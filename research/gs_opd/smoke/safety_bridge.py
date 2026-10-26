import os
os.environ["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from multiprocessing.connection import Listener
import argparse
import copy
import hashlib
import json
import math
import numpy as np
import pickle
import random
import mujoco
from pathlib import Path
import sys
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv
from research.gs_opd.smoke.joint_limit_policy import (
    DEFAULT_JOINT_LIMIT_TOLERANCE_POLICY,
    evaluate_joint_limit,
    joint_type_name,
    normalize_joint_limit_tolerance_policy,
)


JOINT_LIMIT_TOLERANCE_POLICY = DEFAULT_JOINT_LIMIT_TOLERANCE_POLICY

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# This checkpoint was trained and is normally evaluated with the upstream
# robosuite v1.4.0 OSC_POSE ranges. The vendored LIBERO-Safety copy has a
# modified osc_pose.json (all six outputs are +/-2), so apply the upstream
# runtime values to this Bridge environment only.
_PI05_LIBERO_OSC_POS_OUTPUT = 0.05
_PI05_LIBERO_OSC_ROT_OUTPUT = 0.5
_PI05_LIBERO_OSC_KP = 150.0


def _configure_pi05_libero_osc_pose(controller):
    """Apply and verify the LIBERO / Pi0.5 OSC_POSE runtime convention."""
    controller_name = getattr(controller, "name", type(controller).__name__)
    if controller_name != "OSC_POSE":
        raise RuntimeError(
            "Pi0.5 LIBERO requires OSC_POSE, got " f"{controller_name!r}"
        )
    if int(getattr(controller, "control_dim", -1)) != 6:
        raise RuntimeError("OSC_POSE must expose six arm control dimensions")
    if getattr(controller, "use_delta", None) is not True:
        raise RuntimeError("OSC_POSE must use delta commands for Pi0.5 LIBERO")
    if getattr(controller, "use_ori", None) is not True:
        raise RuntimeError("OSC_POSE must control position and orientation")
    if getattr(controller, "impedance_mode", None) != "fixed":
        raise RuntimeError("Pi0.5 LIBERO expects fixed-impedance OSC_POSE")

    input_min = np.asarray(getattr(controller, "input_min", []), dtype=np.float64)
    input_max = np.asarray(getattr(controller, "input_max", []), dtype=np.float64)
    expected_input_min = np.full(6, -1.0, dtype=np.float64)
    expected_input_max = np.full(6, 1.0, dtype=np.float64)
    if not (
        input_min.shape == (6,)
        and input_max.shape == (6,)
        and np.allclose(input_min, expected_input_min)
        and np.allclose(input_max, expected_input_max)
    ):
        raise RuntimeError(
            "Unexpected OSC_POSE normalized input bounds: "
            f"min={input_min.tolist()}, max={input_max.tolist()}"
        )

    # robosuite v1.4 OSC deltas are world-frame: set_goal_position adds the
    # scaled delta to world-coordinate ee_pos, and orientation deltas are
    # pre-multiplied onto ee_ori_mat. Newer controllers may expose the frame.
    input_ref_frame = getattr(controller, "input_ref_frame", None)
    if input_ref_frame not in (None, "world"):
        raise RuntimeError(
            "Expected world-frame OSC_POSE deltas, got " f"{input_ref_frame!r}"
        )

    output_min = np.asarray(
        [-_PI05_LIBERO_OSC_POS_OUTPUT] * 3 + [-_PI05_LIBERO_OSC_ROT_OUTPUT] * 3,
        dtype=np.float64,
    )
    output_max = -output_min
    controller.output_min = output_min.copy()
    controller.output_max = output_max.copy()

    # Match the fixed OSC gains in the upstream v1.4.0 controller config too.
    controller.kp = np.full(6, _PI05_LIBERO_OSC_KP, dtype=np.float64)
    controller.kd = 2.0 * np.sqrt(controller.kp)
    controller.kp_min = np.zeros(6, dtype=np.float64)
    controller.kp_max = np.full(6, 300.0, dtype=np.float64)
    controller.damping_ratio_min = np.zeros(6, dtype=np.float64)
    controller.damping_ratio_max = np.full(6, 10.0, dtype=np.float64)
    controller.position_limits = None
    controller.orientation_limits = None
    controller.uncoupling = True

    # The controller caches the affine input-to-output scale on first use.
    # Invalidate it before probing the actual scaling implementation.
    controller.action_scale = None
    controller.action_input_transform = None
    controller.action_output_transform = None
    probe = np.full(6, 0.5, dtype=np.float64)
    scaled_probe = np.asarray(controller.scale_action(probe), dtype=np.float64)
    expected_probe = np.asarray(
        [_PI05_LIBERO_OSC_POS_OUTPUT * 0.5] * 3
        + [_PI05_LIBERO_OSC_ROT_OUTPUT * 0.5] * 3,
        dtype=np.float64,
    )
    if scaled_probe.shape != (6,) or not np.allclose(scaled_probe, expected_probe):
        raise RuntimeError(
            "OSC_POSE runtime scaling validation failed: "
            f"expected {expected_probe.tolist()}, got {scaled_probe.tolist()}"
        )

    return {
        "controller": controller_name,
        "input_min": input_min.tolist(),
        "input_max": input_max.tolist(),
        "output_min": output_min.tolist(),
        "output_max": output_max.tolist(),
        "position_output_range_m": [-_PI05_LIBERO_OSC_POS_OUTPUT, _PI05_LIBERO_OSC_POS_OUTPUT],
        "rotation_output_range_rad": [-_PI05_LIBERO_OSC_ROT_OUTPUT, _PI05_LIBERO_OSC_ROT_OUTPUT],
        "position_gain_m_per_input": _PI05_LIBERO_OSC_POS_OUTPUT,
        "rotation_gain_rad_per_input": _PI05_LIBERO_OSC_ROT_OUTPUT,
        "control_frame": "world",
        "control_delta": True,
        "kp": controller.kp.tolist(),
        "kd": controller.kd.tolist(),
        "damping_ratio": 1.0,
        "kp_limits": [controller.kp_min.tolist(), controller.kp_max.tolist()],
        "damping_ratio_limits": [
            controller.damping_ratio_min.tolist(),
            controller.damping_ratio_max.tolist(),
        ],
        "impedance_mode": controller.impedance_mode,
        "uncouple_pos_ori": bool(controller.uncoupling),
        "scaled_half_input_probe": scaled_probe.tolist(),
        "source": "upstream robosuite v1.4.0 OSC_POSE defaults",
    }


def _parse_scene_args():
    parser = argparse.ArgumentParser(description="LIBERO-Safety scene Bridge")
    parser.add_argument(
        "--benchmark", default="obstacle_avoidance",
        help="registered LIBERO benchmark name (default: obstacle_avoidance)",
    )
    parser.add_argument("--level", type=int, default=0, help="benchmark level (default: 0)")
    parser.add_argument(
        "--task-id", "--task", dest="task_id", type=int, default=0,
        help="zero-based task ID within the level (default: 0)",
    )
    parser.add_argument(
        "--init-state", type=int, default=0,
        help="zero-based task initialization state (default: 0)",
    )
    parser.add_argument("--seed", type=int, default=0, help="environment seed (default: 0)")
    parser.add_argument("--port", type=int, default=38567, help="local Bridge TCP port (default: 38567)")
    parser.add_argument(
        "--demonstration-horizon",
        type=int,
        default=1000,
        help="isolated horizon; values over 1000 are demonstration-only",
    )
    parser.add_argument(
        "--camera-resolution", type=int, default=256,
        help="square RGB observation resolution for both cameras (default: 256)",
    )
    return parser.parse_args()


scene_args = _parse_scene_args()
if scene_args.camera_resolution <= 0:
    raise ValueError("--camera-resolution must be positive")
if not 1 <= scene_args.demonstration_horizon <= 3600:
    raise ValueError("--demonstration-horizon must be in 1..3600; values over 1000 are demonstration-only")
suite = get_benchmark(scene_args.benchmark)()
task = suite.get_task_by_level_id(scene_args.level, scene_args.task_id)
if task is None:
    raise ValueError(
        f"Unknown LIBERO task: benchmark={scene_args.benchmark!r}, "
        f"level={scene_args.level}, task_id={scene_args.task_id}"
    )
bddl = suite.get_task_bddl_file_path_by_level_id(scene_args.level, scene_args.task_id)
if not bddl or not Path(bddl).is_file():
    raise FileNotFoundError(f"LIBERO BDDL file is unavailable: {bddl!r}")
states = suite.get_task_init_states_by_level_id(scene_args.level, scene_args.task_id)
if states is None or not 0 <= scene_args.init_state < len(states):
    state_count = 0 if states is None else len(states)
    raise IndexError(
        f"init_state {scene_args.init_state} is out of range for "
        f"{scene_args.benchmark}/L{scene_args.level}T{scene_args.task_id} "
        f"({state_count} states)"
    )
scene_metadata = {
    "benchmark": scene_args.benchmark,
    "level": scene_args.level,
    "task_id": scene_args.task_id,
    "task_name": getattr(task, "name", None),
    "instruction": task.language,
    "bddl_path": str(Path(bddl).resolve()),
    "bddl_sha256": hashlib.sha256(Path(bddl).read_bytes()).hexdigest(),
    "init_state_index": scene_args.init_state,
    "init_state_count": len(states),
    "environment_seed": scene_args.seed,
    "camera_resolution": scene_args.camera_resolution,
    "camera_names": ["agentview_image", "robot0_eye_in_hand_image"],
    "standard_evaluation_horizon": 1000,
    "bridge_horizon": scene_args.demonstration_horizon,
    "horizon_mode": (
        "standard_benchmark" if scene_args.demonstration_horizon == 1000
        else "demonstration_only_extension"
    ),
}

print("LOADING SAFETY ENV...", flush=True)
env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    horizon=scene_args.demonstration_horizon,
    camera_heights=scene_args.camera_resolution,
    camera_widths=scene_args.camera_resolution,
)
env.seed(scene_args.seed)
env.reset()
obs = env.set_init_state(states[scene_args.init_state])
controller_runtime_config = _configure_pi05_libero_osc_pose(env.env.robots[0].controller)
print("PI05 LIBERO OSC_POSE RUNTIME CONFIG:", controller_runtime_config, flush=True)
print("LIBERO SCENE CONFIG:", json.dumps(scene_metadata, ensure_ascii=False), flush=True)
last_reward = 0.0
last_done = False


class _PythonObjectState:
    """In-process state for small robosuite helper objects (controller/buffer)."""

    def __init__(self, module, qualname, attributes):
        self.module = module
        self.qualname = qualname
        self.attributes = attributes


_STATE_OBJECT_MODULE_PREFIXES = (
    "robosuite.controllers.interpolators",
    "robosuite.utils.buffers",
)
_ROBOT_BUFFER_FIELDS = (
    "recent_qpos",
    "recent_actions",
    "recent_torques",
    "recent_ee_forcetorques",
    "recent_ee_pose",
    "recent_ee_vel",
    "recent_ee_vel_buffer",
    "recent_ee_acc",
)
_OBSERVABLE_STATE_FIELDS = (
    "_time_since_last_sample",
    "_current_delay",
    "_current_observed_value",
    "_sampled",
)


def _capture_python_value(value):
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, np.generic):
        return value.copy()
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    if isinstance(value, dict):
        return {copy.deepcopy(key): _capture_python_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_capture_python_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_capture_python_value(item) for item in value)

    value_type = type(value)
    module = value_type.__module__
    if module.startswith(_STATE_OBJECT_MODULE_PREFIXES):
        return _PythonObjectState(
            module=module,
            qualname=value_type.__qualname__,
            attributes={
                name: _capture_python_value(item)
                for name, item in vars(value).items()
            },
        )
    raise TypeError(
        "Unsupported mutable snapshot value: "
        f"{module}.{value_type.__qualname__}"
    )


def _restore_python_value(current, saved):
    if isinstance(saved, _PythonObjectState):
        if current is None or type(current).__module__ != saved.module or type(current).__qualname__ != saved.qualname:
            raise RuntimeError(
                "Snapshot helper object no longer matches its saved type: "
                f"{saved.module}.{saved.qualname}"
            )
        for name, item in saved.attributes.items():
            current_value = getattr(current, name, None)
            setattr(current, name, _restore_python_value(current_value, item))
        return current
    if isinstance(saved, np.ndarray):
        return saved.copy()
    if isinstance(saved, dict):
        return {
            copy.deepcopy(key): _restore_python_value(
                current.get(key) if isinstance(current, dict) else None,
                item,
            )
            for key, item in saved.items()
        }
    if isinstance(saved, list):
        current_items = current if isinstance(current, list) else []
        return [
            _restore_python_value(
                current_items[index] if index < len(current_items) else None,
                item,
            )
            for index, item in enumerate(saved)
        ]
    if isinstance(saved, tuple):
        current_items = current if isinstance(current, tuple) else ()
        return tuple(
            _restore_python_value(
                current_items[index] if index < len(current_items) else None,
                item,
            )
            for index, item in enumerate(saved)
        )
    return copy.deepcopy(saved)


def _capture_object_fields(obj, excluded=()):
    return {
        name: _capture_python_value(value)
        for name, value in vars(obj).items()
        if name not in excluded
    }


def _restore_object_fields(obj, saved):
    for name, value in saved.items():
        current = getattr(obj, name, None)
        setattr(obj, name, _restore_python_value(current, value))


def _mujoco_integration_state(sim):
    """Capture MuJoCo's complete integration state, not only qpos and qvel."""
    model = sim.model._model
    data = sim.data._data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(model, state_spec), dtype=np.float64)
    mujoco.mj_getState(model, data, state, state_spec)
    return state


def _set_mujoco_integration_state(sim, state):
    model = sim.model._model
    data = sim.data._data
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.asarray(state, dtype=np.float64)
    expected_size = mujoco.mj_stateSize(model, state_spec)
    if state.shape != (expected_size,):
        raise ValueError(
            f"MuJoCo integration state has shape {state.shape}; expected {(expected_size,)}"
        )
    mujoco.mj_setState(model, data, state, state_spec)
    # mj_forward refreshes derived kinematics from qpos, but the constraint
    # solver may replace qacc_warmstart. Preserve that integration component
    # directly: reapplying the complete state after forward can leave derived
    # xpos / xquat / site_xpos caches inconsistent with the restored qpos.
    qacc_warmstart = data.qacc_warmstart.copy()
    sim.forward()
    data.qacc_warmstart[:] = qacc_warmstart


def _diagnostic_data_from_integration_state(sim):
    """Compute canonical kinematics on scratch data without mutating live state."""
    model = sim.model._model
    data = mujoco.MjData(model)
    state = _mujoco_integration_state(sim)
    state_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, data, state, state_spec)
    mujoco.mj_forward(model, data)
    return data


def _capture_task_object_properties(task_env):
    properties = {}
    for mapping_name in ("objects_dict", "fixtures_dict", "object_sites_dict"):
        for object_name, obj in getattr(task_env, mapping_name, {}).items():
            if hasattr(obj, "object_properties"):
                properties[(mapping_name, object_name)] = copy.deepcopy(obj.object_properties)
    return properties


def _capture_snapshot():
    task_env = env.env
    robot_states = []
    for robot in env.robots:
        controller_state = _capture_object_fields(robot.controller, excluded=("sim",))
        buffer_state = {
            field: _capture_python_value(getattr(robot, field, None))
            for field in _ROBOT_BUFFER_FIELDS
        }
        gripper = getattr(robot, "gripper", None)
        gripper_action = (
            _capture_python_value(gripper.current_action)
            if gripper is not None and hasattr(gripper, "current_action")
            else None
        )
        robot_states.append(
            {
                "controller": controller_state,
                "buffers": buffer_state,
                "torques": _capture_python_value(getattr(robot, "torques", None)),
                "gripper_current_action": gripper_action,
                "has_gripper_current_action": gripper is not None and hasattr(gripper, "current_action"),
            }
        )

    motion_generators = {}
    for name, generator in getattr(task_env, "mocap_motion_generators", {}).items():
        motion_generators[name] = copy.deepcopy(vars(generator))

    observable_states = {}
    for name, observable in getattr(task_env, "_observables", {}).items():
        observable_states[name] = {
            field: copy.deepcopy(getattr(observable, field))
            for field in _OBSERVABLE_STATE_FIELDS
        }

    return {
        # Keep a strong reference and require the exact same compiled model on
        # restore; a hard environment reset can rebuild it with different state.
        "mujoco_model": env.sim.model._model,
        "mujoco_state": _mujoco_integration_state(env.sim),
        "bridge_observation": copy.deepcopy(obs),
        "bridge_reward": last_reward,
        "bridge_done": last_done,
        "task_bookkeeping": {
            field: copy.deepcopy(getattr(task_env, field))
            for field in ("cur_time", "timestep", "done")
        },
        "robots": robot_states,
        "motion_generators": motion_generators,
        "task_object_properties": _capture_task_object_properties(task_env),
        "observable_states": observable_states,
        "obs_cache": copy.deepcopy(task_env._obs_cache),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "python_rng_state": random.getstate(),
    }


def _restore_snapshot(snapshot):
    global obs, last_reward, last_done
    task_env = env.env
    if snapshot["mujoco_model"] is not env.sim.model._model:
        raise RuntimeError("Snapshot belongs to a different compiled MuJoCo model")
    if len(snapshot["robots"]) != len(env.robots):
        raise RuntimeError("Snapshot robot count differs from the active environment")

    _set_mujoco_integration_state(env.sim, snapshot["mujoco_state"])

    for field, value in snapshot["task_bookkeeping"].items():
        setattr(task_env, field, copy.deepcopy(value))

    for robot, saved_robot in zip(env.robots, snapshot["robots"]):
        _restore_object_fields(robot.controller, saved_robot["controller"])
        for field, value in saved_robot["buffers"].items():
            current = getattr(robot, field, None)
            setattr(robot, field, _restore_python_value(current, value))
        robot.torques = _restore_python_value(
            getattr(robot, "torques", None), saved_robot["torques"]
        )
        if saved_robot["has_gripper_current_action"]:
            gripper = getattr(robot, "gripper", None)
            if gripper is None or not hasattr(gripper, "current_action"):
                raise RuntimeError("Snapshot gripper action state is unavailable")
            gripper.current_action = _restore_python_value(
                gripper.current_action, saved_robot["gripper_current_action"]
            )

    generators = getattr(task_env, "mocap_motion_generators", {})
    if set(generators) != set(snapshot["motion_generators"]):
        raise RuntimeError("Snapshot dynamic-object generators differ from the active environment")
    for name, attributes in snapshot["motion_generators"].items():
        generator = generators[name]
        for field, value in attributes.items():
            setattr(generator, field, copy.deepcopy(value))

    for (mapping_name, object_name), properties in snapshot["task_object_properties"].items():
        obj = getattr(task_env, mapping_name)[object_name]
        current_properties = obj.object_properties
        if isinstance(current_properties, dict) and isinstance(properties, dict):
            current_properties.clear()
            current_properties.update(copy.deepcopy(properties))
        else:
            obj.object_properties = copy.deepcopy(properties)

    for name, fields in snapshot["observable_states"].items():
        observable = task_env._observables[name]
        for field, value in fields.items():
            setattr(observable, field, copy.deepcopy(value))
    task_env._obs_cache = copy.deepcopy(snapshot["obs_cache"])
    obs = copy.deepcopy(snapshot["bridge_observation"])
    last_reward = snapshot["bridge_reward"]
    last_done = snapshot["bridge_done"]

    # Restore RNG last: all state-copy operations above are deterministic and must
    # not consume the random stream that the next step will use.
    np.random.set_state(copy.deepcopy(snapshot["numpy_rng_state"]))
    random.setstate(snapshot["python_rng_state"])


def _export_runtime_snapshot():
    snapshot = _capture_snapshot()
    snapshot.pop("mujoco_model", None)
    envelope = {
        "schema_version": "gs_opd_runtime_snapshot.v1",
        "scene_metadata": copy.deepcopy(scene_metadata),
        "model_geometry_sha256": _planning_model_signature(env.sim.model._model),
        "snapshot": snapshot,
    }
    payload = pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL)
    return {
        "status": "complete",
        "schema_version": envelope["schema_version"],
        "model_geometry_sha256": envelope["model_geometry_sha256"],
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload": payload,
        "snapshot_state": _snapshot_diagnostics(canonicalize_kinematics=True),
    }


def _restore_runtime_snapshot(payload):
    if not isinstance(payload, bytes) or not payload:
        raise ValueError("Runtime Snapshot payload must be non-empty bytes")
    envelope = pickle.loads(payload)
    if not isinstance(envelope, dict) or envelope.get("schema_version") != "gs_opd_runtime_snapshot.v1":
        raise ValueError("Runtime Snapshot schema is unsupported")
    expected_scene = envelope.get("scene_metadata")
    scene_fields = (
        "benchmark", "level", "task_id", "init_state_index", "environment_seed",
        "bddl_sha256", "instruction",
    )
    if not isinstance(expected_scene, dict) or any(
        expected_scene.get(field) != scene_metadata.get(field)
        for field in scene_fields
    ):
        raise ValueError("Runtime Snapshot scene metadata differs from the active Bridge")
    model_signature = _planning_model_signature(env.sim.model._model)
    if envelope.get("model_geometry_sha256") != model_signature:
        raise ValueError("Runtime Snapshot model geometry differs from the active Bridge")
    snapshot = envelope.get("snapshot")
    if not isinstance(snapshot, dict) or "mujoco_state" not in snapshot:
        raise ValueError("Runtime Snapshot is missing its physical state")
    snapshot["mujoco_model"] = env.sim.model._model
    _restore_snapshot(snapshot)
    return {
        "status": "restored",
        "model_geometry_sha256": model_signature,
        "snapshot_state": _snapshot_diagnostics(canonicalize_kinematics=True),
        "reward": float(last_reward),
        "done": bool(last_done),
    }


def _json_value(value):
    if value is None:
        return None
    try:
        array = np.asarray(value)
    except Exception:
        return None
    if array.ndim == 0:
        scalar = array.item()
        if isinstance(scalar, (bool, int, float, str)):
            return scalar
        return None
    try:
        return array.tolist()
    except Exception:
        return None


def _repair_context():
    """Read the active controller scaling and robot limits without changing state."""
    sim = env.sim
    model = sim.model._model
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    gravity_norm = float(np.linalg.norm(gravity))
    world_up = (-gravity / gravity_norm).tolist() if gravity_norm > 1e-12 else None
    robots = []
    for index, robot in enumerate(env.robots):
        controller = getattr(robot, "controller", None)
        if controller is None:
            robots.append({"robot_index": index, "status": "unavailable"})
            continue
        joint_ids = np.asarray(getattr(robot, "_ref_joint_indexes", []), dtype=np.int64)
        joint_ranges = None
        joint_limited = None
        joint_names = []
        try:
            joint_ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64).tolist()
            joint_limited = np.asarray(model.jnt_limited[joint_ids], dtype=bool).tolist()
            joint_names = [sim.model.joint_id2name(int(joint_id)) for joint_id in joint_ids]
        except Exception:
            joint_ranges = None
            joint_limited = None
        fields = (
            "control_dim",
            "input_min",
            "input_max",
            "output_min",
            "output_max",
            "control_delta",
            "use_delta",
            "use_ori",
            "kp",
            "kd",
            "impedance_mode",
            "uncoupling",
            "position_limits",
            "orientation_limits",
            "input_ref_frame",
            "eef_name",
        )
        controller_fields = {key: _json_value(getattr(controller, key, None)) for key in fields}
        # robosuite OSC stores the constructor's ``control_delta`` flag as
        # ``use_delta``; expose the effective runtime value under both labels.
        if controller_fields["control_delta"] is None:
            controller_fields["control_delta"] = _json_value(getattr(controller, "use_delta", None))
        robots.append(
            {
                "robot_index": index,
                "status": "available",
                "controller_name": getattr(controller, "name", type(controller).__name__),
                "robot_name": getattr(getattr(robot, "robot_model", None), "name", None),
                "action_dim": int(getattr(robot, "action_dim", 0)),
                "has_gripper": bool(getattr(robot, "has_gripper", False)),
                "fields": controller_fields,
                "joint_names": joint_names,
                "joint_ranges": joint_ranges,
                "joint_limited": joint_limited,
            }
        )
    return {
        "status": "available" if robots and all(r.get("status") == "available" for r in robots) else "incomplete",
        "source": "active robosuite controller and compiled MuJoCo model",
        "world_up": world_up,
        "gravity": gravity.tolist(),
        "model_timestep": float(sim.model.opt.timestep),
        "robots": robots,
    }


def _tracking_value_equal(left, right):
    """Exact equality for captured Snapshot state, including nested arrays."""
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return (
                isinstance(left, np.ndarray)
                and isinstance(right, np.ndarray)
                and left.shape == right.shape
                and left.dtype == right.dtype
                and np.array_equal(left, right)
            )
        except Exception:
            return False
    if isinstance(left, _PythonObjectState) or isinstance(right, _PythonObjectState):
        return (
            isinstance(left, _PythonObjectState)
            and isinstance(right, _PythonObjectState)
            and left.module == right.module
            and left.qualname == right.qualname
            and _tracking_value_equal(left.attributes, right.attributes)
        )
    if isinstance(left, dict) or isinstance(right, dict):
        return (
            isinstance(left, dict)
            and isinstance(right, dict)
            and left.keys() == right.keys()
            and all(_tracking_value_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(_tracking_value_equal(a, b) for a, b in zip(left, right))
        )
    if left is right:
        return True
    try:
        value = left == right
        return bool(value) if np.ndim(value) == 0 else False
    except Exception:
        return False


def _tracking_snapshot_state_id(snapshot):
    model = env.sim.model._model
    instruction = task.language
    digest = hashlib.sha256()
    digest.update(_planning_model_signature(model).encode())
    digest.update(np.asarray(snapshot["mujoco_state"], dtype=np.float64).tobytes())
    digest.update(instruction.encode())
    return digest.hexdigest()


def _tracking_state_object(diagnostic, body_id):
    matches = [
        value
        for value in diagnostic.get("object_states", {}).values()
        if value.get("body_id") == body_id
    ]
    if len(matches) != 1:
        raise ValueError("carried object root state is missing or ambiguous")
    result = matches[0]
    position = np.asarray(result.get("root_pos"), dtype=np.float64)
    quaternion = np.asarray(result.get("root_quat"), dtype=np.float64)
    if (
        position.shape != (3,)
        or quaternion.shape != (4,)
        or not np.isfinite(position).all()
        or not np.isfinite(quaternion).all()
    ):
        raise ValueError("carried object root pose is incomplete")
    quaternion_norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(quaternion_norm) or quaternion_norm <= 1e-12:
        raise ValueError("carried object root quaternion is invalid")
    return position, quaternion / quaternion_norm


def _tracking_body_is_ancestor(model, ancestor, body):
    current = int(body)
    while current > 0:
        if current == int(ancestor):
            return True
        current = int(model.body_parentid[current])
    return current == int(ancestor)


def _tracking_validate_binding(binding, expected_state_id):
    sim = env.sim
    model = sim.model._model
    if not isinstance(binding, dict) or binding.get("state_id") != expected_state_id:
        raise ValueError("carried binding state does not match the requested trajectory state")
    body_id = binding.get("body_id")
    body_name = binding.get("body_name")
    geom_ids = binding.get("geom_ids")
    body_ids = binding.get("body_ids")
    geom_names = binding.get("geom_names")
    semantic_geom_ids = binding.get("semantic_binding_geom_ids")
    semantic_geom_names = binding.get("semantic_binding_geom_names")
    if (
        not isinstance(body_ids, list)
        or not body_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in body_ids)
        or len(set(body_ids)) != len(body_ids)
        or not isinstance(geom_ids, list)
        or not geom_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in geom_ids)
        or len(set(geom_ids)) != len(geom_ids)
        or not isinstance(semantic_geom_ids, list)
        or not semantic_geom_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in semantic_geom_ids)
        or len(set(semantic_geom_ids)) != len(semantic_geom_ids)
        or not isinstance(semantic_geom_names, list)
        or len(semantic_geom_names) != len(semantic_geom_ids)
        or any(not isinstance(value, str) or not value for value in semantic_geom_names)
    ):
        raise ValueError("carried Body/Geom ID arrays are malformed")
    if (
        isinstance(body_id, bool)
        or not isinstance(body_id, int)
        or not 0 < body_id < int(model.nbody)
        or sim.model.body_id2name(body_id) != body_name
        or not isinstance(body_name, str)
        or not isinstance(geom_ids, list)
        or not geom_ids
        or not isinstance(body_ids, list)
        or not body_ids
        or not isinstance(geom_names, list)
        or len(geom_names) != len(geom_ids)
        or any(not isinstance(value, str) or not value for value in geom_names)
        or not isinstance(semantic_geom_ids, list)
        or not isinstance(semantic_geom_names, list)
        or len(semantic_geom_names) != len(semantic_geom_ids)
        or any(value < 0 or value >= int(model.ngeom) for value in geom_ids + semantic_geom_ids)
        or any(value < 0 or value >= int(model.nbody) for value in body_ids)
        or binding.get("binding_verified") is not True
        or binding.get("covers_all_associated_collision_geoms") is not True
    ):
        raise ValueError("carried Body/Geom binding declaration is incomplete")
    registered = _planning_registered_objects(sim.model, env.env)
    if body_id not in registered or registered[body_id]["body_name"] != body_name:
        raise ValueError("carried Body is not a current registered task-object root")
    subtree = _planning_body_subtree(model, body_id)
    if set(body_ids) != subtree:
        raise ValueError("carried Body subtree differs from the verified binding")
    inventory = _geometry_inventory()
    if (
        inventory.get("status") != "complete"
        or inventory.get("robot_classification_status") != "available"
    ):
        raise ValueError("current Geometry Inventory is incomplete")
    inventory_entries = inventory.get("geometries")
    if not isinstance(inventory_entries, list):
        raise ValueError("current Geometry Inventory is malformed")
    inventory_by_id = {
        item.get("geom_id"): item
        for item in inventory_entries
        if isinstance(item, dict)
    }
    if len(inventory_by_id) != len(inventory_entries):
        raise ValueError("current Geometry Inventory has duplicate or malformed Geom IDs")
    pair_geoms = set()
    for pair_index in range(int(model.npair)):
        pair_geoms.add(int(model.pair_geom1[pair_index]))
        pair_geoms.add(int(model.pair_geom2[pair_index]))
    collision_ids = {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in subtree
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
            or geom_id in pair_geoms
        )
    }
    if set(geom_ids) != collision_ids:
        raise ValueError("carried collision Geoms do not cover the complete current Body subtree")
    for geom_id, geom_name in zip(geom_ids, geom_names):
        item = inventory_by_id.get(geom_id)
        if (
            item is None
            or item.get("geom_name") != geom_name
            or item.get("body_id") != body_id
            or item.get("body_name") != body_name
            or item.get("category") != "scene"
        ):
            raise ValueError("carried collision Geom identity conflicts with current model")
    for geom_id, geom_name in zip(semantic_geom_ids, semantic_geom_names):
        item = inventory_by_id.get(geom_id)
        if (
            item is None
            or item.get("geom_name") != geom_name
            or item.get("body_id") != body_id
            or item.get("body_name") != body_name
            or item.get("category") != "scene"
        ):
            raise ValueError(
                "carried semantic Geom identity is outside the verified root Body"
            )
    # The existing Contact Compiler binds contact events to a root Body ID. Do
    # not claim complete object-contact checking for articulated child Bodies.
    if any(int(model.geom_bodyid[geom_id]) != body_id for geom_id in geom_ids):
        raise ValueError("carried_collision_geoms_span_child_bodies_not_supported_by_compiler")
    joint_start = int(model.body_jntadr[body_id])
    joint_count = int(model.body_jntnum[body_id])
    if joint_count != 1 or int(model.jnt_type[joint_start]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("carried object root must expose one free joint")
    if any(int(model.body_jntnum[child]) for child in subtree if child != body_id):
        raise ValueError("articulated carried-object subtree is not supported")
    return inventory, subtree


def _tracking_workspace(robot, diagnostic_data):
    """Compute a model-derived conservative reach sphere for the active Panda."""
    sim = env.sim
    model = sim.model._model
    root_name = getattr(getattr(robot, "robot_model", None), "root_body", None)
    if not isinstance(root_name, str):
        raise ValueError("Panda base Body is unavailable")
    base_id = int(sim.model.body_name2id(root_name))
    joint_ids = np.asarray(getattr(robot, "_ref_joint_indexes", []), dtype=np.int64)
    if joint_ids.ndim != 1 or not len(joint_ids):
        raise ValueError("Panda arm joint inventory is unavailable")
    if np.any(model.jnt_type[joint_ids] != int(mujoco.mjtJoint.mjJNT_HINGE)):
        raise ValueError("workspace reach bound only supports revolute Panda arm joints")
    if np.any(model.jnt_limited[joint_ids] == 0):
        raise ValueError("Panda arm joint limits are incomplete")
    joint_bodies = np.asarray(model.jnt_bodyid[joint_ids], dtype=np.int64)
    prior = base_id
    for body_id in joint_bodies:
        if not _tracking_body_is_ancestor(model, prior, int(body_id)):
            raise ValueError("Panda arm joint order is not a verified kinematic chain")
        prior = int(body_id)
    anchors = np.asarray(diagnostic_data.xanchor[joint_ids], dtype=np.float64)
    base = np.asarray(diagnostic_data.xpos[base_id], dtype=np.float64)
    eef = np.asarray(diagnostic_data.site_xpos[robot.eef_site_id], dtype=np.float64)
    chain = np.vstack((base, anchors, eef))
    if not np.isfinite(chain).all():
        raise ValueError("Panda reach-chain points are non-finite")
    reach = float(np.linalg.norm(np.diff(chain, axis=0), axis=1).sum())
    if not np.isfinite(reach) or reach <= 0:
        raise ValueError("Panda model-derived reach bound is invalid")
    fields = {name: _json_value(getattr(robot.controller, name, None)) for name in (
        "position_limits", "output_min", "output_max", "input_min", "input_max"
    )}
    limits = fields["position_limits"]
    if limits is not None:
        bounds = np.asarray(limits, dtype=np.float64)
        if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[0] > bounds[1]):
            raise ValueError("active OSC Cartesian workspace bounds are malformed")
    return {
        "status": "available",
        "source": "conservative sphere from compiled Panda joint-anchor chain and EEF site",
        "base_body_id": base_id,
        "base_world_m": base.tolist(),
        "maximum_reach_radius_m": reach,
        "controller_position_limits": limits,
    }


def _tracking_workspace_check(position, workspace):
    from research.gs_opd.repair.trajectory_tracking import check_workspace_target

    return check_workspace_target(
        position,
        workspace.get("base_world_m"),
        workspace.get("maximum_reach_radius_m"),
        workspace.get("controller_position_limits"),
    )


def _tracking_joint_limit_check(robot):
    sim = env.sim
    model = sim.model._model
    arm_joint_ids = [int(item) for item in robot._ref_joint_indexes]
    gripper_joint_names = list(getattr(robot, "gripper_joints", []))
    gripper_joint_ids = [int(sim.model.joint_name2id(name)) for name in gripper_joint_names]
    joint_ids = np.asarray(arm_joint_ids + gripper_joint_ids, dtype=np.int64)
    qpos_indices = list(robot._ref_joint_pos_indexes) + list(robot._ref_gripper_joint_pos_indexes)
    if len(joint_ids) == 0 or len(joint_ids) != len(qpos_indices):
        raise ValueError("Panda arm/gripper joint limit indexes are incomplete")
    ranges = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[joint_ids], dtype=bool)
    qpos = np.asarray(sim.data.qpos[qpos_indices], dtype=np.float64)
    if (
        ranges.shape != (len(joint_ids), 2)
        or qpos.shape != (len(joint_ids),)
        or not np.isfinite(ranges).all()
        or not np.isfinite(qpos).all()
        or np.any(ranges[:, 0] > ranges[:, 1])
        or not limited.all()
    ):
        raise ValueError("Panda active arm/gripper joint limits are unavailable")
    outside = (qpos < ranges[:, 0] - 1e-8) | (qpos > ranges[:, 1] + 1e-8)
    return {
        "status": "fail" if outside.any() else "pass",
        "joint_names": [sim.model.joint_id2name(int(joint_id)) for joint_id in joint_ids],
        "qpos": qpos.tolist(),
        "ranges": ranges.tolist(),
        "limited": limited.tolist(),
        "violating_joint_names": [
            sim.model.joint_id2name(int(joint_ids[index]))
            for index in np.flatnonzero(outside)
        ],
    }


def _tracking_gripper_sides(robot):
    gripper = getattr(robot, "gripper", None)
    if gripper is None or type(gripper).__name__ != "PandaGripper":
        raise ValueError("closed-loop grasp evidence requires the active PandaGripper model")
    important = getattr(gripper, "_important_geoms", None)
    if not isinstance(important, dict):
        raise ValueError("PandaGripper important Geom groups are unavailable")
    groups = {}
    for side, keys in (
        ("left", ("left_finger", "left_fingerpad")),
        ("right", ("right_finger", "right_fingerpad")),
    ):
        names = [name for key in keys for name in important.get(key, [])]
        if not names:
            raise ValueError(f"PandaGripper {side} finger Geoms are unavailable")
        adjusted = gripper.correct_naming(names)
        ids = {int(env.sim.model.geom_name2id(name)) for name in adjusted}
        if not ids or any(
            geom_id < 0 or geom_id >= int(env.sim.model._model.ngeom)
            for geom_id in ids
        ):
            raise ValueError(f"PandaGripper {side} Geom IDs do not resolve in the active model")
        groups[side] = ids
    closure = _parse_tracking_gripper_closure(getattr(gripper, "current_action", None))
    return groups, closure


def _parse_tracking_gripper_closure(current_action):
    # The import is deferred until after the Bridge has installed the repo root.
    from research.gs_opd.repair.panda_gripper_state import parse_panda_gripper_closure

    return parse_panda_gripper_closure(current_action)


def _live_grasp_state(binding, expected_instruction):
    """Read current Panda finger contacts for a semantically bound task object."""
    sim = env.sim
    result = {
        "operation": "grasp_state",
        "status": "unavailable",
        "grasp_confirmed": False,
        "instruction": task.language,
        "expected_instruction": expected_instruction,
        "physical_state_unchanged": False,
    }
    before = model_signature = cache_signature = None
    try:
        before = _mujoco_integration_state(sim)
        model_signature = _planning_model_signature(sim.model._model)
        cache_signature = _planning_cache_signature(sim.data._data)
        if (
            not isinstance(expected_instruction, str)
            or not expected_instruction
            or expected_instruction != task.language
        ):
            raise ValueError("live_grasp_instruction_does_not_match_active_task")
        if not isinstance(binding, dict):
            raise ValueError("live_grasp_semantic_binding_is_missing")
        object_id = binding.get("object_id")
        body_id = binding.get("body_id")
        body_name = binding.get("body_name")
        semantic_geom_ids = binding.get("geom_ids")
        semantic_geom_names = binding.get("geom_names")
        model = sim.model._model
        if (
            not isinstance(object_id, str)
            or not object_id
            or isinstance(body_id, bool)
            or not isinstance(body_id, int)
            or body_id <= 0
            or body_id >= int(model.nbody)
            or not isinstance(body_name, str)
            or sim.model.body_id2name(body_id) != body_name
            or not isinstance(semantic_geom_ids, list)
            or not semantic_geom_ids
            or not isinstance(semantic_geom_names, list)
            or len(semantic_geom_names) != len(semantic_geom_ids)
            or any(isinstance(value, bool) or not isinstance(value, int) for value in semantic_geom_ids)
            or len(set(semantic_geom_ids)) != len(semantic_geom_ids)
            or any(not isinstance(value, str) or not value for value in semantic_geom_names)
        ):
            raise ValueError("live_grasp_semantic_body_geom_binding_is_malformed")

        registered = _planning_registered_objects(sim.model, env.env)
        if body_id not in registered or registered[body_id]["body_name"] != body_name:
            raise ValueError("live_grasp_body_is_not_a_registered_task_object")
        subtree = _planning_body_subtree(model, body_id)
        inventory = _geometry_inventory()
        if inventory.get("status") != "complete":
            raise ValueError("live_grasp_geometry_inventory_incomplete")
        inventory_by_id = {
            item["geom_id"]: item for item in inventory.get("geometries", [])
        }
        for geom_id, geom_name in zip(semantic_geom_ids, semantic_geom_names):
            item = inventory_by_id.get(geom_id)
            if (
                item is None
                or item.get("geom_name") != geom_name
                or item.get("body_id") != body_id
                or item.get("body_name") != body_name
                or item.get("category") != "scene"
            ):
                raise ValueError("live_grasp_semantic_geom_conflicts_with_current_model")

        explicit_pair_geoms = set()
        for pair_index in range(int(model.npair)):
            explicit_pair_geoms.add(int(model.pair_geom1[pair_index]))
            explicit_pair_geoms.add(int(model.pair_geom2[pair_index]))
        object_geom_ids = {
            geom_id
            for geom_id in range(int(model.ngeom))
            if int(model.geom_bodyid[geom_id]) in subtree
            and (
                int(model.geom_contype[geom_id]) != 0
                or int(model.geom_conaffinity[geom_id]) != 0
                or geom_id in explicit_pair_geoms
            )
        }
        if not object_geom_ids:
            raise ValueError("live_grasp_object_collision_geometry_is_unavailable")
        if any(
            inventory_by_id.get(geom_id, {}).get("category") != "scene"
            or inventory_by_id.get(geom_id, {}).get("body_id") not in subtree
            for geom_id in object_geom_ids
        ):
            raise ValueError("live_grasp_collision_geom_classification_is_incomplete")
        if len(env.robots) != 1:
            raise ValueError("live_grasp_requires_one_active_Panda_robot")
        gripper_groups, gripper_closure = _tracking_gripper_sides(env.robots[0])
        contacts = _collect_contacts()
        from research.gs_opd.repair.trajectory_bundle_acceptance import (
            classify_opposing_finger_contacts,
        )

        sides = classify_opposing_finger_contacts(
            contacts,
            object_geom_ids=object_geom_ids,
            object_body_ids=subtree,
            gripper_geom_groups=gripper_groups,
        )
        gripper_closed = bool(
            gripper_closure.get("status") == "pass"
            and gripper_closure.get("closed") is True
        )
        grasp_confirmed = (
            contacts.get("status") != "unavailable"
            and sides.get("status") == "pass"
            and gripper_closed
        )
        after = _mujoco_integration_state(sim)
        model_after = _planning_model_signature(sim.model._model)
        cache_after = _planning_cache_signature(sim.data._data)
        unchanged = bool(
            before.shape == after.shape
            and np.array_equal(before, after)
            and model_signature == model_after
            and cache_signature == cache_after
        )
        if not unchanged:
            raise ValueError("live_grasp_read_changed_physical_or_kinematic_state")
        snapshot_identity = {"mujoco_state": before}
        state_id = _tracking_snapshot_state_id(snapshot_identity)
        grasp_status = (
            "pass"
            if grasp_confirmed
            else "unavailable"
            if gripper_closure.get("status") != "pass"
            or contacts.get("status") == "unavailable"
            or sides.get("status") == "unavailable"
            else "not_grasped"
        )
        result.update({
            "status": grasp_status,
            "grasp_confirmed": grasp_confirmed,
            "object_id": object_id,
            "body_id": body_id,
            "body_name": body_name,
            "state_id": state_id,
            "state_fingerprint": hashlib.sha256(before.tobytes()).hexdigest(),
            "gripper_current_action": gripper_closure.get("raw_actuator_action"),
            "gripper_closure": gripper_closure,
            "gripper_closed": gripper_closed,
            "gripper_close_threshold": 0.9,
            "finger_contacts": {
                "status": sides.get("status", "unavailable"),
                "left": bool(sides.get("left")),
                "right": bool(sides.get("right")),
                "contact_count": int(sides.get("contact_count", 0)),
                "matched_contacts": sides.get("matched_contacts", {}),
                "basis": "active PandaGripper finger geom groups against current model-verified object collision geoms",
            },
            "contact_status": contacts.get("status"),
            "physical_state_unchanged": True,
        })
        if not grasp_confirmed:
            if contacts.get("status") == "unavailable":
                result["reason"] = contacts.get("error", "live_contact_data_unavailable")
            elif gripper_closure.get("status") != "pass":
                result["reason"] = (
                    "Panda_gripper_current_action_unavailable:"
                    f"{gripper_closure.get('reason') or 'invalid_state'}"
                )
            elif not gripper_closed:
                result["reason"] = gripper_closure.get("reason") or "Panda_gripper_close_command_below_0.9"
            else:
                result["reason"] = "opposing_left_and_right_finger_contacts_not_both_present"
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["reason"] = result["error"]
        if before is not None:
            try:
                result["physical_state_unchanged"] = bool(
                    np.array_equal(before, _mujoco_integration_state(sim))
                )
            except Exception:
                pass
    return result


def _tracking_contact_sides(contact_result, carried_binding, gripper_groups):
    if not isinstance(contact_result, dict) or contact_result.get("status") == "unavailable":
        return {"status": "unavailable", "left": False, "right": False, "contact_count": 0}
    contacts = contact_result.get("contacts")
    if not isinstance(contacts, list):
        return {"status": "unavailable", "left": False, "right": False, "contact_count": 0}
    object_geoms = set(carried_binding["geom_ids"])
    object_bodies = set(carried_binding["body_ids"])
    sides = {"left": False, "right": False}
    count = 0
    for contact in contacts:
        try:
            pair_geoms = [int(contact["geom1"]["id"]), int(contact["geom2"]["id"])]
            pair_bodies = [int(contact["body1"]["id"]), int(contact["body2"]["id"])]
        except Exception:
            return {"status": "unavailable", **sides, "contact_count": count}
        for side, gripper_geoms in gripper_groups.items():
            for index in (0, 1):
                other = 1 - index
                if (
                    pair_geoms[index] in gripper_geoms
                    and pair_geoms[other] in object_geoms
                    and pair_bodies[other] in object_bodies
                ):
                    sides[side] = True
                    count += 1
    return {
        "status": "pass" if all(sides.values()) else "fail",
        **sides,
        "contact_count": count,
        "basis": "active_PandaGripper_important_geom_groups_contact_object_collision_geoms",
    }


def _tracking_relative_pose(position, quaternion, eef_position, eef_quaternion):
    from research.gs_opd.repair.trajectory_tracking import (
        quaternion_conjugate,
        quaternion_multiply,
        quaternion_to_matrix,
    )

    object_quat = np.asarray(quaternion, dtype=np.float64)
    eef_quat = np.asarray(eef_quaternion, dtype=np.float64)
    rotation = quaternion_to_matrix(object_quat)
    relative_position_local = rotation.T @ (
        np.asarray(eef_position, dtype=np.float64) - np.asarray(position, dtype=np.float64)
    )
    relative_quaternion_local = quaternion_multiply(
        quaternion_conjugate(object_quat), eef_quat
    )
    return relative_position_local, relative_quaternion_local


def _active_environment_action_bounds(env, robot):
    """Read and cross-check the active robosuite and Panda 7-D action bounds."""
    task_env = getattr(env, "env", None)
    if task_env is None:
        raise ValueError("active robosuite environment is unavailable")
    try:
        action_low, action_high = task_env.action_spec
    except Exception as exc:
        raise ValueError(f"active environment action bounds unavailable: {exc}") from exc
    try:
        robot_low, robot_high = robot.action_limits
    except Exception as exc:
        raise ValueError(f"active Panda action limits unavailable: {exc}") from exc

    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)
    robot_low = np.asarray(robot_low, dtype=np.float64)
    robot_high = np.asarray(robot_high, dtype=np.float64)
    if any(
        bounds.shape != (7,) or not np.isfinite(bounds).all()
        for bounds in (action_low, action_high, robot_low, robot_high)
    ):
        raise ValueError("active Panda action bounds must be finite 7-D vectors")
    if (
        np.any(action_low > action_high)
        or np.any(robot_low > robot_high)
        or not np.allclose(action_low, robot_low, rtol=0.0, atol=1e-10)
        or not np.allclose(action_high, robot_high, rtol=0.0, atol=1e-10)
    ):
        raise ValueError("active environment action_spec disagrees with Panda action_limits")
    return action_low, action_high


def _validate_tracking_action(action, action_low, action_high):
    """Fail closed on malformed, non-finite, or out-of-range OSC actions."""
    action = np.asarray(action, dtype=np.float64)
    action_low = np.asarray(action_low, dtype=np.float64)
    action_high = np.asarray(action_high, dtype=np.float64)
    if action.shape != (7,) or not np.isfinite(action).all():
        raise ValueError("computed_action_must_be_finite_7d")
    if (
        action_low.shape != (7,)
        or action_high.shape != (7,)
        or not np.isfinite(action_low).all()
        or not np.isfinite(action_high).all()
        or np.any(action_low > action_high)
    ):
        raise ValueError("active_environment_action_bounds_must_be_finite_7d")
    if np.any(action < action_low - 1e-10) or np.any(action > action_high + 1e-10):
        raise ValueError("computed_action_outside_active_environment_bounds")
    return action


def _track_trajectory_candidate(
    candidate, snapshot, binding, options, candidate_index, reference_plan
):
    from research.gs_opd.repair.trajectory_tracking import (
        body_local_offset_from_world_point,
        body_local_point_to_world,
        quaternion_conjugate,
        quaternion_multiply,
        quaternion_to_matrix,
        quaternion_to_rotation_vector,
        world_pose_tracking_action,
    )

    sim = env.sim
    robot = env.robots[0]
    candidate_id = candidate["candidate_id"]
    result = {
        "candidate_id": candidate_id,
        "direction": candidate.get("direction"),
        "candidate_execution_mode": candidate.get("status", "geometric_candidate"),
        "goal_sphere_center_world_m": candidate.get(
            "goal_sphere_center_world_m"
        ),
        "control_points_world_m": candidate.get("control_points_world_m"),
        "geometric_points_world_m": candidate.get("points_world_m"),
        "status": "error",
        "requested_steps": 0,
        "sphere_center_body_local_offset_m": None,
        "start_object_root_pos_world_m": None,
        "start_object_root_quat_world_wxyz": None,
        "start_object_sphere_center_world_m": None,
        "executed_steps": 0,
        "execution_complete": False,
        "snapshot_start_state_match": False,
        "non_target_collision_detected": False,
        "non_target_collision_events": [],
        "unresolved_target_contact_detected": False,
        "unresolved_target_contact_events": [],
        "video_frames": [],
        "steps": [],
        "failure_reason": None,
        "task_feasibility": {
            "status": "unavailable",
            "scope": "local carried-object path only; no final task-success claim",
            "final_task_success_confirmed": False,
        },
    }
    monitor = _new_contact_monitor()
    try:
        _restore_snapshot(snapshot)
        restore_matches = _tracking_value_equal(snapshot, _capture_snapshot())
        result["snapshot_start_state_match"] = bool(restore_matches)
        diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
        result["start_state"] = diagnostic
        result["start_state_fingerprint"] = hashlib.sha256(
            np.asarray(diagnostic["mujoco_integration_state"], dtype=np.float64).tobytes()
        ).hexdigest()
        if not restore_matches:
            raise RuntimeError("candidate Snapshot restore did not reproduce full captured state")
        object_root_position, object_quaternion = _tracking_state_object(
            diagnostic, binding["body_id"]
        )
        points = np.asarray(candidate["points_world_m"], dtype=np.float64)
        center_start = np.asarray(binding["center_world_m"], dtype=np.float64)
        sphere_center_local = body_local_offset_from_world_point(
            center_start, object_root_position, object_quaternion
        )
        reconstructed_sphere_center = body_local_point_to_world(
            object_root_position, object_quaternion, sphere_center_local
        )
        if (
            np.linalg.norm(reconstructed_sphere_center - center_start)
            > options["state_tolerance_m"]
            or np.linalg.norm(points[0] - reconstructed_sphere_center)
            > options["state_tolerance_m"]
        ):
            raise ValueError("trajectory_start_does_not_match_snapshot_sphere_center")
        result["sphere_center_body_local_offset_m"] = sphere_center_local.tolist()
        result["start_object_root_pos_world_m"] = object_root_position.tolist()
        result["start_object_root_quat_world_wxyz"] = object_quaternion.tolist()
        result["start_object_sphere_center_world_m"] = reconstructed_sphere_center.tolist()
        if len(env.robots) != 1 or int(getattr(robot, "action_dim", 0)) != 7:
            raise ValueError("closed-loop tracker requires one 7-D Panda OSC_POSE action")
        if getattr(robot.controller, "name", type(robot.controller).__name__) != "OSC_POSE":
            raise ValueError("active Panda controller is not OSC_POSE")
        control_timestep = float(env.env.control_timestep)
        if not math.isclose(control_timestep, 0.05, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("active environment is not configured for 20 Hz OSC control")
        sampled = reference_plan
        result["eef_reference_pose_source"] = (
            "explicit_Mink_FK_world_pose_trajectory"
            if sampled.get("eef_reference_poses_world") is not None
            else "carried_sphere_and_live_grasp_transform"
        )
        if not math.isclose(
            float(sampled.get("control_timestep_s", float("nan"))),
            control_timestep,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("preflight_reference_plan_control_timestep_changed")
        references = sampled["reference_points_world_m"]
        result.update({
            "requested_steps": sampled["requested_steps"],
            "reference_path": references.tolist(),
            "path_length_m": sampled["path_length_m"],
            "reference_speed_limit_m_s": options["reference_speed_m_s"],
            "control_timestep_s": control_timestep,
        })
        result["start_contacts"] = _collect_contacts()
        gripper_groups, initial_gripper_closure = _tracking_gripper_sides(robot)
        start_sides = _tracking_contact_sides(
            result["start_contacts"], binding, gripper_groups
        )
        grasp = {
            "status": "unavailable",
            "reason": None,
            "initial_gripper_current_action": initial_gripper_closure.get(
                "raw_actuator_action"
            ),
            "initial_gripper_closure": initial_gripper_closure,
            "initial_finger_contacts": start_sides,
            "per_step": [],
            "maximum_relative_position_error_m": 0.0,
            "maximum_relative_rotation_error_rad": 0.0,
            "physical_grasp_confirmed_by_distance_only": False,
        }
        if initial_gripper_closure.get("status") != "pass":
            reason = initial_gripper_closure.get("reason") or "invalid_current_action"
            grasp.update({
                "status": "unavailable",
                "reason": f"initial_Panda_gripper_action_unavailable:{reason}",
            })
            result["grasp_evidence"] = grasp
            raise RuntimeError(f"initial_grasp_evidence_unavailable: {grasp['reason']}")
        if initial_gripper_closure.get("closed") is not True:
            reason = initial_gripper_closure.get("reason") or "close_command_below_threshold"
            grasp.update({
                "status": "fail",
                "reason": f"initial_Panda_gripper_is_not_closed:{reason}",
            })
            result["grasp_evidence"] = grasp
            raise RuntimeError(f"initial_grasp_evidence_unavailable: {grasp['reason']}")
        if start_sides.get("status") != "pass":
            grasp.update({"status": start_sides.get("status", "unavailable"), "reason": "initial_opposing_finger_contacts_missing"})
            result["grasp_evidence"] = grasp
            raise RuntimeError("initial_grasp_evidence_unavailable: both Panda fingers do not contact the carried object")
        video_stride = options.get("__video_capture_stride")

        def capture_video_frame(frame_step, *, force=False):
            if video_stride is None:
                return
            if (
                not force
                and frame_step != sampled["requested_steps"]
                and frame_step % video_stride != 0
            ):
                return
            rendered = _render_highres(256)
            image = rendered["images"]["rotated_180_rgb"]
            result["video_frames"].append(
                {
                    "step_index": int(frame_step),
                    "width": int(rendered["width"]),
                    "height": int(rendered["height"]),
                    "color_space": rendered["color_space"],
                    "orientation": "rotated_180_rgb",
                    "rgb_bytes": image["data"],
                }
            )

        capture_video_frame(0, force=True)
        robot_state = diagnostic["robot_states"][0]
        explicit_eef_poses = sampled.get("eef_reference_poses_world")
        if explicit_eef_poses is not None:
            explicit_eef_poses = np.asarray(explicit_eef_poses, dtype=np.float64)
            if (
                explicit_eef_poses.shape != (sampled["requested_steps"] + 1, 7)
                or not np.isfinite(explicit_eef_poses).all()
                or np.linalg.norm(
                    explicit_eef_poses[0, :3]
                    - np.asarray(robot_state["eef_pos"], dtype=np.float64)
                ) > options["state_tolerance_m"]
                or np.linalg.norm(
                    quaternion_to_rotation_vector(
                        quaternion_multiply(
                            explicit_eef_poses[0, 3:],
                            quaternion_conjugate(robot_state["eef_quat"]),
                        )
                    )
                ) > options["rotation_tolerance_rad"]
            ):
                raise ValueError("explicit_EEF_reference_start_does_not_match_Snapshot")
            quat_norms = np.linalg.norm(explicit_eef_poses[:, 3:], axis=1)
            if np.any(quat_norms < 1e-12) or np.any(np.abs(quat_norms - 1.0) > 1e-5):
                raise ValueError("explicit_EEF_reference_quaternions_are_invalid")
            explicit_eef_poses = explicit_eef_poses.copy()
            explicit_eef_poses[:, 3:] /= quat_norms[:, None]
        relative_position_local, relative_quaternion_local = _tracking_relative_pose(
            reconstructed_sphere_center,
            object_quaternion,
            robot_state["eef_pos"],
            robot_state["eef_quat"],
        )
        grasp["initial_eef_sphere_center_relative_position_body_local_m"] = (
            relative_position_local.tolist()
        )
        grasp["initial_eef_body_relative_quaternion_local_wxyz"] = (
            relative_quaternion_local.tolist()
        )
        grasp["status"] = "pass"
        workspace = _tracking_workspace(robot, _diagnostic_data_from_integration_state(sim))
        result["workspace_envelope"] = workspace
        initial_eef = np.asarray(robot_state["eef_pos"], dtype=np.float64)
        initial_workspace_check = _tracking_workspace_check(initial_eef, workspace)
        if initial_workspace_check["status"] != "pass":
            raise ValueError(
                initial_workspace_check.get("reason", "initial_EEF_workspace_unavailable")
            )
        initial_joint_check = _tracking_joint_limit_check(robot)
        if initial_joint_check["status"] != "pass":
            raise ValueError("initial_Panda_joint_limit_check_failed")
        initial_object_rotation = quaternion_to_matrix(object_quaternion)
        workspace_targets = (
            explicit_eef_poses[1:, :3]
            if explicit_eef_poses is not None
            else np.stack(
                [value + initial_object_rotation @ relative_position_local for value in references[1:]],
                axis=0,
            )
        )
        for target_eef in workspace_targets:
            workspace_check = _tracking_workspace_check(target_eef, workspace)
            if workspace_check["status"] != "pass":
                raise ValueError(workspace_check.get("reason", "trajectory_target_workspace_unavailable"))
        fields = _repair_context()["robots"][0]["fields"]
        if fields.get("input_ref_frame") not in (None, "world"):
            raise ValueError("active OSC input is not world-frame")
        orientation_limits = fields.get("orientation_limits")
        if orientation_limits is not None:
            bounds = np.asarray(orientation_limits, dtype=np.float64)
            if (
                bounds.shape != (2, 3)
                or not np.isfinite(bounds).all()
                or np.any(bounds != 0)
            ):
                raise ValueError("active OSC orientation clipping cannot be proven inactive")
        tracking_status = "pass"
        tracking_reason = None
        for step_index in range(sampled["requested_steps"]):
            live_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
            object_root_position, object_quaternion = _tracking_state_object(
                live_diagnostic, binding["body_id"]
            )
            sphere_center = body_local_point_to_world(
                object_root_position, object_quaternion, sphere_center_local
            )
            robot_state = live_diagnostic["robot_states"][0]
            actual_eef_position = np.asarray(robot_state["eef_pos"], dtype=np.float64)
            actual_eef_quaternion = np.asarray(robot_state["eef_quat"], dtype=np.float64)
            reference_position = np.asarray(references[step_index + 1], dtype=np.float64)
            object_rotation = quaternion_to_matrix(object_quaternion)
            if explicit_eef_poses is not None:
                target_eef_position = explicit_eef_poses[step_index + 1, :3]
                target_eef_quaternion = explicit_eef_poses[step_index + 1, 3:]
            else:
                target_eef_position = reference_position + object_rotation @ relative_position_local
                target_eef_quaternion = quaternion_multiply(object_quaternion, relative_quaternion_local)
            target_workspace_before_action = _tracking_workspace_check(
                target_eef_position, workspace
            )
            if target_workspace_before_action["status"] != "pass":
                raise ValueError(
                    target_workspace_before_action.get(
                        "reason", "trajectory_target_workspace_unavailable"
                    )
                )
            action = world_pose_tracking_action(
                actual_eef_position,
                actual_eef_quaternion,
                target_eef_position,
                target_eef_quaternion,
                fields,
                gripper_action=1.0,
            )
            action_low, action_high = _active_environment_action_bounds(env, robot)
            action = _validate_tracking_action(action, action_low, action_high)
            desired_output = np.concatenate((target_eef_position - actual_eef_position,
                quaternion_to_rotation_vector(quaternion_multiply(
                    target_eef_quaternion, quaternion_conjugate(actual_eef_quaternion)
                )))
            )
            actual_output = np.asarray(robot.controller.scale_action(action[:6]), dtype=np.float64)
            if (
                actual_output.shape != (6,)
                or not np.isfinite(actual_output).all()
                or not np.allclose(actual_output, desired_output, rtol=0.0, atol=1e-9)
            ):
                raise ValueError("active OSC scaling differs from the requested world pose delta")
            step_contacts = None
            try:
                obs, reward, done, _info = _run_env_step_with_contact_monitor(
                    action.astype(np.float64), step_index, monitor
                )
                last_reward = reward  # noqa: F841  # Preserve branch Snapshot bookkeeping.
                last_done = done  # noqa: F841  # Preserve branch Snapshot bookkeeping.
                diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
                object_root_position, object_quaternion = _tracking_state_object(
                    diagnostic, binding["body_id"]
                )
                sphere_center = body_local_point_to_world(
                    object_root_position, object_quaternion, sphere_center_local
                )
                robot_state = diagnostic["robot_states"][0]
                eef_position = np.asarray(robot_state["eef_pos"], dtype=np.float64)
                eef_quaternion = np.asarray(robot_state["eef_quat"], dtype=np.float64)
                sphere_center_error = float(
                    np.linalg.norm(sphere_center - reference_position)
                )
                eef_error = float(np.linalg.norm(eef_position - target_eef_position))
                current_rel_pos, current_rel_quat = _tracking_relative_pose(
                    sphere_center, object_quaternion, eef_position, eef_quaternion
                )
                relative_pos_error = float(np.linalg.norm(current_rel_pos - relative_position_local))
                relative_rot_delta = quaternion_multiply(
                    current_rel_quat, quaternion_conjugate(relative_quaternion_local)
                )
                relative_rot_error = float(np.linalg.norm(
                    quaternion_to_rotation_vector(relative_rot_delta)
                ))
                step_contacts = _collect_contacts()
                forbidden_scene_ids = set(
                    options.get("__forbidden_scene_body_ids", [])
                )
                unresolved_contact_ids = set(
                    options.get("__unresolved_contact_body_ids", [])
                )
                current_non_target_events = []
                current_unresolved_target_events = []
                for event in monitor.get("_events", {}).values():
                    if event.get("first_control_step") != step_index:
                        continue
                    body_pair = event.get("body_pair")
                    if not isinstance(body_pair, list):
                        continue
                    event_body_ids = {
                        body.get("id")
                        for body in body_pair
                        if isinstance(body, dict)
                        and isinstance(body.get("id"), int)
                        and not isinstance(body.get("id"), bool)
                    }
                    has_forbidden_scene_body = bool(
                        event_body_ids & forbidden_scene_ids
                    )
                    involves_robot_or_carried_book = bool(
                        event.get("robot_involved") is True
                        or binding.get("body_id") in event_body_ids
                    )
                    if has_forbidden_scene_body and involves_robot_or_carried_book:
                        current_non_target_events.append(event)
                    if (
                        event_body_ids & unresolved_contact_ids
                        and involves_robot_or_carried_book
                    ):
                        current_unresolved_target_events.append(event)
                if current_non_target_events:
                    result["non_target_collision_detected"] = True
                    result["non_target_collision_events"].extend(
                        current_non_target_events
                    )
                if current_unresolved_target_events:
                    result["unresolved_target_contact_detected"] = True
                    result["unresolved_target_contact_events"].extend(
                        current_unresolved_target_events
                    )
                finger_sides = _tracking_contact_sides(step_contacts, binding, gripper_groups)
                step_gripper_closure = _parse_tracking_gripper_closure(
                    getattr(robot.gripper, "current_action", None)
                )
                joint_limits = _tracking_joint_limit_check(robot)
                target_workspace = _tracking_workspace_check(target_eef_position, workspace)
                actual_workspace = _tracking_workspace_check(eef_position, workspace)
                grasp_step = {
                    **finger_sides,
                    "eef_sphere_center_relative_position_body_local_m": current_rel_pos.tolist(),
                    "eef_body_relative_quaternion_local_wxyz": current_rel_quat.tolist(),
                    "relative_position_error_m": relative_pos_error,
                    "relative_rotation_error_rad": relative_rot_error,
                    "gripper_current_action": step_gripper_closure.get(
                        "raw_actuator_action"
                    ),
                    "gripper_closure": step_gripper_closure,
                }
                grasp["per_step"].append(grasp_step)
                grasp["maximum_relative_position_error_m"] = max(
                    grasp["maximum_relative_position_error_m"], relative_pos_error
                )
                grasp["maximum_relative_rotation_error_rad"] = max(
                    grasp["maximum_relative_rotation_error_rad"], relative_rot_error
                )
                workspace_status = (
                    "pass"
                    if target_workspace["status"] == "pass" and actual_workspace["status"] == "pass"
                    else "fail"
                )
                reason = None
                if current_non_target_events:
                    reason = "actual_non_target_scene_collision"
                elif current_unresolved_target_events:
                    reason = "unresolved_target_contact_during_local_segment"
                elif joint_limits["status"] != "pass":
                    reason = "active_Panda_joint_limit_violated"
                elif workspace_status != "pass":
                    reason = "Panda_EEF_left_conservative_workspace"
                elif sphere_center_error > options["maximum_sphere_center_tracking_error_m"]:
                    reason = "carried_sphere_center_path_tracking_error_exceeded"
                elif eef_error > options["maximum_eef_tracking_error_m"]:
                    reason = "EEF_pose_tracking_error_exceeded"
                elif finger_sides.get("status") != "pass":
                    reason = "Panda_opposing_finger_grasp_contact_lost"
                elif step_gripper_closure.get("status") != "pass":
                    reason = (
                        "Panda_gripper_current_action_unavailable:"
                        f"{step_gripper_closure.get('reason') or 'invalid_state'}"
                    )
                elif step_gripper_closure.get("closed") is not True:
                    reason = step_gripper_closure.get("reason") or "Panda_gripper_close_command_below_0.9"
                elif relative_pos_error > options["maximum_grasp_relative_position_error_m"]:
                    reason = "gripper_object_relative_position_changed_beyond_limit"
                elif relative_rot_error > options["maximum_grasp_relative_rotation_error_rad"]:
                    reason = "gripper_object_relative_rotation_changed_beyond_limit"
                if reason is not None:
                    tracking_status = "fail"
                    tracking_reason = reason
                    grasp["status"] = "fail"
                    grasp["reason"] = reason
                step = {
                    "step_index": step_index,
                    "reference_sphere_center_world_m": reference_position.tolist(),
                    "actual_object_root_pos_world_m": object_root_position.tolist(),
                    "actual_object_sphere_center_world_m": sphere_center.tolist(),
                    "sphere_center_body_local_offset_m": sphere_center_local.tolist(),
                    "actual_object_root_quat_world_wxyz": object_quaternion.tolist(),
                    "actual_eef_position_world_m": eef_position.tolist(),
                    "actual_eef_quaternion_world_wxyz": eef_quaternion.tolist(),
                    "target_eef_position_world_m": target_eef_position.tolist(),
                    "target_eef_quaternion_world_wxyz": target_eef_quaternion.tolist(),
                    "sphere_center_tracking_error_m": sphere_center_error,
                    "eef_tracking_error_m": eef_error,
                    "grasp_relative_position_error_m": relative_pos_error,
                    "grasp_relative_rotation_error_rad": relative_rot_error,
                    "action": action.tolist(),
                    "requested_osc_output_world_delta": desired_output.tolist(),
                    "controller_scaled_osc_output": actual_output.tolist(),
                    "reward": float(reward),
                    "done": bool(done),
                    "state": diagnostic,
                    "contacts": step_contacts,
                    "non_target_collision_events": current_non_target_events,
                    "unresolved_target_contact_events": current_unresolved_target_events,
                    "substep_contact_coverage": monitor["control_steps"][-1],
                    "joint_limits": joint_limits,
                    "workspace": {
                        "status": workspace_status,
                        "target": target_workspace,
                        "actual": actual_workspace,
                    },
                    "grasp_contacts": finger_sides,
                    "tracking_status": "fail" if reason is not None else "pass",
                    "tracking_failure_reason": reason,
                }
                result["steps"].append(step)
                result["executed_steps"] += 1
                capture_video_frame(
                    step_index + 1, force=bool(reason is not None or done)
                )
                if bool(done):
                    tracking_status = "fail"
                    tracking_reason = "environment_terminated_during_trajectory_tracking"
                    grasp["status"] = "fail"
                    grasp["reason"] = tracking_reason
                    break
                if reason is not None:
                    break
            except Exception:
                raise
        result["execution_complete"] = (
            result["executed_steps"] == result["requested_steps"]
            and tracking_status == "pass"
            and all(not step["done"] for step in result["steps"])
        )
        endpoint_error = (
            result["steps"][-1]["sphere_center_tracking_error_m"]
            if result["steps"]
            else None
        )
        finite_rewards = all(
            isinstance(step.get("reward"), (int, float))
            and np.isfinite(float(step["reward"]))
            for step in result["steps"]
        )
        no_terminal_state = all(not step.get("done", True) for step in result["steps"])
        grasp_held = grasp.get("status") == "pass" and len(grasp["per_step"]) == len(result["steps"])
        endpoint_reached = (
            endpoint_error is not None
            and np.isfinite(endpoint_error)
            and endpoint_error <= options["maximum_sphere_center_tracking_error_m"]
        )
        result["task_feasibility"] = {
            "status": (
                "pass"
                if result["execution_complete"]
                and finite_rewards
                and no_terminal_state
                and grasp_held
                and endpoint_reached
                else "fail"
            ),
            "scope": "complete local carried-object path only; no final task-success claim",
            "final_task_success_confirmed": False,
            "complete_path_executed": result["execution_complete"],
            "no_terminal_done": no_terminal_state,
            "finite_reward_trace": finite_rewards,
            "opposing_finger_grasp_maintained": grasp_held,
            "endpoint_sphere_center_tracking_error_m": endpoint_error,
            "endpoint_within_tracking_tolerance": endpoint_reached,
        }
        result["closed_loop_tracking"] = {
            "status": "pass" if result["execution_complete"] else "fail",
            "reason": tracking_reason,
            "maximum_sphere_center_tracking_error_m": max(
                (step["sphere_center_tracking_error_m"] for step in result["steps"]), default=None
            ),
            "maximum_eef_tracking_error_m": max(
                (step["eef_tracking_error_m"] for step in result["steps"]), default=None
            ),
            "workspace_envelope": workspace,
        }
        result["grasp_evidence"] = grasp
        result["status"] = "ok"
        if not result["execution_complete"] and result["failure_reason"] is None:
            result["failure_reason"] = tracking_reason or "trajectory_tracking_incomplete"
    except Exception as exc:
        result["status"] = "error"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        if "grasp_evidence" not in result:
            result["grasp_evidence"] = {
                "status": "unavailable",
                "reason": result["failure_reason"],
                "physical_grasp_confirmed_by_distance_only": False,
            }
        result["closed_loop_tracking"] = {
            "status": "fail",
            "reason": result["failure_reason"],
        }
    finally:
        result["contact_monitoring"] = _contact_monitor_result(monitor)
    return result


def _track_trajectory_bundle(request):
    global obs, last_reward, last_done
    snapshot_id = request.get("snapshot_id")
    if not isinstance(snapshot_id, str) or snapshot_id not in snapshots:
        return {
            "status": "unavailable",
            "error": "unknown_or_expired_snapshot_id",
            "main_state_restored": False,
            "candidates": [],
        }
    target_snapshot = snapshots[snapshot_id]
    candidates = request.get("candidates")
    binding = request.get("carried_binding")
    options = request.get("options")
    video_capture_stride = request.get("video_capture_stride")
    forbidden_scene_body_ids = request.get("forbidden_scene_body_ids")
    unresolved_contact_body_ids = request.get("unresolved_contact_body_ids")
    if not isinstance(candidates, list) or not candidates or len(candidates) > 24:
        return {
            "status": "unavailable",
            "error": "candidate_count_must_be_in_1_to_24",
            "main_state_restored": False,
            "candidates": [],
        }
    if not isinstance(options, dict):
        return {
            "status": "unavailable",
            "error": "tracking_options_are_missing",
            "main_state_restored": False,
            "candidates": [],
        }
    if video_capture_stride is not None and (
        isinstance(video_capture_stride, bool)
        or not isinstance(video_capture_stride, int)
        or not 1 <= video_capture_stride <= 100
    ):
        return {
            "status": "unavailable",
            "error": "video_capture_stride_must_be_an_integer_in_1_to_100",
            "main_state_restored": False,
            "candidates": [],
        }
    branch_options = dict(options)
    if video_capture_stride is not None:
        branch_options["__video_capture_stride"] = video_capture_stride
    if (
        not isinstance(forbidden_scene_body_ids, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value >= int(env.sim.model._model.nbody)
            for value in forbidden_scene_body_ids
        )
        or len(set(forbidden_scene_body_ids)) != len(forbidden_scene_body_ids)
        or not isinstance(binding, dict)
        or set(forbidden_scene_body_ids).intersection(
            binding.get("body_ids", [])
            if isinstance(binding.get("body_ids"), list)
            else []
        )
    ):
        return {
            "status": "unavailable",
            "error": "forbidden_scene_body_ids_are_missing_or_malformed",
            "main_state_restored": False,
            "candidates": [],
        }
    branch_options["__forbidden_scene_body_ids"] = forbidden_scene_body_ids
    if (
        not isinstance(unresolved_contact_body_ids, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value >= int(env.sim.model._model.nbody)
            for value in unresolved_contact_body_ids
        )
        or len(set(unresolved_contact_body_ids)) != len(unresolved_contact_body_ids)
    ):
        return {
            "status": "unavailable",
            "error": "unresolved_contact_body_ids_are_missing_or_malformed",
            "main_state_restored": False,
            "candidates": [],
        }
    branch_options["__unresolved_contact_body_ids"] = unresolved_contact_body_ids
    expected_context = request.get("repair_context")
    try:
        from research.gs_opd.repair.trajectory_tracking import (
            MAX_CONTROL_STEPS_PER_CANDIDATE,
            MAX_REFERENCE_SPEED_M_S,
            MAX_TOTAL_CONTROL_STEPS,
            plan_bundle_reference_paths,
        )

        active_context = _repair_context()
        if (
            not isinstance(expected_context, dict)
            or expected_context.get("status") != "available"
            or json.dumps(expected_context, sort_keys=True, allow_nan=False)
            != json.dumps(active_context, sort_keys=True, allow_nan=False)
        ):
            raise ValueError("active_repair_context_changed_since_request")
        if len(env.robots) != 1:
            raise ValueError("only one active Panda arm is supported by this tracker")
        if not math.isclose(float(env.env.control_timestep), 0.05, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("active_environment_control_rate_is_not_20_Hz")
        if request.get("trajectory_state_id") != _tracking_snapshot_state_id(target_snapshot):
            raise ValueError("trajectory_bundle_and_snapshot_state_id_mismatch")
        if not _tracking_value_equal(_capture_snapshot(), target_snapshot):
            raise ValueError("active_main_state_does_not_match_the_decision_snapshot")
        _tracking_validate_binding(binding, request["trajectory_state_id"])
        max_steps = options.get("max_control_steps_per_candidate")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or not 1 <= max_steps <= MAX_CONTROL_STEPS_PER_CANDIDATE
        ):
            raise ValueError(
                "max_control_steps_per_candidate_must_be_in_1_to_"
                f"{MAX_CONTROL_STEPS_PER_CANDIDATE}"
            )
        max_total_steps = options.get("max_total_control_steps")
        if (
            isinstance(max_total_steps, bool)
            or not isinstance(max_total_steps, int)
            or not 1 <= max_total_steps <= MAX_TOTAL_CONTROL_STEPS
        ):
            raise ValueError(
                f"max_total_control_steps_must_be_in_1_to_{MAX_TOTAL_CONTROL_STEPS}"
            )
        for field in (
            "reference_speed_m_s",
            "maximum_sphere_center_tracking_error_m",
            "maximum_eef_tracking_error_m",
            "maximum_grasp_relative_position_error_m",
            "maximum_grasp_relative_rotation_error_rad",
            "state_tolerance_m",
            "rotation_tolerance_rad",
        ):
            value = options.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0
            ):
                raise ValueError(f"tracking option {field} must be finite and positive")
        if float(options["reference_speed_m_s"]) > MAX_REFERENCE_SPEED_M_S:
            raise ValueError(
                "reference_speed_m_s_cannot_exceed_"
                f"{MAX_REFERENCE_SPEED_M_S}_m_s"
            )
        candidate_ids = []
        candidate_paths = []
        for candidate in candidates:
            if (
                not isinstance(candidate, dict)
                or not isinstance(candidate.get("candidate_id"), str)
                or not candidate["candidate_id"]
                or candidate.get("status")
                not in (None, "geometric_candidate", "exploratory_inconclusive")
            ):
                raise ValueError("candidate_identity_or_status_is_malformed")
            points = np.asarray(candidate.get("points_world_m"), dtype=np.float64)
            goal = np.asarray(
                candidate.get("goal_sphere_center_world_m"), dtype=np.float64
            )
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or len(points) < 2
                or not np.isfinite(points).all()
                or goal.shape != (3,)
                or not np.isfinite(goal).all()
                or np.linalg.norm(points[-1] - goal) > 1e-6
            ):
                raise ValueError(
                    f"candidate_{candidate['candidate_id']}_path_is_malformed"
                )
            candidate_ids.append(candidate["candidate_id"])
            candidate_paths.append(points)
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate_ids_are_missing_or_duplicated")
        if all(
            candidate.get("eef_reference_poses_world") is not None
            for candidate in candidates
        ):
            reference_plans = []
            planned_total_steps = 0
            for candidate, points in zip(candidates, candidate_paths):
                from research.gs_opd.repair.trajectory_tracking import (
                    _parse_eef_reference_poses,
                )

                eef_poses = _parse_eef_reference_poses(
                    candidate["eef_reference_poses_world"],
                    expected_count=len(points),
                )
                steps = len(points) - 1
                reason = None
                if steps > max_steps:
                    reason = "explicit_EEF_reference_exceeds_candidate_step_budget"
                elif planned_total_steps + steps > max_total_steps:
                    reason = "explicit_EEF_reference_exceeds_total_step_budget"
                accepted = reason is None
                reference_plans.append(
                    {
                        "reference_points_world_m": points,
                        "eef_reference_poses_world": eef_poses if accepted else None,
                        "path_length_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum()),
                        "requested_steps": steps,
                        "budget_status": "accepted" if accepted else "rejected",
                        "budget_rejection_reason": reason,
                        "maximum_reference_displacement_m": float(np.linalg.norm(np.diff(points, axis=0), axis=1).max()),
                        "reference_speed_limit_m_s": None,
                        "control_timestep_s": float(env.env.control_timestep),
                    }
                )
                if accepted:
                    planned_total_steps += steps
        else:
            if any(
                candidate.get("eef_reference_poses_world") is not None
                for candidate in candidates
            ):
                raise ValueError("explicit_EEF_references_must_be_supplied_for_all_candidates")
            reference_plans, planned_total_steps = plan_bundle_reference_paths(
                candidate_paths,
                speed_m_s=float(options["reference_speed_m_s"]),
                control_timestep_s=float(env.env.control_timestep),
                maximum_steps_per_candidate=max_steps,
                maximum_total_steps=max_total_steps,
            )
        reference_plans_by_id = dict(zip(candidate_ids, reference_plans))
    except Exception as exc:
        return {
            "status": "unavailable",
            "error": f"Trajectory tracking preflight failed: {type(exc).__name__}: {exc}",
            "main_state_restored": True,
            "candidates": [],
        }

    caller_snapshot = _capture_snapshot()
    caller_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
    result = {
        "operation": "track_trajectory_bundle",
        "status": "incomplete",
        "snapshot_id": snapshot_id,
        "trajectory_state_id": request["trajectory_state_id"],
        "planned_total_control_steps": planned_total_steps,
        "maximum_total_control_steps": max_total_steps,
        "candidates": [],
        "main_state_restored": False,
        "main_restore_verified": False,
    }
    try:
        for candidate_index, candidate in enumerate(candidates):
            item = {
                "candidate_id": candidate.get("candidate_id") if isinstance(candidate, dict) else None,
                "status": "error",
                "requested_steps": 0,
                "executed_steps": 0,
                "execution_complete": False,
                "snapshot_start_state_match": False,
                "steps": [],
            }
            try:
                if not isinstance(candidate, dict) or not isinstance(candidate.get("candidate_id"), str):
                    raise ValueError("candidate identity is malformed")
                if candidate.get("status") not in (
                    None,
                    "geometric_candidate",
                    "exploratory_inconclusive",
                ):
                    raise ValueError("candidate has an unsupported execution status")
                item = _track_trajectory_candidate(
                    candidate,
                    target_snapshot,
                    binding,
                    branch_options,
                    candidate_index,
                    reference_plans_by_id[candidate["candidate_id"]],
                )
            except Exception as exc:
                item["status"] = "error"
                item["failure_reason"] = f"{type(exc).__name__}: {exc}"
                item.setdefault("steps", [])
            result["candidates"].append(item)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _restore_snapshot(caller_snapshot)
            restored_snapshot = _capture_snapshot()
            restored_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
            state_equal = _tracking_value_equal(caller_snapshot, restored_snapshot)
            diagnostic_equal = _tracking_value_equal(caller_diagnostic, restored_diagnostic)
            result["main_restore_verified"] = bool(state_equal and diagnostic_equal)
            result["main_state_restored"] = bool(state_equal and diagnostic_equal)
            if not result["main_state_restored"]:
                result["main_restore_error"] = "complete_snapshot_or_canonical_diagnostic_mismatch"
        except Exception as exc:
            result["main_state_restored"] = False
            result["main_restore_verified"] = False
            result["main_restore_error"] = f"{type(exc).__name__}: {exc}"
    result["status"] = (
        "complete"
        if result["main_state_restored"]
        and len(result["candidates"]) == len(candidates)
        else "incomplete"
    )
    return result


def _snapshot_diagnostics(canonicalize_kinematics=False):
    sim = env.sim
    task_env = env.env
    diagnostic_data = (
        _diagnostic_data_from_integration_state(sim)
        if canonicalize_kinematics
        else sim.data._data
    )
    diagnostic_body_xpos = getattr(diagnostic_data, "xpos", None)
    if diagnostic_body_xpos is None:
        diagnostic_body_xpos = diagnostic_data.body_xpos
    diagnostic_body_xquat = getattr(diagnostic_data, "xquat", None)
    if diagnostic_body_xquat is None:
        diagnostic_body_xquat = diagnostic_data.body_xquat
    robot_states = []
    for index, robot in enumerate(env.robots):
        robot_state = {
            "robot_index": index,
            "joint_qpos": np.asarray(sim.data.qpos[robot._ref_joint_pos_indexes]).tolist(),
            "joint_qvel": np.asarray(sim.data.qvel[robot._ref_joint_vel_indexes]).tolist(),
            "eef_pos": np.asarray(diagnostic_data.site_xpos[robot.eef_site_id]).tolist(),
        }
        eef_rotation = np.asarray(
            diagnostic_data.site_xmat[robot.eef_site_id], dtype=np.float64
        ).reshape(3, 3)
        eef_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(eef_quaternion, eef_rotation.reshape(-1))
        robot_state["eef_quat"] = eef_quaternion.tolist()
        if getattr(robot, "has_gripper", False):
            robot_state["gripper_qpos"] = np.asarray(
                sim.data.qpos[robot._ref_gripper_joint_pos_indexes]
            ).tolist()
        robot_states.append(robot_state)

    object_states = {}
    for mapping_name in ("objects_dict", "fixtures_dict"):
        for object_name, obj in getattr(task_env, mapping_name, {}).items():
            root_body = getattr(obj, "root_body", None)
            if root_body is None:
                continue
            body_id = sim.model.body_name2id(root_body)
            joints = []
            for joint in getattr(obj, "joints", ()):
                address = sim.model.get_joint_qpos_addr(joint)
                if isinstance(address, tuple):
                    joint_qpos = sim.data.qpos[address[0] : address[1]]
                else:
                    joint_qpos = sim.data.qpos[address]
                joints.extend(np.asarray(joint_qpos).reshape(-1).tolist())
            joint_velocities = []
            for joint in getattr(obj, "joints", ()):
                address = sim.model.get_joint_qvel_addr(joint)
                if isinstance(address, tuple):
                    joint_qvel = sim.data.qvel[address[0] : address[1]]
                else:
                    joint_qvel = sim.data.qvel[address]
                joint_velocities.extend(np.asarray(joint_qvel).reshape(-1).tolist())
            object_states[f"{mapping_name}:{object_name}"] = {
                "body_id": int(body_id),
                "body_name": str(root_body),
                "root_pos": np.asarray(diagnostic_body_xpos[body_id]).tolist(),
                "root_quat": np.asarray(diagnostic_body_xquat[body_id]).tolist(),
                "joint_qpos": joints,
                "joint_qvel": joint_velocities,
            }

    return {
        "mujoco_integration_state": _mujoco_integration_state(sim).tolist(),
        "robot_states": robot_states,
        "object_states": object_states,
        "kinematics_source": (
            "forwarded_copy_of_mujoco_integration_state"
            if canonicalize_kinematics
            else "live_mujoco_data_cache"
        ),
    }


def _collect_contacts():
    """Return MuJoCo contacts without conflating no contact with unavailable data."""
    sim = env.sim
    try:
        contact_count = int(sim.data.ncon)
    except Exception as exc:
        return {
            "status": "unavailable",
            "contacts": [],
            "error": f"Cannot read MuJoCo contact count: {type(exc).__name__}: {exc}",
        }

    if contact_count == 0:
        return {"status": "no_contact", "contacts": []}

    contacts = []
    for contact_index in range(contact_count):
        try:
            contact = sim.data.contact[contact_index]
            geom1_id = int(contact.geom1)
            geom2_id = int(contact.geom2)
            geom1_name = sim.model.geom_id2name(geom1_id)
            geom2_name = sim.model.geom_id2name(geom2_id)
            body1_id = int(sim.model.geom_bodyid[geom1_id])
            body2_id = int(sim.model.geom_bodyid[geom2_id])
            body1_name = sim.model.body_id2name(body1_id)
            body2_name = sim.model.body_id2name(body2_id)
            item = {
                "contact_index": contact_index,
                "geom1": {"id": geom1_id, "name": geom1_name},
                "geom2": {"id": geom2_id, "name": geom2_name},
                "body1": {"id": body1_id, "name": body1_name},
                "body2": {"id": body2_id, "name": body2_name},
                "position": np.asarray(contact.pos, dtype=np.float64).tolist(),
                "signed_contact_distance_m": float(contact.dist),
            }
            try:
                contact_frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
                normal = contact_frame[0].copy()
                normal_norm = float(np.linalg.norm(normal))
                if not np.isfinite(normal).all() or normal_norm <= 1e-12:
                    raise ValueError("contact normal is non-finite or degenerate")
                item["normal_status"] = "available"
                item["normal_geom1_to_geom2"] = (normal / normal_norm).tolist()
            except Exception as exc:
                item["normal_status"] = "unavailable"
                item["normal_geom1_to_geom2"] = None
                item["normal_error"] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            return {
                "status": "unavailable",
                "contacts": contacts,
                "error": (
                    f"Cannot inspect MuJoCo contact {contact_index}: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

        # mj_contactForce only has a solver force when this contact has an
        # address in the constraint system. Keep the geometric contact record
        # even when its force cannot be queried.
        try:
            efc_address = int(contact.efc_address)
            if efc_address < 0:
                raise RuntimeError("contact has no constraint force address")
            force = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(
                sim.model._model,
                sim.data._data,
                contact_index,
                force,
            )
            item["force_status"] = "available"
            item["force_frame"] = "contact"
            item["force_and_torque"] = force.tolist()
        except Exception as exc:
            item["force_status"] = "unavailable"
            item["force_frame"] = None
            item["force_and_torque"] = None
            item["force_error"] = f"{type(exc).__name__}: {exc}"
        contacts.append(item)

    return {"status": "contact", "contacts": contacts}


def _robot_body_ids():
    """Resolve robot bodies from the compiled model's robot body hierarchy."""
    sim = env.sim
    model = sim.model._model
    robot_body_ids = set()
    for robot in env.robots:
        root_name = robot.robot_model.root_body
        root_id = int(sim.model.body_name2id(root_name))
        for body_id in range(int(model.nbody)):
            current_id = body_id
            while current_id > 0:
                if current_id == root_id:
                    robot_body_ids.add(body_id)
                    break
                parent_id = int(model.body_parentid[current_id])
                if parent_id == current_id:
                    break
                current_id = parent_id
    return robot_body_ids


def _robot_joint_limit_inventory(raw_tolerance_policy=None):
    """Return limited robot joints with qpos units and raw live measurements."""
    model = env.sim.model._model
    robot_bodies = _robot_body_ids()
    tolerance_policy = normalize_joint_limit_tolerance_policy(
        raw_tolerance_policy or JOINT_LIMIT_TOLERANCE_POLICY
    )
    joints = []
    for joint_id in range(int(model.njnt)):
        joint_type = int(model.jnt_type[joint_id])
        if (
            int(model.jnt_bodyid[joint_id]) not in robot_bodies
            or not bool(model.jnt_limited[joint_id])
            or joint_type not in (
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
        ):
            continue
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
        kind = joint_type_name(
            joint_type,
            int(mujoco.mjtJoint.mjJNT_HINGE),
            int(mujoco.mjtJoint.mjJNT_SLIDE),
        )
        qpos = float(env.sim.data.qpos[int(model.jnt_qposadr[joint_id])])
        lower, upper = (float(value) for value in model.jnt_range[joint_id])
        joints.append({
            "joint_id": joint_id,
            "joint_name": name,
            "mujoco_jnt_type": joint_type,
            "joint_type": kind,
            "unit": "rad" if kind == "hinge" else "m",
            "limited": True,
            **evaluate_joint_limit(
                joint_name=name,
                joint_type=kind,
                qpos=qpos,
                lower=lower,
                upper=upper,
                policy=tolerance_policy,
            ),
        })
    return {
        "operation": "robot_joint_limit_inventory",
        "status": "complete" if joints else "unavailable",
        "joint_limit_tolerance_policy": tolerance_policy,
        "joints": joints,
        "physical_state_unchanged": True,
    }


def _build_geometry_inventory(
    model, geom_count, robot_body_ids, robot_classification_error=None
):
    """Build a read-only inventory from the active MuJoCo model wrapper."""
    geometries = []
    has_lookup_errors = False
    body_count = int(model._model.nbody)
    compiled_model = model._model
    explicit_pair_geoms = set()
    for pair_index in range(int(compiled_model.npair)):
        explicit_pair_geoms.add(int(compiled_model.pair_geom1[pair_index]))
        explicit_pair_geoms.add(int(compiled_model.pair_geom2[pair_index]))
    bodies = []
    has_body_lookup_errors = False

    for body_id in range(body_count):
        body = {
            "body_id": body_id,
            "body_name": None,
            "parent_body_id": None,
            "parent_body_name": None,
            "geom_ids": [],
        }
        try:
            body["body_name"] = model.body_id2name(body_id)
        except Exception as exc:
            body["body_name_error"] = f"{type(exc).__name__}: {exc}"
            has_body_lookup_errors = True
        try:
            parent_id = int(model._model.body_parentid[body_id])
            body["parent_body_id"] = parent_id
            if 0 <= parent_id < body_count:
                body["parent_body_name"] = model.body_id2name(parent_id)
        except Exception as exc:
            body["parent_body_error"] = f"{type(exc).__name__}: {exc}"
            has_body_lookup_errors = True
        bodies.append(body)

    for geom_id in range(geom_count):
        item = {
            "geom_id": geom_id,
            "geom_name": None,
            "body_id": None,
            "body_name": None,
            "category": "unknown",
            "collision_enabled": False,
            "contype": None,
            "conaffinity": None,
        }
        errors = {}

        try:
            item["geom_name"] = model.geom_id2name(geom_id)
        except Exception as exc:
            errors["geom_name"] = f"{type(exc).__name__}: {exc}"

        try:
            body_id = int(model.geom_bodyid[geom_id])
            item["body_id"] = body_id
            if 0 <= body_id < body_count:
                try:
                    item["body_name"] = model.body_id2name(body_id)
                except Exception as exc:
                    errors["body_name"] = f"{type(exc).__name__}: {exc}"
            else:
                errors["body_id"] = f"Body ID {body_id} is outside model range"

            if 0 <= body_id < body_count and robot_body_ids is not None:
                item["category"] = (
                    "robot" if body_id in robot_body_ids else "scene"
                )
            if 0 <= body_id < body_count:
                bodies[body_id]["geom_ids"].append(geom_id)
            contype = int(compiled_model.geom_contype[geom_id])
            conaffinity = int(compiled_model.geom_conaffinity[geom_id])
            item["contype"] = contype
            item["conaffinity"] = conaffinity
            item["collision_enabled"] = bool(
                contype != 0 or conaffinity != 0 or geom_id in explicit_pair_geoms
            )
        except Exception as exc:
            errors["body_id"] = f"{type(exc).__name__}: {exc}"

        if errors:
            item["errors"] = errors
            has_lookup_errors = True
        geometries.append(item)

    classification_available = robot_body_ids is not None
    status = (
        "complete"
        if classification_available
        and not has_lookup_errors
        and not has_body_lookup_errors
        else "partial"
    )
    result = {
        "status": status,
        "robot_classification_status": (
            "available" if classification_available else "unavailable"
        ),
        "geom_count": geom_count,
        "body_count": body_count,
        "body_inventory_status": (
            "complete" if not has_body_lookup_errors else "partial"
        ),
        "bodies": bodies,
        "geometries": geometries,
    }
    if robot_classification_error is not None:
        result["robot_classification_error"] = robot_classification_error
    return result


def _geometry_inventory():
    """Return current model geometry metadata without changing simulation state."""
    sim = env.sim
    try:
        geom_count = int(sim.model._model.ngeom)
    except Exception as exc:
        return {
            "status": "unavailable",
            "robot_classification_status": "unavailable",
            "geom_count": None,
            "geometries": [],
            "error": f"Cannot read MuJoCo geometry count: {type(exc).__name__}: {exc}",
        }

    robot_body_ids = None
    robot_classification_error = None
    try:
        robot_body_ids = _robot_body_ids()
    except Exception as exc:
        robot_classification_error = f"{type(exc).__name__}: {exc}"

    return _build_geometry_inventory(
        sim.model,
        geom_count,
        robot_body_ids,
        robot_classification_error=robot_classification_error,
    )


def _task_semantic_geometry_bindings():
    """Bind registered BDDL objects to every compiled collision Geom."""
    model = env.sim.model._model
    inventory = _geometry_inventory()
    if inventory.get("status") != "complete":
        raise ValueError("complete_geometry_inventory_unavailable")
    bindings = {}
    for mapping_name in ("objects_dict", "fixtures_dict"):
        for object_name, obj in getattr(env.env, mapping_name, {}).items():
            root_name = getattr(obj, "root_body", None)
            if not isinstance(root_name, str) or not root_name:
                raise ValueError(f"registered_task_object_{object_name}_has_no_root_body")
            root_id = int(env.sim.model.body_name2id(root_name))
            if root_id <= 0 or root_id >= int(model.nbody):
                raise ValueError(f"registered_task_object_{object_name}_root_body_is_invalid")
            subtree = _planning_body_subtree(model, root_id)
            collision_geoms = _collision_geom_ids(subtree)
            root_geoms = _collision_geom_ids({root_id})
            semantic_geoms = {
                geom_id
                for geom_id in range(int(model.ngeom))
                if int(model.geom_bodyid[geom_id]) in subtree
            }
            root_semantic_geoms = {
                geom_id
                for geom_id in range(int(model.ngeom))
                if int(model.geom_bodyid[geom_id]) == root_id
            }
            body_names = {
                int(body_id): env.sim.model.body_id2name(int(body_id))
                for body_id in sorted(subtree)
            }
            collision_body_ids = sorted(
                {int(model.geom_bodyid[geom_id]) for geom_id in collision_geoms}
            )
            geometry_names = {
                int(item["geom_id"]): item["geom_name"]
                for item in inventory["geometries"]
                if item.get("geom_id") in semantic_geoms
            }
            if set(geometry_names) != semantic_geoms:
                raise ValueError(f"registered_task_object_{object_name}_geometry_binding_is_incomplete")
            bindings[object_name] = {
                "object_id": str(object_name),
                "mapping_name": mapping_name,
                "root_body_id": root_id,
                "root_body_name": root_name,
                "root_geom_ids": sorted(root_geoms),
                "root_geom_names": [geometry_names[item] for item in sorted(root_geoms)],
                "root_semantic_geom_ids": sorted(root_semantic_geoms),
                "body_ids": sorted(subtree),
                "body_names": [body_names[item] for item in sorted(subtree)],
                "collision_body_ids": collision_body_ids,
                "collision_body_names": [body_names[item] for item in collision_body_ids],
                "geom_ids": sorted(collision_geoms),
                "geom_names": [geometry_names[item] for item in sorted(collision_geoms)],
                "semantic_geom_ids": sorted(semantic_geoms),
                "semantic_geom_names": [geometry_names[item] for item in sorted(semantic_geoms)],
                "collision_geometry_complete": bool(collision_geoms),
                "geometry_presence": (
                    "compiled_geometry_present"
                    if semantic_geoms
                    else "registered_semantic_body_has_no_compiled_geometry"
                ),
            }
    if not bindings:
        raise ValueError("active_task_has_no_registered_semantic_objects")
    return {
        "status": "complete",
        "bindings": bindings,
        "model_geometry_sha256": _planning_model_signature(model),
    }


def _planning_model_signature(model):
    """Fingerprint compiled geometry and frames used by this read-only query."""
    import hashlib

    digest = hashlib.sha256()
    fields = (
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
    for name in fields:
        values = np.ascontiguousarray(getattr(model, name))
        digest.update(name.encode())
        digest.update(str((values.dtype, values.shape)).encode())
        digest.update(values.tobytes())
    digest.update(bytes(model.names))
    digest.update(np.asarray(model.opt.gravity, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _planning_model_signature_details(model):
    """Expose component digests for zero-action model compatibility audits."""
    import importlib.metadata
    import platform

    fields = (
        "body_parentid", "body_pos", "body_quat", "body_mocapid", "qpos0",
        "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_qposadr",
        "geom_bodyid", "geom_type", "geom_dataid", "geom_pos", "geom_quat",
        "geom_size", "geom_rbound", "geom_contype", "geom_conaffinity",
        "geom_margin", "pair_geom1", "pair_geom2", "pair_margin",
        "mesh_vertadr", "mesh_vertnum", "mesh_vert", "mesh_face",
        "site_bodyid", "site_pos", "site_quat", "site_size", "site_type",
    )
    components = {}
    for name in fields:
        values = np.ascontiguousarray(getattr(model, name))
        components[name] = {
            "dtype": str(values.dtype),
            "shape": list(values.shape),
            "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
        }
    components["names"] = {
        "dtype": "bytes",
        "shape": [len(model.names)],
        "sha256": hashlib.sha256(bytes(model.names)).hexdigest(),
    }
    gravity = np.asarray(model.opt.gravity, dtype=np.float64)
    components["opt.gravity"] = {
        "dtype": str(gravity.dtype),
        "shape": list(gravity.shape),
        "sha256": hashlib.sha256(gravity.tobytes()).hexdigest(),
        "value": gravity.tolist(),
    }
    distributions = {}
    for distribution in ("mujoco", "robosuite", "libero", "numpy", "gym"):
        try:
            distributions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            distributions[distribution] = None
    return {
        "model_geometry_sha256": _planning_model_signature(model),
        "component_digests": components,
        "runtime": {
            "python": platform.python_version(),
            "mujoco_api": getattr(mujoco, "__version__", None),
            "numpy_api": np.__version__,
            "distributions": distributions,
            "bridge_source": str(Path(__file__).resolve()),
            "bddl_path": scene_metadata.get("bddl_path"),
            "bddl_sha256": scene_metadata.get("bddl_sha256"),
        },
    }


def _planning_cache_signature(data):
    import hashlib

    digest = hashlib.sha256()
    for name in ("geom_xpos", "geom_xmat", "xpos", "xquat", "site_xpos", "site_xmat"):
        digest.update(np.ascontiguousarray(getattr(data, name)).tobytes())
    return digest.hexdigest()


def _planning_registered_objects(model_wrapper, task_env):
    entries = {}
    for mapping in ("objects_dict", "fixtures_dict"):
        for key, obj in getattr(task_env, mapping, {}).items():
            root = getattr(obj, "root_body", None)
            if not isinstance(root, str):
                raise ValueError(f"Task object {key!r} has no physical root Body")
            body_id = int(model_wrapper.body_name2id(root))
            if body_id <= 0 or body_id in entries:
                raise ValueError("Task object roots are invalid or ambiguous")
            entries[body_id] = {"key": key, "object": obj, "body_name": root}
    return entries


def _planning_body_subtree(model, root):
    if root <= 0 or root >= int(model.nbody):
        raise ValueError("Planning root Body is outside the active model")
    result = set()
    for body in range(1, int(model.nbody)):
        current, visited = body, set()
        while current:
            if current in visited or current < 0 or current >= int(model.nbody):
                raise ValueError("Compiled Body hierarchy is invalid")
            visited.add(current)
            if current == root:
                result.add(body)
                break
            current = int(model.body_parentid[current])
    return result


def _planning_component_sphere(model, scratch, geom_id, pair_margin):
    """Use the compiled broadphase ball, checking finite primitives/mesh bounds."""
    kind = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    center = np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64)
    rbound = float(model.geom_rbound[geom_id])
    if (
        center.shape != (3,)
        or size.shape != (3,)
        or not np.isfinite(center).all()
        or not np.isfinite(size).all()
        or np.any(size < 0)
        or not np.isfinite(rbound)
        or rbound <= 0
    ):
        raise ValueError(
            f"Geom {geom_id} has no finite positive compiled bounding ball"
        )
    geom = mujoco.mjtGeom
    sources = "compiled geom_rbound centered at scratch geom_xpos"
    if kind == int(geom.mjGEOM_SPHERE):
        checked_radius = float(size[0])
    elif kind == int(geom.mjGEOM_CAPSULE):
        checked_radius = float(size[0] + size[1])
    elif kind == int(geom.mjGEOM_CYLINDER):
        checked_radius = float(np.hypot(size[0], size[1]))
    elif kind == int(geom.mjGEOM_ELLIPSOID):
        checked_radius = float(size.max())
    elif kind == int(geom.mjGEOM_BOX):
        checked_radius = float(np.linalg.norm(size))
    elif kind == int(geom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        if not 0 <= mesh_id < int(model.nmesh):
            raise ValueError(f"Geom {geom_id} mesh association is invalid")
        address, count = (
            int(model.mesh_vertadr[mesh_id]),
            int(model.mesh_vertnum[mesh_id]),
        )
        vertices = np.asarray(
            model.mesh_vert[address : address + count], dtype=np.float64
        )
        if (
            count <= 0
            or vertices.shape != (count, 3)
            or not np.isfinite(vertices).all()
        ):
            raise ValueError(f"Geom {geom_id} compiled mesh vertices are incomplete")
        checked_radius = float(np.linalg.norm(vertices, axis=1).max())
        sources += "; checked against all compiled mesh vertices, not visible pixels"
    else:
        raise ValueError(
            f"Geom {geom_id} type {kind} has no supported finite enclosure"
        )
    margin = max(float(model.geom_margin[geom_id]), pair_margin.get(geom_id, 0.0))
    if (
        not np.isfinite(checked_radius)
        or checked_radius <= 0
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError(f"Geom {geom_id} compiled bound or contact margin is invalid")
    padding = 1e-10 * max(1.0, float(np.abs(center).max()), rbound, checked_radius)
    radius = max(rbound, checked_radius) + margin + padding
    if not np.isfinite(radius):
        raise ValueError(f"Geom {geom_id} enclosure overflow")
    return {
        "geom_id": geom_id,
        "type_id": kind,
        "center_world_m": center.tolist(),
        "geom_rbound_m": rbound,
        "checked_geometry_radius_m": checked_radius,
        "contact_margin_m": margin,
        "numerical_padding_m": padding,
        "radius_m": float(np.nextafter(radius, np.inf)),
        "source": sources,
    }


def _planning_object_sphere(
    binding, inventory, model_wrapper, scratch, registered, state_id
):
    model = model_wrapper._model
    if not isinstance(binding, dict):
        raise ValueError(
            "Planning binding must be a current validated Body/Geom mapping"
        )
    object_id, body = binding.get("object_id"), binding.get("body_id")
    if (
        not isinstance(object_id, str)
        or not object_id
        or isinstance(body, bool)
        or not isinstance(body, int)
    ):
        raise ValueError("Planning binding object/Body ID is invalid")
    if body not in registered or registered[body]["body_name"] != binding.get(
        "body_name"
    ):
        raise ValueError("Bound Body is not a verified registered task-object root")
    geom_count = int(model.ngeom)
    bound_ids, error = _validated_geom_id_list(
        binding.get("geom_ids"), "binding Geoms", geom_count
    )
    names = binding.get("geom_names")
    if error or not isinstance(names, list) or len(names) != len(bound_ids):
        raise ValueError(error or "Planning binding Geom names are incomplete")
    subtree = _planning_body_subtree(model, body)
    by_id = {entry["geom_id"]: entry for entry in inventory["geometries"]}
    for geom_id, name in zip(bound_ids, names):
        current = by_id[geom_id]
        if (
            current["body_id"] != body
            or current["geom_name"] != name
            or current["body_name"] != binding["body_name"]
        ):
            raise ValueError(
                "Semantic Geom identity conflicts with the current Geometry Inventory"
            )
    all_ids = [g for g in range(geom_count) if int(model.geom_bodyid[g]) in subtree]
    if not all_ids or any(by_id[g].get("category") != "scene" for g in all_ids):
        raise ValueError(
            "Full object subtree is empty or robot/scene classification is unavailable"
        )
    pair_margin = {}
    for index in range(int(model.npair)):
        a, b = int(model.pair_geom1[index]), int(model.pair_geom2[index])
        margin = float(model.pair_margin[index])
        if (
            not (0 <= a < geom_count and 0 <= b < geom_count)
            or not np.isfinite(margin)
            or margin < 0
        ):
            raise ValueError("Compiled explicit Contact Pair metadata is incomplete")
        for geom_id in (a, b):
            pair_margin[geom_id] = max(pair_margin.get(geom_id, 0.0), margin)
    collision_ids = [
        g
        for g in all_ids
        if int(model.geom_contype[g]) != 0
        or int(model.geom_conaffinity[g]) != 0
        or g in pair_margin
    ]
    if not collision_ids:
        raise ValueError("Registered object has no confirmed collision Geoms")
    components = [
        _planning_component_sphere(model, scratch, g, pair_margin)
        for g in collision_ids
    ]
    declared = getattr(registered[body]["object"], "contact_geoms", None)
    if declared is not None:
        declared_ids = {int(model_wrapper.geom_name2id(name)) for name in declared}
        if not declared_ids.issubset(set(all_ids)):
            raise ValueError(
                "Task object's declared collision Geoms escape its registered subtree"
            )
    center = np.asarray(scratch.xpos[body], dtype=np.float64)
    if center.shape != (3,) or not np.isfinite(center).all():
        raise ValueError("Canonical object root position is unavailable")
    radius = max(
        float(np.linalg.norm(np.asarray(item["center_world_m"]) - center))
        + item["radius_m"]
        for item in components
    )
    radius = float(
        np.nextafter(
            radius + 1e-10 * max(1.0, radius, float(np.abs(center).max())), np.inf
        )
    )
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Merged object sphere is invalid")
    for item in components:
        g = item["geom_id"]
        item.update({
            "geom_name": by_id[g]["geom_name"],
            "body_id": by_id[g]["body_id"],
            "body_name": by_id[g]["body_name"],
        })
    return {
        "object_id": object_id,
        "body_id": body,
        "body_name": binding["body_name"],
        "registered_task_object": registered[body]["key"],
        "body_ids": sorted(subtree),
        "semantic_binding_geom_ids": bound_ids,
        "semantic_binding_geom_names": names,
        "geom_ids": collision_ids,
        "geom_names": [by_id[g]["geom_name"] for g in collision_ids],
        "excluded_visual_only_geom_ids": sorted(set(all_ids) - set(collision_ids)),
        "components": components,
        "center_world_m": center.tolist(),
        "radius_m": radius,
        "center_definition": "canonical registered root position; radius encloses every collision component",
        "covers_all_associated_collision_geoms": True,
        "binding_verified": True,
        "enclosure_verified": True,
        "state_id": state_id,
        "binding_source": "registered task-object root and current Geometry Inventory identity checks",
        "enclosure_source": "union of all subtree compiled Geom balls including explicit pairs and contact margins",
        "future_articulation_or_motion_verified": False,
    }


def _planning_goal_diagnostics(task_env, model_wrapper, scratch, registered, carried):
    """Audit native task targets without manufacturing a safe object-center goal."""
    result = {
        "status": "goal_unavailable",
        "verified": False,
        "position_world_m": None,
        "position_semantics": None,
        "safe_center_region_verified": False,
        "reason": "Native placement predicates do not specify a verified collision-clear carried-sphere-center region",
        "source": "active task parsed BDDL goal_state and compiled target Body/sites",
        "bddl_file": str(getattr(task_env, "bddl_file_name", "")),
        "targets": [],
    }
    problem = getattr(task_env, "parsed_problem", {})
    goals = problem.get("goal_state") if isinstance(problem, dict) else None
    result["native_goal_predicates"] = copy.deepcopy(goals)
    if not isinstance(goals, (list, tuple)):
        result["reason"] = "Active task goal predicates are unavailable"
        return result
    subject = carried["registered_task_object"]
    by_key = {entry["key"]: (body, entry) for body, entry in registered.items()}
    for predicate in goals:
        if (
            not isinstance(predicate, (list, tuple))
            or len(predicate) != 3
            or predicate[1] != subject
        ):
            continue
        if str(predicate[0]).lower() not in ("on", "in"):
            continue
        target_key = predicate[2]
        target = {"predicate": list(predicate), "verified_center_goal": False}
        site_model = (
            getattr(task_env, "object_sites_dict", {}).get(target_key)
            if isinstance(target_key, str)
            else None
        )
        if site_model is not None and target_key not in by_key:
            target["native_target_site_key"] = target_key
            target_key = getattr(site_model, "parent_name", None)
        if not isinstance(target_key, str) or target_key not in by_key:
            target["reason"] = (
                "Task target is not a directly registered physical object Body"
            )
            result["targets"].append(target)
            continue
        body, entry = by_key[target_key]
        subtree = _planning_body_subtree(model_wrapper._model, body)
        target.update({
            "registered_task_object": target_key,
            "body_id": body,
            "body_name": entry["body_name"],
            "body_origin_world_m_diagnostic_only": np.asarray(
                scratch.xpos[body]
            ).tolist(),
            "body_origin_is_carried_center_goal": False,
            "sites_diagnostic_only": [],
        })
        model = model_wrapper._model
        for site_id in range(int(model.nsite)):
            if int(model.site_bodyid[site_id]) in subtree:
                target["sites_diagnostic_only"].append({
                    "site_id": site_id,
                    "site_name": model_wrapper.site_id2name(site_id),
                    "type_id": int(model.site_type[site_id]),
                    "position_world_m": np.asarray(scratch.site_xpos[site_id]).tolist(),
                    "size": np.asarray(model.site_size[site_id]).tolist(),
                    "safe_center_semantics_verified": False,
                })
        result["targets"].append(target)
    if not result["targets"]:
        result["reason"] = (
            "No direct native placement goal for the physically bound carried object"
        )
    return result


def _planning_geometry(carried_binding, obstacle_bindings, expected_instruction):
    """Read complete planning bounds on scratch data, with exact nonmutation checks."""
    import hashlib

    sim, task_env = env.sim, env.env
    model = sim.model._model
    result = {
        "operation": "planning_geometry",
        "status": "unavailable",
        "query_complete": False,
        "physical_state_unchanged": False,
        "objects": {},
        "goal": {"status": "goal_unavailable"},
        "frame": "world",
        "units": "m",
        "physical_safety_verified": False,
    }
    before = signature = cache = None
    try:
        before = _mujoco_integration_state(sim)
        signature = _planning_model_signature(model)
        cache = _planning_cache_signature(sim.data._data)
        if before.ndim != 1 or not np.isfinite(before).all():
            raise ValueError("Current complete integration state is invalid")
        if (
            not isinstance(expected_instruction, str)
            or not expected_instruction
            or expected_instruction != task.language
        ):
            raise ValueError(
                "Expected task instruction does not match the active Bridge task"
            )
        if not isinstance(obstacle_bindings, list) or not obstacle_bindings:
            raise ValueError(
                "At least one verified forbidden-contact obstacle binding is required"
            )
        if int(getattr(model, "nflex", 0)) != 0:
            raise ValueError(
                "Planning enclosure does not support deformable flex geometry"
            )
        if int(getattr(model, "nplugin", 0)) != 0:
            raise ValueError(
                "Planning query cannot confirm external plugin state is read-only"
            )
        for callback in (
            "get_mjcb_control",
            "get_mjcb_passive",
            "get_mjcb_sensor",
            "get_mjcb_contactfilter",
            "get_mjcb_act_dyn",
            "get_mjcb_act_gain",
            "get_mjcb_act_bias",
        ):
            getter = getattr(mujoco, callback, None)
            if not callable(getter) or getter() is not None:
                raise ValueError(
                    f"Planning scratch forward cannot audit custom callback {callback}"
                )
        inventory = _geometry_inventory()
        result["geometry_inventory"] = inventory
        if (
            inventory.get("status") != "complete"
            or inventory.get("robot_classification_status") != "available"
        ):
            raise ValueError("Current Geometry Inventory is incomplete")
        state_id = hashlib.sha256(
            signature.encode() + before.tobytes() + task.language.encode()
        ).hexdigest()
        result.update({
            "state_id": state_id,
            "model_geometry_sha256": signature,
            "mujoco_integration_state": before.tolist(),
            "instruction": task.language,
            "mujoco_version": mujoco.__version__,
        })
        scratch = _diagnostic_data_from_integration_state(sim)
        registered = _planning_registered_objects(sim.model, task_env)
        all_bindings = [carried_binding, *obstacle_bindings]
        for binding in all_bindings:
            sphere = _planning_object_sphere(
                binding, inventory, sim.model, scratch, registered, state_id
            )
            if sphere["object_id"] in result["objects"]:
                raise ValueError("Planning object IDs are duplicated")
            for prior in result["objects"].values():
                if set(prior["body_ids"]) & set(sphere["body_ids"]):
                    raise ValueError(
                        "Planning bindings have overlapping physical object subtrees"
                    )
            result["objects"][sphere["object_id"]] = sphere
        carried = result["objects"][carried_binding["object_id"]]
        gravity = np.asarray(model.opt.gravity, dtype=np.float64)
        norm = float(np.linalg.norm(gravity))
        if gravity.shape != (3,) or not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("World-up cannot be confirmed from the active gravity")
        result.update({
            "world_up": (-gravity / norm).tolist(),
            "scratch_forward_used": True,
            "carried_object_id": carried_binding["object_id"],
            "obstacle_object_ids": [item["object_id"] for item in obstacle_bindings],
            "goal": _planning_goal_diagnostics(
                task_env, sim.model, scratch, registered, carried
            ),
            "status": "complete",
            "query_complete": True,
        })
    except Exception as exc:
        result["error"] = f"Planning geometry unavailable: {type(exc).__name__}: {exc}"
    finally:
        try:
            after = _mujoco_integration_state(sim)
            check = {
                "integration_state_equal": before is not None
                and np.array_equal(before, after),
                "model_geometry_equal": signature is not None
                and signature == _planning_model_signature(model),
                "live_kinematics_cache_equal": cache is not None
                and cache == _planning_cache_signature(sim.data._data),
            }
            result["state_consistency"] = check
            result["physical_state_unchanged"] = all(check.values())
        except Exception as exc:
            result["state_consistency_error"] = str(exc)
        if not result["physical_state_unchanged"]:
            result.update({
                "status": "unavailable",
                "query_complete": False,
                "error": "Planning query could not prove the complete state/model/caches unchanged",
            })
    return result


def _collision_scene_export():
    """Export active scene collision geometry from scratch data without mutation."""
    import hashlib

    sim = env.sim
    model = sim.model._model
    result = {
        "operation": "collision_scene_export",
        "status": "unavailable",
        "query_complete": False,
        "physical_state_unchanged": False,
        "units": "m",
        "frame": "world",
        "scene_collision_geometries": [],
        "robot_collision_geometries": [],
        "unsupported_geometries": [],
    }
    before = signature = cache = None
    try:
        before = _mujoco_integration_state(sim)
        signature = _planning_model_signature(model)
        cache = _planning_cache_signature(sim.data._data)
        if before.ndim != 1 or not np.isfinite(before).all():
            raise ValueError("Current complete integration state is invalid")
        inventory = _geometry_inventory()
        if inventory.get("status") != "complete" or inventory.get("robot_classification_status") != "available":
            raise ValueError("Current full Geometry Inventory is incomplete")
        scratch = _diagnostic_data_from_integration_state(sim)
        geom_inventory = {int(row["geom_id"]): row for row in inventory["geometries"]}
        pair_margin = {}
        explicit_pair_geoms = set()
        for index in range(int(model.npair)):
            geom_a, geom_b = int(model.pair_geom1[index]), int(model.pair_geom2[index])
            margin = float(model.pair_margin[index])
            if not (0 <= geom_a < int(model.ngeom) and 0 <= geom_b < int(model.ngeom)) or not np.isfinite(margin) or margin < 0:
                raise ValueError("Compiled explicit Contact Pair metadata is incomplete")
            explicit_pair_geoms.update((geom_a, geom_b))
            for geom_id in (geom_a, geom_b):
                pair_margin[geom_id] = max(pair_margin.get(geom_id, 0.0), margin)
        type_names = {
            int(value): name.removeprefix("mjGEOM_").lower()
            for name, value in getattr(mujoco.mjtGeom, "__members__", {}).items()
        }
        scene_rows, robot_rows, unsupported = [], [], []
        merged_meshes = {}
        for geom_id in range(int(model.ngeom)):
            item = geom_inventory.get(geom_id)
            if item is None or item.get("errors"):
                raise ValueError(f"Geom identity inventory is incomplete for {geom_id}")
            if not item.get("collision_enabled"):
                continue
            category = item.get("category")
            if category not in ("scene", "robot"):
                raise ValueError(f"Geom {geom_id} lacks a verified robot/scene identity")
            geom_type = int(model.geom_type[geom_id])
            type_name = type_names.get(geom_type, f"geom_{geom_type}")
            base = {
                "geom_id": geom_id,
                "geom_name": item.get("geom_name"),
                "body_id": item.get("body_id"),
                "body_name": item.get("body_name"),
                "category": category,
                "geom_type_id": geom_type,
                "geom_type": type_name,
                "contype": int(model.geom_contype[geom_id]),
                "conaffinity": int(model.geom_conaffinity[geom_id]),
                "explicit_contact_pair": geom_id in explicit_pair_geoms,
                "geom_margin_m": float(model.geom_margin[geom_id]),
                "pair_margin_m": float(pair_margin.get(geom_id, 0.0)),
                "geom_size_m": np.asarray(model.geom_size[geom_id], dtype=np.float64).tolist(),
                "geom_rbound_m": float(model.geom_rbound[geom_id]),
                "position_world_m": np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64).tolist(),
                "rotation_world": np.asarray(scratch.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3).tolist(),
            }
            if category == "robot":
                robot_rows.append(base)
                continue
            margin = max(float(model.geom_margin[geom_id]), pair_margin.get(geom_id, 0.0))
            local_center = np.zeros(3, dtype=np.float64)
            if type_name == "plane":
                # CuRobo's finite obstacle API has no infinite plane primitive.
                # The active tabletop scene lies inside this broad bounded floor patch.
                half = np.asarray([2.5, 2.5, 0.01], dtype=np.float64)
                local_center = np.asarray([0.0, 0.0, -0.01], dtype=np.float64)
                shape = "cuboid"
                base["source"] = "bounded_5m_workspace_enclosure_of_active_MuJoCo_infinite_plane"
            elif type_name == "mesh":
                mesh_id = int(model.geom_dataid[geom_id])
                if not 0 <= mesh_id < int(model.nmesh):
                    raise ValueError(f"Geom {geom_id} has invalid compiled mesh association")
                address, count = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
                vertices = np.asarray(model.mesh_vert[address : address + count], dtype=np.float64)
                if count <= 0 or vertices.shape != (count, 3) or not np.isfinite(vertices).all():
                    raise ValueError(f"Geom {geom_id} compiled mesh vertices are incomplete")
                face_address, face_count = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
                faces = np.asarray(model.mesh_face[face_address : face_address + face_count], dtype=np.int64)
                if face_count <= 0 or faces.shape != (face_count, 3) or np.any(faces < 0) or np.any(faces >= count):
                    raise ValueError(f"Geom {geom_id} compiled mesh faces are incomplete")
                rotation = np.asarray(scratch.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                position = np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64)
                vertices_world = vertices @ rotation.T + position[None, :]
                body_id = int(item["body_id"])
                merged = merged_meshes.setdefault(body_id, {
                    "body_id": body_id,
                    "body_name": item.get("body_name"),
                    "geom_ids": [],
                    "geom_names": [],
                    "vertices_world_m": [],
                    "faces": [],
                    "source": "exact_triangles_from_compiled_MuJoCo_collision_meshes",
                })
                vertex_offset = len(merged["vertices_world_m"])
                merged["geom_ids"].append(geom_id)
                merged["geom_names"].append(item.get("geom_name"))
                merged["vertices_world_m"].extend(vertices_world.tolist())
                merged["faces"].extend((faces + vertex_offset).reshape(-1).tolist())
                continue
            elif type_name == "box":
                half = np.asarray(model.geom_size[geom_id], dtype=np.float64)
                shape = "cuboid"
                base["source"] = "compiled_MuJoCo_box"
            elif type_name == "sphere":
                radius = float(model.geom_size[geom_id, 0])
                shape = "sphere"
                base["source"] = "compiled_MuJoCo_sphere"
            else:
                sphere = _planning_component_sphere(model, scratch, geom_id, pair_margin)
                scene_rows.append({
                    **base,
                    "shape": "sphere",
                    "center_world_m": sphere["center_world_m"],
                    "radius_m": sphere["radius_m"],
                    "source": sphere["source"],
                })
                continue
            rotation = np.asarray(scratch.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            center = np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64) + rotation @ local_center
            if shape == "cuboid":
                if half.shape != (3,) or not np.isfinite(half).all() or np.any(half <= 0):
                    raise ValueError(f"Geom {geom_id} has invalid finite box dimensions")
                scene_rows.append({
                    **base,
                    "shape": shape,
                    "center_world_m": center.tolist(),
                    "rotation_world": rotation.tolist(),
                    "half_extents_world_axes_m": (half + margin).tolist(),
                    "source": base.get("source", "compiled_MuJoCo_box"),
                })
            else:
                if not np.isfinite(radius) or radius <= 0 or not np.isfinite(margin):
                    raise ValueError(f"Geom {geom_id} has invalid sphere dimensions")
                scene_rows.append({
                    **base,
                    "shape": shape,
                    "center_world_m": center.tolist(),
                    "radius_m": float(radius + margin),
                    "source": base.get("source", "compiled_MuJoCo_sphere"),
                })
        for body_id, merged in sorted(merged_meshes.items()):
            if not merged["vertices_world_m"] or not merged["faces"]:
                raise ValueError(f"Body {body_id} merged compiled collision mesh is empty")
            scene_rows.append({
                **merged,
                "name": f"mujoco_body_{body_id}_compiled_mesh",
                "shape": "mesh",
            })
        result.update({
            "model_geometry_sha256": signature,
            "state_id": hashlib.sha256(signature.encode() + before.tobytes()).hexdigest(),
            "mujoco_integration_state": before.tolist(),
            "geometry_inventory": inventory,
            "scene_collision_geometries": scene_rows,
            "robot_collision_geometries": robot_rows,
            "unsupported_geometries": unsupported,
            "scratch_forward_used": True,
            "status": "complete" if not unsupported else "partial",
            "query_complete": not unsupported,
        })
    except Exception as exc:
        result["error"] = f"Collision scene export failed: {type(exc).__name__}: {exc}"
    finally:
        try:
            after = _mujoco_integration_state(sim)
            checks = {
                "integration_state_equal": before is not None and np.array_equal(before, after),
                "model_geometry_equal": signature is not None and signature == _planning_model_signature(model),
                "live_kinematics_cache_equal": cache is not None and cache == _planning_cache_signature(sim.data._data),
            }
            result["state_consistency"] = checks
            result["physical_state_unchanged"] = all(checks.values())
        except Exception as exc:
            result["state_consistency_error"] = f"{type(exc).__name__}: {exc}"
        if not result["physical_state_unchanged"]:
            result.update({
                "status": "unavailable",
                "query_complete": False,
                "error": "Collision scene export could not prove state/model/caches unchanged",
            })
    return result


def _scratch_robot_fk_audit(raw_joint_positions):
    """Compare requested arm FK poses on scratch MuJoCo data only."""
    sim = env.sim
    model = sim.model._model
    robot = env.robots[0]
    qpos_indices = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    q_values = np.asarray(raw_joint_positions, dtype=np.float64)
    result = {
        "operation": "scratch_robot_fk_audit",
        "status": "unavailable",
        "physical_state_unchanged": False,
        "model_geometry_sha256": _planning_model_signature(model),
        "joint_qpos_indices": qpos_indices.tolist(),
        "joint_names": [],
        "poses": [],
    }
    if (
        q_values.ndim != 2
        or q_values.shape[1:] != (len(qpos_indices),)
        or not len(q_values)
        or not np.isfinite(q_values).all()
    ):
        result["error"] = "joint_positions_must_be_finite_N_by_active_arm_dof"
        return result

    joint_names = []
    for qpos_index in qpos_indices:
        matches = [
            joint_id for joint_id in range(int(model.njnt))
            if int(model.jnt_qposadr[joint_id]) == int(qpos_index)
        ]
        if len(matches) != 1:
            result["error"] = "active_arm_qpos_index_does_not_map_to_one_joint"
            return result
        joint_id = matches[0]
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_names.append(str(name))
        if bool(model.jnt_limited[joint_id]):
            lower, upper = (float(value) for value in model.jnt_range[joint_id])
            if np.any(q_values[:, len(joint_names) - 1] < lower) or np.any(
                q_values[:, len(joint_names) - 1] > upper
            ):
                result["error"] = f"requested_fk_pose_exceeds_physical_limit:{name}"
                result["joint_names"] = joint_names
                return result

    live_state_before = _mujoco_integration_state(sim)
    model_signature_before = _planning_model_signature(model)
    cache_before = _planning_cache_signature(sim.data._data)
    scratch = _diagnostic_data_from_integration_state(sim)
    body_names = [
        getattr(robot.robot_model, "root_body", "robot0_base"),
        *[f"robot0_link{index}" for index in range(8)],
        "robot0_right_hand",
        "gripper0_right_gripper",
        "gripper0_eef",
    ]
    body_ids = {}
    for name in body_names:
        try:
            body_ids[name] = int(sim.model.body_name2id(name))
        except Exception:
            continue
    poses = []
    for pose_index, q in enumerate(q_values):
        scratch.qpos[qpos_indices] = q
        mujoco.mj_forward(model, scratch)
        bodies = {}
        for name, body_id in body_ids.items():
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, np.asarray(scratch.xmat[body_id]).reshape(9))
            bodies[name] = {
                "body_id": body_id,
                "position_world_m": np.asarray(scratch.xpos[body_id], dtype=np.float64).tolist(),
                "quaternion_world_wxyz": (quaternion / np.linalg.norm(quaternion)).tolist(),
            }
        eef_id = int(robot.eef_site_id)
        eef_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            eef_quaternion, np.asarray(scratch.site_xmat[eef_id]).reshape(9)
        )
        eef_quaternion /= np.linalg.norm(eef_quaternion)
        poses.append({
            "pose_index": pose_index,
            "arm_qpos_rad": q.tolist(),
            "bodies": bodies,
            "eef_site_id": eef_id,
            "eef_position_world_m": np.asarray(
                scratch.site_xpos[eef_id], dtype=np.float64
            ).tolist(),
            "eef_quaternion_world_wxyz": eef_quaternion.tolist(),
        })

    live_state_after = _mujoco_integration_state(sim)
    cache_after = _planning_cache_signature(sim.data._data)
    unchanged = bool(
        np.array_equal(live_state_before, live_state_after)
        and model_signature_before == _planning_model_signature(model)
        and cache_before == cache_after
    )
    result.update({
        "status": "complete" if unchanged else "incomplete",
        "physical_state_unchanged": unchanged,
        "model_geometry_sha256": model_signature_before,
        "joint_names": joint_names,
        "root_body_name": getattr(robot.robot_model, "root_body", None),
        "poses": poses,
        "scratch_forward_used": True,
        "active_state_equal": bool(np.array_equal(live_state_before, live_state_after)),
        "active_model_equal": model_signature_before == _planning_model_signature(model),
        "active_kinematics_cache_equal": cache_before == cache_after,
    })
    return result


def _scratch_robot_path_contact_audit(
    raw_arm_joint_states,
    object_id,
    raw_gripper_joint_states=None,
    raw_eef_to_object_transform=None,
    raw_world_object_transform=None,
    fingertip_contact_scope="none",
):
    """Audit exact MuJoCo contacts along hypothetical arm states on scratch data.

    The only optional contact exception is the active task's verified left/right
    fingertip pads against the requested target object's compiled collision
    Geoms, and only at the final supplied state. An optional object transform
    is expressed as T_eef_object in meters and is used to carry the target in
    scratch data; the live integration state is never modified or stepped.
    """
    sim = env.sim
    model = sim.model._model
    q_values = np.asarray(raw_arm_joint_states, dtype=np.float64)
    gripper_values = None if raw_gripper_joint_states is None else np.asarray(
        raw_gripper_joint_states, dtype=np.float64
    )
    transform_eef_object = None if raw_eef_to_object_transform is None else np.asarray(
        raw_eef_to_object_transform, dtype=np.float64
    )
    result = {
        "operation": "scratch_robot_path_contact_audit",
        "status": "unavailable",
        "physical_state_unchanged": False,
        "object_id": object_id,
        "states": [],
    }
    if (
        not isinstance(object_id, str)
        or q_values.ndim != 2
        or q_values.shape[1:] != (7,)
        or not len(q_values)
        or not np.isfinite(q_values).all()
    ):
        result["error"] = "object_id_and_finite_N_by_7_arm_joint_states_required"
        return result
    if gripper_values is not None and (
        gripper_values.shape != (len(q_values), 2)
        or not np.isfinite(gripper_values).all()
    ):
        result["error"] = "gripper_joint_states_must_be_finite_N_by_2"
        return result
    if transform_eef_object is not None and (
        transform_eef_object.shape != (4, 4)
        or not np.isfinite(transform_eef_object).all()
        or not np.allclose(transform_eef_object[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(
            transform_eef_object[:3, :3].T @ transform_eef_object[:3, :3],
            np.eye(3), atol=1e-6,
        )
        or not np.isclose(np.linalg.det(transform_eef_object[:3, :3]), 1.0, atol=1e-6)
    ):
        result["error"] = "T_eef_object_must_be_a_finite_SE3_transform_in_meters"
        return result
    world_object_transform = None if raw_world_object_transform is None else np.asarray(
        raw_world_object_transform, dtype=np.float64
    )
    if world_object_transform is not None and (
        world_object_transform.shape != (4, 4)
        or not np.isfinite(world_object_transform).all()
        or not np.allclose(world_object_transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8)
        or not np.allclose(
            world_object_transform[:3, :3].T @ world_object_transform[:3, :3],
            np.eye(3), atol=1e-6,
        )
        or not np.isclose(np.linalg.det(world_object_transform[:3, :3]), 1.0, atol=1e-6)
    ):
        result["error"] = "T_world_object_must_be_a_finite_SE3_transform_in_meters"
        return result
    if transform_eef_object is not None and world_object_transform is not None:
        result["error"] = "carried_and_fixed_world_object_transforms_are_mutually_exclusive"
        return result
    if fingertip_contact_scope not in ("none", "final_state", "all_states"):
        result["error"] = "fingertip_contact_scope_must_be_none_final_state_or_all_states"
        return result

    bindings_result = _task_semantic_geometry_bindings()
    binding = bindings_result.get("bindings", {}).get(object_id)
    if bindings_result.get("status") != "complete" or not isinstance(binding, dict):
        result["error"] = "active_BDDL_target_geometry_binding_unavailable"
        return result
    problem = getattr(env.env, "parsed_problem", {})
    goals = problem.get("goal_state") if isinstance(problem, dict) else None
    target_predicate = next((
        list(predicate)
        for predicate in goals or []
        if isinstance(predicate, (list, tuple))
        and len(predicate) == 3
        and predicate[1] == object_id
    ), None)
    if target_predicate is None:
        result["error"] = "active_BDDL_target_predicate_is_unavailable"
        return result
    context = _task_success_first_grasp_context({
        "expected_instruction": task.language,
        "target_predicate": target_predicate,
    })
    if context.get("status") != "complete" or context.get("read_only") is not True:
        result["error"] = "verified_fingertip_geometry_context_unavailable"
        return result
    target_geom_ids = {int(value) for value in binding.get("root_geom_ids", [])}
    pad_ids = {
        int(row["geom_id"])
        for side in ("left", "right")
        for row in context.get("fingertip_pad_geometries", {}).get(side, [])
    }
    if not target_geom_ids or not pad_ids or len(pad_ids) != 2:
        result["error"] = "verified_two_sided_fingertip_or_target_geom_ids_unavailable"
        return result
    allowed_pairs = {
        tuple(sorted((pad_id, target_geom_id)))
        for pad_id in pad_ids
        for target_geom_id in target_geom_ids
    }

    robot = env.robots[0]
    arm_qpos = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    gripper_qpos = np.asarray(robot._ref_gripper_joint_pos_indexes, dtype=np.int64)
    if len(arm_qpos) != 7 or len(gripper_qpos) != 2:
        result["error"] = "active_robot_arm_or_gripper_qpos_layout_is_not_7_plus_2"
        return result
    live_before = _mujoco_integration_state(sim)
    model_before = _planning_model_signature(model)
    cache_before = _planning_cache_signature(sim.data._data)
    scratch = _diagnostic_data_from_integration_state(sim)
    robot_body_ids = {
        int(row["body_id"])
        for row in _geometry_inventory().get("geometries", [])
        if row.get("category") == "robot" and row.get("collision_enabled") is True
    }
    target_body_id = int(binding["root_body_id"])
    free_joint_id = int(model.body_jntadr[target_body_id])
    if transform_eef_object is not None and (
        int(model.body_jntnum[target_body_id]) != 1
        or int(model.jnt_type[free_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
    ):
        result["error"] = "carried_target_root_is_not_a_single_free_joint"
        return result
    object_qpos = int(model.jnt_qposadr[free_joint_id])
    eef_site_id = int(robot.eef_site_id)
    states = []
    for state_index, arm_q in enumerate(q_values):
        scratch.qpos[arm_qpos] = arm_q
        if gripper_values is not None:
            scratch.qpos[gripper_qpos] = gripper_values[state_index]
        mujoco.mj_forward(model, scratch)
        if transform_eef_object is not None:
            eef_pose = np.eye(4, dtype=np.float64)
            eef_pose[:3, :3] = np.asarray(scratch.site_xmat[eef_site_id]).reshape(3, 3)
            eef_pose[:3, 3] = np.asarray(scratch.site_xpos[eef_site_id], dtype=np.float64)
            object_pose = eef_pose @ transform_eef_object
            object_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(object_quat, object_pose[:3, :3].reshape(9))
            scratch.qpos[object_qpos:object_qpos + 7] = np.r_[object_pose[:3, 3], object_quat]
            mujoco.mj_forward(model, scratch)
        elif world_object_transform is not None:
            object_quat = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(object_quat, world_object_transform[:3, :3].reshape(9))
            scratch.qpos[object_qpos:object_qpos + 7] = np.r_[
                world_object_transform[:3, 3], object_quat
            ]
            mujoco.mj_forward(model, scratch)
        contacts = []
        illegal_contacts = []
        for contact_index in range(int(scratch.ncon)):
            contact = scratch.contact[contact_index]
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            body_a, body_b = int(model.geom_bodyid[geom_a]), int(model.geom_bodyid[geom_b])
            if body_a not in robot_body_ids and body_b not in robot_body_ids:
                continue
            pair = tuple(sorted((geom_a, geom_b)))
            target_fingertip_pair = pair in allowed_pairs
            allowed = bool(
                target_fingertip_pair
                and fingertip_contact_scope != "none"
                and (
                    fingertip_contact_scope == "all_states"
                    or state_index == len(q_values) - 1
                )
            )
            row = {
                "geom_ids": [geom_a, geom_b],
                "geom_names": [
                    sim.model.geom_id2name(geom_a), sim.model.geom_id2name(geom_b)
                ],
                "body_ids": [body_a, body_b],
                "distance_m": float(contact.dist),
                "is_verified_fingertip_target_pair": target_fingertip_pair,
                "allowed": allowed,
            }
            contacts.append(row)
            if not allowed:
                illegal_contacts.append(row)
        states.append({
            "state_index": state_index,
            "arm_joint_positions_rad": arm_q.tolist(),
            "gripper_joint_positions_rad": (
                gripper_values[state_index].tolist() if gripper_values is not None else None
            ),
            "contact_count": len(contacts),
            "illegal_contact_count": len(illegal_contacts),
            "contacts": contacts,
            "illegal_contacts": illegal_contacts,
        })
    live_after = _mujoco_integration_state(sim)
    cache_after = _planning_cache_signature(sim.data._data)
    unchanged = bool(
        np.array_equal(live_before, live_after)
        and model_before == _planning_model_signature(model)
        and cache_before == cache_after
    )
    illegal_count = sum(row["illegal_contact_count"] for row in states)
    result.update({
        "status": "complete" if unchanged else "incomplete",
        "physical_state_unchanged": unchanged,
        "active_model_and_cache_unchanged": bool(
            model_before == _planning_model_signature(model)
            and cache_before == cache_after
        ),
        "target_geom_ids": sorted(target_geom_ids),
        "verified_fingertip_pad_geom_ids": sorted(pad_ids),
        "allowed_contact_policy": (
            f"only verified left/right fingertip-pad to target-object geom pairs ({fingertip_contact_scope})"
            if fingertip_contact_scope != "none" else "no robot-scene contact exception"
        ),
        "state_count": len(states),
        "illegal_contact_count": illegal_count,
        "states": states,
    })
    return result


def _scratch_carried_geom_distance_audit(raw_joint_states, object_id):
    """Measure exact compiled Geom distances in hypothetical grasp-preserving scratch states."""
    sim = env.sim
    model = sim.model._model
    model_wrapper = sim.model
    q_values = np.asarray(raw_joint_states, dtype=np.float64)
    result = {
        "operation": "scratch_carried_geom_distance_audit",
        "status": "unavailable",
        "physical_state_unchanged": False,
        "model_geometry_sha256": _planning_model_signature(model),
        "object_id": object_id,
        "states": [],
    }
    if (
        not isinstance(object_id, str)
        or q_values.ndim != 2
        or q_values.shape[1:] != (7,)
        or not len(q_values)
        or not np.isfinite(q_values).all()
    ):
        result["error"] = "object_id_and_finite_N_by_7_joint_states_required"
        return result
    bindings_result = _task_semantic_geometry_bindings()
    binding = bindings_result.get("bindings", {}).get(object_id)
    if bindings_result.get("status") != "complete" or not isinstance(binding, dict):
        result["error"] = "active_BDDL_object_geometry_binding_unavailable"
        return result

    body_id = int(binding["root_body_id"])
    carried_geom_ids = [int(value) for value in binding["root_geom_ids"]]
    robot = env.robots[0]
    arm_qpos = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    if len(arm_qpos) != 7:
        result["error"] = "active_robot_arm_qpos_is_not_7d"
        return result
    joint_names = []
    for qpos_index in arm_qpos:
        joint_id = next(
            (jid for jid in range(int(model.njnt)) if int(model.jnt_qposadr[jid]) == int(qpos_index)),
            None,
        )
        if joint_id is None:
            result["error"] = "active_arm_qpos_index_has_no_model_joint"
            return result
        joint_names.append(str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)))
    joint_name_to_id = {
        str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)): jid
        for jid in range(int(model.njnt))
    }
    for q_index, q_value in enumerate(q_values):
        for q_col, joint_name in enumerate(joint_names):
            joint_id = joint_name_to_id[joint_name]
            if bool(model.jnt_limited[joint_id]):
                lower, upper = (float(v) for v in model.jnt_range[joint_id])
                if q_value[q_col] < lower or q_value[q_col] > upper:
                    result["error"] = f"scratch_joint_state_exceeds_physical_limit:{joint_name}"
                    return result

    live_state_before = _mujoco_integration_state(sim)
    model_signature_before = _planning_model_signature(model)
    cache_before = _planning_cache_signature(sim.data._data)
    scratch = _diagnostic_data_from_integration_state(sim)
    free_joint_id = int(model.body_jntadr[body_id])
    if (
        int(model.body_jntnum[body_id]) != 1
        or int(model.jnt_type[free_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
    ):
        result["error"] = "carried_object_root_is_not_one_free_joint"
        return result
    object_qpos = int(model.jnt_qposadr[free_joint_id])
    object_pose_start = np.eye(4, dtype=np.float64)
    object_pose_start[:3, :3] = np.asarray(scratch.xmat[body_id]).reshape(3, 3)
    object_pose_start[:3, 3] = np.asarray(scratch.xpos[body_id], dtype=np.float64)
    eef_site_id = int(robot.eef_site_id)
    eef_pose_start = np.eye(4, dtype=np.float64)
    eef_pose_start[:3, :3] = np.asarray(scratch.site_xmat[eef_site_id]).reshape(3, 3)
    eef_pose_start[:3, 3] = np.asarray(scratch.site_xpos[eef_site_id], dtype=np.float64)
    eef_to_object = np.linalg.inv(eef_pose_start) @ object_pose_start

    inventory = _geometry_inventory()
    if inventory.get("status") != "complete":
        result["error"] = "complete_geometry_inventory_unavailable"
        return result
    robot_geom_ids = sorted(
        int(row["geom_id"])
        for row in inventory["geometries"]
        if row.get("category") == "robot" and row.get("collision_enabled") is True
    )
    scene_geom_ids = sorted(
        int(row["geom_id"])
        for row in inventory["geometries"]
        if row.get("category") == "scene"
        and row.get("collision_enabled") is True
        and int(row["geom_id"]) not in set(carried_geom_ids)
    )
    query_geom_ids = sorted(set(scene_geom_ids + robot_geom_ids))
    distance_fn = getattr(mujoco, "mj_geomDistance", None)
    if not callable(distance_fn):
        result["error"] = "MuJoCo_build_has_no_mj_geomDistance"
        return result
    geom_names = {
        geom_id: model_wrapper.geom_id2name(geom_id)
        for geom_id in set(carried_geom_ids + query_geom_ids)
    }
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    requested_pairs = 0
    attempted_pairs = 0
    censored_pairs = 0
    pair_errors = []
    states = []
    for state_index, arm_q in enumerate(q_values):
        scratch.qpos[arm_qpos] = arm_q
        mujoco.mj_forward(model, scratch)
        eef_pose = np.eye(4, dtype=np.float64)
        eef_pose[:3, :3] = np.asarray(scratch.site_xmat[eef_site_id]).reshape(3, 3)
        eef_pose[:3, 3] = np.asarray(scratch.site_xpos[eef_site_id], dtype=np.float64)
        object_pose = eef_pose @ eef_to_object
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, object_pose[:3, :3].reshape(9))
        scratch.qpos[object_qpos:object_qpos + 7] = np.r_[object_pose[:3, 3], quat]
        mujoco.mj_forward(model, scratch)
        per_body = {}
        nearest = None
        near_pairs = []
        for geom_a in carried_geom_ids:
            for geom_b in query_geom_ids:
                requested_pairs += 1
                attempted_pairs += 1
                fromto = np.zeros(6, dtype=np.float64)
                try:
                    distance = float(distance_fn(model, scratch, geom_a, geom_b, 5.0, fromto))
                    if not np.isfinite(distance):
                        raise ValueError("non_finite_geom_distance")
                    if distance >= 5.0:
                        censored_pairs += 1
                        continue
                    body_b = int(model.geom_bodyid[geom_b])
                    body_name_b = model_wrapper.body_id2name(body_b)
                    pair = {
                        "carried_geom_id": int(geom_a),
                        "carried_geom_name": geom_names.get(geom_a),
                        "other_geom_id": int(geom_b),
                        "other_geom_name": geom_names.get(geom_b),
                        "other_body_id": body_b,
                        "other_body_name": body_name_b,
                        "distance_m": distance,
                        "mesh_approximation": (
                            int(model.geom_type[geom_a]) == mesh_type
                            or int(model.geom_type[geom_b]) == mesh_type
                        ),
                    }
                    current = per_body.get(str(body_b))
                    if current is None or distance < current["minimum_distance_m"]:
                        per_body[str(body_b)] = {
                            "other_body_id": body_b,
                            "other_body_name": body_name_b,
                            "minimum_distance_m": distance,
                            "closest_geom_pair": pair,
                        }
                    if nearest is None or distance < nearest["distance_m"]:
                        nearest = pair
                    if distance <= 0.02:
                        near_pairs.append(pair)
                except Exception as exc:
                    if len(pair_errors) < 25:
                        pair_errors.append({
                            "carried_geom_id": int(geom_a),
                            "other_geom_id": int(geom_b),
                            "error": f"{type(exc).__name__}: {exc}",
                        })
        states.append({
            "state_index": state_index,
            "arm_qpos_rad": arm_q.tolist(),
            "object_pose_world": {
                "position_m": object_pose[:3, 3].tolist(),
                "quaternion_wxyz": quat.tolist(),
            },
            "nearest_pair": nearest,
            "minimum_distance_by_other_body": sorted(
                per_body.values(), key=lambda row: row["minimum_distance_m"]
            ),
            "near_pairs_le_20mm": sorted(near_pairs, key=lambda row: row["distance_m"]),
        })
    live_state_after = _mujoco_integration_state(sim)
    cache_after = _planning_cache_signature(sim.data._data)
    state_unchanged = bool(
        np.array_equal(live_state_before, live_state_after)
        and model_signature_before == _planning_model_signature(model)
        and cache_before == cache_after
    )
    result.update({
        "status": "complete" if state_unchanged and attempted_pairs == requested_pairs and not pair_errors else "incomplete",
        "physical_state_unchanged": state_unchanged,
        "active_integration_state_equal": bool(np.array_equal(live_state_before, live_state_after)),
        "active_model_and_cache_equal": bool(
            model_signature_before == _planning_model_signature(model)
            and cache_before == cache_after
        ),
        "timestep": int(env.env.timestep),
        "object_root_body_id": body_id,
        "object_geom_ids": carried_geom_ids,
        "robot_comparison_geom_count": len(robot_geom_ids),
        "scene_comparison_geom_count": len(scene_geom_ids),
        "requested_pair_count": requested_pairs,
        "attempted_pair_count": attempted_pairs,
        "censored_at_5m_pair_count": censored_pairs,
        "pair_error_count": len(pair_errors),
        "pair_errors": pair_errors,
        "source_object_to_eef_transform_world_relative": eef_to_object.tolist(),
        "distance_sign_convention": "negative=overlap, zero=touching, positive=separation",
        "distance_method": "mj_geomDistance on independent MjData; carried object rigidly follows the actual t=215 EEF-to-object transform; no mj_step or live qpos write",
        "mesh_approximation_possible": True,
        "states": states,
    })
    return result


def _validated_geom_id_list(raw_ids, label, geom_count):
    if not isinstance(raw_ids, list) or not raw_ids:
        return None, f"{label} must be a non-empty array of Geom IDs"
    geom_ids = []
    for index, value in enumerate(raw_ids):
        if not isinstance(value, int) or isinstance(value, bool):
            return None, f"{label}[{index}] must be an integer Geom ID"
        if value < 0 or value >= geom_count:
            return None, f"{label}[{index}]={value} is outside [0, {geom_count})"
        geom_ids.append(value)
    if len(set(geom_ids)) != len(geom_ids):
        return None, f"{label} contains duplicate Geom IDs"
    return geom_ids, None


def _geom_set_min_distance(geom_ids_a, geom_ids_b, distmax_m=10.0):
    """Query pairwise signed distances on forwarded scratch data only."""
    sim = env.sim
    model = sim.model._model
    model_wrapper = sim.model
    geom_count = int(model.ngeom)
    result = {
        "operation": "geom_set_min_distance",
        "status": "unavailable",
        "query_complete": False,
        "distance_reliable": False,
        "minimum_distance_m": None,
        "minimum_pair": None,
        "geom_ids_a": geom_ids_a if isinstance(geom_ids_a, list) else None,
        "geom_ids_b": geom_ids_b if isinstance(geom_ids_b, list) else None,
        "distmax_m": distmax_m,
        "pair_count_requested": None,
        "pair_count_attempted": 0,
        "pair_count_censored_at_distmax": 0,
        "pair_error_count": 0,
        "pair_errors": [],
        "mesh_approximation": False,
        "warnings": [],
        "scratch_forward_used": False,
        "active_integration_state_unchanged": None,
    }
    ids_a, error_a = _validated_geom_id_list(geom_ids_a, "geom_ids_a", geom_count)
    ids_b, error_b = _validated_geom_id_list(geom_ids_b, "geom_ids_b", geom_count)
    if error_a or error_b:
        result["errors"] = [error for error in (error_a, error_b) if error]
        return result
    if set(ids_a) & set(ids_b):
        result["errors"] = ["The two Geom sets must be disjoint"]
        return result
    try:
        distmax = float(distmax_m)
    except (TypeError, ValueError, OverflowError):
        result["errors"] = ["distmax_m must be a finite positive number"]
        return result
    if not np.isfinite(distmax) or distmax <= 0.0:
        result["errors"] = ["distmax_m must be a finite positive number"]
        return result
    distance_fn = getattr(mujoco, "mj_geomDistance", None)
    if not callable(distance_fn):
        result["errors"] = ["This MuJoCo build does not expose mj_geomDistance"]
        return result

    result["geom_ids_a"] = ids_a
    result["geom_ids_b"] = ids_b
    result["distmax_m"] = distmax
    result["pair_count_requested"] = len(ids_a) * len(ids_b)
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    try:
        live_state_before = _mujoco_integration_state(sim)
        scratch_data = _diagnostic_data_from_integration_state(sim)
        result["scratch_forward_used"] = True
    except Exception as exc:
        result["errors"] = [f"Cannot construct forwarded scratch data: {type(exc).__name__}: {exc}"]
        return result

    best_distance = None
    best_pair = None
    for geom_a in ids_a:
        for geom_b in ids_b:
            result["pair_count_attempted"] += 1
            geom_pair = {"geom_id_a": geom_a, "geom_id_b": geom_b}
            try:
                fromto = np.zeros(6, dtype=np.float64)
                distance = float(distance_fn(model, scratch_data, geom_a, geom_b, distmax, fromto))
                if not np.isfinite(distance):
                    raise ValueError(f"mj_geomDistance returned non-finite distance {distance!r}")
                if distance >= distmax:
                    result["pair_count_censored_at_distmax"] += 1
                    continue
                if not np.isfinite(fromto).all():
                    raise ValueError("mj_geomDistance returned non-finite closest-point coordinates")

                body_a = int(model.geom_bodyid[geom_a])
                body_b = int(model.geom_bodyid[geom_b])
                type_a = int(model.geom_type[geom_a])
                type_b = int(model.geom_type[geom_b])
                mesh_pair = type_a == mesh_type or type_b == mesh_type
                result["mesh_approximation"] = result["mesh_approximation"] or mesh_pair
                geom_pair.update(
                    {
                        "distance_m": distance,
                        "geom_a": {
                            "id": geom_a,
                            "name": model_wrapper.geom_id2name(geom_a),
                            "body_id": body_a,
                            "body_name": model_wrapper.body_id2name(body_a),
                            "type_id": type_a,
                        },
                        "geom_b": {
                            "id": geom_b,
                            "name": model_wrapper.geom_id2name(geom_b),
                            "body_id": body_b,
                            "body_name": model_wrapper.body_id2name(body_b),
                            "type_id": type_b,
                        },
                        "closest_segment_world": fromto.tolist(),
                        "mesh_approximation": mesh_pair,
                    }
                )
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = geom_pair
            except Exception as exc:
                result["pair_error_count"] += 1
                if len(result["pair_errors"]) < 10:
                    result["pair_errors"].append(
                        {
                            **geom_pair,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )

    try:
        live_state_after = _mujoco_integration_state(sim)
        state_unchanged = np.array_equal(live_state_before, live_state_after)
        result["active_integration_state_unchanged"] = bool(state_unchanged)
    except Exception as exc:
        state_unchanged = False
        result["active_state_check_error"] = f"{type(exc).__name__}: {exc}"

    result["minimum_distance_m"] = best_distance
    result["minimum_pair"] = best_pair
    complete = bool(
        state_unchanged
        and result["pair_count_attempted"] == result["pair_count_requested"]
        and result["pair_error_count"] == 0
        and result["pair_count_censored_at_distmax"] == 0
        and best_pair is not None
    )
    result["query_complete"] = complete
    result["distance_reliable"] = complete
    result["status"] = "complete" if complete else "partial"
    result["distance_sign_convention"] = "negative=overlap, zero=touching, positive=separation"
    result["distance_method"] = "mujoco.mj_geomDistance on scratch MjData from current integration state"
    if result["pair_count_censored_at_distmax"]:
        result["warnings"].append(
            "One or more Geom pairs reached distmax_m; the reported minimum may be censored"
        )
    if result["pair_error_count"]:
        result["warnings"].append(
            f"{result['pair_error_count']} Geom pair distance queries failed; at most 10 are listed"
        )
    if not state_unchanged:
        result["warnings"].append("The active MuJoCo integration state changed during the query")
    if result["mesh_approximation"]:
        result["warnings"].append(
            "Mesh distance can be approximate; this estimate is exploratory and is not a collision-safety proof"
        )
    result["not_a_safety_proof"] = True
    return result


def _gripper_geometry_inventory():
    """Resolve collision Geom IDs from robosuite's active gripper models."""
    sim = env.sim
    robot_body_ids = None
    robot_body_error = None
    try:
        robot_body_ids = _robot_body_ids()
    except Exception as exc:
        robot_body_error = f"{type(exc).__name__}: {exc}"

    grippers = []
    for robot_index, robot in enumerate(env.robots):
        gripper_value = getattr(robot, "gripper", None)
        if isinstance(gripper_value, dict):
            gripper_items = list(gripper_value.items())
        else:
            gripper_items = [(None, gripper_value)]

        for arm_name, gripper in gripper_items:
            entry = {
                "robot_index": robot_index,
                "arm": arm_name,
                "model_class": (
                    f"{type(gripper).__module__}.{type(gripper).__qualname__}"
                    if gripper is not None
                    else None
                ),
                "status": "unavailable",
                "geometries": [],
                "errors": [],
            }
            if gripper is None:
                entry["errors"].append("robot has no gripper model")
                grippers.append(entry)
                continue

            try:
                geom_names = gripper.contact_geoms
            except Exception as exc:
                entry["errors"].append(
                    f"Cannot read gripper contact_geoms: {type(exc).__name__}: {exc}"
                )
                grippers.append(entry)
                continue

            if not isinstance(geom_names, (list, tuple)) or not geom_names:
                entry["errors"].append("gripper model has no contact geom names")
                grippers.append(entry)
                continue

            seen_geom_ids = set()
            for geom_name in geom_names:
                if not isinstance(geom_name, str) or not geom_name:
                    entry["errors"].append(f"invalid gripper contact geom name: {geom_name!r}")
                    continue
                try:
                    geom_id = int(sim.model.geom_name2id(geom_name))
                    if geom_id < 0 or geom_id >= int(sim.model._model.ngeom):
                        raise ValueError(f"Geom ID {geom_id} is outside the active model")
                    actual_name = sim.model.geom_id2name(geom_id)
                    if actual_name != geom_name:
                        raise ValueError(
                            f"name lookup returned {actual_name!r} for {geom_name!r}"
                        )
                    body_id = int(sim.model.geom_bodyid[geom_id])
                    if robot_body_ids is None or body_id not in robot_body_ids:
                        raise ValueError(
                            f"gripper geom {geom_name!r} does not map to a confirmed robot body"
                        )
                    if geom_id in seen_geom_ids:
                        continue
                    seen_geom_ids.add(geom_id)
                    entry["geometries"].append(
                        {
                            "geom_id": geom_id,
                            "geom_name": actual_name,
                            "body_id": body_id,
                            "body_name": sim.model.body_id2name(body_id),
                        }
                    )
                except Exception as exc:
                    entry["errors"].append(
                        f"Cannot map gripper geom {geom_name!r}: {type(exc).__name__}: {exc}"
                    )
            if entry["geometries"] and not entry["errors"]:
                entry["status"] = "complete"
            elif not entry["errors"]:
                entry["errors"].append("no gripper contact Geom IDs were resolved")
            grippers.append(entry)

    complete = bool(grippers) and robot_body_ids is not None and all(
        item["status"] == "complete" for item in grippers
    )
    result = {
        "status": "complete" if complete else "partial" if grippers else "unavailable",
        "robot_classification_status": (
            "available" if robot_body_ids is not None else "unavailable"
        ),
        "robot_body_ids": sorted(robot_body_ids) if robot_body_ids is not None else None,
        "model_body_count": int(sim.model._model.nbody),
        "model_geom_count": int(sim.model._model.ngeom),
        "grippers": grippers,
    }
    if robot_body_error is not None:
        result["robot_classification_error"] = robot_body_error
    return result


def _task_success_first_grasp_context(request):
    """Resolve an active BDDL target and its collision/pad geometry read-only."""
    if request.get("expected_instruction") != task.language:
        raise ValueError("task_instruction_does_not_match_active_BDDL_task")
    model = env.sim.model._model
    data = env.sim.data._data
    bindings = _task_semantic_geometry_bindings()["bindings"]
    problem = getattr(env.env, "parsed_problem", {})
    goals = problem.get("goal_state") if isinstance(problem, dict) else None
    if not isinstance(goals, (list, tuple)):
        raise ValueError("active_BDDL_goal_state_is_unavailable")
    candidates = [
        list(predicate)
        for predicate in goals
        if isinstance(predicate, (list, tuple))
        and (
            len(predicate) == 3
            and str(predicate[0]).lower() in ("in", "on")
            and predicate[1] in bindings
            or len(predicate) == 2
            and str(predicate[0]).lower() in ("close", "open")
            and predicate[1] in bindings
        )
    ]
    requested = request.get("target_predicate")
    if requested is not None:
        requested = list(requested) if isinstance(requested, (list, tuple)) else None
        if requested is None or requested not in candidates:
            raise ValueError("requested_target_predicate_is_not_an_active_BDDL_object_goal")
        predicate = requested
    elif len(candidates) == 1:
        predicate = candidates[0]
    else:
        raise ValueError("active_BDDL_has_ambiguous_object_goals; pass_an_exact_active_target_predicate")

    target_id = str(predicate[1])
    binding = bindings[target_id]
    root_id = int(binding["root_body_id"])
    if root_id <= 0 or root_id >= int(model.nbody):
        raise ValueError("active_BDDL_target_root_body_is_invalid")
    root_position = np.asarray(data.xpos[root_id], dtype=np.float64)
    root_rotation = np.asarray(data.xmat[root_id], dtype=np.float64).reshape(3, 3)
    geom_rows = []
    for geom_id in sorted(int(value) for value in binding["geom_ids"]):
        geom_position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
        geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        geom_type = int(model.geom_type[geom_id])
        geom_type_name = next(
            (
                name.removeprefix("mjGEOM_").lower()
                for name, value in getattr(mujoco.mjtGeom, "__members__", {}).items()
                if int(value) == geom_type
            ),
            f"geom_{geom_type}",
        )
        size = np.asarray(model.geom_size[geom_id], dtype=np.float64).copy()
        center_offset_geom = np.zeros(3, dtype=np.float64)
        geometry_source = "compiled_geom_size"
        mesh_vertex_count = None
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[geom_id])
            vertex_start = int(model.mesh_vertadr[mesh_id])
            vertex_count = int(model.mesh_vertnum[mesh_id])
            vertices = np.asarray(model.mesh_vert[vertex_start : vertex_start + vertex_count], dtype=np.float64)
            if vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices):
                lower, upper = vertices.min(axis=0), vertices.max(axis=0)
                center_offset_geom = (lower + upper) * 0.5
                size = (upper - lower) * 0.5
                geometry_source = "compiled_MuJoCo_mesh_vertex_bounds"
                mesh_vertex_count = int(vertex_count)
            else:
                raise ValueError(f"active_mesh_collision_geometry_has_no_vertices:{geom_id}")
        center_root = root_rotation.T @ (geom_position - root_position)
        rotation_root = root_rotation.T @ geom_rotation
        center_root += rotation_root @ center_offset_geom
        geom_rows.append({
            "geom_id": geom_id,
            "geom_name": str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)),
            "body_id": int(model.geom_bodyid[geom_id]),
            "geom_type": int(model.geom_type[geom_id]),
            "geom_type_name": geom_type_name,
            "size_m": size.tolist(),
            "center_root_m": center_root.tolist(),
            "rotation_root": rotation_root.tolist(),
            "center_world_m": geom_position.tolist(),
            "rotation_world": geom_rotation.tolist(),
            "geometry_source": geometry_source,
            "mesh_vertex_count": mesh_vertex_count,
        })

    robot = env.robots[0]
    eef_site_id = int(robot.eef_site_id)
    eef_position = np.asarray(data.site_xpos[eef_site_id], dtype=np.float64)
    eef_rotation = np.asarray(data.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
    inventory = _gripper_geometry_inventory()
    if inventory.get("status") != "complete" or not inventory.get("grippers"):
        raise ValueError("active_gripper_geometry_inventory_is_incomplete")
    geometries = inventory["grippers"][0].get("geometries", [])
    pads = {"left": [], "right": []}
    for item in geometries:
        name = str(item["geom_name"]).lower()
        if "pad" not in name:
            continue
        if "finger1" in name or "leftfinger" in name or "left_finger" in name:
            side = "left"
        elif "finger2" in name or "rightfinger" in name or "right_finger" in name:
            side = "right"
        else:
            continue
        geom_id = int(item["geom_id"])
        pads[side].append({
            "geom_id": geom_id,
            "geom_name": str(item["geom_name"]),
            "center_eef_m": (
                eef_rotation.T @ (np.asarray(data.geom_xpos[geom_id], dtype=np.float64) - eef_position)
            ).tolist(),
        })
    if len(pads["left"]) != 1 or len(pads["right"]) != 1:
        raise ValueError("active_Panda_model_must_expose_one_pad_collision_geom_per_finger_side")
    left_center = np.asarray(pads["left"][0]["center_eef_m"], dtype=np.float64)
    right_center = np.asarray(pads["right"][0]["center_eef_m"], dtype=np.float64)
    arm_ids = np.asarray(robot._ref_joint_indexes, dtype=np.int64)
    arm_qpos = np.asarray(data.qpos[model.jnt_qposadr[arm_ids]], dtype=np.float64)
    gripper_qpos = np.asarray(data.qpos[robot._ref_gripper_joint_pos_indexes], dtype=np.float64)
    gripper_joint_ids = np.asarray(
        [env.sim.model.joint_name2id(name) for name in robot.gripper_joints], dtype=np.int64
    )
    gripper_joint_ranges = np.asarray(model.jnt_range[gripper_joint_ids], dtype=np.float64)
    if gripper_joint_ranges.shape != (len(gripper_qpos), 2):
        raise ValueError("active_gripper_joint_ranges_do_not_match_gripper_qpos")
    integration = np.zeros(
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION), dtype=np.float64
    )
    mujoco.mj_getState(model, data, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
    scratch = mujoco.MjData(model)
    max_pad_separation = 0.0
    for corner in (
        [gripper_joint_ranges[0, side0], gripper_joint_ranges[1, side1]]
        for side0 in (0, 1)
        for side1 in (0, 1)
    ):
        mujoco.mj_setState(model, scratch, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
        scratch.qpos[robot._ref_gripper_joint_pos_indexes] = np.asarray(corner, dtype=np.float64)
        mujoco.mj_forward(model, scratch)
        positions = []
        for side in ("left", "right"):
            geom_id = int(pads[side][0]["geom_id"])
            positions.append(np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64))
        max_pad_separation = max(max_pad_separation, float(np.linalg.norm(positions[0] - positions[1])))
    return {
        "operation": "task_success_first_grasp_context",
        "status": "complete",
        "read_only": True,
        "instruction": str(task.language),
        "native_goal_predicate": predicate,
        "native_goal_value": bool(env.env._eval_predicate(predicate)),
        "target_object_id": target_id,
        "target_binding": binding,
        "target_root_pose_world": {
            "position_m": root_position.tolist(),
            "rotation": root_rotation.tolist(),
        },
        "target_collision_geometries": geom_rows,
        "eef_pose_world": {
            "position_m": eef_position.tolist(),
            "rotation": eef_rotation.tolist(),
        },
        "fingertip_pad_geometries": pads,
        "symmetric_pad_center_eef_m": ((left_center + right_center) * 0.5).tolist(),
        "gripper_pad_separation_m": float(np.linalg.norm(left_center - right_center)),
        "active_open_gripper_qpos": gripper_qpos.tolist(),
        "gripper_joint_ranges": gripper_joint_ranges.tolist(),
        "maximum_pad_separation_m": max_pad_separation,
        "arm_qpos": arm_qpos.tolist(),
        "model_geometry_sha256": _planning_model_signature(model),
        "bddl_sha256": scene_metadata.get("bddl_sha256"),
    }


def _task_success_first_articulation_context(request):
    """Resolve a real hinged/slide fixture path using scratch MuJoCo state."""
    if request.get("expected_instruction") != task.language:
        raise ValueError("task_instruction_does_not_match_active_BDDL_task")
    predicate = request.get("predicate")
    if not isinstance(predicate, (list, tuple)) or len(predicate) != 2:
        raise ValueError("articulation_context_requires_a_native_unary_fixture_goal")
    predicate = [str(item) for item in predicate]
    if predicate[0].lower() not in {"close", "open"}:
        raise ValueError("articulation_context_supports_only_native_Close_or_Open_goals")
    if predicate not in env.env.parsed_problem.get("goal_state", []):
        raise ValueError("requested_fixture_predicate_is_not_an_active_BDDL_goal")
    fixture_id = predicate[1]
    fixture = getattr(env.env, "fixtures_dict", {}).get(fixture_id)
    if fixture is None:
        raise ValueError(f"active_BDDL_fixture_binding_is_missing:{fixture_id}")
    model, live = env.sim.model._model, env.sim.data._data
    root = int(env.sim.model.body_name2id(fixture.root_body))
    descendants = _planning_body_subtree(model, root)
    joint_rows = []
    for joint_id, body_id in enumerate(model.jnt_bodyid):
        body_id = int(body_id)
        if body_id not in descendants or not bool(model.jnt_limited[joint_id]):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)):
            continue
        qadr = int(model.jnt_qposadr[joint_id])
        dofadr = int(model.jnt_dofadr[joint_id])
        current = float(live.qpos[qadr])
        lower, upper = (float(x) for x in model.jnt_range[joint_id])
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_rows.append({
            "joint_id": int(joint_id), "joint_name": str(joint_name or f"joint_{joint_id}"),
            "body_id": body_id, "body_name": str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)),
            "joint_type": "hinge" if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE) else "slide",
            "qpos_address": qadr, "dof_address": dofadr, "position": current,
            "range": [lower, upper],
        })
    if not joint_rows:
        raise ValueError("active_fixture_has_no_limited_hinge_or_slide_joint")

    articulation = getattr(fixture, "object_properties", {}).get("articulation", {})
    close_ranges = articulation.get(f"default_{predicate[0].lower()}_ranges", [])
    target_ranges = []
    if isinstance(close_ranges, (list, tuple)) and len(close_ranges) == 2:
        target_ranges = [float(close_ranges[0]), float(close_ranges[1])]
    if predicate[0].lower() == "close" and not target_ranges:
        target_ranges = [0.0, 0.0]
    if predicate[0].lower() == "open" and not target_ranges:
        target_ranges = [joint_rows[0]["range"][0], joint_rows[0]["range"][0]]

    selected = None
    target_position = None
    best_distance = -1.0
    for row in joint_rows:
        lo, hi = row["range"]
        clipped = [float(np.clip(value, lo, hi)) for value in target_ranges]
        candidate_target = float(np.mean(clipped))
        current = row["position"]
        distance = abs(current - candidate_target) / max(hi - lo, 1e-9)
        if distance > best_distance:
            selected, target_position, best_distance = row, candidate_target, distance
    assert selected is not None and target_position is not None

    joint_body = int(selected["body_id"])
    moving_bodies = _planning_body_subtree(model, joint_body)
    geom_ids = [
        geom_id for geom_id, body_id in enumerate(model.geom_bodyid)
        if int(body_id) in moving_bodies
        and (int(model.geom_contype[geom_id]) or int(model.geom_conaffinity[geom_id]))
    ]
    if not geom_ids:
        raise ValueError("active_articulated_link_has_no_collision_geometry")
    integration = _mujoco_integration_state(env.sim)
    scratch = mujoco.MjData(model)
    qadr, dofadr = int(selected["qpos_address"]), int(selected["dof_address"])
    samples = np.linspace(float(selected["position"]), target_position, 9)
    path_by_geom = {int(geom_id): [] for geom_id in geom_ids}
    hinge_origin = np.asarray(live.xpos[joint_body], dtype=np.float64)
    radial_scores = {}
    for angle in samples:
        mujoco.mj_setState(model, scratch, integration, mujoco.mjtState.mjSTATE_INTEGRATION)
        scratch.qpos[qadr] = float(angle)
        scratch.qvel[dofadr] = 0.0
        mujoco.mj_forward(model, scratch)
        hinge_origin = np.asarray(scratch.xpos[joint_body], dtype=np.float64)
        for geom_id in geom_ids:
            center = np.asarray(scratch.geom_xpos[geom_id], dtype=np.float64)
            path_by_geom[int(geom_id)].append(center.tolist())
            if len(path_by_geom[int(geom_id)]) == 1:
                radial_scores[int(geom_id)] = float(np.linalg.norm(center - hinge_origin))
    handle_geom = max(geom_ids, key=lambda geom_id: radial_scores[int(geom_id)])
    centers = np.asarray(path_by_geom[int(handle_geom)], dtype=np.float64)
    axis_local = np.asarray(model.jnt_axis[int(selected["joint_id"])], dtype=np.float64)
    axis_world = np.asarray(live.xmat[joint_body], dtype=np.float64).reshape(3, 3) @ axis_local
    axis_world /= max(float(np.linalg.norm(axis_world)), 1e-12)
    radial = centers[0] - np.asarray(live.xpos[joint_body], dtype=np.float64)
    radial -= axis_world * float(np.dot(radial, axis_world))
    radial /= max(float(np.linalg.norm(radial)), 1e-12)
    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(handle_geom))
    return {
        "operation": "task_success_first_articulation_context",
        "status": "complete", "read_only": True,
        "fixture_id_from_active_BDDL": fixture_id,
        "fixture_root_body_id": root,
        "fixture_root_body_name": str(fixture.root_body),
        "native_goal_predicate": predicate,
        "native_goal_value": bool(env.env._eval_predicate(predicate)),
        "joint": {**selected, "target_position": target_position, "close_open_range": target_ranges},
        "articulated_contact_geom_id": int(handle_geom),
        "articulated_contact_geom_name": str(geom_name or f"geom_{handle_geom}"),
        "contact_point_path_world_m": centers.tolist(),
        "outward_direction_world": radial.tolist(),
        "model_geometry_sha256": _planning_model_signature(model),
        "bddl_sha256": scene_metadata.get("bddl_sha256"),
    }


def _validated_render_resolution(resolution):
    """Validate an explicit native square render size without coercion."""
    if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 0:
        raise ValueError(
            f"Render resolution must be a positive integer, got {resolution!r}"
        )
    return resolution


def _render_highres(resolution=512):
    """Render the existing primary camera without advancing or rebuilding sim."""
    resolution = _validated_render_resolution(resolution)
    sim = env.sim
    if getattr(sim, "_render_context_offscreen", None) is None:
        raise RuntimeError(
            "Offscreen render context is not initialized; refusing to create one "
            "because its constructor calls sim.forward()"
        )

    render_context = sim._render_context_offscreen
    visual_global = sim.model.vis.global_
    old_offscreen_size = (visual_global.offwidth, visual_global.offheight)
    old_camera_state = (render_context.cam.type, render_context.cam.fixedcamid)
    state_before = _mujoco_integration_state(sim).copy()
    try:
        raw_rgb = np.ascontiguousarray(
            sim.render(
                width=resolution,
                height=resolution,
                camera_name="agentview",
                depth=False,
            )
        )
    finally:
        # robosuite may grow the offscreen framebuffer and changes these visual
        # settings while selecting the requested camera. Keep the prior render
        # configuration for the normal 256x256 observation path.
        visual_global.offwidth, visual_global.offheight = old_offscreen_size
        render_context.cam.type, render_context.cam.fixedcamid = old_camera_state
    state_after = _mujoco_integration_state(sim)
    if not np.array_equal(state_before, state_after, equal_nan=True):
        raise RuntimeError(
            "MuJoCo integration state changed during high-resolution render; "
            "discarding rendered image"
        )
    if raw_rgb.shape != (resolution, resolution, 3) or raw_rgb.dtype != np.uint8:
        raise RuntimeError(
            "Unexpected high-resolution render result: "
            f"shape={raw_rgb.shape}, dtype={raw_rgb.dtype}"
        )

    rotated_rgb = np.ascontiguousarray(np.rot90(raw_rgb, k=2))

    def image_payload(image):
        return {
            "shape": tuple(image.shape),
            "dtype": str(image.dtype),
            "data": image.tobytes(),
        }

    return {
        "camera_name": "agentview",
        "width": resolution,
        "height": resolution,
        "color_space": "RGB",
        "render_api": "env.sim.render",
        "native_resolution": True,
        "interpolation_used": False,
        "raw_orientation": "unmodified offscreen renderer row order",
        "rotated_orientation": "raw RGB rotated 180 degrees",
        "physical_state_unchanged": True,
        "images": {
            "raw_rgb": image_payload(raw_rgb),
            "rotated_180_rgb": image_payload(rotated_rgb),
        },
    }


def _decode_geom_segmentation(object_segmentation, geom_count, geom_object_type):
    """Convert robosuite's (MuJoCo object type, object id) image to Geom IDs."""
    object_segmentation = np.asarray(object_segmentation)
    if (
        object_segmentation.ndim != 3
        or object_segmentation.shape[2] != 2
        or not np.issubdtype(object_segmentation.dtype, np.integer)
    ):
        raise RuntimeError(
            "Unsupported robosuite segmentation result; expected integer "
            f"HxWx2 object-type/object-id array, got "
            f"shape={object_segmentation.shape}, dtype={object_segmentation.dtype}"
        )

    object_types = object_segmentation[:, :, 0]
    object_ids = object_segmentation[:, :, 1]
    geom_pixels = object_types == geom_object_type
    invalid_geom_pixels = geom_pixels & (
        (object_ids < 0) | (object_ids >= geom_count)
    )
    if np.any(invalid_geom_pixels):
        bad_id = int(object_ids[invalid_geom_pixels][0])
        raise RuntimeError(
            f"Segmentation returned invalid Geom ID {bad_id}; model has "
            f"{geom_count} Geoms"
        )

    background_pixels = (object_types == -1) & (object_ids == -1)
    geom_ids = np.full(object_ids.shape, -2, dtype=np.int32)
    geom_ids[background_pixels] = -1
    geom_ids[geom_pixels] = object_ids[geom_pixels].astype(np.int32)
    return geom_ids


def _render_geom_segmentation(resolution=512):
    """Capture same-state RGB and object segmentation from the primary camera."""
    resolution = _validated_render_resolution(resolution)
    sim = env.sim
    if getattr(sim, "_render_context_offscreen", None) is None:
        raise RuntimeError(
            "Offscreen render context is not initialized; refusing to create one "
            "because its constructor calls sim.forward()"
        )

    render_context = sim._render_context_offscreen
    visual_global = sim.model.vis.global_
    old_offscreen_size = (visual_global.offwidth, visual_global.offheight)
    old_camera_state = (render_context.cam.type, render_context.cam.fixedcamid)
    state_before = _mujoco_integration_state(sim).copy()
    try:
        rgb_before = np.ascontiguousarray(
            sim.render(
                width=resolution,
                height=resolution,
                camera_name="agentview",
                depth=False,
                segmentation=False,
            )
        )
        object_segmentation = sim.render(
            width=resolution,
            height=resolution,
            camera_name="agentview",
            depth=False,
            segmentation=True,
        )
        rgb_after = np.ascontiguousarray(
            sim.render(
                width=resolution,
                height=resolution,
                camera_name="agentview",
                depth=False,
                segmentation=False,
            )
        )
    finally:
        visual_global.offwidth, visual_global.offheight = old_offscreen_size
        render_context.cam.type, render_context.cam.fixedcamid = old_camera_state

    state_after = _mujoco_integration_state(sim)
    if not np.array_equal(state_before, state_after, equal_nan=True):
        raise RuntimeError(
            "MuJoCo integration state changed during Geom segmentation render"
        )
    if rgb_before.shape != (resolution, resolution, 3) or rgb_before.dtype != np.uint8:
        raise RuntimeError(
            "Unexpected RGB render result: "
            f"shape={rgb_before.shape}, dtype={rgb_before.dtype}"
        )
    if rgb_after.shape != rgb_before.shape or not np.array_equal(rgb_before, rgb_after):
        raise RuntimeError(
            "RGB output changed across the segmentation render; pixel alignment "
            "cannot be confirmed"
        )

    geom_count = int(sim.model._model.ngeom)
    geom_ids = _decode_geom_segmentation(
        object_segmentation,
        geom_count=geom_count,
        geom_object_type=int(mujoco.mjtObj.mjOBJ_GEOM),
    )
    visible_geom_ids = np.unique(geom_ids[geom_ids >= 0]).tolist()
    visible_geoms = []
    for geom_id in visible_geom_ids:
        body_id = int(sim.model.geom_bodyid[geom_id])
        visible_geoms.append(
            {
                "geom_id": int(geom_id),
                "geom_name": sim.model.geom_id2name(int(geom_id)),
                "body_id": body_id,
                "body_name": sim.model.body_id2name(body_id),
                "pixel_count": int(np.count_nonzero(geom_ids == geom_id)),
            }
        )

    rgb = np.ascontiguousarray(rgb_before)
    geom_ids = np.ascontiguousarray(geom_ids, dtype=np.int32)

    def image_payload(image):
        return {
            "shape": tuple(image.shape),
            "dtype": str(image.dtype),
            "data": image.tobytes(),
        }

    return {
        "camera_name": "agentview",
        "width": resolution,
        "height": resolution,
        "color_space": "RGB",
        "render_api": "env.sim.render",
        "native_resolution": True,
        "interpolation_used": False,
        "raw_orientation": "unmodified offscreen renderer row/column order",
        "rgb_segmentation_pixel_consistency": True,
        "physical_state_unchanged": True,
        "geom_id_background_value": -1,
        "non_geom_object_value": -2,
        "model_geom_count": geom_count,
        "background_pixel_count": int(np.count_nonzero(geom_ids == -1)),
        "non_geom_pixel_count": int(np.count_nonzero(geom_ids == -2)),
        "visible_geoms": visible_geoms,
        "images": {
            "rgb": image_payload(rgb),
            "geom_ids": image_payload(geom_ids),
        },
    }


def _collision_geom_ids(body_ids):
    """Return every active collision Geom attached to the selected Bodies."""
    model = env.sim.model._model
    body_ids = set(int(value) for value in body_ids)
    explicit_pair_geoms = set()
    for pair_index in range(int(model.npair)):
        explicit_pair_geoms.add(int(model.pair_geom1[pair_index]))
        explicit_pair_geoms.add(int(model.pair_geom2[pair_index]))
    return {
        geom_id
        for geom_id in range(int(model.ngeom))
        if int(model.geom_bodyid[geom_id]) in body_ids
        and (
            int(model.geom_contype[geom_id]) != 0
            or int(model.geom_conaffinity[geom_id]) != 0
            or geom_id in explicit_pair_geoms
        )
    }


_STATIC_INSTALLATION_MODEL_SIGNATURE_CACHE = {"model": None, "signature": None}
_VERIFIED_STATIC_INSTALLATION_CONTACTS = (
    {
        "model_geometry_sha256": "9e2f374b19a09a123e715dcefc0cebaa11d647b4029d4546bc6b52a2944d451e",
        "robot_geom_id": 90,
        "robot_geom_name": "mount0_pedestal_col",
        "robot_body_id": 22,
        "robot_body_name": "mount0_pedestal",
        "scene_geom_id": 14,
        "scene_geom_name": None,
        "scene_body_id": 1,
        "scene_body_name": "study_table",
    },
)


def _fixed_body_ancestor_chain(model, body_id):
    """Return a verified fixed Body chain, or fail if any ancestor is jointed."""
    chain = []
    current = int(body_id)
    while True:
        joints = [
            joint_id
            for joint_id in range(int(model.njnt))
            if int(model.jnt_bodyid[joint_id]) == current
        ]
        if joints:
            raise ValueError(
                f"static_installation_body_has_jointed_ancestor:body={current},joints={joints}"
            )
        chain.append(current)
        if current == 0:
            return chain
        parent = int(model.body_parentid[current])
        if parent == current or parent < 0 or parent >= int(model.nbody):
            raise ValueError("static_installation_body_ancestor_chain_is_invalid")
        current = parent


def _normalize_static_installation_contact_pairs(
    raw_pairs, model, robot_body_ids, robot_geom_ids, other_scene_groups
):
    """Validate the narrowly pinned L0T1 fixed mount/table contact exception."""
    if not isinstance(raw_pairs, list):
        raise ValueError("allowed_static_installation_contact_pairs_must_be_an_array")
    if not raw_pairs:
        return set(), []
    if len(raw_pairs) != 1 or len(_VERIFIED_STATIC_INSTALLATION_CONTACTS) != 1:
        raise ValueError("static_installation_contact_pair_is_not_in_the_verified_allowlist")

    expected = _VERIFIED_STATIC_INSTALLATION_CONTACTS[0]
    pair = raw_pairs[0]
    if not isinstance(pair, dict) or pair != expected:
        raise ValueError("static_installation_contact_pair_identity_is_not_in_the_verified_allowlist")

    cache = _STATIC_INSTALLATION_MODEL_SIGNATURE_CACHE
    if cache["model"] is not model:
        cache["model"] = model
        cache["signature"] = _planning_model_signature(model)
    if cache["signature"] != expected["model_geometry_sha256"]:
        raise ValueError("static_installation_contact_model_fingerprint_is_not_verified")

    robot_geom = int(expected["robot_geom_id"])
    scene_geom = int(expected["scene_geom_id"])
    robot_body = int(expected["robot_body_id"])
    scene_body = int(expected["scene_body_id"])
    if (
        robot_geom not in robot_geom_ids
        or scene_geom < 0
        or scene_geom >= int(model.ngeom)
        or robot_body not in robot_body_ids
        or int(model.geom_bodyid[robot_geom]) != robot_body
        or int(model.geom_bodyid[scene_geom]) != scene_body
        or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom)
        != expected["robot_geom_name"]
        or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, scene_geom)
        != expected["scene_geom_name"]
        or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, robot_body)
        != expected["robot_body_name"]
        or mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, scene_body)
        != expected["scene_body_name"]
    ):
        raise ValueError("static_installation_contact_geom_or_body_identity_mismatch")
    if not any(
        scene_geom in group["geom_ids"] and scene_body in group["body_ids"]
        for group in other_scene_groups
    ):
        raise ValueError("static_installation_contact_scene_geom_is_not_a_verified_other_scene_geom")

    robot_chain = _fixed_body_ancestor_chain(model, robot_body)
    scene_chain = _fixed_body_ancestor_chain(model, scene_body)
    pair_key = tuple(sorted((robot_geom, scene_geom)))
    evidence = {
        **expected,
        "robot_fixed_body_ancestor_ids": robot_chain,
        "scene_fixed_body_ancestor_ids": scene_chain,
        "verification": "model_fingerprint_and_geom_body_identity_match; both ancestor chains contain no joints",
    }
    return {pair_key}, [evidence]


def _static_installation_pairs_for_group(policy, group):
    """Return only verified mount/scene pairs whose scene Geom is in this group."""
    return {
        pair
        for pair in policy.get("allowed_static_installation_contact_pair_ids", set())
        if any(
            geom_id not in policy["robot_geom_ids"] and geom_id in group["geom_ids"]
            for geom_id in pair
        )
    }


def _is_allowed_static_installation_contact(policy, pair_key):
    return tuple(sorted(int(value) for value in pair_key)) in policy.get(
        "allowed_static_installation_contact_pair_ids", set()
    )


def _normalize_safety_geometry_policy(raw_policy):
    """Verify semantic Body/Geom bindings before substep safety checks."""
    model = env.sim.model._model
    if not isinstance(raw_policy, dict):
        raise ValueError("safety_geometry_policy_is_missing")
    margin = raw_policy.get("minimum_clearance_m")
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not np.isfinite(float(margin))
        or float(margin) < 0.0
    ):
        raise ValueError("minimum_clearance_m_must_be_non_negative")
    joint_limit_tolerance_policy = normalize_joint_limit_tolerance_policy(
        raw_policy.get("joint_limit_tolerance_policy", JOINT_LIMIT_TOLERANCE_POLICY)
    )
    robot_bodies = _robot_body_ids()
    if not robot_bodies:
        raise ValueError("complete_robot_body_inventory_unavailable")
    robot_geoms = _collision_geom_ids(robot_bodies)
    if not robot_geoms:
        raise ValueError("complete_robot_collision_geom_inventory_unavailable")
    inventory = _geometry_inventory()
    if inventory.get("status") != "complete":
        raise ValueError("complete_geometry_inventory_unavailable")
    inventory_by_id = {
        item.get("geom_id"): item
        for item in inventory.get("geometries", [])
        if isinstance(item, dict)
    }
    groups = []
    seen_body_ids = set()
    seen_geom_ids = set()

    def parse_groups(raw_groups, group_name, minimum_default):
        if not isinstance(raw_groups, list):
            raise ValueError(f"{group_name}_must_be_an_explicit_array")
        parsed = []
        for index, item in enumerate(raw_groups):
            if not isinstance(item, dict):
                raise ValueError(f"{group_name}[{index}]_must_be_an_object")
            object_id = item.get("object_id")
            body_ids = item.get("body_ids")
            body_names = item.get("body_names")
            geom_ids = item.get("geom_ids")
            if (
                not isinstance(object_id, str)
                or not object_id
                or not isinstance(body_ids, list)
                or not body_ids
                or not isinstance(body_names, list)
                or len(body_names) != len(body_ids)
                or not isinstance(geom_ids, list)
                or not geom_ids
            ):
                raise ValueError(f"{group_name}[{index}]_binding_is_incomplete")
            minimum_body_id = 0 if group_name == "other_scene_objects" else 1
            if any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < minimum_body_id
                or value >= int(env.sim.model._model.nbody)
                for value in body_ids
            ) or len(set(body_ids)) != len(body_ids):
                raise ValueError(f"{group_name}[{index}]_body_ids_are_invalid")
            if any(
                not isinstance(name, str)
                or env.sim.model.body_id2name(body_id) != name
                for body_id, name in zip(body_ids, body_names)
            ):
                raise ValueError(f"{group_name}[{index}]_body_names_do_not_match_model")
            if set(body_ids) & (robot_bodies | seen_body_ids):
                raise ValueError(f"{group_name}[{index}]_overlaps_robot_or_prior_binding")
            expected_geoms = _collision_geom_ids(body_ids)
            if set(geom_ids) != expected_geoms or not expected_geoms:
                raise ValueError(f"{group_name}[{index}]_does_not_cover_all_collision_geoms")
            if any(
                inventory_by_id.get(geom_id, {}).get("category") != "scene"
                or inventory_by_id.get(geom_id, {}).get("body_id") not in set(body_ids)
                for geom_id in geom_ids
            ):
                raise ValueError(f"{group_name}[{index}]_geom_identity_is_unverified")
            if set(geom_ids) & seen_geom_ids:
                raise ValueError(f"{group_name}[{index}]_duplicates_prior_geometry")
            object_margin = item.get("minimum_clearance_m", minimum_default)
            if (
                isinstance(object_margin, bool)
                or not isinstance(object_margin, (int, float))
                or not np.isfinite(float(object_margin))
                or float(object_margin) < 0.0
            ):
                raise ValueError(f"{group_name}[{index}]_clearance_must_be_non_negative")
            group = {
                "object_id": object_id,
                "body_ids": set(body_ids),
                "geom_ids": set(geom_ids),
                "minimum_clearance_m": float(object_margin),
                "policy_class": group_name,
            }
            parsed.append(group)
            groups.append(group)
            seen_body_ids.update(body_ids)
            seen_geom_ids.update(geom_ids)
        return parsed

    forbidden_groups = parse_groups(
        raw_policy.get("forbidden_objects"),
        "forbidden_objects",
        max(0.010, float(margin)),
    )
    target_groups = parse_groups(
        raw_policy.get("allowed_target_objects", []),
        "allowed_target_objects",
        max(0.010, float(margin)),
    )
    other_groups = parse_groups(raw_policy.get("other_scene_objects", []), "other_scene_objects", 0.0)
    static_installation_raw = raw_policy.get("allowed_static_installation_contact_pairs")
    if static_installation_raw is None:
        # Existing L0T1 policy builders predate this field. Apply the trusted
        # installation registry only when Bridge metadata identifies its exact
        # compiled model; the validator below independently fingerprints it.
        static_installation_raw = [
            dict(spec)
            for spec in _VERIFIED_STATIC_INSTALLATION_CONTACTS
            if scene_metadata.get("model_geometry_sha256")
            == spec["model_geometry_sha256"]
        ]
    (
        static_installation_pair_ids,
        static_installation_pair_bindings,
    ) = _normalize_static_installation_contact_pairs(
        static_installation_raw,
        model,
        robot_bodies,
        robot_geoms,
        other_groups,
    )
    support_geom_ids = raw_policy.get("allowed_carried_support_contact_geom_ids", [])
    if (
        not isinstance(support_geom_ids, list)
        or any(isinstance(value, bool) or not isinstance(value, int) for value in support_geom_ids)
        or len(set(support_geom_ids)) != len(support_geom_ids)
        or not set(support_geom_ids).issubset(
            set().union(*(item["geom_ids"] for item in other_groups)) if other_groups else set()
        )
    ):
        raise ValueError("allowed_carried_support_contact_geoms_must_be_verified_other_scene_geometries")
    if not forbidden_groups:
        raise ValueError("at_least_one_forbidden_contact_object_is_required_for_takeover")
    allowed_body_ids = set().union(*(item["body_ids"] for item in target_groups)) if target_groups else set()
    allowed_pairs = raw_policy.get("allowed_contact_pairs", [])
    if not isinstance(allowed_pairs, list):
        raise ValueError("allowed_contact_pairs_must_be_an_array")
    pair_ids = set()
    allowed_pair_min_clearance = {}
    for index, pair in enumerate(allowed_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"allowed_contact_pairs[{index}]_must_be_an_object")
        robot_geom = pair.get("robot_geom_id")
        scene_geom = pair.get("scene_geom_id")
        if (
            isinstance(robot_geom, bool)
            or not isinstance(robot_geom, int)
            or robot_geom not in robot_geoms
            or isinstance(scene_geom, bool)
            or not isinstance(scene_geom, int)
            or scene_geom < 0
            or scene_geom >= int(env.sim.model._model.ngeom)
            or int(env.sim.model._model.geom_bodyid[scene_geom]) not in allowed_body_ids
        ):
            raise ValueError(f"allowed_contact_pairs[{index}]_does_not_match_verified_objects")
        pair_key = tuple(sorted((robot_geom, scene_geom)))
        minimum = pair.get("minimum_clearance_m", -0.002)
        if (
            isinstance(minimum, bool)
            or not isinstance(minimum, (int, float))
            or not np.isfinite(float(minimum))
            or float(minimum) < -0.002
        ):
            raise ValueError(
                f"allowed_contact_pairs[{index}]_penetration_tolerance_exceeds_2mm"
            )
        pair_ids.add(pair_key)
        allowed_pair_min_clearance[pair_key] = float(minimum)
    object_groups = {item["object_id"]: item for item in groups}
    forbidden_pairs = raw_policy.get("forbidden_object_pairs", [])
    if not isinstance(forbidden_pairs, list):
        raise ValueError("forbidden_object_pairs_must_be_an_array")
    normalized_forbidden_pairs = []
    seen_forbidden_body_pairs = set()
    for index, pair in enumerate(forbidden_pairs):
        if not isinstance(pair, dict):
            raise ValueError(f"forbidden_object_pairs[{index}]_must_be_an_object")
        object_a = object_groups.get(pair.get("object_id_a"))
        object_b = object_groups.get(pair.get("object_id_b"))
        if (
            object_a is None
            or object_b is None
            or object_a is object_b
            or set(pair.get("geom_ids_a", [])) != object_a["geom_ids"]
            or set(pair.get("geom_ids_b", [])) != object_b["geom_ids"]
            or set(pair.get("body_ids_a", [])) != object_a["body_ids"]
            or set(pair.get("body_ids_b", [])) != object_b["body_ids"]
        ):
            raise ValueError(f"forbidden_object_pairs[{index}]_does_not_match_verified_bindings")
        body_pair = tuple(sorted((tuple(sorted(object_a["body_ids"])), tuple(sorted(object_b["body_ids"])))) )
        if body_pair in seen_forbidden_body_pairs:
            raise ValueError(f"forbidden_object_pairs[{index}]_is_duplicated")
        seen_forbidden_body_pairs.add(body_pair)
        normalized_forbidden_pairs.append({
            "object_id_a": object_a["object_id"],
            "object_id_b": object_b["object_id"],
            "body_ids_a": object_a["body_ids"],
            "body_ids_b": object_b["body_ids"],
            "geom_ids_a": object_a["geom_ids"],
            "geom_ids_b": object_b["geom_ids"],
        })

    carried_raw = raw_policy.get("carried_object")
    carried_object = None
    if carried_raw is not None:
        if not isinstance(carried_raw, dict):
            raise ValueError("carried_object_must_be_an_object_or_null")
        carried_group = object_groups.get(carried_raw.get("object_id"))
        binding = carried_raw.get("grasp_binding")
        if carried_group is None or not isinstance(binding, dict):
            raise ValueError("carried_object_must_match_a_verified_scene_binding")
        if (
            binding.get("object_id") != carried_group["object_id"]
            or binding.get("body_id") not in carried_group["body_ids"]
            or not isinstance(binding.get("body_name"), str)
            or set(binding.get("geom_ids", [])) != _collision_geom_ids({binding.get("body_id")})
            or len(binding.get("geom_names", [])) != len(binding.get("geom_ids", []))
        ):
            raise ValueError("carried_object_grasp_binding_is_incomplete_or_mismatched")
        body_id = int(binding["body_id"])
        if env.sim.model.body_id2name(body_id) != binding["body_name"]:
            raise ValueError("carried_object_body_name_does_not_match_model")
        grasp = _live_grasp_state(binding, task.language)
        if grasp.get("status") != "pass" or grasp.get("physical_state_unchanged") is not True:
            raise ValueError(f"carried_object_grasp_is_not_verified:{grasp.get('reason')}")
        free_joints = [
            int(model.body_jntadr[body_id]) + offset
            for offset in range(int(model.body_jntnum[body_id]))
            if int(model.jnt_type[int(model.body_jntadr[body_id]) + offset])
            == int(mujoco.mjtJoint.mjJNT_FREE)
        ]
        if len(free_joints) != 1:
            raise ValueError("carried_object_root_must_have_exactly_one_free_joint")
        eef_site_id = int(env.robots[0].eef_site_id)
        live = env.sim.data._data
        eef_rotation = np.asarray(live.site_xmat[eef_site_id], dtype=np.float64).reshape(3, 3)
        object_rotation = np.asarray(live.xmat[body_id], dtype=np.float64).reshape(3, 3)
        world_eef = np.eye(4, dtype=np.float64)
        world_eef[:3, :3] = eef_rotation
        world_eef[:3, 3] = live.site_xpos[eef_site_id]
        world_object = np.eye(4, dtype=np.float64)
        world_object[:3, :3] = object_rotation
        world_object[:3, 3] = live.xpos[body_id]
        eef_object = np.linalg.inv(world_eef) @ world_object
        allowed_carried_pairs = carried_raw.get("allowed_contact_pairs", [])
        if not isinstance(allowed_carried_pairs, list):
            raise ValueError("carried_object_allowed_contact_pairs_must_be_an_array")
        carried_pair_ids = set()
        carried_pair_minimum = {}
        for pair_index, pair in enumerate(allowed_carried_pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"carried_object_allowed_contact_pairs[{pair_index}]_must_be_an_object")
            carried_geom = pair.get("carried_geom_id")
            scene_geom = pair.get("scene_geom_id")
            required = pair.get("minimum_clearance_m", -0.002)
            if (
                isinstance(carried_geom, bool) or not isinstance(carried_geom, int)
                or carried_geom not in carried_group["geom_ids"]
                or isinstance(scene_geom, bool) or not isinstance(scene_geom, int)
                or scene_geom not in set().union(*(item["geom_ids"] for item in groups))
                or isinstance(required, bool) or not isinstance(required, (int, float))
                or not np.isfinite(float(required)) or float(required) < -0.002
            ):
                raise ValueError(f"carried_object_allowed_contact_pairs[{pair_index}]_is_invalid")
            scene_group = next(item for item in groups if scene_geom in item["geom_ids"])
            if (
                scene_group["object_id"] == carried_group["object_id"]
                or scene_group["policy_class"] != "allowed_target_objects"
            ):
                raise ValueError(f"carried_object_allowed_contact_pairs[{pair_index}]_is_not_a_declared_target")
            key = tuple(sorted((carried_geom, scene_geom)))
            carried_pair_ids.add(key)
            carried_pair_minimum[key] = float(required)
        allowed_support_pairs = carried_raw.get("allowed_support_contact_pairs", [])
        if not isinstance(allowed_support_pairs, list):
            raise ValueError("carried_object_allowed_support_contact_pairs_must_be_an_array")
        for pair_index, pair in enumerate(allowed_support_pairs):
            if not isinstance(pair, dict):
                raise ValueError(f"carried_object_allowed_support_contact_pairs[{pair_index}]_must_be_an_object")
            carried_geom = pair.get("carried_geom_id")
            scene_geom = pair.get("scene_geom_id")
            required = pair.get("minimum_clearance_m", -0.002)
            if (
                isinstance(carried_geom, bool) or not isinstance(carried_geom, int)
                or carried_geom not in carried_group["geom_ids"]
                or isinstance(scene_geom, bool) or not isinstance(scene_geom, int)
                or scene_geom not in support_geom_ids
                or isinstance(required, bool) or not isinstance(required, (int, float))
                or not np.isfinite(float(required)) or float(required) < -0.002
            ):
                raise ValueError(f"carried_object_allowed_support_contact_pairs[{pair_index}]_is_invalid")
            key = tuple(sorted((carried_geom, scene_geom)))
            if key in carried_pair_ids:
                raise ValueError("carried_object_support_contact_pair_duplicates_an_existing_pair")
            carried_pair_ids.add(key)
            carried_pair_minimum[key] = float(required)
        finger_groups, _ = _tracking_gripper_sides(env.robots[0])
        expected_gripper_geoms = set().union(*finger_groups.values())
        carried_object = {
            "object_id": carried_group["object_id"],
            "body_id": body_id,
            "free_joint_id": free_joints[0],
            "free_joint_qpos_adr": int(model.jnt_qposadr[free_joints[0]]),
            "free_joint_dof_adr": int(model.jnt_dofadr[free_joints[0]]),
            "geom_ids": set(carried_group["geom_ids"]),
            "gripper_geom_ids": expected_gripper_geoms,
            "eef_to_object_transform": eef_object,
            "allowed_contact_pair_ids": carried_pair_ids,
            "allowed_pair_minimum_clearance_m": carried_pair_minimum,
            "minimum_clearance_m": carried_group["minimum_clearance_m"],
            "binding": binding,
        }
        if not carried_object["gripper_geom_ids"].issubset(robot_geoms):
            raise ValueError("carried_object_gripper_geoms_do_not_match_the_active_robot")
    classified_scene_body_ids = set().union(*(item["body_ids"] for item in groups)) if groups else set()
    all_active_collision_geoms = _collision_geom_ids(range(int(model.nbody)))
    all_scene_collision_body_ids = {
        int(model.geom_bodyid[geom_id])
        for geom_id in all_active_collision_geoms
        if int(model.geom_bodyid[geom_id]) not in robot_bodies
    }
    if all_scene_collision_body_ids != classified_scene_body_ids:
        missing = sorted(all_scene_collision_body_ids - classified_scene_body_ids)
        extra = sorted(classified_scene_body_ids - all_scene_collision_body_ids)
        raise ValueError(
            f"scene_collision_body_classification_incomplete:missing={missing},extra={extra}"
        )
    return {
        "minimum_clearance_m": float(margin),
        "robot_body_ids": robot_bodies,
        "robot_geom_ids": robot_geoms,
        "forbidden_objects": forbidden_groups,
        "checked_objects": groups,
        "allowed_target_objects": target_groups,
        "other_scene_objects": other_groups,
        "allowed_contact_pair_ids": pair_ids,
        "allowed_pair_minimum_clearance_m": allowed_pair_min_clearance,
        "allowed_static_installation_contact_pair_ids": static_installation_pair_ids,
        "allowed_static_installation_contact_pair_bindings": static_installation_pair_bindings,
        "allowed_target_body_ids": allowed_body_ids,
        "classified_scene_body_ids": classified_scene_body_ids,
        "forbidden_scene_body_ids": set().union(*(item["body_ids"] for item in forbidden_groups)) if forbidden_groups else set(),
        "other_scene_body_ids": set().union(*(item["body_ids"] for item in other_groups)) if other_groups else set(),
        "allowed_carried_support_contact_geom_ids": set(support_geom_ids),
        "forbidden_object_pairs": normalized_forbidden_pairs,
        "carried_object": carried_object,
        "joint_limit_tolerance_policy": joint_limit_tolerance_policy,
        "source": raw_policy.get("source"),
    }


def _minimum_geom_clearance(model, data, geom_ids_a, geom_ids_b, excluded_pairs=None):
    """Use conservative rbound pruning and exact MuJoCo distances for the minimum."""
    if not geom_ids_a or not geom_ids_b:
        return {"status": "incomplete", "minimum_distance_m": None, "pair": None}
    excluded_pairs = excluded_pairs or set()
    pairs = []
    for geom_a in geom_ids_a:
        for geom_b in geom_ids_b:
            if tuple(sorted((int(geom_a), int(geom_b)))) in excluded_pairs:
                continue
            lower_bound = float(
                np.linalg.norm(data.geom_xpos[geom_a] - data.geom_xpos[geom_b])
                - model.geom_rbound[geom_a]
                - model.geom_rbound[geom_b]
            )
            pairs.append((lower_bound, int(geom_a), int(geom_b)))
    if not pairs:
        return {"status": "complete", "minimum_distance_m": None, "pair": None,
                "attempted_pairs": 0, "requested_pairs": 0}
    pairs.sort(key=lambda item: item[0])
    minimum = None
    minimum_pair = None
    attempted = 0
    censored_pairs = 0
    for lower_bound, geom_a, geom_b in pairs:
        if minimum is not None and lower_bound >= minimum:
            break
        segment = np.zeros(6, dtype=np.float64)
        distance = float(mujoco.mj_geomDistance(model, data, geom_a, geom_b, 10.0, segment))
        attempted += 1
        if not np.isfinite(distance) or not np.isfinite(segment).all():
            return {"status": "incomplete", "minimum_distance_m": None, "pair": None,
                    "attempted_pairs": attempted, "requested_pairs": len(pairs)}
        if distance >= 10.0:
            censored_pairs += 1
            continue
        if minimum is None or distance < minimum:
            minimum = distance
            minimum_pair = {
                "robot_geom_id": geom_a,
                "robot_geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_a),
                "object_geom_id": geom_b,
                "object_geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_b),
                "closest_points_world": segment.tolist(),
            }
    if minimum is None:
        complete = censored_pairs == len(pairs)
    else:
        complete = all(lower_bound >= minimum for lower_bound, _, _ in pairs[attempted:])
    return {
        "status": "complete" if complete else "incomplete",
        "minimum_distance_m": minimum if minimum is not None else 10.0 if complete else None,
        "pair": minimum_pair,
        "attempted_pairs": attempted,
        "requested_pairs": len(pairs),
        "minimum_is_lower_bound": minimum is None and complete,
        "distance_censored_pairs": censored_pairs,
    }


def _exact_geom_pair_distance(model, data, geom_a, geom_b):
    segment = np.zeros(6, dtype=np.float64)
    distance = float(mujoco.mj_geomDistance(model, data, int(geom_a), int(geom_b), 10.0, segment))
    if not np.isfinite(distance) or not np.isfinite(segment).all():
        return None
    return distance


def _substep_safety_geometry(monitor, contact_result):
    """Measure all robot-to-forbidden Geom clearances after each physical substep."""
    policy = monitor.get("safety_policy")
    if policy is None:
        return None
    sim = env.sim
    model = sim.model._model
    item = {
        "status": "complete",
        "minimum_clearance_m": None,
        "minimum_all_clearance_m": None,
        "minimum_carried_forbidden_clearance_m": None,
        "clearances_by_object": {},
        "carried_clearances_by_object": {},
        "forbidden_contacts": [],
        "self_contacts": [],
        "unresolved_contacts": [],
        "forbidden_object_contacts": [],
        "unallowed_scene_contacts": [],
        "contact_identities": [],
        "joint_limit_violations": [],
        "strict_joint_limit_violations": [],
        "allowed_static_installation_contact_pairs": policy[
            "allowed_static_installation_contact_pair_bindings"
        ],
    }
    try:
        if contact_result.get("status") not in ("contact", "no_contact"):
            item["status"] = "incomplete"
        for group in policy["checked_objects"]:
            allowed_pairs = {
                pair for pair in policy["allowed_contact_pair_ids"]
                if next(
                    geom_id for geom_id in pair
                    if geom_id not in policy["robot_geom_ids"]
                ) in group["geom_ids"]
            } if group["policy_class"] == "allowed_target_objects" else set()
            excluded_pairs = allowed_pairs | _static_installation_pairs_for_group(
                policy, group
            )
            clearance = _minimum_geom_clearance(
                model,
                sim.data._data,
                policy["robot_geom_ids"],
                group["geom_ids"],
                excluded_pairs,
            )
            item["clearances_by_object"][group["object_id"]] = clearance
            if clearance.get("status") != "complete":
                item["status"] = "incomplete"
            distance = clearance.get("minimum_distance_m")
            if distance is not None:
                if item["minimum_all_clearance_m"] is None or distance < item["minimum_all_clearance_m"]:
                    item["minimum_all_clearance_m"] = distance
                if group["policy_class"] == "forbidden_objects" and (
                    item["minimum_clearance_m"] is None or distance < item["minimum_clearance_m"]
                ):
                    item["minimum_clearance_m"] = distance
                if distance < group["minimum_clearance_m"]:
                    item["status"] = "unsafe"
                    item["forbidden_contacts"].append({
                        "geometry_clearance_requirement_violated": True,
                        "object_id": group["object_id"],
                        "distance_m": distance,
                        "minimum_required_m": group["minimum_clearance_m"],
                    })
            for pair in allowed_pairs:
                distance = _exact_geom_pair_distance(model, sim.data._data, *pair)
                required = policy["allowed_pair_minimum_clearance_m"].get(pair)
                if distance is None or required is None:
                    item["status"] = "incomplete"
                else:
                    item.setdefault("allowed_contact_clearances_m", {})[
                        f"{pair[0]}:{pair[1]}"
                    ] = distance
                    if distance < required:
                        item["status"] = "unsafe"
                        item["forbidden_contacts"].append({
                            "allowed_pair_penetration_exceeded": True,
                            "geom_ids": list(pair),
                            "distance_m": distance,
                            "minimum_allowed_distance_m": required,
                        })
        carried = policy.get("carried_object")
        if carried is not None:
            for group in policy["checked_objects"]:
                if group["object_id"] == carried["object_id"]:
                    continue
                allowed_pairs = {
                    pair for pair in carried["allowed_contact_pair_ids"]
                    if next(geom_id for geom_id in pair if geom_id in carried["geom_ids"])
                    in carried["geom_ids"]
                    and next(geom_id for geom_id in pair if geom_id not in carried["geom_ids"])
                    in group["geom_ids"]
                }
                clearance = _minimum_geom_clearance(
                    model,
                    sim.data._data,
                    carried["geom_ids"],
                    group["geom_ids"],
                    allowed_pairs,
                )
                item["carried_clearances_by_object"][group["object_id"]] = clearance
                if clearance.get("status") != "complete":
                    item["status"] = "incomplete"
                distance = clearance.get("minimum_distance_m")
                required = (
                    group["minimum_clearance_m"]
                    if group["policy_class"] == "forbidden_objects"
                    else 0.0
                )
                if distance is not None:
                    if item["minimum_all_clearance_m"] is None or distance < item["minimum_all_clearance_m"]:
                        item["minimum_all_clearance_m"] = distance
                    if group["policy_class"] == "forbidden_objects" and (
                        item["minimum_carried_forbidden_clearance_m"] is None
                        or distance < item["minimum_carried_forbidden_clearance_m"]
                    ):
                        item["minimum_carried_forbidden_clearance_m"] = distance
                    if distance < required:
                        item["status"] = "unsafe"
                        item["forbidden_contacts"].append({
                            "carried_object_clearance_requirement_violated": True,
                            "carried_object_id": carried["object_id"],
                            "object_id": group["object_id"],
                            "distance_m": distance,
                            "minimum_required_m": required,
                        })
                for pair in allowed_pairs:
                    distance = _exact_geom_pair_distance(model, sim.data._data, *pair)
                    required_pair = carried["allowed_pair_minimum_clearance_m"].get(pair)
                    if distance is None or required_pair is None:
                        item["status"] = "incomplete"
                    elif distance < required_pair:
                        item["status"] = "unsafe"
                        item["forbidden_contacts"].append({
                            "carried_object_allowed_pair_penetration_exceeded": True,
                            "geom_ids": list(pair),
                            "distance_m": distance,
                            "minimum_allowed_distance_m": required_pair,
                        })
        for contact in contact_result.get("contacts", []):
            geom_a = int(contact["geom1"]["id"])
            geom_b = int(contact["geom2"]["id"])
            body_a = int(contact["body1"]["id"])
            body_b = int(contact["body2"]["id"])
            item["contact_identities"].append({
                "contact_index": int(contact.get("contact_index", len(item["contact_identities"]))),
                "geom1": contact["geom1"],
                "geom2": contact["geom2"],
                "body1": contact["body1"],
                "body2": contact["body2"],
                "position_world_m": contact.get("position"),
                "normal_geom1_to_geom2_world": contact.get("normal_geom1_to_geom2"),
                "signed_contact_distance_m": contact.get("signed_contact_distance_m"),
            })
            pair_key = tuple(sorted((geom_a, geom_b)))
            if body_a in policy["robot_body_ids"] and body_b in policy["robot_body_ids"]:
                item["self_contacts"].append(contact)
            elif body_a in policy["robot_body_ids"] or body_b in policy["robot_body_ids"]:
                scene_body = body_b if body_a in policy["robot_body_ids"] else body_a
                if scene_body in policy["forbidden_scene_body_ids"]:
                    item["forbidden_contacts"].append(contact)
                elif scene_body in policy["allowed_target_body_ids"]:
                    if pair_key not in policy["allowed_contact_pair_ids"]:
                        item["forbidden_contacts"].append(contact)
                elif scene_body in policy["other_scene_body_ids"]:
                    if not _is_allowed_static_installation_contact(policy, pair_key):
                        item["unallowed_scene_contacts"].append(contact)
                else:
                    item["unresolved_contacts"].append(contact)
            else:
                bodies = {body_a, body_b}
                if any(
                    bodies & pair["body_ids_a"] and bodies & pair["body_ids_b"]
                    for pair in policy["forbidden_object_pairs"]
                ):
                    item["forbidden_object_contacts"].append(contact)
        if item["forbidden_contacts"] or item["self_contacts"]:
            item["status"] = "unsafe"
        if item["forbidden_object_contacts"]:
            item["status"] = "unsafe"
        if item["unallowed_scene_contacts"]:
            item["status"] = "unsafe"
        if item["unresolved_contacts"]:
            item["status"] = "incomplete"
        # Check every limited Panda hinge/slide joint at every native MuJoCo
        # substep; EEF tracking cannot certify redundant-joint feasibility.
        robot_bodies = _robot_body_ids()
        for joint_id in range(int(model.njnt)):
            if (
                int(model.jnt_bodyid[joint_id]) not in robot_bodies
                or not bool(model.jnt_limited[joint_id])
                or int(model.jnt_type[joint_id])
                not in (int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE))
            ):
                continue
            qpos_address = int(model.jnt_qposadr[joint_id])
            value = float(sim.data.qpos[qpos_address])
            lower, upper = (float(v) for v in model.jnt_range[joint_id])
            joint_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_JOINT, int(joint_id)
            )
            joint_kind = joint_type_name(
                int(model.jnt_type[joint_id]),
                int(mujoco.mjtJoint.mjJNT_HINGE),
                int(mujoco.mjtJoint.mjJNT_SLIDE),
            )
            measurement = evaluate_joint_limit(
                joint_name=str(joint_name),
                joint_type=joint_kind,
                qpos=value,
                lower=lower,
                upper=upper,
                policy=policy["joint_limit_tolerance_policy"],
            )
            if not measurement["strict_criterion_pass"]:
                item["strict_joint_limit_violations"].append({
                    "joint_id": int(joint_id),
                    **measurement,
                })
            if not measurement["operational_pass"]:
                item["status"] = "unsafe"
                item["joint_limit_violations"].append({
                    "joint_id": int(joint_id),
                    **measurement,
                })
        if item["minimum_clearance_m"] is None and policy["forbidden_objects"]:
            item["status"] = "incomplete"
    except Exception as exc:
        item.update({"status": "incomplete", "error": f"{type(exc).__name__}: {exc}"})
    monitor["geometry_substeps"].append(item)
    return item


def _safety_geometry_at_current_state(raw_policy):
    """Read current full-Geom margin and contact policy without advancing physics."""
    policy = _normalize_safety_geometry_policy(raw_policy)
    monitor = {"safety_policy": policy, "geometry_substeps": []}
    result = _substep_safety_geometry(monitor, _collect_contacts())
    result["sampling_point"] = "current_live_MuJoCo_state"
    result["minimum_forbidden_clearance_m"] = result.get("minimum_clearance_m")
    result["policy_source"] = policy["source"]
    return result


def _new_contact_monitor(safety_policy=None):
    normalized_policy = _normalize_safety_geometry_policy(safety_policy) if safety_policy is not None else None
    robot_body_ids = None
    robot_classification_error = None
    try:
        robot_body_ids = _robot_body_ids()
    except Exception as exc:
        robot_classification_error = f"{type(exc).__name__}: {exc}"

    expected_substeps = None
    expected_error = None
    try:
        task_env = env.env
        control_timestep = float(task_env.control_timestep)
        model_timestep = float(task_env.model_timestep)
        expected_substeps = int(control_timestep / model_timestep)
        if expected_substeps <= 0:
            raise ValueError(f"invalid substep count {expected_substeps}")
    except Exception as exc:
        expected_error = f"{type(exc).__name__}: {exc}"

    return {
        "expected_substeps_per_control_step": expected_substeps,
        "expected_substeps_error": expected_error,
        "robot_body_ids": robot_body_ids,
        "robot_classification_status": (
            "available" if robot_body_ids is not None else "unavailable"
        ),
        "robot_classification_error": robot_classification_error,
        "captured_physics_substeps": 0,
        "contact_substeps": 0,
        "no_contact_substeps": 0,
        "unavailable_contact_substeps": 0,
        "control_steps": [],
        "geometry_substeps": [],
        "safety_policy": normalized_policy,
        "_events": {},
    }


def _contact_pair_classification(body1_id, body2_id, robot_body_ids):
    if robot_body_ids is None:
        return "unavailable", None
    body1_robot = body1_id in robot_body_ids
    body2_robot = body2_id in robot_body_ids
    if body1_robot and body2_robot:
        return "robot_robot", True
    if body1_robot or body2_robot:
        return "robot_scene", True
    return "scene_scene", False


def _record_substep_contacts(monitor, control_step_index, physics_substep_index,
                             candidate_substep_index, contact_result):
    status = contact_result.get("status", "unavailable")
    if status not in ("contact", "no_contact"):
        status = "unavailable"

    # Count each geom pair once per physics substep, even when the solver
    # reports several contact points for the same pair.
    pairs_in_substep = {}
    malformed_contact = False
    for contact in contact_result.get("contacts", []):
        try:
            geom1 = contact["geom1"]
            geom2 = contact["geom2"]
            body1 = contact["body1"]
            body2 = contact["body2"]
            geom1_id = int(geom1["id"])
            geom2_id = int(geom2["id"])
            if geom1_id <= geom2_id:
                pair_key = (geom1_id, geom2_id)
                ordered_geoms = (geom1, geom2)
                ordered_bodies = (body1, body2)
            else:
                pair_key = (geom2_id, geom1_id)
                ordered_geoms = (geom2, geom1)
                ordered_bodies = (body2, body1)
            pair = pairs_in_substep.setdefault(
                pair_key,
                {
                    "geoms": ordered_geoms,
                    "bodies": ordered_bodies,
                    "contact_point_count": 0,
                    "force_norms": [],
                    "force_unavailable_count": 0,
                    "first_contact_sample": None,
                },
            )
            pair["contact_point_count"] += 1
            if pair["first_contact_sample"] is None:
                position = contact.get("position")
                normal = contact.get("normal_geom1_to_geom2")
                normal_status = contact.get("normal_status", "unavailable")
                try:
                    position_array = np.asarray(position, dtype=np.float64)
                    if position_array.shape != (3,) or not np.isfinite(position_array).all():
                        raise ValueError("contact position is malformed")
                    pair["first_contact_sample"] = {
                        "position": position_array.tolist(),
                        "position_status": "available",
                        "normal": normal,
                        "normal_status": normal_status,
                        "normal_geom1_to_geom2": [geom1_id, geom2_id],
                    }
                except Exception:
                    pair["first_contact_sample"] = {
                        "position": None,
                        "position_status": "unavailable",
                        "normal": normal,
                        "normal_status": normal_status,
                        "normal_geom1_to_geom2": [geom1_id, geom2_id],
                    }
            if contact.get("force_status") == "available":
                force = np.asarray(contact.get("force_and_torque"), dtype=np.float64)
                if force.shape == (6,) and np.isfinite(force[:3]).all():
                    pair["force_norms"].append(float(np.linalg.norm(force[:3])))
                else:
                    pair["force_unavailable_count"] += 1
            else:
                pair["force_unavailable_count"] += 1
        except Exception:
            # The contact reader normally returns complete records or marks the
            # sample unavailable. Keep the coverage status conservative if a
            # malformed item reaches this aggregation layer.
            malformed_contact = True

    if malformed_contact:
        status = "unavailable"
    if status == "contact":
        monitor["contact_substeps"] += 1
    elif status == "no_contact":
        monitor["no_contact_substeps"] += 1
    else:
        monitor["unavailable_contact_substeps"] += 1

    for pair_key, pair in pairs_in_substep.items():
        event = monitor["_events"].get(pair_key)
        if event is None:
            body1_id = int(pair["bodies"][0]["id"])
            body2_id = int(pair["bodies"][1]["id"])
            classification, robot_involved = _contact_pair_classification(
                body1_id, body2_id, monitor["robot_body_ids"]
            )
            event = {
                "geom_pair": [dict(item) for item in pair["geoms"]],
                "body_pair": [dict(item) for item in pair["bodies"]],
                "classification": classification,
                "robot_involved": robot_involved,
                "first_control_step": int(control_step_index),
                "first_physics_substep": int(physics_substep_index),
                "first_candidate_physics_substep": int(candidate_substep_index),
                "occurrence_count": 0,
                "contact_point_count": 0,
                "max_force_norm": None,
                "force_sample_count": 0,
                "force_unavailable_count": 0,
                "first_contact_position_world": None,
                "contact_position_status": "unavailable",
                "first_contact_normal_geom1_to_geom2_world": None,
                "contact_normal_status": "unavailable",
                "contact_normal_geom_ids": None,
                "first_contact_sample_recorded": False,
            }
            monitor["_events"][pair_key] = event
        if not event["first_contact_sample_recorded"] and pair["first_contact_sample"] is not None:
            sample = pair["first_contact_sample"]
            event["first_contact_position_world"] = sample["position"]
            event["contact_position_status"] = sample["position_status"]
            event["first_contact_normal_geom1_to_geom2_world"] = sample["normal"]
            event["contact_normal_status"] = sample["normal_status"]
            event["contact_normal_geom_ids"] = sample["normal_geom1_to_geom2"]
            event["first_contact_sample_recorded"] = True
        event["occurrence_count"] += 1
        event["contact_point_count"] += pair["contact_point_count"]
        event["force_unavailable_count"] += pair["force_unavailable_count"]
        for force_norm in pair["force_norms"]:
            event["force_sample_count"] += 1
            if event["max_force_norm"] is None or force_norm > event["max_force_norm"]:
                event["max_force_norm"] = force_norm

    return status


def _finish_contact_control_step(monitor, control_step):
    expected = control_step["expected_physics_substeps"]
    captured = control_step["captured_physics_substeps"]
    observed = control_step["contact_samples"]
    if control_step.get("capture_error"):
        status = "unavailable" if captured == 0 else "incomplete"
    elif expected is not None and captured == expected and observed["unavailable"] == 0:
        status = "complete"
    else:
        status = "incomplete"
    control_step["status"] = status
    control_step["unobserved_expected_substeps"] = (
        max(0, expected - captured) if expected is not None else None
    )
    monitor["control_steps"].append(control_step)
    return control_step


def _run_env_step_with_contact_monitor(action, action_index, monitor):
    """Observe contacts after each native MjSim.step during one env.step call."""
    sim = env.sim
    expected = monitor["expected_substeps_per_control_step"]
    control_step = {
        "control_step_index": int(action_index),
        "status": "unavailable",
        "expected_physics_substeps": expected,
        "captured_physics_substeps": 0,
        "contact_samples": {"contact": 0, "no_contact": 0, "unavailable": 0},
        "capture_error": None,
    }
    candidate_substep_start = monitor["captured_physics_substeps"]
    prior_step = sim.__dict__.get("step", None)
    had_instance_step = "step" in sim.__dict__
    original_step = sim.step
    physics_substep_index = 0
    observer_installed = False

    def monitored_step(*args, **kwargs):
        nonlocal physics_substep_index
        result = original_step(*args, **kwargs)
        substep_index = physics_substep_index
        physics_substep_index += 1
        monitor["captured_physics_substeps"] += 1
        control_step["captured_physics_substeps"] += 1
        prior_sample_counts = (
            monitor["contact_substeps"],
            monitor["no_contact_substeps"],
            monitor["unavailable_contact_substeps"],
        )
        try:
            contacts = _collect_contacts()
            sample_status = _record_substep_contacts(
                monitor,
                action_index,
                substep_index,
                candidate_substep_start + substep_index,
                contacts,
            )
        except Exception as exc:
            sample_status = "unavailable"
            (monitor["contact_substeps"],
             monitor["no_contact_substeps"],
             monitor["unavailable_contact_substeps"]) = prior_sample_counts
            monitor["unavailable_contact_substeps"] += 1
            control_step["capture_error"] = (
                f"{type(exc).__name__}: {exc}"
            )
            contacts = {"status": "unavailable", "contacts": []}
        geometry = _substep_safety_geometry(monitor, contacts)
        if geometry is not None and geometry.get("status") != "complete":
            control_step["capture_error"] = (
                control_step.get("capture_error")
                or "substep_geometry_safety_" + str(geometry.get("status"))
            )
        control_step["contact_samples"][sample_status] += 1
        return result

    try:
        sim.step = monitored_step
        observer_installed = True
    except Exception as exc:
        control_step["capture_error"] = (
            f"Cannot install per-instance MjSim.step observer: {type(exc).__name__}: {exc}"
        )

    try:
        return env.step(action)
    finally:
        if observer_installed:
            try:
                if had_instance_step:
                    sim.step = prior_step
                else:
                    del sim.__dict__["step"]
            except Exception as exc:
                control_step["capture_error"] = (
                    f"Cannot restore MjSim.step observer: {type(exc).__name__}: {exc}"
                )
        else:
            # A failed assignment should not have replaced the class method;
            # restore the prior instance lookup state defensively.
            if had_instance_step:
                sim.__dict__["step"] = prior_step
            else:
                sim.__dict__.pop("step", None)
        monitor["_last_control_step"] = _finish_contact_control_step(
            monitor, control_step
        )


def _run_joint_trajectory_step_with_contact_monitor(request, action_index, monitor):
    """Execute one bounded 7-DoF torque-controlled MuJoCo trajectory sample."""
    global obs, last_reward, last_done

    sim = env.sim
    task_env = env.env
    if len(env.robots) != 1:
        raise ValueError("joint_trajectory_control_requires_one_robot")
    robot = env.robots[0]
    model, data = sim.model._model, sim.data._data
    q_target = np.asarray(request.get("joint_target_qpos_rad"), dtype=np.float64)
    if q_target.shape != (7,) or not np.isfinite(q_target).all():
        raise ValueError("joint_target_qpos_rad_must_be_finite_7D")
    qidx = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    didx = np.asarray(robot._ref_joint_vel_indexes, dtype=np.int64)
    actuator_ids = np.asarray(robot._ref_joint_actuator_indexes, dtype=np.int64)
    if qidx.shape != (7,) or didx.shape != (7,) or actuator_ids.shape != (7,):
        raise ValueError("active_robot_arm_joint_or_actuator_mapping_is_not_7D")
    q_start = np.asarray(data.qpos[qidx], dtype=np.float64).copy()
    q_delta = q_target - q_start
    control_dt = float(task_env.control_timestep)
    model_dt = float(task_env.model_timestep)
    substeps = int(round(control_dt / model_dt))
    if substeps != 25 or not math.isclose(control_dt, substeps * model_dt, abs_tol=1e-10):
        raise ValueError("joint_trajectory_control_requires_25_MuJoCo_substeps")
    lower = np.asarray(model.jnt_range[[
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in robot.robot_model.joints
    ], 0], dtype=np.float64)
    upper = np.asarray(model.jnt_range[[
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name))
        for name in robot.robot_model.joints
    ], 1], dtype=np.float64)
    if np.any(q_target < lower) or np.any(q_target > upper):
        raise ValueError("joint_target_exceeds_compiled_MuJoCo_position_limits")
    gripper_action = request.get("gripper_action")
    if gripper_action is not None:
        if isinstance(gripper_action, bool) or not isinstance(gripper_action, (int, float)):
            raise ValueError("gripper_action_must_be_a_finite_scalar")
        if not np.isfinite(float(gripper_action)) or not -1.0 <= float(gripper_action) <= 1.0:
            raise ValueError("gripper_action_must_be_within_normalized_range")
        robot.grip_action(robot.gripper, np.asarray([gripper_action], dtype=np.float64))

    model_mass = np.empty((model.nv, model.nv), dtype=np.float64)
    torque_bounds = np.asarray(model.actuator_ctrlrange[actuator_ids], dtype=np.float64)
    desired_velocity = q_delta / control_dt
    kp = float(request.get("joint_position_kp", 50.0))
    if not np.isfinite(kp) or not 5.0 <= kp <= 100.0:
        raise ValueError("joint_position_kp_must_be_between_5_and_100")
    control_step = {
        "control_step_index": int(action_index),
        "status": "incomplete",
        "expected_physics_substeps": substeps,
        "captured_physics_substeps": 0,
        "contact_samples": {"contact": 0, "no_contact": 0, "unavailable": 0},
        "capture_error": None,
    }
    candidate_substep_start = monitor["captured_physics_substeps"]
    substep_contacts = []
    applied_joint_torques = []
    prior_step = sim.__dict__.get("step", None)
    had_instance_step = "step" in sim.__dict__
    original_step = sim.step
    step_index = 0

    def monitored_step(*args, **kwargs):
        nonlocal step_index
        result = original_step(*args, **kwargs)
        contacts = _collect_contacts()
        sample_status = _record_substep_contacts(
            monitor,
            action_index,
            step_index,
            candidate_substep_start + step_index,
            contacts,
        )
        geometry = _substep_safety_geometry(monitor, contacts)
        if geometry is not None and geometry.get("status") != "complete":
            control_step["capture_error"] = "substep_geometry_monitoring_incomplete"
        control_step["contact_samples"][sample_status] += 1
        control_step["captured_physics_substeps"] += 1
        monitor["captured_physics_substeps"] += 1
        substep_contacts.append(contacts)
        step_index += 1
        return result

    task_env._set_mocap_motion()
    task_env.timestep += 1
    try:
        sim.step = monitored_step
        for substep in range(substeps):
            sim.forward()
            alpha = (substep + 1) / substeps
            q_reference = q_start + alpha * q_delta
            q_velocity = np.asarray(data.qvel[didx], dtype=np.float64)
            q_acceleration = kp * (q_reference - data.qpos[qidx]) + (
                2.0 * math.sqrt(kp) * (desired_velocity - q_velocity)
            )
            mujoco.mj_fullM(model, model_mass, data.qM)
            torque = (
                model_mass[np.ix_(didx, didx)] @ q_acceleration
                + np.asarray(data.qfrc_bias[didx], dtype=np.float64)
            )
            if not np.isfinite(torque).all():
                raise FloatingPointError("joint_torque_command_is_nonfinite")
            # Keep motor saturation at the compiled actuator bounds.
            data.ctrl[actuator_ids] = np.clip(
                torque, torque_bounds[:, 0], torque_bounds[:, 1]
            )
            applied_joint_torques.append(np.asarray(data.ctrl[actuator_ids], dtype=np.float64).tolist())
            sim.step()
            task_env._update_observables()
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                raise FloatingPointError("MuJoCo_state_became_nonfinite_during_joint_control")
            if np.any(data.qpos[qidx] < lower - 1e-7) or np.any(data.qpos[qidx] > upper + 1e-7):
                raise RuntimeError("MuJoCo_arm_joint_crossed_compiled_position_limit")
        task_env.cur_time += control_dt
        reward, done, info = task_env._post_action(None)
        done = bool(task_env._check_success())
        info["cost"] = task_env._check_constraint(done)
        last_reward, last_done = reward, done
        obs = task_env._get_observations()
    finally:
        if had_instance_step:
            sim.step = prior_step
        else:
            sim.__dict__.pop("step", None)
        control_step["status"] = (
            "complete"
            if control_step["captured_physics_substeps"] == substeps
            else "incomplete"
        )
        monitor["_last_control_step"] = _finish_contact_control_step(monitor, control_step)

    info["applied_joint_torques_nm_per_substep"] = applied_joint_torques
    return obs, reward, done, info, substep_contacts


def _contact_monitor_result(monitor):
    control_steps = monitor["control_steps"]
    expected_counts = [
        step["expected_physics_substeps"] for step in control_steps
    ]
    expected_total = (
        sum(expected_counts)
        if expected_counts and all(count is not None for count in expected_counts)
        else None
    )
    unobserved_counts = [
        step["unobserved_expected_substeps"] for step in control_steps
    ]
    unobserved_total = (
        sum(unobserved_counts)
        if unobserved_counts and all(count is not None for count in unobserved_counts)
        else None
    )
    if control_steps and all(step["status"] == "complete" for step in control_steps):
        status = "complete"
    elif not control_steps or all(step["status"] == "unavailable" for step in control_steps):
        status = "unavailable"
    else:
        status = "incomplete"
    events = [monitor["_events"][key] for key in sorted(monitor["_events"])]
    return {
        "status": status,
        "sampling_point": "immediately_after_each_successful_MjSim.step",
        "expected_physics_substeps_per_control_step": monitor[
            "expected_substeps_per_control_step"
        ],
        "expected_physics_substeps_total": expected_total,
        "unobserved_expected_substeps": unobserved_total,
        "expected_substeps_error": monitor["expected_substeps_error"],
        "captured_physics_substeps": monitor["captured_physics_substeps"],
        "contact_substeps": monitor["contact_substeps"],
        "no_contact_substeps": monitor["no_contact_substeps"],
        "unavailable_contact_substeps": monitor["unavailable_contact_substeps"],
        "robot_classification_status": monitor["robot_classification_status"],
        "robot_classification_error": monitor["robot_classification_error"],
        "control_steps": control_steps,
        "geometry_monitoring": {
            "status": (
                "unavailable"
                if monitor.get("safety_policy") is None
                else "complete"
                if len(monitor.get("geometry_substeps", [])) == monitor["captured_physics_substeps"]
                and all(item.get("status") == "complete" for item in monitor.get("geometry_substeps", []))
                else "unsafe"
                if any(item.get("status") == "unsafe" for item in monitor.get("geometry_substeps", []))
                else "incomplete"
            ),
            "minimum_clearance_m": min(
                (item["minimum_clearance_m"] for item in monitor.get("geometry_substeps", [])
                 if item.get("minimum_clearance_m") is not None),
                default=None,
            ),
            "substeps": monitor.get("geometry_substeps", []),
        },
        "events": events,
        "interpretation": "Observed physical contacts only; no safety verdict is assigned.",
    }


def _normalize_branch_candidates(raw_candidates):
    if not isinstance(raw_candidates, (list, tuple)):
        raise ValueError("candidates must be a list of candidate action sequences")

    normalized = []
    seen_ids = set()
    for candidate_index, candidate in enumerate(raw_candidates):
        if isinstance(candidate, dict):
            candidate_id = candidate.get("candidate_id", f"candidate-{candidate_index}")
            raw_actions = candidate.get("actions")
        else:
            candidate_id = f"candidate-{candidate_index}"
            raw_actions = candidate

        if isinstance(candidate_id, bool) or not isinstance(candidate_id, (str, int)):
            raise ValueError(
                f"candidate {candidate_index} candidate_id must be a string or integer"
            )
        id_key = (type(candidate_id).__name__, str(candidate_id))
        if id_key in seen_ids:
            raise ValueError(f"Duplicate candidate_id: {candidate_id!r}")
        seen_ids.add(id_key)

        try:
            actions = np.asarray(raw_actions, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"candidate {candidate_id!r} actions must be a numeric array"
            ) from exc
        if actions.ndim != 2 or actions.shape[1:] != (7,) or actions.shape[0] == 0:
            raise ValueError(
                f"candidate {candidate_id!r} actions must have shape (steps, 7) "
                "with at least one step"
            )
        if not np.isfinite(actions).all():
            raise ValueError(f"candidate {candidate_id!r} contains a non-finite action")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "actions": [action.copy() for action in actions],
            }
        )
    return normalized


def _branch_rollout(snapshot_id, raw_candidates, safety_policy=None,
                    capture_runtime_snapshots=False):
    global obs, last_reward, last_done

    if not isinstance(snapshot_id, str) or snapshot_id not in snapshots:
        raise ValueError(f"Unknown snapshot_id: {snapshot_id!r}")
    candidates = _normalize_branch_candidates(raw_candidates)
    target_snapshot = snapshots[snapshot_id]

    # Capture the caller's live state before touching the simulator. Every
    # candidate restores target_snapshot, and this finally block restores the
    # state that was active when branch_rollout was requested.
    main_snapshot = _capture_snapshot()
    main_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
    result = {
        "snapshot_id": snapshot_id,
        "candidates": [],
        "main_state_restored": False,
    }
    try:
        for candidate in candidates:
            candidate_result = {
                "candidate_id": candidate["candidate_id"],
                "status": "ok",
                "requested_steps": len(candidate["actions"]),
                "executed_steps": 0,
                "steps": [],
            }
            contact_monitor = _new_contact_monitor(safety_policy)
            try:
                _restore_snapshot(target_snapshot)
                candidate_result["start_state"] = _snapshot_diagnostics()
                for action_index, action in enumerate(candidate["actions"]):
                    obs, reward, done, _info = _run_env_step_with_contact_monitor(
                        action, action_index, contact_monitor
                    )
                    last_reward = reward
                    last_done = done
                    state = _snapshot_diagnostics()
                    contacts = _collect_contacts()
                    step_record = {
                            "step_index": action_index,
                            "action": action.tolist(),
                            "reward": float(reward),
                            "done": bool(done),
                            "state": state,
                            "contacts": contacts,
                            "observation": pack(obs, reward, done),
                            "safety_geometry": contact_monitor["geometry_substeps"][-1]
                            if contact_monitor.get("geometry_substeps")
                            else None,
                            "substep_contact_coverage": contact_monitor[
                                "control_steps"
                            ][-1],
                        }
                    if capture_runtime_snapshots:
                        step_record["terminal_runtime_snapshot"] = _export_runtime_snapshot()["payload"]
                    candidate_result["steps"].append(step_record)
                    candidate_result["executed_steps"] += 1
                    if done:
                        break
            except Exception as exc:
                candidate_result["status"] = "error"
                candidate_result["error"] = f"{type(exc).__name__}: {exc}"
            finally:
                candidate_result["contact_monitoring"] = _contact_monitor_result(
                    contact_monitor
                )
            result["candidates"].append(candidate_result)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _restore_snapshot(main_snapshot)
            restored_snapshot = _capture_snapshot()
            restored_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
            result["main_state_restored"] = bool(
                _tracking_value_equal(main_snapshot, restored_snapshot)
                and _tracking_value_equal(main_diagnostic, restored_diagnostic)
            )
            result["main_restore_verified"] = result["main_state_restored"]
            if not result["main_state_restored"]:
                result["main_restore_error"] = "caller_snapshot_or_diagnostic_mismatch"
        except Exception as exc:
            result["main_state_restored"] = False
            result["main_restore_verified"] = False
            result["main_restore_error"] = f"{type(exc).__name__}: {exc}"

    return result


def _osc_prefix_probe(request):
    """Preview a feedback OSC_POSE prefix in an isolated, fully monitored branch."""
    from research.gs_opd.repair.trajectory_tracking import (
        quaternion_conjugate,
        quaternion_multiply,
        quaternion_to_rotation_vector,
        world_pose_tracking_action,
    )

    snapshot_id = request.get("snapshot_id")
    if not isinstance(snapshot_id, str) or snapshot_id not in snapshots:
        raise ValueError("unknown_or_expired_snapshot_id")
    target_snapshot = snapshots[snapshot_id]
    path_qpos = np.asarray(request.get("jointspace_path_qpos"), dtype=np.float64)
    path_poses = np.asarray(request.get("eef_reference_poses_world"), dtype=np.float64)
    native_action_prefix_raw = request.get("native_action_prefix")
    native_action_prefix = None
    if native_action_prefix_raw is not None:
        native_action_prefix = np.asarray(native_action_prefix_raw, dtype=np.float64)
    start_index = request.get("path_start_index")
    max_steps = request.get("max_control_steps")
    gripper_action = request.get("gripper_action")
    risk_trigger = request.get("risk_trigger_distance_m")
    risk_rearm = request.get("risk_rearm_distance_m")
    risk_crossing_armed = request.get("risk_crossing_armed")
    max_joint_error = request.get("maximum_joint_path_deviation_rad")
    max_eef_error = request.get("maximum_eef_tracking_error_m")
    max_orientation_error = request.get("maximum_eef_orientation_error_rad")
    qpos_reference_mode = request.get("qpos_reference_mode", "enforced")
    if qpos_reference_mode not in {"enforced", "diagnostic_only"}:
        raise ValueError("qpos_reference_mode_must_be_enforced_or_diagnostic_only")
    if (
        path_qpos.ndim != 2
        or path_qpos.shape[1:] != (7,)
        or len(path_qpos) < 2
        or not np.isfinite(path_qpos).all()
        or path_poses.shape != (len(path_qpos), 7)
        or not np.isfinite(path_poses).all()
        or np.any(np.abs(np.linalg.norm(path_poses[:, 3:], axis=1) - 1.0) > 1e-5)
    ):
        raise ValueError("jointspace_path_or_eef_reference_poses_are_malformed")
    if (
        isinstance(start_index, bool)
        or not isinstance(start_index, int)
        or not 0 <= start_index < len(path_qpos) - 1
        or isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or not 1 <= max_steps <= 1200
    ):
        raise ValueError("path_start_index_or_max_control_steps_are_invalid")
    if native_action_prefix is not None and (
        qpos_reference_mode != "diagnostic_only"
        or native_action_prefix.ndim != 2
        or native_action_prefix.shape[1:] != (7,)
        or len(native_action_prefix) < 1
        or len(native_action_prefix) > max_steps
        or not np.isfinite(native_action_prefix).all()
    ):
        raise ValueError("native_action_prefix_requires_finite_N_by_7_actions_and_diagnostic_qpos_mode")
    if (
        isinstance(gripper_action, bool)
        or not isinstance(gripper_action, (int, float))
        or not np.isfinite(float(gripper_action))
    ):
        raise ValueError("native_7d_gripper_action_must_be_explicit_and_finite")
    for name, value in (
        ("risk_trigger_distance_m", risk_trigger),
        ("risk_rearm_distance_m", risk_rearm),
        ("maximum_joint_path_deviation_rad", max_joint_error),
        ("maximum_eef_tracking_error_m", max_eef_error),
        ("maximum_eef_orientation_error_rad", max_orientation_error),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"{name}_must_be_finite_and_positive")
    if float(risk_rearm) <= float(risk_trigger):
        raise ValueError("risk_rearm_distance_m_must_exceed_entry_distance")
    if not isinstance(risk_crossing_armed, bool):
        raise ValueError("risk_crossing_armed_must_be_an_explicit_boolean")
    policy = _normalize_safety_geometry_policy(request.get("safety_geometry_policy"))
    expected_state_hash = request.get("current_state_sha256")
    live_state = _mujoco_integration_state(env.sim)
    live_state_hash = hashlib.sha256(live_state.tobytes()).hexdigest()
    if expected_state_hash != live_state_hash:
        raise ValueError("active_environment_state_changed_before_snapshot_prefix_probe")
    if not _tracking_value_equal(_capture_snapshot(), target_snapshot):
        raise ValueError("active_main_state_does_not_match_prefix_probe_snapshot")
    model, sim = env.sim.model._model, env.sim
    robot = env.robots[0]
    qpos_addresses = np.asarray(robot._ref_joint_pos_indexes, dtype=np.int64)
    if len(qpos_addresses) != 7:
        raise ValueError("active_robot_arm_qpos_is_not_7d")
    cursor_qpos_error = float(np.linalg.norm(
        np.asarray(sim.data.qpos[qpos_addresses]) - path_qpos[start_index]
    ))
    if qpos_reference_mode == "enforced" and cursor_qpos_error > float(max_joint_error):
        raise ValueError("path_cursor_does_not_match_current_real_qpos_within_limit")
    if getattr(robot.controller, "name", type(robot.controller).__name__) != "OSC_POSE":
        raise ValueError("active_robot_controller_is_not_native_OSC_POSE")
    if int(getattr(robot, "action_dim", 0)) != 7:
        raise ValueError("active_robot_action_space_is_not_native_7d")

    scratch = mujoco.MjData(model)
    integration_spec = mujoco.mjtState.mjSTATE_INTEGRATION
    mujoco.mj_setState(model, scratch, live_state, integration_spec)
    q_path_fk_errors = []
    if native_action_prefix is None:
        for index in range(start_index, min(len(path_qpos), start_index + max_steps + 1)):
            scratch.qpos[qpos_addresses] = path_qpos[index]
            mujoco.mj_forward(model, scratch)
            quaternion = np.empty(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quaternion, np.asarray(scratch.site_xmat[robot.eef_site_id]).reshape(9))
            quaternion /= np.linalg.norm(quaternion)
            position_error = float(np.linalg.norm(scratch.site_xpos[robot.eef_site_id] - path_poses[index, :3]))
            orientation_error = float(np.linalg.norm(quaternion_to_rotation_vector(
                quaternion_multiply(path_poses[index, 3:], quaternion_conjugate(quaternion))
            )))
            q_path_fk_errors.append(max(position_error, orientation_error))
        if max(q_path_fk_errors, default=math.inf) > 1e-6:
            raise ValueError("RRT_EEF_reference_does_not_match_current_compiled_model_FK")

    caller_snapshot = _capture_snapshot()
    caller_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
    result = {
        "operation": "osc_prefix_probe",
        "status": "incomplete",
        "complete": False,
        "safe": False,
        "snapshot_id": snapshot_id,
        "current_state_sha256": live_state_hash,
        "path_start_index": start_index,
        "requested_steps": min(
            max_steps,
            len(path_qpos) - 1 - start_index,
            len(native_action_prefix) if native_action_prefix is not None else max_steps,
        ),
        "executed_steps": 0,
        "qpos_reference_mode": qpos_reference_mode,
        "action_source": (
            "explicit_native_7d_action_prefix"
            if native_action_prefix is not None else "world_pose_path_feedback"
        ),
        "qpos_cursor_reference_error_rad": cursor_qpos_error,
        "maximum_joint_path_deviation_rad": float(max_joint_error),
        "actions": [],
        "steps": [],
        "main_state_restored_verified": False,
        "stop_reason": None,
    }
    initial_clearance = request.get("current_minimum_forbidden_clearance_m")
    if initial_clearance is None or not np.isfinite(float(initial_clearance)):
        raise ValueError("current_forbidden_object_clearance_is_unknown")
    risk_crossing_pending = bool(risk_crossing_armed)
    try:
        _restore_snapshot(target_snapshot)
        if not _tracking_value_equal(target_snapshot, _capture_snapshot()):
            raise RuntimeError("Snapshot_restore_did_not_reproduce_prefix_start_state")
        fields = _repair_context()["robots"][0]["fields"]
        monitor = _new_contact_monitor(request.get("safety_geometry_policy"))
        action_low, action_high = _active_environment_action_bounds(env, robot)
        movement_history = []
        goal_position = path_poses[-1, :3]
        for local_index in range(result["requested_steps"]):
            path_index = start_index + local_index + 1
            before = _snapshot_diagnostics(canonicalize_kinematics=True)["robot_states"][0]
            target_pose = path_poses[path_index]
            action = (
                native_action_prefix[local_index]
                if native_action_prefix is not None
                else world_pose_tracking_action(
                    before["eef_pos"],
                    before["eef_quat"],
                    target_pose[:3],
                    target_pose[3:],
                    fields,
                    gripper_action=float(gripper_action),
                )
            )
            action = _validate_tracking_action(action, action_low, action_high).astype(np.float32)
            obs, reward, done, _info = _run_env_step_with_contact_monitor(action, local_index, monitor)
            last_reward = reward
            last_done = done
            post = _snapshot_diagnostics(canonicalize_kinematics=True)
            robot_state = post["robot_states"][0]
            qpos = np.asarray(robot_state["joint_qpos"], dtype=np.float64)
            eef_position = np.asarray(robot_state["eef_pos"], dtype=np.float64)
            eef_quaternion = np.asarray(robot_state["eef_quat"], dtype=np.float64)
            q_error = float(np.linalg.norm(qpos - path_qpos[path_index]))
            eef_error = float(np.linalg.norm(eef_position - target_pose[:3]))
            eef_rotation_error = float(np.linalg.norm(quaternion_to_rotation_vector(
                quaternion_multiply(target_pose[3:], quaternion_conjugate(eef_quaternion))
            )))
            geom_samples = monitor["geometry_substeps"][-monitor["expected_substeps_per_control_step"]:]
            geom_status = (
                "complete"
                if len(geom_samples) == monitor["expected_substeps_per_control_step"]
                and all(sample.get("status") == "complete" for sample in geom_samples)
                else "unsafe"
                if any(sample.get("status") == "unsafe" for sample in geom_samples)
                else "incomplete"
            )
            if geom_status != "complete":
                result["stop_reason"] = "substep_geometry_safety_" + geom_status
            observation = pack(obs, reward, done, include_snapshot_state=True)
            step = {
                "step_index": local_index,
                "path_index": path_index,
                "action": action.tolist(),
                "reference_qpos": path_qpos[path_index].tolist(),
                "reference_eef_pose": target_pose.tolist(),
                "actual_qpos": qpos.tolist(),
                "actual_eef_position_world_m": eef_position.tolist(),
                "actual_eef_quaternion_wxyz": eef_quaternion.tolist(),
                "joint_path_deviation_rad": q_error,
                "joint_path_deviation_exceeds_frozen_reference_limit": q_error > float(max_joint_error),
                "qpos_reference_mode": qpos_reference_mode,
                "eef_tracking_error_m": eef_error,
                "eef_orientation_error_rad": eef_rotation_error,
                "reward": float(reward),
                "done": bool(done),
                "observation": observation,
                "contacts": _collect_contacts(),
                "safety_geometry_substeps": geom_samples,
                "terminal_runtime_snapshot": _export_runtime_snapshot()["payload"],
            }
            result["steps"].append(step)
            result["actions"].append(action.tolist())
            result["executed_steps"] += 1
            current_margin = min(
                (sample.get("minimum_clearance_m") for sample in geom_samples
                 if sample.get("minimum_clearance_m") is not None),
                default=None,
            )
            if done:
                result["stop_reason"] = "environment_terminated_during_isolated_prefix"
                break
            if geom_status != "complete":
                break
            if native_action_prefix is None and (
                (qpos_reference_mode == "enforced" and q_error > float(max_joint_error))
                or eef_error > float(max_eef_error)
                or eef_rotation_error > float(max_orientation_error)
            ):
                result["stop_reason"] = "planned_joint_or_eef_reference_deviation"
                break
            if current_margin is not None:
                if not risk_crossing_pending and current_margin >= float(risk_rearm):
                    risk_crossing_pending = True
                if risk_crossing_pending and current_margin <= float(risk_trigger):
                    risk_crossing_pending = False
                    result["stop_reason"] = "narrow_channel_entry_replan"
                    break
            movement_history.append(float(np.linalg.norm(eef_position - np.asarray(before["eef_pos"], dtype=np.float64))))
            if len(movement_history) >= 12 and sum(movement_history[-12:]) < 0.002:
                result["stop_reason"] = "motion_stagnation"
                break
        result["contact_monitoring"] = _contact_monitor_result(monitor)
        result["status"] = "complete"
        result["complete"] = result["executed_steps"] > 0
        result["safe"] = bool(
            result["complete"]
            and result["contact_monitoring"].get("status") == "complete"
            and result["contact_monitoring"].get("geometry_monitoring", {}).get("status") == "complete"
            and all(not step["done"] for step in result["steps"])
        )
        result["last_path_index"] = result["steps"][-1]["path_index"] if result["steps"] else start_index
    except Exception as exc:
        result["status"] = "incomplete"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _restore_snapshot(caller_snapshot)
            restored_snapshot = _capture_snapshot()
            restored_diagnostic = _snapshot_diagnostics(canonicalize_kinematics=True)
            result["main_state_restored_verified"] = bool(
                _tracking_value_equal(caller_snapshot, restored_snapshot)
                and _tracking_value_equal(caller_diagnostic, restored_diagnostic)
            )
            if not result["main_state_restored_verified"]:
                result["main_restore_error"] = "caller_state_or_diagnostic_mismatch"
        except Exception as exc:
            result["main_state_restored_verified"] = False
            result["main_restore_error"] = f"{type(exc).__name__}: {exc}"
    if not result["main_state_restored_verified"]:
        result["status"] = "incomplete"
        result["complete"] = False
        result["safe"] = False
    return result


def pack(obs, reward=0.0, done=False, include_snapshot_state=False):
    images = {}
    for k in ("agentview_image", "robot0_eye_in_hand_image"):
        if k in obs:
            arr = np.ascontiguousarray(obs[k])
            images[k] = {
                "shape": tuple(arr.shape),
                "dtype": str(arr.dtype),
                "data": arr.tobytes(),
            }
    state = {}
    for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
        if k in obs:
            state[k] = np.asarray(obs[k]).tolist()
    try:
        native_task_success = bool(env.env._check_success())
    except Exception:
        native_task_success = None
    result = {
        "images": images,
        "state": state,
        "reward": float(reward),
        "done": bool(done),
        "task_success": native_task_success,
        "timestep": int(env.env.timestep),
        "instruction": task.language,
        "scene_metadata": scene_metadata,
    }
    if include_snapshot_state:
        result["snapshot_state"] = _snapshot_diagnostics(canonicalize_kinematics=True)
    return result


def _native_task_goal_status(predicate=None):
    """Read active BDDL goal predicates through the native evaluator."""
    task_env = env.env
    goal_states = task_env.parsed_problem.get("goal_state")
    if not isinstance(goal_states, list) or not goal_states:
        raise RuntimeError("active_BDDL_goal_state_is_unavailable")
    normalized = [tuple(str(token).lower() for token in state) for state in goal_states]
    if predicate is not None:
        if not isinstance(predicate, (list, tuple)) or not predicate:
            raise ValueError("native_goal_query_requires_a_nonempty_predicate_array")
        requested = tuple(str(token).lower() for token in predicate)
        if requested not in normalized:
            raise ValueError("requested_predicate_is_not_an_active_BDDL_goal")
        index = normalized.index(requested)
        return {
            "operation": "native_goal_query",
            "status": "complete",
            "predicate": list(goal_states[index]),
            "value": bool(task_env._eval_predicate(goal_states[index])),
            "timestep": int(task_env.timestep),
        }
    values = [
        {"predicate": list(state), "value": bool(task_env._eval_predicate(state))}
        for state in goal_states
    ]
    return {
        "operation": "native_task_status",
        "status": "complete",
        "native_goal_values": values,
        "task_success": bool(task_env._check_success()),
        "timestep": int(task_env.timestep),
        "reward": float(last_reward),
        "done": bool(last_done),
    }


def _native_goal_query_details(predicate):
    """Read one native On relation and its compiled live target site."""
    if not isinstance(predicate, (list, tuple)) or len(predicate) != 3:
        raise ValueError("native_goal_query_requires_a_three_item_relation")
    predicate = list(predicate)
    goals = copy.deepcopy(env.env.parsed_problem.get("goal_state"))
    if predicate not in goals:
        raise ValueError("requested_predicate_is_not_an_active_BDDL_goal")
    subject, region = predicate[1], predicate[2]
    site_object = env.env.object_sites_dict.get(region)
    if site_object is None:
        raise ValueError(f"compiled_BDDL_site_is_unavailable:{region}")
    site_id = int(env.sim.model.site_name2id(region))
    site_position = np.asarray(env.sim.data.site_xpos[site_id], dtype=np.float64)
    site_rotation = np.asarray(env.sim.data.site_xmat[site_id], dtype=np.float64).reshape(3, 3)
    subject_object = env.env.get_object(subject)
    subject_body = int(env.sim.model.body_name2id(subject_object.root_body))
    subject_position = np.asarray(env.sim.data.body_xpos[subject_body], dtype=np.float64)
    subject_quaternion = np.asarray(env.sim.data.body_xquat[subject_body], dtype=np.float64)
    return {
        "operation": "native_goal_query",
        "status": "complete",
        "read_only": True,
        "predicate": predicate,
        "predicate_is_native_goal": True,
        "predicate_value": bool(env.env._eval_predicate(predicate)),
        "region": {
            "name": region,
            "parent_name": site_object.parent_name,
            "site_id": site_id,
            "site_position_world_m": site_position.tolist(),
            "site_rotation_world": site_rotation.tolist(),
            "site_size_m": np.asarray(site_object.size, dtype=np.float64).tolist(),
            "site_local_position_m": np.asarray(site_object.site_pos, dtype=np.float64).tolist(),
            "site_local_quaternion_wxyz": np.asarray(site_object.site_quat, dtype=np.float64).tolist(),
        },
        "subject": {
            "name": subject,
            "body_id": subject_body,
            "body_name": subject_object.root_body,
            "root_position_world_m": subject_position.tolist(),
            "root_quaternion_wxyz": subject_quaternion.tolist(),
        },
    }

scene_metadata["model_geometry_sha256"] = _planning_model_signature(env.sim.model._model)
listener = Listener(("127.0.0.1", scene_args.port), authkey=b"gs-opd-local")
print("SAFETY BRIDGE READY", flush=True)
snapshots = {}
next_snapshot_id = 1

try:
    while True:
        conn = listener.accept()
        try:
            while True:
                try:
                    request = conn.recv()
                except EOFError:
                    break

                if request["op"] == "observe":
                    # Return the observation produced during Bridge startup
                    # without resetting the environment or advancing physics.
                    # This is useful for same-initialization RGB/geometry
                    # capture because reset can sample a different state.
                    result = pack(
                        obs,
                        last_reward,
                        last_done,
                        include_snapshot_state=bool(
                            request.get("include_snapshot_state", False)
                        ),
                    )
                    if request.get("safety_geometry_policy") is not None:
                        try:
                            result["safety_geometry"] = _safety_geometry_at_current_state(
                                request.get("safety_geometry_policy")
                            )
                        except Exception as exc:
                            result["safety_geometry"] = {
                                "status": "incomplete",
                                "minimum_forbidden_clearance_m": None,
                                "forbidden_contacts": [],
                                "self_contacts": [],
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                elif request["op"] == "reset":
                    # Reset may hard-rebuild the compiled model; prior snapshots
                    # are no longer valid for the new simulation instance.
                    snapshots.clear()
                    env.reset()
                    obs = env.set_init_state(states[scene_args.init_state])
                    scene_metadata["model_geometry_sha256"] = _planning_model_signature(
                        env.sim.model._model
                    )
                    controller_runtime_config = _configure_pi05_libero_osc_pose(
                        env.env.robots[0].controller
                    )
                    print(
                        "PI05 LIBERO OSC_POSE RUNTIME CONFIG:",
                        controller_runtime_config,
                        flush=True,
                    )
                    last_reward = 0.0
                    last_done = False
                    result = pack(obs)
                elif request["op"] == "step":
                    action = np.asarray(request["action"], dtype=np.float32)
                    assert action.shape == (7,)
                    obs, reward, done, info = env.step(action)
                    last_reward = reward
                    last_done = done
                    result = pack(
                        obs,
                        reward,
                        done,
                        include_snapshot_state=bool(request.get("include_snapshot_state", False)),
                    )
                    if request.get("include_contacts", False):
                        result["contacts"] = _collect_contacts()
                elif request["op"] == "step_monitored":
                    action = np.asarray(request["action"], dtype=np.float32)
                    assert action.shape == (7,)
                    start_contacts = _collect_contacts()
                    monitor = _new_contact_monitor(request.get("safety_geometry_policy"))
                    obs, reward, done, info = _run_env_step_with_contact_monitor(
                        action, int(request.get("action_index", 0)), monitor
                    )
                    last_reward = reward
                    last_done = done
                    result = pack(obs, reward, done, include_snapshot_state=True)
                    result["contact_monitoring"] = _contact_monitor_result(monitor)
                    result["contacts"] = _collect_contacts()
                    result["start_contacts"] = start_contacts
                    result["safety_geometry"] = monitor["geometry_substeps"][-1] if monitor.get("geometry_substeps") else None
                elif request["op"] == "joint_trajectory_step":
                    monitor = _new_contact_monitor()
                    try:
                        obs, reward, done, info, substep_contacts = (
                            _run_joint_trajectory_step_with_contact_monitor(
                                request, int(request.get("action_index", 0)), monitor
                            )
                        )
                        result = pack(obs, reward, done, include_snapshot_state=True)
                        result["control_mode"] = "joint_space_mass_matrix_PD_torque"
                        result["control_action_space"] = "7D_joint_target_qpos_rad"
                        result["pi05_native_action_compatible"] = False
                        result["contact_monitoring"] = _contact_monitor_result(monitor)
                        result["substep_contacts"] = substep_contacts
                        result["applied_joint_torques_nm_per_substep"] = info.get("applied_joint_torques_nm_per_substep")
                        result["contacts"] = _collect_contacts()
                        result["task_cost"] = info.get("cost")
                    except Exception as exc:
                        result = {
                            "operation": "joint_trajectory_step",
                            "status": "failed",
                            "error": f"{type(exc).__name__}: {exc}",
                            "control_mode": "joint_space_mass_matrix_PD_torque",
                            "control_action_space": "7D_joint_target_qpos_rad",
                            "pi05_native_action_compatible": False,
                            "contact_monitoring": _contact_monitor_result(monitor),
                        }
                elif request["op"] == "snapshot":
                    try:
                        snapshot_id = f"snapshot-{next_snapshot_id:06d}"
                        snapshots[snapshot_id] = _capture_snapshot()
                        next_snapshot_id += 1
                        result = {
                            "snapshot_id": snapshot_id,
                            "snapshot_state": _snapshot_diagnostics(canonicalize_kinematics=True),
                        }
                    except Exception as exc:
                        result = {"error": f"Snapshot failed: {type(exc).__name__}: {exc}"}
                elif request["op"] == "export_runtime_snapshot":
                    try:
                        result = _export_runtime_snapshot()
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "error": f"Runtime Snapshot export failed: {type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "restore_runtime_snapshot":
                    try:
                        result = _restore_runtime_snapshot(request.get("payload"))
                    except Exception as exc:
                        result = {
                            "status": "restore_failed",
                            "error": f"Runtime Snapshot restore failed: {type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "restore":
                    snapshot_id = request.get("snapshot_id")
                    if not isinstance(snapshot_id, str) or snapshot_id not in snapshots:
                        result = {"error": f"Unknown snapshot_id: {snapshot_id!r}"}
                    else:
                        try:
                            _restore_snapshot(snapshots[snapshot_id])
                            result = pack(
                                obs,
                                last_reward,
                                last_done,
                                include_snapshot_state=True,
                            )
                            result["restored_snapshot_id"] = snapshot_id
                        except Exception as exc:
                            result = {"error": f"Restore failed: {type(exc).__name__}: {exc}"}
                elif request["op"] == "branch_rollout":
                    try:
                        result = _branch_rollout(
                            request.get("snapshot_id"),
                            request.get("candidates"),
                            request.get("safety_geometry_policy"),
                            bool(request.get("capture_runtime_snapshots", False)),
                        )
                    except Exception as exc:
                        result = {
                            "error": f"Branch rollout failed: {type(exc).__name__}: {exc}",
                            "main_state_restored": None,
                        }
                elif request["op"] == "track_trajectory_bundle":
                    try:
                        result = _track_trajectory_bundle(request)
                    except Exception as exc:
                        result = {
                            "operation": "track_trajectory_bundle",
                            "status": "incomplete",
                            "error": (
                                "Trajectory tracking failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                            "main_state_restored": False,
                            "main_restore_verified": False,
                            "candidates": [],
                        }
                elif request["op"] == "trajectory_tracking_capabilities":
                    result = {
                        "status": "available",
                        "native_osc_pose_tracking": True,
                        "explicit_eef_reference_poses_world": True,
                        "substep_contact_monitoring": True,
                        "full_snapshot_restore_verification": True,
                    }
                elif request["op"] == "geometry_inventory":
                    try:
                        result = _geometry_inventory()
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "robot_classification_status": "unavailable",
                            "geom_count": None,
                            "geometries": [],
                            "error": (
                                "Geometry inventory failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                elif request["op"] == "compiled_model_fingerprint_details":
                    try:
                        result = _planning_model_signature_details(
                            env.sim.model._model
                        )
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "error": (
                                "Compiled model fingerprint detail query failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                elif request["op"] == "robot_joint_limit_inventory":
                    try:
                        result = _robot_joint_limit_inventory(
                            request.get("joint_limit_tolerance_policy")
                        )
                    except Exception as exc:
                        result = {
                            "operation": "robot_joint_limit_inventory",
                            "status": "unavailable",
                            "joints": [],
                            "physical_state_unchanged": True,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "task_semantic_geometry_bindings":
                    try:
                        result = _task_semantic_geometry_bindings()
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "bindings": {},
                            "error": (
                                "Task semantic geometry binding failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                elif request["op"] == "native_task_status":
                    try:
                        result = _native_task_goal_status()
                    except Exception as exc:
                        result = {
                            "operation": "native_task_status",
                            "status": "unavailable",
                            "native_goal_values": [],
                            "task_success": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "inspect_contacts":
                    try:
                        result = _collect_contacts()
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "contacts": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "native_goal_query":
                    try:
                        result = _native_goal_query_details(request.get("predicate"))
                    except Exception as exc:
                        result = {
                            "operation": "native_goal_query",
                            "status": "unavailable",
                            "value": None,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "geom_set_min_distance":
                    try:
                        result = _geom_set_min_distance(
                            request.get("geom_ids_a"),
                            request.get("geom_ids_b"),
                            request.get("distmax_m", 10.0),
                        )
                    except Exception as exc:
                        result = {
                            "operation": "geom_set_min_distance",
                            "status": "unavailable",
                            "query_complete": False,
                            "distance_reliable": False,
                            "minimum_distance_m": None,
                            "minimum_pair": None,
                            "error": f"Geom distance query failed: {type(exc).__name__}: {exc}",
                            "not_a_safety_proof": True,
                        }
                elif request["op"] == "safety_geometry_check":
                    try:
                        result = _safety_geometry_at_current_state(
                            request.get("safety_geometry_policy")
                        )
                    except Exception as exc:
                        result = {
                            "status": "incomplete",
                            "minimum_forbidden_clearance_m": None,
                            "forbidden_contacts": [],
                            "self_contacts": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "planning_geometry":
                    try:
                        result = _planning_geometry(
                            request.get("carried_binding"),
                            request.get("obstacle_bindings"),
                            request.get("expected_instruction"),
                        )
                    except Exception as exc:
                        result = {"operation": "planning_geometry", "status": "unavailable",
                                  "query_complete": False, "physical_state_unchanged": False,
                                  "error": f"Planning geometry failed: {type(exc).__name__}: {exc}"}
                elif request["op"] == "collision_scene_export":
                    try:
                        result = _collision_scene_export()
                    except Exception as exc:
                        result = {
                            "operation": "collision_scene_export",
                            "status": "unavailable",
                            "query_complete": False,
                            "physical_state_unchanged": False,
                            "scene_collision_geometries": [],
                            "robot_collision_geometries": [],
                            "unsupported_geometries": [],
                            "error": f"Collision scene export failed: {type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "scratch_robot_fk_audit":
                    try:
                        result = _scratch_robot_fk_audit(
                            request.get("arm_joint_positions_rad")
                        )
                    except Exception as exc:
                        result = {
                            "operation": "scratch_robot_fk_audit",
                            "status": "unavailable",
                            "physical_state_unchanged": False,
                            "poses": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "scratch_carried_geom_distance_audit":
                    try:
                        result = _scratch_carried_geom_distance_audit(
                            request.get("arm_joint_states_rad"),
                            request.get("object_id"),
                        )
                    except Exception as exc:
                        result = {
                            "operation": "scratch_carried_geom_distance_audit",
                            "status": "unavailable",
                            "physical_state_unchanged": False,
                            "states": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "scratch_robot_path_contact_audit":
                    try:
                        result = _scratch_robot_path_contact_audit(
                            request.get("arm_joint_states_rad"),
                            request.get("object_id"),
                            request.get("gripper_joint_states_rad"),
                            request.get("T_eef_object_m"),
                            request.get("T_world_object_m"),
                            request.get("fingertip_contact_scope", "none"),
                        )
                    except Exception as exc:
                        result = {
                            "operation": "scratch_robot_path_contact_audit",
                            "status": "unavailable",
                            "physical_state_unchanged": False,
                            "states": [],
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "rrt_connect":
                    try:
                        import hashlib
                        from research.gs_opd.repair.mujoco_jointspace_planning import plan_current_scene

                        expected_instruction = request.get("expected_instruction")
                        if expected_instruction != task.language:
                            raise ValueError("task_instruction_does_not_match_active_BDDL_task")
                        expected_model = request.get("model_geometry_sha256")
                        actual_model = _planning_model_signature(env.sim.model._model)
                        if expected_model != actual_model:
                            raise ValueError("compiled_model_geometry_signature_changed")
                        current_state = _mujoco_integration_state(env.sim)
                        current_state_hash = hashlib.sha256(current_state.tobytes()).hexdigest()
                        if request.get("start_state_sha256") != current_state_hash:
                            raise ValueError("active_7d_qpos_start_state_changed_before_planning")
                        policy = _normalize_safety_geometry_policy(
                            request.get("safety_geometry_policy")
                        )
                        result = plan_current_scene(
                            env,
                            request.get("goal_pose_world"),
                            policy,
                            start_state_sha256=current_state_hash,
                            expected_instruction=expected_instruction,
                            actual_instruction=task.language,
                            ik_seed_qpos=request.get("ik_seed_qpos"),
                            planner_config=request.get("planner_config"),
                        )
                        result["operation"] = "rrt_connect"
                        result["start_state_sha256"] = current_state_hash
                        result["model_geometry_sha256"] = actual_model
                        result["safety_policy_source"] = request.get("safety_geometry_policy", {}).get("source")
                    except Exception as exc:
                        result = {
                            "operation": "rrt_connect",
                            "status": "unavailable",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "live_state_unchanged": None,
                        }
                elif request["op"] == "task_success_first_context":
                    try:
                        result = _task_success_first_grasp_context(request)
                    except Exception as exc:
                        result = {
                            "operation": "task_success_first_grasp_context",
                            "status": "unavailable",
                            "read_only": True,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "task_success_first_articulation_context":
                    try:
                        result = _task_success_first_articulation_context(request)
                    except Exception as exc:
                        result = {
                            "operation": "task_success_first_articulation_context",
                            "status": "unavailable",
                            "read_only": True,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "task_success_first_rrt":
                    try:
                        import hashlib
                        from research.gs_opd.repair.mujoco_jointspace_planning import plan_current_scene

                        expected_instruction = request.get("expected_instruction")
                        if expected_instruction != task.language:
                            raise ValueError("task_instruction_does_not_match_active_BDDL_task")
                        expected_model = request.get("model_geometry_sha256")
                        actual_model = _planning_model_signature(env.sim.model._model)
                        if expected_model != actual_model:
                            raise ValueError("compiled_model_geometry_signature_changed")
                        current_state = _mujoco_integration_state(env.sim)
                        current_state_hash = hashlib.sha256(current_state.tobytes()).hexdigest()
                        if request.get("start_state_sha256") != current_state_hash:
                            raise ValueError("active_7d_qpos_start_state_changed_before_planning")
                        planner_config = request.get("planner_config")
                        if not isinstance(planner_config, dict) or planner_config.get(
                            "require_safe_configurations"
                        ) is not False:
                            raise ValueError("task_success_first_RRT_requires_explicit_record_only_mode")
                        policy = _normalize_safety_geometry_policy(
                            request.get("safety_geometry_policy")
                        )
                        result = plan_current_scene(
                            env,
                            request.get("goal_pose_world"),
                            policy,
                            start_state_sha256=current_state_hash,
                            expected_instruction=expected_instruction,
                            actual_instruction=task.language,
                            ik_seed_qpos=request.get("ik_seed_qpos"),
                            planner_config=planner_config,
                        )
                        result["operation"] = "task_success_first_rrt"
                        result["planning_mode"] = "kinematic_route_with_record_only_safety_diagnostics"
                        result["safety_certified"] = False
                        result["start_state_sha256"] = current_state_hash
                        result["model_geometry_sha256"] = actual_model
                        result["safety_policy_source"] = request.get("safety_geometry_policy", {}).get("source")
                    except Exception as exc:
                        result = {
                            "operation": "task_success_first_rrt",
                            "status": "unavailable",
                            "planning_mode": "kinematic_route_with_record_only_safety_diagnostics",
                            "safety_certified": False,
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "live_state_unchanged": None,
                        }
                elif request["op"] == "screen_grasp_candidates":
                    try:
                        import hashlib
                        from research.gs_opd.repair.mujoco_jointspace_planning import screen_grasp_candidates

                        expected_instruction = request.get("expected_instruction")
                        if expected_instruction != task.language:
                            raise ValueError("task_instruction_does_not_match_active_BDDL_task")
                        actual_model = _planning_model_signature(env.sim.model._model)
                        if request.get("model_geometry_sha256") != actual_model:
                            raise ValueError("compiled_model_geometry_signature_changed")
                        current_state = _mujoco_integration_state(env.sim)
                        current_state_hash = hashlib.sha256(current_state.tobytes()).hexdigest()
                        if request.get("start_state_sha256") != current_state_hash:
                            raise ValueError("active_state_changed_before_candidate_screen")
                        policy = _normalize_safety_geometry_policy(
                            request.get("safety_geometry_policy")
                        )
                        finger_groups, _closure = _tracking_gripper_sides(env.robots[0])
                        finger_geom_ids = sorted(
                            finger_groups["left"] | finger_groups["right"]
                        )
                        required_finger_suffixes = {
                            "finger1_collision",
                            "finger1_pad_collision",
                            "finger2_collision",
                            "finger2_pad_collision",
                        }
                        finger_geom_names = {
                            str(env.sim.model.geom_id2name(int(geom_id)))
                            for geom_id in finger_geom_ids
                        }
                        matched_finger_suffixes = {
                            suffix
                            for name in finger_geom_names
                            for suffix in required_finger_suffixes
                            if name.endswith(suffix)
                        }
                        if (
                            len(finger_geom_ids) != 4
                            or matched_finger_suffixes != required_finger_suffixes
                        ):
                            raise ValueError(
                                "candidate_screen_requires_exact_four_Panda_finger_mesh_pad_geoms"
                            )
                        result = screen_grasp_candidates(
                            env,
                            request.get("candidates"),
                            policy,
                            target_object_id=request.get("target_object_id"),
                            target_root_body_id=request.get("target_root_body_id"),
                            left_finger_geom_ids=sorted(finger_groups["left"]),
                            right_finger_geom_ids=sorted(finger_groups["right"]),
                            gripper_geom_ids=finger_geom_ids,
                            closed_gripper_qpos=request.get("closed_gripper_qpos"),
                            predicted_stable_object_pose_world=request.get("predicted_stable_object_pose_world"),
                            start_state_sha256=current_state_hash,
                            expected_instruction=expected_instruction,
                            actual_instruction=task.language,
                            clearance_m=float(request.get("clearance_m", 0.010)),
                        )
                        result["model_geometry_sha256"] = actual_model
                        result["safety_policy_source"] = request.get("safety_geometry_policy", {}).get("source")
                    except Exception as exc:
                        result = {
                            "operation": "screen_grasp_candidates",
                            "status": "unavailable",
                            "screening_type": "offline_scratch_only_not_physical_execution",
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "live_state_unchanged": None,
                            "candidates": [],
                        }
                elif request["op"] == "osc_prefix_probe":
                    try:
                        result = _osc_prefix_probe(request)
                    except Exception as exc:
                        result = {
                            "operation": "osc_prefix_probe",
                            "status": "incomplete",
                            "complete": False,
                            "safe": False,
                            "main_state_restored_verified": False,
                            "failure_reason": f"{type(exc).__name__}: {exc}",
                            "actions": [],
                            "steps": [],
                        }
                elif request["op"] == "gripper_geometry_inventory":
                    try:
                        result = _gripper_geometry_inventory()
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "robot_classification_status": "unavailable",
                            "robot_body_ids": None,
                            "model_body_count": None,
                            "model_geom_count": None,
                            "grippers": [],
                            "error": (
                                "Gripper geometry inventory failed: "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        }
                elif request["op"] == "grasp_state":
                    result = _live_grasp_state(
                        request.get("carried_binding"),
                        request.get("expected_instruction"),
                    )
                elif request["op"] == "repair_context":
                    try:
                        result = _repair_context()
                    except Exception as exc:
                        result = {
                            "status": "unavailable",
                            "robots": [],
                            "error": f"Repair context failed: {type(exc).__name__}: {exc}",
                        }
                elif request["op"] == "render_highres":
                    try:
                        result = _render_highres(request.get("resolution", 512))
                    except Exception as exc:
                        result = {
                            "error": (
                                "High-resolution render failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        }
                elif request["op"] == "geom_segmentation":
                    try:
                        result = _render_geom_segmentation(request.get("resolution", 512))
                    except Exception as exc:
                        result = {
                            "error": (
                                "Geom segmentation render failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        }
                else:
                    result = {"error": "Unknown operation"}

                conn.send(result)
        finally:
            conn.close()
finally:
    listener.close()
    env.close()
