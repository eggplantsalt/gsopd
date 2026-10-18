import os
os.environ["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from multiprocessing.connection import Listener
import copy
import hashlib
import json
import math
import numpy as np
import random
import mujoco
from pathlib import Path
import sys
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

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


suite = get_benchmark("obstacle_avoidance")()
task = suite.get_task_by_level_id(0, 0)
bddl = suite.get_task_bddl_file_path_by_level_id(0, 0)

print("LOADING SAFETY ENV...", flush=True)
env = OffScreenRenderEnv(
    bddl_file_name=bddl,
    camera_heights=256,
    camera_widths=256,
)
env.seed(0)
env.reset()
states = suite.get_task_init_states_by_level_id(0, 0)
obs = env.set_init_state(states[0])
controller_runtime_config = _configure_pi05_libero_osc_pose(env.env.robots[0].controller)
print("PI05 LIBERO OSC_POSE RUNTIME CONFIG:", controller_runtime_config, flush=True)
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
        robot_state = diagnostic["robot_states"][0]
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
        for value in references[1:]:
            target_eef = value + initial_object_rotation @ relative_position_local
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
                if joint_limits["status"] != "pass":
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
                or candidate.get("status") not in (None, "geometric_candidate")
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
                if candidate.get("status") not in (None, "geometric_candidate"):
                    raise ValueError("candidate has an unsupported execution status")
                item = _track_trajectory_candidate(
                    candidate,
                    target_snapshot,
                    binding,
                    options,
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
            object_states[f"{mapping_name}:{object_name}"] = {
                "body_id": int(body_id),
                "body_name": str(root_body),
                "root_pos": np.asarray(diagnostic_body_xpos[body_id]).tolist(),
                "root_quat": np.asarray(diagnostic_body_xquat[body_id]).tolist(),
                "joint_qpos": joints,
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


def _build_geometry_inventory(
    model, geom_count, robot_body_ids, robot_classification_error=None
):
    """Build a read-only inventory from the active MuJoCo model wrapper."""
    geometries = []
    has_lookup_errors = False
    body_count = int(model._model.nbody)

    for geom_id in range(geom_count):
        item = {
            "geom_id": geom_id,
            "geom_name": None,
            "body_id": None,
            "body_name": None,
            "category": "unknown",
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
        except Exception as exc:
            errors["body_id"] = f"{type(exc).__name__}: {exc}"

        if errors:
            item["errors"] = errors
            has_lookup_errors = True
        geometries.append(item)

    classification_available = robot_body_ids is not None
    status = (
        "complete"
        if classification_available and not has_lookup_errors
        else "partial"
    )
    result = {
        "status": status,
        "robot_classification_status": (
            "available" if classification_available else "unavailable"
        ),
        "geom_count": geom_count,
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


def _render_highres():
    """Render the existing primary camera without advancing or rebuilding sim."""
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
                width=512,
                height=512,
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
    if raw_rgb.shape != (512, 512, 3) or raw_rgb.dtype != np.uint8:
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
        "width": 512,
        "height": 512,
        "color_space": "RGB",
        "render_api": "env.sim.render",
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


def _render_geom_segmentation():
    """Capture same-state RGB and object segmentation from the primary camera."""
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
                width=512,
                height=512,
                camera_name="agentview",
                depth=False,
                segmentation=False,
            )
        )
        object_segmentation = sim.render(
            width=512,
            height=512,
            camera_name="agentview",
            depth=False,
            segmentation=True,
        )
        rgb_after = np.ascontiguousarray(
            sim.render(
                width=512,
                height=512,
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
    if rgb_before.shape != (512, 512, 3) or rgb_before.dtype != np.uint8:
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
        "width": 512,
        "height": 512,
        "color_space": "RGB",
        "render_api": "env.sim.render",
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


def _new_contact_monitor():
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


def _branch_rollout(snapshot_id, raw_candidates):
    global obs, last_reward, last_done

    if not isinstance(snapshot_id, str) or snapshot_id not in snapshots:
        raise ValueError(f"Unknown snapshot_id: {snapshot_id!r}")
    candidates = _normalize_branch_candidates(raw_candidates)
    target_snapshot = snapshots[snapshot_id]

    # Capture the caller's live state before touching the simulator. Every
    # candidate restores target_snapshot, and this finally block restores the
    # state that was active when branch_rollout was requested.
    main_snapshot = _capture_snapshot()
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
            contact_monitor = _new_contact_monitor()
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
                    candidate_result["steps"].append(
                        {
                            "step_index": action_index,
                            "action": action.tolist(),
                            "reward": float(reward),
                            "done": bool(done),
                            "state": state,
                            "contacts": contacts,
                            "substep_contact_coverage": contact_monitor[
                                "control_steps"
                            ][-1],
                        }
                    )
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
            result["main_state_restored"] = True
        except Exception as exc:
            result["main_state_restored"] = False
            result["main_restore_error"] = f"{type(exc).__name__}: {exc}"

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
    result = {
        "images": images,
        "state": state,
        "reward": float(reward),
        "done": bool(done),
        "instruction": task.language,
    }
    if include_snapshot_state:
        result["snapshot_state"] = _snapshot_diagnostics(canonicalize_kinematics=True)
    return result

listener = Listener(("127.0.0.1", 38567), authkey=b"gs-opd-local")
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

                if request["op"] == "reset":
                    # Reset may hard-rebuild the compiled model; prior snapshots
                    # are no longer valid for the new simulation instance.
                    snapshots.clear()
                    env.reset()
                    obs = env.set_init_state(states[0])
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
                        result = _render_highres()
                    except Exception as exc:
                        result = {
                            "error": (
                                "High-resolution render failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        }
                elif request["op"] == "geom_segmentation":
                    try:
                        result = _render_geom_segmentation()
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
