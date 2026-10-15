import os
os.environ["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from multiprocessing.connection import Listener
import copy
import numpy as np
import random
import mujoco
from libero.libero.benchmark import get_benchmark
from libero.libero.envs import OffScreenRenderEnv

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
    sim.forward()


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


def _snapshot_diagnostics():
    sim = env.sim
    task_env = env.env
    robot_states = []
    for index, robot in enumerate(env.robots):
        robot_state = {
            "robot_index": index,
            "joint_qpos": np.asarray(sim.data.qpos[robot._ref_joint_pos_indexes]).tolist(),
            "joint_qvel": np.asarray(sim.data.qvel[robot._ref_joint_vel_indexes]).tolist(),
            "eef_pos": np.asarray(sim.data.site_xpos[robot.eef_site_id]).tolist(),
        }
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
                "root_pos": np.asarray(sim.data.body_xpos[body_id]).tolist(),
                "root_quat": np.asarray(sim.data.body_xquat[body_id]).tolist(),
                "joint_qpos": joints,
            }

    return {
        "mujoco_integration_state": _mujoco_integration_state(sim).tolist(),
        "robot_states": robot_states,
        "object_states": object_states,
    }

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
        result["snapshot_state"] = _snapshot_diagnostics()
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
                            "snapshot_state": _snapshot_diagnostics(),
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
                else:
                    result = {"error": "Unknown operation"}

                conn.send(result)
        finally:
            conn.close()
finally:
    listener.close()
    env.close()
