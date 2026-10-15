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
    # mj_forward recomputes derived quantities and can also update integration
    # inputs such as qacc_warmstart while solving constraints. Reapply the saved
    # integration state after forward so restoring does not silently replace
    # state captured at the snapshot boundary.
    mujoco.mj_setState(model, data, state, state_spec)


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
            try:
                _restore_snapshot(target_snapshot)
                candidate_result["start_state"] = _snapshot_diagnostics()
                for action_index, action in enumerate(candidate["actions"]):
                    obs, reward, done, _info = env.step(action)
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
                        }
                    )
                    candidate_result["executed_steps"] += 1
                    if done:
                        break
            except Exception as exc:
                candidate_result["status"] = "error"
                candidate_result["error"] = f"{type(exc).__name__}: {exc}"
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
                else:
                    result = {"error": "Unknown operation"}

                conn.send(result)
        finally:
            conn.close()
finally:
    listener.close()
    env.close()
