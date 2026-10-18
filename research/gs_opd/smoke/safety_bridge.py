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
        "geom_ids": collision_ids,
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
