import os
os.environ["LIBERO_CONFIG_PATH"] = "/workspace/zy_workspace/.libero_safety_config"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from multiprocessing.connection import Listener
import numpy as np
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

def pack(obs, reward=0.0, done=False):
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
    return {
        "images": images,
        "state": state,
        "reward": float(reward),
        "done": bool(done),
        "instruction": task.language,
    }

listener = Listener(("127.0.0.1", 38567), authkey=b"gs-opd-local")
print("SAFETY BRIDGE READY", flush=True)

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
                    env.reset()
                    obs = env.set_init_state(states[0])
                    result = pack(obs)
                elif request["op"] == "step":
                    action = np.asarray(request["action"], dtype=np.float32)
                    assert action.shape == (7,)
                    obs, reward, done, info = env.step(action)
                    result = pack(obs, reward, done)
                else:
                    result = {"error": "Unknown operation"}

                conn.send(result)
        finally:
            conn.close()
finally:
    listener.close()
    env.close()
