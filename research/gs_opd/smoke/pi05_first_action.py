import os
from pathlib import Path
from multiprocessing.connection import Client

os.environ["EMBODIED_PATH"] = "/workspace/zy_workspace/RLinf/examples/embodiment"

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from rlinf.models import get_model
from rlinf.envs.sim.libero.utils import get_libero_image, get_libero_wrist_image, quat2axisangle

ckpt = Path("/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft")
assert (ckpt / "model.safetensors").is_file(), f"Checkpoint missing: {ckpt}"

with initialize_config_dir(config_dir="/workspace/zy_workspace/RLinf/evaluations/libero", version_base=None):
    cfg = compose(config_name="libero_spatial_openpi_pi05_smoke")

cfg.rollout.model.model_path = str(ckpt)
OmegaConf.update(cfg.rollout.model, "openpi.torch_compile", False, force_add=True)

print("MODEL PATH:", ckpt, flush=True)
print("NORM STATS EXISTS:", (ckpt / "norm_stats.json").is_file(), flush=True)
print("LOADING PI0.5...", flush=True)

model = get_model(cfg.rollout.model)
model = model.to("cuda:0").eval()
print("MODEL LOADED", flush=True)

conn = Client(("127.0.0.1", 38567), authkey=b"gs-opd-local")
try:
    conn.send({"op": "reset"})
    data = conn.recv()

    obs = {
        k: np.frombuffer(v["data"], dtype=v["dtype"]).reshape(v["shape"]).copy()
        for k, v in data["images"].items()
    }
    obs.update({
        k: np.asarray(v, dtype=np.float32)
        for k, v in data["state"].items()
    })

    main = np.ascontiguousarray(get_libero_image(obs))
    wrist = np.ascontiguousarray(get_libero_wrist_image(obs))
    state = np.concatenate([
        obs["robot0_eef_pos"],
        quat2axisangle(obs["robot0_eef_quat"].copy()),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)

    env_obs = {
        "main_images": torch.from_numpy(main).unsqueeze(0),
        "wrist_images": torch.from_numpy(wrist).unsqueeze(0),
        "states": torch.from_numpy(state).unsqueeze(0),
        "task_descriptions": [data["instruction"]],
    }

    print("PREDICTING...", flush=True)
    with torch.no_grad():
        actions, _ = model.predict_action_batch(env_obs, mode="eval")

    print("ACTIONS SHAPE:", tuple(actions.shape), flush=True)
    action = actions[0, 0].detach().float().cpu().numpy()
    assert action.shape == (7,) and np.isfinite(action).all()
    print("FIRST ACTION:", action.tolist(), flush=True)

    conn.send({"op": "step", "action": action.tolist()})
    result = conn.recv()
    print("REWARD:", result["reward"], "DONE:", result["done"], flush=True)
    print("===== PI05 SAFETY FIRST ACTION PASS =====", flush=True)
finally:
    conn.close()
