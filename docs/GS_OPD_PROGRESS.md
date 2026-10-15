# GS-OPD / π₀.₅ × LIBERO-Safety 进度

## 当前环境与基线

- 仓库：`/workspace/zy_workspace/RLinf`；本次修改时 HEAD 为 `61e2401f50eac200b18da2bbcd15b88813f0ac85`。
- Python：仓库 `.venv/bin/python` 为 Python 3.11.14。
- Checkpoint：`/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft`。此前评测日志记录从该目录加载 `model.safetensors`，并从 checkpoint 中加载 OpenPI LIBERO norm stats。
- 已完成的普通 LIBERO-Spatial task 0 闭环基线：240 步、1 条 trajectory，`success_once=1.0`、`success_at_end=1.0`、`return=1.0`。当时记录的 Git HEAD 为 `fccbb19c`；日志目录为 `logs/20261014-19:19:37-libero_spatial_openpi_pi05_smoke`，指标在 `metrics.log`，完整运行记录在 `eval_embodiment.log`。该结果是普通 LIBERO-Spatial 基线，不是 LIBERO-Safety 结果。

## 本次改动

修改 `research/gs_opd/smoke/pi05_rollout.py`：

- 新增 `--max-steps`，默认 `240`；只替换原先固定的循环上限，结果字典中的原有统计字段保持不变。
- 新增可选 `--video-out <path.mp4>`。不传该参数时不收集视频帧，也不调用视频 writer。
- 启用视频时，reset 后先记录初始 `agentview_image`；每次 action 的 `step` 返回后立即记录新帧。帧通过 RLinf 的 `get_libero_image()` 处理，按训练预处理方式旋转 180°，并转换为连续数组。
- 结束时调用 RLinf 现有 `save_rollout_video()` 写 MP4；该函数使用仓库已有的 `imageio`，未新增依赖。视频帧数应为实际执行动作数加初始帧数。

## On-Policy Trace 第一版

`research/gs_opd/data/trajectory_writer.py` 新增 `TrajectoryWriter`，`pi05_rollout.py` 新增可选 `--trace-out <path.npz>`。未指定时不导入 writer、不收集 trace 数据；模型推理、action chunk、环境 step 和视频帧路径保持原样。

启用 trace 后，每次 policy 决策记录 Bridge 原始 RGB 双相机图像、原始机器人状态字段、当前指令、零起始 chunk 索引和完整预测动作 chunk。每个实际执行的 action 另存零起始全局 step 索引、所属 chunk 与 chunk 内 action 索引、执行动作、step 返回的 reward/done 以及执行后双相机和机器人状态。trace 图像保留原始方向；既有 MP4 仍使用 RLinf 的 180° 图像预处理。

NPZ 通过 `numpy.savez_compressed` 写入，不使用 object 数组或 pickle：

| 字段 | 内容 |
|---|---|
| `metadata_json` | JSON 标量，含 schema 版本、suite/level/task、instruction、checkpoint、max/actual steps、chunk/transition 数量及索引基数。 |
| `decision_chunk_indices`、`decision_step_indices`、`decision_instructions` | 每个决策 chunk 的索引、生成前已执行步数和指令。 |
| `decision_agentview_image`、`decision_robot0_eye_in_hand_image` | 每个 chunk 决策状态的 raw RGB。 |
| `decision_robot0_eef_pos`、`decision_robot0_eef_quat`、`decision_robot0_gripper_qpos` | 每个 chunk 决策状态的 Bridge 机器人状态。 |
| `predicted_actions`、`predicted_action_offsets` | 拼接后的完整预测动作；chunk `i` 对应 `predicted_actions[offsets[i]:offsets[i+1]]`。offsets 支持不同 chunk 长度。 |
| `step_indices`、`step_chunk_indices`、`step_action_indices` | 每个真实执行 step 的全局零起始索引、所属预测 chunk 和 chunk 内动作下标。 |
| `executed_actions`、`step_rewards`、`step_dones` | 实际发给 Bridge 的动作及其 step reward/done。 |
| `post_agentview_image`、`post_robot0_eye_in_hand_image`、`post_robot0_eef_pos`、`post_robot0_eef_quat`、`post_robot0_gripper_qpos` | 每个动作执行后的 raw Bridge 观测。 |

## 验证情况

- 已运行 `PYTHONPYCACHEPREFIX=/tmp/rlinf-pycache .venv/bin/python -m py_compile research/gs_opd/data/trajectory_writer.py research/gs_opd/smoke/pi05_rollout.py`，两个文件语法检查通过。
- 本次没有启动 Safety Bridge、没有运行 rollout，也没有生成 trace NPZ 或新 MP4；NPZ 字段布局已在源码定义，实际落盘读取和帧/状态对齐待人工短程验证。
- Safety Bridge 使用 `research/gs_opd/smoke/safety_bridge.py`，监听本机 `127.0.0.1:38567`；Rollout 返回的初始和 step 观测都包含 `agentview_image`。

## MuJoCo Snapshot / Restore 第一版

### 接口审计与保存范围

- 已审计 LIBERO-Safety 的 `libero/libero/envs/env_wrapper.py`、`libero/libero/envs/bddl_base_domain.py`，以及 vendored robosuite 1.4 的 `utils/binding_utils.py`、`controllers/osc.py`、`controllers/interpolators/linear_interpolator.py`、`robots/single_arm.py` 和 `models/grippers/panda_gripper.py`。
- LIBERO `ControlEnv.get_sim_state()` 使用的 `MjSimState.flatten()` 仅保存 MuJoCo `time/qpos/qvel`，其 `set_state_from_flattened()` 也只恢复这三项；它不足以独立支持确定性回放。
- Safety venv 中 MuJoCo 版本为 `3.3.2`。Bridge 改用 `mujoco.mj_getState` / `mj_setState` 的 `mjSTATE_INTEGRATION`，保存 MuJoCo integration state（物理积分变量、`ctrl`/外力输入、约束激活、warm-start、mocap、userdata、插件 state 和 integration history），恢复后调用 `sim.forward()` 重建派生量。已用最小 MuJoCo 模型验证该 venv 的 get/set API 可用；没有启动完整环境或 Rollout。
- Python 侧同时保存并恢复 robosuite `OSC_POSE` 控制器字段及位置/姿态插值器状态、机器人历史缓冲、Panda 夹爪累计 `current_action`、task 的 `cur_time/timestep/done`、LIBERO 动态物体 motion generator 的内部字段、可变物体显示属性、Observable 的采样状态和 `_obs_cache`，Bridge 最近一次观测/reward/done，以及进程级 `numpy.random` 和 `random` 状态。当前实现针对运行中的同一 Bridge 环境和同一编译模型，通过进程内 Snapshot ID 引用，不写磁盘；再次 Reset 会使现有 Snapshot ID 失效。
- Bridge 原有 `reset` 和 `step` 请求/响应字段保持原样；只有 Smoke 请求显式传 `include_snapshot_state=true` 时，step 响应才额外包含 MuJoCo、机器人关节/末端和场景对象诊断状态。Snapshot/Restore 为新增操作；Restore 返回保存时的 Bridge 观测，并附上状态诊断。

### Smoke 与限制

- 新增 `research/gs_opd/smoke/snapshot_smoke.py`：通过现有 `multiprocessing.connection` Bridge 协议执行 Reset → Snapshot S → 固定 4 个 7 维小幅动作 → Restore S → 重放相同动作。逐步比较 MuJoCo integration state、机器人状态和物体/fixture 状态；默认绝对误差阈值 `1e-6`，输出 PASS/FAIL。
- 保存范围覆盖已审计的 LIBERO-Safety BDDL/robosuite 路径，但 Snapshot 是同一进程、同一模型的临时状态，不支持跨进程或磁盘恢复。当前 Bridge 使用的 OSC 控制器和插值器、LIBERO motion generator 及全局 RNG 已纳入；发现不支持的控制器 helper 状态时 Snapshot 会报错，而不是静默略过。
- 尚不能保证恢复自定义控制器/传感器闭包中的私有 RNG 或外部状态、MuJoCo 插件的进程外状态、渲染器/GPU 内部缓存，以及第三方任务在 `model` 上自行修改但未纳入 task object properties 的任意字段。当前普通 Safety 环境没有在 step 中发现这类额外状态；实际确定性仍需运行下面的 Smoke 命令核验。

### 语法与 API 验证

- 已通过 Safety Python 环境语法检查：

  ```bash
  PYTHONPYCACHEPREFIX=/tmp/libero-safety-pycache \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python -m py_compile \
    research/gs_opd/smoke/safety_bridge.py \
    research/gs_opd/smoke/snapshot_smoke.py
  ```

- MuJoCo `3.3.2` 最小模型 get/set API 探针通过，integration state 长度为 `38`，修改后恢复的 free body x 坐标回到 `0.000000`。
- Safety venv 下 `libero.libero.benchmark` 与 `OffScreenRenderEnv` import 检查通过；运行时输出了 robosuite private macro 建议和 Gym deprecation warning。
- 尚未创建 LIBERO-Safety 仿真实例、执行 Snapshot Smoke 或运行完整 Rollout；等待人工审查和实际环境验证。

### Snapshot Smoke 启动命令

终端 A 启动 LIBERO-Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端 B 执行一致性测试：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/snapshot_smoke.py --atol 1e-6
```

测试会在控制台输出每步 MuJoCo、机器人和物体状态最大绝对误差，最后给出 `PASS` 或 `FAIL`；不会生成轨迹文件。

## 10 步 Trace 采集命令

先在终端 A 启动 Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
.venv/bin/python research/gs_opd/smoke/safety_bridge.py
```

再在终端 B 运行 10 步采集：

```bash
cd /workspace/zy_workspace/RLinf
.venv/bin/python research/gs_opd/smoke/pi05_rollout.py \
  --max-steps 10 \
  --trace-out /workspace/zy_workspace/RLinf/logs/gs_opd/pi05_safety_l0t0_10step_trace.npz
```

预期 NPZ 路径：`/workspace/zy_workspace/RLinf/logs/gs_opd/pi05_safety_l0t0_10step_trace.npz`。需要同时保存视频时，可再传 `--video-out <path.mp4>`；trace 使用 raw RGB，视频仍按原有图像预处理旋转 180°。

## 人工审查后的短程启动方式

先在终端 A 启动 Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
.venv/bin/python research/gs_opd/smoke/safety_bridge.py
```

再在终端 B 执行 20 步 smoke，并保存视频：

```bash
cd /workspace/zy_workspace/RLinf
.venv/bin/python research/gs_opd/smoke/pi05_rollout.py \
  --max-steps 20 \
  --video-out /workspace/zy_workspace/RLinf/logs/gs_opd/pi05_safety_l0t0_20step.mp4
```

预期视频路径：`/workspace/zy_workspace/RLinf/logs/gs_opd/pi05_safety_l0t0_20step.mp4`。视频仅在传入 `--video-out` 时写出；若提前 success/done，实际帧数按实际执行步数计算。

## 下一步

1. 人工审查 Snapshot 状态字段与 Bridge 操作，再执行上述一致性 Smoke；重点核对恢复点误差和重放误差。
2. 执行 10 步 trace 采集命令，用 `numpy.load(..., allow_pickle=False)` 读取 NPZ，核对 chunk offsets、step→chunk/action 映射和 decision/post 观测顺序；若同时保存视频，检查 MP4 可解码与帧顺序。
3. 短程 Snapshot 与数据核验通过后，再按需运行 240 步 LIBERO-Safety task 0，并扩展到更多 suite 任务；后续再基于已审查轨迹设计 Teacher、Repair 与训练流程。

## MuJoCo Action Branching 第一版

### Bridge 行为与接口

`research/gs_opd/smoke/safety_bridge.py` 新增 `branch_rollout` 操作。客户端请求使用现有本机 multiprocessing Bridge 连接，格式如下：

```python
{
    "op": "branch_rollout",
    "snapshot_id": "snapshot-000001",
    "candidates": [
        {"candidate_id": "candidate-A", "actions": [[0.04, 0, 0, 0, 0, 0, 0]]},
        {"candidate_id": "candidate-B", "actions": [[-0.04, 0, 0, 0, 0, 0, 0]]},
    ],
}
```

Bridge 会先校验 Snapshot ID 和全部候选动作（形状为 `steps × 7`、非空且为有限数值），再保存调用时的主环境状态。每条候选执行前都从请求指定的同一个 Snapshot 恢复；动作按序调用现有 `env.step()`，遇到 `done` 后停止该候选。候选结果只返回给本次请求，不进入主 Rollout 的 step/trace/video 记录。操作结束时（包括候选执行异常）通过 `finally` 尝试恢复调用前的主环境状态，并返回 `main_state_restored`；若恢复失败，响应附带 `main_restore_error`。

响应结构摘要：

| 字段 | 内容 |
|---|---|
| `snapshot_id` | 分支起点 Snapshot ID。 |
| `main_state_restored` | Bridge 是否成功完成调用前主环境状态恢复。 |
| `candidates[]` | 各候选独立结果，含 ID、`status`、请求/实际步数、起始状态和逐步记录。 |
| `candidates[].start_state` | Snapshot 恢复后、执行动作前的 MuJoCo integration state、机器人状态和物体/fixture 状态。 |
| `candidates[].steps[]` | 零起始步索引、实际动作、reward、done、执行后物理状态和接触信息。 |
| `steps[].contacts.status` | `no_contact` 表示 MuJoCo 可查询且 `ncon == 0`；`contact` 表示已枚举一个或多个接触；`unavailable` 表示接触计数或接触条目无法读取。 |
| `steps[].contacts.contacts[]` | 每个接触的 geom/body ID 与名称、接触位置，以及可用时的 6 维 contact-frame 力/力矩。每项以 `force_status` 单独标记力数据 `available` 或 `unavailable`。 |

单个候选失败会在该候选结果中返回 `status: "error"` 和错误信息，Bridge 随后仍会尝试从目标 Snapshot 执行其他候选；分支操作最终仍会尝试恢复调用前的主状态。无效 Snapshot 或候选输入会在环境被修改前返回错误。分支响应不包含相机图像，也不写视频或轨迹文件。

### Branch Smoke

新增 `research/gs_opd/smoke/branch_smoke.py`。测试流程现在严格区分分支起点和调用前主环境状态：Reset → 保存 S₀ → 主环境执行四个非零 7 维动作并保存 S₁ 诊断 → 从 S₀ 执行候选 A、B、A-repeat → 核对主环境仍处于 S₁。脚本还要求 S₀ 与 S₁ 的最大状态差大于 `--atol`（默认 `1e-6`），防止两者意外相同导致恢复断言失去区分力；逐候选起点须与 S₀ 在容差内一致，A 与 A-repeat 的逐步状态/reward/done/contact 结果须一致，分支后主环境须与 S₁ 一致。逐步输出 MuJoCo integration、机器人和物体状态相对候选起点的最大绝对变化，以及完整接触事件 JSON；`no_contact` 与 `unavailable` 会按 Bridge 状态原样显示。

### 运行命令

终端 A 使用 LIBERO-Safety Python 环境启动 Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端 B 执行分支 Smoke：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/branch_smoke.py --atol 1e-6
```

### 限制与验证状态

- 本次只做语法检查，没有启动 Bridge、创建仿真实例或执行 Smoke；S₀/S₁ 差异、重复候选误差及主状态恢复仍待人工运行验证。
- Restore 诊断已扩展：分别输出 `mujoco_integration_state`、`robot_states`、`object_states` 的最大绝对误差及对应字段/数组下标；新增 S₁ 上执行非零探针动作后直接 Restore S₁ 的对照，另比较分支结束状态与分支调用前状态，从而区分普通 Snapshot/Restore 与 Branching 的恢复误差。
- 根因证据与最小修复：原 `_set_mujoco_integration_state()` 调用 `mj_setState()` 后执行 `sim.forward()`，而 forward 求解约束时可能更新 `mjSTATE_INTEGRATION` 包含的 `qacc_warmstart`。在 LIBERO-Safety MuJoCo 3.3.2 环境用一个带接触的最小 MuJoCo 模型复现：接触数为 1，模型 `nq=7`、`nv=6`，integration state 中 warm-start 段从 index 14 开始；执行 `mj_setState` + `mj_forward` 后 index 16（`qacc_warmstart[2]`）从 `8.056839445962867e-10` 变为 `7.197303360572629e-10`。再次 `mj_setState` 可精确恢复保存的向量。Bridge 现在在 `sim.forward()` 后再次应用原始 integration state，保留 forward 计算的派生量并恢复求解器 warm-start 等状态。原 `_restore_snapshot()` 中控制器字段恢复只更新 Python 控制器/缓冲对象，不执行仿真步或写 MuJoCo 状态，因此没有证据表明它会覆盖该物理状态。
- 用户报告的 `0.6662` 尚未通过增强版 Smoke 定位到具体字段；上面的 API 探针证实了 Restore 路径存在 integration state 被 forward 改写的问题，但该数值是否完全来自该分量，需由 Smoke 新增的逐分量/索引输出确认。未运行完整 Bridge Smoke，也不据此宣称端到端已通过。
- Snapshot/分支仍只支持当前 Bridge 进程中的同一编译 MuJoCo 模型；Reset 会清除旧 Snapshot。已知 Snapshot 尚不能保证覆盖自定义控制器/传感器闭包私有 RNG、插件进程外状态、渲染器/GPU 缓存和第三方任务任意外部副作用，见上文 Snapshot 限制。
- 逐步对象诊断复用现有 `_snapshot_diagnostics()`：列出 BDDL `objects_dict` / `fixtures_dict` 中具有 `root_body` 的对象根位姿、四元数及对象关节位置；未注册对象不会有单独命名条目，但其 MuJoCo integration state 仍包含在完整物理状态向量中。
- MuJoCo 返回的力/力矩是接触坐标系下的六维值；没有约束力地址或 API 调用失败时，该接触几何信息仍保留，但 `force_status` 为 `unavailable`。本功能只记录物理事实，不把接触分类为安全违规。
- 下一步由人工在 LIBERO-Safety 环境运行 Branch Smoke，检查 S₁ 探针确实改变状态、直接 Restore S₁ 的各分量误差、`S0_vs_S1_max_abs_delta > 1e-6`、重复 A 的误差、候选起点对 S₀ 的误差、分支调用者状态与分支后状态误差、分支后状态对 S₁ 的误差和接触记录；全部条件满足时脚本输出 `PASS`。若仍失败，先按输出的最大误差字段/下标判断是否为其他遗漏状态，再继续修复，不放宽容差。
