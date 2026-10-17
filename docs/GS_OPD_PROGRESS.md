Warning: truncated output (original token count: 35761)
Total output lines: 1244

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
| `decision_safety_json`（仅 Safety Collection） | 与各决策索引对齐的 JSON 字符串，包含 A0/Teacher 规则/当前绑定/Compiler 结果/Repair 搜索状态/A+ 和训练标签有效性。 |

## 验证情况

- 已运行 `PYTHONPYCACHEPREFIX=/tmp/rlinf-pycache .venv/bin/python -m py_compile research/gs_opd/data/trajectory_writer.py research/gs_opd/smoke/pi05_rollout.py`，两个文件语法检查通过。
- 本次没有启动 Safety Bridge、没有运行 rollout，也没有生成 trace NPZ 或新 MP4；NPZ 字段布局已在源码定义，实际落盘读取和帧/状态对齐待人工短程验证。
- Safety Bridge 使用 `research/gs_opd/smoke/safety_bridge.py`，监听本机 `127.0.0.1:38567`；Rollout 返回的初始和 step 观测都包含 `agentview_image`。

## Continuous On-Policy Safety Collection

`pi05_rollout.py` 新增可选 `--on-policy-safety-collection --on-policy-safety-out <episode.npz>`。该模式仅面向仓库当前 Bridge 源码固定的 `obstacle_avoidance / level 0 / task 0`。启动前解析 Safety IR v2 和语义消歧结果，并把目标 Body/Geom 名称、Geom ID、gripper 分类及 forbid-contact 规则与活动 Bridge Geometry Inventory 核对；不把 `object_4` 或任何物理绑定默认套用到其他场景。`--object-id` 必须由调用者明确提供。

每个 π₀.₅ 决策后，Rollout 在当前状态创建 Snapshot，完整分支评估 A0，并调用 Contact Compiler。明确 `violation` 时调用既有 `run_verifier_guided_repair()`，候选从同一 Snapshot 独立分支检查；`safe` 或 `unknown` 时跳过 Repair。当前版本会在主环境执行前选择已验证动作：A0 完整验证为 `safe` 时执行 A0；明确 `violation` 且 Repair 的 A+ 完整通过安全、任务进度和动作约束检查时执行 A+。unknown、Repair 失败、动作形状/有限性或安全状态对应关系校验失败、Snapshot 恢复未确认时，记录明确失败原因并在执行当前 Chunk 前终止 Episode。每次实际动作执行后读取 Bridge 新观测，Student 据此生成下一 Chunk。

NPZ 保留 Student 原始 `predicted_actions`，并通过 `executed_actions` 只记录真实发给 Bridge 的动作。`decision_safety_json` 增加 `selected_action_chunk`、`selected_action_source`、`selected_candidate_id`、`execution_selection_valid` 和 `execution_selection_failure_reason`，明确每个决策选中的动作及验证对应关系；A+ 只在实际执行时进入后续 `executed_actions`，未执行候选不会伪装成真实轨迹。其余 Teacher IR、任务指令和对象绑定、A0 Compiler 输出、Repair 状态、候选搜索摘要、`training_label_valid`、失败原因和分支恢复标志继续保存。为控制体积，Repair 摘要保留每个搜索候选的动作、幅度/窗口、Safety/进度/约束结论与失败原因，不重复嵌入候选分支的完整物理状态；A0 Contact Compiler 证据及其完整 Contact Monitoring 仍保留。NPZ 外同时生成同名 `.summary.json`，记录 Episode 内 `safe/violation/unknown` 数量、Repair 调用次数、有效 A+ 数量、有效训练标签数、Repair 状态分布、失败原因和候选拒绝原因。

### 固定任务运行命令

先启动仓库内固定场景的 Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

另一个终端运行完整 On-Policy Episode。下面的 `object_4` 和 JSON 路径是同一已有 Teacher/语义绑定组合；程序仍会对活动物理模型重新核验：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/RLinf/.venv/bin/python \
  research/gs_opd/smoke/pi05_rollout.py \
  --on-policy-safety-collection \
  --max-steps 240 \
  --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
  --object-id object_4 \
  --semantic-result /workspace/zy_workspace/RLinf/logs/gs_opd/vlm_semantic_disambiguation_object_4_raw_outline_ab/semantic_disambiguation.json \
  --on-policy-safety-out /workspace/zy_workspace/RLinf/logs/gs_opd/on_policy_safety_l0t0_episode.npz
```

若当前 IR 中另有已消歧且 Body 已解析的 `target_object`，可增加成对的 `--task-object-id <id> --task-object-semantic-result <json>`，以等长违规前缀的目标物体状态变化提供 Repair 任务进度证据。未提供时，Repair 仍要求可用的正向环境 Reward；仅 EEF 位移不足以生成 A+。`--trace-out` 与该模式互斥，视频可通过既有 `--video-out` 同时保存。运行会真正执行 π₀.₅ 和通过验证的主环境 A0/A+，不应把该命令当成无动作 Smoke。

### 验证与结果统计

- 每个 Episode 的终端统计和 `.summary.json` 包含 `a0_verdict_counts.safe/violation/unknown`、`repair_attempts`、`valid_a_plus_count`、`valid_training_label_count`、`failure_reason_counts` 与 `candidate_rejection_reason_counts`；NPZ 的 `metadata_json` 内也保存相同汇总。
- 新增轻量测试覆盖 safe/unknown 分支、violation 复用 Snapshot、通过验证的 A+ 选择与动作执行字段区分、无有效动作时 fail-closed 终止、主状态恢复失败时必须终止、Episode 计数，以及 NPZ 保存的预测/实际动作对应关系。完成测试/语法检查不代表真实 LIBERO-Safety 场景已成功采集；首次实际运行需人工检查 summary、`decision_safety_json`、`selected_action_source` 和实际 `executed_actions`。
- 本次未启动 Safety Bridge、未执行模型推理或真实 Episode；只做轻量测试与文件检查。

### 已验证动作执行（2026-10-17）

- On-Policy Collection 不再把 Repair 结果仅作为标签后继续执行原始 A0。完整安全 A0 才能作为 A0 执行；A0 明确违规时，只有既有 Repair 输出的完整 A+ 同时满足候选安全、任务进度、动作约束、完整 Chunk 和 Snapshot 恢复检查才会被选中执行。
- 选取前再次校验预测 A0 与安全记录完全对应，A0/A+ 形状一致、全部有限、决策 verdict 与完整分支状态一致。A+ 还需与 Repair 被选中的候选、有效训练监督步数及验证记录完全对应。unknown、Repair 失败、数据无效或恢复失败均在当前 Chunk 执行前终止 Episode，并记录原因。
- `predicted_actions` 始终保存 Student 原始 A0；逐步 `executed_actions` 和 `TrajectoryWriter.add_step()` 只记录实际送入 Bridge 的选中动作；安全记录保存选中动作、来源和候选 ID。真实执行后的新观测继续驱动后续 Student 决策。
- 此次只完成源码与轻量测试，不启动模型、Bridge、仿真或训练；真实环境下 A+ 的动作效果仍待人工短程验证。

## MuJoCo Snapshot / Restore 第一版

### 接口审计与保存范围

- 已审计 LIBERO-Safety 的 `libero/libero/envs/env_wrapper.py`、`libero/libero/envs/bddl_base_domain.py`，以及 vendored robosuite 1.4 的 `utils/binding_utils.py`、`controllers/osc.py`、`controllers/interpolators/linear_interpolator.py`、`robots/single_arm.py` 和 `models/grippers/panda_gripper.py`。
- LIBERO `ControlEnv.get_sim_state()` 使用的 `MjSimState.flatten()` 仅保存 MuJoCo `time/qpos/qvel`，其 `set_state_from_flattened()` 也只恢复这三项；它不足以独立支持确定性回放。
- Safety venv 中 MuJoCo 版本为 `3.3.2`。Bridge 改用 `mujoco.mj_getState` / `mj_setState` 的 `mjSTATE_INTEGRATION`，保存 MuJoCo integration state（物理积分变量、`ctrl`/外力输入、约束激活、warm-start、mocap、userdata、插件 state 和 integration history），恢复后调用 `sim.forward()` 重建派生量。已用最小 MuJoCo 模型验证该 venv 的 get/set API 可用；没有启动完整环境或 Rollout。
- Python 侧同时保存并恢复 robosuite `OSC_POSE` 控制器字段及位置/姿态插值器状态、机器人历史缓冲、Panda 夹爪累计 `current_action`、task 的 `cur_time/timestep/done`、LIBERO 动态物体 motion generator 的内部字段、可变物体显示属性、Observable 的采样状态和 `_obs_cache`，Bridge 最近一次观测/reward/done，以及进程级 `numpy.random` 和 `random` 状态。当前实现针对运行中的同一 Bridge 环境和同一编译模型，通过进程内 Snapshot ID 引用，不写磁盘；再次 Reset 会使现有 Snapshot ID 失效。
- Bridge 原有 `reset` 和 `step` 请求/响应字段保持原样；只有 Smoke 请求显式传 `include_snapshot_state=true` 时，step 响应才额外包含 MuJoCo、机器人关节/末端和场景对象诊断状态。Snapshot/Restore 为新增操作；Restore 返回保存时的 Bridge 观测，并附上状态诊断。

### Snapshot/Restore 派生缓存一致性修复

- 真实 Step 70 检查中，integration state 和机器人关节状态在 `1e-6` 内一致；差异仅落在 `eef_pos` 与黑碗的 `root_pos/root_quat`。审计 vendored robosuite v1.4 `BaseEnv.step()` 确认每个内部物理步先 `sim.forward()` 再 `sim.step()`，且最后一次 `mj_step` 后直接更新 observable 并返回，没有再做一次 forward。因此控制步边界的 `qpos` 已是积分后的值，而 `site_xpos/body_xpos/body_xquat` 可能仍来自最后一次 `mj_forward`；Restore 的 forward 把缓存同步到当前 qpos，于是旧诊断把陈旧缓存与同步后的缓存相比较。它不能单独证明物理状态恢复错误。
- `_set_mujoco_integration_state()` 保留 `mj_setState → sim.forward` 顺序，随后只直接恢复 `qacc_warmstart`，不再整段重写 integration state。这样保留 forward 求解前快照中的 warm-start，同时避免 forward 后重写位置状态的风险。Safety 环境的小型接触模型测试确认 forward 需要恢复的 integration 量是 warm-start，并检查恢复后完整 integration state 仍逐项相同。
- `_snapshot_diagnostics(canonicalize_kinematics=True)` 在临时 `MjData` 上从当前 integration state 执行 forward，再读取 EEF 和物体刚体派生位姿；活动仿真数据不被改写。Bridge Snapshot/Restore 响应及显式 `include_snapshot_state` 诊断使用这一路径。分支内部逐步状态诊断仍读取仿真数据，不插入额外 forward，也不改变动作执行顺序。
- Step 70 Smoke 现在先执行同一 Snapshot 的原地 `Snapshot → Restore`，按原 `1e-6` 阈值比较诊断；通过后再从该 Snapshot 执行两次完全相同的归档 Action Chunk。Smoke 比较两候选的起点、逐步物理状态、动作/reward/done、Contact 事件和监测汇总；要求两个候选均完整执行且子步接触监测完整，并独立核验分支结束后的主状态恢复。任何往返差异、候选差异或恢复失败均不会报告为通过。
- 静态验证通过：`gsopd_teacher` 环境下相关 Teacher/Compiler/Grounding/Replay unittest 50 项通过；LIBERO-Safety 环境下 Snapshot/Restore MuJoCo 小模型和 OSC Bridge unittest 5 项通过；Bridge、Replay Smoke 和测试文件 `py_compile`、`git diff --check` 通过。测试没有连接真实 Safety Bridge。MuJoCo 小模型验证状态设置、派生量同步，以及相同 Snapshot 上相同物理步序列的 state/contact 确定性；归档 Step 70 的 7 维 Action Chunk 双候选一致性和真实场景碰撞仍待手动 Bridge Smoke 验证，不能据此声称真实 LIBERO-Safety Chunk 已一致重放。

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

该 Smoke 保留 Reset 后的短动作 Restore/续跑检查，并报告状态误差；Step 70 的实时归档重放 Smoke 另执行 Snapshot 原地往返和重复分支验证。二者都不是完整 Rollout。

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
| `candidates[].steps[]` | 零起始步索引、实际动作、reward、done、执行后物理状态、兼容保留的控制步接触信息，以及该动作的子步采集覆盖统计。 |
| `candidates[].contact_monitoring` | 候选级子步覆盖状态、各控制步的 expected/captured 数量、contact/no_contact/unavailable 计数和紧凑的 geom/body 接触事件摘要。 |
| `steps[].contacts.status` | `no_contact` 表示 MuJoCo 可查询且 `ncon == 0`；`contact` 表示已枚举一个或多个接触；`unavailable` 表示接触计数或接触条目无法读取。 |
| `steps[].contacts.contacts[]` | 每个接触的 geom/body ID 与名称、接触位置，以及可用时的 6 维 contact-frame 力/力矩。每项以 `force_status` 单独标记力数据 `available` 或 `unavailable`。 |

单个候选失败会在该候选结果中返回 `status: "error"` 和错误信息，Bridge 随后仍会尝试从目标 Snapshot 执行其他候选；分支操作最终仍会尝试恢复调用前的主状态。无效 Snapshot 或候选输入会在环境被修改前返回错误。分支响应不包含相机图像，也不写视频或轨迹文件。

### Branch Smoke

新增 `research/gs_opd/smoke/branch_smoke.py`。测试流程现在严格区分分支起点和调用前主环境状态：Reset → 保存 S₀ → 主环境执行四个非零 7 维动作并保存 S₁ 诊断 → 从 S₀ 执行候选 A、B、A-repeat → 核对主环境仍处于 S₁。脚本还要求 S₀ 与 S₁ 的最大状态差大于 `--atol`（默认 `1e-6`），防止两者意外相同导致恢复断言失去区分力；逐候选起点须与 S₀ 在容差内一致，A 与 A-repeat 的逐步状态/reward/done/contact 结果须一致，分支后主环境须与 S₁ 一致。逐步输出 MuJoCo integration、机器人和物体状态相对候选起点的最大绝对变化、控制步接触状态及子步采集覆盖统计；默认只打印紧凑事件汇总，`--verbose-contacts` 才打印原有控制步 Contact JSON。`no_contact` 与 `unavailable` 明确区分。

### 子步级 Contact Recording

#### 调用链审计

- LIBERO `OffScreenRenderEnv.step()` 将动作交给底层环境，并在物理 step 返回后按需做观测图像噪声处理；LIBERO Safety 的 `BDDLBaseDomain.step()` 在进入 robosuite 控制步前更新 mocap，再调用父类 `step()`，随后执行任务成功/约束检查。
- 当前 vendored robosuite 1.4 的 `robosuite/environments/base.py::MujocoEnv.step()` 按 `int(control_timestep / model_timestep)` 循环：`sim.forward()` → `_pre_action()` → `sim.step()` → `_update_observables()`。本环境的 `robosuite/utils/binding_utils.py::MjSim.step()` 调用 `mujoco.mj_step(model, data)`。因此只在 `env.step()` 返回后读取一次 Contact…23761 tokens truncated…义映射，任务进度绑定从该映射解析。
- [x] 轻量 On-Policy Collection 测试、Python 语法检查和 `git diff --check` 通过；没有启动仿真、模型或 Rollout。
- [ ] 人工运行短程采集并检查 NPZ 标签、绑定及 Episode 汇总。

## On-Policy Preflight：允许的 Gripper 接触规则（2026-10-17）

- 检查实际 `logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt` 后确认：`object_1` 的抓取规则和 `object_3` 的放置规则均为 `gripper contact allow`；另有独立 Object-to-Object 规则 `object_1 contact object_4 forbid`。当前 Compiler 只为 Gripper-to-Object `forbid` 生成 `compiled_rule`，`allow` 返回 `None` 是正常结果。
- Preflight 改为按选中对象的 Gripper 规则决策处理：结构合法的唯一 `allow` 不要求 forbid 专用 `compiled_rule`；`forbid` 仍必须成功编译。未知决策、多个/冲突规则、非单一 gripper 部位或不支持的 phase 继续 fail-closed。Object-to-Object 两侧 Body/Geom 校验没有改变。
- 回归测试使用与当前 L0T0 相同的 `object_1 allow + object_1→object_4 forbid` 规则组合，通过真实 `compile_forbid_contact()` 和绑定校验逻辑验证 Preflight 能通过；另覆盖 Gripper forbid 编译失败、Object-to-Object 绑定不完整时拒绝，以及旧 Gripper-only forbid 正常编译。
- 本次未启动 Safety Bridge、π₀.₅ 或 Rollout。完成相关单测、语法检查和 `git diff --check` 后，已提交并 Push 本修复；实际 80 步运行仍待人工重新执行。

**检查清单（Checklist）**

- [x] 从真实 Teacher 文件确认 Gripper allow 与 Object-to-Object forbid 规则。
- [x] 合法 allow 不再被要求提供 forbid-only `compiled_rule`；forbid 编译失败仍拒绝。
- [x] 绑定校验与其他 unsupported/conflicting 规则的 fail-closed 行为保留。
- [x] 相关单测、语法检查、`git diff --check` 通过；无仿真运行。
- [x] 修复已 Push；等待人工重跑 80 步采集。

## 跨 Action Chunk 的 S70→S75 接触诊断 Smoke（2026-10-17）

- 新增 `research/gs_opd/smoke/cross_chunk_contact_smoke.py`，针对已归档的 80 步 On-Policy Episode。只读检查实际 NPZ 确认决策步为 `0, 5, …, 75`，预测动作偏移覆盖 0～80；使用既有归档加载器按 `decision_step_indices` 和 `predicted_action_offsets` 提取 Step 70、75 的两个 5×7 A⁰，并分别与对应 `executed_actions` 完全相等。没有调用 Student、Teacher 或 SAM 3，也没有重新生成动作。
- Smoke 重置固定 Bridge，逐项重放 Step 0～69；当前任务指令、双相机可用性和 S70 机器人观测必须通过。`--allow-image-mismatch` 只将 S70 的 RGB 像素差异保留为诊断；机器人状态仍按 `1e-6` 比较。随后核对 Teacher 全部 object-contact 规则涉及的双对象语义 Body/Geom 绑定与活动 Geometry Inventory。
- 前置检查通过后，对 S70 创建同一个 Snapshot 并做 Snapshot→Restore 往返检查；一次 `branch_rollout` 从该 Snapshot 独立运行 A=`A⁰₇₀`（5 步）和 B=`A⁰₇₀ || A⁰₇₅`（10 步）。每条分支都要求完整控制步和完整物理子步 Contact Monitoring；复用现有 `compile_forbid_contact()` 和 Safety IR v2 / 语义映射分别计算 verdict。分支返回后另外比较 Bridge 恢复标志和主环境 Snapshot 诊断。
- Smoke 对 A/B 前五步精确核验回传动作，对物理状态、每步 contacts、子步覆盖、Control-step 监控和前缀内首次接触身份/位置/法向按 `1e-6` 比较，并单独比较第五步结束状态。B 的首次 object-to-object violation 索引、物理子步、Body/Geom、世界接触位置/法向和最大可用力均从 Compiler evidence 计算；报告会说明首次违规是否位于零基索引 5～9，不预设任何碰撞步或 verdict。
- 与归档 Step 75 观测的可用字段比较当前覆盖 EEF 位置和夹爪关节位置；Bridge 的 `branch_rollout` 逐步记录不含 RGB 或 EEF 四元数，因此这些字段明确标为 unavailable，不能声称完整视觉观测精确复现。`diagnostic_complete` 表示诊断完整，不表示找到安全动作；B 是原始归档动作拼接，任何 verdict 都不是 Repair 成功证明。
- 对真实归档做了只读文件预检：Step 70/75 分别位于决策索引 14/15；两个预测 Chunk 均为 5×7，且与归档对应已执行动作逐元素相同。Fake Bridge 测试使用真实 Contact Compiler，覆盖完整性、分支前缀、S75 可用状态字段、Compiler 首次违规时间、绑定失败、子步监测不完整和图像差异门槛；7 项 unittest、Python 语法检查通过。没有连接 Bridge、启动仿真或模型，所以这里不报告真实接触结果。

### 手动真实运行命令

终端一启动固定 LIBERO-Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二运行 S70→S75 两分支诊断。此命令不加载 π₀.₅ 或 VLM；移除 `--allow-image-mismatch` 会要求 S70 双相机逐像素一致：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/cross_chunk_contact_smoke.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --allow-image-mismatch \
  --out logs/gs_opd/cross_chunk_contact_s70_s75.json
```

**检查清单（Checklist）**

- [x] 按归档决策索引和 offsets 提取 S70/S75 两个原始 A⁰，并确认它们等于相应已执行动作。
- [x] 新增同一 Snapshot 下 5 步 A 与 10 步 B 的独立分支、完整子步监测和真实 Compiler 判定。
- [x] 检查前五步分支一致性、S75 可用状态字段及分支后主环境恢复；图像豁免不放宽物理状态、绑定、动作、覆盖或 Restore 检查。
- [x] Fake Bridge + 真实 Compiler 的 7 项单测及 Python 语法检查通过；未启动 Bridge、仿真、Student 或 VLM。
- [ ] 人工运行真实 Bridge Smoke，审查 A/B 的实际完整性、Safety verdict 和首次碰撞证据；不得将 B 的结果当成 Repair 成功证明。

## 跨 Chunk 状态一致性诊断修复（2026-10-17）

- 用户提供的真实实验观察为：5 步分支 safe，10 步分支在 control step 5 / physics substep 8 出现 Object-to-Object violation，前五步分支一致且原分支主环境恢复误差为零。本次没有重新运行该仿真，也没有把这些观察写成程序判定条件。
- 在当前 80 步 NPZ 上独立读取 Step 74 的 post-observation 与 Step 75 的 decision-observation：EEF position、EEF quaternion、夹爪位置最大绝对误差均为 `0`；agentview 与 eye-in-hand 图像均逐像素一致（差异像素 `0`）。Smoke 启动 Bridge 前先执行同一离线检查；字段缺失或不一致会停止。
- Snapshot、候选起点、分支后主环境及最后恢复检查现在将 `kinematics_source` 字符串单独列为诊断元数据。仅当两侧该值都是字符串时才从物理状态比较中剥离；数值数组、其他字段和结构仍逐项比较，绝对容差保持 `1e-6`，非有限数值也判为失败。来源字符串不同会记录 `matched=false` 元数据，但不会单独导致物理状态失败。
- 原 Branch 第五步诊断与 NPZ S75 decision observation 是跨接口来源比较。该结果继续完整记录可比较数值及缺失字段（例如 Branch 不提供 EEF quaternion），明确标记为非同源诊断、不作为最终 S75 一致性门槛，也不把它隐藏成成功。真正的严格比较在分支完成并检查主状态后，从原 S70 Snapshot 执行 Restore，再通过普通 `step` 接口执行归档 `A⁰₇₀` 五步：Bridge 环境返回的双相机、EEF position/quaternion、夹爪状态和指令与 NPZ S75 decision 逐项比较；图像必须像素完全相同，状态仍用 `1e-6`，`--allow-image-mismatch` 不作用于这项比较。
- 第五个普通 `step` 同时请求 `include_snapshot_state`；该 MuJoCo/机器人/物体物理诊断与 Branch A 第五步诊断按 `1e-6` 比较，并单独报告运动学来源字符串。无论同源观测、Branch 诊断或其他比较是否失败，清理路径都会再次 Restore S70，并严格核验恢复状态；最终恢复失败会使 Smoke 失败。
- Fake Bridge 回归覆盖来源字符串单独不同、数值差异超出 `1e-6`、S74/S75 归档不吻合、同源环境观测吻合/不吻合，以及最后 Restore 失败；保留真实 Contact Compiler 的 5/10 步检查、跨 Chunk 首次违规证据、物理子步监测覆盖和分支恢复检查。当前相关测试共 11 项，未连接 Bridge、启动仿真或模型。

**检查清单（Checklist）**

- [x] 离线核对归档 Step 74 post 与 Step 75 decision 的三项机器人状态和两路 RGB；当前文件五项均精确一致。
- [x] 仅把字符串型 `kinematics_source` 从物理值比较中分离；其他差异仍受 `1e-6` 严格门控。
- [x] 增加 Restore→普通 step 五步→比较→最终 Restore 路径，最终 Restore 在比较失败时仍执行。
- [x] 新增来源元数据、同源状态匹配/不匹配和最终恢复失败测试；本次仅做静态测试，不启动 Bridge 或仿真。
- [ ] 人工运行下方原 S70→S75 Smoke 命令，检查新的同源比较与最终 S70 恢复结果。

## Near–Far 跨 Chunk 间距预警 Smoke（2026-10-17）

- `safety_bridge.py` 新增只读 `geom_set_min_distance` 操作。调用方传入当前模型中的两组 Geom ID，Bridge 将 MuJoCo `mjSTATE_INTEGRATION` 复制到新建的 scratch `MjData`，执行 `mj_forward`，再对 Geom 笛卡尔积调用 `mujoco.mj_geomDistance`；不会在真实 `env.sim.data` 上执行 forward 或改写仿真状态。结果包含最小有符号距离、对应 Geom/Body、最近点线段、查询覆盖、`distmax_m` 截断、异常和活动积分状态往返检查。
- 每组 Geom 必须非空、组内 ID 唯一、两组互不相交且在当前模型范围内。任一组合达到距离上界、查询出错、距离/最近点非有限或活动积分状态发生变化时，结果不完整，Smoke 将距离标为 unavailable。Mesh 组合会设置近似标记并附说明；`mj_geomDistance` 结果只用于间距趋势探索，不是碰撞安全证明。
- 新增 `research/gs_opd/smoke/near_far_gap_smoke.py`。它使用既有归档加载器校验并提取 S65/S70 决策前缀与 S70 的五步预测 Chunk，Reset 后重放实际历史动作，在 S65、S70、S71–S75 查询由当前 Geometry Inventory 和 `semantic_bindings_l0t0.json` 解析出的 `object_1` / `object_4` 当前 Body 全部 Geom 组合。Teacher 需恰有一条这两个对象间的 `forbid contact` 规则；两侧绑定身份必须与当前库存一致，程序不硬编码 Geom ID。
- S70 风险预测仅用 S70 间距及 S70 的 A⁰ 五步分支轨迹；不会提取或使用 A⁰₇₅ 生成预测。`decision_safety_json` 中 S75 的历史违规只在 S70 预测冻结后作为事后真值单独校验，并要求其完整 Chunk、全部物理子步覆盖、规则、Body/Geom ID 与当前库存一致。5、10、20 mm 三个阈值固定写入程序；报告各阈值 S70→S75 触发情况、首次可知触发步、相邻间距变化、持续接近定义及到历史首碰撞的控制步/物理子步提前量。RGB 只作差异诊断；S65、S70、S75 指令与机器人状态仍以 `1e-6` 严格核验。
- 增加 9 项轻量测试：固定阈值与事后提前量、距离不可用 fail-closed、RGB 仅诊断、S75 物理绑定及历史接触覆盖校验，以及真实 MuJoCo 小球模型的 scratch 距离、状态不变、`distmax` 截断、非法 Geom、非有限结果和 API 缺失。语法检查和 `git diff --check` 通过。没有连接 Safety Bridge、重放归档动作或启动模型；尚无实际 LIBERO 场景的距离、趋势或提前预警结论。

### 手动运行命令

终端一启动 LIBERO-Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二重放 S65→S70，再仅用 A⁰₇₀ 查询至 S75，并将 JSON 保存至指定路径：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/near_far_gap_smoke.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --out logs/gs_opd/near_far_gap_s65_s75.json
```

**检查清单（Checklist）**

- [x] Bridge 距离查询在 scratch `MjData` 上计算；无效、截断和 Mesh 近似显式标注，不输出安全证明。
- [x] Smoke 使用归档 A⁰₇₀ 和当前 Geometry Inventory 校验双对象绑定；5/10/20 mm 阈值固定，S75 历史违规与 S70 预测隔离。
- [x] 9 项 unittest、Python 语法检查和 `git diff --check` 通过；没有启动仿真或模型。
- [ ] 人工运行 Bridge Smoke，核对 S65/S70/S75 状态一致性、间距查询覆盖和真实阈值触发结果。

## S70 端点几何引导的提前局部修正可行性 Smoke（2026-10-17）

- 新增 `research/gs_opd/smoke/early_repair_s70_smoke.py`。Smoke 只用归档加载器读取 S70 决策、此前 70 个真实执行动作及 A⁰₇₀ 五步 Chunk；不提取 A⁰₇₅，也不调用 Student、Teacher 推理或 `run_verifier_guided_repair()`。先重置并重放到 S70，严格核验 instruction/机器人状态，再创建同一 S70 Snapshot。
- 先通过真实 `compile_forbid_contact()` 和完整 5 步 Contact Monitoring 确认 A⁰₇₀ 为 safe。随后从 Snapshot 顺序执行 A⁰₇₀，仅为获取 S75 端点的最近 Geom witness；用已验证 Body 身份确定 witness 方向（移动物体最近点减障碍物最近点）及两侧横向方向。不会伪造违规/碰撞事件。最近点或方向缺失、距离查询不完整时直接 fail closed。
- 候选生成复用 `build_repair_actions()`、`_candidate_constraints()` 和 OSC `repair_context`。首版使用 3 个几何方向、2 个平滑窗口和 4 个小幅度组合，最多 24 条；CLI `--max-candidates` 上限为 36。候选只改已反归一化控制器尺度上的 XYZ 输入，姿态/夹爪保持 A⁰ 原值；检查 OSC 6 维输入范围、平滑修正、输出动作、活跃 Workspace 和关节限制。
- 所有可执行候选通过单次 `branch_rollout` 从相同 S70 Snapshot 独立执行完整五步；Contact Compiler 必须判 safe，物理子步监测必须 complete。对分支端点再从 S70 Snapshot 经普通 `step` 重放，要求与 Branch 端点诊断在 `1e-6` 内一致后才查询间隙；每次重放及整个 Smoke 结束都执行 Restore 并核验 S70 主状态。
- 任务进度复用 `evaluate_task_progress()`，比较 A⁰ 与候选相同的完整 5 步，不使用空前缀，也不把 EEF 位移单独当任务成功。首版依赖已验证 `target_object` Body 的沿 A⁰ 搬运方向位移代理；无可比物体运动、短 Chunk、Reward/物体证据不足均拒绝候选，并保留代理局限。候选 S75 最小间隙还必须比 A⁰ 增加超过 `1e-6 m`；余量阈值计数只用于探索统计。只有物理安全、完整覆盖、任务进度、动作约束、间隙提升与 Restore 全部通过时，候选才计为 `accepted_for_probe`；它仍不是最终 A⁺，`training_label_valid` 固定为 false。
- JSON 逐候选记录动作、几何方向、幅度/窗口、完整接触监测摘要、Compiler verdict、同五步任务进度、动作/Workspace 约束、S75 间隙和拒绝原因；另分别统计 5/10/20 mm 余量的已测候选数与安全且保留进度候选数。几何余量是探索指标，不证明碰撞安全或未来 Chunk 安全。
- 新增 12 项纯离线单测，覆盖最近点顺序/反向、平滑候选数量上限、控制范围、姿态与夹爪不变、同五步进度及其失败、要求候选间隙实际增加、间隙计数、安全/监测 fail-closed、所有探测门槛通过但不生成训练标签，以及 step 失败时仍执行 Snapshot Restore。已运行测试和语法检查；没有启动 Bridge、仿真或模型，尚未得出 S70 实际修正可行性结果。

### 端点物理状态比较门槛修复

- `replay_and_measure_endpoint()` 仍完整保存 Branch 与顺序 `step(include_snapshot_state=True)` 的全部诊断比较结果；其中 EEF/物体派生运动学缓存差异现在明确标为诊断信息，不再阻断终点距离查询。
- 是否允许查询距离改为严格比较双方完整 `mujoco_integration_state` 向量：必须均存在、均为一维数值数组、形状一致、值均有限，且 `rtol=0`、`atol=1e-6`。任何缺失、形状错误、非有限值或积分状态超差仍拒绝查询；完整诊断 JSON 和主环境 Snapshot 恢复核验保留。
- 新增 2 项回归测试：积分状态完全一致而 EEF 缓存相差 `0.331 mm` 时仍查询端点距离；积分状态差 `2e-6` 时拒绝查询并仍完成最终 Snapshot 恢复。未放宽动作、Compiler、任务进度、间隙提升或恢复验收。

### 手动运行命令

终端一启动 LIBERO-Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二运行最多 24 条（不超过 36 条）S70 几何引导探测；报告默认写入指定 JSON：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/early_repair_s70_smoke.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --max-candidates 36 \
  --out logs/gs_opd/early_repair_s70.json
```

**检查清单（Checklist）**

- [x] S70 A⁰ 必须经真实 Compiler 判 safe；候选方向来自 A⁰₇₀ 实际五步端点的最近 Geom 位置，不构造虚假碰撞证据。
- [x] 所有候选都受 36 条硬上限、同一 Snapshot Branch、完整 Contact Monitoring、Compiler、同五步搬运进度、OSC/Workspace/关节和间隙 Restore 检查约束。
- [x] 12 项轻量单测、Python 语法检查和 `git diff --check` 通过；不产生 A⁺ 训练标签，未启动仿真或模型。
- [ ] 人工运行 Bridge Smoke 并审查逐候选 JSON；不把几何余量或五步探测通过外推为未来安全。

## S70→S75′ Student 闭环前瞻验证 Smoke（2026-10-17）

- 新增 `research/gs_opd/smoke/early_repair_student_lookahead_smoke.py`。它读取已有 `logs/gs_opd/early_repair_s70.json`，只选择 `accepted_for_probe=true` 且保存的 `endpoint_gap_m >= 0.010` 的修正候选，按文件顺序最多选 2 条；候选不足时不降低阈值，并明确设置 `no_eligible_repair_candidates`。无合格修正候选时仍可单独评估原始 A⁰ 路线。
- 从归档重放前 70 个实际动作，严格核对任务指令与 S70 机器人状态，校验 Teacher 两个对象绑定与当前 Geometry Inventory，再建立唯一 S70 Snapshot。A⁰ 与所选修正候选在同一次 `branch_rollout` 请求中从该 Snapshot 各自执行五步；每条路线必须完整执行、Contact Monitoring 子步覆盖完整、Contact Compiler verdict 为 safe、当前控制器动作范围有效，并通过与 A⁰ 同五步的目标物体任务进度比较，才会进入 S75′ 的真实观测和 Student 推理阶段。
- 每条通过当前五步门槛的路线都会先恢复并严格核验 S70，再在主 Bridge 环境逐步执行该路线的五个动作，以获取各自真实的 S75/S75′ RGB、机器人状态、reward/done 和最终 Snapshot。分支端点与普通 `step(include_snapshot_state=true)` 的完整诊断均保留；物理一致性以完整 `mujoco_integration_state` 按 `1e-6` 比较，派生运动学缓存差异单独诊断。随后从该路线的 S75 Snapshot 重新调用 π₀.₅ 预测五步，并通过另一个 `branch_rollout` 完整记录物理子步接触、调用同一 Teacher/Compiler 判定，最后核验 S75 分支恢复和 S70 主状态恢复。
- Student 配置与 `pi05_rollout.py` 对齐：默认 checkpoint `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft`、`libero_spatial_openpi_pi05_smoke` Hydra 配置、`get_model(...).to("cuda:0").eval()`、`openpi.torch_compile=false`、`predict_action_batch(mode="eval")`；不导入会立即加载模型/启动 Rollout 的 `pi05_rollout.py`。图像使用 `get_libero_image()` 与 `get_libero_wrist_image()` 旋转 180°，状态编码为 EEF position + `quat2axisangle(EEF quaternion)` + gripper qpos。预测结果必须是有限 `(5,7)` 动作且通过当前 OSC 输入范围检查。
- JSON 逐路线记录 S70 五步安全、Compiler 与首次违规证据、任务进度、S75 真实 RGB 尺寸与状态、黑碗/香水瓶绑定间隙、该状态重新预测的完整 Student 动作、后续五步的完整物理子步覆盖、Safety verdict、首次违规控制步/物理子步、相对 A⁰ Student 路线的同五步任务进度诊断、异常及 Snapshot 恢复结果。数据只写报告 JSON，不写入 NPZ 或 A⁺ 训练集；几何间隙不是安全证明。
- 新增 7 项纯轻量单测，覆盖候选 `>=10mm` 选择及最多 2 条、无候选时不放宽阈值、坏动作拒绝、当前 Chunk fail-closed 门槛、Student `(5,7)`/有限值校验和违规时间映射。单测、Python 语法检查与 `git diff --check` 通过；未启动 Bridge、MuJoCo 仿真或 π₀.₅ 模型，因此尚无实际 S75′ 安全结论。

### 手动运行命令

终端一启动 LIBERO-Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二运行 S70→S75′ Student 前瞻验证（会加载 π₀.₅ 并运行有限的 Bridge 分支仿真）：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf:/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/early_repair_student_lookahead_smoke.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz \
  --early-repair-json logs/gs_opd/early_repair_s70.json \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --checkpoint /workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft \
  --out logs/gs_opd/early_repair_student_lookahead.json
```

**检查清单（Checklist）**

- [x] 候选选择严格使用既有 `accepted_for_probe` 和 `endpoint_gap_m >= 10mm`，最多 2 条；原始 A⁰ 路线始终独立纳入。
- [x] S70 当前 Chunk 与每条 S75′ Student Chunk 均走现有 Snapshot、Branch、完整 Contact Monitor 和 Teacher Contact Compiler；后续 Student 动作由该路线真实 S75 观测重新推理。
- [x] 7 项轻量单测、语法检查和 `git diff --check` 通过；未启动模型或仿真，未生成训练标签。
- [ ] 人工运行 Smoke，检查各路线当前 Chunk 门槛、S75′ 新预测、后续 Contact verdict/首次违规时间及最终 S70 恢复。
