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

每个 π₀.₅ 决策后，Rollout 在当前状态创建 Snapshot，完整分支评估 A0，并调用 Contact Compiler。明确 `violation` 时调用既有 `run_verifier_guided_repair()`，候选从同一 Snapshot 独立分支检查；`safe` 或 `unknown` 时跳过 Repair。当前版本在主环境执行前选择已验证动作：A0 完整验证为 `safe` 时执行 A0；A0 明确违规且 Repair 的 A+ 完整通过安全、任务进度和动作约束检查时执行 A+。unknown、Repair 失败、动作数据无效、选择动作与验证结果不对应或 Snapshot 恢复未确认时，记录原因并在当前 Chunk 前终止 Episode。每个实际动作执行后的新观测继续输入 Student 生成下一 Chunk。

NPZ 保留 Student 原始 `predicted_actions`；`executed_actions` 和逐步 post-observation 只记录真实发送给 Bridge 的选中动作。`decision_safety_json` 保存每个决策的 Teacher IR、任务指令和对象绑定、A0 Compiler 输出、Repair 状态、候选搜索摘要、A+（若有效）、`training_action_chunk`、`training_label_valid`、选中的 `selected_action_chunk`、`selected_action_source`、`selected_candidate_id`、执行选择有效性、失败原因和分支恢复标志。未执行候选不会伪装成实际轨迹。为控制体积，Repair 摘要保留每个搜索候选的动作、幅度/窗口、Safety/进度/约束结论与失败原因，不重复嵌入候选分支的完整物理状态；A0 Contact Compiler 证据及其完整 Contact Monitoring 仍保留。NPZ 外同时生成同名 `.summary.json`，记录 Episode 内 `safe/violation/unknown` 数量、Repair 调用次数、有效 A+ 数量、有效训练标签数、Repair 状态分布、失败原因和候选拒绝原因。

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

若当前 IR 中另有已消歧且 Body 已解析的 `target_object`，可增加成对的 `--task-object-id <id> --task-object-semantic-result <json>`，以等长违规前缀的目标物体状态变化提供 Repair 任务进度证据。未提供时，Repair 仍要求可用的正向环境 Reward；仅 EEF 位移不足以生成 A+。`--trace-out` 与该模式互斥，视频可通过既有 `--video-out` 同时保存。运行会真正执行 π₀.₅ 和主环境 Student A0，不要把该命令当作无动作 Smoke。

### 验证与结果统计

- 每个 Episode 的终端统计和 `.summary.json` 包含 `a0_verdict_counts.safe/violation/unknown`、`repair_attempts`、`valid_a_plus_count`、`valid_training_label_count`、`failure_reason_counts` 与 `candidate_rejection_reason_counts`；NPZ 的 `metadata_json` 内也保存相同汇总。
- 新增轻量测试覆盖安全 A0 与有效 A+ 选择、unknown/Repair 失败/恢复失败/无效动作均停止执行、Episode 终止原因分类，以及 NPZ 中预测 A0 与实际 A+ 的区分。完成测试/语法检查不代表真实 LIBERO-Safety 场景已成功采集；人工运行需检查 summary、`decision_safety_json`、选中来源和实际 `executed_actions`。
- 本次未启动 Safety Bridge、未执行模型推理或真实 Episode；只做轻量测试与文件检查。

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
- 当前 vendored robosuite 1.4 的 `robosuite/environments/base.py::MujocoEnv.step()` 按 `int(control_timestep / model_timestep)` 循环：`sim.forward()` → `_pre_action()` → `sim.step()` → `_update_observables()`。本环境的 `robosuite/utils/binding_utils.py::MjSim.step()` 调用 `mujoco.mj_step(model, data)`。因此只在 `env.step()` 返回后读取一次 Contact 会遗漏中间子步曾出现、随后消失的接触。
- 本实现只在每个 `branch_rollout` 候选调用 `env.step(action)` 的时间窗口内，临时遮蔽当前 `MjSim` **实例**的 `step` 属性。包装器原样调用原绑定方法一次，在该次 `mj_step` 完成后立即读取 `data.ncon/contact` 和可用的 `mj_contactForce`，随后 robosuite 的 `_update_observables()` 与下一子步继续原有顺序。没有拆分控制动作、提前调用 `forward`、更改模型参数或全局替换 MuJoCo 函数；每个控制步 `finally` 恢复实例原有 `step` 属性。
- 机器人 body 映射从 `env.robots[*].robot_model.root_body` 解析编译模型 body ID，再沿模型 `body_parentid` 结构收集其后代（包含机器人层级中的 mount/gripper）。Contact 双方以 `geom_bodyid` 查对应 body 集合并分为 `robot_robot`、`robot_scene`、`scene_scene`；如果映射不可用，分类为 `unavailable`，不使用 geom 名称前缀猜测。

#### 新增结果字段与覆盖语义

`candidates[].steps[]` 保留原有 `state`、`reward`、`done` 和控制步后的 `contacts`，新增 `substep_contact_coverage`。候选级 `contact_monitoring` 汇总每个控制步采集数量、子步样本状态及接触事件。事件按无序 geom ID 对聚合：一次 `occurrence_count` 表示该 geom 对在一个物理子步中出现过（同一子步多个接触点不会重复计 occurrence）；另累计接触点数、首次控制步/该控制步内物理子步/候选全局子步、geom/body ID 与名称，以及有数据时最大的接触坐标系力向量前三项的欧氏范数。没有力样本时 `max_force_norm` 为 `null`，并保留力样本可用/不可用计数。

覆盖状态按每个动作预期子步数 `int(control_timestep / model_timestep)` 与实际包装到的 `sim.step()` 次数核对：

- `complete`：该候选所有已执行控制步均采到预期数量，且所有 Contact 查询成功。若事件列表为空，这只表示完整采样范围内未观测到接触。
- `incomplete`：只捕获到部分子步、预期步数不可读，或有一个及以上子步 Contact 查询失败。事件列表仍可能包含已观测事实，但不能据此断言没有其他接触。
- `unavailable`：没有可用的子步接触样本，例如无法安装实例级观察器，或候选在开始物理步前失败。

Smoke 默认逐控制步打印 `contact/no_contact/unavailable` 计数、`captured/expected` 和候选事件摘要；不会逐子步输出原始 Contact JSON。除原有 Snapshot/Restore 与分支一致性条件外，`PASS` 还要求 A、B、A-repeat 三个候选的子步 Contact 覆盖状态均为 `complete`、机器人 body 映射可用，且 A 与 A-repeat 的候选事件汇总一致；若采集或分类不可用则输出 `FAIL`。事件只记录物理事实，不代表安全/危险判定。

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
- 本次子步 Contact 改动同样只做源码审计和语法检查，没有在 LIBERO-Safety 仿真实例中运行 Branch Smoke；运行时观察器是否成功安装、预期物理子步数是否匹配，以及实际 Contact/力数据可用性仍须按 Smoke 输出确认。
- Restore 诊断已扩展：分别输出 `mujoco_integration_state`、`robot_states`、`object_states` 的最大绝对误差及对应字段/数组下标；新增 S₁ 上执行非零探针动作后直接 Restore S₁ 的对照，另比较分支结束状态与分支调用前状态，从而区分普通 Snapshot/Restore 与 Branching 的恢复误差。
- 根因证据与最小修复：原 `_set_mujoco_integration_state()` 调用 `mj_setState()` 后执行 `sim.forward()`，而 forward 求解约束时可能更新 `mjSTATE_INTEGRATION` 包含的 `qacc_warmstart`。在 LIBERO-Safety MuJoCo 3.3.2 环境用一个带接触的最小 MuJoCo 模型复现：接触数为 1，模型 `nq=7`、`nv=6`，integration state 中 warm-start 段从 index 14 开始；执行 `mj_setState` + `mj_forward` 后 index 16（`qacc_warmstart[2]`）从 `8.056839445962867e-10` 变为 `7.197303360572629e-10`。再次 `mj_setState` 可精确恢复保存的向量。Bridge 现在在 `sim.forward()` 后再次应用原始 integration state，保留 forward 计算的派生量并恢复求解器 warm-start 等状态。原 `_restore_snapshot()` 中控制器字段恢复只更新 Python 控制器/缓冲对象，不执行仿真步或写 MuJoCo 状态，因此没有证据表明它会覆盖该物理状态。
- 用户报告的 `0.6662` 尚未通过增强版 Smoke 定位到具体字段；上面的 API 探针证实了 Restore 路径存在 integration state 被 forward 改写的问题，但该数值是否完全来自该分量，需由 Smoke 新增的逐分量/索引输出确认。未运行完整 Bridge Smoke，也不据此宣称端到端已通过。
- Snapshot/分支仍只支持当前 Bridge 进程中的同一编译 MuJoCo 模型；Reset 会清除旧 Snapshot。已知 Snapshot 尚不能保证覆盖自定义控制器/传感器闭包私有 RNG、插件进程外状态、渲染器/GPU 缓存和第三方任务任意外部副作用，见上文 Snapshot 限制。
- 逐步对象诊断复用现有 `_snapshot_diagnostics()`：列出 BDDL `objects_dict` / `fixtures_dict` 中具有 `root_body` 的对象根位姿、四元数及对象关节位置；未注册对象不会有单独命名条目，但其 MuJoCo integration state 仍包含在完整物理状态向量中。
- MuJoCo 返回的力/力矩是接触坐标系下的六维值；没有约束力地址或 API 调用失败时，该接触几何信息仍保留，但 `force_status` 为 `unavailable`。本功能只记录物理事实，不把接触分类为安全违规。
- 子步采样覆盖当前审计到的 vendored robosuite `MjSim.step()` 路径。若后续 robosuite 版本改用其他集成路径、C 层内部执行多个不可见子步，或在 `mj_step` 后立即清除/重建 contact buffer，当前观察器可能无法覆盖；expected/captured 数量不匹配时会报告 `incomplete`，Contact 查询失败时报告 `unavailable`，空事件列表不能被解释为“无接触”或安全结论。力值只在 `mj_contactForce` 可读时汇总，采用 contact frame 的法向/切向力三分量范数。
- 下一步由人工在 LIBERO-Safety 环境运行 Branch Smoke，检查 S₁ 探针确实改变状态、直接 Restore S₁ 的各分量误差、`S0_vs_S1_max_abs_delta > 1e-6`、重复 A 的误差、候选起点对 S₀ 的误差、分支调用者状态与分支后状态误差、分支后状态对 S₁ 的误差和接触记录；全部条件满足时脚本输出 `PASS`。若仍失败，先按输出的最大误差字段/下标判断是否为其他遗漏状态，再继续修复，不放宽容差。

## VLM Safety Teacher 第一版

### 可复用接口审计

- 仓库已有 `VLMRewardModel`，通过 Transformers 加载本地视觉语言模型并执行生成；该实现耦合 RLinf reward worker、Torch、模型 Processor 和 reward parser，不适合作为 Safety Bridge 独立 Smoke 的直接调用入口。
- `EmbodiedAPIRewardWorker` 已将图像编码为 data URL，并通过 `InferenceHTTPClient.chat_completion()` 调用 OpenAI-compatible `/v1/chat/completions`。`InferenceHTTPClient` 的同步请求基于 requests、异步请求基于 aiohttp，当前接口没有 API Key 配置参数。
- 第一版沿用已存在的多模态 Chat Completions 消息结构（文本 + `image_url`），提供可替换的 `TeacherBackend` Protocol 和 `OpenAICompatibleBackend`。为避免 Safety Python 环境额外依赖 RLinf HTTP client 的 requests/aiohttp，Teacher Backend 使用 Python 标准库完成 HTTP 调用及 RGB PNG data URL 编码，不新增依赖。
- 当前 shell 的 `GS_OPD_TEACHER_BACKEND`、`GS_OPD_TEACHER_BASE_URL`、`GS_OPD_TEACHER_MODEL` 和 `GS_OPD_TEACHER_API_KEY` 均未配置。本次没有调用 VLM，也没有生成或填写任何模拟 Teacher 结果。

### 模块与输出 Schema

新增目录 `research/gs_opd/teacher/`：

| 文件 | 职责 |
|---|---|
| `schema.py` | JSON Schema 定义、枚举约束、重复键/非有限值拒绝、字段校验和错误类型。 |
| `prompt.py` | 结构化 system/user prompt，明确事实、推断、不确定性和物理 ID 禁止项。 |
| `backend.py` | `TeacherBackend` 替换接口、真实 OpenAI-compatible 多模态 HTTP Backend 和环境变量配置。 |
| `teacher.py` | 调用 Backend 并只返回通过 Schema 校验的输出。 |
| `smoke.py` | 从 Safety Bridge Reset 读取双相机 RGB 与 Task Instruction，调用配置 Backend 并打印已验证输出。 |
| `__init__.py` | 导出公开 Schema、Backend 与 Teacher API。 |

Schema 版本为 `gs_opd.safety_teacher.v1`。顶层字段为 `schema_version`、`objects`、`global_uncertainties`。每个物体包含：

| 字段 | 约束与含义 |
|---|---|
| `name`、`description` | 非空语义名称和视觉描述；不用于物理模型绑定。 |
| `role` | `target_object`、`placement_target`、`avoid_object`、`tool`、`obstacle`、`other` 或 `unknown`。 |
| `visibility` | `visible`、`partially_visible`、`not_visible` 或 `uncertain`。 |
| `contact_rule` | `allowed`、`forbidden` 或 `unknown`；Prompt 要求只有任务语义明确支持时才给 allowed/forbidden。 |
| `robot_parts` | `gripper`、`left_finger`、`right_finger`、`end_effector`、`wrist`、`arm`、`base`、`other`、`unknown` 中的一项或多项。 |
| `reason` | 简短说明角色、接触规则及涉及部位的理由。 |
| `visual_facts` | 可直接从图像确认的陈述，每条标注 `agentview_image` 或 `robot0_eye_in_hand_image`。 |
| `inferences` | 结合图像和任务指令得到的推断；每项包含 `statement` 与 `[0, 1]` 的 `confidence`。 |
| `uncertainties` | 该物体身份、可见性或语义规则尚未确定的事项。 |

输出结构不包含 geom/body ID、坐标或视觉到物理的绑定字段；所有对象和嵌套字段均禁止额外属性，解析器还拒绝文本中的 geom/body ID 标记。JSON 解析拒绝重复键、NaN/Infinity、Markdown 包裹和字段/枚举错误；遇到错误直接抛出可定位字段路径的 `TeacherOutputError`，不会尝试修补或伪造输出。`SAFETY_OUTPUT_JSON_SCHEMA` 可供后续服务端或数据管线复用，当前运行时字段验证不依赖 `jsonschema`。

### Backend 配置

使用支持多模态 OpenAI Chat Completions 的真实服务（例如已启动的 vLLM/SGLang OpenAI-compatible endpoint），配置：

| 环境变量 | 必需 | 默认值 | 作用 |
|---|---:|---|---|
| `GS_OPD_TEACHER_BACKEND` | 是 | 无 | 当前支持 `openai_compatible`。 |
| `GS_OPD_TEACHER_BASE_URL` | 是 | 无 | 服务 Base URL，如 `http://127.0.0.1:8000/v1`；代码请求其 `/chat/completions`。 |
| `GS_OPD_TEACHER_MODEL` | 是 | 无 | 服务端已加载的多模态模型 ID/别名。 |
| `GS_OPD_TEACHER_API_KEY` | 否 | 无 | 需要 Bearer 鉴权时设置；不需要鉴权的本地服务可留空。 |
| `GS_OPD_TEACHER_TIMEOUT_S` | 否 | `90` | 单次请求超时秒数，必须为正数。 |
| `GS_OPD_TEACHER_MAX_TOKENS` | 否 | `1800` | 最大生成 token 数，必须为正整数。 |
| `GS_OPD_TEACHER_TEMPERATURE` | 否 | `0` | 采样温度，范围 `[0, 2]`。 |
| `GS_OPD_TEACHER_JSON_MODE` | 否 | `0` | 设为 `1` 时额外请求 `response_format=json_object`；仅对支持该参数的服务启用。 |

自定义本地 Backend 可实现 `TeacherBackend.generate(instruction, images) -> str`，并将实例传给 `SafetyTeacher`；不要在 Backend 失败时回退到静态示例结果。

### Safety Bridge Teacher Smoke

终端 A 启动已配置 LIBERO-Safety 环境的 Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端 B 配置一个实际运行中的 VLM 服务，并读取当前 Bridge Reset 观测：

```bash
cd /workspace/zy_workspace/RLinf
export GS_OPD_TEACHER_BACKEND=openai_compatible
export GS_OPD_TEACHER_BASE_URL=http://127.0.0.1:8000/v1
export GS_OPD_TEACHER_MODEL=your-served-vlm-model
# 如服务要求鉴权，再设置实际的 key：
# export GS_OPD_TEACHER_API_KEY=...

PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  -m research.gs_opd.teacher.smoke --host 127.0.0.1 --port 38567
```

Smoke 只 Reset 并读取 `agentview_image`、`robot0_eye_in_hand_image` 和 `instruction`，不会执行机器人动作。缺少 Backend 配置时退出码为 `2` 并列出必需变量；Bridge、HTTP 或 Schema 错误退出码为 `1`；只有真实服务返回并通过校验后才打印 Teacher JSON。

### 验证状态与下一步

- 本次源码语法检查通过；没有启动 Safety Bridge、没有连接真实 VLM 服务，也没有执行模型推理 Smoke。
- 当前环境尚缺 Backend 名称、服务 URL 和模型 ID；需提供一个已运行、可接收两张 `image_url` 图像的 OpenAI-compatible VLM endpoint，并按上表设置配置。若服务要求鉴权，还需配置 API Key。
- 尚未验证不同 vLLM/SGLang 版本对双图输入、`response_format` 参数和该 JSON Schema Prompt 的兼容性，也未评估模型输出的语义准确率、置信度校准或接触规则质量。下一步先配置真实服务运行 Teacher Smoke，人工审核对象角色、事实/推断边界和 `unknown` 使用情况，再考虑向后续 Compiler 提供输入；当前没有实现视觉分割、物理绑定、Compiler、Repair 或训练逻辑。

### Teacher Smoke 原始响应保存与请求耗时（2026-10-15）

- `research/gs_opd/teacher/smoke.py` 新增可选 `--raw-out <path.json>`。Smoke 只调用一次 `backend.generate()`；成功收到响应后，先按 UTF-8 原文写入指定路径，再调用现有 `parse_teacher_output()`。因此 Schema 校验失败时仍保留完整模型文本。路径的父目录会自动创建；写入失败会明确报告并以失败状态退出。
- 使用示例：在现有 Teacher Smoke 命令后添加 `--raw-out /tmp/teacher_raw.json`。未提供参数时，不写文件，保留原有输出行为。
- 终端会输出单次 `generate()` 调用耗时（秒），请求异常时也会输出已耗时间；该计时不包含文件写入与 Schema 校验。
- 本次只做 Python 语法检查，未连接 Safety Bridge、未调用真实 VLM；实际原始响应保存和耗时输出待配置服务后验证。

### Teacher Smoke 双相机采集（2026-10-15）

- `research/gs_opd/teacher/smoke.py` 新增 `--images-out-dir <目录>`：保存 `agentview_image.png` 与 `robot0_eye_in_hand_image.png`，并写入 `capture_metadata.json`，其中包含原始任务指令、RGB/uint8 信息及各图像的宽高和文件名。
- PNG 通过 Backend 已有 RGB 编码函数从 Bridge 返回的原始帧编码；没有旋转、裁剪或额外图像预处理。未指定输出目录时不保存图像，现有 `--raw-out`、请求计时及默认 VLM Smoke 路径保持可用。
- 新增 `--capture-only`；该模式要求同时指定 `--images-out-dir`，读取一次 Bridge Reset 并保存图像/元数据后退出，不创建 VLM Backend、不调用模型或执行 Schema 校验。示例：`python -m research.gs_opd.teacher.smoke --images-out-dir /tmp/gs_opd_capture --capture-only`。
- 本次仅计划运行 Python 语法检查与 `git diff --check`；不启动 Bridge、不运行 Smoke、不连接真实模型。图像文件的运行时生成与 RGB 像素一致性仍待后续人工运行验证。

## GS-OPD Safety IR v2 Schema / Validator

- 新增 `research/gs_opd/teacher/schema_v2.py`，单独定义 v2 JSON Schema 常量 `SAFETY_OUTPUT_V2_JSON_SCHEMA`、`validate_teacher_output_v2()`、`parse_teacher_output_v2()` 和 `TeacherOutputV2Error`；不导入或修改 v1 Validator，也未接入现有 Prompt、Backend、Smoke 或 Bridge。
- v2 强制顶层 `schema_version`, `objects`, `rules`, `global_uncertainties`。Object 字段为 `id,name,role,visibility,mobility,visual_facts,inferences,uncertainties`；Rule 字段为 `id,subject_parts,target_id,predicate,decision,phase,reason`。所有层级均拒绝额外字段；Validator 检查非空文本、对象和规则 ID 各自唯一、规则目标引用、合法机器人部位、枚举、`[0,1]` 有限 confidence，以及文本中显式 MuJoCo geom/body ID。
- v1 相机、role、visibility 和 robot-part 枚举在新模块中独立定义并保持同值。解析器拒绝重复 JSON 键和 NaN/Infinity。`unknown` 保持原值；Schema 不按 role 合成规则。
- 新增 `research/gs_opd/teacher/test_schema_v2.py`，只依赖 Python 标准库 `unittest`。仓库 Python 执行 `python -m unittest -v research.gs_opd.teacher.test_schema_v2`：10 项通过；`python -m py_compile research/gs_opd/teacher/schema_v2.py research/gs_opd/teacher/test_schema_v2.py` 与 `git diff --check` 均通过。
- 本次仅验证离线 Schema/Validator；没有调用 VLM、MuJoCo，也未改变 v1 行为。v2 尚未接入 Prompt 或服务端约束生成流程。

## GS-OPD Safety IR v2 Teacher Prompt

- 新增 `research/gs_opd/teacher/prompt_v2.py`，提供独立 `SYSTEM_PROMPT` 和 `build_user_prompt(task_instruction, camera_names)`；从 `schema_v2.py` 读取版本及枚举，明确列出全部顶层、object、visual fact、inference 和 rule 字段，与 v2 JSON Schema 对齐。
- Prompt 要求从双相机和任务语义识别环境对象、使用本地语义 ID、区分 visual facts / inferences / uncertainties，并排除机器人本体与 MuJoCo ID。接触规则仅表达机器人部位到环境物体的 contact；目标抓取、放置关系、非目标意外碰撞分别说明；证据不足保留 `unknown`，不按非目标角色自动生成禁令。
- 新增标准库 unittest `research/gs_opd/teacher/test_prompt_v2.py`，3 项测试通过：`python -m unittest -v research.gs_opd.teacher.test_prompt_v2`。Prompt 和测试文件语法检查、`git diff --check` 均通过。
- 未调用真实 VLM，未修改或接入现有 v1 Prompt、Backend、Schema、Smoke 或 Bridge；v2 Prompt 的实际模型输出尚未验证。

## Teacher Smoke 可选 Safety IR v2 接入（2026-10-15）

- `research/gs_opd/teacher/smoke.py` 新增 `--teacher-version {v1,v2}`，默认 `v1`。v1 仍调用原 `generate()`、v1 Prompt/Parser 和现有 `json_object` 配置路径；v2 调用 Backend 的 `generate_v2()`，采用 `prompt_v2.py`，并使用 `parse_teacher_output_v2()`，不经过 v1 Parser。
- `research/gs_opd/teacher/backend.py` 的 v2 请求在 Chat Completions `response_format.type=json_schema` 下直接传入 `SAFETY_OUTPUT_V2_JSON_SCHEMA`；不会把 Schema 模式退化成 `json_object`。Thinking 配置继续应用于两个版本。若服务以 HTTP 错误拒绝 Schema 请求，报错会明确指出 JSON Schema 模式/关键字不兼容，且说明未执行无约束回退。
- 原有 `--raw-out`（保存后再校验）、`--images-out-dir`、`--capture-only` 和单次请求耗时统计均保留；一次 Smoke 只调用一个版本的一个 Backend 生成入口。
- 新增 `research/gs_opd/teacher/test_smoke_versions.py`，mock Bridge 与 HTTP，不连接真实服务；验证 v1 默认 Prompt/`json_object`/Parser、v2 Prompt/Schema `response_format`/v2 Parser，以及 Schema 请求拒绝时不回退。Smoke 版本、Prompt v2、Schema v2 单测共 16 项全部通过；修改文件语法检查及 `git diff --check` 通过。
- 未调用真实 VLM。兼容性限制：当前 v2 JSON Schema 在 `subject_parts` 使用 `uniqueItems`；[vLLM XGrammar 文档](https://docs.vllm.ai/en/stable/api/vllm/v1/structured_output/backend_xgrammar/) 将该关键字列为不支持项。使用不支持它的 vLLM 结构化输出后端时，请求会明确失败，不会剥除该约束或静默退化。JSON Schema 请求结构参考 [vLLM Structured Outputs 示例](https://docs.vllm.ai/en/stable/examples/features/structured_outputs/)。

## vLLM 0.31.0 v2 `uniqueItems` 兼容性修复（2026-10-15）

- `backend.py` 在构造 v2 `response_format` 时对 `SAFETY_OUTPUT_V2_JSON_SCHEMA` 执行 `copy.deepcopy()`，只从生成用副本的 `rules.items.properties.subject_parts` 删除 `uniqueItems`。`minItems`、机器人部位 `items.enum`、required、`additionalProperties` 和其他 Schema 内容保持不变；请求仍使用 JSON Schema 约束，不会退化成 `json_object`。
- 原始 Schema 常量未修改。v2 `validate_teacher_output_v2()` 和 `parse_teacher_output_v2()` 仍拒绝重复 `subject_parts`；模型若生成重复部位，将在本地校验阶段失败。
- 回归测试检查生成请求 Schema 仅缺少 `uniqueItems`、原始常量仍保留该约束、其余关键约束不变，并验证 Validator 与 Parser 拒绝重复部位。Teacher Smoke、Prompt v2、Schema v2 相关 unittest 共 17 项通过；Python 语法检查及 `git diff --check` 通过。
- 未启动 VLM 或执行推理。其他 vLLM Schema 兼容错误仍按服务端错误明确报出，不会通过移除其他关键字或切换无约束输出来规避。

## MuJoCo Geometry Inventory 调试工具（2026-10-15）

- Safety Bridge 新增只读 `geometry_inventory` 操作。它枚举当前编译模型的每个 Geom，并返回 `geom_id`、`geom_name`、`body_id`、`body_name` 和 `category`。分类复用 `_robot_body_ids()`：由当前机器人根 Body 和 MuJoCo `body_parentid` 层级判断 `robot` / `scene`；机器人映射或某项归属 Body ID 无法确认时，分类为 `unknown` 并保留错误信息。没有场景名称或固定 Geom ID 清单。
- 新增 `research/gs_opd/smoke/geometry_inventory_smoke.py`，通过现有本机 Bridge 协议发送 `{"op":"geometry_inventory"}`，将完整响应保存为 JSON。它不 Reset、不 step、不修改仿真状态，也不调用 Teacher。
- 运行方式：终端 A 按现有命令启动 Safety Bridge；终端 B 执行：

  ```bash
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/geometry_inventory_smoke.py \
    --out /tmp/geometry_inventory.json
  ```

  服务端需处于已加载模型状态；输出包含清单状态、分类统计和文件路径。
- 静态语法检查及轻量 fake-model 分类/失败回退验证通过；未启动真实仿真或 VLM，因此当前场景的 Geom 数量、名称及运行时分类尚待人工采集确认。

## SAM 3 文本提示分割 Smoke（2026-10-15）

- 新增 `research/gs_opd/grounding/sam3_smoke.py`。脚本使用 v2 Validator 读取 `--teacher-json`，按 `--object-id` 精确查找对象，再将其语义 `name` 传给 SAM 3 文本提示；Object ID 不传给模型。
- 调用指定 SAM 3 安装中的官方 `build_sam3_image_model(checkpoint_path=..., load_from_HF=False)`、`Sam3Processor.set_image()` 和 `set_text_prompt()`。默认根据 PyTorch CUDA 可用性选择设备；不会下载权重。每个返回候选分别保存二值 Mask PNG、原图叠加 PNG，并在 `sam3_results.json` 记录分数、像素坐标框、掩码面积及路径。零候选时记录 `status: no_candidates` 和空列表，`selected_candidate` 始终为 `null`。
- 实际运行命令（输出路径可按需替换）：

  ```bash
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/RLinf \
    /workspace/zy_workspace/envs/gsopd_sam3/bin/python \
    research/gs_opd/grounding/sam3_smoke.py \
    --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
    --object-id object_1 \
    --image /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0/agentview_image.png \
    --checkpoint /workspace/zy_workspace/checkpoints/gs_opd/sam3/sam3.pt \
    --out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/sam3_smoke
  ```

- 上述输入文件均已在本地检查存在，Teacher JSON 通过 v2 Validator，`object_1` 当前对应 `akita black bowl`；对应图像 capture metadata 的指令是“pick up the akita black bowl on the stove and place it on the plate”。
- 当前只进行语法和 mock 结果保存/零候选轻量测试；未加载 SAM 3、启动 GPU 推理或修改 Python 环境。真实候选数量、Mask 质量与叠加图待人工运行检查；候选是模型输出，不代表已确认的物理对象对应关系。

### SAM 3 CUDA BF16 推理上下文修复

- `sam3_smoke.py::run_smoke()` 将 `processor.set_image()` 和 `set_text_prompt()` 放入同一个 `torch.inference_mode()` 区域；CUDA 可用时再包入 `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`，CPU 路径仅使用 inference mode。
- 本次已通过 Python 语法检查；未加载模型、运行 GPU 推理或修改 SAM 3 官方代码、权重及环境。

### BF16 Tensor 转 NumPy 修复

- `_to_numpy()` 对 PyTorch Tensor 先 detach；仅当 dtype 为 `torch.bfloat16` 时转为 `torch.float32`，然后继续转 CPU/NumPy。其他 dtype 原样处理，包括保持 Bool Mask 为 NumPy `bool`。
- 已运行 CPU 侧轻量回归检查，覆盖 BF16 数值和 NumPy dtype，以及 Bool Mask dtype/值保持；未启动 GPU 推理。
- 可复现检查命令：

  ```bash
  PYTHONPATH=/workspace/zy_workspace/RLinf \
    /workspace/zy_workspace/envs/gsopd_sam3/bin/python - <<'PY'
  import numpy as np
  import torch
  from research.gs_opd.grounding.sam3_smoke import _to_numpy

  values = _to_numpy(torch.tensor([1.25, -2.5], dtype=torch.bfloat16))
  assert values.dtype == np.float32
  np.testing.assert_allclose(values, [1.25, -2.5])
  mask = _to_numpy(torch.tensor([[True, False]], dtype=torch.bool))
  assert mask.dtype == np.bool_
  np.testing.assert_array_equal(mask, [[True, False]])
  print("BF16 and Bool conversion regression: PASS")
  PY
  ```

### SAM 3 置信度阈值参数

- `sam3_smoke.py` 新增 `--confidence-threshold`，默认 `0.5`，只接受闭区间 `[0, 1]`。参数传入 `Sam3Processor(..., confidence_threshold=...)`，并在 `sam3_results.json` 的 `confidence_threshold` 字段记录。
- CPU mock 检查覆盖默认/自定义阈值传递、结果元数据及越界输入拒绝；语法检查通过。未运行 GPU 推理。

## MuJoCo 高分辨率相机采集 Smoke（2026-10-15）

- 已审计 Safety Bridge 启动的 `OffScreenRenderEnv` 与 vendored robosuite 1.4：环境初始化时会创建 offscreen render context；已有 context 下 `sim.render(width, height, camera_name)` 直接使用当前 `MjData` 和指定相机，不调用 reset、step 或 forward。若 context 不存在，新操作会报错，不尝试创建，因为 context 构造函数会调用 `sim.forward()`。
- Safety Bridge 新增只读 `render_highres` 操作，直接从 `agentview` 渲染 512×512 RGB。返回原始 renderer 行序图像和它旋转 180° 后的图像，并在渲染前后比较 MuJoCo integration state；状态不一致就报错且不返回图像。robosuite 扩大 framebuffer 时修改的 offscreen 尺寸及临时 camera 选择会在调用后恢复。环境 camera sensor 仍按现有 256×256 配置工作。
- 新增 `research/gs_opd/smoke/highres_capture.py`，只发送 `{"op":"render_highres"}`，把两张真实渲染图保存为 PNG，并写入 `capture_metadata.json`。客户端校验图像 shape、dtype、payload 长度和 Bridge 的物理状态未变化标记。
- 运行命令：

  ```bash
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/safety_bridge.py
  ```

  另开终端运行采集：

  ```bash
  cd /workspace/zy_workspace/RLinf
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/highres_capture.py \
    --out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/highres_capture
  ```

- Bridge 与客户端语法检查及 fake render / mocked Bridge 写图测试通过；未启动仿真或模型推理。真实 GL 渲染、图像质量及运行时 integration state 比较仍待人工执行验证。

## MuJoCo Geom ID 分割图采集（2026-10-16）

- Safety Bridge 新增只读 `geom_segmentation` 操作，复用启动时已经存在的 offscreen renderer，从当前 `agentview` 直接渲染 512×512 RGB 和 robosuite 的 MuJoCo object segmentation。segmentation 的每像素 `(object_type, object_id)` 仅在类型为 `mjOBJ_GEOM` 时解码为模型 Geom ID；`(-1, -1)` 是背景并写为 `-1`，其他可见非 Geom 对象写为 `-2`，不会误报成背景。可见 Geom 元数据动态包含 Geom ID/名称、所属 Body ID/名称和像素数。
- 不创建 renderer、不调用 Reset、step 或 forward。Bridge 在渲染前后比较 MuJoCo integration state；并在 segmentation 渲染前后再次渲染 RGB，要求 512×512 RGB 数组逐像素一致，否则拒绝返回采集结果。渲染期间修改的 framebuffer 尺寸和临时相机选择会恢复。
- `research/gs_opd/smoke/geom_segmentation_capture.py` 保留 renderer 原始方向输出 `agentview_512_rgb.png`、`geom_ids.npy`、`geom_segmentation_overlay.png`，并以相同的 180° 变换输出正向 `agentview_512_rgb_rotated_180.png`、`geom_ids_rotated_180.npy`、`geom_segmentation_overlay_rotated_180.png`。正向 RGB 与 Geom ID 图使用相同旋转，逐像素坐标一致，可用于后续 SAM Mask 对齐；元数据为每个文件标注坐标方向。Overlay 按真实 Geom ID 着色，背景保留 RGB，非 Geom 像素用洋红提示；不做对象名称匹配、IoU 或安全判断。
- `--compare-rgb` 明确比较已有的正向 RGB PNG 与本次 180° 旋转后的 RGB，执行严格逐像素比较；尺寸不符、文件不可读或像素不一致会在元数据中记录并使客户端以失败状态退出。需确保两张图来自相同场景状态和渲染条件。
- 两个终端运行方式：

  ```bash
  # 终端 A：启动 Safety Bridge 和已加载的 LIBERO-Safety 环境
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/safety_bridge.py
  ```

  ```bash
  # 终端 B：采集当前 agentview 的 RGB 和 Geom ID 分割图
  cd /workspace/zy_workspace/RLinf
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/geom_segmentation_capture.py \
    --out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/geom_segmentation
  ```

  可选地增加 `--compare-rgb /path/to/existing_forward_rgb.png`；输入应是与 SAM 3 一致的 180° 正向 RGB，并且场景状态、相机和渲染条件相同才预期逐像素匹配。
- Python 语法检查及 mock 写图/旋转一致性/正向 RGB 匹配与不匹配轻量测试通过；`git diff --check` 通过。未启动真实仿真或 GPU/模型推理，真实 GL segmentation 输出仍待人工运行验证。

## SAM 3 × MuJoCo 物理实例 IoU 匹配 Smoke（2026-10-16）

- 新增独立 `research/gs_opd/grounding/sam3_mujoco_iou_smoke.py`。它读取 SAM 3 `sam3_results.json` 中所有候选 Mask 和原始 score、`geom_ids_rotated_180.npy` 及 `geom_metadata.json`，检查 SAM 实际输入 RGB 与元数据中的 180° 正向 RGB 尺寸相同且逐像素相同，并校验 Geom ID 方向、可见 ID 清单及像素计数。尺寸、方向或输入图像不匹配时明确失败。
- 以 Geom 元数据中的 owning `body_id` 聚合同一 Body 下的可见 Geom，计算所有 Candidate 对每个 Body 组的 IoU 矩阵。默认 `--min-iou 0.1`，仅唯一且非零、达到阈值的最大 IoU 标为 `matched_by_iou`；零重叠、并列、空 Mask 或低于阈值保留 `unresolved`。每候选保留 SAM 原始 score、所有对象 IoU、最佳重叠对象的 Body/Geom ID 和分数，并保存 SAM Mask 与最佳物理重叠区域的并排叠加图。SAM score 不参与匹配，也不用于推断 Teacher 对象身份。
- 当前 Geom capture metadata 没有 Body parent links，因此只合并共享同一 owning Body ID 的 Geom；不会猜测跨多个 Body 的关节连接对象属于同一个实例。输出 JSON 会明确记录此限制。该 Smoke 不做语义消歧、IoU 之外的身份判断或 Compiler。
- 运行示例（SAM 结果中的 `image` 必须是该 Geom capture 的正向 RGB，且文件内容严格一致）：

  ```bash
  cd /workspace/zy_workspace/RLinf
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/grounding/sam3_mujoco_iou_smoke.py \
    --sam3-results /path/to/sam3_output/sam3_results.json \
    --geom-ids /path/to/geom_capture/geom_ids_rotated_180.npy \
    --geom-metadata /path/to/geom_capture/geom_metadata.json \
    --out-dir /path/to/sam3_mujoco_iou
  ```

- 输出 `sam3_mujoco_iou_results.json`（含完整 IoU 矩阵和 unresolved 原因）及每个候选的 `candidate_NNN_iou_comparison.png`。新增标准库 `unittest` 覆盖同 Body Geom 聚合、完整矩阵、低 SAM score 仍按 IoU 匹配、高分但无重叠时 unresolved、坐标方向及像素不一致拒绝。3 项测试、语法检查和 `git diff --check` 通过；未运行 SAM 3 推理或真实 MuJoCo 环境。

## VLM 语义 Candidate 消歧 Smoke（2026-10-16）

- 新增独立 `research/gs_opd/grounding/vlm_semantic_disambiguation_smoke.py`。输入为 Safety IR v2 JSON、目标 `object_id`、SAM 3 结果和对应 IoU 结果。Safety IR 由现有 v2 Parser 校验；读取 SAM 结果记录的原始 RGB 与候选 Mask，并标出 `candidate_index`。
- 当前视觉输入为两张同尺寸图：`original_rgb.png` 是按输入 RGB 像素原样保存的原图；`numbered_candidates.png` 仅覆盖 1 像素候选轮廓和候选编号文字，不填充候选内部。两张图在同一次 VLM 请求中发送。Prompt 明确规定编号和轮廓只是定位辅助、属于人工标注，不能用于颜色、材质或外观判断；这些判断只能依据未修改原图。Teacher 对象描述、SAM 候选编号、Backend 配置、响应 Schema 和物理关联逻辑与既有实现相同。
- 调用现有 `create_backend_from_env()` 配置的 OpenAI-compatible Teacher Backend transport，附带 Teacher 对象的名称、角色、视觉事实、推断、不确定性及相关接触规则。VLM 请求使用独立 JSON Schema，输出严格限定为 `object_id`、`candidate_index`、`reason`、`status`；`candidate_index` 只能是已有候选编号或 null。VLM 不会收到 SAM score、IoU、Body/Geom ID 或物理对象名称，也不以 SAM 最高分代替语义判断。原始 VLM 文本先保存，再进行本地严格校验。
- VLM 只有返回 `resolved` 且引用有效候选时，Smoke 才读取该候选已有的 IoU `best_match` 并附加 Body/Geom ID。若 IoU 本身 unresolved，语义 Candidate 结果仍保留，但物理关联继续标记 unresolved；VLM 也可返回 unresolved 和 null Candidate。
- 为与先前错误选择 Candidate 0 的运行进行单变量对照，继续使用相同 Teacher JSON、`object_4`、SAM/IoU 文件及 VLM 服务配置；本次使用新的输出目录，保留此前结果不被覆盖。运行前需配置已有 Teacher Backend 环境变量：`GS_OPD_TEACHER_BACKEND=openai_compatible`、`GS_OPD_TEACHER_BASE_URL`、`GS_OPD_TEACHER_MODEL`；若服务启用鉴权，再配置 `GS_OPD_TEACHER_API_KEY`。实际运行命令（模型名和 URL 替换为当前已启动服务的配置）：

  ```bash
  cd /workspace/zy_workspace/RLinf
  export GS_OPD_TEACHER_BACKEND=openai_compatible
  export GS_OPD_TEACHER_BASE_URL=http://127.0.0.1:8000/v1
  export GS_OPD_TEACHER_MODEL=your-served-vlm-model
  # 服务要求鉴权时，设置实际 API Key；按需设置 GS_OPD_TEACHER_ENABLE_THINKING=0 或 1。

  PYTHONPATH=/workspace/zy_workspace/RLinf \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    -m research.gs_opd.grounding.vlm_semantic_disambiguation_smoke \
    --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
    --object-id object_4 \
    --sam3-results /workspace/zy_workspace/RLinf/logs/gs_opd/sam3_smoke_512/sam3_results.json \
    --iou-results /workspace/zy_workspace/RLinf/logs/gs_opd/sam3_mujoco_iou/sam3_mujoco_iou_results.json \
    --out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/vlm_semantic_disambiguation_object_4_raw_outline_ab
  ```

- 以上 Teacher/SAM/IoU 输入文件在本地存在，SAM 与 IoU 的 Candidate/图像坐标关联通过轻量输入审计，当前 `object_4` 名称为 `small metal bowl`。mock unittest 覆盖单次 Backend 请求、双图顺序与像素内容、原图逐像素保真、候选内部不填色、标注免责声明、结构化 Schema/非法候选拒绝、unresolved 行为及成功后的 Body/Geom 关联。3 项单元测试、Python 语法检查和 `git diff --check` 通过；未调用真实 VLM。旧输出目录保持不变，可直接与 `...object_4_raw_outline_ab` 对照。

## Teacher 视觉定位点 Smoke（2026-10-16）

- 新增独立 `research/gs_opd/grounding/teacher_visual_point_smoke.py`。脚本读取并校验 Safety IR v2、目标 `object_id`、用户指定的 180° 正向原始 RGB、SAM 3 Mask 和 MuJoCo IoU 结果；要求传入 RGB 与 SAM 输入图像及 IoU 来源 Geom capture 的正向 RGB 逐像素一致，且尺寸/坐标方向元数据吻合。
- 现有 Teacher Backend 只收到目标对象的名称、角色、可见事实、推断、不确定性、相关 Safety IR 规则和单张未标注原始 RGB。请求的 JSON Schema 只约束 `status,x,y,reason`，不传 Candidate 编号、Mask、分数、IoU、Body/Geom ID 或物理对象名称。`x,y` 是左上角原点、范围 `[0,1]` 的归一化坐标；`located` 必须给出两个有限坐标，`unresolved` 必须令两者均为 null。收到响应后先保存原文，再进行严格字段/坐标校验。
- Smoke 将坐标以 `round-half-up` 映射到 `(width-1,height-1)` 像素索引；仅当该像素落在唯一一个 SAM Mask 内，且该候选 IoU 状态为 `matched_by_iou`、Body/Geom 元数据内部一致时，才返回 `resolved` 和 Candidate/Body/Geom ID。Mask 无命中、多个 Mask 重叠、物理匹配 unresolved 或 Teacher 点位 unresolved 都保留整体 `unresolved`。输出定位点原图可视化、原始 VLM 文本、结构化 JSON；可选 `--semantic-result` 将点位映射 Candidate 与既有消歧结果 Candidate/物理 ID 作一致性对照。
- 对当前 `object_4`，已有最新轮廓辅助消歧结果为 Candidate 1（Body 26 / Geom 167）；先前填色图版本曾选择 Candidate 0。下面命令把新定位点结果与最新轮廓版本做对照；不会覆盖两份历史消歧输出。模型 URL/名称需与当前已启动 Teacher 服务一致：

  ```bash
  cd /workspace/zy_workspace/RLinf
  export GS_OPD_TEACHER_BACKEND=openai_compatible
  export GS_OPD_TEACHER_BASE_URL=http://127.0.0.1:8000/v1
  export GS_OPD_TEACHER_MODEL=your-served-vlm-model
  # 服务启用鉴权时设置 GS_OPD_TEACHER_API_KEY；Thinking 可按服务配置为 0 或 1。

  PYTHONPATH=/workspace/zy_workspace/RLinf \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    -m research.gs_opd.grounding.teacher_visual_point_smoke \
    --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
    --object-id object_4 \
    --image /workspace/zy_workspace/RLinf/logs/gs_opd/highres_capture/agentview_512_rotated_180.png \
    --sam3-results /workspace/zy_workspace/RLinf/logs/gs_opd/sam3_smoke_512/sam3_results.json \
    --iou-results /workspace/zy_workspace/RLinf/logs/gs_opd/sam3_mujoco_iou/sam3_mujoco_iou_results.json \
    --semantic-result /workspace/zy_workspace/RLinf/logs/gs_opd/vlm_semantic_disambiguation_object_4_raw_outline_ab/semantic_disambiguation.json \
    --out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/teacher_visual_point_object_4
  ```

- 3 项轻量 unittest 覆盖严格响应解析、VLM 请求只包含语义和单张原图、唯一 Mask 命中且物理匹配成功、重叠 Mask 和物理匹配 unresolved 时不输出选择，以及与既有 Candidate/Body/Geom 对照；语法检查及 `git diff --check` 通过。对本地 `object_4` 输入做了无模型的文件/尺寸/坐标关联审计：`small metal bowl`、512×512、4 个 Mask，现有轮廓辅助消歧结果为 Candidate 1。未调用真实 VLM 或运行耗时推理。

## Contact Safety Compiler Smoke（2026-10-16）

- 新增 `research/gs_opd/smoke/contact_safety_compiler_smoke.py`，目前只识别 Safety IR v2 中显式的 `predicate=contact`、`decision=forbid`、`subject_parts=["gripper"]`、`phase=any` 规则。不会从 `avoid_object` 等角色推导禁令；无规则、规则冲突/不支持、目标语义未解析或 Body/Geom 绑定不一致时输出 `unknown`。
- 复用现有语义消歧 JSON 的 `physical_association`，并在当前 Bridge 会话查询现有 `geometry_inventory`，核对目标 Body/Geom ID 和名称确属当前 scene 模型。为了精确区分手爪与机械臂其他部分，Safety Bridge 只新增只读 `gripper_geometry_inventory`：通过当前 robosuite gripper model 的 `contact_geoms` 名称及模型 `geom_name2id()` 得到精确 Geom ID，并核实其所属 Body 在机器人层级中；不按名称前缀推断。若任一夹爪几何身份无法完整确认，规则判定为 `unknown`。
- Smoke 在同一 Bridge 连接内获取几何清单、创建 Snapshot，并调用原有 `branch_rollout` 接口执行用户传入的候选动作序列；不重写或拆分动作执行。只有主状态恢复、机器人 Body 分类可用、每个已执行控制步的物理子步采集完整、没有 unavailable 子步且所有 Contact pair 身份可与当前模型核对时，未观察到夹爪-目标 Body 接触才判为本规则的 `safe`。精确的 gripper Geom 与绑定目标 Body 接触输出 `violation` 和首次控制步/物理子步、接触次数、Contact pair 及可用最大力证据；其他机器人部位与目标接触单独列出，不计作夹爪违规。双方都不属于绑定目标 Body 的有效接触事件会被忽略，即使其中一方是 gripper Geom；身份不一致仍会使判定为 `unknown`。整体结果按候选汇总：任一已证明违规即 `violation`，所有候选均完整安全才 `safe`，其余为 `unknown`。
- 结果范围仅是已编译的 `forbid contact(gripper, object_id)` 规则，不代表其他安全规则或未执行动作安全。输出包括 `contact_safety_compiler_results.json`、Bridge 当前完整 Geom 清单、夹爪 Geom 清单及所用 `branch_rollout_result.json`。测试覆盖精确夹爪接触、机械臂其他部位接触、完整无接触记录、监控缺失、夹爪身份不完整；未启动 Bridge 或执行仿真。
- 使用 `object_4` 的当前 v2/语义结果做实际实验时，先在 LIBERO-Safety 环境启动 Bridge：

  ```bash
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/safety_bridge.py
  ```

  另开终端，准备一组用于该实验的 7 维动作序列，再运行 Smoke。下面动作只是小幅探测示例；如需判断特定行为，请用对应分支动作替换：

  ```bash
  cd /workspace/zy_workspace/RLinf
  cat > /tmp/gs_opd_contact_probe.json <<'JSON'
  [
    {
      "candidate_id": "contact-probe",
      "actions": [
        [0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00],
        [0.00, 0.04, 0.00, 0.00, 0.00, 0.00, 0.00],
        [-0.04, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
      ]
    }
  ]
  JSON

  PYTHONPATH=/workspace/zy_workspace/RLinf \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/contact_safety_compiler_smoke.py \
    --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
    --object-id object_4 \
    --semantic-result /workspace/zy_workspace/RLinf/logs/gs_opd/vlm_semantic_disambiguation_object_4_raw_outline_ab/semantic_disambiguation.json \
    --candidates-json /tmp/gs_opd_contact_probe.json \
    --out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/contact_safety_compiler_object_4
  ```

- 6 项轻量 unittest、Python 语法检查和 `git diff --check` 通过；测试包含有效的非目标物体接触（含 gripper Geom）应被忽略。仅使用 mock Bridge/联系记录验证编译逻辑，没有运行真实分支 Rollout。真实模型下的 gripper Geom 查询和 Contact 判定待按上面的命令验证。

## π₀.₅ 首个 Action Chunk Contact Safety 探测（2026-10-16）

- `research/gs_opd/smoke/pi05_rollout.py` 新增可选 `--contact-safety-probe` 模式，要求同时提供 `--teacher-json`、`--object-id`、`--semantic-result` 和 `--probe-out-dir`。模型按现有流程根据 Reset Observation 生成首个完整 Chunk 后，脚本不发送任何真实 `step`；它使用 Rollout 当前 Bridge 连接查询 Geom/Gripper 清单、创建 Snapshot，再将同一 Chunk 作为单个候选传给现有 `branch_rollout`。
- 分支接触监控结果传给 `contact_safety_compiler_smoke.py` 的 `compile_forbid_contact()`，使用 Safety IR v2 和语义消歧文件计算 `safe/violation/unknown`。输出目录中的 `first_chunk_contact_safety_probe.json` 保存原始预测动作 Chunk、Bridge 清单、完整分支结果/接触证据、编译结果和 `main_state_restored`。只有 Bridge 明确报告主状态恢复时该字段才为 true。
- 探测模式运行后立即结束，不会继续 Rollout；为避免写出没有真实执行 Transition 的轨迹，不能与 `--trace-out` 同用。未启用探测参数时，Rollout、Video、Trace 流程不变。探测时可选 `--video-out` 只包含初始观测帧，不应将其当作动作视频。
- 轻量 Bridge Mock 测试验证 Geometry/Gripper 查询、Snapshot、完整动作 Chunk 的分支请求顺序，确认未发送 `step`，并检查动作及分支证据落盘。Python 语法检查、该测试和 `git diff --check` 通过；未运行模型或仿真。
- 真实验证需先按下列命令启动 Safety Bridge，再运行 π₀.₅ 首 Chunk 探测。输出目录独立保存本次检查结果：

  ```bash
  # 终端 A：LIBERO-Safety 环境启动 Bridge
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/safety_bridge.py
  ```

  ```bash
  # 终端 B：生成首个模型 Chunk、分支试执行并编译接触结果；不会执行真实动作
  cd /workspace/zy_workspace/RLinf
  PYTHONPATH=/workspace/zy_workspace/RLinf \
    /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
    research/gs_opd/smoke/pi05_rollout.py \
    --contact-safety-probe \
    --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
    --object-id object_4 \
    --semantic-result /workspace/zy_workspace/RLinf/logs/gs_opd/vlm_semantic_disambiguation_object_4_raw_outline_ab/semantic_disambiguation.json \
    --probe-out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/pi05_first_chunk_contact_probe_object_4
  ```

## Verifier-Guided Action Repair v1（2026-10-16）

- 新增 `research/gs_opd/repair/verifier_guided_repair.py`，提供无模型/环境副作用的可复用搜索接口 `run_verifier_guided_repair()`。`pi05_rollout.py` 增加可选 `--verifier-guided-repair` 入口，使用 π₀.₅ 正常预测的首个、已反归一化 `[T,7]` Action Chunk 作为 A0；对 A0 与后续每个候选都调用同一 `snapshot_id` 下的 `branch_rollout`，并复用 `compile_forbid_contact()`。该入口在探测后立即结束，不对主环境发送 `step`，不改变未启用该参数时的 Rollout、Video、Trace 路径。
- Safety Bridge 新增只读 `repair_context`，返回当前加载 OSC 的输入/输出范围、有效 `use_delta`、位置限幅、MuJoCo gravity/world-up、控制 timestep 和 Panda 关节范围。分支状态的对象诊断增加 `body_id/body_name`。子步 Contact event 新增首个世界坐标接触位置、MuJoCo contact frame 第一轴法向、法向原始 Geom 顺序及可用状态；物理 step、动作缩放、Snapshot/Restore、分支和 Contact 采集时序未改。contact normal 的 `geom1→geom2` 方向用本地 MuJoCo 双球接触模型做了轻量确认，真实 LIBERO-Safety 环境还未运行验证。
- 搜索仅在 A0 的 Compiler 结果为明确 `violation` 时启动；A0 为 `safe` 时报告 `skipped_original_safe`，`unknown`、不完整监控或无法将违规 Contact 与完整位置/法向关联时报告 `skipped_unknown`，不构造 A+。Repair direction 由首个违规步、违规 Geom 对、接触法向、接触点与该控制步结束后 EEF 位置的相对向量、违规前观测到的 EEF 运动方向构造：障碍法向外侧、接触切平面两侧方向和按真实 MuJoCo 重力确定的上方方向。接触位置/法向有物理子步时间精度，Bridge 目前只在控制步结束时保存 EEF pose，所以 EEF 相对向量是时间近似值，字段会明确标注。候选若因 `done` 提前结束，将记录实际检查步数并拒绝作为完整 Chunk 标签；不会把未执行动作输出为 A+。
- 修正是在已反归一化的 world XYZ 控制量上添加控制步位移 bias；以 Bridge 实际 `input_min/max`、`output_min/max` 计算 OSC 增益，按输入裁剪后的有效基动作构造候选，动作维度 3–6（姿态和夹爪）保持逐元素相同。每个候选使用接触步前后连续动作上的平滑包络，至少修正两个动作；如果修正会越过 OSC 输入范围、被裁剪抹掉、窗口太短、控制器目标限位或机器人关节限位检查失败，则拒绝该候选。配置的 Cartesian `position_limits` 若存在会检查每步目标；若未配置，则用 Bridge 返回的模型关节范围和分支实际 EEF 状态作约束。当前控制配置中的 Cartesian 位置限幅可能为 `null`，因此实际任务工作空间边界仍需真实运行核验，不能将本约束解释为通用工作空间安全证明。

### 搜索参数与任务进度门槛

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| 初始每步幅度 | `clip(0.5 × 违规前观测 EEF 单步位移中位数, 0.002, 0.012) m` | 如果没有可用实际 EEF 位移，则用活动 OSC 尺度换算的基动作幅度；不是固定 action-space epsilon。 |
| 每步最大幅度 | `0.04 m` | 单个控制步的世界 XYZ 修正上限。 |
| 累计修正路程 | `0.08 m` | 各修正控制步向量模长之和上限。 |
| 时间窗口 | 前/后各 `1、2、3` 步 | 以首个违规控制步为中心，超出 Chunk 的部分截断；lead 区间首个动作有非零权重。第 0 步违规时只能从该步开始避让。 |
| 搜索扩张 | `×2`，最多 `96` 次候选 | 指数扩张超过上限时将最后一级截到 `0.04 m` 并实际构造/评估该幅度；达到上限仍无合格候选则不输出 A+。 |
| 安全后收缩 | 二分最多 `3` 轮，直到区间 `≤0.001 m` | 在发现可行候选后缩小不必要偏移；最终在合格候选中按相对 A0 的 world-XYZ 修正 L2 范数选择。 |
| 动作平滑 | 相邻修正控制量变化 `≤0.025 m` | 限制 bias 的突变；基动作原有变化不由 Repair 重写。 |
| EEF 轨迹诊断 | 至少 `5 mm`，且达到 A0 违规前正向位移的 `70%` 时标记为保留 | 只比较相同的违规前动作前缀，用于描述局部轨迹是否保持；不作为独立任务成功/进度证据。 |
| 已绑定任务物体进度 | 保留基线可观测位移/关节变化的 `70%` | 仅在额外提供语义为 `target_object` 且物理 Body 已解析时启用，并比较 A0 与候选等长违规前缀的状态变化。 |

- 进度比较以首个违规控制步 `k` 为界：基线 A0 与候选的 EEF、目标物体状态、前缀 reward 都比较动作 `[0,k)` 执行后的状态/奖励；完整 Chunk reward 另行记录，用于保留已获得的正向 LIBERO 成功奖励。候选 EEF 位移只保留为同前缀轨迹诊断，绝不单独证明任务进度。任务证据必须来自候选正向环境 reward、保留 A0 的正向 reward，或已物理绑定 `target_object` 在同长度违规前缀中的可观测进展；证据不足时不生成 A+。若提供 `--task-object-id` 与对应 `--task-object-semantic-result`，程序校验 Teacher `target_object` 角色、resolved 语义绑定、当前模型 Body 身份，再检查该物体状态变化；绑定不一致/状态缺失时不接受该对象代理。done 单独不当作成功信号。
- 若 Snapshot/分支报告主状态恢复失败，立即停止搜索并输出 `aborted_main_state_restore_failed`。其它情况没有合格 A+ 时 `repaired_action_chunk_A_plus=null`，JSON 保留失败原因。结果 `verifier_guided_repair.json` 包括 A0、首违例控制步/物理子步、Geom/Body、接触位置/法向与机器人运动、方向基、控制器参数、每次候选动作、时间窗口、幅度、分支接触与 reward/done、Compiler verdict、任务进度指标、动作/工作空间检查、拒绝原因和最终 A+。
- 本次定点修复增加 Chunk 首步/第 1 步/末步违规的窗口权重测试；lead 首个动作现在会提前产生非零修正。幅度指数序列超过 `0.04 m` 时截到并实际测试合法上界。A0 与候选的进度比较统一到违规步前缀 `[0,k)`，候选完整 Chunk 的 EEF 位移不能再与 A0 的违规前缀对比；EEF 仅是轨迹诊断，不能独立作为任务进度。Bridge 实际执行步数、step 记录数或请求步数任一短于 Chunk 时，候选记为 `rejected_incomplete_chunk`，报告 requested/executed/recorded 步数和 `training_label_valid=false`，且不输出部分或完整 A+ 标签。只有完整且通过全部门槛的候选才记录 `validated_action_steps` 并输出有效 A+。
- `RepairConfig` 可由后续 On-Policy 调用方注入以调整搜索边界；默认实现不改模型、Bridge 动作协议或主环境，不包含学习逻辑。轻量标准库 unittest 覆盖 A0 安全跳过、unknown、Contact 证据到方向/连续多步窗口、Chunk 边界窗口、多尺度扩张与收缩及最大幅度评估、搜索失败、EEF 单独不构成任务进度、对齐前缀的任务物体进度、缺少法向、候选提前终止拒绝，以及 baseline/candidate 主状态恢复失败；同时测试首 Chunk Probe 的 Bridge 请求顺序且确认不发送 `step`。15 项 Repair 核心测试、1 项 Repair Probe 集成 Mock、6 项现有 Contact Compiler 测试及 1 项既有 Contact Probe 回归测试，共 23 项通过；语法检查和 `git diff --check` 通过。以上均不是真实 Safety Bridge/π₀.₅ 环境有效性验证，不能宣称 Repair 已在真实场景成功。

### 真实首 Chunk Repair 探测命令

先在一个终端用 LIBERO-Safety Python 环境启动已更新的 Bridge（需重启旧服务以加载 `repair_context` 与 Contact 扩展）：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

另开终端运行 π₀.₅ Repair 探测。没有正向环境 reward 或已物理绑定的任务物体进展证据时，Repair 会拒绝候选并将 A+ 留空；EEF 局部位移只记录为同前缀轨迹诊断：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/RLinf/.venv/bin/python \
  research/gs_opd/smoke/pi05_rollout.py \
  --verifier-guided-repair \
  --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/v2_l0t0_raw.json \
  --object-id object_4 \
  --semantic-result /workspace/zy_workspace/RLinf/logs/gs_opd/vlm_semantic_disambiguation_object_4_raw_outline_ab/semantic_disambiguation.json \
  --repair-out-dir /workspace/zy_workspace/RLinf/logs/gs_opd/verifier_guided_repair_object_4
```

如果任务的 `target_object` 已有独立物理消歧结果，可加上 `--task-object-id <task_object_id> --task-object-semantic-result <task_object_semantic_result.json>`，用该对象的真实 Bridge 状态变化加强进度检查。运行后先人工检查 `verifier_guided_repair.json` 中的 `status`、`main_state_restored`、`first_violation`、`task_progress`、`action_constraints` 和 A+；该模式不会把 A+ 执行到主 Rollout。

**检查清单（Checklist）**

- [x] 实现首 Chunk A0 分支验证、明确违规门控和基于接触几何的局部方向生成。
- [x] 实现依赖活动 OSC 参数的平滑多动作窗口、幅度扩张及安全后幅度收缩。
- [x] 仅接受完整 Chunk、Compiler 安全、任务特定进度证据和动作/关节约束均通过的候选；部分执行候选不产生训练标签。
- [x] 保存 A0、违规位置、方向、参数、候选证据、进度指标、恢复状态和最终 Repair JSON。
- [x] 完成 23 项相关轻量单测、语法检查及 `git diff --check`。
- [ ] 在真实 LIBERO-Safety Bridge 与 π₀.₅ 环境验证 contact normal、控制器尺度、Restore 和候选有效性。

## π₀.₅ → LIBERO-Safety 7 维动作链路审计（2026-10-16）

### 动作传递与归一化

1. `research/gs_opd/smoke/pi05_rollout.py:259-277` 组装 EEF state/camera/task 输入，调用 `predict_action_batch(..., mode="eval")`，取 batch 0 并断言动作形状为 `[T, 7]`。`pi05_rollout.py:295-297` 对每行直接 `.tolist()` 发给 Bridge；Rollout 不在这里再缩放、裁剪或改写动作。`T` 为 5：评测配置 `evaluations/libero/libero_spatial_openpi_pi05_smoke.yaml` 设置 `num_action_chunks: 5`；OpenPI `pi05_libero` 的网络 horizon 为 10，但运行时 output transform 按 `action_chunk=5` 截取前五行。
2. `rlinf/models/embodiment/openpi/tasks/eval.py:77-78` 将环境 observation 送入 Pi0Eval；`rlinf/models/embodiment/openpi/env_io.py:87-100` 在 `decode_actions()` 调用输出 pipeline 并将结果作为 float32 env actions 返回。pipeline 在 `rlinf/models/embodiment/openpi/transforms/pipeline.py:102-105` 按顺序执行模型输出 transform、`Unnormalize(norm_stats)` 和数据集输出 transform。该 checkpoint 对应 `pi05_libero`：`rlinf/models/embodiment/openpi/dataconfig/__init__.py:107-116` 指定 Pi0.5、10 步 horizon、`extra_delta_transform=False`；OpenPI 创建数据配置时按非 PI0 模型启用 quantile normalization（`VLA-Arena/vla_arena/models/openpi/src/openpi/training/config.py:219-232`）。因此这里的返回动作已从模型归一化空间反变换，后续不是再对归一化模型输出做修正。
3. LIBERO 输出 transform `rlinf/models/embodiment/openpi/policies/libero_policy.py:104-118` 只保留动作前 7 维；本配置的 `extra_delta_transform=False`，不会再对前六维套用 `AbsoluteActions`。checkpoint 的实际统计文件为 `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft/physical-intelligence/libero/norm_stats.json`，其 quantile 反变换使用每维 `q01/q99`。前七维 q01→q99 为：`[-0.7152, 1.0217]`、`[-0.8907, 1.0487]`、`[-2.1055, 0.7261]`、`[-3.3144, -1.4901]`、`[-2.4146, 2.7202]`、`[-0.6158, 1.1151]`、`[-1.0000, 0.9996]`。这些是统计反变换的端点，不是输出裁剪范围；OpenPI `Unnormalize` 不负责把结果限制到 robosuite 控制范围。
4. `research/gs_opd/smoke/safety_bridge.py:1319-1326` 的 `step` 分支将请求向量转为 `np.float32` 后调用 `env.step(action)`；LIBERO `bddl_base_domain.py:1251-1261` 只有 action_dim 为 4 时才重排成 OSC_POSITION 命令，当前 7 维路径原样进入 robosuite。`robot_env.py:558-584` 按机器人 action_dim 分发，Panda `single_arm.py:236-245` 将前 6 维交给 OSC_POSE、最后一维交给 gripper。robosuite `base.py:378-393` 在一次 20 Hz 控制步内重复 25 次 2 ms 物理子步；控制器 set_goal 只在首个子步更新，期间反复运行控制器。

### 7 维含义、坐标系与范围

| 维度 | 当前 LIBERO/Panda 路径含义 | 坐标/单位 | 当前控制范围及后处理 |
| --- | --- | --- | --- |
| 0 | EEF 目标位置增量 X | MuJoCo world frame，米 | Safety Bridge 初始化时校正为 OSC 输入 `[-1, 1]` → 目标增量 `[-0.05, 0.05]` m；`set_goal_position()` 将该增量加到当前 world `site_xpos`。 |
| 1 | EEF 目标位置增量 Y | MuJoCo world frame，米 | 同维 0。 |
| 2 | EEF 目标位置增量 Z | MuJoCo world frame，米 | 同维 0。 |
| 3 | 旋转增量 axis-angle X 分量 | world/spatial 旋转向量，弧度 | Safety Bridge 初始化时校正为 OSC 输入 `[-1, 1]` → 旋转增量 `[-0.5, 0.5]` rad/分量；姿态目标无额外 orientation limit。 |
| 4 | 旋转增量 axis-angle Y 分量 | world/spatial 旋转向量，弧度 | 同维 3。 |
| 5 | 旋转增量 axis-angle Z 分量 | world/spatial 旋转向量，弧度 | 同维 3。 |
| 6 | Panda gripper 开合方向命令 | 无量纲，输入 `[-1, 1]` | `PandaGripper.format_action()` 用符号决定开/合，幅值不是夹爪宽度；其 `current_action` 每次调用以 0.01 增减并饱和到 `[-1, 1]`，再映射到夹爪执行器控制范围。 |

- 几何和坐标结论依据：`single_arm.py` 的 EEF observable 从 `sim.data.site_xpos`/world EEF orientation 取状态；`osc.py:234-237` 调用 `scale_action()`；`osc.py:202-245` 在 `control_delta=True` 时把前三维作为位置 delta、后三维作为轴角 delta；`control_utils.py:132-145` 计算 `current_position + delta`，`control_utils.py:172-177` 用 `R(delta) @ current_orientation` 更新姿态。该左乘顺序表示姿态增量按 world/spatial 轴表达。OSC 最终产生力/力矩并由 Panda 机器人 actuator torque limits 裁剪；这不是对 EEF 目标位置的保护性限幅。
- **修复前的尺度风险：** LIBERO-Safety vendored `osc_pose.json` 曾把六个 OSC 输出上下限都配置为 ±2，覆盖控制器构造函数默认值。Bridge 当前已在每次环境 Reset/硬重建后将运行时参数校正到 checkpoint 对应的标准 LIBERO 值，具体证据和换算见下方修复记录。
- 方向命名限制：动作前三维是 world X/Y/Z，不是图像像素的左/右/上。要按 `agentview` 图像语义生成左/右候选，需用当前 camera extrinsics 将像素方向投影到 world/table 平面；现有 Rollout 没有提供这项变换。把 world `+Z` 称作“上”也应以当前模型重力/场景坐标约定验证后再使用。[需运行时确认 camera/world 对齐]

### 首版局部修正建议

- 可以直接在已经 `Unnormalize` 的 Action Chunk 前三维构造 world-axis 候选；不要在模型归一化输出上直接加相同数值，也不要改维度 3–6。首版候选建议只沿单个 world axis 产生正/负小偏置，并对控制器有效输入区间 `[-1, 1]` 做显式检查，遇到基动作已饱和或修正经 clip 后没有变化时不要把它当成有效候选。
- 以当前 OSC JSON 的增益估算，`ε=0.0025` 对应每个被修正控制步约 5 mm 的目标位置偏移；这是小幅探测的起点，不是安全阈值。若五行都加常量，该目标路径偏移会逐步累计。首版可先只改第一个动作，或对一个很短的前缀使用有界/渐消 bias，并保留其余 action 值；在相同 Snapshot 下用 `branch_rollout` 比较原 Chunk、`+ε`、`-ε`，核查实际 EEF 轨迹、接触和目标是否恢复，再决定候选策略。
- “左/右/上”应先落成明确的 world `±X/±Y/±Z`，或者先从真实 camera extrinsics 求出图像方向在世界坐标中的向量；仅根据图像观感任意改 X/Y/Z 会把方向标签和物理运动混淆。每次修正还需保留 chunk 时序、动作有限值、控制器 clip、环境工作空间/桌面边界、夹爪维度不变，并对全部候选执行同一 Contact Safety Compiler 检查。

**检查清单（Checklist）**

- [x] 跟踪模型输出、checkpoint 反归一化、Bridge、LIBERO step、OSC 和 Panda gripper。
- [x] 标明 7 维语义、坐标、统计反变换及控制器输入/输出范围。
- [x] 记录首版 XYZ 候选修正建议和坐标/累积偏移约束。
- [x] Bridge 启动及 Reset 后打印并验证 active OSC_POSE 参数和输入到物理目标的实际缩放。
- [ ] 在真实 LIBERO-Safety rollout 中确认 EEF 位移与任务表现改善，并核验 camera/world 方向关系。

## π₀.₅ OSC_POSE 尺度异常排查与 Bridge 修复（2026-10-16）

### 根因与配置证据

- 失败日志和本次测量显示：未校正时 active OSC_POSE 的 6 维 `output_min/max` 均为 `[-2, 2]`，7 维 Rollout 不做额外缩放，240 步实际 EEF 累计运动为 `12.889 m`。在 controller 的仿射缩放后，xyz 动作 `0.1–0.5` 形成 `0.2–1.0 m` 的单步 world-frame 位置目标增量，足以解释观测到的过大位移。
- 训练约定可由本地 Pi0.5 配置和上游默认评测链路交叉核对：`rlinf/models/embodiment/openpi/dataconfig/__init__.py` 的 `pi05_libero` 使用 `extra_delta_transform=False`，Rollout 的 `Unnormalize(norm_stats)` 后将 LIBERO 7 维 delta action 原样送入环境；官方 LIBERO `ControlEnv` 默认使用 `OSC_POSE` 并调用 `load_controller_config(default_controller=controller)`，OpenPI 官方 LIBERO evaluator 创建 `OffScreenRenderEnv` 时也不覆盖控制器配置。robosuite **v1.4.0** 官方 `osc_pose.json` 定义 xyz 输出 `±0.05 m`、旋转输出 `±0.5 rad`、`kp=150`、`damping_ratio=1`、固定阻抗与 `control_delta=true`。本地 LIBERO-Safety vendored 的同名配置相较该官方文件被改为全部输出 `±2`，并将 `kp` 从 `150` 提高到 `750`、`kp_limits` 从 `[0,300]` 提高到 `[0,1000]`。
- 控制坐标约定来自 robosuite 1.4 的实际实现：位置 delta 加到 world 坐标 `ee_pos`；姿态轴角增量先转成旋转矩阵并左乘当前 EEF 姿态，即 world/spatial 增量。每维输入仍裁剪在 `[-1,1]`，夹爪第 7 维走 Panda gripper，不受以下 OSC 修复影响。

### 最小修复与尺度对比

仅修改 GS-OPD Safety Bridge 初始化路径，不改 π₀.₅ 输出、不改 LIBERO-Safety/robosuite 文件、不改 Repair 算法。Bridge 在首次 Reset 和客户端每次 `reset`（该环境启用 hard reset，可能重建 controller）后，对活动 OSC_POSE 应用并检查上游配置：`output_min/max=[-0.05,-0.05,-0.05,-0.5,-0.5,-0.5] / [0.05,0.05,0.05,0.5,0.5,0.5]`、`kp=150`、`kd=2√150`、`kp_limits=[0,300]`、固定阻抗、delta 命令和解耦控制。输入范围、控制维数、控制器类型或 delta 约定不符时启动/Reset 报错，不静默继续。还会调用控制器真实 `scale_action(0.5)` 验证输出 `[0.025,0.025,0.025,0.25,0.25,0.25]`；`repair_context` 返回 active gains、ranges 和限制供后续 Repair 正确读取。

| 分量 | 修复前 | 修复后 | 尺度差异 | π₀.₅ 典型输入 `0.1–0.5` 对应目标增量 |
| --- | --- | --- | --- | --- |
| XYZ 位置 | `±2 m / 输入单位` | `±0.05 m / 输入单位` | 输出斜率降低 `40×` | `0.2–1.0 m` → `0.005–0.025 m` |
| 轴角旋转 | `±2 rad / 输入单位` | `±0.5 rad / 输入单位` | 输出斜率降低 `4×` | `0.2–1.0 rad` → `0.05–0.25 rad` |
| OSC 固定刚度 `kp` | `750` | `150` | 降为 `1/5` | 作用于闭环跟踪刚度；不是动作缩放因子 |

官方配置证据：[robosuite v1.4.0 OSC_POSE config](https://raw.githubusercontent.com/ARISE-Initiative/robosuite/v1.4.0/robosuite/controllers/config/osc_pose.json)、[OpenPI Pi0.5 LIBERO training config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)、[OpenPI LIBERO evaluator](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)。Bridge 会将实际 active 参数打印到服务端终端；无需修改或覆盖 vendored 文件。

### 验证与重新评测

- 新增 `test_safety_bridge_controller.py`，用轻量假控制器验证旧 `±2/kp=750` 参数会被设置为目标值、实际仿射 `scale_action()` 输出正确、非 OSC_POSE 或非 delta controller 会拒绝启动，并静态确认每次 hard reset 后都会重应用配置。3 项测试通过；Bridge 和测试文件 Python 语法检查、`git diff --check` 通过。
- 已用 LIBERO-Safety 环境启动新 Bridge 到环境初始化阶段；真实 `OSC_POSE` runtime 输出验证为位置 `±0.05 m`、旋转 `±0.5 rad`、`kp=150`，实际 `scale_action(0.5)` 输出 `[0.025,0.025,0.025,0.25,0.25,0.25]`。新进程随后因端口 `38567` 已被占用而退出。对当前占用端口的服务只读调用 `repair_context`，确认它仍报告旧的 `±2` 配置；没有对该服务执行 Reset 或动作。因此重跑前必须正常停止旧 Bridge、再用更新后的脚本启动新 Bridge。π₀.₅ rollout 未运行，修复后的真实 EEF 位移、累计运动和成功率仍待重新验证；之前的 `12.889 m` 是修复前测量，不能当作修复后的结果。

先正常停止仍占用 `127.0.0.1:38567` 的旧 Safety Bridge，再启动新版 Bridge，确认服务端日志出现 `output_min/max`、`kp=150` 和 `scaled_half_input_probe`：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

另开终端用同一 π₀.₅ checkpoint 重跑 240 步固定 LIBERO-Safety 任务，保留视频和 Trace 便于核对 EEF 路径：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/RLinf/.venv/bin/python \
  research/gs_opd/smoke/pi05_rollout.py \
  --max-steps 240 \
  --trace-out /workspace/zy_workspace/RLinf/logs/gs_opd/pi05_l0t0_osc_scale_fix.npz \
  --video-out /workspace/zy_workspace/RLinf/logs/gs_opd/pi05_l0t0_osc_scale_fix.mp4
```

复测时重点比较动作量级、`robot0_eef_pos` 累计路径、成功/结束标志和视频；若尺度已正确但任务仍失败，再依据新 Trace 分析策略/坐标对齐，不对模型输出做手工缩放。

**检查清单（Checklist）**

- [x] 对照 Pi0.5 LIBERO 数据配置、OpenPI 正常 evaluator、LIBERO 控制器加载路径和 robosuite v1.4.0 官方 OSC 配置。
- [x] 在 Bridge 初始化与每次硬 Reset 后应用局部运行时配置，并验证 controller 实际缩放函数的结果。
- [x] 增加控制参数轻量回归测试；语法检查和 diff 检查通过。
- [ ] 人工重启 Bridge 并运行同一任务，确认 EEF 位移、任务成功率和新日志表现。

## Safety IR v2 搬运物体避障规则与离线漏检修复（2026-10-16）

### 审计与真实视觉检查

- 原 Safety IR v2 只有机器人部位到环境物体的 `rules`，没有表达“被搬运物体不能碰撞另一个物体”的字段；Contact Compiler 因此只检查 `gripper ↔ object_4` 等已编译的手部接触规则，`scene_scene` 物体对接触不会触发这条规则。
- 已用本地 Qwen3.8-27B vLLM 对归档中 Step 74 的原始 256×256 双相机 RGB 和任务指令运行 v2 Teacher。原方向与旋转 180° 的结果都没有识别中央小物体，也未输出 payload-object 禁止接触规则；这一输入路径仍会漏掉该遮挡物。
- 随后用同一固定场景的原始高分辨率双相机图像和未标注中央区域裁剪图做真实 VLM 检查。聚焦提示下，Qwen 输出了可校验的 `object_contact_rules`：将搬运碗和“small ridged cylinder”分为不同语义对象，并生成 payload→障碍物 `forbid contact`。VLM 没有把该物体命名为 perfume bottle；它只依据视觉描述识别为小型有棱纹圆柱。此输出保存于 `logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt`。因此高分辨率聚焦输入能够表达视觉风险，但不能声称标准 Step 74 输入已解决识别漏检，也不能声称 VLM 识别了瓶子的精确类别。
- SAM 3 在 VLM 给出的“small ridged cylinder”语义及视觉事实提示下产生候选；IoU 后处理把语义消歧候选绑定到当前仿真模型里的一个场景 Body/Geom。针对搬运碗，加入任务指令消歧后，VLM 选中 stove 上的 bowl 候选（IoU `0.9793`）；中央小物体候选 IoU 为 `0.1427`。VLM 只看原始 RGB、候选轮廓/编号、Teacher 语义和可选指令，不接收 MuJoCo ID/名称；物理 ID 只由本地 SAM Mask、Geom 分割和 IoU 后处理关联。关联记录为 `logs/gs_opd/semantic_bindings_l0t0.json`。低 IoU 结果与 SAM 候选不确定性仍是限制，不能视作普遍可靠的 Grounding。

### 最小实现

- `teacher/schema_v2.py` 和 `teacher/prompt_v2.py` 增加可选顶层 `object_contact_rules`，字段为 `id, subject_id, target_id, predicate, decision, phase, reason`。Validator 校验对象引用、不同 subject/target、ID 唯一和枚举；旧 v2 JSON 缺少该字段时仍可读取。Prompt 要求区分任务 payload 与可见路径障碍，不能仅因非目标角色而禁止接触；保留正常支撑面和目标放置接触，证据不足时输出 `unknown` 或记录不确定性。
- `smoke/contact_safety_compiler_smoke.py` 只从明确的语义消歧结果编译 object-object 规则，并对照 Bridge 几何清单校验当前模型身份。完整子步 Contact Recording 若发现绑定的 payload 与障碍 Body 接触，则返回 `violation` 并附控制步/物理子步、Geom/Body、接触位置/法向/力证据；身份冲突、缺失绑定或采集不完整返回 `unknown`。无关场景接触、夹爪抓取及正常支撑/放置接触不触发障碍规则。没有将 VLM 名称直接匹配到 Geom，也没有硬编码场景对象名或 ID。
- `grounding/vlm_semantic_disambiguation_smoke.py` 可选接收任务指令与当前 Geom Inventory；在发给 VLM 前排除机器人候选和无可靠 IoU 物理绑定的候选，且不把物理 ID/名称传给 VLM。`grounding/sam3_smoke.py` 可选把 Teacher 的 agentview 可见事实加入 SAM 文本提示，默认行为不变。
- 新增 `smoke/recompile_archive_contact_safety.py`，仅离线复用 NPZ 中已有的逐物理子步接触记录重编译，不启动仿真、Teacher 或完整 Rollout。

### 离线结果与限制

- 输入 `logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz` 的 27 个已归档决策，经新规则与真实 VLM/SAM/IoU 语义绑定重编译，结果为 `safe=24, violation=3, unknown=0`。旧记录在决策 Step 70 将原始 chunk 标为 safe；新编译在首个 chunk 的控制步偏移 4（Step 74）检出 `object_1` 搬运碗与 `object_4` 视觉识别障碍绑定 Body 的接触：两组接触发生于物理子步 8/9，记录到 17/16 次接触点，最大可用接触力约 `5.55/4.30`。Step 75 与 80 的后续分支也检出持续接触。证据来自归档 MuJoCo Contact 记录；Teacher 视觉语义来自上面单独保存的真实 VLM 原始响应，二者没有互相冒充。
- 本归档中的正常 gripper-target 抓取、搬运物体与当前支撑物/放置区域接触没有被这条 payload-obstacle 规则误报；其余 24 个决策保持 safe。该离线复核只验证已有轨迹，不代表实时 Rollout 的 Teacher、SAM 和 Compiler 已在每个状态在线运行。
- Schema、Prompt、Compiler、VLM 候选消歧和 SAM 提示的相关 unittest 共 39 项通过；修改的 Python 文件通过 `py_compile`，`git diff --check` 通过。未运行仿真或完整 Rollout。
- 当前 Rollout 调用链没有被改动。若实时调用方没有为规则涉及的每个语义对象传入已解析的 SAM/IoU 绑定，Compiler 应返回 unknown，不能据此继续声称安全。下一步需在同一固定任务中接通完整绑定和规则输入，先做短程分支验证；同时改进 Step 74 视角/分辨率或复核提示，避免依赖初始高分辨率识别来覆盖搬运期间的遮挡变化。

### 验证命令

离线复现上述 27 个决策重编译结果（不运行仿真或模型）：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/envs/gsopd_teacher/bin/python \
  research/gs_opd/smoke/recompile_archive_contact_safety.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --object-id object_1 \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --geometry-inventory logs/gs_opd/contact_safety_compiler_object_4/bridge_geometry_inventory.json \
  --gripper-inventory logs/gs_opd/contact_safety_compiler_object_4/bridge_gripper_geometry_inventory.json \
  --out logs/gs_opd/offline_object_contact_recompile.json
```

后续真实验证应先用 Safety Bridge 对固定 L0T0 场景提供所有 Teacher 对象的 SAM/IoU 语义绑定，再从同一 Snapshot 对短 Action Chunk 执行 `branch_rollout` 并调用 Contact Compiler；确认 Step 74 的对象对接触返回 violation、正常抓放/支撑对不触发后，再考虑接入 Rollout。当前没有新增会直接启动完整 Rollout 的命令。

**检查清单（Checklist）**

- [x] 审计 Teacher Prompt、Safety IR v2、SAM 3 候选绑定及 Compiler 的原有对象接触覆盖缺口。
- [x] 用真实 vLLM 对原始 Step 74 观测检查并保存结果；如实记录 256×256 路径的漏检，以及高分辨率聚焦输入的有限识别结果。
- [x] 增加向后兼容的 object-object 规则、严格字段/引用校验及语义绑定驱动的 Compiler 检查。
- [x] 用已有 NPZ 离线重编译，检出 Step 74 及后续两段接触；完成抓取和支撑接触回归覆盖。
- [ ] 在短程真实 Bridge 分支中验证在线对象绑定、Teacher 规则输入与正常抓放误报率；标准分辨率下的障碍视觉识别仍需改进。

## 归档 Step 70 状态重放与真实 Bridge Contact Compiler Smoke（2026-10-17）

### Smoke 流程

- 新增 `research/gs_opd/smoke/replay_archive_contact_smoke.py`。脚本从归档的 `step_indices` 逐项提取已执行 Step 0 到目标决策步前一动作，要求索引连续、未提前 done；再从对应决策的预测动作偏移读取完整 Action Chunk，并核对它与归档 Step 70～74 已执行动作逐元素一致。该核对只保证数据来源与完整性，不把 Step 74 写进安全判定。
- 默认严格模式通过现有 Bridge `reset` 重置固定任务，再逐个发送历史 `step` 动作。到目标决策点后，比较任务指令、EEF 位置、四元数、夹爪关节位置（`state_atol=1e-6`），以及 `agentview_image` 和 `robot0_eye_in_hand_image` 两张原始 RGB（逐像素完全相同）。默认任一不符都会在 Snapshot 和分支动作之前停止。
- 可选 `--allow-image-mismatch` 只将两张已成功解码的相机图像差异视为诊断信息；它不放宽任务指令、机器人状态、归档完整 Action Chunk 或语义绑定门槛。此模式仍要求两路当前 RGB 可读取。输出会保存当前相机 PNG、逐像素绝对差异 PNG（图像尺寸不一致时记录尺寸并跳过差异 PNG），并在 JSON 中记录不同像素数、最大通道差、尺寸和 dtype。没有执行当前 RGB 的 VLM/SAM 重识别；旧语义绑定只是拿来与活动模型 Geometry Inventory 核对，绝不代表本次视觉识别成功。
- 只有指令和机器人状态通过后才查询 Geometry/Gripper Inventory。Smoke 对 Safety IR v2 的每条 object-contact rule 涉及的 subject/target 语义对象逐一校验：语义结果必须存在、对应 Teacher 对象名称/角色必须一致、历史 Body/Geom 身份必须由当前完整 Geometry Inventory 解析并匹配，且不同语义对象不能冲突绑定到同一 Body。缺失、冲突或 Inventory 不完整都会在 Snapshot 前终止。该核验只证明历史物理绑定与当前模型清单一致，不提供当前图像语义重新识别证据。
- 绑定核验通过后才创建 Snapshot。随后将归档预测 Chunk 原样提交一次 `branch_rollout`，并调用既有 `compile_forbid_contact()`，输入 Safety IR v2 和全对象语义绑定。结果要求候选完整执行、逐控制步物理子步 Contact 覆盖完整；任一缺失时输出 `unknown` 或明确失败，不能将未完整分支当成 Safe。
- Snapshot 创建后先立即 Restore 同一个 ID，并按 `state_atol` 比较 canonical integration/robot/object diagnostics；未通过时不运行候选。随后一次 `branch_rollout` 请求提交同一 Snapshot 下两条字节相同的归档 Action Chunk，分别从该 Snapshot 开始执行。Smoke 要求两候选都完整执行、每一步物理子步监测完整，并比较候选起始状态、逐步物理状态、action、reward/done、Contact 事件和完整 Contact Monitoring 结构。数值误差仍受 `state_atol`（默认 `1e-6`）约束。分支前后用 Snapshot 诊断检查主环境；还要求 Bridge `main_state_restored=true`。输出 JSON 保存往返比较、重复候选一致性、图像比较、绑定核验、Action Chunk、逐步 reward/done、接触覆盖、Compiler 判定、接触证据和恢复结果。`current_scene_physical_collision` 与历史回放诊断分开记录：`historical_decision_state_matches`、`historical_decision_images_pixel_exact`、`historical_decision_observation_matches` 分别描述指令/机器人状态（按 `state_atol`）、图像和完整决策观测是否匹配；分支状态只报告当前场景分支的物理结果，不将图像不一致的分支称为历史精确复现。Bridge 分支报告 violation 但主状态恢复检查失败时，最终 Smoke 仍报告恢复失败。

### 静态预检与测试状态

- 只读预检真实 `logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz` 成功：目标决策索引为 14，前缀含 70 个 `[7]` 动作，Step 70 预测 Chunk 为 5×7 且与归档对应的 Step 70～74 执行动作逐元素相同；两路归档图像均为 256×256 RGB，机器人状态包含 EEF 位置/四元数和夹爪关节位置。本次没有连接真实 Bridge、执行物理仿真或运行 Rollout，因此还没有真实状态匹配/碰撞复现结论。
- 新增的 unittest 使用 Fake Bridge 覆盖：从 NPZ 选择连续前缀和原始 Chunk、严格模式的图像不匹配阻断、允许图像差异时继续当前场景分支并保存当前 RGB/差异图、图像允许模式仍拒绝机器人状态不匹配、缺少绑定或当前 Geometry Inventory 冲突时禁止 Snapshot/Branch、立即 Snapshot/Restore 失败阻断、两次相同分支结果一致性校验、完整分支/现有 Compiler/主状态恢复校验、短 Chunk 不得通过完整性检查。额外的 MuJoCo 小模型测试检查 Euler step 边界缓存同步、原地 Snapshot/Restore integration state 保持以及相同动作序列的 state/contact 重复性；这些测试不替代真实 LIBERO-Safety 验证。

### 手动真实验证

终端一：在 LIBERO-Safety Python 环境启动固定 L0T0 Safety Bridge。

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二：重放 70 个归档动作，核验观测后先做一次 Snapshot→Restore 原地往返，再从同一 Snapshot 分别运行两次完全相同的 Step 70～74 Action Chunk；此命令不调用 π₀.₅，分支动作不会进入主 Rollout。以下使用图像差异诊断模式；移除 `--allow-image-mismatch` 即恢复默认严格的逐像素图像门槛。

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/replay_archive_contact_smoke.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --object-id object_1 \
  --decision-step 70 \
  --allow-image-mismatch \
  --out logs/gs_opd/replay_archive_step70_contact_smoke.json
```

若指令、机器人状态、当前 Geometry Inventory 绑定、原地往返、重复分支一致性、完整子步接触覆盖或 Restore 校验未通过，实验会在相应阶段停止或返回 unknown。图像不同但其他前置检查通过时，允许检测当前场景接触；最终状态使用 `current_scene_physical_collision_detected` 等当前分支结果，并独立记录历史决策状态与图像匹配情况。真实 Bridge 命令需要人工运行；Smoke 单测不替代该验证。

**检查清单（Checklist）**

- [x] 新增独立 Bridge 客户端 Smoke，复用现有 Reset/Step/Snapshot/Branch/Compiler 接口。
- [x] 保留严格默认模式；新增可选图像差异诊断模式，任务/机器人状态与双相机可用性仍是硬门槛。
- [x] Snapshot 前检查规则涉及的 Teacher 语义绑定与当前 Geometry Inventory 一致；显式记录未进行当前视觉重识别。
- [x] 在分支前执行 Snapshot→Restore 原地往返检查；从一个 Snapshot 重复执行同一归档 Chunk，比较完整状态和 Contact 结果。
- [x] 检查原始 Action Chunk 的归档执行一致性、每个分支的完整步数、物理子步覆盖和分支后主状态恢复。
- [x] 只读预检真实 NPZ；完成 Fake Bridge unittest、Python 语法检查和 `git diff --check`。
- [ ] 在固定 L0T0 的真实 Safety Bridge 环境运行 Smoke，确认重放观测匹配并由当前绑定/Geometry Inventory 将接触编译为 violation。

## Step 70 Object-to-Object Geometry-Guided Repair Smoke（2026-10-17）

### 实现内容

- `research/gs_opd/repair/verifier_guided_repair.py` 的 `locate_first_violation()` 现在同时读取原有 Gripper-to-Object `evidence` 和 `object_contact_checks[].evidence`。对象接触证据必须能关联到编译器根据当前语义绑定生成的 forbid 规则、subject/target Body ID，以及分支 Contact Monitoring 中控制步、物理子步、Geom/Body ID 和名称完全相同的唯一原始事件；接触位置或法向不可用、绑定冲突或事件无法唯一匹配时返回明确错误，不推测方向。
- 对 Object-to-Object 碰撞，Repair 将规则 subject 的 Body 作为正在搬运的对象。语义绑定中的 Geom 是视觉定位锚点，不要求它恰好是产生碰撞的 MuJoCo 碰撞 Geom；碰撞 Geom 通过原始事件中同序的 Geom/Body 对归属到已绑定 Body。方向按 `contact_normal_geom_ids` 的实际 MuJoCo 顺序解释：subject 位于法向第一 Geom 时取 `-normal`，位于第二 Geom 时取 `+normal`。原有 Gripper-to-Object 法向兼容路径保留。
- 新增 `research/gs_opd/smoke/object_geometry_guided_repair_smoke.py`。它复用归档加载、Reset/Step 重放、指令/机器人状态比较、当前 Geometry Inventory 绑定校验和现有 `Snapshot` / `restore` / `branch_rollout` / Compiler 接口。A0 必须完整执行且接触监控完整，并由当前 Compiler 明确判为 Object-to-Object violation；否则不会进入 Repair 搜索。
- Smoke 查询 Bridge 的只读 `repair_context`，使用 Safety IR 中唯一的 `target_object` Body 作为任务进度依据（多个目标时要求显式指定）。它调用现有 `run_verifier_guided_repair()`，候选仍使用既有方向、窗口、多尺度幅度、动作约束及任务进度门槛。每批候选通过一个 `branch_rollout` 请求从同一个 Snapshot 独立开始；每次分支后额外比较主环境 Snapshot diagnostics。任何 Restore 不匹配都会清空 A+ 并终止为失败。
- JSON 结果包含 A0、首个接触的规则/对象/Body/Geom/时间/位置/法向及原始事件、活动控制器参数、所有搜索尝试的方向/幅度/窗口/动作、分支完整性、Compiler 结果、任务进度、动作约束、恢复诊断和 A+（若通过）。结果分类区分 `no_safe_candidate_found` 与 `safe_candidates_failed_task_progress`；短 Chunk 的目标物体位移仍只是已有 Repair 定义的时间对齐进度代理，不等同任务完成证明。

### 静态验证与当前限制

- 新增几何证据单测覆盖：语义锚点 Geom 不同于碰撞 Geom、subject 在事件 Geom 对两侧时的离障碍方向，以及依据真实法向 Geom 顺序反转法向的情况。新增 Smoke Fake Bridge 测试确认 A0 门槛、同一 Snapshot 分支请求、主状态检查和 A+ 输出。
- Repair 核心、新 Smoke、On-Policy Repair/Collection、归档 Contact Smoke 和 Contact Compiler 共 50 项 unittest 通过。已执行 Python 语法检查与 `git diff --check`；没有连接 Bridge、启动仿真、调用 VLM 或运行完整 Rollout，因此当前没有真实 A+ 或真实修复成功结论。
- 真实运行依赖一份含 `object_contact_rules` 的有效 Safety IR v2，以及 `semantic_bindings_l0t0.json` 中这些规则对象当前场景的绑定。下方使用已有的 `initial_center_distinction_teacher_raw.txt`；不要用缺少对象接触规则的旧原始 Teacher JSON 替代。`--allow-image-mismatch` 只放宽归档图像逐像素一致门槛，任务指令、机器人状态、当前 Body/Geom 绑定、完整 Chunk、接触监控、任务进度和 Snapshot 恢复门槛仍然生效。

### 手动真实实验命令

终端一启动固定 L0T0 Safety Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二重放归档 Step 0～69，在当前 Step 70 状态验证 A0，并执行受验证的几何引导 Repair 搜索。脚本不会调用 π₀.₅，也不会把 A+ 执行到主环境：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/object_geometry_guided_repair_smoke.py \
  --archive logs/gs_opd/on_policy_safety_l0t0_osc_fixed.npz \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --decision-step 70 \
  --allow-image-mismatch \
  --out logs/gs_opd/object_geometry_guided_repair_step70.json
```

可选 `--object-id` 和 `--task-object-id` 只在 Safety IR 中存在多个匹配对象时使用；它们必须分别对应 forbid object-contact subject 和有当前物理绑定的 `target_object`。搜索预算可用 `--max-candidate-evaluations` 调整，默认沿用核心 Repair 配置的 96 次上限。

**检查清单（Checklist）**

- [x] 扩展首个违规定位，支持对象接触证据与原始子步 Contact 的严格关联，并保持旧 Gripper-to-Object 兼容。
- [x] 按真实碰撞 Geom 顺序和两侧 Body 身份计算搬运物体离障碍方向，不硬编码场景名称/ID/碰撞步。
- [x] 新增独立归档重放 Repair Smoke；A0 violation、全 Chunk、完整 Contact Monitoring、任务进度、动作约束和主环境恢复均为必要条件。
- [x] 增加定点单测；相关 50 项 unittest 通过，Python 语法检查及 `git diff --check` 通过。
- [ ] 人工启动 Safety Bridge，运行 Step 70 Smoke 并审查是否存在安全且保持任务进度的 A+。

## On-Policy Collection 接入 Object-to-Object IR v2（2026-10-17）

- `pi05_rollout.py` 的 On-Policy preflight 新增 `--semantic-results-json`，读取 Teacher Object ID 到语义消歧结果的映射。含 `object_contact_rules` 的 IR 会复用归档 Smoke 的 `_validate_current_bindings()`，逐个核对规则 subject/target 的 Teacher 名称与角色、当前 Geometry Inventory 中的 Body/Geom 身份，并把完整映射传给 preflight、A0 和每轮 Repair 候选的 `compile_forbid_contact()`。
- 纯 Object-to-Object IR 可通过 preflight，不再要求不存在的 Gripper `compiled_rule`；若 IR 同时含旧 Gripper-to-Object 规则，仍要求对应规则成功编译。旧 `--semantic-result` 入口保留，原有 G2O 绑定和可选独立任务对象文件仍兼容。对象间规则的绑定缺失或与当前模型冲突时，Episode 在采集前失败关闭。
- 使用映射文件时，默认从 Safety IR 唯一的 `target_object` 找出 Repair 任务进度绑定，并从同一映射解析其当前物理 Body；多个目标时需要显式 `--task-object-id`，且该对象仍须在映射中。逐次决策记录完整绑定映射与校验结果。RGB、机器人状态、指令和 A0/A+ 标签仍由同一次当前 Bridge 观测写入，不读取归档历史图像。只有完整验证的安全 A0 或 A+ 标为有效；Restore 未确认则立即终止 Episode。
- 新增轻量测试覆盖纯 Object-to-Object preflight、两对象映射校验和传递、由同一映射解析任务进度 Body、映射不一致时 fail-closed、旧 G2O A0/候选均收到语义映射，以及 80 步 CLI 参数解析。尚未启动仿真或 Rollout，因此没有新的真实采集结论。

### 人工 80 步以内采集命令

先按前文命令启动 LIBERO-Safety Bridge。随后在 π₀.₅ 推理环境中运行固定 L0T0 的短程 On-Policy 采集；`object_1` 是当前 Safety IR 的搬运对象，同时从语义映射解析为任务进度目标：

```bash
cd /workspace/zy_workspace/RLinf
.venv/bin/python research/gs_opd/smoke/pi05_rollout.py \
  --on-policy-safety-collection \
  --max-steps 80 \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --object-id object_1 \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --on-policy-safety-out logs/gs_opd/on_policy_safety_l0t0_object_v2_80step.npz
```

检查输出 NPZ 的 `decision_safety_json`、`training_label_valid` 和同名 `.summary.json`。本次完成代码与轻量测试；此命令会启动真实 π₀.₅ Student，并执行至多 80 个主环境动作，需人工启动。

**检查清单（Checklist）**

- [x] 新增映射型语义绑定 CLI，保留旧单对象语义文件参数。
- [x] 复用当前 Geometry Inventory 绑定校验，校验 Object-to-Object 规则引用的两侧 Body/Geom。
- [x] 纯 Object-to-Object IR 不再依赖 Gripper `compiled_rule`；旧 Gripper-to-Object preflight 保持编译要求。
- [x] A0 与 Repair 候选 Compiler 调用接收完整语义映射，任务进度绑定从该映射解析。
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

## GS-OPD Step 2.1：通用 Near–Far 风险预警接口

- 新增 `research/gs_opd/repair/near_far.py`，提供 `forecast_near_far_risk()`。调用者传入当前 Bridge、包含 `snapshot_id` 和 `snapshot_state` 的 Snapshot、有限 `(5, 7)` 原始 A⁰ Chunk、已验证且不重叠的目标/障碍物 Geom ID 集合及风险阈值；接口只从该 Snapshot 执行这五步，不读取归档、固定决策步或未来动作。
- 接口在决策状态及每步执行后复用 `geom_set_min_distance` 查询，返回逐状态有符号间隙、固定阈值告警、首次触发步、相邻间隙变化/持续接近判断和 Chunk 终点最近 Geom Pair。阈值默认沿用 5/10/20 mm，持续接近沿用全部相邻间隙变化均小于 `-1e-9 m` 的原判据。结果明确标记几何间隙不是安全证明。
- 开始模拟前和结束时均恢复并按 `1e-6` 比较 Snapshot 诊断。任何距离查询、动作步执行或恢复不完整都会得到 `unknown`；无法确认无风险时不会输出 `no_warning`。确定观察到阈值越界的警告仍会保留，同时完整性状态为 unknown。
- 原 `near_far_gap_smoke.py` 的距离查询及阈值/趋势分析改为调用共享函数，归档 S65/S70/S75 校验和历史标签仍由 Smoke 自己处理。
- Near–Far 结果现在附带 Snapshot ID 与当前五步 A⁰ 的 SHA-256 指纹，供后续通用 Repair 接口确认预警与动作来源一致。
- 轻量测试覆盖阈值风险和持续接近、无预警、距离查询 unknown、最终恢复失败，以及非固定决策步索引。Near-Far 新接口及原 Smoke 回归测试共 13 项通过，Python 语法和 `git diff --check` 通过；未启动 Safety Bridge、仿真或模型。

**检查清单（Checklist）**

- [x] 新接口只使用调用方传入的 Snapshot、当前 A⁰ Chunk 和已验证 Geom 集合。
- [x] 接口模拟后始终尝试恢复并严格核验原 Snapshot；查询不完整和恢复不符 fail-closed。
- [ ] 人工在真实 Bridge 环境调用接口，核对当前观测、逐步间隙与恢复结果；未运行实验前不宣称真实场景预警有效。

## GS-OPD Step 2.2：通用 Near–Far 几何引导提前 Repair 搜索

- 新增 `research/gs_opd/repair/early_repair.py` 的 `run_early_repair_search()`，接受任意当前决策 Snapshot、五步 A⁰、Near–Far 结果、Safety IR v2、Teacher 语义绑定和当前 Geometry/Gripper Inventory、Repair Context、任务物体绑定及 A⁰ 分支验证结果。模块不读取归档、不固定决策步或对象 ID，也不选择候选或创建训练标签。
- 只有当前 A⁰ 完整执行、子步 Contact Monitoring 完整、Contact Compiler 明确判为 safe，且 Near–Far 结果为 complete/warning 时才进入候选搜索。接口核对 Near–Far 的 Snapshot ID、动作指纹、六个间距样本和目标/障碍物 Geom 集合，并核对 A⁰ 物理验证结果对同一 Snapshot 和动作指纹的来源声明。
- 候选方向、平滑窗口、多尺度幅度、动作约束、完整五步分支、Contact Compiler、同时间范围任务进度和终点间隙改善复用既有提前 Repair Smoke 函数。每条候选都从同一 Snapshot 分支验证；端点间隙复测后检查恢复。返回全部合格候选的 ID、动作和验证结果，不提供候选选择或 A⁺ 标签。
- 若 Near–Far unknown、A⁰ 安全证明失败、任一候选监测/Compiler/任务进度/间隙验证不完整，或 Snapshot 恢复无法确认，结果均为不可执行。完整搜索但没有合格候选时返回 `no_qualified_candidates`；有候选但存在未完成验证时返回 `not_executable`。
- 旧 `early_repair_s70_smoke.py` 的算法和默认候选 ID 保持兼容；新增可选 `candidate_prefix` 供通用调用生成非归档标识。
- 使用 `gsopd_teacher` 环境运行 Repair、通用入口、旧 Smoke、Near–Far 和 Student Lookahead 相关测试共 59 项，54 项通过、5 项因需要 LIBERO-Safety MuJoCo 环境而跳过；Python 语法检查和 `git diff --check` 通过。新增接口测试使用 Mock Bridge，没有启动真实仿真或模型，因此尚未验证通用入口在真实 Bridge 上的候选表现。

**检查清单（Checklist）**

- [x] 覆盖 S70 以外决策索引、多个合格候选、无合格候选、Near–Far unknown、A⁰ 物理证明失败、候选验证不完整和 Snapshot 恢复失败。
- [x] 旧 S70 Smoke 候选 ID 默认行为通过回归测试。
- [ ] 在真实 Bridge 上为当前任意决策状态提供已验证输入并人工运行；未运行前不声称存在真实可用候选。

## GS-OPD Step 2.3：On-Policy Near–Far 与 Early Repair 执行集成

- 在现有 `run_on_policy_safety_collection_decision()` 中，复用 A⁰ 的同一次 Snapshot、完整五步 Branch 与 Contact Compiler 结果。仅当 Compiler 明确判定 safe 且分支接触监测完整时，才进行 Near–Far 预警；A⁰ violation 继续走原有 Verifier-Guided Repair。
- Near–Far 对象对必须来自 Safety IR 中唯一的 `forbid contact` object-to-object 规则，并通过现有语义绑定校验、当前 Geometry Inventory、任务物体绑定解析目标和障碍 Geom 集合。缺少唯一规则、有效绑定、完整间距样本或 Snapshot 恢复证明时，当前 Chunk 在主环境执行前终止。
- Near–Far complete/no_warning 时选择并执行 A⁰。complete/warning 时，将同一 Snapshot、A⁰ 指纹、原始分支与 safe Compiler 候选及全部当前验证输入交给 `run_early_repair_search()`；在通过现有完整校验及终点间隙改善门槛的合格候选中，按 `verification.endpoint_gap_m` 降序选择，gap 并列时按 `candidate_id` 升序稳定选择。缺少、NaN 或非有限间隙不参加排序；若无有限且通过既有间隙改善门槛的候选则 fail-closed 终止。
- 将 `selection_policy`、所选候选 ID/终点间隙、A⁰ 基线终点间隙、所选间隙增量，以及合格候选间隙最小值/最大值和无效间隙候选排除原因写入既有 decision safety record。实际选择的 A⁺ 与训练标签仍逐项比对；`TrajectoryWriter` 和 NPZ Schema 未修改。
- 新增单测覆盖最大 gap 选择、相同 gap 的 ID 稳定排序、缺失/NaN/无穷 gap 排除及全部 gap 无效时主环境执行前终止，并保留 safe A⁰、Near–Far unknown、无合格候选、恢复失败和原有 A⁰ violation→Verifier-Guided Repair 回归。On-Policy 测试 27 项通过；本次未启动模型或仿真，真实环境下候选表现尚未验证。

**检查清单（Checklist）**

- [x] safe/no-warning 执行已验证 A⁰；safe/warning 仅执行完整验证并恢复成功且终点间隙最大的 A⁺；violation 仍使用现有 Verifier-Guided Repair。
- [x] Near–Far 几何集合来自 Safety IR、已校验语义绑定和当前 Geometry Inventory；复用 A⁰ 的 Branch/Compiler 结果。
- [x] unknown、无合格候选、动作无效或恢复未确认的路径在当前 Chunk 主环境 `step` 前终止；测试及静态检查通过。
- [ ] 人工启动 Bridge 并运行受限 On-Policy Episode，检查 Near–Far 预警、选中候选 ID、实际动作与执行后 Snapshot/任务状态。

## Early Repair 受控幅度扩大配置

- 仅将旧 Early Repair Smoke 的平滑 XYZ 候选幅度从 `0.004/0.008/0.012/0.016 m` 调整为 `0.008/0.016/0.024/0.032 m`，并将 `PROBE_CONFIG.maximum_amplitude_m_per_step` 从 `0.016 m` 调至 `0.032 m`。
- 通用 `repair/early_repair.py` 在候选生成和候选评估时均传入 `probe.PROBE_CONFIG`；候选幅度网格由它调用的 `probe.generate_clearance_candidates()` 读取 `AMPLITUDES_M`。因此通用入口复用这次幅度网格及上限，无需更改通用 Repair 算法。
- 三种几何方向、两个平滑窗口、随机选择方式、Contact Compiler、安全约束、任务进度、累计绕行上限 `0.08 m` 和 Near–Far 阈值 `5/10/20 mm` 均保持原值。没有修改 S75 Verifier-Guided Repair。
- Early Repair 旧 Smoke 与通用接口相关单测共 23 项通过；三份 Python 文件语法检查及 `git diff --check` 通过。本次未启动模型或仿真，新的幅度配置尚待人工运行实验验证效果。

**检查清单（Checklist）**

- [x] 候选测试核验 24 个网格项的幅度值、两个平滑窗口、绕行上限及 Near–Far 阈值。
- [x] 通用 Early Repair 测试核验候选生成与评估均收到同一 `PROBE_CONFIG`。
- [ ] 人工运行 S70 Smoke，比对新旧幅度配置下安全候选数、任务进度和终点间隙；未运行前不宣称效果提升。

## On-Policy Early Repair 按终点间隙选择候选

- 已移除 Early Repair 最终随机抽样。通过 `_validated_early_repair_candidates()` 的现有物理安全、任务进度、动作、间隙改善、Snapshot 恢复和动作一致性检查后，再要求 `verification.endpoint_gap_m`、A⁰ 基线间隙与已报告间隙增量均为有限数值且互相一致，并复核原有 `MIN_GAP_IMPROVEMENT_M` 门槛。
- 选择终点间隙最大的有效候选；数值并列时按候选 ID 字典序升序选择。无效 gap 候选会记录排除原因，不参与排序；没有剩余合格候选时在主环境执行前 fail-closed。
- `decision_safety_json` 顶层记录策略、所选候选/间隙、A⁰ 基线间隙、间隙增量、合格候选间隙范围、数量和 gap 排除原因。A⁺、训练标签和实际执行动作仍由现有精确一致性校验绑定。
- On-Policy 单测 27 项通过，Python 语法检查和 `git diff --check` 通过；未启动模型或仿真。

**检查清单（Checklist）**

- [x] 最大间隙选择与 candidate ID 并列排序有回归测试。
- [x] 缺失、NaN、正负无穷 gap 不进入选择；全部不可用时不会执行动作。
- [ ] 在人工运行的 On-Policy Episode 中检查写入的 gap 统计及实际执行候选。

## Repair 任务进度：Free Joint qpos 仅作诊断

- 调整 `research/gs_opd/repair/verifier_guided_repair.py::_task_object_progress()`：任务物体推进是否通过，仅以绑定物体 `root_pos` 的基线方向平移投影与既有 70% 阈值判定。Free Joint `joint_qpos` 仍记录变化范数和投影比例，但不再作为额外通过门槛。
- 缺少物体状态/有限 `root_pos`、基线平移不足以形成可靠方向、候选平移投影低于 70% 时仍不能通过该进度检查；Joint qpos 变化不能替代平移证据。其他 Reward、Contact Compiler、Snapshot、动作约束与候选搜索判定未改。
- 该辅助函数由 Verifier-Guided Repair 与 Early Repair Smoke 共用；两条调用路径的相关测试一并回归。S40 对应单测通过完整 `evaluate_task_progress()` 验证平移保留率 81%、Joint qpos 投影 68% 时总体任务进度通过；另有测试验证平移低于 70% 仍失败，缺失平移状态仍返回 unavailable。
- 三个相关测试模块共 43 项通过；本次未启动模型或仿真。

**检查清单（Checklist）**

- [x] Joint qpos 变化仅作为 JSON 诊断字段，不参与 `_task_object_progress()` 的通过判定。
- [x] Verifier-Guided Repair、通用 Early Repair 和旧 S70 Smoke 相关单测通过。
- [ ] 人工复核 S40 实验的任务进度报告，确认平移及 qpos 诊断字段符合预期。

## Early Repair 任务可行性：绕行与抓持保持

- 仅调整 `research/gs_opd/smoke/early_repair_s70_smoke.py::assess_candidate()` 使用的 Early Repair 专用判定；共享 `verifier_guided_repair.py::evaluate_task_progress()` 和 Violation Repair 均未修改。
- 原 A⁰ 方向的 70% `root_pos` 平移投影仍写入 `task_progress.original_a0_direction_70_percent_diagnostic`，并明确标记 `used_as_hard_gate=false`。候选可偏离 A⁰ 原方向，不再仅因此被否决。
- 新判定要求 A⁰ 与候选完整执行五步，起始及五个后置状态都含有效绑定任务物体 `root_pos` 和 EEF 位置；同时检查 `done`、奖励缺失/非有限值及对既有正奖励的逐前缀保持。没有正向奖励的终止会拒绝候选。
- 抓持保持使用 A⁰ 五步中物体与 EEF 的相对距离稳定性及共同运动作为启发式证据；最大相对距离偏差容许值为 0.05 m。基线抓持证据不足时返回 `unavailable`，候选物体相对 EEF 明显偏离时返回失败。该运动学启发式不能证明真实接触抓持；报告保留其局限，且 `training_label_valid` 与 `high_quality_training_label_created` 均保持 false。
- Contact Compiler、完整接触监测、动作/工作空间约束、Snapshot 恢复、间隙改善门槛及最大 gap 候选选择未改变。轻量测试覆盖偏离 A⁰ 方向但保持相对距离的安全绕行、物体丢失、任务物体状态缺失、正奖励丢失和无成功奖励的提前终止。
- Early Repair、通用 Early Repair 与 Student Lookahead 相关测试共 35 项通过；Python 语法检查和 `git diff --check` 通过。未启动模型或仿真，因此尚未用真实 S40 候选验证新的可行性门槛。

**检查清单（Checklist）**

- [x] 70% A⁰ 方向平移投影仅作为诊断记录，不再单独否决 Early Repair 候选。
- [x] 五步物体/EEF 状态、任务物体绑定、终止/奖励和相对距离抓持证据 fail-closed；不自动生成任务训练标签。
- [x] 未改共享 Verifier-Guided Repair、Contact Compiler、安全约束、Snapshot、gap 改善或最大 gap 选择逻辑。
- [ ] 人工重跑 S40 Early Repair Smoke，检查通过的候选是否仍满足接触安全、间隙改善及抓持保持诊断；未运行前不宣称真实场景已有合格候选。

## Early Repair：区分评估完成与候选合格

只读审计 `logs/gs_opd/repair_detour_feasibility_45step.npz` 最后一个 decision（S35），确认整批终止来自任务状态契约不一致：`evaluate_early_repair_task_feasibility()` 返回正常的证据不足结果 `unavailable`，而 `_candidate_validation_complete()` 仅接受 pass/fail，将其误报为 `candidate_validation_incomplete`。

| S35 归档检查项 | 实际结果 |
| --- | --- |
| 候选数量 | 24 |
| task_progress.status / reason | 24 条均为 `unavailable / baseline_grasp_evidence_unreliable` |
| 物体与 EEF 五步状态 | 24 条均完整 |
| Contact Compiler / 对象间规则 | 24 条均为 safe |
| 完整五步执行 / Contact Monitoring / 动作约束 | 24 条均通过 |
| endpoint replay / main_state_restored | 24 条均为 complete / true |
| 终点间隙改善 | 16 条；另 8 条同时被间隙改善门槛拒绝 |
| 搜索前后 Snapshot 恢复 | 均匹配，最大物理数值误差为 0 |

- 全部候选共享的 A⁰ 基线 EEF 路程为 **4.693110 mm**，低于现有 **5 mm** 门槛；物体路程为 3.482200 mm（门槛 2 mm），相对距离范围为 0.877623 mm（容许值 50 mm）。基线抓持证据不足的具体触发项是 EEF 路程；这次保持所有抓持条件原值，24 条仍不合格。
- 任务判定新增 `evaluation_complete`：完整状态、有效绑定和 reward/done 输入均可读取后，正常的 pass、fail 或证据不足 unavailable 都可表示已完成评估。缺失状态、绑定或 reward/done 时仍为 false；候选保留原始具体拒绝原因。
- 通用搜索分别检查完成性和合格性。已完成但 unavailable 的候选不能进入 `qualified_candidates`、不能执行或生成标签；它们不再单独否决其他完整合格候选。全部候选 unavailable 时返回 `no_qualified_candidates`，保持不可执行。缺少物理/接触监测、Compiler unknown、endpoint replay 不完整或 Snapshot 恢复失败仍令整批不可执行。
- 原 NPZ 未改写；修复仅改变后续搜索的状态分类，不把 S35 的 16 条间隙改善候选视为可用。生成动作、幅度、间隙门槛、最大 gap 选择和 Violation Repair 未改。
- Early Repair、旧 Smoke、Student Lookahead 和 On-Policy 相关单测共 **68 项通过**，覆盖混合 pass/unavailable、全部 unavailable、物理验证不完整、接触 unknown、恢复失败及状态/奖励缺失；四份修改 Python 文件语法检查与 `git diff --check` 通过。测试使用现有 `gsopd_teacher` 环境，未启动模型或仿真。

**检查清单（Checklist）**

- [x] S35 最后一个 decision 的 24 条真实候选已只读审计，并记录实际抓持证据不足原因。
- [x] 评估完成不等于候选合格；unavailable 不生成执行动作或训练标签，也不污染其他合格候选。
- [x] 真正的物理验证、接触监测和恢复不完整继续失败关闭，68 项单测及静态检查通过。
- [ ] 人工重跑后核对：同类全部证据不足的搜索应准确报告 `no_qualified_candidates`，不期待本修复自动产生 S35 合格动作。

## Early Repair：使用三维相对位置检查局部连续性

- 移除 Early Repair 专用可行性判定中基线 EEF 至少运动 5 mm、物体至少运动 2 mm 的前提。两者路程继续作为诊断记录，静止或低速状态不再仅因运动量不足被拒绝；共享 Violation Repair 的门槛未改。
- 对起始及五个动作后状态计算世界坐标三维向量 `r(t) = object.root_pos(t) - EEF.pos(t)`。记录完整六个相对位置，检查基线、候选各自相对初始状态的最大三维变化，以及候选与 A⁰ 同步状态的最大三维偏差；继续沿用 0.05 m 容许值。旧标量距离变化保留为诊断，不能替代三维检查。
- 相对位置连续时，pass 仅表示五步局部运动学可行，`grasp_evidence.status=kinematic_consistency_only`；`physical_grasp_confirmed`、`task_success_confirmed` 和 `high_quality_training_label_created` 均为 false。绝对距离接近不作为物理抓持的证据，静止持物与附近独立支撑的静止物体在这些位置数据下可能不可区分。
- 基线三维相对位置明显不稳定时返回已完成但 unavailable 的 `baseline_grasp_evidence_unreliable`；候选相对位置明显偏离时返回 fail；状态或 reward/done 缺失继续返回评估不完整。上轮完成性/合格性契约保持有效，证据不足候选不进入合格列表。
- Contact Compiler、动作/工作空间约束、Snapshot、奖励/终止判定、gap 改善及最大 gap 选择、Near–Far 和 Violation Repair 未修改。此项位置启发式不能证明力闭合、接触抓持或任务成功；世界坐标相对位置也会随真实持物旋转而改变，当前没有进行旋转补偿，可能保守拒绝此类候选。
- 在现有 `gsopd_teacher` 环境运行相关测试 **71 项通过**，覆盖静止、低速、侧向绕行、物体脱离、距离不变但三维方向改变、证据不足、缺失状态以及原有奖励/恢复/选择回归；三份 Python 文件语法检查和 `git diff --check` 通过。未启动模型或仿真，未重写历史 NPZ；不能根据 S35 的旧标量摘要宣称其 16 条间隙改善候选已通过新的三维检查。

**检查清单（Checklist）**

- [x] 最低运动量不再是 Early Repair 的抓持保持前提，路程仅作诊断。
- [x] 检查完整三维相对位置，区分局部运动学可行、抓持未确认与明确脱离；不声称启发式 pass 是任务成功或高质量示范。
- [x] 71 项单测及语法/diff 检查通过，其他安全、奖励与恢复门槛保持原值。
- [ ] 人工重跑受限 Episode，核对真实 S35 候选的六帧相对向量、物理安全、间隙及执行后任务表现。

## On-Policy Early Repair：Brake-and-Steer 候选

- `generate_clearance_candidates()` 新增默认关闭的 `include_brake_and_steer=False`；旧 S70 Smoke 的默认 24 条 XYZ 偏置网格、动作数值和 ID 保持一致。当前只有 On-Policy Rollout 调用的 `run_early_repair_search()` 显式传 true，在原网格后增加最多 12 次候选生成尝试，总预算保持 36。
- 用已有最近 Geom 的单位 away、正负侧向方向和 MuJoCo world_up 构造候选。A⁰ XYZ 按 OSC input/output 中点及 gain 换算为世界坐标位移 `d`，朝障碍物的投影为 `p = min(dot(d, away), 0)`。现有平滑窗口权重为 `w`，总修正为 `w * (-p * away + steering)`；原本远离障碍物的分量不制动。通过 gain 反算输入，不对候选做 clipping，旋转及夹爪值逐项保留。
- 两个现有窗口各尝试六种组合：纯制动；正侧向 8/16 mm；负侧向 8/16 mm；沿 world_up 抬升 16 mm。保留原 XYZ 幅度网格和全部限值；制动与转向合成后的真实每步修正、累计修正及 slew 均须通过原门槛。越界或无有效修正的候选在物理执行前明确拒绝。
- 新诊断包含 gain/仿射中点、A⁰ 世界位移、朝近投影、平滑权重、制动与转向分量、从实际动作差重建的完整 `physical_xyz_bias_by_step_m`、总修正幅度和各项拒绝原因。完整修正诊断已通过真实 `_candidate_constraints()` 的轻量回归测试；没有用额外侧移幅度代替制动加转向的总幅度。
- Safety 报告新增 `candidate_generation_counts`，分别记录 XYZ 偏置与 Brake-and-Steer 的生成数及拒绝数。离线测试的朝近动作样例实际生成 **24 + 12 = 36 条**；零 XYZ 动作样例生成 **24 + 10 = 34 条**，两条纯制动因无修正被拒绝；预算设为 30 时为 **24 + 6 = 30 条**。这些数字是生成层单测结果，不是实际 Episode 的物理合格候选数量。
- 新候选仍从同一 Snapshot 走完整五步 Branch、Contact Compiler、任务可行性、动作/Workspace、终点 gap 和恢复检查，再沿用最大 gap 选择。Near–Far、Verifier-Guided Repair、训练标签语义和 NPZ Schema 未修改；候选生成不构成安全、任务成功或高质量示范的证明。
- 相关单测共 **80 项通过**（Early Repair 与旧 Smoke 46 项，Student Lookahead 与 On-Policy 34 项）；四份 Python 文件语法检查和 `git diff --check` 通过。未启动模型或仿真，尚未验证 S35/S40 等真实状态下新增候选的安全通过数、间隙改善及实际任务效果。

**检查清单（Checklist）**

- [x] 旧 24 条网格默认不变；On-Policy 显式启用额外最多 12 条，总预算不超过 36。
- [x] 朝近分量制动、远离分量保留、侧移/抬升、完整修正诊断、OSC 仿射尺度、各项动作上限和旧路径兼容性均有单测。
- [x] 所有物理安全、任务可行性、gap、Snapshot 恢复及最大 gap 选择门槛继续复用，静态检查和 80 项单测通过。
- [ ] 人工运行受限 On-Policy Episode，按两类候选统计静态拒绝、物理安全、任务可行性、gap 改善及最终实际执行来源。

## Geometry-Guided Local Escape and Recovery（2026-10-18）

### 当前目标与实现范围

- 总体路线仍为真实观测 → Teacher 语义规则/已验证物理绑定 → Snapshot 物理验证 → 实际执行安全动作 → 从真实新观测继续 Student Rollout → 收集可追溯数据。本次完成局部 Escape 与 Episode 内 Recovery 的第一版代码；没有进行真实 Episode 性能验收，也没有训练。
- 新增 `research/gs_opd/repair/local_recovery.py`。当前 On-Policy Object-to-Object 路径直接调用该模块；无新增模式开关、固定决策步、历史动作或场景物体 ID。已有 Safety IR、绑定/Inventory 验证、完整 Contact Compiler、Bridge、Near–Far 和共享 Verifier-Guided Repair 实现未修改。Gripper-only 路径保留原验证逻辑。
- 当前仅接受一个明确的、已绑定的搬运任务物体 → 障碍物 forbid-contact 对。指令、物理 Body 对和连续实际状态必须保持一致；不把当前绑定套用到其他场景。

### 候选、专用约束与物理验收

- 默认尝试 **26 条**：停止朝近运动 1 条、保持位置命令 1 条；8/16/24 mm 三档下，左右侧移、抬升、左右侧移结合抬升、后撤各 1 条，共 18 条；另保留原构造器的三方向 × 两窗口、8 mm 偏置 baseline 共 6 条。生成尝试总数上限 36，不用旧 24 条网格占满预算；旧 S70 Smoke 的默认网格和行为没有修改。
- 直接构造每步世界 XYZ 位移，再用活动 OSC 的 input/output 中点与 gain 反算最终输入。以最后一个真实主环境命令为起点，前两步以 0.5、1.0 权重过渡，随后维持目标；姿态及夹爪维度逐项保留 A⁰。全部动作先量化为实际 Bridge 使用的 float32，再做分支验证和动作指纹检查；不做 clipping。
- 新候选使用显式 `recovery_absolute_control` 策略：真实 OSC 输入范围、世界坐标 delta 约定、相邻世界位移命令变化（含上一真实动作边界）不超过 50 mm、活动 Workspace 和 MuJoCo 关节限位均须通过。50 mm 是本版命令连续性策略，不能解释为硬件安全认证。活动控制器明确未设置笛卡尔 `position_limits` 时记录此事实，不虚构 Workspace。缺少必要配置/状态返回不可执行。
- 直接 Recovery 的相对 A⁰ 每步/累计修正量只作诊断，不应用旧 32/80 mm 搜索预算；六条 legacy baseline 继续使用原相对修正约束。每步真实世界修正向量、最终输入、控制尺度、预算策略及逐条拒绝原因均写入报告。
- A⁰ 当前 safe 但 Near–Far warning，或 A⁰ 明确 violation 时，可从当前 Snapshot 最近 Geom 对生成 Recovery。方向来自当前几何，不把已碰撞 A⁰ 的终点当作避障起点。每条候选独立从同一 Snapshot 完整执行五步，验证起点积分状态、实际动作、全物理子步 Contact Monitoring、全部 Teacher 规则、Workspace、任务物体/EEF 连续性、奖励/终止，以及终点重放和主状态恢复。任何必要物理验证不完整、Compiler unknown 或恢复失败均 fail-closed。
- 任务连续性比较候选起始及五步后的三维 `object.root_pos - EEF.pos`，沿用已有 50 mm 容许值，检查奖励保留和无成功奖励的终止；不再用已碰撞 A⁰ 的终点或方向作为 Recovery 的硬参照。只有当前安全 Chunk 的局部运动学证据；`physical_grasp_confirmed`、`task_success_confirmed` 和高质量示范标志不据此置 true。正常无风险 A⁰ 的抓取/放置不套用 Recovery 持物启发式。

### 跨 Chunk 状态与真实执行

- 合格候选先选择具有真实任务物体侧向进展（至少 2 mm）的同侧绕行，其次抬升，再次有限后撤，最后有限制动/静止。优先达到合理间隙，之后比较有效侧向进展和命令路程，最后按 candidate ID 稳定排序；不再一律最大化终点 gap。所有分类都先通过相同物理安全与任务可行性门槛。
- Episode 状态保留绕行侧别、世界侧向轴、实际间隙历史和阶段。生成候选时保持同侧切向；首次真实侧向执行后锁定侧别，拒绝反向折返。Snapshot 预测不会推进这些状态。
- 仅在 A⁰ 完整安全且 Near–Far 解除预警、实际间隙超过 20 mm，并连续两段真实执行支持不再接近后，下一决策退出 Recovery、恢复正常 Student 动作。确认阶段继续检查侧向一致性；实际成功奖励单独记录，不从几何指标推断成功。
- 有界策略：单次 Recovery 最多 12 个 Chunk；Episode 最多 24 个 Recovery A⁺、4 次进入；单次最多后撤 2 段、静止/纯制动保底 1 段、连续无进展 3 段。没有合格候选立即停止；计数只由实际执行更新，Episode 总预算不会因退出重入而清零。
- `pi05_rollout.py` 真正发送选中 A⁺，随后从普通 `step` 的真实双相机/状态继续 Student 推理。主环境每步附带只读积分状态；与所选分支以 `1e-6` 比较，并核对 reward/done，异常立即停止后续动作。Chunk 后重新查询实际间隙，与分支终点严格核验，再提交 Recovery 状态。
- `TrajectoryWriter` 保留原 NPZ 字段：`predicted_actions` 是原始 A⁰，`executed_actions` 是实际发送的 A⁰/A⁺。`decision_safety_json` 记录候选类型、修正量、侧别、起点/A⁰/A⁺ 终点间隙、侧向进展、阶段、拒绝统计、实际逐步核验及执行覆盖。部分执行不能把未执行后缀标为完整监督标签；动作不匹配仍保留真实转换并使标签失效。安全 Chunk 标签与最终任务成功、高质量完整示范明确分开。

### 离线证据与测试

只读使用 `logs/gs_opd/brake_steer_60step.npz` 的真实 A⁰、上一实际动作、历史绑定及已保存控制尺度诊断进行生成检查，没有仿真、没有重新推理、没有改写归档：

| 归档决策 | 生成 / 控制约束拒绝 | 超出旧相对预算的直接 Recovery | 制动最大 / 累计相对 A⁰ 修正 |
| --- | --- | --- | --- |
| S50 | 26 / 0 | 20 | 43.963 / 176.817 mm |
| S55 | 26 / 0 | 20 | 41.698 / 185.383 mm |

这些数值包含从上一实际命令平滑过渡后的完整五步修正，不等于旧窗口纯制动的统计。离线只证明生成和输入/命令连续性检查可通过；没有验证碰撞、Workspace 路径、抓持、任务效果或 Snapshot 恢复。

- CPU 环境：`/workspace/zy_workspace/envs/gsopd_teacher/bin/python`。七个直接相关 unittest 模块 **126 项通过**。全部 24 个 GS-OPD unittest 模块 **213 项：206 通过，7 跳过**；跳过项为会创建 MuJoCo 最小模型的 Snapshot 两项和 GeomDistance 五项，遵守本次不启动仿真的要求，不能把它们算作已验证。
- 五份改动 Python 文件的 `py_compile` 与 `git diff --check` 通过；源文件及进度文档没有截断标记，原有 On-Policy 决策函数完整保留。
- 新模块、测试和 Writer 的 Ruff lint/格式检查通过。原有 Rollout 与 On-Policy 测试文件分别仍有 10/2 项历史 Ruff 提示（导入布局、版权头等），与本次起点 `8cea42a6` 对比没有新增；未为清理旧格式大范围重写 Rollout。
- 新增 `test_local_recovery.py`，通过确定性 Bridge 协议替身调用真实规则/绑定验证、Near–Far、Compiler 及 Recovery 实现。覆盖强朝近动作超旧预算、真实输入越界、Workspace/slew、碰撞和 unknown 拒绝、违规 A⁰ Recovery、同侧持续、退出后正常执行、有限后撤/无进展/重入、真实状态异常立即停止、奖励/终止、缺失状态、任务脱离，以及 NPZ 原预测/实际动作和部分执行标签对应。旧 Near–Far/S70/Verifier 回归通过。
- **证据限制：** 协议替身不是 MuJoCo，离线输入合法不是物理安全。相对位置连续不能证明物理抓持，世界坐标检查可能保守拒绝持物旋转；Mesh 距离仍为近似值。没有运行模型、真实 Bridge、完整 Episode 或训练。真实任务成功率、绕行效率、冻结/后撤次数、实际计算耗时和完整 Episode 恢复可靠性仍需人工实验。

### 人工运行命令与下一步

离线生成诊断（无需 Bridge/模型）：

```bash
cd /workspace/zy_workspace/RLinf
/workspace/zy_workspace/envs/gsopd_teacher/bin/python \
  -m research.gs_opd.repair.local_recovery \
  --archive logs/gs_opd/brake_steer_60step.npz --steps 50 55
```

终端一，人工启动现有固定任务 Bridge：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

终端二，使用现有 π₀.₅ 环境运行受限的真实连续采集。这会实际执行受验证动作；程序仍核对当前 Teacher/双对象物理绑定，不适用于其他场景：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/RLinf/.venv/bin/python \
  research/gs_opd/smoke/pi05_rollout.py \
  --on-policy-safety-collection --max-steps 80 \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --object-id object_4 \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --on-policy-safety-out logs/gs_opd/local_escape_recovery_80step.npz \
  --video-out logs/gs_opd/local_escape_recovery_80step.mp4
```

**检查清单（Checklist）**

- [x] 新 Recovery 不受旧相对 A⁰ 搜索预算误阻断，最终真实控制输入和物理验证没有放宽；旧 Smoke/共享 Repair 保持原行为。
- [x] 真实执行后才更新跨 Chunk 状态；动作、标签、观测及实际覆盖对应，失败路径停止发送后续动作。
- [x] 生成离线诊断和 206 项无模型/仿真单测通过；7 项 MuJoCo 用例明确跳过。
- [ ] 人工运行受限 Episode，核对真实安全通过数、同侧绕行、危险解除后的任务恢复、奖励成功及所有停止原因；尚不宣称真实场景 Recovery 已成功。

## 独立几何轨迹 Bundle：Waypoint 与 B-Spline（2026-10-18）

### 当前目标与全局路线

- 新增 `research/gs_opd/repair/trajectory_bundle.py` 和 `test_trajectory_bundle.py`，实现从当前搬运物体包围球中心到已验证目标侧中心的完整三维几何候选。本阶段仅生成与初筛；后续若接入执行，仍需控制转换、Snapshot/Contact Compiler 和真实任务可行性验证。
- 模块仅使用标准库与现有 NumPy，未导入或修改 OSC、Recovery、Near–Far、Teacher、Safety Bridge 或 On-Policy 主循环。所有报告和候选均明确 `physical_safety_verified=false`、`executable=false`，不生成训练标签或自动选定最终路径。
- 源码审计确认，现有 Geometry Inventory 返回 Geom/Body 身份及 robot/scene 分类，未提供完整几何包围尺寸。此模块要求调用方显式提供同状态、已核验的保守包围球和目标中心来源，不从名称、VLM 描述、Body 原点或历史位置推断半径/目标。**真实任务输入的完整几何包围范围与目标中心坐标来源：[需补充]**；本轮结果来自明确标识的合成几何测试。

### 输入与生成接口

接口为 `generate_trajectory_bundle(scene, BundleConfig(...))`，返回可直接写入 JSON 的诊断。`scene` 的输入契约如下：

| 字段 | 要求 |
| --- | --- |
| `frame / units / state_id` | 固定 `world / m`，非空当前状态标识 |
| `world_up` | 有限非零三维向量，模块归一化后构造局部方向 |
| `carried` | 搬运物体的完整保守包围球，字段见下方 |
| `obstacles` | 至少一个障碍物包围球；与搬运物体的 `object_id` 全部唯一 |
| 每个包围球 | `object_id, center_world_m[3], radius_m>0, state_id`；需 `binding_verified=true, enclosure_verified=true` 及非空 `binding_source, enclosure_source` |
| `goal` | `position_world_m[3], position_semantics="carried_sphere_center", verified=true, source, state_id`；位置含义是期望搬运球中心，不自动等同于放置物体的 Body 原点 |

- 全部包围球和目标的 `state_id` 必须与当前输入相同。声明及来源字段仅用于拒绝缺失、未确认或混合状态输入，模块不能独立证明调用方的物理绑定/几何来源真实可靠。
- `sphere_from_aabb(lower_world_m, upper_world_m)` 可把调用方已核验的完整世界 AABB 转成保守球。球半径覆盖整个对角线，并处理中心舍入；不能用只有局部可见表面的 AABB 代替完整物体尺寸。
- 默认 18 条，允许整数 12～24 条；循环分配左/右/上三类。根据起点→目标轴及 `world_up` 建立方向，选择对直线路径间隙最小的障碍物作为采样参照，同时对所有输入障碍物进行初筛。没有物体 ID/场景名称分支。
- 分级采样偏移为膨胀半径的 1.2～2.6 倍，交替使用两种纵向展开；构造入口、绕行峰点和出口，再形成七个控制点的端点夹持三次 B-Spline。曲线端点严格为当前中心与目标中心，内部控制点不是必须经过的插值点。起终点重合或任务轴与 `world_up` 平行、无法定义左右时明确报错。
- 每条候选记录方向及世界向量、Waypoint、B-Spline 控制点/节点、完整离散路径、近似弧长、终点目标侧投影、逐障碍间隙与拒绝原因；弧长为离散折线长度，不包含速度或时间参数。

### 保守间隙初筛与可视化

- 障碍物膨胀半径包含搬运球半径、障碍球半径及默认 10 mm 间隙要求。先检查起点、终点和离散曲线；没有合格路线时保留全部拒绝候选并报告通过数为零。
- 不仅检查采样点。用三次 B-Spline 导数控制点的最大范数给出全曲线的参数导数范数上界 `L`，均匀参数采样间隔为 `Δu`，连续间隙下界为 `最小采样间隙 - L·Δu/2 - 数值余量`。只有该保守下界满足间隙要求才通过几何筛选；采样点通过但下界不足时明确标记 `curve_clearance_bound_inconclusive`。
- 默认至少 65 个点，根据 1 mm 误差目标自适应增加，最多 4097 点；达到上限后仍使用实际误差界，不会因稀疏采样而静默放行。
- `save_trajectory_bundle(bundle, out_dir)` 保存 `trajectory_bundle.json` 和 `trajectory_bundle.svg`。SVG 提供 XY/XZ/YZ 等比例投影，按左右/上方向着色，拒绝曲线为灰色虚线；悬停可查看候选 ID 和原因。圆表示障碍球加搬运球及余量，JSON 保留全采样，SVG 仅为显示适度抽样。
- **局限：** 初筛只在输入保守包围成立、障碍物静止和路径参考点确为搬运球中心的条件下成立。未覆盖机器人/夹爪体积、姿态、支撑接触、动力学、可达性、动作连续性或任务成功；几何通过不能作为 MuJoCo 验证通过或可执行证明。

### 测试与使用

- 现有 `gsopd_teacher` CPU 环境下，新模块 **16 项 unittest 通过**；结合 Near–Far、Local Recovery 和旧 S70 Smoke 回归共 **76 项通过**。测试使用合成数据，不启动模型、Bridge 或仿真。
- 两个新增 Python 文件的 `py_compile`、Ruff lint/格式检查，以及 `git diff --check` 通过；实际文件范围仅为新规划器、对应测试和本进度文档。
- 测试覆盖解析三次 Bezier 对照、12～24 条预算与确定性、世界方向旋转、AABB 包围（含大坐标舍入）、搬运球半径膨胀、采样点之间穿越、连续间隙界、额外障碍物拒绝、全拒绝、状态/来源/单位/目标语义校验、ID 无关性及 JSON/SVG/CLI。
- 基础合成场景的直线穿过障碍球，默认生成 **18 条**（左/右/上各 6），全部通过球形几何初筛；额外障碍物及非法起终点测试会产生拒绝。该数字不是当前真实搬运任务的路径通过数。

运行上述轻量测试：

```bash
cd /workspace/zy_workspace/RLinf
/workspace/zy_workspace/envs/gsopd_teacher/bin/python -m unittest \
  research.gs_opd.repair.test_trajectory_bundle \
  research.gs_opd.repair.test_near_far \
  research.gs_opd.repair.test_local_recovery \
  research.gs_opd.smoke.test_early_repair_s70_smoke
```

调用方准备符合上表的真实核验输入后，可使用以下离线入口。`--scene-json` 文件必须由调用方提供，不应把测试夹具中的来源声明作为真实几何证据：

```bash
cd /workspace/zy_workspace/RLinf
/workspace/zy_workspace/envs/gsopd_teacher/bin/python \
  -m research.gs_opd.repair.trajectory_bundle \
  --scene-json /path/to/verified_geometry.json \
  --out-dir logs/gs_opd/trajectory_bundle \
  --candidate-count 18 --clearance-m 0.01
```

**检查清单（Checklist）**

- [x] 独立 12～24 条三维 B-Spline 候选、保守球间隙界、JSON/SVG 及逐条拒绝原因已实现。
- [x] 新增 16 项及相关回归 76 项通过，未修改现有控制/执行/安全判定路径。
- [ ] 补充同状态完整包围球和目标侧中心的实际核验来源，再检查真实场景几何可行性；本轮没有仿真或物理安全结论。

## 真实规划几何只读查询与 Scene Smoke（2026-10-18）

> 历史实现记录：本节记录只读 Bridge 几何查询及最初依赖最终放置 Goal 的 Scene Smoke。局部绕行目标及当前 Smoke 行为以文档后面的“局部绕障 Trajectory Bundle 终点”一节为准。

### 实现与几何来源审计

- `safety_bridge.py` 新增只读 `planning_geometry` 操作；新增独立 `trajectory_bundle_scene_smoke.py` 和对应测试。当前局部目标是将已核验的语义物体绑定转换为同一真实状态下的完整碰撞包围球；全局路线中的控制转换、Snapshot 轨迹验证与实际任务执行仍是后续阶段。
- 审计了当前 MuJoCo **3.3.2** 的编译模型字段与官方粗筛实现：碰撞粗筛使用 `geom_xpos` 为球心、`geom_rbound` 为半径。[MuJoCo 3.3.2 实现](https://github.com/google-deepmind/mujoco/blob/3.3.2/src/engine/engine_collision_driver.c#L118) 支持这一中心/半径约定；本轮没有把可见 Mask、Teacher 名称或 Body 原点直接当成完整几何边界。
- 查询从活动 LIBERO `objects_dict/fixtures_dict` 确认语义绑定的根 Body，逐项检查当前 Geometry Inventory 的 Body/Geom ID 与名称，再遍历完整 Body 子树。包含 `contype/conaffinity` 启用的所有 Geom，以及由显式 Contact Pair 启用的零掩码 Geom；视觉专用且无显式碰撞对的 Geom 单独列出，不作为碰撞边界。语义绑定原先只命中一个可见 Geom，也不会限制完整碰撞 Geom 集合。
- 每个组件读取编译 `geom_rbound`、scratch `geom_xpos`，并用编译 primitive 尺寸或**全部编译 Mesh 顶点**复核半径；包含实际 Geom/Contact Pair margin 及数值余量。以 canonical 根 Body 位置为合并球心，取 `max(组件球心距 + 组件半径)` 得到保守合并球。这是有完整半径证明的物体包围球中心定义，不是放置目标坐标推断。
- 返回 `object_id`、根/子 Body IDs、完整碰撞 Geom IDs、原绑定 Geom IDs、排除的视觉 Geom、每个组件的世界坐标/半径、合并球、同状态标识及来源。缺少有效绑定、无碰撞 Geom、非有限/无效半径、未知或无限几何类型、重叠物体子树等情况明确返回 unavailable/error；不返回部分成功作为完整边界。
- 仅在由积分状态重建的 scratch `MjData` 上做 `mj_forward`。查询前后严格核验完整 integration state、所用编译几何/坐标框架指纹及 live kinematics cache 均未改变。`state_id` 为模型几何、积分状态和任务指令的内容指纹，不是可恢复的 Snapshot ID。未审计的控制/传感等全局回调、外部插件和 flex 模型会明确拒绝查询。

### 目标 G 的真实限制

- 静态审计当前固定任务 BDDL，目标为 `On(搬运物体, 放置目标)`；活动任务注册表可将其关联到实际目标 Body 和附属编译 Site。接口返回这些物理 ID、Body 原点、Site 位置/尺寸及原生谓词，全部标记为诊断信息。
- 审计 `LIBERO-Safety/libero/libero/envs/predicates/base_predicates.py::On` 和 `object_states/base_object_states.py::ObjectState.check_ontop()`：当前实现依赖接触与相对高度，没有给出经验证的、对完整搬运物体球心安全的目标区域。因此**本轮没有可靠依据从目标 Body 原点或 Site 中心生成 G**，也没有加入半径/高度虚构偏移。
- 当前 Bridge 的原生目标审计会返回 `goal.status="goal_unavailable"`、`verified=false`、`position_world_m=null`。这并不否认几何包围球查询本身可以成功。最初的 Scene Smoke 因依赖最终安全放置区域而不生成候选；当前版本已改为依据同状态障碍球和 BDDL `On` 方向构造受限的局部绕行 Goal，详见后续章节。
- 客户端仅在收到同状态、带明确物理目标 Body、`carried_sphere_center` 语义及已验证安全区域声明的目标后，才会调用**原有** `generate_trajectory_bundle(..., candidate_count=18)`。当前原生 `On` 路径没有这样的目标证明。该条件分支的测试使用明确标识的合成协议夹具，不能作为真实目标、18 条真实轨迹或任务成功的证据。
- 局部绕行 Smoke 不再要求最终放置安全球心区域，但通过初筛后仍需要独立控制/物理/任务验证；最终放置区域仍未由本轮证明。本轮不修改 B-Spline 算法、Near–Far、Local Recovery、OSC、Student、Compiler 或训练链路。

### Scene Smoke 的验证与输出

- 解析现有 Safety IR v2，复用 `_validate_current_bindings()` 和 forbid-pair 校验；对选中搬运物体的**全部禁止接触障碍物**请求完整边界。默认从唯一合法搬运 subject 推断；不按物体名称匹配 Geom，不硬编码对象 ID。
- 指令优先取已有语义绑定的 `task_instruction`，需所有已提供声明一致，也可用 `--expected-instruction` 明确传入；与活动 Bridge 的任务指令严格相等。没有指令来源或绑定/规则冲突时，在几何请求前停止。沿用旧绑定不代表完成了当前 RGB 视觉重识别，报告明确记录 false。
- 前后各执行一次只读 `planning_geometry`，重复核对同模型绑定、完整球包围证据、状态指纹、完整 integration state 精确相等、几何边界和目标诊断相同。期间不发送 reset、step、branch 或 restore。若状态变化，禁止保存候选 Bundle。
- 输出目录需为空或不含本 Smoke 的保留文件，避免与旧状态结果混淆。总是尽量保存 `scene_smoke_report.json`、`planning_geometry.json` 和 `planning_geometry_after.json`。只有目标证明与前后状态一致性全部通过，才保存 `scene.json` 及既有生成器的 `trajectory_bundle.json/.svg`、几何通过数量和拒绝统计。
- 报告分别记录几何查询是否成功、目标是否可用、请求/实际候选数和轨迹可执行性。`physical_safety_verified`、`trajectory_executability_verified` 均为 false；goal unavailable 的 CLI 退出码为 2，不应将其误写为规划成功。

### 静态验证结果

- 使用现有 `/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python`，MuJoCo 3.3.2，新增 **15 项**测试及几何 Bundle、Near–Far 距离、控制器回归共 **43 项通过**。只编译合成小模型并调用只读 `mj_forward`，未执行 `mj_step/env.step`，未创建真实 LIBERO 环境、启动 Bridge 服务或模型。
- 覆盖完整子 Body/隐藏碰撞组件、零掩码显式碰撞对、全部 Mesh 顶点、canonical 坐标与 stale live cache、非有限/未知边界、错误绑定/任务、只含视觉 Geom、自定义回调拒绝、原点/Site 不冒充 G、前后状态变化、指令缺失/冲突、旧输出保护和协议输出。
- 静态 AST 比对确认原有 **34 个 Bridge 函数/类定义均未改变**；新增代码仅为规划查询辅助函数与一个 dispatch 分支。真实 L0T0 的查询成功与否仍待人工运行，43 项测试通过不能替代这一结果。
- 另对现有真实 Teacher/语义绑定文件与**归档** `contact_safety_compiler_object_4/bridge_geometry_inventory.json` 做只读离线 preflight，规则、Body/Geom 身份和已有任务指令校验通过。这仅验证文件组合可构造请求，不是当前 Bridge 的完整几何查询结果。
- 三份改动 Python 文件的 `py_compile`、两个新文件的 Ruff lint/格式、新增 Bridge 辅助代码的 Ruff/格式及 `git diff --check` 通过。Bridge 全文件仍有 4 项历史 Ruff 问题（旧导入布局、版权头/空行），与修改前相同、无新增；没有为此调整原有初始化/导入顺序。

### 人工运行命令

服务端需加载新增操作；若已有旧进程，请在你选择的状态时机人工重启。以下仍使用现有固定任务初始化；Smoke 本身不重置或重放环境：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

另一终端只读查询服务端当时的真实状态，输出到新目录：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  -m research.gs_opd.smoke.trajectory_bundle_scene_smoke \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --out-dir logs/gs_opd/trajectory_bundle_scene_v1
```

当时旧版客户端若遇到原生 `On` 关系，会报告 `goal_unavailable`；该行为已由后续局部绕行版本替代。该历史判断不是当前 Smoke 的预期结果。

**检查清单（Checklist）**

- [x] 完整碰撞 Geom 的保守包围球、状态/来源信息及只读 Scene Smoke 已实现；15 项新增及相关回归 43 项通过。
- [x] 放置目标 Body/Site 只作物理诊断，无法证明安全球心目标时明确 goal_unavailable。
- [ ] 人工确认真实 L0T0 包围球查询成功和前后状态完全一致；本轮未运行服务端或真实查询。
- [ ] 真实局部绕行几何及其物理/任务可行性待后续验证；最终放置安全球心区域也仍未被证明。

## 局部绕障 Trajectory Bundle 终点（2026-10-18）

### 变更

- `trajectory_bundle_scene_smoke.py` 从同一次 `planning_geometry` 响应构造 `local_bypass_goal`。起点 S 是搬运物体完整碰撞包围球中心；障碍 O 来自已校验的 forbid-contact 对象包围球。BDDL `On` 目标保留为任务语义参考，并用于定位任务方向上的候选障碍；局部绕行单位方向改为 `D = (O - S) / |O - S|`。
- 局部终点按当前几何计算：`G = O + D × (r_carried + r_obstacle + safety_clearance + exit_clearance + numeric_pad)`。原安全间隙仍为 10 mm；独立出口余量 `exit_clearance_m` 默认 20 mm，可由函数参数或 CLI `--exit-clearance-m` 配置。G 必须越过 O 到达相对 S 的另一侧，并保留两球半径、安全间隙和出口余量；盘子位置不再限制 G 是否越过障碍。旧错误条件正是导致真实日志被拦截的原因：障碍物任务方向投影 0.262936 m、横向 0.210115 m，所需中心距 0.230447 m，G 的任务方向投影越过盘子参考点。
- `trajectory_bundle.py::_scene_input()` 区分 `placement_goal` 与 `local_bypass_goal`。局部 Goal 必须带当前状态下的搬运/障碍球绑定、目标 BDDL `On` 身份与状态、S→O 单位方向及包围球间隙证明；生成器重算并验证 S–O 非退化、方向一致、G 位于障碍另一侧，并保留“原安全间隙 + 出口余量”。不要求局部方向与盘子方向相同，也不要求 G 位于盘子投影之前。旧的 `placement_goal` 输入路径未修改。
- 局部 G 仅表示“越过障碍后的几何采样终点”，不是最终盘面放置位置，也不表示碰撞安全或动作可执行。原有 18 条左右/右侧/上方 B-Spline 与连续曲线保守间隙初筛未改变；JSON/SVG 均显式写明 `local_bypass_goal`。每条候选和 Bundle 仍为 `physical_safety_verified=false`、`executable=false`。
- 成功时输出目录包含前后只读查询 `planning_geometry.json` / `planning_geometry_after.json`、含真实查询球体与局部 G 的 `scene.json`、18 条完整候选及拒绝原因的 `trajectory_bundle.json`、三视图 `trajectory_bundle.svg` 和带几何通过数、拒绝统计、状态一致性及非可执行声明的 `scene_smoke_report.json`。

### 验证与限制

- 轨迹生成器、Scene Smoke、Near–Far 几何 API 和 OSC 控制器相关单测共 **54 项通过**。本次日志数值构成的非共线合成场景中，18/18 条候选通过原连续几何初筛，且 G 的任务方向投影超过盘子参考投影；起点/终点不安全场景保持全拒绝。测试还覆盖 S 与 O 重合、非法方向/状态来源、球形间隙不足、默认/自定义出口余量及全部候选拒绝。
- B-Spline `_screen_curve()` 的采样间误差界与 10 mm 安全下界没有变化；20 mm 出口余量是额外空间，不能视为物理安全证明。
- Python 语法检查、Ruff lint 和 `git diff --check` 通过。`ruff format --check` 指出文件中本次修改区域以外的历史格式差异；为避免无关重排，保留原格式。
- 本轮没有启动 Bridge、连接真实 L0T0 状态或执行环境动作。单测使用小型合成 MuJoCo 模型；真实包围球查询、现场 BDDL 目标解析和真实 18 条候选通过数仍待下方手动 Smoke。静态通过不代表真实轨迹可执行或安全。

### 当前 L0T0 手动 Smoke 命令

在已经处于目标 L0T0 状态的 Bridge 上运行客户端；不要为了本次查询重启或重置正在使用的 Bridge。客户端只发送 `geometry_inventory` 与两次只读 `planning_geometry` 请求，不发送 reset/step。每次使用新的输出目录：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  -m research.gs_opd.smoke.trajectory_bundle_scene_smoke \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --expected-instruction "pick up the akita black bowl on the stove and place it on the plate" \
  --exit-clearance-m 0.02 \
  --out-dir logs/gs_opd/trajectory_bundle_scene_local_bypass_l0t0_v3
```

命令中的 Teacher、语义绑定和任务指令只用于当前已有 L0T0 检查；Smoke 仍会将 Body/Geom 绑定与当前 Inventory、任务指令与 Bridge 活动任务、前后积分状态与几何指纹逐项核对。真实候选数量、球间隙初筛通过数和拒绝原因以这次运行生成的 JSON 为准，运行前不预设结果。

**检查清单（Checklist）**

- [x] 局部绕行方向由同状态搬运球心指向所选障碍球心确定；BDDL `On` 目标保留为任务参考，目标 Body 原点没有被当作 G。
- [x] 54 项相关单测、语法检查、Ruff lint 和 diff 检查通过；日志数值合成场景候选通过数 18/18，不安全起点/终点场景各拒绝 18/18。
- [ ] 在当前真实 L0T0 Bridge 上运行上面的只读 Smoke，检查实际几何包围球、18 条候选的通过/拒绝情况和查询前后状态。
- [ ] 后续仍需用独立 OSC/MuJoCo 完整接触验证确认任何候选的物理安全与可执行性。

## 局部出口区域与多终点轨迹采样（2026-10-18）

### 当前生成行为

- 保留已核验的同状态 Teacher/Body/Geom 绑定、完整障碍物保守包围球、`S→O` 单位绕障方向，以及 10 mm 安全间隙和 20 mm 默认出口余量。BDDL `On` 目标继续校验并记录为任务语义上下文，不参与局部几何方向、出口位置或可行域限制。
- 原单点 G 现在是局部出口区域的中心基准。生成器在垂直于 `S→O` 的平面内建立确定性的正交 `lateral/up` 基；区域半边长为 `local_exit_region_scale × (r_carried + r_obstacle + clearance + exit_clearance)`。默认 `local_exit_region_scale=0.5`，CLI 可设置 `(0, 1]` 范围内的值。区域法向坐标固定为 0，因此每个出口点都处于障碍物后侧的同一出口平面；区域范围随同状态包围球尺度变化，不使用场景世界坐标偏置。
- 默认 18 条候选按左绕、右绕、上绕各 6 条确定性生成。每族的幅度按区域半边长分级；左/右族沿相反 lateral 方向，上绕族沿 up 方向并交替微量 lateral 偏移，因此 18 个 `Gᵢ` 都有不同的三维坐标。每个点都由正交局部基和区域相对坐标重建，不改变 S→O 绕行法向距离。
- `trajectory_bundle.json` 分别保存 `local_exit_region` 与每条候选的 `goal_world_m`、`goal_local_coordinates_m`、方向、完整 Waypoint/B-Spline 控制点、离散曲线、长度、终点误差、逐个绑定障碍物的连续间隙下界和拒绝原因。所有曲线共用同一个真实查询起点 S，并以各自 Gᵢ 作为精确控制点和曲线终点。XY/XZ/YZ SVG 绘出出口区域投影和每条候选的独立终点标记。
- B-Spline 构造、全曲线采样间误差界、10 mm 连续间隙筛选以及全部已绑定障碍物检查没有放宽。端点区域只做保守几何初筛：每个候选仍标记 `physical_safety_verified=false`、`executable=false`，不能作为 MuJoCo 安全或动作可执行证据。

### 静态验证与真实场景状态

- 当前规划器与只读 Scene Smoke 两个测试文件共 **47 项 unittest 通过**。默认合成局部绕行场景中 18/18 条轨迹通过原连续球间隙初筛；测试也覆盖 18 个不同出口点、三族方向、精确曲线端点、尺度退化/越界、附加障碍物、端点不安全和全部候选拒绝。
- 四个相关 Python 文件的语法检查、仓库环境 `/workspace/zy_workspace/RLinf/.venv/bin/ruff check` 及 `git diff --check` 通过；没有安装工具或依赖。
- 本轮没有启动 Bridge、创建/重置仿真、执行环境动作或查询真实 L0T0 几何。以上数量和通过率仅来自合成静态夹具，不能推断真实场景的包围球尺寸、候选通过数、物理安全性或轨迹可执行性。

### 真实 L0T0 只读 Smoke 命令

服务端需运行包含 `planning_geometry` 的当前 Bridge 实现，并处于你要检查的 L0T0 状态。若 Bridge 已经运行，不要为了本 Smoke 重置环境；在另一终端使用新输出目录：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  -m research.gs_opd.smoke.trajectory_bundle_scene_smoke \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --expected-instruction "pick up the akita black bowl on the stove and place it on the plate" \
  --exit-clearance-m 0.02 \
  --local-exit-region-scale 0.5 \
  --out-dir logs/gs_opd/trajectory_bundle_scene_local_bypass_region_l0t0_v1
```

该客户端只请求 Geometry Inventory 和两次只读 `planning_geometry`，用于比较查询前后的状态；不会 reset、step 或执行候选。真实候选通过数以新目录的 `scene_smoke_report.json` 与 `trajectory_bundle.json` 为准。

**检查清单（Checklist）**

- [x] 单点局部 G 扩展为尺度受限的三维局部出口区域；18 条候选分别记录不同 Gᵢ 与完整几何诊断。
- [x] 47 项相关 unittest、Python 语法检查、`git diff --check` 通过；静态合成场景 18/18 几何初筛通过。
- [ ] 在真实 L0T0 Bridge 上执行只读 Smoke 并检查实际候选统计及前后状态一致性。
- [ ] 使用独立完整接触/动力学验证任何候选的物理安全、可执行性及任务可行性。

## Geometry-Guided Trajectory Bundle 闭环跟踪组件（2026-10-18）

### 接口与验证流程

- 新增 `research/gs_opd/repair/trajectory_tracking.py::track_trajectory_bundle()`。它接收当前决策状态的 Snapshot ID、内存中的同状态 `trajectory_bundle.py` Bundle、已核验的搬运对象 Body/Geom binding、Safety IR v2、语义绑定、当前 Geometry Inventory、Gripper Inventory 和活动 `repair_context`。Bundle 的几何候选必须仍标记 `executable=false`、`physical_safety_verified=false`；本接口只跟踪 `passes_geometric_screen=true` 的候选，最多 24 条。
- 新增 Bridge `track_trajectory_bundle` 操作。调用前要求主环境完整 Snapshot 与请求 Snapshot 完全相同，且 Bundle 的模型/积分状态/任务指令指纹一致。每条候选开始前都从该 Snapshot 恢复并验证完整 Snapshot；全部候选结束后恢复并逐项核验调用前的 Snapshot 与 canonical 状态。候选起始积分状态指纹还会跨候选比较；任一恢复或起点校验不通过，所有候选均不可执行。
- Bridge 将物体球心曲线按弧长和 20 Hz 环境控制周期重采样，参考速度默认 0.04 m/s 且不允许提高；默认单候选最多 480 步、整束最多 7,200 步，硬上限分别为 600 和 12,000 步。所需控制步按 `ceil(curve_length / (reference_speed * 0.05))` 预先计算；例如 0.567 m 路径在 0.04 m/s 下需要 284 步。Bridge 与客户端都会在分支动作前检查单候选及整束预算，超限会明确拒绝，不截断路径或提高速度。起始时从同一 Snapshot 的 Body `root_pos/root_quat` 和已验证几何球心计算球心在 Body 局部坐标中的刚性偏移；后续每步都用真实 Body 位姿将该偏移变换回世界系，轨迹参考、EEF 目标、跟踪误差及终点误差一律使用 `sphere_center`。报告分别给出 `actual_object_root_pos_world_m`、`actual_object_root_quat_world_wxyz`、`actual_object_sphere_center_world_m`、`reference_sphere_center_world_m` 及 `sphere_center_body_local_offset_m`，不再假设 root 与球心重合。随后依据真实球心与 EEF 位姿建立 EEF 相对偏移。每步通过活动 OSC_POSE 的世界坐标 delta 和当前 input/output bounds 反算 7 维动作。动作在调用环境前必须通过有限值、归一化 action range、控制器 scaling 一致性和 7 维环境 action spec 检查；没有静默裁剪。目标受控制器 position limits（若配置）和 Panda 模型关节锚点链导出的保守可达半径检查，并在每步核验实际 EEF 工作范围及臂/夹爪关节 qpos limits。若活动控制器 orientation clipping 不能证明关闭，则拒绝该候选。
- 初始抓持不是前提假设：只在活动 `PandaGripper` 的 current close action 有效、MuJoCo 中绑定对象同时与模型定义的左/右指面 Geom 接触时开始执行。每步继续检查两侧接触、夹爪闭合命令及 EEF—物体相对位姿变化；丢失接触、位姿异常、动作越界、Workspace/关节限制失败、提前 done 或路径误差超限都会拒绝。抓持输出明确记录为接触支持的抓持保持证据，不以距离不变单独宣称抓持成功。
- 现有 Compiler 的机器人部位映射只完整支持 `gripper`。IR 中存在 `unknown` 或当前 Compiler 无法映射的其他机器人部位规则时，客户端在发送分支请求前拒绝；所有明确的 gripper 规则目标都会逐一调用现有 Compiler，未分类接触仍会使候选无法通过。
- 每步记录重采样球心参考、Body root 位姿、真实球心及其 Body-local 偏移、EEF 位姿、跟踪误差、实际七维 action、实际 OSC scaling、抓持相对位姿、Contact Monitor 结果、reward/done、关节与工作空间检查及 canonical 状态。结果中的 `goal_sphere_center_world_m` 明确表示球心目标。Bridge 复用 `_run_env_step_with_contact_monitor()` 获取 `env.step()` 内每个 MuJoCo 物理子步；客户端复用真实 `compile_forbid_contact()` 和全部 `object_contact_rules` 检查。允许的抓持接触、Teacher 禁止接触、Teacher 明确允许的其他物体接触和未分类接触分别输出；不明接触及不完整监测 fail-closed。
- 合格条件还要求完整执行到路径终点、有限 reward trace、未提前终止、逐步路径跟踪通过以及抓持接触保持。`task_feasibility` 只声明“本段局部搬运路径已完成且未终止”，不会声称最终任务成功，也不会生成训练标签。每条候选独立保留原始几何路径、所有控制步诊断、Compiler 结果与拒绝原因；只有全部物理、动作、抓持、任务段和恢复验证都通过的 Candidate ID 才进入 `executable_safe_candidate_ids`。

### 静态验证与限制

- 新增 `research/gs_opd/repair/test_trajectory_tracking.py`，当前 22 项 unittest 通过：覆盖双指抓持通过/未抓持拒绝、Body 原点与球心有偏移及旋转变化、零偏移、Bundle 与同状态球心绑定冲突及返回偏移冲突拒绝、目标超出保守 Workspace、OSC 输入越界不裁剪、禁止的物体/夹爪接触、未分类碰撞 fail-closed、抓持丢失、Snapshot 恢复失败、多候选隔离、20 Hz 限速、0.567 m 路径超过旧 240 步上限后仍在新预算内、单候选及整束真实超预算拒绝，以及拒绝通过提高参考速度绕过时长预算。测试通过现有 Contact Compiler，但 Bridge 响应和物理状态使用合成夹具。
- 当前检查只验证了静态代码和 Python 协议替身。没有启动 Safety Bridge、MuJoCo 或 π₀.₅，也没有在真实 Student 持物状态运行；因此没有任何真实 Panda 抓持、动力学跟踪、真实碰撞或 Snapshot 分支成功结论。合成测试不能代替真实 MuJoCo 物理验证。
- 2026-10-19 定点修复将规划包围球 `sphere_center` 与 MuJoCo Body `root_pos` 分开处理：从决策 Snapshot 的球心和 Body root 位姿计算刚性局部偏移，后续按实时 Body 四元数更新球心；路径起点、EEF 目标、误差、终点和报告统一按球心比较。轨迹弧长按完整离散 B-Spline 点列求和，步数以速度上限 0.04 m/s 和实际已校验 20 Hz 周期计算，并受 480/7,200 默认及 600/12,000 硬上限约束。静态回归覆盖 0.567 m 路径需 284 步、单条与整束超预算以及禁止提高参考速度绕过预算。新增代码仍未在真实 MuJoCo 执行。
- Workspace 约束使用活动控制器笛卡尔限制（若有）、模型关节锚点链给出的保守外包半径、实际 EEF 路径检查和关节 qpos limits。外包半径不是完整逆运动学可达性证明；无法跟踪时由每步真实反馈误差拒绝。指面持续接触和相对位姿阈值是操作性保持证据，不证明任意摩擦/载荷下的抓持裕量。mesh/几何候选初筛仍不构成物理安全证明。

## 轨迹跟踪 Body/Geom 绑定契约修复（2026-10-19）

- 检查了真实记录 `logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance_v2.json`：黑碗根 Body 为 `akita_black_bowl_1_main`，语义识别 Geom 与规划查询列出的碰撞 Geom 集分离，且记录中的 `body_ids` 只含根 Body。旧 `planning_geometry.objects.*` 缺少 `semantic_binding_geom_names` 和碰撞 `geom_names`；候选在跟踪前报 `carried Body/Geom binding coverage is inconsistent`。旧记录未改写。
- 统一字段语义：`semantic_binding_geom_ids/names` 保留当前 SAM3/IoU 语义绑定的 Geom；`geom_ids/names` 仅表示完整碰撞 Geom 集。Safety Bridge `planning_geometry` 从当前 Geometry Inventory 返回两组独立 ID/名称，并为每个碰撞 Geom 返回同序 `components`。语义 Geom 可以是 visual-only，也可以同时属于碰撞集合。
- Scene Smoke 的 `_check_query()` 分别核验语义绑定、碰撞 ID/名称顺序、组件和 Inventory 身份。Tracker 客户端分别按 Inventory 核验语义与碰撞 Geom、组件顺序，并确认所选 Candidate 与语义关联一致。Bridge 在跟踪前按活动模型重新核验根 Body/子树、完整有效碰撞 Geom 集、名称和语义 Geom 身份。Contact Compiler 仍只支持碰撞 Geom 位于单一根 Body；跨对象、跨 Body、覆盖不全及不支持的 child-body 碰撞均拒绝，没有增加关节物体支持。
- 新增合成 MuJoCo 接口链回归：从 `planning_geometry` 响应经过 `_check_query()`、Tracker 客户端校验到 Bridge 前置校验；覆盖语义/碰撞 Geom 分离、两者重合、缺失或错序名称、跨对象 Geom、不完整碰撞覆盖及 child Body 拒绝。
- 本次只完成单元与静态验证；未启动 Student、真实 Safety Bridge 或真实物理验收。Step 55 物理结果仍待人工重跑确认。

**检查清单（Checklist）**

- [x] 语义 Geom 与完整碰撞 Geom 使用独立字段并分别校验当前 Inventory 身份。
- [x] 碰撞名称与 Geom ID 顺序严格对应；Bridge 继续复算碰撞集合并 fail-closed。
- [ ] 在真实 Student Step 55 状态重新运行物理验收，确认候选跟踪结果。

## 轨迹束控制预算改为候选级准入（2026-10-19）

- 检查真实记录 `logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance_v3.json`：Step 50 报告 18/18 条几何初筛通过，但 `branch_candidate_count=0`；整批在 `trajectory_requires_498_steps_exceeding_limit_480` 处中止，尚未进入物理分支。
- `plan_bundle_reference_paths()` 现在先为每条完整离散曲线独立计算弧长和所需控制步数。超过 480 步的候选记录具体长度、步数和 `trajectory_requires_N_steps_exceeding_limit_480` 原因；其余候选按方向分组轮询，确定性使用最多 7,200 步整束预算，超过剩余预算的候选单独记录 `trajectory_requires_N_steps_exceeding_remaining_bundle_budget_R`。不截断路径、不提高 20 Hz/0.04 m/s，也不扩大配置上限。
- 仅预算合格候选会发送给 Bridge。客户端将每条分支结果与几何拒绝、单候选预算拒绝及整束预算拒绝按原 Candidate ID 合并，并保留原始顺序；预算拒绝记录固定为不可执行、未进行物理跟踪、无训练标签。控制预算通过不代表物理安全。若没有候选准入 Bridge，则明确记录未发送分支请求，不伪称完成 Snapshot 恢复检查。
- 回归测试复现 18 条候选中前三条各 424 步、其后为 498 步及更长的输入：424 步候选进入 Bridge 请求，超 480 步候选独立拒绝，18 条均可追溯；另验证 18 条 424 步候选时整束准入不超 7,200 步，且左/右/上方向轮询保留多样性。
- 相关单测、语法、Ruff 和 `git diff --check` 已通过；没有启动真实 Student 或 MuJoCo。v3 的物理分支验收仍需人工重跑，不能以预算准入测试代替物理跟踪结果。

**检查清单（Checklist）**

- [x] 18 条候选的预算判定与结果记录相互独立；Bridge 只接收合格项。
- [x] 整束步数受原 7,200 默认上限约束，方向采用确定性轮询。
- [ ] 重跑真实 Step 50 Student 物理验收，核对进入分支的候选及 Snapshot 恢复结果。

### 在 Near–Far 命中的当前决策状态调用

物理验收必须从 **Student 已经真实抓住搬运物体** 的活环境决策状态开始，且调用过程中 Student 尚未执行下一动作。不要从 Bridge 冷启动后的未抓持 L0T0 初始状态运行，也不要用历史 Snapshot、RGB 或动作替代当前状态。最短入口是在持久运行的 Student/On-Policy 进程中，确认当前真实抓持后用同一 Bridge connection 依次读取 Snapshot、活动控制器/几何清单和 `repair_context`，并从这个 Snapshot 生成 `planning_geometry`、绑定和 Bundle，再调用下面接口：

```python
# connection 是已连接到当前 Safety Bridge 的 multiprocessing.Connection。
# bundle 与 binding 必须由当前决策状态生成，Teacher/语义绑定需已校验。
def bridge_request(connection, request):
    connection.send(request)
    response = connection.recv()
    if not isinstance(response, dict) or response.get("error"):
        raise RuntimeError(f"Safety Bridge request failed: {response!r}")
    return response


from research.gs_opd.repair.trajectory_tracking import (
    TrackingConfig,
    track_trajectory_bundle,
)

snapshot_reply = bridge_request(connection, {"op": "snapshot"})
repair_context = bridge_request(connection, {"op": "repair_context"})
geometry_inventory = bridge_request(connection, {"op": "geometry_inventory"})
gripper_inventory = bridge_request(connection, {"op": "gripper_geometry_inventory"})

result = track_trajectory_bundle(
    connection=connection,
    snapshot_id=snapshot_reply["snapshot_id"],
    bundle=current_state_bundle,
    carried_binding=current_state_carried_binding,
    repair_context=repair_context,
    teacher=current_safety_ir_v2,
    semantic_results=current_semantic_bindings,
    geometry_inventory=geometry_inventory,
    gripper_inventory=gripper_inventory,
    config=TrackingConfig(
        max_control_steps_per_candidate=480,
        max_total_control_steps=7200,
    ),
)
print(result["status"], result["executable_safe_candidate_ids"])
```

服务端需使用包含新操作的当前 Bridge 版本。启动命令仍使用既有固定 LIBERO-Safety 初始化；该 Bridge 启动本身会初始化固定场景，**不要在已有运行中的 Student 决策时为该组件单独重启或 Reset 环境**：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

该组件可直接由之后 Near–Far 选中的同状态调用；目前尚未接入 On-Policy 主循环，也没有独立 CLI 或真实执行输出目录。

**检查清单（Checklist）**

- [x] 新增可复用客户端接口和闭环 Bridge 操作；世界坐标物体路径通过反馈目标 EEF 位姿转换为活动 OSC 动作，并复用子步 Contact Monitor、Contact Compiler 和 Snapshot。
- [x] 22 项合成单元测试及 Python 语法检查通过；测试覆盖抓持、球心刚性偏移、路径时长预算、碰撞、约束、恢复和候选隔离。
- [ ] 在真实 Student 已抓持的决策状态确认左右指面真实接触、完整候选闭环、Compiler 判定及分支/调用者状态恢复。
- [ ] 实测 OSC 跟踪误差、控制步预算、Workspace 拒绝率和操作性抓持保持条件是否适用于真实轨迹。

## 一次性真实 MuJoCo Trajectory Bundle 验收入口（2026-10-19）

### Rollout 集成

- `pi05_rollout.py` 新增显式参数 `--trajectory-bundle-physical-validation --trajectory-validation-out <report.json>`。需要 Safety IR v2 与完整 `--semantic-results-json`；搬运对象优先从唯一的 Object-to-Object forbid subject 解析，也可用 `--trajectory-subject-object-id` 消除规则存在多个搬运主体时的歧义。该入口不生成 Trace/NPZ 标签，也不与 probe 或 On-Policy 数据采集模式组合。
- Student 使用既有动作推理与 `env.step()` 流程。Rollout 在每个真实控制步后通过同一个持久 `conn` 请求只读 `grasp_state`；Bridge 按当前 PandaGripper 模型的左右指面 Geom 分类、当前 MuJoCo 接触对、语义绑定对应的当前 Body/Geom 身份及 gripper close action 判断抓持。需要 **3 个连续、状态指纹不同** 的实际状态都满足夹爪闭合阈值与左右指面同时接触，才在下一个 Student 决策边界触发一次验收；绝对 rollout 步数不用于判断抓持。
- 触发时先捕获 Snapshot，再重新读取并记录当前 Safety IR v2 和完整语义绑定文件的 SHA-256 与内容；随后从同一状态重新校验 Teacher 全部 Object-to-Object 绑定、抓持证据、完整规划几何、活动 OSC 配置和 Gripper Inventory。MuJoCo integration state、`state_id` 必须与 Snapshot/抓持样本吻合。复用 `trajectory_bundle_scene_smoke` 的局部绕行目标构造和 `trajectory_bundle.py` 默认 18 条三维出口 Bundle，再调用既有 `track_trajectory_bundle()`。几何初筛拒绝数与通过数都保留，只对初筛通过的路线运行独立 Snapshot 分支。
- 每条实际分支保留预算、控制步、EEF/物体球心误差、root pose 与 sphere center、双指接触和夹爪状态、完整 Contact Monitor、Compiler、任务段与抓持连续性、Snapshot 起点/恢复诊断及拒绝原因。分支函数结束后 Rollout 再显式 Restore 一次并严格比较 canonical Snapshot 诊断；随后通过只读几何查询核对状态一致。Restore/绑定/规则/预算/监测等任何失败都会写为明确阻塞状态。每个候选只在 Snapshot 分支执行，绝不提交主环境，也不产生训练标签。
- 验收结束后原 Student Rollout 继续使用同一个 Bridge 连接，按原动作生成与执行流程前进。验收候选不会影响后续 Student 动作；用于 Rollout 的 `A0` 未改写。若完整 Snapshot 恢复或恢复后几何一致性检查失败，Rollout 会在下一条 Student 主环境动作前停止；Tracker 内候选拒绝但调用者状态恢复通过时，Rollout 可继续原 Student 流程。若 Episode 步数用尽仍无稳定抓持状态，会输出 `stable_grasp_not_found_within_episode`；如果抓持样本不足、预检失败或候选均未通过，也分别保存证据与原因。

### 静态回归及物理结果边界

- 新增 `trajectory_bundle_acceptance.py` 与 14 项轻量测试，覆盖真实左右指面分类、单指/错误物体身份/接触 unavailable fail-closed、连续不同状态稳定性和接触丢失复位、同一 Bridge 连接的 Snapshot/几何/Tracker 调用顺序、未达到稳定抓持时不创建 Snapshot、同状态校验失败后仍执行恢复，以及恢复失败/恢复后几何查询失败的 fail-closed 结果和恢复异常后阻止下一条 Student 动作。Trajectory Bundle/Tracking/Scene/Bridge 与现有 Rollout/On-Policy/probe 回归共 117 项通过；新增 Python 文件语法检查和新模块 Ruff 检查通过，`git diff --check` 通过。
- 这是合成协议和数值回归，不是真实 MuJoCo 结果。本次没有启动 Bridge、π₀.₅、模型或仿真；真实抓持状态、几何通过数、实际分支步数、接触与 Compiler 结果、跟踪误差和恢复结果都尚未产生。全候选被拒绝也代表分支已评估，不代表通过物理验收。

### 手动运行

服务端在第一个终端启动新版 Bridge。服务端初始化固定 LIBERO-Safety 环境；需要在干净的 Episode 开始前启动。不要在已有 Student 状态期间重启它：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py
```

第二个终端启动 π₀.₅。Rollout 自己只建立一个 Bridge `Client`，这个 `conn` 同时承载 reset、实际 Student step、抓持查询、Snapshot、planning geometry、分支跟踪及最终恢复：

```bash
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf:/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/RLinf/.venv/bin/python \
  research/gs_opd/smoke/pi05_rollout.py \
  --max-steps 240 \
  --trajectory-bundle-physical-validation \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --trajectory-validation-out logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance.json \
  --video-out logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance.mp4
```

若 Teacher 中存在多个合法 forbid subject，命令需额外提供经过确认的 `--trajectory-subject-object-id <Teacher Object ID>`。输出 JSON 的 `candidate_count/geometric_screen_pass_count/branch_candidate_count`、`candidate_outcome/candidate_rejection_reason_counts`、每个候选的 `requested_steps/executed_steps`、`contact_compiler`、`task_feasibility`、`grasp_evidence`、`rejection_reasons` 和 `restore` 是实验核查依据；顶层 `main_environment_candidate_execution=false` 与 `training_label_created=false` 应保持不变。

**检查清单（Checklist）**

- [x] 使用主 Rollout 已有 Bridge 连接，新增只读真实抓持查询；只在连续真实双指接触且夹爪闭合后触发一次物理验收。
- [x] 复用现有 Safety IR/绑定复核、几何 Scene Smoke、Bundle、OSC Tracker、Contact Compiler、Snapshot 分支和状态恢复逻辑。
- [x] 117 项相关合成 unittest、语法检查和 diff 检查通过；新模块 Ruff 检查通过。
- [ ] 在真实 π₀.₅ Episode 中确认稳定抓持被检测，并核查所有真实几何候选与分支结果、Compiler、任务段和最终主环境恢复。

## PandaGripper 闭合状态解析修复（2026-10-19）

- 已审计当前使用的 `/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4/robosuite/models/grippers/panda_gripper.py` 和 `gripper_model.py`。`GripperModel.__init__()` 将 `current_action` 初始化为 `zeros(dof)`，Panda 的 `dof=1`，因此初始状态是形状 `(1,)` 的 `[0]`。`PandaGripper.format_action()` 随后以 `[-1, +1] * speed * sign(action)` 更新该字段，`speed=0.01`；源码注释明确 `action=+1` 表示闭合、`action=-1` 表示打开。首次格式化后 `current_action` 是形状 `(2,)` 的相反数执行器对：负/正方向表示闭合，正/负方向表示打开。Panda XML 的左右位置执行器控制范围分别为 `[0, 0.04]` 和 `[-0.04, 0]`，与这组配对符号相符。
- 新增 `repair/panda_gripper_state.py` 作为只读的共享解释器。初始单维零值是有效的中性状态，但不构成闭合证据；双执行器值必须有限、处于 `[-1, 1]` 且大小相等、符号相反。只有负/正配对才产生闭合命令进度，正/负配对明确解释为打开，闭合命令进度达到原有 `0.9` 才报告已闭合。该进度表示 robosuite 控制状态，不是测得的夹指开口或抓持成功；左右指真实接触仍独立检查。
- `safety_bridge.py` 的只读 `grasp_state` 与轨迹跟踪初始/逐步抓持检查现在共用该解析器，并同时返回原始 `current_action`、解析进度/状态、左右指接触及具体拒绝原因。`trajectory_tracking.py` 对每个分支的初始和逐步证据重新解析原始执行器值，并核对 Bridge 的结构化闭合诊断；格式错误、打开方向、低于阈值、单指接触及缺失/矛盾数据均 fail-closed。原有连续 3 个不同状态门控、阈值、OSC、Snapshot、Compiler 和几何逻辑未调整。
- 新增 `test_panda_gripper_state.py` 覆盖初始化零值、闭合/打开执行器配对、部分闭合、形状/方向/范围/非有限值错误和分支证据冲突；测试还直接执行本机 vendored `PandaGripper.format_action()` 方法，核对实际产生的闭合和打开双执行器值。Bridge 抓持查询测试覆盖双指接触下的闭合、初始化中性和打开状态。已有左右指面分类回归继续覆盖双指、单指及不可用接触。目标环境 `unittest` 共 45 项通过（Parser、Bridge 验收、Trajectory Tracking）；Python 语法检查、相关 Ruff 检查和 `git diff --check` 通过。真实 Episode 和 MuJoCo 分支未启动。

重新运行一次 240 步真实物理验收仍使用同一 Bridge 连接。先在第一个终端启动 Bridge，再在第二个终端运行 Rollout：

```bash
# Terminal 1: start the LIBERO-Safety Bridge
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python \
  research/gs_opd/smoke/safety_bridge.py

# Terminal 2: run one 240-step π₀.₅ physical-acceptance Episode
cd /workspace/zy_workspace/RLinf
PYTHONPATH=/workspace/zy_workspace/RLinf:/workspace/zy_workspace/LIBERO-Safety \
  /workspace/zy_workspace/RLinf/.venv/bin/python \
  research/gs_opd/smoke/pi05_rollout.py \
  --max-steps 240 \
  --trajectory-bundle-physical-validation \
  --teacher-json logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json logs/gs_opd/semantic_bindings_l0t0.json \
  --trajectory-validation-out logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance.json \
  --video-out logs/gs_opd/trajectory_bundle_l0t0_physical_acceptance.mp4
```

**检查清单（Checklist）**

- [x] 以本机 robosuite 源码为依据处理 PandaGripper 的初始化与已格式化双执行器状态。
- [x] Bridge 实时抓持、跟踪分支初始/逐步状态及客户端分支验收使用一致的负/正闭合极性与 `0.9` 阈值。
- [x] 合成单元测试通过；真实 240 步抓持统计和物理分支结果等待人工运行核查。
## Trajectory Tracking action bounds wrapper fix (2026-10-19)

- Audited the local LIBERO-Safety source at `/workspace/zy_workspace/LIBERO-Safety/libero/libero/envs/env_wrapper.py`: `OffScreenRenderEnv` inherits `ControlEnv`, whose `env` member is the underlying robosuite task environment. `OffScreenRenderEnv` forwards `robots` and `sim`, while its outer wrapper does not expose `action_spec`.
- Audited vendored robosuite v1.4 source: `RobotEnv.action_spec` concatenates every robot's `action_limits`; Panda `SingleArm.action_limits` concatenates controller limits and gripper limits. The Bridge now reads `env.env.action_spec`, checks its finite 7-D bounds against the active Panda `robot.action_limits`, and retains finite shape, action range, and OSC scaling validation.
- Adjacent accesses in the tracking path (`env.robots`, `env.sim`, and `env.env.control_timestep`) match the audited wrapper API; no further wrapper-attribute mismatch was found.
- Added regression cases for an outer wrapper without `action_spec`, valid nested robosuite bounds matching Panda limits, missing/non-finite/invalid-shape/inconsistent bounds, and malformed/non-finite/out-of-range actions.
- Validation: the Bridge controller, Snapshot, and trajectory tracking unittest modules passed (40 passed, 2 MuJoCo-dependent tests skipped); `py_compile`, Ruff (with the Bridge's existing `CPY001,I001,E402` exclusions), and `git diff --check` passed. Tests ran with Python 3.12.14 from the existing `vllm-env` environment.
- No Student or MuJoCo physical run was started. v5 physical tracking, grasp retention, tracking errors, and contact results remain to be checked manually.
- Run blocker found during handoff: the working-tree `research/gs_opd/smoke/pi05_rollout.py` is currently zero bytes, while `HEAD` (`d685b776`) contains a 116,531-byte version. This file is outside this task's allowed edit scope and was not changed; restore/review it before launching the v5 Student command.

Checklist:

- [x] Read local LIBERO and robosuite v1.4 wrapper/action-limit sources.
- [x] Make the Bridge read bounds from the nested robosuite environment and validate against active Panda limits.
- [x] Preserve fail-closed action shape, finite-value, bounds, and OSC-scale checks.
- [ ] Re-run v5 physical acceptance and inspect executed steps, grasp retention, tracking errors, and contacts.

## Mink 引导关节轨迹与隔离物理执行（2026-10-19）

本轮目标已从固定曲线运动学检查改为：复用 v5 Step 55 的三条原始球心曲线和真实抓持状态，生成 Panda 关节轨迹，并在当前真实场景模型的隔离 `MjData` 中通过力矩执行器逐步执行。当前已完成一条左绕局部轨迹，另外两条保留独立失败结果。没有接入 Student 或正常 Bridge 的执行路径。

### 实现与控制接口

- `research/gs_opd/smoke/mink_panda_model.py` 从当前 LIBERO-Safety 的 `OffScreenRenderEnv` 编译完整真实场景并临时导出 MJB，核对归档任务、452 项积分状态、Panda 七关节名/位置范围、EEF 起点及当前 Body/Geom 名称和归属；从活动 PandaGripper 的 `important_geoms` 读取左右指面，未使用示例机器人或硬编码物体 ID。
- 归档模型指纹 `085817d1…b2e840` 与当前模型 `40fd3ac3…408d2` 不同。该差异单独保留；本轮关节、速度及 EEF 起点在 `1e-6` 内吻合，所有归档 Geom ID/名称/Body 归属与当前模型核对后，使用**当前模型**的碰撞和动力学继续实验。
- `research/gs_opd/smoke/mink_panda_physical_smoke.py` 直接读取 v5 预算合格的 `trajectory-001/left`、`trajectory-002/right`、`trajectory-003/above`。保持原球心 B-Spline 引导，沿采样曲线按 0.02 m/s、20 Hz 重新离散；使用 v5 真实物体姿态、球心—EEF 局部偏移和相对四元数构造 EEF 目标。Mink `FrameTask + PostureTask` 在位置目标优先、姿态误差不超过原 0.20 rad 的条件下调整实际跟踪位置与冗余关节构型。曲线生成器未修改。
- 使用实际编译模型的 `ConfigurationLimit`、经核验的 Panda `VelocityLimit`，固定非手臂自由度；通过 Mink `CollisionAvoidanceLimit` 检查当前机器人自碰撞和机器人—场景 Geom，并在规划 scratch data 中按照刚性抓持假设变换完整搬运物体，对真实 Geom 做接触检查。该 scratch 的物体 qpos 设置仅用于规划检查；物理执行只在初始恢复时设置归档状态，之后通过 `mj_step` 推进，未通过设置 qpos 伪造执行。
- 第一轮完整左/上 IK 路线在真实执行中分别于 771/656 步后因请求力矩超出实际执行器范围被拒绝，没有裁剪继续。最终加入保守的关节参考速度预算 0.6 rad/s、相邻 20 Hz 参考速度变化预算 2.0 rad/s²，并在必要时暂停引导点推进；这些是独立实验的参考整形预算，不是放宽机器人限制。每条轨迹默认最多 1200 个控制步。
- 独立关节执行器使用 robosuite v1.4 `joint_pos.py` 同类质量矩阵 PD 控制，并加入目标关节速度前馈：`tau = M_arm @ (50*(q_ref-q) + 2*sqrt(50)*(dq_ref-dq)) + qfrc_bias_arm`。控制在真实 0.002 s 物理周期更新，位置参考在 20 Hz 关节路径节点之间连续插值。验证真实执行器为单位传动比力矩 motor，并按模型 `actuator_ctrlrange` 检查每次请求；没有把关节角伪装成 OSC 动作、改变控制器配置或静默裁剪力矩。夹爪保持归档中的真实位置控制值 `[0,0]`。
- 每个 `mj_step` 后读取真实接触、力、位置/法向并汇总；区分 Teacher 明确允许的 gripper—payload 抓取接触、禁止的 payload—obstacle 接触、机器人自碰撞、其他机器人—场景碰撞和其他场景物体支撑接触。局部搬运段中的其他 payload 接触保守拒绝。复用现有 `compile_forbid_contact()` 对已绑定搬运物体及全部 Object-to-Object 规则编译，其他机器人接触另外保守拒绝；没有改 Compiler 语义。抓持保持在每个 20 Hz 控制结束状态检查双指真实接触和相对位姿，接触监测本身覆盖所有 500 Hz 子步。
- 每条路线使用同一积分状态新建独立数据，结束后重新恢复并逐项比较完整积分状态。输出保留全部关节参考、实际关节状态、球心/EEF、抓持相对位姿、接触事件、首次拒绝原因及恢复结果。完整结束的路线还需确认从对应家族方向越过障碍物平面并到达远侧；这项方向检查本身不作为碰撞安全证明。

### 已运行的真实结果

实际运行命令（复用已安装的独立环境，任务变更后未再安装依赖）：

```bash
cd /workspace/zy_workspace/RLinf
/workspace/zy_workspace/envs/gsopd_mink/bin/python \
  research/gs_opd/smoke/mink_panda_physical_smoke.py \
  --out-dir logs/gs_opd/mink_panda_physical_v3
```

| 路线 | 完整关节规划 | 真实控制步 | 物理结果 |
| --- | --- | --- | --- |
| `trajectory-001` 左绕 | 842 步 | **842/842** | 局部物理段通过；Compiler safe；21,050/21,050 物理子步完整监测；主状态恢复误差 0 |
| `trajectory-002` 右绕 | 第 127 个引导点拒绝 | 0 | 当前真实 Geom 检出黑碗与木柜接触（动态读取的 `akita_black_bowl_1_g23` / `wooden_cabinet_1_g1`）；未执行 |
| `trajectory-003` 上绕 | 842 步 | 64/842 | 双指接触丢失，立即拒绝；恢复误差 0；已执行前缀的 Compiler safe 不能解释为整条轨迹通过 |

左绕实测指标：球心终点/最大引导误差 **0.0116642 m**；最大 EEF 相对关节计划位置误差 **0.00841391 m**；实际 EEF 相对起点姿态最大偏差 **0.122619 rad**（低于 0.20 rad）；抓持相对位置最大变化 **0.00456042 m**，相对姿态最大变化 **0.139331 rad**；所有控制结束状态均保有左右指接触。实际物理子步的关节速度最大占允许上限 **28.10%**；控制结束状态下 joint5 最小位置余量 **2.67982 rad**。监测事件只有允许的抓取接触与其他场景物体支撑接触，未记录禁止的黑碗—障碍物、机械臂自碰撞或其他机械臂—场景碰撞。

- 原始实验完整报告：`logs/gs_opd/mink_panda_physical_v3/report.json`；各候选的 `*_joint_plan.json` 保存完整七关节参考，`*_physical.json` 保存真实执行状态和接触证据。
- 可比较简洁结果：`logs/gs_opd/mink_panda_physical_v3/experiment_summary.json`。它在原始真实记录上追加方向/出口检查和指标汇总，保留原报告 SHA256 与执行源码 SHA256；这一步没有重新仿真。左绕实测轨迹在障碍物平面上的家族方向投影为 0.255186 m，终点前向投影为 0.221156 m，方向/远侧检查通过。此报告不篡改原始执行记录。
- 可查看三视图：`logs/gs_opd/mink_panda_physical_v3/trajectory-001.svg`（灰：几何引导；蓝：Mink 规划球心；红：真实 `mj_step` 球心轨迹）。另外两条也有独立 SVG。
- 当前代码已把方向/远侧检查和主要指标合并入之后运行的路线结果；已有 v3 的同项结果来自真实轨迹的离线追加检查。完整左绕耗时是 42.1 秒仿真时间，不能解释为真实机器人墙钟耗时或最终放置成功。

### 环境、验证与后续

- 复用 `/workspace/zy_workspace/envs/gsopd_mink`：Python 3.11.16、Mink 0.0.5、MuJoCo 3.3.2、NumPy 1.26.4、qpsolvers 4.13.0、quadprog 0.1.13。此前使用 `uv` 与清华源创建；本轮任务变更后没有重装依赖。真实场景导出使用原 LIBERO Python 3.10.21 / robosuite 1.4.0，未改两个原有虚拟环境或任何锁文件。
- 位置范围来自当前编译模型；Panda 关节速度来自已经核对位置范围的 [Franka 官方 Panda joint limits](https://github.com/frankarobotics/franka_ros/blob/develop/franka_description/robots/panda/joint_limits.yaml)：前四轴 2.175 rad/s、后三轴 2.610 rad/s。没有用 OSC XYZ 范围替代关节速度限制。
- 13 项标准库 unittest 覆盖坐标与刚性偏移、归档路线筛选、接触分类、逐子步汇总、关节参考加速度约束、真实 motor 步进、力矩越界拒绝、完整积分状态恢复和指定绕行侧/远侧检查。它们是合成接口回归，不能替代上述真实场景结果。
- 当前只证明**此 Step 55 状态下的一条独立关节力矩控制局部绕障段**通过了所列检查。没有在正常 OSC Bridge 上执行该轨迹，没有验证最终放置奖励或 Episode 成功，没有生成训练标签。20 Hz 双指/相对位姿证据不等价于证明任意瞬间的刚性无滑移抓持。正式接入仍需在更多 Student 真实抓持 Snapshot 上复验，并明确正常系统的关节轨迹执行入口；不可把本结果直接当成七维 OSC Action Chunk。

**检查清单（Checklist）**

- [x] 当前真实模型及 Step 55 积分状态驱动 Mink 规划，模型指纹差异作为独立诊断保留。
- [x] 完成一条七关节轨迹，并通过真实力矩执行器完整运行 842 步、21,050 个物理子步。
- [x] 保留右绕规划碰撞拒绝和上绕抓持丢失拒绝；没有强行宣布三条均成功。
- [x] 检查完整监测、Compiler、关节限制、抓持相对运动、终点/家族方向以及隔离恢复；保存 JSON/SVG。
- [ ] 扩展到更多真实抓持状态与完整任务，确定正式关节执行接口；尚未接入 On-Policy 主链路或宣称最终安全绕障任务成功。

## 实时 Student → Mink → OSC Snapshot Smoke（2026-10-19）

### 本次范围与必要实现调整

- 本次使用新启动的独立 Safety Bridge 和 π₀.₅ Student Episode；没有加载或重放历史 Step 55 Snapshot、动作或 RGB。Student 通过同一个 Bridge 连接执行 rollout、实时抓持查询、Snapshot 和左绕分支请求。只选中一个左绕几何路线 `trajectory-001`；没有执行右绕、上绕或其他物理候选，没有修改训练目标、生成训练标签或 Push。
- 842 步 Mink 参考的 OSC 分支原先存在 600 步硬上限，通用 `TrackingConfig` 默认值为 480。保留通用默认 480，仅将单候选硬上限和本 Smoke 的 Mink 步数预算设为 1200；实际分支按 Mink 返回的明确步数设置请求预算。OSC 参考速度对齐 Mink 的 0.02 m/s，避免用 0.04 m/s 重采样后与显式 EEF 参考点数不一致。语法及接口探针确认 842 步、0.02 m/s 参考可通过 Tracker 配置和预算接口；本次实时 Mink 规划未成功返回轨迹，因此没有声称当前 Episode 的路线实际为 842 步。
- LIBERO MJB 导出改用既有 Safety Python，Mink worker 继续使用 `/workspace/zy_workspace/envs/gsopd_mink/bin/python`。传递 Mink Python 参数时保留 venv 符号链接，避免解析到不含 Mink 包的基础解释器。当前 `scene.carried` 与 obstacle 条目从同一 Snapshot 的 Geometry 查询补入经验证的 Body/Geom 身份。

### 真实运行结果与首要阻塞

三次 Student Episode 均在线运行到真实双指抓持并在下一个决策边界停止；执行步数分别为 45、50、50。第三次运行在决策步 50 选中左绕 `trajectory-001`，其现场 B-Spline 路径长度为 0.850178531 m。没有旧 Step 55 数据参与。

最终报告：`logs/gs_opd/mink_panda_physical_v3/live_left_osc_snapshot_smoke_20261019_attempt3/live_left_osc_snapshot_report.json`；完整终端输出：同目录 `smoke_console.log`。本次唯一首要阻塞发生在 Mink worker `plan_route()`：

```text
KeyError: 'eef_quaternion_world_wxyz'
  research/gs_opd/smoke/mink_panda_physical_smoke.py:461
```

因此 Mink 规划没有产生可用轨迹，OSC Snapshot 分支实际执行 **0 步**。真实 EEF/黑碗跟踪误差没有样本，记为 unavailable；Contact Compiler 未运行，没有 Compiler verdict。更早的启动错误已定位并绕过：Mink Python 被 `.resolve()` 解析后报告 `No module named 'mink'`；此前 LIBERO 导出误用 Mink 环境时也曾报告 `No module named 'yaml'`。这些不再是最终阻塞。没有继续扩展或尝试其他路线。

| 项目 | 实测结果 |
| --- | --- |
| 最终 Student 执行步数 | 50；在决策步 50 触发当前状态验收并停止下一条 Student 动作 |
| 实时抓持 | 通过；闭合命令比例 1.0，左右指均接触当前黑碗 Body `akita_black_bowl_1_main`，9 个匹配接触 |
| 左绕 Mink 规划 | worker 已启动但在 `plan_route()` 抛 KeyError；未返回 Mink 轨迹 |
| OSC Snapshot 分支步数 | 0/未开始 |
| EEF/黑碗跟踪误差 | unavailable（没有执行 OSC 控制步） |
| Contact Compiler | not run（没有分支步） |
| Snapshot 恢复 | 初始 Snapshot→Restore 最大状态误差 0.0；OSC `repair_context` 完全匹配；收尾 Restore verified。由于分支未执行，这不代表分支后的控制器恢复验证 |

### Snapshot 覆盖审计与验证边界

- `_capture_snapshot()` 保存活动控制器除 `sim` 外的全部实例字段、机器人状态缓冲、夹爪 `current_action`、MuJoCo integration state、任务/观测缓存、RNG 和任务对象属性；不支持的可变对象会使 Snapshot 失败，不会静默略过。Smoke 实时抓持状态下的 Snapshot→Restore 成功，物理诊断最大误差为 0.0，`repair_context` 中 OSC_POSE 配置比较完全匹配，收尾 Restore 也通过。
- Tracker 在 branch 前会将当前完整 `_capture_snapshot()` 与决策 Snapshot 精确比较；但本次未进入 `track_trajectory_bundle()`，所以不能声称真实 OSC branch 修改控制器状态后已完成完整往返验证。当前控制器状态覆盖的实现与 Snapshot 捕获已运行验证，branch 后恢复仍待修复 Mink 规划错误后在同一入口验证。

### 验证与继续条件

- 相关 Python 文件 `py_compile` 通过；842 步规划预算接口探针通过；Mink venv import 与启动参数接口通过。所有这些只验证语法/接口，不代替物理结果。
- Bridge 曾以 PID 151675 从当前工作树启动，运行时能力握手报告 native OSC_POSE、显式 EEF reference、子步 Contact Monitoring 与 full Snapshot restore；Student 退出后已关闭该独立 Bridge。三次报告均记录最终状态 Restore verified。
- 当前继续的唯一条件：修复或确认 `plan_route()` 初始化结果缺少 `eef_quaternion_world_wxyz` 的原因，再用同一条实时左绕入口完成真实分支；在此之前不要扩展候选或据此报告 OSC 跟踪、Contact Compiler 或 branch 恢复通过。

## Mink live-plan 四元数初始化回归修复（2026-10-19）

- 修复前记录：上节 `attempt3` 的真实 Student 抓持正常，但 Mink worker 在 `plan_route()` 第 461 行 append `eef_quaternion_world_wxyz` 时因结果字典未初始化该字段而抛 `KeyError`；Mink 规划未完成，OSC 实际执行 **0 步**，跟踪误差无样本，Contact Compiler 未运行。
- 最小生产修复仅在 `plan_route()` 的 `result` 初始化中加入 `"eef_quaternion_world_wxyz": []`。新增回归测试直接调用真实 `plan_route()`，使用小型 MuJoCo 模型运行到一个规划控制步，并核对关节、EEF 位置、EEF 四元数及球心轨迹长度均为 2；没有手工构造消费端 plan，也没有改 B-Spline、Snapshot 或训练接口。
- 验证：Mink 环境下该 `plan_route()` 回归测试通过；两个 Python 文件的语法编译通过。没有安装依赖。
- 复用 `attempt3` 的 `live_mink_plan_input.json`、MJB 和 metadata，未启动 Student 或 Bridge，独立重跑当前左绕 Mink worker。worker 越过第 461 行并正常完成规划：`trajectory-001/left`，850 个控制步，四类轨迹长度各 **851**（含初始状态），`first_failure=null`，绕行方向/远侧检查 `pass`；worker 自身物理动作数为 0。结果保存在同目录 `live_mink_plan_after_quaternion_fix.json`。当前 live 参考为 850 步，规划预算 1200；旧 480 Tracker 默认值不会限制该 worker 的规划，但本次没有重启 Student，因此没有宣称 OSC Tracker 消费端已执行这条 850 步参考。

| 结果项 | 修复前真实 Smoke | 修复后本次独立 worker |
| --- | --- | --- |
| 第一次失败原因 | `KeyError: 'eef_quaternion_world_wxyz'`，`plan_route()` 第 461 行 | 无；规划完成，`failure_reason=null` |
| Mink 轨迹 | 未返回 | 850 控制步；关节/EEF 位置/EEF 四元数/球心各 851 点 |
| OSC 实际执行步数 | 0 | 0（本次只运行 Mink worker，未启动 Student/Bridge/OSC） |
| EEF/黑碗跟踪误差 | 无 OSC 样本 | 未运行 OSC，故无新样本 |
| 双指抓持 | Student 在线状态通过：闭合比例 1.0，左右指均接触黑碗，9 个匹配接触 | 复用同一 Snapshot 输入；worker 报告规划初始双指接触检查通过 |
| Contact Compiler | 未运行 | 未运行（无 OSC/物理分支） |
| Snapshot 恢复 | 初始 Snapshot→Restore 最大状态误差 0.0，`repair_context` 完全匹配，收尾 Restore verified；分支未开始 | worker 未执行物理动作或修改 Snapshot；没有 branch 后控制器恢复数据 |

- 本轮唯一生产代码改动和真实 worker 输出解决了当前首要规划阻塞；没有用单测代替 MuJoCo 规划。OSC 分支实际控制步数、跟踪误差、Compiler 和 branch 后恢复仍沿用上表所述的未执行状态，不据规划成功推断通过。
- 未启动 Student 是本轮有意采用的验证顺序：上次失败实验的模型文件仍在，先按要求以保存输入单独复跑 Mink worker。没有新增路线、重放历史 Step 55、接管主环境或 Push。

**检查清单（Checklist）**

- [x] 只补齐 `plan_route()` 的 EEF 四元数列表初始化。
- [x] 回归测试真实调用 `plan_route()` 并检查四类轨迹长度一致。
- [x] 用上次保存的 live input 和当前模型独立运行左绕 Mink worker；规划 850 步完成，所有四类轨迹各 851 点。
- [ ] 用新 Student Snapshot 在正常 Bridge 上运行左绕 Mink→OSC 分支，并记录实际控制、跟踪、Compiler 与 branch 后 Snapshot 恢复证据。

## 真实 Student → Mink → 原生 OSC_POSE 左绕 Snapshot Smoke（attempt4，2026-10-19）

- 使用独立的新 Safety Bridge（日志显示 `SAFETY BRIDGE READY`）与现有 `mink_osc_snapshot_smoke.py` 入口；能力握手通过显式 EEF pose、子步接触监测与完整 Snapshot 恢复检查。一次过早连接在 Bridge 监听就绪前返回 `ConnectionRefusedError`，保存在 `bridge_startup_race.log`；Bridge ready 后原入口重跑成功。输出目录为 `logs/gs_opd/mink_panda_physical_v3/live_left_osc_snapshot_smoke_20261019_attempt4/`，没有覆盖前次报告。
- Student 实际运行 50 步，在当前抓持状态触发唯一左绕分支。实时抓持通过：闭合比例 1.0，黑碗 `akita_black_bowl_1_main` 左右指都接触，Snapshot 时 7 个匹配接触。新鲜当前状态下 Mink 规划完成 **847 步**，四类轨迹均 **848 点**；与上一次已保存 Snapshot 的 850 步不同，这是本次 Student Snapshot 的实时路径。
- 原生 `OSC_POSE` 使用显式 Mink FK EEF 参考执行 **108/847 步**。首次拒绝在 0-based 控制索引 107：`gripper_object_relative_rotation_changed_beyond_limit`，抓持相对旋转误差 **0.204605 rad**，超过 0.20 rad 限制。拒绝时左右指仍接触（6 个接触），闭合命令保持 1.0；抓持相对位置误差 0.008205 m。OSC EEF 跟踪误差全段最大 **0.003352 m**，黑碗球心跟踪误差最大 **0.012877 m**；首次拒绝步的 EEF 误差为 0.002956 m，黑碗球心误差为 0.012877 m。接触子步监测完整：108 控制步的 2700/2700 个物理子步均被采样。
- Contact Compiler 总 verdict 为 **`unknown`**：`object_1` 的 Teacher 明确允许夹爪—目标物接触，结果为 `safe`；`object_3` 因 `semantic_disambiguation_result_is_not_an_object` 返回 `unknown`。接触分类记录 101 个事件，其中 19 个 `allowed`、82 个 `unknown`。分支因此拒绝，不能报告为安全通过。
- Snapshot 恢复通过：分支前完整状态与 OSC controller context 的比较均匹配，最大数值误差为 0；108 步分支后恢复 `snapshot-000001`，完整状态比较 0 个 mismatch、最大数值误差 0，恢复后的抓持复核也通过（闭合比例 1.0，左右指接触，7 个匹配接触）。这次完成了真实 branch 后恢复验证。

真实产物：

- 完整 JSON（含每步状态与错误）：`logs/gs_opd/mink_panda_physical_v3/live_left_osc_snapshot_smoke_20261019_attempt4/live_left_osc_snapshot_report.json`
- 真实控制台日志：同目录 `smoke_console.log`；Bridge 日志：`safety_bridge.log`
- Mink 规划：`live_mink_plan.json`；Student 视频：`student_grasp_search.mp4`
- Mink／OSC EEF／黑碗球心轨迹对比：`live_left_osc_snapshot_report.svg`；场景参考轨迹：`current_state_scene/trajectory_bundle.svg`

本次未修改规划算法、Student、Snapshot 或训练接口；没有运行其他路线、重放历史 Step 55、接管主环境或 Push。该左绕分支真实执行并在抓持旋转误差限制处拒绝，Compiler 结果保持 `unknown`，但完整 Snapshot 恢复验证通过。

**检查清单（Checklist）**

- [x] 新 Safety Bridge 与 Student 实际运行；使用一个当前状态左绕路线进入原生 OSC_POSE Snapshot 分支。
- [x] 记录 108/847 OSC 步、首个失败、EEF/黑碗误差、双指抓持、Compiler 和分支后完整 Snapshot 恢复。
- [x] 保存完整 JSON、控制台/Bridge 日志、Student 视频与轨迹 SVG；未自动 Push。
