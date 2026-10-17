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

每个 π₀.₅ 决策后，Rollout 在当前状态创建 Snapshot，完整分支评估 A0，并调用 Contact Compiler。明确 `violation` 时调用既有 `run_verifier_guided_repair()`，候选从同一 Snapshot 独立分支检查；`safe` 或 `unknown` 时跳过 Repair。分支结束后主进程继续执行 Student 原始 A0，绝不把 A+ 发给主环境。安全完整的 A0 可作为有效安全标签；只有 Repair 完整 Chunk、Compiler、安全监控、任务进度证据和动作约束全部通过时，A+ 才是有效标签。未知、无有效 Repair、短执行 Chunk 均不产生训练标签。Snapshot/Branch Rollout 无法确认主状态恢复时，立即停止 Episode，决策已记录但不执行主环境动作。

NPZ 继续包含 Trace 的原始双相机图像、状态、完整 A0 和实际主环境动作/后置 Observation。新增 `decision_safety_json` 保存每个决策的完整 Teacher IR、任务指令和对象绑定、A0 Compiler 输出、Repair 状态、候选搜索摘要、A+（若有效）、`training_action_chunk`、`training_label_valid`、失败原因和分支恢复标志。为控制体积，Repair 摘要保留每个搜索候选的动作、幅度/窗口、Safety/进度/约束结论与失败原因，不重复嵌入候选分支的完整物理状态；A0 Contact Compiler 证据及其完整 Contact Monitoring 仍保留。NPZ 外同时生成同名 `.summary.json`，记录 Episode 内 `safe/violation/unknown` 数量、Repair 调用次数、有效 A+ 数量、有效训练标签数、Repair 状态分布、失败原因和候选拒绝原因。

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
- 新增轻量测试覆盖 safe/unknown 跳过 Repair、violation 复用 Snapshot 并记录 A+、主状态恢复失败时标记必须终止、Episode 计数，以及可选 NPZ 标签落盘和普通 Trace 兼容性。完成测试/语法检查不代表真实 LIBERO-Safety 场景已成功采集；首次实际运行需人工检查 summary、`decision_safety_json` 和 `training_label_valid`。
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
