# GS-OPD 当前局部目标

## 当前状态：L0T1/state0 几何 Grounding 采集完成，等待复核（2026-10-21）

- Safety Bridge 已支持 benchmark、level、task、init-state、seed 和相机分辨率参数；默认行为仍是 obstacle_avoidance/L0T0/state0、seed 0、256×256。`reset` 使用同一套场景参数。
- 本地核验后选用不同 BDDL 的 obstacle_avoidance/L0T1/state0：`pick up the book and place it in the back compartment of the caddy`，50 个 init states。只执行初始化/观察采集，没有 Student、动作 step、规划、Repair 或训练。
- 新增只读 `observe` 后，正式 Grounding 从 Bridge 启动的初始化观测开始，随后读取 `geom_segmentation` 和 `geometry_inventory`，采集客户端不 reset/step。两相机 512×512 RGB、Geom segmentation 512×512 int32；同状态前后 Snapshot diagnostics 与双相机缓存图像一致，agentview RGB 与 segmentation RGB 原始方向逐像素匹配。
- 完整 Geometry Inventory 含 28 Bodies、288 Geoms；book/caddy/bottle 的 BDDL 实体名命中三个不同 Body，且各有可见 Geom 像素。Teacher “white book” 与 BDDL `black_book_1` 的颜色/身份关系仍未确认；brown caddy、green bottle 也尚无 Teacher 语义绑定。
- 前次 Teacher 实际输入图像另存为 `teacher_input_reference/` 并仅作对照。本次与之像素不同，虽 task/init-state 配置相同，不能确认实际物理状态一致，因此不跨 reset 混配。完整 RGB、Geom ID、overlay、Body/Geom 清单和对齐报告在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_init0_geom_grounding_20261021/`。
- 两相机 512×512 RGB PNG、原始 RGB NPZ、场景元信息、BDDL 对比、冻结 Qwen3.8 Teacher 原始与校验 JSON、Schema 结果和 SHA-256 清单保存在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_init0_scene_observation_20261021/`。Safety IR v2 输出 6 个语义物体、6 条机器人接触规则、1 条被搬运 book→bottle 禁止接触规则；后侧隔间可见性和 bottle 相对位置仍有不确定性。无任何旧场景 Teacher JSON 或物体绑定参与。
- 当前等待复核 Grounding 图像、物理对象可分性和书籍颜色/身份差异；复核前不运行 SAM3/VLM 语义消歧，不启动 Student Rollout。

## 已完成：π₂ Batch Size 3 纯 Student 闭环评测（2026-10-21）

- 在原三 Episode π₂ 累计数据集上，从原始 SFT Base 新建 rank-32 LoRA，完成一次 10,000 步训练。相较既有 π₂ 配置只把 `actor.micro_batch_size` 和 `actor.global_batch_size` 从 2 改为 3，并使用独立实验标识；学习率、数据采样、模型与 Flow Matching 设置保持一致。
- 训练前确认官方训练 DataLoader 的 6,897 个窗口覆盖三个 Episode，batch size 为 3，三个阶段各曝光 2,299 次；实际 Batch 3 前向/反向通过，峰值 allocated/reserved 显存约 17.64/22.35 GiB。正式训练完成 10,000/10,000 步，保存 5k/10k Checkpoint、阶段固定评估和可独立重载的 Adapter。
- 独立从两个训练的原始 `step_metrics.jsonl` 重算后，Batch 2 首/末 100 步 Loss 均值为 `0.0256858893` / `0.0012088185`；Batch 3 为 `0.0253006398` / `0.0011320326`。Batch 3 进度报告中误抄的 Batch 2 数值已更正，训练日志与权重未改动。
- 使用 Batch 3 Adapter 做了一次纯 Student 闭环 Episode：848 步、reward `1.0`、`done=True`。完整子步监测下显式 Teacher 接触规则编译结果为 `safe`，但整体物理安全仍为 `unknown`。前 60 步 EEF 路线与 π₁ 对齐点平均距离 3.83 mm；Batch 2 失败轨迹对应值为 8.97 mm。
- 训练产物位于 `/opt/data/private/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_bs3_20261021_attempt1/`；闭环与对比证据位于 `/opt/data/private/gs_opd/eval/pi05_pi2_cumulative_fresh_bs3_obstacle_avoidance_l0t0_state0_seed20261020_20261021_attempt1/`。详情见 `docs/GS_OPD_PROGRESS.md`；等待复核，不继续 Rollout 或训练。

## π₂ 原始累计经验训练与存储整理基线（2026-10-21）

- 使用累计 LeRobot v2.1 三 Episode 数据集和已验证的三阶段均衡采样，完成 **10,000 optimizer steps**。训练始终从原始 SFT Checkpoint `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft` 加载冻结基座，并新建 rank-32 Action Expert LoRA；`initial_adapter_path: null`、`resume_dir: null`，没有加载或合并 attempt3 π₁ Adapter。
- 基座非 LoRA 权重 667/667 键完整覆盖，抽样权重最大差为 0；667 个基座参数张量冻结，13,860,864 个 LoRA 参数独立训练。AdamW 与学习率调度器从头初始化，固定 LR `1e-4`，未改变 OpenPI Flow Matching、归一化或模型结构。
- Batch Size 2 训练 loss 首 100 步均值 `0.025686`、末 100 步均值 `0.001209`。三阶段固定评估覆盖 step 0、2,000、4,000、6,000、10,000；π₂ Adapter 与完整训练权重在相同 Observation 上推理差为 0。
- 完整训练报告、逐步 Loss、配置、预检、step 0 与阶段评估、5,000/10,000 步 Checkpoint、独立 π₂ Adapter 及导出验证均保存在 `logs/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_20261021_attempt1/`。
- 本轮没有 MuJoCo Rollout、物理安全判定或新任务成功结论；等待复核训练与评估结果后再决定下一步。
- π₂ 训练目录已完整校验并迁至 `/opt/data/private/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_20261021_attempt1/`；原 workspace 路径保留为软链接。后续 SFT、评测、Rollout 和新数据集默认输出已分别指向数据盘 `sft/eval/rollouts/datasets`。旧 SFT 基座、π₁ Adapter、成功 Left 和累计数据集仍从原路径读取正常。
- 存储迁移与路径读取验证已完成；不启动训练或 Rollout，等待复核后再做 π₂ 与 π₁ 闭环评测。

累计数据来源与训练评估详见 `docs/GS_OPD_PROGRESS.md`。总体路线见 `docs/GS_OPD_ROADMAP.md`。

## 已完成：attempt3 LoRA 训练与纯 Student 闭环评测（2026-10-20）

- 从原始 Student 初始化 rank-32 Action Expert LoRA，完成 10,000 optimizer steps。冻结范围、完整 845 帧 Episode、三阶段均衡采样、官方 OpenPI DataLoader、RLinf `SFTRunner`、原生 Flow Matching Loss 和归一化保持不变；搬运阶段固定示范 Observation 的 Flow MSE 从 0.152026 降至 0.000982，10×7 Action MAE 从 0.183005 降至 0.009061。
- attempt3 Adapter 与原始 Student Checkpoint 组合后，在 LIBERO-Safety obstacle_avoidance L0T0 初始状态 0 完成一次纯 Student 回合：711 个环境动作步，reward 1.0，`done=True`；Adapter runtime 日志确认 rank 32、216 张量、13,860,864 参数已加载。未使用 B-Spline/Mink 接管或在线修正。
- 原生 20 Hz OSC_POSE、action horizon/chunk 10、7D 执行动作和原 norm stats 保持；保存了完整 MP4、NPZ（七维动作、双相机 Observation、EEF 与黑碗物理状态、接触记录）、Safety Compiler 审计和路线对比图。针对 `object_1→object_4` forbid 规则，Compiler 对 17,775 个子步完成审计并返回 `safe`；保守包围球余量仅约 0.19 mm，且该结论依赖现有 `object_4` 语义绑定（IoU 0.1427），不得外推到其他物体或规则。
- 本次产物位于 `logs/gs_opd/eval/pi05_libero_full_trajectory_lora_constant_lr_attempt3_obstacle_avoidance_l0t0_20261020/`；详细结果见 `docs/GS_OPD_PROGRESS.md` 和 `evaluation_report.json`。之前 LoRA Student 路径已按历史清理记录删除，因此图中明确标缺失。

attempt3 训练与单回合评测基线已完成；其后数据准备状态见本页开头。初始 Student、成功示范、LeRobot 数据、attempt3 Adapter 及评测原始证据保留。
