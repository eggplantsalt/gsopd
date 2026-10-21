# GS-OPD 当前局部目标

## 当前状态：π₂ 累计经验训练与存储整理完成，等待复核（2026-10-21）

- 使用累计 LeRobot v2.1 三 Episode 数据集和已验证的三阶段均衡采样，完成 **10,000 optimizer steps**。训练始终从原始 SFT Checkpoint `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft` 加载冻结基座，并新建 rank-32 Action Expert LoRA；`initial_adapter_path: null`、`resume_dir: null`，没有加载或合并 attempt3 π₁ Adapter。
- 基座非 LoRA 权重 667/667 键完整覆盖，抽样权重最大差为 0；667 个基座参数张量冻结，13,860,864 个 LoRA 参数独立训练。AdamW 与学习率调度器从头初始化，固定 LR `1e-4`，未改变 OpenPI Flow Matching、归一化或模型结构。
- 训练 loss 首 100 步均值 `0.025686`、末 100 步均值 `0.001209`。三阶段固定评估覆盖 step 0、2,000、4,000、6,000、10,000；π₂ Adapter 与完整训练权重在相同 Observation 上推理差为 0。
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
