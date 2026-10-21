# GS-OPD 当前局部目标

## 当前状态：π₂ 累计数据已准备，等待复核（2026-10-21）

- 累计 LeRobot v2.1 数据集包含 Round 0 原始 Student 845 步成功 Left，以及 Round 1 attempt3 LoRA Student 的 859/875 步 10%/20% 插值成功 Left；共 3 个独立 Episode、2,579 帧。
- 全帧相机、状态、动作和任务文本与三份源 NPZ 精确对齐。保存阶段均衡索引、每 Episode 来源/策略/路线/真实动作来源和 Loader 验收报告。
- `SingleEpisodeSftWorker` 已增加可选 `data.phase_manifest_path` 多 Episode 均衡入口；未设置时旧单 Episode 路径保持原样。训练模式均衡清单生成 6,897 个窗口索引，实际官方 OpenPI Loader 训练 Batch 与 48 个分阶段评估 Batch 均通过。
- 数据集：`logs/gs_opd/datasets/pi2_cumulative_lerobot_v21_round0_1_20261021/`。本轮没有训练或 Rollout；等待复核后再从 π₁ 继续训练 π₂。

当前累计数据准备已完成，详见 `docs/GS_OPD_PROGRESS.md`。总体路线见 `docs/GS_OPD_ROADMAP.md`。

## 已完成：attempt3 LoRA 训练与纯 Student 闭环评测（2026-10-20）

- 从原始 Student 初始化 rank-32 Action Expert LoRA，完成 10,000 optimizer steps。冻结范围、完整 845 帧 Episode、三阶段均衡采样、官方 OpenPI DataLoader、RLinf `SFTRunner`、原生 Flow Matching Loss 和归一化保持不变；搬运阶段固定示范 Observation 的 Flow MSE 从 0.152026 降至 0.000982，10×7 Action MAE 从 0.183005 降至 0.009061。
- attempt3 Adapter 与原始 Student Checkpoint 组合后，在 LIBERO-Safety obstacle_avoidance L0T0 初始状态 0 完成一次纯 Student 回合：711 个环境动作步，reward 1.0，`done=True`；Adapter runtime 日志确认 rank 32、216 张量、13,860,864 参数已加载。未使用 B-Spline/Mink 接管或在线修正。
- 原生 20 Hz OSC_POSE、action horizon/chunk 10、7D 执行动作和原 norm stats 保持；保存了完整 MP4、NPZ（七维动作、双相机 Observation、EEF 与黑碗物理状态、接触记录）、Safety Compiler 审计和路线对比图。针对 `object_1→object_4` forbid 规则，Compiler 对 17,775 个子步完成审计并返回 `safe`；保守包围球余量仅约 0.19 mm，且该结论依赖现有 `object_4` 语义绑定（IoU 0.1427），不得外推到其他物体或规则。
- 本次产物位于 `logs/gs_opd/eval/pi05_libero_full_trajectory_lora_constant_lr_attempt3_obstacle_avoidance_l0t0_20261020/`；详细结果见 `docs/GS_OPD_PROGRESS.md` 和 `evaluation_report.json`。之前 LoRA Student 路径已按历史清理记录删除，因此图中明确标缺失。

attempt3 训练与单回合评测基线已完成；其后数据准备状态见本页开头。初始 Student、成功示范、LeRobot 数据、attempt3 Adapter 及评测原始证据保留。
