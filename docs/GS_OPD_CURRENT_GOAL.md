# GS-OPD 当前局部目标

## 已完成：完整轨迹 Action Expert LoRA 过拟合与真实 Student 评测（2026-10-19）

- 数据：由原始成功轨迹生成单 Episode、845 步 LeRobot v2.1；包含 Student 抓取 55 步、B-Spline OSC 搬运 749 步、Student 放置 41 步。836 个无 Padding horizon-10 窗口全部保留，包括跨阶段窗口；Observation–Action 逐步对齐。
- 训练：初始 Student + 原 norm stats；RLinf 官方 OpenPI SFT DataLoader / SFTRunner / 原生 Flow Matching Loss；rank-32 `gemma_300m_lora` 只训练 Action Expert LoRA，冻结 VLM 和所有基础权重；三阶段每 Epoch 均衡到各 750 次曝光，完成 3 Epoch / 3375 Step。
- 拟合：固定分阶段窗口 Flow Matching MSE：抓取 0.017022→0.012987，搬运 0.131135→0.087699，放置 0.028254→0.023418；整体 0.058804→0.041368。独立 adapter 与完整训练 Checkpoint 在固定真实 Observation 上推理完全一致。
- 评测：LIBERO-Safety `obstacle_avoidance` L0T0 初始状态 0，纯 Student 使用 Horizon 10 和原生 7D Action；设置 1000 步预算后在 120 步因 reward=1、`done=True` 成功终止。原始物理轨迹在 step 40–119 共记录到 80 个双指同时接触目标碗的控制步样本。
- 训练与评测报告、adapter、MP4 和完整 NPZ 见 `docs/GS_OPD_PROGRESS.md` 最新章节和 `logs/gs_opd/sft/pi05_libero_full_trajectory_lora_20261019_attempt1/`、`logs/gs_opd/eval/pi05_libero_full_trajectory_lora_20261019_attempt1/`。

本次固定状态的纯 Student 闭环任务真实成功；结果不代表多初始状态泛化，也不构成安全证明。没有修改 `object_3` 语义绑定或让 Compiler 输出 safe，没有连续重训、批量采集或 Push。
