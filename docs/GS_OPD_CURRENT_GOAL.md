# GS-OPD 当前局部目标

## 已完成：恒定 1e-4 学习率的完整轨迹 LoRA 过拟合尝试（2026-10-20）

- 按用户选择，从原始 Student 初始化新的 rank-32 Action Expert LoRA；冻结范围、845 步 LeRobot Episode、三阶段均衡采样、官方 OpenPI DataLoader、RLinf `SFTRunner`、原生 Flow Matching Loss 与归一化保持不变。AdamW 重新初始化，学习率恒定 `1e-4`，无 Warmup，预算 500 步。
- attempt1 第 1 步因 LR 浮点比较过严误停；最小修正容差后，attempt2 到第 18 步触发 Loss 发散保护（0.444459，为前 10 步中位数的 11.94 倍），Loss/梯度/LR 均有限。未降 LR 或继续训练。
- 固定 Observation 的训练前指标见 `docs/GS_OPD_PROGRESS.md`；未到达 100/200/300/500 步评估，故不能判断搬运拟合是否改善。全量失败 Checkpoint 已删除，仅保留逐步记录与报告；本轮没有可用 Adapter。

该局部实验已停止。原始 Student Checkpoint、成功示范和 LeRobot 数据集仍保留。没有启动 Student 或 MuJoCo，也没有 Push。总体路线见 `docs/GS_OPD_ROADMAP.md`。
