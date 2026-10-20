# GS-OPD 当前局部目标

## 已完成：845 步完整轨迹 LoRA 过拟合，恒定 1e-4（2026-10-20）

- 从原始 Student 初始化新 rank-32 Action Expert LoRA；非 LoRA 权重完整加载，冻结范围、完整 845 帧 Episode、三阶段均衡采样、官方 OpenPI DataLoader、RLinf `SFTRunner`、原生 Flow Matching Loss 和归一化保持不变。AdamW 重新初始化，LR 恒定 `1e-4`、零 Warmup、无衰减。
- 完成 10,000 optimizer steps（约 89.6 分钟），在训练前及 2,000/4,000/6,000/10,000 步对同一 48 个真实 Observation、固定随机条件评估三个阶段的 Flow MSE 与反归一化 10×7 Action MAE。所有训练 Loss、梯度范数和 LR 有限；不再因单个有限 Batch 尖峰停止。
- 搬运阶段最终 Flow MSE 为 0.000982（基线 0.152026），真实动作 MAE 为 0.009061（基线 0.183005）。独立 Adapter 已成功导出并重载验证，与完整训练 checkpoint 的固定 Observation 推理最大动作差为 0。
- 报告、Loss 曲线、逐步/定点评估日志、完整 checkpoint 与 Adapter 位于 `logs/gs_opd/sft/pi05_libero_full_trajectory_lora_constant_lr_20261020_attempt3/`。没有启动 Student 或 MuJoCo；此结果只确认示范状态上的拟合，不证明闭环泛化或安全。

该局部训练目标已完成，后续等待新的明确目标。原始 Student Checkpoint、成功示范和 LeRobot 数据集保留；本次成功实验权重保留。未 Push。总体路线见 `docs/GS_OPD_ROADMAP.md`。
