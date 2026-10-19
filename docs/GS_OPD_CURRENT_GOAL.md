# GS-OPD 当前局部目标

## 已完成：单轨迹 π₀.₅ SFT 过拟合（2026-10-19）

- 数据：`logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/lerobot_bspline_osc_v21`，749 帧、1 Episode；训练限制为 740 个完整 10 步窗口，末尾 padding 窗口全部排除。
- 初始权重：`/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft`，未覆盖；训练只使用原有 π₀.₅ / `pi05_libero` / Flow Matching Loss / norm stats。
- 结果：单卡 RTX 4090、冻结 VLM、batch 2、3 Epoch、1110 optimizer step；固定 32 个训练窗口的模型空间 Flow Matching MSE 为 0.108403 → 0.000176912。
- Checkpoint：`logs/gs_opd/sft/pi05_libero_lerobot_bspline_osc_v21_overfit_20261019_attempt4/checkpoints/global_step_1110/actor/`；RLinf 模型工厂重载成功，训练后 Action Expert 参数签名差值为 0。
- 报告：`docs/GS_OPD_PROGRESS.md` 的“单轨迹 π₀.₅ OpenPI SFT 过拟合”章节及输出目录内 `report.json`、`step_metrics.jsonl`、`checkpoint_reload.json`、`run.log`。

本局部目标已完成。该固定评估集合来自训练 Episode，只能证明这条轨迹可被模型拟合；不构成泛化或 MuJoCo 闭环验证。原始数据、初始 Checkpoint、Student 和控制链路均保持原样；本轮没有启动 MuJoCo、Student 或批量采集，也未 Push。

后续若继续，按新的任务范围决定是否增加真实 Episode 或评估该独立 Checkpoint；不得把这次单轨迹训练结果表述为新 Student 的闭环成功。
