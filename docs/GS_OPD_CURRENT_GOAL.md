# GS-OPD 当前局部目标

## 已完成：恢复 B-Spline 出口基线并诊断 attempt3 Student 接管（2026-10-20）

- `local_exit_region_scale` 的 `BundleConfig` 与 CLI 默认值恢复到已验证的 0.5，相关测试断言同步；`test_trajectory_bundle` 27 项通过。
- 只离线分析已有 attempt3 主环境轨迹：422 步 OSC 安全完成到 Above 局部出口，随后 Student 528 步使 EEF 朝远离盘子的方向升高并偏移；任务失败更符合 Student 接管后的行为。后续 Student 段缺少逐步黑碗物理坐标，因此不把 EEF 路径冒充为黑碗路径。
- `trajectory-003` 路线第 386 步存在通过几何间隙和当步抓持/接触检查的较早候选检查点，但该位置未测试控制权交接且比 OSC 终点更远离盘子，暂不作为新的默认方案。
- 本次没有运行 MuJoCo、训练或改动规划/安全逻辑。改进方向仅记在进度报告中，待明确批准后再实施。

## 已完成：attempt3 LoRA 训练与纯 Student 闭环评测（2026-10-20）

- 从原始 Student 初始化 rank-32 Action Expert LoRA，完成 10,000 optimizer steps。冻结范围、完整 845 帧 Episode、三阶段均衡采样、官方 OpenPI DataLoader、RLinf `SFTRunner`、原生 Flow Matching Loss 和归一化保持不变；搬运阶段固定示范 Observation 的 Flow MSE 从 0.152026 降至 0.000982，10×7 Action MAE 从 0.183005 降至 0.009061。
- attempt3 Adapter 与原始 Student Checkpoint 组合后，在 LIBERO-Safety obstacle_avoidance L0T0 初始状态 0 完成一次纯 Student 回合：711 个环境动作步，reward 1.0，`done=True`；Adapter runtime 日志确认 rank 32、216 张量、13,860,864 参数已加载。未使用 B-Spline/Mink 接管或在线修正。
- 原生 20 Hz OSC_POSE、action horizon/chunk 10、7D 执行动作和原 norm stats 保持；保存了完整 MP4、NPZ（七维动作、双相机 Observation、EEF 与黑碗物理状态、接触记录）、Safety Compiler 审计和路线对比图。针对 `object_1→object_4` forbid 规则，Compiler 对 17,775 个子步完成审计并返回 `safe`；保守包围球余量仅约 0.19 mm，且该结论依赖现有 `object_4` 语义绑定（IoU 0.1427），不得外推到其他物体或规则。
- 本次产物位于 `logs/gs_opd/eval/pi05_libero_full_trajectory_lora_constant_lr_attempt3_obstacle_avoidance_l0t0_20261020/`；详细结果见 `docs/GS_OPD_PROGRESS.md` 和 `evaluation_report.json`。之前 LoRA Student 路径已按历史清理记录删除，因此图中明确标缺失。

训练与单回合评测目标均已完成，后续等待新的明确目标。初始 Student、成功示范、LeRobot 数据、attempt3 Adapter 及本次评测原始证据保留。总体路线见 `docs/GS_OPD_ROADMAP.md`。
