# GS-OPD 当前局部目标

## 当前主线：恢复单场景过拟合成功执行基线（2026-10-20）

- Episode 风险层调度及其对 rollout 触发的门控已从 `pi05_rollout.py` 移除；Near–Far 独立研究、旧日志和三层采集产物均保留。稳定抓持后原有 B-Spline/OSC 直接路径和通用 Snapshot 候选验收路径不再受 Episode 采样层影响。
- 固定成功基线：原成功示范 `55 Student 抓取 + 749 direct B-Spline/OSC Left + 41 Student 放置 = 845 步，reward=1`；attempt3 LoRA 纯 Student 回合 `711 步成功`。直接 Left 与 18 候选 Snapshot/随机接管路径保持明确区分。
- 已按历史 `launch.json` 用原始 Student 做一次新目录真实复现：第 50 步抓持后启动直接 Left，82 OSC 步后因 `bilateral_grasp_or_gripper_closure_lost` 失败；总 132 步，未碰撞记录、任务未成功。原始 launch 未固定 `--seed`，本次也沿用未设种子；当前第 50 步稳定抓持起点生成 843 个参考点（1082 步预算），历史第 55 步起点为 840 点（1079 步预算），直接路线按实时抓持状态生成。本次不调整算法、不再重试。
- 37 项相关单元测试、语法检查和 CLI 入口检查通过。复现视频/NPZ/JSON 保存在 `logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_restore_20261020_attempt1/`；历史成功数据和配置仍在 attempt1 原目录。详见 `docs/GS_OPD_PROGRESS.md`。

该次入口恢复与一次复现已完成；下一步可从历史成功 `launch.json` 直接进入 `--direct-bspline-osc-left`，使用新输出目录。此入口保留但当前复现并未成功，不能把历史成功当作当前代码的确定性保证。没有训练、修改模型/数据或 Push。

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
