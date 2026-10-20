# GS-OPD 总体路线

## 总体目标

在 π₀.₅ 原生动作空间和现有安全约束下，让 Student 从真实成功执行数据学习避障操作，并在 MuJoCo 中完成可复现的 Student 闭环任务。数据来源、动作来源和安全判定要可追溯；Compiler 的 `unknown` 不得伪装成 `safe`，普通行为克隆动作不得误标为安全修正。

## 阶段与当前状态

1. **安全输入与物体语义**：沿用 Teacher、Safety IR、实时几何绑定和 Contact Compiler；`object_3` 尚缺真实语义绑定时继续保持 `unknown`，不补造映射。
2. **真实成功执行**：已有一次真实 Student→B-Spline→原生 OSC 左绕成功回合：749 步 B-Spline 左绕后由 π₀.₅ 接续 41 步，取得真实任务奖励 1.0；完整轨迹和视频保存在 `live_left_bspline_osc_adaptive_20261019_attempt1`。
3. **数据格式与读取**：已有 749 步 OSC 专用 LeRobot 数据集；现已另从原始成功 NPZ 生成完整 845 步单 Episode（55 抓取 / 749 搬运 / 41 放置），逐步验证 Observation–Action 对齐，保留 836 个 horizon-10 无 Padding 窗口及跨阶段窗口，继续使用初始 Student 对应 norm stats。
4. **单轨迹学习链路**：此前 845 步 rank-32 Action Expert LoRA 过拟合和续训的派生权重已按空间清理要求删除，汇总保留在进度记录。2026-10-20 从原始 Student 初始化的新恒定 `1e-4` LR 尝试复用了官方 OpenPI DataLoader / `SFTRunner` / 原生 Flow Matching Loss 与三阶段均衡采样；step 18 Loss 达到前 10 步中位数的 11.94 倍，按停止保护终止。未到达 step 100 固定评估，也没有保留 Checkpoint 或 Adapter。
5. **训练后 Student 评测与后续验证**：此前 rank-32 LoRA Checkpoint 在 LIBERO-Safety `obstacle_avoidance` L0T0 初始状态 0 的纯 Student 闭环真实完成任务；单回合为 120 步、reward 1.0、`done=True`，已启动的 10 回合固定状态评测为 10/10 成功。之后的离线配对诊断显示 LoRA 在固定示范 observation 上更接近 B-Spline OSC 搬运动作，但抓取和放置第一动作误差变大；本次 Adapter 继续拟合又因搬运固定误差停滞而提前结束。上述都只是示范拟合或固定初始状态结果，不构成多状态泛化或安全证明。`object_3` 仍无真实语义绑定，Compiler `unknown` 不变。此次没有再启动 Student、MuJoCo、Bridge 或批量采集；后续工作等待新的明确目标。

## 主要记录

- 实验与历史阻塞：`docs/GS_OPD_PROGRESS.md`
- 当前局部目标：`docs/GS_OPD_CURRENT_GOAL.md`
- 本地数据与原始 Student Checkpoint 仍保留；过拟合派生权重、重复 DCP、训练/评测日志及视频轨迹已于 2026-10-20 清理（约 26.43 GiB），数值结论保留在进度记录。
