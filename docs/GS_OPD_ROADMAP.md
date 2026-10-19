# GS-OPD 总体路线

## 总体目标

在 π₀.₅ 原生动作空间和现有安全约束下，让 Student 从真实成功执行数据学习避障操作，并在 MuJoCo 中完成可复现的 Student 闭环任务。数据来源、动作来源和安全判定要可追溯；Compiler 的 `unknown` 不得伪装成 `safe`，普通行为克隆动作不得误标为安全修正。

## 阶段与当前状态

1. **安全输入与物体语义**：沿用 Teacher、Safety IR、实时几何绑定和 Contact Compiler；`object_3` 尚缺真实语义绑定时继续保持 `unknown`，不补造映射。
2. **真实成功执行**：已有一次真实 Student→B-Spline→原生 OSC 左绕成功回合：749 步 B-Spline 左绕后由 π₀.₅ 接续 41 步，取得真实任务奖励 1.0；完整轨迹和视频保存在 `live_left_bspline_osc_adaptive_20261019_attempt1`。
3. **数据格式与读取**：已有 749 步 OSC 专用 LeRobot 数据集；现已另从原始成功 NPZ 生成完整 845 步单 Episode（55 抓取 / 749 搬运 / 41 放置），逐步验证 Observation–Action 对齐，保留 836 个 horizon-10 无 Padding 窗口及跨阶段窗口，继续使用初始 Student 对应 norm stats。
4. **单轨迹学习链路**：此前的 1110 Step 全参数 Expert 训练与当前 3375 Step rank-32 Action Expert LoRA 结果分别保留。当前 LoRA 实验冻结 VLM 和全部基础权重、三阶段均衡训练，分别报告抓取/搬运/放置固定 Flow Matching 拟合；独立 adapter 能重载，且与完整 SFT Checkpoint 在固定 Observation 上推理完全一致。
5. **训练后 Student 评测与后续验证**：LoRA checkpoint 在 LIBERO-Safety `obstacle_avoidance` L0T0 初始状态 0 的纯 Student 闭环中，以 7D 原生动作、horizon 10 完成真实任务。此前单回合为 120 步、reward 1.0、`done=True`；随后在暂停指令生效前，已启动的 10 回合固定状态评测队列自行结束，10/10 均 reward 1.0、`done=True`（步数见进度记录），每回合视频与 NPZ 保留。最新离线配对动作诊断发现 LoRA 对 B-Spline OSC 搬运第一动作 MAE 降低 17.3%，而抓取 / 放置第一动作误差分别增加 41.8% / 21.6%；不能据此宣称训练后各阶段都更接近示范，也不构成安全证明或多状态泛化结论。`object_3` 仍无真实语义绑定，Compiler `unknown` 不变。Safety Bridge 已停止；当前不自动扩展采集、训练或环境评测，后续按新的明确目标决定。

## 主要记录

- 实验与历史阻塞：`docs/GS_OPD_PROGRESS.md`
- 当前局部目标：`docs/GS_OPD_CURRENT_GOAL.md`
- 本地数据与 Checkpoint 路径、训练曲线和重载核验：见进度记录的单轨迹 SFT 章节。
