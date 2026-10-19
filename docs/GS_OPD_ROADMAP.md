# GS-OPD 总体路线

## 总体目标

在 π₀.₅ 原生动作空间和现有安全约束下，让 Student 从真实成功执行数据学习避障操作，并在 MuJoCo 中完成可复现的 Student 闭环任务。数据来源、动作来源和安全判定要可追溯；Compiler 的 `unknown` 不得伪装成 `safe`，普通行为克隆动作不得误标为安全修正。

## 阶段与当前状态

1. **安全输入与物体语义**：沿用 Teacher、Safety IR、实时几何绑定和 Contact Compiler；`object_3` 尚缺真实语义绑定时继续保持 `unknown`，不补造映射。
2. **真实成功执行**：已有一次真实 Student→B-Spline→原生 OSC 左绕成功回合：749 步 B-Spline 左绕后由 π₀.₅ 接续 41 步，取得真实任务奖励 1.0；完整轨迹和视频保存在 `live_left_bspline_osc_adaptive_20261019_attempt1`。
3. **数据格式与读取**：已将成功回合的 749 步真实 B-Spline OSC 动作保存为 LeRobot v2.1 Episode，并验证 RLinf 官方 OpenPI SFT DataLoader 使用对应 Checkpoint 的 norm stats。训练时排除 9 个缺少完整 10 步真实动作的末尾窗口。
4. **单轨迹学习链路**：已用官方 RLinf SFT Runner、Pi0.5 原生 Flow Matching Loss 和初始 Student Checkpoint 完成 3 Epoch / 1110 Step 单轨迹过拟合，独立 Checkpoint 可重新加载。该结果证明数据到反向传播及保存恢复链路正确，不证明留出集泛化或新闭环成功。
5. **后续验证与扩展**：后续工作应另行限定是否进行独立状态/回合评估或增加具有真实来源标签的 Episode；维持原生 7D 环境动作、模型 32D 内部动作定义、norm stats、安全阈值及失败即停约束。多 Episode 训练和新 Student MuJoCo 验证均尚未由本次单轨迹实验完成。

## 主要记录

- 实验与历史阻塞：`docs/GS_OPD_PROGRESS.md`
- 当前局部目标：`docs/GS_OPD_CURRENT_GOAL.md`
- 本地数据与 Checkpoint 路径、训练曲线和重载核验：见进度记录的单轨迹 SFT 章节。
