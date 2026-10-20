# GS-OPD 当前局部目标

## 当前主线：单条 Left 自适应 OSC Snapshot 复验已完成（2026-10-20）

- 在多候选 On-Policy 路径中复用历史 Left 的 `0.020 m/s` 球心参考点采样与 12 mm 自适应推进；Snapshot 分支和验收通过后的主环境函数配置了相同推进规则。单 waypoint/总追赶上限为 32/240 步，Safety Compiler、关节限位、双指抓持、接触安全阈值和路线生成保持原样。主环境函数增加逐步关节限位核对，但本次候选未通过，因此该执行函数未实际运行。历史直接 Left 函数保留其专用速度/追赶常量。
- 使用原始 Student、seed 0，仅对 Left `trajectory-001` 做一次真实 Snapshot 验证。第 50 步稳定双指抓持后生成 18 条几何候选，单 Left 分支的 846 个参考步预算中真实执行 198 步；0-based 控制索引 197 首次触发 `robot0_joint7=2.9031289221 rad`，超过 `2.8973 rad` 上限 `0.0058289221 rad`。故候选不完整、不进入合格池。
- 越限时左右指都仍接触黑碗（9 个匹配接触），闭合命令 `[-1.0,1.0]`；抓持相对位置/旋转误差最大分别为 `0.009744 m` / `0.183919 rad`，在原门槛内。分支接触采样完整、Compiler 对这段不完整轨迹返回 `safe`、Snapshot 恢复最大误差 0；这些不构成完整安全路线验收。
- 因候选失败，主环境 OSC 为 0 步、Student 接续为 0 步。没有获得完整成功安全轨迹，不继续试其他候选。逐步 NPZ、视频、完整验收 JSON、启动参数、结果和 Student 日志保存在 `logs/gs_opd/eval/trajectory_bundle_adaptive_left_l0t0_20261020_attempt1/`。
- 回归：轨迹跟踪 **39 项 + 2 子测试**、轨迹束与 Snapshot 验收 **46 项 + 65 子测试**、OSC Snapshot/rollout contact probe **6 项**均通过，涉及 Python 文件 `py_compile` 通过。真实报告保留 Tracker 在关节限位失败当步的一个次级 waypoint 标记误报；之后已修正验收器并加回归，没有重新运行物理环境。

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
