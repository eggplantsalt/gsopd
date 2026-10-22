# GS-OPD 当前局部目标

## L0T2 成功示范数据验收与 LeRobot 适配完成（2026-10-22；等待复核）

本轮只读验收完整 1093 步示范并生成隔离的 LeRobot v2.1 专家候选集；没有执行训练、Rollout 或机器人动作。动作边界为 294 π₀.₅ `vla` +425 Pot 2 `expert_osc` +135 Grasp Retargeting `expert_osc` +239 最终放置 `expert_osc`。1093 行 action、双相机 Observation、物理状态、接触、任务状态、reward/done 及逐步源引用全部核对；t=719、854、1021 的状态与 RGB 衔接一致。t=294 首个 OSC 输入是从前缀后状态重新记录的真实 RGB，与 VLA 回放最后 post 图像不逐像素相同（两视图最大像素差 114/117）；原输入保留，未复制替代帧。

原生逐步谓词状态只在 406/1093 行非空：VLA 294 行及 Pot 2 OSC 425 行中的 393 行缺值，其他 374 行有值。缺值保留为空；逐步指令、动作、双相机、奖励、Done 和接触引用均保留。候选索引标明每行谓词状态可用性，LeRobot 样本不伪造缺失谓词。

环境在 t=1053 首次返回 reward=1、done=true，后续 40 步仍返回终止状态，只用于撤离和物理稳定证据，不进入任务监督集。LeRobot `0.3.3` / v2.1 候选数据位于 `/opt/data/private/gs_opd/datasets/lerobot_l0t2_expert_osc_candidate_20261022_attempt1/`，分成 425、135、199 三个 Snapshot 连续段，共 759 个终止前 `expert_osc` 样本；759 帧已逐帧通过 Reader 回读校验。294 个 `vla` 动作保留为原策略轨迹与失败证据，终止后 40 个 OSC 动作只保留为撤离/稳定证据。VLA 动作 21–56 的 Ginger Beer 接触（峰值 12.096849 N）仍在记录中；全链与专家候选都不标为安全成功。完整示范 1093 步，超出标准 horizon 1000 共 93 步。

验收报告 `evidence/data_integrity_report.json`、样本索引 `evidence/expert_supervision_sample_index.json`、转换验收 `evidence/lerobot_expert_candidate_conversion_report.json` 与可复现脚本均位于最终示范归档。检查发现紧凑 `complete_action_chain_index.npz` 中路径字符串定长截断（source 路径 958 行、物理状态路径 664 行）；完整 JSONL 和新样本索引的路径完整，后续读取应使用它们。等待复核，不启动训练。

## L0T2 完整任务成功示范执行阶段（前一阶段，2026-10-22）

从已验证的 t=854 Pot 1 稳定抓持快照恢复，在独立 demonstration-only Bridge（horizon=1300）中使用既定 Pot 1 目标 `[-0.1127581, -0.2626613, 0.9966100] m`、yaw 180°，按“高位远离 Pot 2 绕行—Cook Region 西南侧进入—下降—松爪—撤离—稳定”路线完成真实物理执行。t=854→1093 共 239 个 Pot 1 OSC 动作，t=1021 慢速下降达到原局部动作上限后从同一真实状态继续原路线；没有重抓取、VLA 推理或额外候选。

最终 Pot 1/Pot 2 原生 `On(..., flat_stove_1_cook_region)` 与 `Turnon(flat_stove_1)` 同时为真；Bridge 实际 step return 为 `reward=1.0, done=true`。Pot 1 在 t=1053 释放，t=1089–1093 连续 5 个状态稳定。Pot 2 在最终放置段位移 0 m；新机器人–非目标碰撞为 0，关节检查全通过。Bridge 只读状态中的 `task_done=false` 原样保留；验收依据为原生谓词、真实 reward/done、释放及稳定条件。完整轨迹 1093 步，超过标准评测 horizon=1000 共 93 步，标为 `demonstration_success_over_budget`，不是标准 Benchmark 成功。

轨迹由 294 步原始 π₀.₅ Base-only 前缀、425 步 Pot 2 OSC Near-Goal Recovery、135 步 Pot 1 Grasp Retargeting、239 步 Pot 1 搬运放置组成。原 VLA 前缀动作与原归档动作逐步一致，但动作 21–56 有真实 Ginger Beer 接触（峰值 12.096849 N），并保留回放时碰撞物积分状态尾部差异；因此 `safety_success=false`，不得作为安全成功示范。未训练。

完整证据归档：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt3/`。入口 `README.md`；验收 `evidence/success_acceptance_report.json`；动作/源数组索引 `evidence/training_demo_traceability_index.json`；1093 步记录 `evidence/complete_standardized_action_records.jsonl` 与 `evidence/complete_action_chain_index.npz`；三路完整视频位于 `evidence/videos/complete_demonstration_*.mp4`；成功 Runtime Snapshot 为 `evidence/moka_pot_1_complete_task_success_runtime_snapshot.pkl`。

**本轮停止，等待复核；不启动训练、不继续执行机器人动作。**

# GS-OPD 当前局部目标

## L0T2 双 Moka Pot 联合布局诊断（2026-10-22；完成并等待复核）

本轮已完成离线布局可行性诊断。t=854 的 Pot 1/Pot 2 起点状态已在隔离 scratch MuJoCo 中精确恢复；Pot 2 保持原位时存在三组“静态几何 + 原生谓词 + 局部夹爪进入/撤离”均通过的目标。推荐 Pot 1 root 位姿为 `[-0.1127581, -0.2626613, 0.9966100] m`、yaw 180°；Pot 2 仍在快照位姿。二壶 collision Geom 间隙 15.272 mm，各自 Burner 接触数 30/28，三项原生谓词同时成立；局部 7 点 IK 通道最小非目标间隙 37.814 mm，无机器人接触且关节范围有效。

没有执行机器人动作或任何 MuJoCo 物理步进；候选位姿只在独立 scratch 副本短暂设置，候选后 Integration State 精确恢复，scratch 已停止。主 Demo Bridge、t=854 源快照、BDDL、reward、物理模型和 horizon 未改。结果不代表完整转运扫掠可行，也不代表真实释放、任务成功或安全成功。

证据归档：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_joint_pot_layout_feasibility_20261022_attempt1/README.md` 与 `evidence/layout_feasibility_report.json`。原 t=967 的强接触仍需从实际路径/动态交互解释，静态终点诊断只排除了“当前 Pot 2 占位导致没有任何可行联合终点”这一假设。

**当前局部工作已完成，停止等待复核。** 复核前不继续 OSC 搬运/放置，不训练。

