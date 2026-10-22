# GS-OPD 当前局部目标

## L0T2 Round 1 Safety Recovery：路线部分执行后跟踪误差停止（2026-10-23；完成，等待复核）

从原始 π₀.₅ t=10 真实状态开始，本轮严格重放并验证前 10 个动作及 Runtime Snapshot，模型/BDDL 指纹和机器人、物体状态逐项匹配。依据此前 Pot 2 t=141–143 稳定抓取的 EEF–物体局部变换构造预抓取目标；未重新抓取、未运行 t=10 后的 VLA。

执行一条“先 −World-X/+Z 退让，再沿负 World-Y 外侧绕行”的路线。初始 8 个 OSC 步完成方向反转；随后沿 cubic B-Spline 执行 198/381 步，总 206 步至 t=216。全机器人碰撞几何在 5150/5150 物理子步均有接触/间隙记录，机器人到 Ginger Beer、White Place Box 实际接触均为 0；全程最小间隙 63.800/111.804 mm。Pot 2 位移数值为 0，Bottle 位移 2.27 µm，Place Box 位移为 0。

路线因 EEF 位置跟踪误差达到 40.135 mm（阈值 40 mm）停止；关节未越限，距离预抓取目标仍 433.061 mm，未到达预抓取、未抓取。reward=0、done=false。故本轮没有获得完整、无碰撞到达预抓取的安全恢复轨迹；已执行部分也不作为整体安全认证。L0T2 Teacher IR 缺失、Contact Compiler 未运行，语义安全 verdict 未验证。

独立归档：`/opt/data/private/gs_opd/rollouts/l0t2_round1_safety_recovery_20261023_attempt1/`。机器报告、实际动作/Observation/接触/间隙、Snapshot、路线及双相机视频均已保存。原 1093 步成功示范与 LeRobot 数据未改。不继续物理尝试或训练，停止等待复核。



## L0T2 Ginger Beer 避障干预时机诊断（2026-10-23；完成，等待复核）

当前只诊断原始 π₀.₅ 的安全干预时机，不创建新的绕障候选或执行 OSC 避障。本轮回放动作 1–20 并测量 t=0/5/10/15/20 的实时几何间隙；t=0 与 t=20 分别精确匹配原始 Rollout 初态和已验证 t=20 快照，模型几何 SHA-256 为 `1a0c219c…f7a9f752`。完整活动机器人碰撞 Geom 到 Bottle 的最小间隙从 t=0 的 119.9 mm 降到 t=20 的 4.4 mm；到 Place Box 仍为 103.4 mm。EEF 持续朝 Bottle 的 +X/+Y 方向运动。

上一轮三条路线虽然方向不同，首个 OSC 目标只请求约 1–2 mm 位移，但实际首步都沿 +X 运动 4.27–4.52 mm。第二个 OSC 动作内均以右夹爪手掌 Geom 74 接触 Bottle Geom 222；这是残余 VLA 动量加上 t=20 间隙仅 4.42 mm 的共同起始错误。

建议下次复核通过后从 t=15 干预，先沿 −World-X 后退并略向 +Z 抬升。该时点掌部尚有 36.1 mm 间隙，OSC 世界系每轴可发出 ±50 mm 的目标变化；这不保证单步实现该位移。t=16–19 未采样，t=20 大幅反向制动未物理验证；**精确最晚可行干预点及建议动作仍待物理验证**。本轮没有执行退让或训练。

证据归档：`/opt/data/private/gs_opd/rollouts/ginger_beer_intervention_timing_l0t2_20261023_attempt1/`。完成后停止，等待复核。


## L0T2 Geometry-Guided 局部安全修正：本轮候选验证完成，等待复核（2026-10-22）

当前局部目标是为原始 π₀.₅ 动作 21–56 的 Ginger Beer 碰撞寻找可继续抓取 Pot 2 的安全接近方式。本轮已可靠重建动作 21 前 t=20 状态，验证同模型快照恢复，并完成 18 条候选生成及最多 3 条不同方向的隔离 OSC 物理验证。

t=20 Snapshot 恢复通过，模型几何指纹为 `1a0c219c…f7a9f752`。运行时绑定确认 Pot 2 Body 24、Ginger Beer Body 25、White Place Box Body 31；BDDL 原生约束与目标已存档。Left/Near、Right/Mid、Above/Far 三条路线都在第二个 OSC 动作 t=22 因夹爪碰到 Ginger Beer 而触发 BDDL 禁止接触；每条 2 步后停止，关节限位通过，Contact Monitor 子步覆盖完整。没有抵达预抓取区，预抓取可达性和任务重新进入仍未证明。18 条球形几何筛选通过 0 条，原因是保守球体覆盖起点与目标区域，不可据此给出路线安全结论。

证据归档：`/opt/data/private/gs_opd/rollouts/geometry_guided_safety_bundle_l0t2_20261022_attempt1/`。完整数据与失败原因见 `evidence/geometry_guided_pregrasp_execution_report.json`；t=20 快照为 `evidence/collision_prestate_t20_runtime_snapshot.pkl`。

本轮结束，等待复核。未经复核不继续绕障动作；不训练、不执行任务后续阶段。


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

