# GS-OPD 当前局部目标

## 当前局部目标：t=223 OSC 恢复状态已保存（2026-10-23；等待复核）

本轮已从上一轮真实 `t=173` Runtime Snapshot 恢复并完成严格逐字段 roundtrip。旧 RRT 关节偏差为0.209865 rad L2，其中0.209718 rad（99.93%）处于EEF Jacobian零空间；EEF任务分量仅0.007858 rad，实际位置/姿态误差2.830 mm/0.007704 rad。原joint1停止门槛主要捕获冗余解漂移。

从真实t=173状态重解预抓取目标IK，并使用固定种子20261024的7D Joint-Space RRT-Connect得到150点 `offline_feasible` 路径。Bottle/Place Box密集全几何间隙14.289/12.722 mm，满足未修改的10 mm硬门槛，也超过旧路径10.872 mm的最小障碍间隙。通过原生OSC_POSE只执行前50步至t=223；实际到目标距离减少37.318 mm，最大EEF误差3.344 mm；1,250个物理子步监测无机器人环境接触、自接触或限位违规。t=223快照restore→re-export全部字段精确匹配，文件可用于恢复。

当前末端距离预抓取目标371.476 mm，尚未抵达、未抓取Pot 2；reward=0、done=false。本段只证明有限的安全物理进展，不能标记为预抓取到达、完整Safety Recovery或任务成功。新Bridge已关闭，不继续执行其余路径。

完整报告：`/opt/data/private/gs_opd/rollouts/l0t2_round1_t173_replan_osc50_20261023_attempt1/evidence/t173_replanned_osc_execution_report.md`；末态快照：同目录 `evidence/t173_replanned_osc_terminal_runtime_snapshot.pkl`。下一步待复核后再决定是否从可恢复t=223快照继续沿路线执行；不运行VLA、训练、抓取或放置。

**本轮结束，等待复核。**

---

## 历史局部目标：L0T2 RRT q0→q1 OSC 首段验证已停止（2026-10-23）

本轮从原始 π₀.₅ Student t=10 Runtime Snapshot 恢复后，只执行离线 Joint-Space RRT 路径的 q0→q1 首段。恢复前按双 reset 初始化顺序核对模型指纹、BDDL、Integration State、控制器/缓冲、task bookkeeping、RNG以及q0/FK，均通过；完整65点保存扫掠的 Bottle/Place Box 最小间隙为38.886/112.217 mm。

正确张开夹爪的原生7D OSC_POSE分支执行163步，至任务t=173、参考弧长76.88%。EEF最大跟踪误差8.997 mm，最大姿态误差0.035743 rad；整段已执行物理状态无Bottle/Place Box接触、无自接触，实测间隙至少38.628/111.011 mm。实际关节逐渐偏离RRT参考，第163步第1关节偏差0.150257 rad触发本轮关节通道硬停止。该末态到q1 EEF仍有51.643 mm/0.128868 rad，q误差L2为0.281536 rad；未到q1，没有执行后续段。

区分结论：EEF参考局部可跟踪；q0→q1的7D关节路径未被OSC跟踪完成；已执行部分无观察到的障碍接触，但不构成整段路径认证或Safety Success。没有证据证明EEF目标不可达；直接瓶颈是原生6D OSC_POSE不约束冗余关节解，在线q偏差达到停止门槛。无需再在本轮扩大动作或放宽阈值。

完整证据：`/opt/data/private/gs_opd/rollouts/l0t2_round1_rrt_q0_q1_osc_validation_20261023_attempt1/`，详见 `evidence/q0_q1_osc_validation_report.md`。t=173末态Snapshot已restore→re-export核验并恢复回t=10；JSONL/NPZ包含动作、双相机Observation、qpos/EEF、4,075个物理子步接触与间隙，两路视频已保存。另有错误夹爪符号的隔离失败记录，不并入正确分支结果。

**本轮工作完成，停止等待复核。**不执行q1→q2/q2→q3、不抓取或放置、不训练、不运行Rollout；既有任务成功示范及LeRobot数据保持不变。

---

## 历史目标：L0T2 t=10 RRT-Connect 路线离线可行（2026-10-23）

t=291分支已停止。本轮在已验证的原始 π₀.₅ t=10 状态上，以当前 Pot2 场景重算历史成功预抓取姿态的 q_goal；q_start、q_goal 的碰撞、关节限位及端点余量检查通过。固定种子双向7D Joint-Space RRT-Connect第11次迭代连通，shortcut后路径2.730396 rad，4个路点、257个密集验证样本。

Bottle/Place Box全机器人最小余量分别14.879/10.872 mm；全程没有机器人–环境或自接触，关节限位和终点FK通过。Pot2为目标物：末端夹爪仅在预抓取目标100 mm、姿态差0.25 rad范围内可进入其10 mm邻域，终段右指尖最小间隙0.033 mm但无穿透/接触，机械臂其余连杆保持至少10 mm。

此路线只标记 offline_feasible，不代表原生 OSC 可执行或安全成功。无环境步进、OSC、Rollout、训练；已有示范和数据集未改。证据归档 /opt/data/private/gs_opd/rollouts/l0t2_round1_jointspace_rrt_connect_20261023_attempt1/。当前停止等待复核。


## 历史局部目标：L0T2 t=10 关节空间任务导向规划（上一轮；离线候选均失败）

正式停止 t=291 分支。从已验证 π₀.₅ t=10 Integration State 构造 q_start；Bridge Observation 缓存值与其有1.058 mrad关节/0.586 mm EEF差异，q_start按Integration State和原始第10步物理状态确认，缓存差异已留档。按成功 Pot2 state118 的物体相对预抓取 EEF 位姿重建当前目标，并在当前场景重新求 q_goal；不直接假定历史绝对关节解。q_goal IK残差小于3 mm/0.03 rad，因Pot2目标Body位姿相同而数值收敛到历史 state118关节解。

3条完整关节空间路线均在离线全几何检查中失败：直连边穿过Bottle；西南通道路线前段有余量，但最后接目标边触碰Place Box且Bottle不足10 mm；起点抬升调腕路线末段穿过Bottle。关节限制和这些完整关节路径上的Jacobian状态合规，终点FK误差在数值精度内；没有任何真实OSC、环境步进、Rollout或训练。完整路径和逐段原因见 `/opt/data/private/gs_opd/rollouts/l0t2_round1_task_directed_jointspace_20261023_attempt1/evidence/jointspace_path_planning_report.md`。

少量中间位姿 IK 探索显示当前指定交越点陷入近奇异局部解；这不证明目标不可达。基于三条直线关节边都失败，RRT-Connect是值得下一轮审核的离线C-space搜索方法，但并非已证明唯一解法，本轮未运行新规划库。等待复核；已有成功示范和数据集保持不变。

---
## 历史诊断：t=291 首段抬升接入预检（attempt1）

已从 t=291 快照严格恢复并通过完整状态一致性核验。原成功抓取轨迹第118步的真实无接触 EEF 状态重建了 Pot 2 预抓取目标。唯一分阶段路线的离线预检在首段抬升第8/159点未通过 IK 残差要求：位置残差3.646 mm；关节未越限且无环境接触，但 EEF Jacobian 条件数升至约246,237，呈严重腕部病态。没有执行任何 Bridge 控制动作，t=291 原状态已重新导出并确认一致。

该报告仅记录首条固定姿态竖直抬升路线；后续 attempt2 已依照当前目标检查两条替代 IK 接入路线。报告：`/opt/data/private/gs_opd/rollouts/l0t2_round1_pregrasp_reentry_20261023_attempt1/evidence/pregrasp_reentry_report.md`。

---


## L0T2 Round 1 自适应跟踪物理验证（2026-10-23；失败后停止，等待复核）

从兼容恢复后的 t=216 Snapshot 用真实 EEF 跟踪原 B-Spline 前向路线，参数为参考速度 0.020 m/s、12 mm 推进门槛、每点最多32步追赶、40 mm硬停止。75步执行到 t=291，在参考点39因32步追赶用尽、误差仍为16.529 mm停止。该处机器人无碰撞、关节未越限，最大障碍间隙 Ginger Beer 175.105 mm / Place Box 202.194 mm；尚距预抓取位姿444.244 mm，不能验收为局部安全恢复成功。

失败证据说明自适应推进已生效，但局部闭环没能追上冻结目标；没有证据证明目标不可达，也没有动作饱和或碰撞证据。t=291 分支快照已保存，Bridge 已精确回到输入 t=216 并停止。本轮无进一步动作、抓取、放置或训练。

报告与数据：`/opt/data/private/gs_opd/rollouts/l0t2_round1_adaptive_safety_recovery_20261023_attempt1/`。等待复核；原完整任务成功示范与 LeRobot 数据保留不变。

---


## L0T2 Round 1 OSC/B-Spline 跟踪失败复核（2026-10-23；只读完成，等待复核）

只读复核 t=10→216 分支的 198 条 B-Spline OSC 记录、动作/状态、t=216 Runtime Snapshot 及实际调用的 OSC_POSE 映射代码；没有恢复 Bridge、执行动作、Rollout 或训练。误差在索引 159–162（约 10 mm）开始偏离稳定平台；索引 170 / task t=189 起连续增长，11.619→13.332 mm；到索引 197 / t=216 达 40.135 mm。参考点始终每步推进约 3 mm；实际单步位移由 140–159 的 2.962 mm 降到 170–179 的 2.667 mm、180–189 的 2.375 mm，参考游标未按误差减速或暂停。

6D OSC 动作最大归一化分量 0.674（无 pose action 饱和），Panda 关节限位检查全通过；坐标/动作变换与 robosuite 当前 EEF 世界系 delta 约定一致，未发现符号翻转证据。最有依据的根因是后段 OSC 响应下降时仍按固定 0.06 m/s 推进参考；局部目标可达性与历史内部扭矩饱和无法确认。t=216 到预抓取点直线距离为 0.4331 m，比 t=10 的 0.3340 m 更远 99.0 mm，虽已走到绕障参考的 51.97%。

t=216 Snapshot 的 317 维 Integration State 与最后动作记录完全一致，模型/BDDL/任务元数据、controller/buffers、observation cache 和 RNG 均有保存；但该快照的 restore→re-export roundtrip 尚未测试。唯一推荐的未验证修正：恢复并核验 t=216 后，从真实 EEF 状态重锚到原路线前向最近参数，对连接段重新检查完整机器人几何；采用不高于 0.04 m/s 的弧长参考，并在位置误差超过 10 mm 时冻结游标，追上后再推进。保持 40 mm 停止阈值，不改 B-Spline 几何。方案尚未物理验证。

独立诊断：`/opt/data/private/gs_opd/rollouts/l0t2_round1_safety_recovery_20261023_attempt1/evidence/round1_tracking_failure_diagnostic.md`。原始物理记录、已验证成功示范及训练候选均未改。本轮结束，等待复核。

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

