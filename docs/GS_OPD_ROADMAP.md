# GS-OPD 总体路线

## 最新局部状态（2026-10-23）：L0T2 Round 1 绕障部分执行，无接触但未到预抓取，等待复核

以原始 π₀.₅真实 t=10 Snapshot 恢复（前 10 步原动作重放，模型、BDDL、MuJoCo 状态、控制器、任务 bookkeeping、RNG 全部核对通过），并按历史 Pot 2 实际抓取变换构造可重接入的预抓取目标。沿一条先 −World-X/+Z 制动退让、后负 World-Y 外侧 cubic B-Spline 的 EEF 路线执行 206 个 OSC_POSE 控制步，到达 timestep 216 时因 EEF 跟踪误差 40.135 mm（阈值 40 mm）停止；仍距预抓取目标 433.061 mm。关节检查无越限。

每个控制步的 25 个 MuJoCo 子步均保存了全机器人对 Ginger Beer/White Place Box 的接触与几何最小间隙。5150 个物理子步中机器人障碍接触事件为 0；全程最小间隙分别 63.800 mm 和 111.804 mm，Pot 2 位移数值为 0。该结论仅适用于已执行的部分路线：没有到达预抓取，也没有整条路线通过或安全认证。Teacher IR 未找到，Contact Compiler 未运行，语义 verdict 未验证。原 1093 步 L0T2 成功示范及 LeRobot 候选数据保持不变。

独立证据：`/opt/data/private/gs_opd/rollouts/l0t2_round1_safety_recovery_20261023_attempt1/`。包含机器/文本报告、逐步 Observation–Action 与子步物理记录、视频、路线和源代码副本。当前停止等待复核；复核前不再执行 OSC 避障、不启动训练。

## 下一总体目标

继续以 L0T2 为目标进行安全恢复研究，但先复核本轮 OSC 中段 40 mm 位置误差的成因和日志。需要决定是从 t=10 起点调整离散速度/路径段落，还是选择更早可行的几何干预点；任何后续动作必须单独审批复核后，在可恢复 Snapshot 分支验证，继续以全机器人几何接触监测和关节限位为约束。Teacher/Compiler 在 L0T2 的状态仍未验证；不得据此将本轮部分无碰撞前缀写成安全示范。原始 1093 步任务成功示范及 LeRobot 候选集不改，不训练。



## 最新局部状态（2026-10-23）：L0T2 Ginger Beer 干预时机已诊断；退让待物理验证

在原始 π₀.₅ 动作前缀的 t=0/5/10/15/20 状态对全机器人活动碰撞 Geom 作实时 clearance 查询。t=0、t=20 与归档物理状态精确一致；Bottle 间隙由 119.9 mm 递减至 t=20 的 4.4 mm，Place Box 在 t=20 仍有 103.4 mm 机器人间隙。EEF 从 t=0 到 t=20 持续沿 +X/+Y 向 Bottle 靠近，最新区间速度约 0.14 m/s。

上一轮三条 OSC 分支的首步只要求约 1–2 mm 位置修正，但继承速度使 EEF 实际前进 4.27–4.52 mm；第二步同一掌部 Geom 74–Bottle Geom 222 发生接触。下一次复核后的实验建议为 t=15 先向 −World-X 退让并略向 +Z 抬升；此时掌部 clearance 为 36.1 mm，OSC 每轴可发出 ±50 mm 的目标变化，但实际退让仍待验证。t=16–19 和 t=20 大幅反向命令没有物理验证，故精确最晚干预点仍不确定，建议路径不能标为安全。

本轮只读完成，没有生成新轨迹或动作。独立证据 `/opt/data/private/gs_opd/rollouts/ginger_beer_intervention_timing_l0t2_20261023_attempt1/`，详细结论见 `evidence/intervention_timing_report.md`。当前暂停等待复核。


## 最新总体状态（2026-10-22）：L0T2 示范/数据验收完成；Geometry-Guided 接近修正尚未找到有效路线

L0T2 的 1093 步真实任务成功示范与 759 条隔离 LeRobot `expert_osc` 候选数据已完成验收；完整示范含历史 VLA Ginger Beer 接触，未标安全成功。当前局部安全阶段从原始 π₀.₅ 第 20 步状态开始。已按原始 Bridge 初始化与 Rollout reset 顺序精确重放动作 1–20，导出并成功恢复 t=20 Runtime Snapshot（Integration State 精确一致，模型几何指纹 `1a0c219c…f7a9f752`）。

现有候选生成器输出 18 条 Left/Right/Above、Near/Mid/Far EEF 参考路径；保守球形筛选通过数 0/18。三条不同方向/范围的 OSC Snapshot 分支各在第二个动作与 Ginger Beer 发生接触，触发原生 BDDL `CheckRobotContact`，未进入预抓取区域。联系力峰值分别 0.817、0.936、0.942 N，Contact Monitor 覆盖完整，Panda 关节限位均通过。任务重新进入可行性仍未知，局部安全路线数量为 0；不将局部结果解释为完整任务成功或安全认证。

证据归档 `/opt/data/private/gs_opd/rollouts/geometry_guided_safety_bundle_l0t2_20261022_attempt1/`。本阶段等待复核；后续如获准继续，应先依据 t=20 的实测正向关节速度建立可验证的起始避碰控制，再重新筛选路线，保持 BDDL、模型、OSC 默认控制、抓取模块、成功示范和 LeRobot 数据集不变。


## 最新总体状态（2026-10-22）：L0T2 成功示范完成数据验收；候选专家集等待复核

完整链 1093 步已核对，动作分段为 294 VLA +425 OSC +135 Grasp Retargeting +239 OSC。LeRobot v2.1 独立候选集包括终止前 759 个 `expert_osc` 样本，分成 425/135/199 三个 Snapshot 连续 episode；读回逐帧校验全部通过。VLA 294 步留作原始轨迹与 Ginger Beer 碰撞证据；done 后 40 步留作撤离/稳定证据，不进入任务监督池。所有样本没有安全认证。

数据集位于 `/opt/data/private/gs_opd/datasets/lerobot_l0t2_expert_osc_candidate_20261022_attempt1/`；审计、样本索引与转换脚本保存在成功示范归档的 `evidence/`。完整示范仍标记 over-budget（1093 步，标准 horizon=1000，超 93 步），任务成功不等于安全成功。紧凑 action index 的两列路径字符串有定长截断；完整 JSONL 和样本 sidecar 保留全路径，报告已标明该限制。当前只等待复核，不启动训练、Rollout 或其他场景实验。

## 总体目标

在 π₀.₅ 原生动作空间和现有安全约束下，让 Student 从真实成功执行数据学习避障操作，并在 MuJoCo 中完成可复现的 Student 闭环任务。数据来源、动作来源和安全判定要可追溯；Compiler 的 `unknown` 不得伪装成 `safe`，普通行为克隆动作不得误标为安全修正。

## 历史诊断（2026-10-22）：L0T2 Stove 双壶终点布局已证实静态可行

只读检查 t=854 Demo Snapshot，并从同一 Snapshot 在独立 scratch MuJoCo 副本测试 16 个 Pot1 目标位姿。保持 Pot2 原位时，8 个目标有静态无冲突支撑及同时满足两个原生 On 和 Stove Turnon 的联合状态；3 个还通过了局部夹爪进场/反向撤离几何检查。推荐 Pot1 root `[-0.1127581,-0.2626613,0.9966100] m`、yaw 180°，Pot1–Pot2 完整碰撞几何间隙 15.272 mm；Pot1/Pot2 Burner 支撑接触 30/28 个，局部夹爪最小非目标间隙 37.814 mm，7 点 IK 全部可达、无机器人场景接触。Pot2 不必为终点布局而重新放置。

只在 scratch 中临时改候选 qpos 并作 `mj_forward`/距离/接触/谓词检查，没有物理步进、控制动作或更改主快照/任务。故此结果只说明“静态终点和局部夹爪进出通道可行”，并未验证从 t=854 到目标的完整携物扫掠、动态交互、实际释放或任务 reward；既有 t=967 的 40.139 N 接触仍是实际搬运路径风险。独立归档 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_joint_pot_layout_feasibility_20261022_attempt1/`。当前暂停后续 OSC 直到复核。

## 复核后的下一步

先复核 759 条 OSC 专家候选的示范质量、状态起点和路径引用，再决定是否将其并入训练转换。没有复核前不启动训练或 Rollout；不改变原始 VLA 动作/Observation，不认证安全。L0T1 与 L0T0 保持暂停。

# GS-OPD 总体路线

## 最新局部状态（2026-10-22）：L0T2 t=854 搬运到位后下降触发持续 Pot1–Pot2 接触；停止等待复核

从已验证 Pot 1 双指抓持快照 t=854 恢复；隔离 Demo Bridge horizon=1300，标准评测 horizon=1000，模型、BDDL、物理和 reward 未改。第一条分段 OSC_POSE 路线用了 113 步，完成主动抬升至 Z=1.19936 m、高位西侧绕行及 Cook Region 上空进场。t=967 的慢速下降产生此前不存在的 Pot1–Pot2 物理接触，峰值 40.139 N / 25 个子步，超过硬停止阈值；Pot 1 尚未达到释放高度。Pot 2 位移最大 0.733 mm，Pot2 On 和 Stove Turnon 仍真；Pot1 On=false、reward=0、done=false。所有已记录 Panda 关节检查通过。第二条不同初始方向的路线因明确持续挤压停止条件而未运行。

离散 OBB sweep 未预测该下降碰撞，因此不能作为 clearance 证明；真实 MuJoCo 接触作为权威结果。失败动作、Observation、物体状态、接触/力、视频及失败快照单独保存，未并入示范链。实验后 Bridge 精确恢复至 t=854 起点，Integration State 最大误差 0，双指抓持及已有 Pot2 On/Stove Turnon 均复核通过。完整任务、标准 Benchmark、超预算示范与安全成功均未取得。历史 VLA Ginger Beer 接触保留。详细归档为 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt2/`。

## 下一总体目标

继续在 LIBERO-Safety `obstacle_avoidance/L0T2/state0` 获取真实完整任务示范，但先对照 t=967 实际接触点核查携物时 Pot1、Pot2 的全碰撞几何和 Stove Cook Region 的可用占位。只有在新的路径能解释并避免本轮实际下降碰撞后，才可在复核后设计有限的不同路线。保持 BDDL、reward、物理模型、标准 horizon、关节限制及任务判据不变；不直接改物体位姿，不把部分执行标成成功或安全示范。L0T0 保持暂停，不启动训练。

## 历史项目状态与阶段记录（2026-10-22）

第一版通用 Grasp Retargeting 已成功迁移 Pot 1 抓取，t=966 Pot 1 仍处于稳定双指抓持并已接近 Stove。Pot 2 的原生 On 与 Stove Turnon 保持成立，Pot 1 On 未成立。此次只修复 demonstration-only Snapshot 兼容恢复并扩展隔离 Episode horizon：从原始 t=966 快照成功恢复，在同一个模型兼容实例内将底层 horizon 调至1300；扩展前后物理状态不变。当前 Bridge 已具备334步余量，但没有执行放置动作。标准评测仍为 horizon1000。等待复核后再续做 OSC 放置。

## 总体目标

在 π₀.₅ 原生动作空间和现有安全约束下，让 Student 从真实成功执行数据学习避障操作，并在 MuJoCo 中完成可复现的 Student 闭环任务。数据来源、动作来源和安全判定要可追溯；Compiler 的 `unknown` 不得伪装成 `safe`，普通行为克隆动作不得误标为安全修正。

## 阶段与当前状态

1. **安全输入与物体语义**：沿用 Teacher、Safety IR、实时几何绑定和 Contact Compiler。现有 L0T0 的 `object_3` 已有真实 SAM/IoU 与活动 Geometry Inventory 绑定，仅限该固定任务输入使用；其他缺少可靠证据的对象/规则继续保持 `unknown`，不补造映射。
2. **真实成功执行**：已有一次真实 Student→B-Spline→原生 OSC 左绕成功回合：749 步 B-Spline 左绕后由 π₀.₅ 接续 41 步，取得真实任务奖励 1.0；完整轨迹和视频保存在 `live_left_bspline_osc_adaptive_20261019_attempt1`。
3. **数据格式与读取**：已有 749 步 OSC 专用 LeRobot 数据集；现已另从原始成功 NPZ 生成完整 845 步单 Episode（55 抓取 / 749 搬运 / 41 放置），逐步验证 Observation–Action 对齐，保留 836 个 horizon-10 无 Padding 窗口及跨阶段窗口，继续使用初始 Student 对应 norm stats。
4. **单轨迹学习链路**：2026-10-20 从原始 Student 初始化新的 rank-32 Action Expert LoRA，使用恒定 `1e-4` LR、AdamW 重新初始化、官方 OpenPI DataLoader、`SFTRunner`、原生 Flow Matching Loss 和三阶段均衡采样，完成 10,000 optimizer steps。固定示范状态上的 OSC 搬运 Flow MSE 从 0.152026 降至 0.000982，真实 10×7 Action MAE 从 0.183005 降至 0.009061；独立 Adapter 已成功重载并与训练 checkpoint 推理一致。该结果确认完整轨迹监督拟合，不代表新闭环成功、安全泛化或真实执行能力。
5. **训练后 Student 评测与后续验证**：attempt3 新 Adapter 已与原始 Student 组合，在 LIBERO-Safety `obstacle_avoidance` L0T0 初始状态 0 完成一次纯 Student MuJoCo 回合：711 步、reward 1.0、`done=True`。完整 MP4、七维动作、双相机 Observation、EEF/黑碗物理状态和 17,775 个子步接触监测均已保存。针对现有 Safety IR 的 object_1/object_4 禁止接触规则，活动 Geometry Inventory 绑定通过核验，Compiler 在全子步记录上判为 `safe`；碗与该障碍的最近保守包围球间距约 0.19 mm，须保留这一很小余量以及 object_4 原语义匹配 IoU 0.1427 的置信限制。路径对比显示 Student 以更紧的 +World-Y 路径绕过障碍。此前 LoRA Student 物理路径文件已被空间清理，本次没有替代或重建该历史曲线。现有 `object_3` 语义绑定只对固定 L0T0 输入有证据，不从 reward 推导其规则安全结论。
   - 单独的 attempt3 On-Policy 候选接管 Smoke 完整执行了 422 步 Above 原生 OSC 路线，再由 Student 执行 528 步但未成功。OSC 末端位于局部出口而非目标盘；后续 EEF 到盘子的距离由约 0.371 m 增至 0.628 m，诊断更支持 Student 接管后未进入朝盘下降/放置，而非 OSC 绕障失败。该 Student 段没有黑碗逐步世界坐标，具体物体运动仍以原视频为限；详见进度报告。
6. **跨 Episode 风险分层采集（历史/暂停）**：此前验证了 Far `(60,100] mm`、Mid `(30,60] mm`、Near `≤30 mm` 三层，并完成三次真实 L0T0 回合；数据和报告保存在 `logs/gs_opd/stratified_far_mid_near_20261020_attempt2/`。Episode 调度入口已从活动 `pi05_rollout.py` 主线移除，Near–Far 独立研究实现和原 5/10/20 mm 安全阈值未改。该采集不是当前默认工作流，详见 `docs/GS_OPD_PROGRESS.md`。
7. **多候选 On-Policy 安全轨迹采集（L0T0 历史工作，当前暂停）**：Snapshot 分支、候选池验收和验收后主环境 OSC 均将额外的软件 Panda `joint_limit` 改为逐步诊断，不因越限本身提前拒绝；qpos、越限幅度和 `joint_limit_compliant` 保留，不能将越限轨迹标为满足全部机械约束。MuJoCo 原生关节约束、抓持、跟踪、碰撞和 Contact Compiler 保持不变，历史示范、LoRA 与训练未改。回归为 78 项及 2 个子测试通过，Python `py_compile` 和 `git diff --check` 通过。
   - 原始 Student、seed 0 的单次当前磁感线采集生成 18 条候选，7 条在总预算内执行 Snapshot 分支，另 11 条因预算未执行；7 条都因原有抓持、跟踪或接触分类约束而提前失败，完整通过数为 0，Snapshot 恢复通过。首个当前候选 Left `trajectory-001` 在控制步 206 因抓持相对旋转误差越限而停止（207/846 步），不是因关节软件检查。5/7 分支含关节越限诊断，最大 `0.0073798359 rad`；另两条分支合规。主环境 OSC 与 Student 接续没有触发。
   - Student 的真实主环境前 50 步关节均合规、最大越限 `0 rad`，并观测到 4 个连续有效状态的稳定双指抓持；任务 reward 0、`done=false`。实际 episode Contact Compiler verdict `safe`，但事件分类为 `unknown`（294 条黑碗与炉灶 burner 接触没有唯一 Teacher allow rule），所以接触安全未验证。没有完整成功安全轨迹。证据位于 `logs/gs_opd/eval/trajectory_bundle_joint_limit_diagnostic_l0t0_20261020_attempt1/`，详见 `docs/GS_OPD_PROGRESS.md`。按本轮边界没有批量重试、复现历史 Left 或调参。
8. **L0T2 单物体 Near-Goal Recovery（等待复核）**：已用 294 个归档原始动作恢复 π₀.₅ 在第 141 步形成的 Moka Pot 2 稳定抓持，并从该 Snapshot 完成一次隔离 OSC 放置修正。Bridge 原生 `On(moka_pot_2, flat_stove_1_cook_region)` 连续 5 个稳定状态成立，释放后有真实 burner 支撑接触。该成功片段含 425 个 OSC 动作与可恢复单物体成功态，但完整 T2 reward 仍为 0、`done=false`，且原始前缀有机器人–Ginger Beer 碰撞；不得标注成完整或安全成功示范。详细证据保存在独立归档 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_mokapot2_near_goal_osc_recovery_20261022_attempt1/`。


9. **可复用 Grasp Retargeting（首版物理验证通过，等待复核）**：新增通用物体坐标抓取模板提取、运行时 Body/Geom 绑定和标准化 `expert_osc` action record。Pot 2 来源到 Pot 1 目标实测稳定双指抓持，并抬升 30.657 mm；恢复快照精确验证通过。未搬运/放置，T2 仍未完成；由于源 VLA 前缀已有 Ginger Beer 接触，组合历史不是安全成功示范。证据见 `docs/GS_OPD_PROGRESS.md` 与独立归档 `grasp_retargeting_l0t2_20261022_attempt1`。

10. **L0T2 Pot 1 搬运与 Stove 放置（本轮局部执行因标准 horizon 不足而停止，等待复核）**：从 Pot 1 已验证抓持 Snapshot（t=854）恢复，在正常 horizon 1000 内评估三条由真实 Cook Region 与器物几何构造的 OSC 放置候选。两条 clearance 分支各运行 14 步；`west_north_diagonal` 部分搬运 112 步，至 t=966 时仍抓持 Pot 1，需下降约 133 mm。按实测速率估算下降、释放、撤离及稳定还需约 119 步，剩余 34 步，因此停止。Pot 2 `On` 和 Stove `Turnon` 保持真，Pot 1 `On` 为假，reward=0、`done=false`；下降时 Pot 1–Pot 2 峰值接触力 71.20 N，Pot 2 移动 0.303 mm；无新机器人–非目标碰撞，Panda 限位 112/112 通过。任务成功与安全成功均为 false；历史 Ginger Beer 接触保留。独立数据和报告位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_mokapot1_stove_placement_20261022_attempt1/`。下一阶段待复核后从更早的 Pot 2 抓持状态压缩完整搬运链；不继续从 t=966 搜索或扩展 horizon。

## 主要记录

- 实验与历史阻塞：`docs/GS_OPD_PROGRESS.md`
- 当前局部目标：`docs/GS_OPD_CURRENT_GOAL.md`
- 原始 Student Checkpoint、成功示范和 LeRobot 数据集仍保留。attempt3 训练日志、报告、Adapter 和 RLinf checkpoint，以及本次 Student rollout 视频、NPZ、Compiler 审计和轨迹图，均保留于各自独立目录。详细记录见 `docs/GS_OPD_PROGRESS.md`。
