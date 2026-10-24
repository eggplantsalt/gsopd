# GS-OPD 总体路线

## 最新总体状态：单次重规划执行至t=463后关节路线偏差触发安全停止（2026-10-24）

执行线已从t=359缓存路径续接到t=398；在准确Snapshot恢复和单次既有RRT-Connect后，又安全真实执行至t=463。合计104动作、2,600物理子步，Native/Operational Safety通过、双指抓持与finger-only接触规则保持、关节限位违规0；Strict-10mm统计因机器人非手指—Pot 2约4.36 mm而未通过，Ginger Beer/Place Box禁触边界仍满足。

最新193点路线离线offline_feasible且其模型、t=398 Integration State与策略指纹及路径已归档；它不等于OSC可执行。t=463实测7D qpos参考偏差0.182045 rad超过冻结0.18 rad，EEF误差仍小，执行器停止。证据支持“OSC_POSE执行对关节冗余参考跟踪不足”作为问题方向，但具体根因尚未通过控制内部量测确认。不能放宽阈值，也不把这解释成几何规划不可达。

本阶段没有Pot 2 On、释放、Pot 1操作、原生任务成功或Round 0种子；标准1000步和demo-only 1800步成功均未取得。104/450动作已用于这段连续物理Episode，唯一规划调用额度已耗尽。完整事件、OA、逐步Snapshot、视频、续接审计与规划路线见进度文档引用的两个t359/t398归档。

后续门槛是从严格恢复的t=463状态做一次新的、有界执行授权，先定位OSC_POSE与7D关节路点在冗余关节上的交接失配，再规划/执行至Pot 2放置，且维持所有当前安全标准。只读恢复复现入口已列在GS_OPD_CURRENT_GOAL.md；不得回放旧路径，也不开始Pot 1、LoRA或安全轨迹种子认证，直到Pot 2真实On里程碑完成。


## 前一阶段总体状态：缓存路线安全执行至t=359（2026-10-24）

归档路线具有完整257点qpos/EEF路径、t=329 Integration State、模型指纹和安全策略指纹。首条有效候选在RRT iteration 10；首次可行墙钟时刻没有保存。后来搜索继续到117次迭代/7,979次构型检查/360.055 s，Recovery外层累计361.403 s超过360 s限时。为满足本轮目标，直接从严格恢复的同一t=329状态复用完整路线，没有重跑搜索；状态、模型、策略和路线端点匹配。完整路径复用归档的全几何验证记录，本轮对起始状态和实际前缀重做验证。

修复transport持物时的预演夹爪策略为neutral `0.0`。四段隔离OSC预演共32动作、800子步通过；其后30个真实7D OSC动作、750子步均安全，Pot 2保持双指接触。全程0个禁触/未知/自碰撞接触、0个Operational关节限位违规；最大qpos路径偏差0.122807 rad，最大EEF位置/姿态误差7.326 mm/0.036344 rad。Pot 2真实位移75.474 mm，竖直上升32.529 mm；robot–Pot 2、Ginger Beer、Place Box最小净距分别4.370、18.776、42.177 mm。分类为Native Safety通过、Operational Safety通过、Strict-10mm不通过。

t=359时动作上限触发停止，仍在transport，Pot 1/2 On均false、Stove Turnon=true、reward=0、done=false；这是授权预算耗尽，未见规划—控制接口不兼容或安全阻塞。无放置、Pot 1抓放或完整任务成功，未建立Round 0种子。剩余路线索引30、t=359 Snapshot与Integration State已持久化；严格恢复核验尚未进行。本轮预算已用完，续跑前需新的动作预算并先验证该Snapshot与bundle，不得从t=329重搜或跨Snapshot拼接。

运行数据和复现入口见 [最新进度记录](GS_OPD_PROGRESS.md) 与 `research/gs_opd/repair/run_t329_native_task_suffix.py`。下一阶段依序：严格恢复t=359并校验模型/Integration State/策略；从缓存路径索引30安全预演；在新的明确物理预算内继续Pot 2 transport、放置、Pot 1抓放与原生谓词验收。完整种子仍要求从兼容的原始安全t=10状态连续重放及统一复验；t=329局部后缀不能补成完整前缀，也不启动LoRA。历史1093步示范与所有LeRobot数据保留。


## 上一总体状态：t=329严格恢复通过；首次任务续跑被RRT时间预算阻断（2026-10-24)

已复用Episode 9 t=249→329真实抬升分支的t=329 Runtime Snapshot并严格恢复。模型SHA与Snapshot SHA核验通过，Pot 2双指稳定持有、四Geom finger-only接触、当前操控状态及native predicates均由实时状态确认；新的TaskPhaseController自然进入transport。源局部分支验收Native Safety与Operational Safety通过，Strict-10mm未通过；当前t=329状态Operational preflight通过。

一次既有RRT-Connect运行生成 `offline_feasible` 路线：117次迭代、7,979构型检查、2条route候选；所选路径的最小禁触余量17.5466 mm，Pot 2非手指几何间隙4.3703 mm（对象门槛2 mm），携物到Ginger Beer/Place Box分别17.5466/17.5807 mm。规划器耗时360.055 s、Recovery耗时361.403 s，超过360 s限时约1.4 s；引擎按策略fail-closed，0个物理动作，未验证该路线的OSC可执行性。

当前没有Pot 2放置、Pot 1抓放或原生任务成功。后缀级Native Safety/Operational Safety/Strict-10mm均因0动作而未评估，不能标为通过或违规；本轮安全种子、标准1000步成功、扩展1800步成功均未取得。归档 `/opt/data/private/gs_opd/rollouts/l0t2_round0_t329_native_task_suffix_20261024_194230/`，入口 `research/gs_opd/repair/run_t329_native_task_suffix.py`。后续需新的连续执行授权来解决规划超时与OSC闭环交接；最终种子必须从安全t=10开始连续执行及复验。禁止把不兼容Snapshot拼接成任务轨迹，不训练LoRA；保留历史1093步示范、Episode 8/9及全部LeRobot数据。


## 上一总体状态：t=249真实抬升与接触自由退出通过；完整安全种子仍未取得（2026-10-24）

Round 0仍要求原生任务成功、全过程安全且证据完整的L0T2示范；当前没有安全种子。本轮使用唯一授权的t=249隔离分支，从按正式reset顺序严格恢复且Integration State精确匹配的Episode 9 Snapshot出发。局部OSC_POSE Lift Controller闭环生成80个动作、2,000个完整物理子步。Pot 2于t=311真实上升30.443 mm，阈值动作全部25子步无Pot 2–桌面接触；随后19动作、475子步继续无该支撑接触，终态t=329上升39.129 mm，双指持有稳定。

Safety Bridge现按MuJoCo joint type报告单位：hinge=rad、slide=m。`gripper0_finger_joint2`为范围`[-0.04,0] m`的slide。原生严格范围判断保持各单位1e-8；独立Operational finger-slide容差8 μm按Episode 9历史post-control qpos校准，不更改MJCF、qpos或BDDL。t=236起始越界4.179718 μm与旧安全汇总的“0”不一致，是因为旧Episode9的5,975个子步没有保存joint-limit字段，旧审计把缺失空列表计为0；目前应标为“未记录”，而非通过。单位与边界回归测试8/8通过。

验收分类：本局部分支 **Native Safety pass、Operational Safety pass、Strict-10mm fail**。Pot 2非手指机器人最小间隙4.316 mm；Ginger Beer 10.193 mm、Place Box 47.592 mm。接触规则未扩大：仅Geom 76/77/79/80允许finger–Pot 2接触，穿透上限2 mm；硬禁触对象、自碰撞、BDDL与关节模型限制仍受检查。原生任务仍false（两只Pot的On均false，reward 0、done false），旧t=11→249前缀没有逐子步接触身份，故局部结果不构成完整安全种子。

分支归档：`/opt/data/private/gs_opd/rollouts/l0t2_round0_t249_operational_lift_20261024_branch1/`；复现：`python -m research.gs_opd.repair.run_episode9_t249_lift_branch --maximum-actions 80`。下一道门是在具备逐子步finger-only安全证据的合法完整Episode内完成Pot 2放置、Pot 1抓取/放置及原生成功验收；本轮不启动完整Episode、不训练LoRA、不重做t=236抓取搜索。历史示范、Episode和LeRobot数据原样保留。

## 上一轮总体状态：任务优先安全重构仍未取得物理抓持；8个t=236分支均重复首步限位失败（2026-10-24）

Round 0仍以一条完整、安全、原生任务成功的L0T2轨迹为第一道验收；当前没有种子。已把净距参数改为逐对象配置：Pot 2非手指机器人Geom首档2 mm、必要时0 mm且不允许穿透；Ginger Beer/Place Box默认10 mm，只在实测正间隙确认其为阻塞时逐个试5 mm再2 mm；指尖与Pot 2只允许四个核验Geom、穿透最多2 mm。Bridge、RecoveryConfig、RRT及离线候选筛选已支持这些分档，原生BDDL约束与模型关节range保持不变。

严格t=236恢复通过，但本轮8个物理分支各执行首个动作后均因已张开的Panda夹爪被继续发送`open=-1`而报finger joint低于机械限位4.186×10⁻⁶ m（该finger joint是slide，旧日志曾误标为rad），未进入有效接近/抓持验证。每动作25子步、双相机、接触/间隙、Snapshot等证据已归档；Pot 2、Ginger Beer、Place Box的已测最小几何净距均超过10 mm。未观察到禁触物接触、自碰撞或Pot2手掌接触。Bridge保留按MuJoCo joint type解释的原生严格判据；入口改成从张开饱和位置向闭合方向更新后以neutral保持。8个物理分支预算已消耗。

归档：`/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024/episode1_t236_relaxed_clearance_physical_branches_20261024/`。入口：`research/gs_opd/repair/run_t236_physical_grasp_branches.py`。下一停靠点是在新获批物理预算内验证夹爪饱和时的保持策略，再继续目标阶段；不能把本轮分支标为安全抓取成功或Strict-10mm Safe。历史1093步示范、Episode 8/9和全部LeRobot数据保留。

## 前一阶段总体状态：handoff死循环已修复；48组候选均未通过精确MuJoCo检查（2026-10-24）

Round 0目标仍是先产生一条完整安全成功L0T2示范；目前没有安全重接入构型、物理抓取或成功种子。t=236的重复handoff来自恢复全局12 mm/0.08 rad与Approach 8 mm/0.06 rad判据分裂：EEF误差11.267 mm时恢复层回报成功，阶段层仍判未到；9.208 mm的hand–Pot 2预测余量使同一动作连续被安全拒绝，循环94次。两层现已统一按当前阶段的判据，且同状态无进展handoff默认至多3次后fail-closed；25项相关回归测试通过。

在恢复校验通过的同一t=236 Snapshot上，MuJoCo scratch筛查24组腕姿/方向/冗余IK候选，0/24完整可行（12组预抓取几何/自碰撞、12组无关节限位内IK）；CuRobo Franka GPU多目标IK另生成24组候选，96个阶段目标中52个CuRobo IK成功，但联合`plan_grasp`在预抓取规划失败。全部候选回到当前MuJoCo模型与指Geom 76/77/79/80白名单复筛后仍0/24可行：15组预抓取IK失败、7组预抓取Geom/自碰撞、2组Approach触犯Pot 2的10 mm/接触语义。RRT候选0、物理分支0、执行动作0。CuRobo只提供提议，不认证安全。

因此本轮结论是有界筛查未找到安全抓持/抬升终点或可达路径，不能扩大成全局不可达证明；也未测试物理闭爪、30 mm真实抬升或后续Pot 2放置/Pot 1重接入。当前保留已采集的t=236前缀及Snapshot，不新增Episode、不训练LoRA。复现入口是`research/gs_opd/repair/search_t236_pot2_grasp_reentries.py`和`research/gs_opd/repair/run_t236_curobo_grasp_reentries.py`，报告位于Round 0归档的`episode1_t236_grasp_search_20261024/`与`episode1_t236_curobo_search_20261024b/`。

后续路线只在产生新的、有来源的抓取/接近假设后继续有限离线搜索；任何真实OSC_POSE Snapshot分支都必须先通过精确全机器人预抓取、Approach、闭爪稳定持有、≥30 mm抬升和safe-exit几何筛选，且仍使用每次RRT最多3候选、50/60 mm风险滞回、进展型预算与停滞退出。不得降低10 mm约束或扩大finger whitelist。历史1093步任务成功示范、Episode 8/9和所有LeRobot数据保持原样。

## 上一轮总体状态：Round 0首条种子未取得；一次授权Episode在Approach处停机（2026-10-24）

### 结果

t=249直接restore失败是探针遗漏正式入口的原生`reset`顺序；reset后模型SHA与归档完全一致，编译字段变化只涉及`body_pos`，运行依赖和BDDL指纹相同。t=249 Runtime Snapshot在正式顺序下原生恢复成功、317维integration state精确一致。由于外部`TaskPhaseController`不在Bridge Snapshot中，本轮按要求没有拼接t=249后端状态，而从原始Student t=10 Snapshot启动新Episode。

复用`tsfs_01_axis2_rot+30`且未重筛24候选。TaskPhaseController自然推进到Pot 2 Approach。t=236之前的226动作、5,650/5,650个子步安全记录通过：执行中的Ginger Beer最小余量12.122 mm、Place Box 48.828 mm；Panda接触许可精确限于四个finger collision/pad Geom，Palm/hand持续受约束。

Approach下一动作的隔离预测检测到Palm/hand—Pot 2净距9.208 mm，低于10 mm要求；动作被拒绝。EEF距恢复目标11.267 mm，已在12 mm位置容差内，但阶段仍为`approach`；同一7D动作触发94次相同自动接管/handoff循环，没有物理推进。按停滞 fail-closed 规则停止。停机阶段是Pot 2 Approach，非规划不可达、非物理抓取失败：RRT到达预抓取并通过；失败是Approach末端动作将仍受约束的hand几何带入Pot 2 10 mm边界，现有phase handoff在相同timestep循环。

### 验收和资源边界

当前新Episode任务成功=false、完整物理安全成功=false（已执行前缀通过但未完成任务）、标准1000步成功=false、1800步扩展示范成功=false。没有30 mm抬升或任何放置，因此不生成Round 0种子。两路视频、226步动作/子步事件、t=236 Runtime Snapshot和从MP4恢复的部分OA数据保存在`/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024/episode1/`。TrajectoryWriter未finalize；失败记录不用于训练。没有重跑候选、插值扩增或LoRA。

本次增量搜索为0个新抓取候选、2条在线RRT（6,332构型检查、120.18秒）、226个物理动作。历史24组候选、选中预检路线4,448检查/101.818秒作为先验保留，不计作本轮重搜。 t=10安全前缀、原始1093步任务成功示范和全部LeRobot数据保留。

### 下一停靠点

授权的一次完整Episode已用完，本轮不继续物理执行。下一步先对失败动作做局部静态诊断：令真实`approach`目标/阶段切换在hand/Palm及全机械臂Geoms仍保持≥10 mm时可行，同时只对四个真实手指Geom允许目标物接触；修复同timestep无动作handoff循环的早退保护。不得改10 mm硬边界、不得把hand加入白名单、不得重开24候选搜索。任何新完整物理Episode需要新的明确预算。

## 历史总体路线状态：Task-Success-First 安全重构未达示范验收（2026-10-23）

### Round 0 主线

复用已归档1093步原生任务成功演示的阶段/抓取/放置先验，但绝不回放其21–56 Ginger Beer违规动作。由原始π₀.₅ t=10安全Snapshot出发，先搜索满足当前布局下任务需要的安全Pot 2抓持与抬升状态，再用全机器人7D RRT-Connect寻找可到达的安全路径；获得物理抬升后才从该新状态重接已有任务阶段控制器，连续重新生成并验收完整原生任务。Round 0只在原生完整任务成功、全程禁触规则通过且25子步证据完整后才可产生安全示范候选；否则不训练LoRA。

### 当前证据与结论

- 历史只读审计确认动作21–56共36步发生Ginger Beer接触（峰值12.096849 N）；t=20到t=118瓶子移动351.670 mm，而两只Pot及Place Box位置不变。Pot 2历史闭爪到t=141抓稳时自身又移动10.602 mm，因此以历史物体相对抓持变换为先验，在安全t=10物体坐标中重新表达，并搜索多方向/多腕姿/多7D冗余构型。
- 24个实质不同离线候选中1个通过全部scratch抓持/抬升/退出筛选。对其3条RRT路线候选的预检用时101.818秒、4,448构型检查；实际选定路线最小禁触余量25.031 mm，50/60 mm风险滞回、每次RRT最多3条候选及基于进展的总预算保持不变。
- 两次独立从已验证t=10 Snapshot启动的物理Episode预算已用完。Episode 8（237动作）到稳定抓持后被Pot 2与桌面支持接触的策略误报停止；Episode 9（239动作）通过真实双指稳定抓持，但切换到 `lift` 时目标夹爪接触白名单被清空，再次误报并在首个抬升动作前停止。Episode 9的5,975/5,975子步均有完整记录，无禁触/自碰撞，Ginger Beer与Place Box最小余量分别10.1907和47.4795 mm。30 mm真实抬升未验证，Pot 2 native On=false；其它任务阶段未执行。`task_success_first_result.json`和完整视频、Snapshot、动作/接触轨迹在TSFS归档目录。
- 已修正现有阶段接触策略，使经验证的被持目标仍由夹爪持有时保留精确目标夹爪Geom接触许可（穿透限值−2 mm）；其它机械臂Geom、Ginger Beer、Place Box的10 mm硬边界保持不变。该修复未在物理执行中复验，不能标作lift-safe。

### 验收与下一道门

路径可以到达安全任务抓持构型，且真实双指抓持已经得到；但Episode终止于 `lift` 阶段开始前，因此缺少真实≥30 mm Pot 2抬升、抬升后退出/携物几何和后续任务重接入验证。完整任务、完整安全成功、标准1000步Benchmark成功与demo-only扩展成功均未取得，没有Round 0安全成功候选。两次Episode已达到授权上限，当前停止物理运行。

下一道门是严格恢复Episode 9 t=249 Snapshot，逐项确认Bridge/Snapshot模型和BDDL、物体姿态、夹爪两侧持物、任务阶段锚点和传感器/控制器状态与存档动作末态完全兼容；只有确认这仍是同一合法物理Episode的无缝恢复，才可从该状态验证真实抬升和退出。若连续性无法证明，则先取得新的Episode预算，再启动从t=10的独立尝试。历史1093步示范及现有LeRobot数据保持原样。

## Historical task-phase regression follow-up（2026-10-23；由上方Task-Success-First结果更新）

1. **通用安全恢复核心：保留。** Student风险接管、7D RRT-Connect、原生OSC_POSE滚动控制、隔离 Snapshot 候选验证、真实完整机器人几何和25个物理子步检查均保持。禁触对象10 mm硬余量不变。
2. **预算回归：已修复并覆盖测试。** 单次RRT候选路线仍最多3条；阶段重规划预算只计有真实进展的控制段后重规划，最多12次/阶段、72次全局。移除了规划调用累计3次即停止的阶段门槛；连续无进展和相同重规划原因仍 fail-closed。风险边界用50 mm进入、60 mm实际退出的滞回，避免同一风险带重复无效触发。
3. **run9恢复状态筛选：完成。** 旧 t=238 与 t=258 状态分别只有0.419/2.277 mm Pot2全机器人几何间隙，不符合新10 mm目标接触规则；没有放宽规则。逐步记录筛得符合新规则的 t=240 Snapshot，Pot2间隙30.9 mm、Ginger Beer/Place Box至少24.4 mm。
4. **新阶段控制器有限验证：预抓取与安全接近通过，抓取端点被拒绝。** t=240恢复后30控制到预抓取（局部恢复成功）、18控制到 t=287。下一控制候选预测会使 robot link5 到 Ginger Beer 仅9.83 mm，已隔离拒绝；3种有来源抓取变换端点均被 RRT 判定不安全。闭爪、稳定抓持与抬升尚未开始，这是当前首次确定阻塞。
5. **任务/安全验收：均未取得。** 依赖抓取关卡的 t=10与t=0完整任务没有启动；两个Pot放置、炉具 Turnon 和原生谓词验收未执行。任务成功、完整安全成功、标准1000步Horizon成功、扩展示范成功均为 false，没有安全示范候选，没有覆盖既有数据或LoRA。
6. **历史任务阶段先验：仅作可追溯专家输入。** 原生 π₀.₅ t=118–141 Pot2抓取、425步Pot2放置、135步Grasp Retargeting和239步最终放置已复核。239步续接在t=1093满足全任务谓词，使用1300步demo-only horizon且原始前缀有Ginger Beer物理接触，不满足本项目完整安全成功标签。历史动作不拼接，所有新控制需从当前状态重算。
7. **下一优先事项：** 取得新的有来源且能通过完整机器人端点检查的Pot2抓取姿态/接近方向；最多三个候选。之后先在隔离环境验证连续双指稳定抓持和真实30 mm抬升，再允许最多两次完整Episode尝试。若候选均不安全，停止L0T2专用搜索并报告专家输入需求，不增加预算掩盖可达性或控制问题。

完整复现命令及运行记录见 [research/gs_opd/AUTOMATIC_TASK_RECOVERY.md](../research/gs_opd/AUTOMATIC_TASK_RECOVERY.md) 与 [GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md)。

---

## Previous Automatic Safety Recovery checkpoint（2026-10-23；stage-controller follow-up supersedes task-stage status below）

1. **通用安全恢复机制：已实现。** 状态机、7D 当前状态 RRT-Connect、隔离控制前缀 Snapshot 预演、原生 OSC_POSE 分段执行、全 Geom / 25子步监测、偏差/停滞/窄通道自主重规划、预算和 fail-closed 状态已集成。10 mm 禁止接触硬余量不可降低。完整当前状态和可复现 CLI 见 [`research/gs_opd/AUTOMATIC_SAFETY_RECOVERY.md`](../research/gs_opd/AUTOMATIC_SAFETY_RECOVERY.md)。
2. **L0T2 预抓取自动避障：已验证到局部目标。** t=10 Snapshot运行228控制步/3次重规划；Episode从 t=0运行258控制步/2次重规划。均自动到达按 Pot 2 实时位姿重算的历史成功抓取预抓取先验，Episode 的结果为 `local_recovery_success=true`。完整任务、完整安全和标准1000步 Horizon均未成功。
3. **L0T2 抓取/转运/放置：待任务专家输入。** 当前必须提供基于真实现态的抓取控制目标、held-object转运与放置目标/控制器、阶段转移、容许的目标接触对象/Geom及其可追溯来源。当前对 `grasp` 阶段显式失败，不复制历史动作、不猜目标。不得将历史先验说成 VLM Teacher输出。
4. **携物绕障：通用几何接口已补，L0T0 任务实测待做。** Bridge验证实时双指抓持，从实际 EEF–物体变换在 RRT scratch中移动物体，并对真实物体 Geom逐个物理子步监测。该绑定已经在归档 L0T2 Pot 2 双指抓持状态通过只读 Bridge 集成检查，完整查询 Ginger Beer、Place Box、Pot 1、Stove 与其余场景对象，禁止对象最小间隙204.2 mm；未推进物理状态。需要 L0T0 活动任务约束、禁止/允许接触的可靠映射和阶段目标后，才可运行端到端验证。这不等于 L0T0 任务验证成功。
5. **安全示范与训练：暂缓。** 目前没有完整任务成功且满足禁止接触约束的样本，故没有安全成功候选，不启动LoRA；1093步历史示范和其它记录不覆盖。

**验收分类固定区分：** 局部恢复成功 ≠ 完整任务成功 ≠ 完整安全成功 ≠ 标准1000步 Horizon内成功。未来仅在真实任务成功、完整禁止接触约束有全程证据且物理子步覆盖完整时，才可标记安全成功候选；扩展 Horizon必须单独标注。

---

## 最新局部状态（2026-10-23）：L0T2 t=10 关节空间 RRT-Connect 获得离线可行路线

从已验证的原始 π₀.₅ t=10 Runtime Snapshot构造 q_start；从历史 Pot2 成功预抓取的物体相对 EEF 位姿在当前场景重求 q_goal。两端的完整机器人碰撞几何、关节限位、Ginger Beer/Place Box 10 mm余量以及目标 Pot2接近例外均单独核验。固定随机种子20261023的双向7D RRT-Connect第11次迭代连通；Shortcut接受5条边，最终关节路径2.730396 rad、EEF路径0.697530 m，包含4个关节路点和257个密集样本。

完整路径逐样本复检通过。Bottle和Place Box最小余量为14.879/10.872 mm；目标Pot2最小间隙0.033 mm对应终端指尖无接触近接，其余机械臂部分与Pot2至少128.719 mm；无机器人环境接触、自接触或关节越限。模型和BDDL指纹匹配。独立归档 /opt/data/private/gs_opd/rollouts/l0t2_round1_jointspace_rrt_connect_20261023_attempt1/。

本结果只说明 offline_feasible，不证明 OSC 动态跟踪或真实抓取可行。没有环境步进、动作、Rollout、训练，未修改成功示范和LeRobot数据集。当前等待复核。

## 下一总体目标

复核离线路径及接近例外规则。其后如重新进入物理验证阶段，从同一 t=10 Snapshot 独立恢复，使用原生 OSC_POSE 监测全程接触、EEF误差及关节状态；不得把离线通过自动视作 OSC_executable 或 safety_success。已有1093步任务成功示范及759条LeRobot专家候选保持原样。


## 历史局部状态（2026-10-23；上一轮人工构型候选0/3）

正式停止 t=291，复用已验证的原始 π₀.₅ t=10 Runtime Snapshot，在独立 scratch MuJoCo 中以Integration State和原始前10步物理记录构造 q_start。缓存Observation与状态存在1.058 mrad关节及0.586 mm EEF差异，规划采用Integration State并留存该差异。历史成功 state118 的物体相对预抓取位姿用于目标；在当前Pot2 Body上重新求得 q_goal，IK位置/姿态残差低于3 mm/0.03 rad，终端限位余量0.609 rad，目标q数值与历史解相同是同一Body位姿重建后的求解结果。

3条完整关节空间候选均经全15个机器人碰撞Geom和Bottle(137)/Place Box(4)/Pot2(15)/Pot1(15)/Stove(21)逐采样检查，EEF采样≤5 mm、Robot Geom中心位移≤4 mm、关节增量≤0.02 rad。直连边穿透Bottle 41.52 mm；西南路线最后一段碰Place Box 4.94 mm并使Bottle余量降到8.15 mm；起点抬升调腕路线穿透Bottle 27.13 mm。路径样本均未越关节限位，最小关节余量0.609 rad，完整候选Jacobian σmin最低0.05455。没有候选通过，也没有OSC、Rollout、VLA或训练。

完整离线证据：`/opt/data/private/gs_opd/rollouts/l0t2_round1_task_directed_jointspace_20261023_attempt1/`。这不是可执行或安全成功认证。1093步任务成功示范与759条LeRobot候选均未改。

## 历史下一步建议（上一轮）

本轮只支持“简单直线关节边及少量人工中间构型不足以连通安全目标”这一结论；没有证明预抓取目标全局不可达。待复核后，可在同一t=10状态和同一完整碰撞距离检查器上实现/评估关节空间RRT-Connect，保持Bottle/Place Box 10 mm余量、原关节限位和3 mm IK残差门槛。RRT-Connect是目前有证据支持的下一项离线搜索方法，但未证明数学上唯一必要；本轮没有安装或运行新规划库。离线通过后仍需独立评审，再以原生OSC_POSE从同一Snapshot进行真实物理可执行性验证。


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

## 历史复核建议（截至2026-10-22）

先复核 759 条 OSC 专家候选的示范质量、状态起点和路径引用，再决定是否将其并入训练转换。没有复核前不启动训练或 Rollout；不改变原始 VLA 动作/Observation，不认证安全。L0T1 与 L0T0 保持暂停。

## 历史总体路线记录

## 最新局部状态（2026-10-22）：L0T2 t=854 搬运到位后下降触发持续 Pot1–Pot2 接触；停止等待复核

从已验证 Pot 1 双指抓持快照 t=854 恢复；隔离 Demo Bridge horizon=1300，标准评测 horizon=1000，模型、BDDL、物理和 reward 未改。第一条分段 OSC_POSE 路线用了 113 步，完成主动抬升至 Z=1.19936 m、高位西侧绕行及 Cook Region 上空进场。t=967 的慢速下降产生此前不存在的 Pot1–Pot2 物理接触，峰值 40.139 N / 25 个子步，超过硬停止阈值；Pot 1 尚未达到释放高度。Pot 2 位移最大 0.733 mm，Pot2 On 和 Stove Turnon 仍真；Pot1 On=false、reward=0、done=false。所有已记录 Panda 关节检查通过。第二条不同初始方向的路线因明确持续挤压停止条件而未运行。

离散 OBB sweep 未预测该下降碰撞，因此不能作为 clearance 证明；真实 MuJoCo 接触作为权威结果。失败动作、Observation、物体状态、接触/力、视频及失败快照单独保存，未并入示范链。实验后 Bridge 精确恢复至 t=854 起点，Integration State 最大误差 0，双指抓持及已有 Pot2 On/Stove Turnon 均复核通过。完整任务、标准 Benchmark、超预算示范与安全成功均未取得。历史 VLA Ginger Beer 接触保留。详细归档为 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_final_placement_demo_20261022_attempt2/`。

## 历史下一总体目标（2026-10-22）

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
