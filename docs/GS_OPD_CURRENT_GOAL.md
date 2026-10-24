# GS-OPD 当前局部目标

## 当前局部目标：从t=463真实持物状态重建Pot 2放置路线；本轮按跟踪偏差fail-closed停止（2026-10-24）

本轮严格审计并复用t=359剩余257点路线的索引30。恢复状态、模型、策略、夹爪及Pot 2双指抓持核验通过；路线实际执行39步后在t=398超过0.18 rad关节路径跟踪限值，EEF误差仍在既定范围。对精确t=398 Snapshot严格恢复后只做一次既有RRT-Connect（193路点、199.249秒、4,590构型检查），保存了完整路线，再真实执行65步到t=463。

连续新增动作合计104/450，2,600/2,600子步均通过Native Safety与Operational Safety，无禁触/自碰撞/未许可接触、无Operational关节越限，Pot 2双指持有持续。Strict-10mm统计未通过，因Pot 2机器人非手指间隙约4.36 mm；当前2 mm目标物Operational门槛和Ginger Beer/Place Box禁触约束均未违反。t=463关节路线误差0.182045 rad越过冻结0.18 rad，而EEF误差为4.238 mm/0.034841 rad；唯一规划预算已用，故按maximum_replanning_attempts_exhausted停止。

t=463仍为transport，Pot 2 On=false，Pot 1 On=false，Stove Turnon=true，reward=0、done=false。没有Pot 2放置/释放、Pot 1阶段、完整任务成功或Round 0种子；不启动LoRA。完整合并审计为 /opt/data/private/gs_opd/rollouts/l0t2_round0_t398_pot2_single_replan_20261024_214225/posthoc_review.json，重规划路线及状态/策略指纹在同目录的replanned_route_t398.npz与JSON sidecar。OA、双相机、每步25子步、安全事件与Snapshot保留在t359和t398两个连续归档中。

本轮规划调用已用尽，当前任务暂停在t=463真实Snapshot；后续需有新预算，先针对7D RRT路点和OSC_POSE冗余关节跟踪偏差作有界诊断，再从严格恢复的t=463状态重规划。不放宽0.18 rad、安全余量、禁触规则，不重放旧动作，不把局部前缀拼成完整种子。

只读审计入口：

/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python -m research.gs_opd.repair.resume_t359_pot2_placement --audit-only
/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python -m research.gs_opd.repair.resume_t398_pot2_single_replan --audit-only --source-run /opt/data/private/gs_opd/rollouts/l0t2_round0_t359_pot2_placement_20261024_213308


## 前一局部目标：从t=359安全续接Pot 2 transport；本轮30步预算已用尽（2026-10-24）

上一阶段“RRT有可行路线但超时零动作”已通过规划结果复用解决：从严格恢复的t=329状态校验Integration State、模型/策略指纹和路线端点后，直接加载归档257点完整路线，没有再次运行360秒RRT。缓存路线的首次可行候选在iteration 10；该时刻墙钟时间未记录。持物预演夹爪命令已改为neutral `0.0`，并记录来源。

4段独立预演共32 OSC动作/800子步全部通过；之后授权内真实执行30动作/750子步至t=359。Pot 2双指抓持在全部750子步稳定，未发生禁触、未知或自碰撞接触，Operational关节限位违规为0。EEF与Pot 2分别移动74.187/75.474 mm，Pot 2真实上升32.529 mm；关节、EEF跟踪均在既有阈值内。机器人非手指–Pot 2最小净距4.370 mm、Ginger Beer 18.776 mm、Place Box 42.177 mm。因此 **Native Safety和Operational Safety通过；Strict-10mm不通过**。

在t=359因30个动作上限耗尽停止，阶段仍为 `transport`，Pot 1/2 On=false、Stove Turnon=true、reward=0、done=false。这里是预算停止，不是规划或控制失败；未完成放置、原生成功或Round 0种子。t=359续接Snapshot和剩余路径bundle已保存，但bundle尚未做恢复预验证：`/opt/data/private/gs_opd/rollouts/l0t2_round0_t329_native_task_suffix_20261024_205500/snapshots/main_timestep_000359.pkl` 与同目录 `route_resume_bundle_timestep_000359.npz`。恢复后必须核对Integration State SHA `8d74c65c32250f4542bec0efab17a536e92b2b3ab9ba5b463604108683981874`、Snapshot SHA `ce7c004543a0fc19217c46c65ccfd82a561bb90999dfce0f75cd405dbb82ab47`，然后由路径索引30继续。

物理归档及审计详见 `GS_OPD_PROGRESS.md` 最新节和 `/opt/data/private/gs_opd/rollouts/l0t2_round0_t329_native_task_suffix_20261024_205500/route_reuse_prefix_audit.json`。本轮复现入口为 `python -m research.gs_opd.repair.run_t329_native_task_suffix --route-archive /opt/data/private/gs_opd/rollouts/l0t2_round0_t329_native_task_suffix_20261024_194230`。本轮真实动作预算已用完；任何续跑需新授权，并先严格恢复t=359，不重复路线搜索，也不将t=329后缀与缺失前缀拼接。历史任务成功示范、Episode档案和LeRobot数据保持原样，不训练LoRA。


## 上一阶段局部目标：从t=329继续L0T2任务；首次后缀因RRT时间预算停止（2026-10-24）

t=329 Episode 9来源Runtime Snapshot已按正式reset顺序严格恢复：模型与Snapshot SHA匹配，手臂、EEF、夹爪、Pot 2位姿核对一致；实时双指持有和finger-only接触核验通过。新TaskPhaseController基于当前物理状态初始化为 `transport`，Pot 1/2 On仍false，炉具Turnon为true。

本次唯一后缀调用既有RRT-Connect，117次迭代、7,979构型检查后记录两条离线可行路线，并选中一条。所选路线最小禁触余量17.5466 mm；Pot 2非手指间隙4.3703 mm（高于该对象2 mm门槛），Ginger Beer/Place Box携物间隙17.5466/17.5807 mm。但RRT调用耗时360.055 s、Recovery阶段合计361.403 s，超过配置360 s规划预算约1.4 s，Safety Engine在transport阶段fail-closed。物理动作0、Episode仍t=329，因此没有搬运或接触违规证据。

该本轮一次后缀预算已结束，不自动重试相同规划。源t=249→329分支保持Native/Operational Safety通过、Strict-10mm未通过；本次0动作后缀的Native/Operational/Strict-10mm轨迹验收均未评估。原生任务成功=false、标准1000步成功=false、扩展1800步成功=false，未建立Round 0种子。详见 `GS_OPD_PROGRESS.md` 最新节与归档 `/opt/data/private/gs_opd/rollouts/l0t2_round0_t329_native_task_suffix_20261024_194230/`；入口为 `python -m research.gs_opd.repair.run_t329_native_task_suffix`。

后续完整种子仍必须在获得新的连续执行预算后，从安全t=10状态执行完整单Episode并统一复验。不可把t=249→329来源分支或本次t=329恢复后缀与缺失前缀拼接；不启动LoRA。历史示范与LeRobot数据保持不变。


## 上一已验证局部里程碑：t=249真实抬升与接触自由退出通过（2026-10-24）

本轮唯一隔离物理分支从 Episode 9 t=249 Runtime Snapshot 按正式 native reset 顺序严格恢复：模型几何SHA `1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752`，317维Integration State SHA `ff38bea1ff53331bbf97d4f8f70fe6301ab5e81635b7187ee70ffbc2bc258094`、时间步249、Pot 2双指抓持和四指Geom白名单均通过。source/re-export Snapshot字节哈希不同，但积分状态逐元素一致；未跳过指纹检查或改写qpos。Bridge报告中的hinge单位为rad、slide为m；`gripper0_finger_joint2` 范围为`[-0.04,0] m`。Native严格关节判据保留1e-8（按关节单位），独立Operational finger-slide容差8 μm由Episode 9历史动作末端数据校准；t=236初值越限4.179718 μm，不能据此推断旧前缀逐子步限位通过。

使用独立局部Lift Controller从恢复后的真实EEF与Pot 2姿态闭环生成OSC_POSE动作，夹爪neutral `0`保持已闭合命令。80动作/2,000物理子步均保存；t=311时Pot 2真实竖直上升30.443 mm，阈值动作完整25子步无Pot 2–桌面支撑接触，之后19动作/475子步也保持支撑接触清零，t=329终态上升39.129 mm。双指抓持保持；最大合法指尖穿透0.5273 mm，其他机器人Geoms无违规接触，无自碰撞、Operational限位违规、Ginger Beer/Place Box禁触或原生任务完成事件。

验收单独报告：Native Safety pass；Operational Safety pass；Strict-10mm fail（机器人非手指–Pot 2最小4.316 mm；Ginger Beer 10.193 mm、Place Box 47.592 mm）。原生任务仍未成功：Pot 1/2 On均false，reward=0、done=false。因此这只是t=249起Pot 2局部抬升与安全退出成功，不是完整Episode或Round 0安全种子。旧t=11→249前缀逐子步原始接触身份不完整，不能将其升级为满足当前finger-only规则的完整安全前缀。

归档 `/opt/data/private/gs_opd/rollouts/l0t2_round0_t249_operational_lift_20261024_branch1/` 含双相机Observation–Action、82,490条逐子步接触Geom/Body身份、完整25子步安全记录、Snapshot、视频、奖励/done及原生谓词。测试：单位与边界回归8/8通过，相关模块`py_compile`通过。复现入口：`python -m research.gs_opd.repair.run_episode9_t249_lift_branch --maximum-actions 80`。本轮不启动完整Episode、候选重搜或LoRA；历史1093步示范及全部LeRobot数据保留。

## 历史局部目标：t=236抓取分支停止，8次物理分支预算已用完（2026-10-24；被上方t=249实测进展取代）

### 本轮变化

将安全净距改为逐对象显式配置：Pot 2非手指机器人Geom可设2 mm、必要时0 mm且不得穿透；Ginger Beer/Place Box仍默认10 mm，只在实际正间隙低于当前档位时才逐对象尝试5 mm、2 mm；目标夹爪许可精确限制到finger Geom 76/77/79/80，最大穿透2 mm。Bridge几何策略校验、RecoveryConfig、MuJoCo RRT与scratch候选筛选器均不再写死10 mm作为所有对象的最低可配置值；每个对象的碰撞分组阈值仍独立执行。Stage/BDDL、MuJoCo关节边界与任务谓词未改。

从t=236归档Snapshot按正式native reset后恢复，模型SHA `1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752`与Integration State SHA `a299ea309f78435c868306716bfa403834d20f3599473793b25612e93207c0f8`完全匹配。首次物理分支揭示：夹爪已在张开机械限位时，OSC仍重复发送`gripper=-1`，指关节 `gripper0_finger_joint2`（MuJoCo slide）在25子步积分中的最大越界为 **4.186×10⁻⁶ m**（qpos −0.0400041863，限位 −0.04）。安全监测因此正确fail-closed。原生严格比较值保持`1e-8`对应单位；分支入口先发闭合方向更新、后续以neutral保持。历史记录中标为rad属于诊断单位错误，详见当前进度首节的单位勘误。

已执行 **8个隔离分支、每支1个控制动作、共8动作**，每动作均有25个物理子步、双相机Observation、接触/间隙、reward/done、原生任务状态、grasp state与可恢复Snapshot；没有成功分支。各分支首个动作都被同一夹爪限位数值告警终止，实际上重复了这一基础控制问题，未完成设计的腕姿/接近差异验证。几何最小值在这些有限前缀中为Pot 2非手指 **10.571 mm**、Ginger Beer **11.734 mm**、Place Box **48.536 mm**；未观察到自碰撞、禁触物接触或Palm–Pot穿透。尽管该有限前缀实际距离均超过10 mm，Joint-limit监测未通过，所以不能标记为Strict-10mm Safe。

最近物理里程碑仍为Pot 2 `approach`。未实现稳定双指抓持、30 mm真实抬升、safe exit、Pot 2放置、Pot 1抓放或原生完整任务成功；没有启动条件性的完整Episode预算，没有建立Round 0种子，也没有训练LoRA。8分支上限已经耗尽，本轮不再发起物理动作。数据归档及可复现入口：`/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024/episode1_t236_relaxed_clearance_physical_branches_20261024/`、`research/gs_opd/repair/run_t236_physical_grasp_branches.py`。

### 下一道门

先读审计已归档分支的joint-limit事件与首动作，再由用户授权新的有限物理分支预算后，从严格恢复的同一t=236 Snapshot重试。新执行保留`1e-8 rad`限位比较精度，验证全开夹爪的1%闭合方向更新与neutral保持策略；之后才有资格验证逼近、闭爪、连续双指持有、≥30 mm抬升及退出。只有该目标抓取/抬升通过，先前条件授权的最多两次完整Episode才可使用。

## 前一阶段：修复Approach重复handoff并完成48组离线筛查（2026-10-24）

### 阶段判据冲突与停滞保护

t=236的EEF位置误差为11.2673916 mm。运行配置的全局恢复阈值是12 mm/0.08 rad，Approach阶段阈值是8 mm/0.06 rad。安全Bridge拒绝下一动作（hand到Pot 2为9.207675 mm），物理timestep保持236；Recovery Engine按全局阈值认为恢复目标已到，TaskPhaseController按Approach阈值认为阶段未到。运行时因此循环“安全拒绝→自动接管→恢复目标成功→task expert handoff→同一动作再拒绝”94次。

现在`TaskPhaseController.is_pose_goal_reached()`统一提供阶段阈值判定，Recovery Engine和阶段状态更新使用同一结果；Approach依然要求8 mm/0.06 rad。Recovery Engine增加默认3次的同timestep/同阶段/无物理进展handoff上限，超限即fail-closed。没有放宽精度、安全余量或10 mm禁触要求。`test_task_phase_controller`、`test_automatic_recovery_runtime`和`test_automatic_recovery`定向测试25/25通过。

### t=236安全抓持候选与CuRobo复核

起点是归档真实t=236 Snapshot，经原生reset后恢复；模型SHA `1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752`、Snapshot SHA `355c59e82470702a2e7953d2fb00c45dc3e8c4cb2da1a2f48a2a7acae50d7a28`和状态SHA `a299ea309f78435c868306716bfa403834d20f3599473793b25612e93207c0f8`匹配。接触许可仅为四个finger collision/pad Geom 76/77/79/80，合法穿透上限2 mm；其余Panda Geom、Ginger Beer、Place Box仍需至少10 mm。

复筛了48组有界不同候选（原MuJoCo 24组 + CuRobo生成方向/腕姿24组）：原组中12组预抓取几何/自碰撞失败、12组IK未找到；CuRobo组中15组未在MuJoCo约束内找到预抓取IK、7组全几何/自碰撞失败、2组最终接近路径越过Pot 2 10 mm/接触语义。CuRobo GPU批量IK的96个阶段目标有52个优化解，但`MotionPlanner.plan_grasp`在当前到预抓取阶段失败；即便IK成功也不能当作MuJoCo构型或安全轨迹。两轮精确MuJoCo筛选均0个完成approach、stable grasp、lift≥30 mm与safe exit完整筛查的候选；RRT路线0，没有物理分支，也没有物理动作。

本轮停止类别是“当前有界候选均未通过精确MuJoCo验证”，尚未证明安全抓取终点全局不存在；物理抓取和后续任务重接入均未测试。新增脚本`research/gs_opd/repair/run_t236_curobo_grasp_reentries.py`与`curobo_t236_plan_grasps.py`保存重现入口，JSON证据在`/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024/episode1_t236_curobo_search_20261024b/`。当前没有通过离线筛查的候选，因此不开始OSC物理分支、完整Episode或训练。历史1093步示范、Episode 8/9和现有LeRobot数据保留。

## 上一轮唯一获批Episode：Approach停滞后fail-closed（2026-10-24）

### Snapshot初始化兼容性根因

原t=249探针漏掉正式入口在Bridge启动后执行的原生`reset`。零动作重放显示：Bridge启动模型指纹为`75c8db457aa094e5efc43050a4f07192a9e8dca1379e84a67f4854800466d574`，与归档不符；原生reset后变为归档指纹`1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752`。启动与reset之间唯一变化的编译字段为`body_pos`（32×3）；Python 3.10.21、MuJoCo 3.3.2、robosuite 1.4.0、LIBERO 0.1.0、NumPy 1.26.4及BDDL SHA均一致。按正式顺序恢复Episode 9 t=249 Snapshot成功，317维integration state逐元素相同，Pot 2双指接触与原生谓词`Pot1 On=false, Pot2 On=false, Stove Turnon=true`均恢复可见。未绕过指纹校验或改写物理状态。

### 唯一新Episode结果

从严格核验的原始Student t=10 Snapshot启动，TaskPhaseController在本Episode自然进入`pregrasp`（t=10）和`approach`（t=213）。复用已验证候选`tsfs_01_axis2_rot+30`，没有重筛24组候选。RRT-Connect接受两条在线路径规划（共6,332构型检查、120.18秒；路径最小禁触净距分别25.031/13.339 mm）。

在全局t=236、`approach`阶段，实际已提交的226个动作/5,650个MuJoCo子步仍安全：Ginger Beer最小余量12.122 mm、Place Box 48.828 mm、Pot 2最近机器人几何净距11.815 mm；每步25/25接触子步完整，无机器人接触、自碰撞、未许可环境接触或未知项。运行中实际白名单ID为76/77/79/80，分别对应finger1 collision、finger1 pad、finger2 collision、finger2 pad；hand/palm ID 74未许可。接触许可没有扩展到其它机器人Geom或物体。

隔离预测发现Approach下一动作会使`gripper0_hand_collision`与`moka_pot_2_g12`只剩9.208 mm，低于10 mm硬余量；该非手指动作未提交。恢复目标位置误差已在12 mm位置容差内（11.267 mm），但任务阶段仍是`approach`；同一个7D预测动作被拒绝并触发94次“unsafe expert prefix→local recovery success→expert handoff”，没有物理进展。我据此发SIGINT fail-closed并关闭Bridge/视频写入器。最远仅到Pot 2 `approach`，没有闭爪稳定抓持、30 mm抬升、放置或Pot 1阶段。t=236原生状态为Pot 1 On=false、Pot 2 On=false、Stove Turnon=true、reward=0、done=false。

事件日志、t=236 Snapshot、两路各227帧视频和恢复的部分OA NPZ保存在`/opt/data/private/gs_opd/rollouts/l0t2_round0_tsfs_safe_seed_20261024/episode1/`。NPZ中的动作、关节/物体状态和子步安全记录来自原始事件日志；相机图像由同步MP4解码（有损），不是原始RGB数组。SIGINT发生在TrajectoryWriter finalize前，故没有标准`observation_action_trace.npz`；该失败尝试不作为训练数据或种子。

| 验收项 | 当前结果 |
|---|---|
| t=249 Snapshot（正式reset顺序） | **恢复通过；317维物理积分状态精确一致** |
| 本Episode实际物理前缀安全 | 通过：226动作、5,650/5,650子步；最小Ginger余量12.122 mm |
| 实际Pot 2稳定抓持 / 真实抬升≥30 mm | 未达到 / 未执行 |
| 原生完整任务成功 / 完整安全成功 | false / false |
| 标准1000步成功 / 1800步扩展示范成功 | false / false |
| 本轮重新筛选24候选 | 0；复用既有TSFS-01 |
| Round 0种子 / LoRA | 未建立 / 未启动 |

用户授权的唯一Episode已用完，不再启动第二次。接下来需针对“Approach目标已到位但下一步hand净距不足，控制器在同状态handoff循环”作最小局部修复；任何后续物理执行都需新的预算授权。可复现配置生成入口为`research/gs_opd/repair/prepare_tsfs01_episode_config.py`，执行入口为`research/gs_opd/repair/automatic_recovery_cli.py`；完整失败摘要为归档目录内`episode1_fail_closed_summary.md/json`。历史1093步示范、Episode 8/9以及全部LeRobot数据保留。

## 历史结果：从已验证 Pot 2 稳定抓持状态完成安全抬升重接入（2026-10-23）

### 本轮已完成

- 只读审计并保留历史1093步L0T2任务成功链；第一处禁触违规是动作21–56的Ginger Beer接触。t=10、t=719、t=854、t=1093精确Snapshot存在；t=141、t=156、t=1021、t=1053没有对应精确Snapshot。逐里程碑的关节/物体相对姿态/原生谓词与文件可用性见最新审计JSON。
- 生成同一组24个实质不同的抓取与抬升候选；1组通过离线全机器人Geometry、关节、双指接触与抬升/退出筛选。RRT-Connect从原始Student t=10前缀的安全终态规划到预抓取，随后真实OSC接近，风险余量、滞回、全机器人Geometry与10 mm障碍物约束未放宽。
- Episode 9在同一合法物理Episode内真实达到 Pot 2稳定双指抓持（t=239起稳定持有），截至t=249位姿总移动3.071 mm，Ginger Beer最小余量10.1907 mm、Place Box 47.4795 mm。完整239步有5,975/5,975个子步记录；无禁触或自碰撞。引擎在进入 `lift` 后、首次抬升动作之前 fail-closed。

### 停止原因与验收

两次Episode预算已经用完，本轮不启动第三次。Episode 8因Pot 2仍支撑在桌面时，`unmapped_scene_environment`把目标物—桌面接触误认作障碍违规而停在 `stable_grasp`。Episode 9确认了真实抓持，但离开 `stable_grasp` 时安全策略清空目标夹爪接触许可；因此同一真实指抓持接触被误判为Pot 2 10 mm间隙违规（−0.528 mm）。该接触属于任务必需的夹持，不是 Ginger Beer 或 Place Box 接触；在此前全部物理子步中，禁触余量和接触记录均通过。未完成30 mm真实抬升，Pot 2原生On仍为false，后续Pot 2放置、Pot 1抓放与全任务谓词均未重接入。

已把现有策略修正为：只要实时抓持核验仍确认当前任务目标由夹爪持有，就维持当前任务目标的精确夹爪Geom接触许可，最大穿透保持2 mm；其它机器人Geom与障碍物禁触检查不变。该代码修复尚未物理复验。当前结果：真实安全抓持成立，安全抬升/任务成功/完整安全成功未得到验证；不建立Round 0安全成功示范候选，不修改历史示范或LeRobot数据，不训练LoRA。

归档：`/opt/data/private/gs_opd/rollouts/task_success_first_safety_reconstruction_l0t2_20261023_attempt1/`；机器可读结果为 `evidence/task_success_first_result.json`。后续若继续，先从Episode 9 的精确t=249 Snapshot做严格兼容性与抓持状态恢复核验，再只推进真实抬升与退出；若无法把该恢复证明为同一连续物理Episode，则需另行授权新的Episode预算。可复现CLI及固定t=10入口见 [GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md)。

## 历史状态：Pot 2 阶段抓取端点阻塞（已由上方Task-Success-First真实运行更新）

### 已完成

- 修复旧 maximum_plan_attempts_per_phase=3 把连续控制中的 RRT 调用数错误当作阶段预算的问题。每次 RRT 内最多3候选；实际推进后的阶段重规划预算最多12次（全局72），停滞或重复失败原因提前 fail-closed。
- 加入基于实时几何间隙的风险边界滞回：50 mm进入、60 mm实际退出后重新武装。完整25子步检查与10 mm Ginger Beer / Place Box硬余量保持不变。
- run9旧目标与新目标相差53.2 mm，且目标接触规则不同。旧 run9三条路线的禁触余量为12.054/11.283/12.471 mm，触发风险和窄通道重规划两次后于 t=258 到旧目标；新 t=10 三条路线余量19.571/18.485/18.448 mm，但连续控制段仍触发跟踪偏差、风险带和窄通道事件，旧三调用上限在 t=128、距目标232.6 mm时截断。现已删除该调用次数上限。
- 按新几何规则从 run9 逐步记录选取安全的 t=240 Snapshot（Pot 2间隙30.9 mm，Ginger Beer/Place Box至少24.4 mm）；严格恢复后30步到预抓取，另18步安全接近至 t=287。
- t=287下一 OSC 前缀在隔离 Snapshot 中预测 link5–Ginger Beer 为9.83 mm，未提交。三个来源明确的抓取目标变体均因 goal_configuration_is_unsafe 被 RRT 端点检查拒绝，触发每阶段最多3个目标候选的停止线。

### 当前阻塞与结果

首个确定阻塞是**现有三种抓取目标构型都没有满足完整机器人几何的安全终点**。t=270 已取得预抓取局部恢复成功，随后安全接近到 t=287；完整任务、完整安全、标准1000步 Horizon和扩展示范成功均未取得。没有闭爪、稳定双指抓持或真实抬升30 mm，不能据此判断目标空间全局不可达。需要新的可追溯抓取姿态或接近方向先验，并先通过同一全机器人几何与接触规则。历史动作不回放，当前动作仍由真实状态重新生成。

因此本轮不启动 t=10 后续完整任务或原生 t=0 Episode。Pot 2 放置、Pot 1 抓放、炉具激活和原生谓词验收未执行。任务成功、完整安全成功、标准1000步成功及扩展示范成功均为 false；未生成安全成功训练候选、不覆盖历史数据、不运行 LoRA。历史 425步 Pot2放置、135步抓取迁移、239步最终放置只保留为专家先验和既有结果；239步接续虽在 t=1093 完成原生任务，但超出标准Horizon且源前缀有 Ginger Beer 接触，不能标成安全示范。

自动恢复回归、阶段状态机、抓取 retargeting 等完整 repair 测试共174项通过，修改模块编译检查通过。实验归档及精确分项见 [GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md) 与 [research/gs_opd/AUTOMATIC_TASK_RECOVERY.md](../research/gs_opd/AUTOMATIC_TASK_RECOVERY.md)。

**下一步：** 在三候选上限内取得可信且端点安全的 Pot 2 抓取变换/接近方向；若没有这样的任务先验，则明确停止 L0T2 专用扩展并记录需要的专家输入。通过后才继续闭爪、连续双指抓持、30 mm抬升验证；该关卡成功后才能启动最多两次独立完整 Episode 尝试。

以下既有局部目标与人工分段 OSC 实验记录是历史证据，保留原样。

---


## 当前局部目标：t=323 接续完成150步并安全停在窄段前（2026-10-23；等待复核）

严格恢复t=323 Snapshot后，同一RRT-Connect从真实7D qpos重规划；旧路线偏差使直接复用关节游标不合适。新的113点 `offline_feasible` 路线满足Ginger Beer/Place Box 10 mm离线门槛，全程最小17.957/12.556 mm；路径首次低于20 mm处为参考索引205。

本轮用原生OSC_POSE执行150步至t=473，步数上限先于窄段边界触发。预抓取距离缩短135.989 mm至166.034 mm；EEF误差最大2.937 mm/0.008320 rad；真实障碍间隙最低33.962/80.286 mm，3,750物理子步没有禁止接触、自碰撞或关节越限。Pot未移动，未到预抓取或抓取，reward=0、done=false。终态快照与活动Bridge导出逐字节一致并可恢复，Bridge已关闭。

窄余量段仍须在后续获准物理执行中单独验证；当前结果不能标为完整Safety Success。证据位于`/opt/data/private/gs_opd/rollouts/l0t2_round1_t323_replan_osc150_20261023_attempt1/`，报告`evidence/t323_replanned_osc150_execution_report.md`。

**本轮结束，等待复核。**没有执行Student Rollout、放置或训练，历史成功示范和LeRobot数据未修改。

---

## 当前局部目标：从 t=223 重规划后执行100步，安全到达 t=323 并等待复核

已从严格验证的 t=223快照恢复。旧RRT剩余路径与真实7D qpos不一致，故从实际状态用同一MuJoCo RRT-Connect重新生成一条161点 `offline_feasible` 全路径；Ginger Beer/Place Box离线全几何最小间隙21.278/12.722 mm，保持10 mm门槛。

原生OSC_POSE在隔离Bridge执行100步至t=323，达到预算即停。EEF最大误差3.256 mm/0.007034 rad，预抓取距离减少69.453 mm至302.023 mm。2,500物理子步全量监测无环境/自接触、无障碍接触、无限位违规；真实最小间隙29.568/174.958 mm。Pot 1/Pot 2位姿未变，环境reward=0、done=false，预抓取尚未到达。t=323 Runtime Snapshot和Bridge末态导出逐字节匹配，SHA-256 `c333a23a…f818f410`；隔离Bridge已关闭。

本段仅为通过物理监测的有限安全进展，不是完整路线认证或Safety Success。完整路径的 Place Box 最小离线余量为12.722 mm；本轮前缀实测仍为174.958 mm，因此窄余量段尚未在线验证。末态可供复核后续续接。完整报告：`/opt/data/private/gs_opd/rollouts/l0t2_round1_t223_replan_osc100_20261023_attempt1/evidence/t223_replanned_osc100_execution_report.md`；Observation–Action JSONL/NPZ、视频、路径、物体位移审计、恢复检查及终态快照均在同一归档。

**本轮结束，停止等待复核。**不执行后续抓取、放置、VLA、Rollout或训练，不修改既有示范/LeRobot数据。

---


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

