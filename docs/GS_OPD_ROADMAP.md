# SafeTRAIL Stage A 路线图状态（2026-10-30）

覆盖11/30（9项Standard、2项Extended）。OAH/L0T1已新增760动作完整原生成功Episode并通过连续性/接触/BDDL审计。当前定向验证OAH/L1T1：t578合法持物前缀的offset0放置路线推进220动作至t798、目标距离0.228m后在下一路点停滞；现从精确t798物理Snapshot分支测试offset1容器朝向和20mm OSC步距，Standard剩余202动作。证据与运行配置见Current Goal及任务矩阵。

## 2026-10-30 15:31 CST route00 real descent is progressing


## 当前执行状态（2026-10-30）

覆盖12/30（10 Standard、2 Extended）。contact-only保留每子步接触和BDDL审计，精确Runtime Snapshot支持同Episode下游分支。L1T1预算保留修正已验证：最终插入路线成功时Stage A不再重复预留96步，避免剩余时限被固定扣除。下一项OA/L1T4已有t759持物状态，瓶颈是目标XY误差2.6mm后垂直OSC未跟踪、而非joint limit；计划尝试offset2 direct region route与20mm步距，并确保后续Close关节目标仍在Standard余量内。
覆盖11/30（9 Standard、2 Extended）。contact-only审计保留每子步真实接触与BDDL检查，省去Stage A不需要的全Geom clearance扫掠；精确Runtime Snapshot用于同Episode下游分支。L0T1已验证offset1原生区域OSC可完成放置。OAH/L1T1 offset0路线已从t578物理推进至t798，目标距离降至0.228m且抓持稳定，之后因24步OSC停滞结束；当前分支从t798 Snapshot测试offset1朝向并将步距由8mm提高到20mm，避免重复t1–t798。
At t282,36 actions/900 monitored substeps complete; both finger contacts hold. Bowl root lowered ~54.9mm to z1.08937m; controller has entered horizontal region_progress with32.9mm to the current waypoint. XY goal distance has moved only4.4mm during the mostly vertical descent, so keep monitoring rather than classify stagnation. Native success remains false.

## 2026-10-30 15:28 CST fixed continuation reached placement branch00

Exact t246 resume passed the former NameError and entered real `region_layout_12_direct_region_progress`; 21 durable camera frames appeared. Executor/Bridge remain active, action count and BDDL outcome pending.

## 2026-10-30 15:24 CST same-episode continuation launched

Cold-start continuation is running from the verified t246 handoff Snapshot inside the prior V3 archive; new run archive `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T0_reachability_biased_dev03_fix_continuation_01`, source payload SHA256 `f400d1206d951477b1f62f5bb3f2e8d2d6e4893bedaa98e74794fda002099ec8`. Horizon1000, reachability-biased native On transport. Preserve and audit the 246-action/6150-substep prefix; wait for MuJoCo placement and native BDDL result.

## 2026-10-30 15:14 CST OAH/L0T0 grasp skill physically verified; task still active

Physical branch from exact t0 completed214 actions and5350/5350 monitored substeps, 0 incomplete; stable bilateral grasp22 states, bowl lift36.4506mm. RRT preflight had failed; generic active-BDDL geometry high-approach OSC fallback succeeded in real grasp/lift execution. Candidate and records saved at `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T0_reachability_biased_dev_03/v3_run/task_success_engine/task_success_first_executor_obstacle_avoidance_human_L0T0_20261030_141837_336613925/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_human_L0T0_20261030_142606_922900193`. This is not native task success. V3/executor remain active; bridge PID152477/53532 started downstream execution. Continue placement and native BDDL audit before any coverage increment.

## 2026-10-30 15:09 CST OAH/L0T0 branch remains CPU-active

Physical grasp branch ~27m43s from Snapshot with no durable records/terminal report. Four run PIDs active, Bridge ~101% CPU. Matched prior branch record-save reference36m42s; continue observing current branch.

## 2026-10-30 15:01 CST OAH/L0T0 branch continues

Selected grasp physical branch ~20m from Snapshot, process active/CPU-bound, no physical records or terminal report emitted. Keep current code loaded; await branch outcome.

## 2026-10-30 14:57 CST OAH/L0T0 dev03 physical grasp status

Physical branch age ~15m24s since its 14:41:48 Snapshot, no durable post-action rows yet; Bridge process is CPU-active. Matched dev02 branch records took36m42s to persist. Continue current run; no code mutation while child imports/executed source.

## 2026-10-30 14:53 CST OAH/L0T1 zero-pass review broadens fallback admission

历史screen中15个碰撞拒绝候选的pregrasp/grasp IK全reachable，但预抓距离-99.46mm记为聚合`unmapped_scene_environment`；目标book距机器人约13mm，报告没有保存aggregate pair身份。该碰撞来源未证实，碰撞状态不能作为Stage A物理尝试硬门槛。screen-pass-first fallback只排除显式IK不可达项；预抓screen通过和approach碰撞出现较晚只作为排序偏好；pass数不足预检名额时有限追加其余IK可达候选，并记录真实contacts，成功仍需原生BDDL。

## 2026-10-30 14:50 CST screen-refusal evidence refines zero-pass fallback

OAH/L0T0 dev03的14个screen拒绝候选中，13项有reachable pregrasp/grasp IK，失败主要出现在末段约5mm finger-target penetration。Stage A可探索这种候选但不能把screen拒绝当作成功；使用screen-pass-first候选序列；pass数量少于本轮RRT预检名额时，将IK可达的screen拒绝项追加到后面补足名额。明确IK不可达才硬排除；pregrasp screen通过及晚发生接触只是排序偏好，不作物理尝试硬门槛。真实分支逐步记录接触并以BDDL作最终判断。

# SafeTRAIL Stage A 当前检查点（2026-10-30）

## 2026-10-30 14:48 Asia/Shanghai grasp-branch duration reference

同候选在dev02从branch-start Snapshot到214步抓取records mtime间隔36分42秒；dev03 current physical branch from14:41:48到14:47:56约6分钟。预计进度检查按旧run用时节奏安排，但不假定新分支结果/时长相同。Progress文件顶部记录两个source timestamp。


## 2026-10-30 14:45:10 Asia/Shanghai branch elapsed-time correction

OAH/L0T0 dev03 Bridge PID141507总时长18m+包含screen与RRT；选中的grasp物理branch-start Snapshot实际在14:41:48 CST生成，到14:44:53约3m。该branch目前没有post-action records；不要把Bridge进程时长当成物理branch时长。


## 2026-10-30 14:31 Asia/Shanghai shared Stage A grasp-search decision

候选screen当前在24项中过10项，dev03已选中screen-pass项进入physical grasp. 后续0-pass任务可疑似被collision screen过度阻断；活动run结束后优先补一个“pass优先、零pass才有限physical探索screen碰撞候选”的Stage A fallback，完整保留MuJoCo contacts，并绝不赋予安全资格。成功判据保持native BDDL task_success on continuous physical trajectory.


## 2026-10-30 14:27 Asia/Shanghai live checkpoint

OAH/L0T0 dev03启动未报错，V3 manifest=`running`。合法t0 identity已在grasp分支snapshot manifest重新核实；Student-first 32步已保存并排除，PID 141379 grasp-search及PID 141507 Bridge执行task-specific screen。专家动作尚未写出。Stage A失败/成功判断仍以连续MuJoCo行动与原生BDDL终态为准。archive `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T0_reachability_biased_dev_03/v3_run`。

## 2026-10-30 14:29 Asia/Shanghai live checkpoint

OAH/L0T0 dev03 candidate screen完成：24 grasp candidates，10 task-approach可行、14目标表面碰撞/提前pad接触，live state保持不变、t0指纹精确匹配。executor继续RRT预检与候选选择；专家物理动作仍未出现，screen pass不作为最终成功。报告在progress文档顶部给出的run归档。

## 2026-10-30 14:30 Asia/Shanghai live checkpoint

OAH/L0T0 dev03由10个screen通过候选中选中`geom111_jaw0_approach2_-1_flip-1__rrtgenerated_geometry_high_approach`，已生成物理branch-start Snapshot。抓姿与dev02成功抓取的candidate一致，便于将本轮place-height算法影响隔离在后续放置阶段。MuJoCo post-action抓持记录还未出现，当前不记任务进展/成功。详情见progress顶部。

## 2026-10-30 14:20 Asia/Shanghai checkpoint

当前覆盖9/30。OAH/L0T0 dev02以native On=false终止于t=433；selected抓取、handoff、branch02来源433步连续、10,825/10,825子步完整，失败原因为24步区域路点停滞。诊断显示几何走廊清障把运输抬至1.3263m，EEF平移响应衰减并反向，目标抓持仍保持。详见进度文件terminal checkpoint和旧run `terminal_failure_audit.json`。

当前自主执行dev03 Standard合法t0冷启动（PID 140520），使用新`reachability_biased`路线：On任务先降低至目标上方10–16cm再做侧向运输；In任务沿用compiled-obstacle高度。此策略减少高位可达性风险，碰撞/BDDL均由真实物理结果裁决。新archive：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T0_reachability_biased_dev_03/v3_run`。dev02 V3 wrapper因对既存输出目录重复mkdir失败；dev03使用全新唯一目录。覆盖达到30/30前继续迭代。


## 2026-10-30 14:00 Asia/Shanghai live status

当前覆盖仍为9/30（7 Standard、2 Extended）。OAH/L0T0 dev02 Standard冷启动持续运行；executor 112228与Bridge 113961存活，wall预算剩余约89分钟。branch00/01为互斥失败路线；branch02至t=408/step_index=161，162 placement动作，active candidate共408动作/10,200预期子步。原生Region XY距离从187.6 mm降至51.5 mm，碗根仍高于目标约0.394 m，25子步监测及抓持确认完整，task_success=false。详情与审计边界见GS_OPD_PROGRESS.md顶部和当前目标文档。

只读诊断显示OAH/L0T1既有24项屏幕集中在单一book geom92的8个基姿及深度偏移（9 IK不可达、15碰撞）；存在未定位的恒定`unmapped_scene_environment`负间隙及部分hand-book重叠。原候选池不重跑；后续先形成通用候选覆盖/静态接触诊断改进，并按dev02终态重排任务。


当前覆盖9/30（7项Standard、2项Extended）；三条旧报告未明确的Snapshot边界已独立逐项验证，见GS_OPD_BASELINE_SNAPSHOT_AUDIT.json。 当前21项未解决任务的最近失败证据已分类：A候选筛查11、F区域进展4、B/D/H/I/J各1；分类见GS_OPD_FAILURE_CLASSIFICATION.md。L2T2 dev03启用多目标调度，但首个Pot1 native placement未完成：24个抓取候选4个可行，185步抓取/抬升（35.94 mm）和32步handoff后，三条互斥placement分支各447步；最终selected branch02在90-action目标到达预算内超时。终态pot1/pot2 On均false、炉灶Turnon=true、task_success=false。selected V3审计来源齐全（664动作/16,600子步），但无成功Episode连续性且未授予安全认证。证据与限制见`GS_OPD_PROGRESS.md`顶部及任务矩阵。

L2T3两次首步Safety Bridge EOFError暂存，L2T4既有Standard尝试已失败。依覆盖矩阵，`obstacle_avoidance_human/L0T0` dev02现已从其identity-checked合法t0开始Standard 1000步独立冷启动，使用当前bundle、14400秒wall-time上限，不载入目标成功示范或其他run物理候选；t0身份核验和24候选screen已完成，10候选可行、live state未变化；32步Student-first段（800子步）已逐行核实为完整监测且task_success=false，V3最终审计待完成。终态和selected Episode审计待完成。 dev02专家抓取候选已形成：214动作、stable bilateral连续22状态、碗抬升36.45 mm；handoff已完成32动作/800子步（t215–246）；branch00 direct已留下transport-failed Snapshot（186动作、最后t432）；branch01 keypose-side当前至t416（step_index=169，170动作，phase=transport_region_progress_region_progress），最新Snapshot重算目标XY距离31.5 mm。当前候选前缀214抓取+32 handoff+170 branch01动作=416动作；两个放置分支互斥且不拼接，V3最终selected stream待审计。branch起点integration state与handoff匹配。身份、预算、可恢复Snapshot与下一步见[机器检查点](GS_OPD_SAFE_TRAIL_CHECKPOINT.json)。L2T2的三个placement分支保持互斥，不拼接；后续在公共路点跟踪/停滞能力修复后回访。
## L1T2结果与下一覆盖项（2026-10-29）

L1T2 Standard 1000步仅到t=595，pot1 On=true、pot2 On=false；同一合法t0 Episode从已提交t595 snapshot延长至horizon 5000后，在t=1183满足所有BDDL目标并`task_success=true`。V3接受Stage7并完成完整审计：1183个连续7D动作、29,575个MuJoCo子步，逐动作子步监测完整，Snapshot边界已核验。故该配置计为Extended Demonstration Success，不计Standard Success。executor在任务完成后的grasp-only guard报出`environment_done_during_grasp_only_branch`，但最终native status为complete、V3 engine return code=0。安全认证、Strict-10mm与训练资格未通过/未授予；这是Stage A任务可行性工作，不作安全认证声明。

截至2026-10-29的历史检查点共有8/30项成功轨迹（6 Standard、2 Extended）；L1T1 dev06已将当前覆盖推进到9/30。Native Region Progress运输/放置修复及L1T1实证见本文件顶部与`GS_OPD_PROGRESS.md`。

`obstacle_avoidance/L1T0` 已完成Standard 1000步冷启动原生成功：合法t0初始 `On(akita_black_bowl_1,plate_1)=false`，t=402时原生On与 `task_success` 均为true，reward=1、done=true。V3审计确认成功Episode共402动作、10,050个MuJoCo子步，t=1–402连续；Student首分支精确回滚并丢弃，后续145步专家抓取、32步Student handoff与225步专家放置组成同一条成功历史，阶段Snapshot边界通过核验。285步几何监测为unsafe，Strict-10mm、安全认证与训练资格均未通过；稳定释放未评估。Standard成功覆盖为3/30；计入L0T2与L1T2两个扩展horizon成功后，共5/30配置有成功轨迹。

Task-Feasible候选筛查源码复核发现旧 `task_approach_only` 路径对BDDL禁触对象仍应用默认10mm余量。L0T3/T4的48个候选都因实际碰撞或IK失败被拒，没有仅因10mm余量而拒绝的候选，所以此问题不解释两轮失败。现已修正任务筛查仅基于实际碰撞/穿透和IK可达性；Strict-10mm与2mm边界的定义及安全认证逻辑保持不变。L1T0结果使用修正前指纹`d0ea082cf6d31e6c7b4e5f6a55a80a95a8efe8e3f8a157c634b3106e4ad12e2b`；后续运行使用源码bundle指纹`6b75e613b78f6662ad650d3a51d892bfcbbe6150120847aeea927ee788706371`。

`obstacle_avoidance/L1T1`的dev01/dev02历史筛查24项中15项实际碰撞、9项IK不可达；dev03/dev04随后诊断并精确处理固定mount/table安装接触，dev05进入真实抓放但受过高清障路点阻塞。dev06修正世界up清障高度后Standard 1000步于t=613成功；详细动作审计和安全边界见本文件顶部及矩阵。

`obstacle_avoidance/L1T2` Standard 1000未成功（t=595仅pot1 On）；同Episode的5000步扩展续接已在t=1183全部原生目标为true。详情与安全认证边界见本文件前段和[GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md)，机器状态见唯一矩阵[GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv)。L1T3 dev02已完成且筛查失败；下一配置为L1T4。

# L1T0抓持与抬升阶段检查点（2026-10-29，历史记录）

在这条阶段检查点中，24个task-approach候选有11个通过；V3经8次RRT预检后真实执行一个候选，稳定抓住黑碗并抬升36.864 mm。145/145动作记录、3,625/3,625子步连续完整。随后同一Episode继续完成On任务，最终成功与全程审计见本文顶部；L0T3/L0T4拒绝原因和10mm筛查边界修正也已单独说明。

# obstacle_avoidance/L1T0覆盖扫描启动记录（2026-10-29，历史）

L0T3/L0T4首轮均在候选筛查阶段失败后，当时转测 `obstacle_avoidance/L1T0`。合法t0由现有batch模块的reset/export薄流程新鲜生成，明确 `no_prior_experience=True`；之后以现有V3做Standard冷启动。矩阵记录t0身份和运行状态。

# L0T3/L0T4完成首轮扫描（2026-10-29）

两项均从identity-checked合法t0独立运行现有V3后，在抓姿/开口接近筛选阶段耗尽：L0T3 24候选0可行（21碰撞、3 IK）；L0T4 24候选0可行（24碰撞）。两轮RRT和专家物理动作均为0；被回滚的Student分支分别31/775与32/800，记录完整。两份报告的V3 `physical_action_audit` 标记incomplete，是因为没有专家动作源；这不代表存在未监测的成功动作。当时计划先对 `obstacle_avoidance/L1T0` 做覆盖扫描，再根据多任务证据修复共享候选能力。

# obstacle_avoidance_human/L0T4首轮最终结果（2026-10-29）

fresh t0冷启动完成后，首轮24抓取候选中1个可行。选定抓取+handoff+keypose-side放置失败路径616动作/15,400子步审计完整，但t=616时In、Close及`task_success`均false，放置支路因24动作waypoint stagnation终止；几何状态中420项标记unsafe，安全认证未授予。报告和分支证据以`GS_OPD_PROGRESS.md`的L0T4条目及任务矩阵为准；当前覆盖已转至`obstacle_avoidance_human/L1T0`。

# 历史执行状态快照（2026-10-29）

`obstacle_avoidance/L0T3` 的首次标准冷启动在抓姿可行性搜索阶段停止：24个候选、0个通过（21碰撞约束、3 IK不可达），无RRT路线尝试、无专家物理分支、无接受动作。Student 31动作分支精确回滚且完整审计。当时证据定位到抓姿/开口接近候选不足，不支持延长Horizon；当时下一项为 `obstacle_avoidance/L0T4`。

`libero_90/L0T6` attempt14终态为t=519、native `Open=false`、`task_success=false`，原因为 `articulated_fixture_live_follow_joint_progress_stalled`；该支线已关闭。三项基线仍为L0T0/L0T1 Standard、L0T2 Extended。逐项状态唯一见 [GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv)。

# 历史执行次序校正（2026-10-29）

- 已核实本地注册的30个避障任务配置；状态唯一来源为 [GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv)。
- `obstacle_avoidance/L0T3` 的标准1000步、init-state 0、seed 0冷启动当时正在运行；t0 Snapshot的BDDL、模型、payload和integration-state身份均已记录。等待完成后审计并更新矩阵。
- `libero_90/L0T6` attempt14已在t=519以原生 `Open=false` 结束，`articulated_fixture_live_follow_joint_progress_stalled`；保留为额外失败诊断并关闭后续轮次。
- 当时下一步为L0T3审计、`obstacle_avoidance/L0T4`冷启动，然后按覆盖优先顺序完成其余23项。既有L0T0/L0T1 Standard和L0T2 Extended成功基线保持不变。

# 当前总体路线：SafeTRAIL Stage A LIBERO-Safety 避障轨迹生成（2026-10-29）

## 2026-10-30局部路线状态

dev14之前的首轮扫描覆盖30项，原生成功覆盖8/30。dev_11放置RRT在20秒内仅6次迭代/373次检查后超时；dev_12启动时因端口占用无物理动作。dev_13生成193点全几何放置路线（63.325秒、10迭代、1,185检查）并执行355个放置动作，但最终直线插入时mug持续接触微波结构，24动作位置/姿态进展低于停滞门槛；t=607时In/Close均false。总尝试审计607动作/15,175完整子步，安全监测记录412个unsafe动作，不计成功覆盖。dev14插入RRT和后续只读预检未找到可执行目标。之后L1T1 dev06以世界竖直清障修正完成Standard成功，当前覆盖9/30；L1T3 dev02仍在抓取筛查失败，下一配置为`obstacle_avoidance/L1T4`。详见`GS_OPD_PROGRESS.md`和`GS_OPD_CURRENT_GOAL.md`。

目标是在合法任务初始状态上自动搜索、真实执行并完成 LIBERO-Safety 原生 BDDL 任务，为 SafeTRAIL 安全避障轨迹生成建立物理可行数据。主实验范围是 `obstacle_avoidance` 和 `obstacle_avoidance_human` 两个 Suite，共30个 L0/L1/L2 × T0–T4 配置。每次成功要求完整连续的物理动作记录和原生 `task_success=true`；标准1000步与 demonstration-only 扩展分别统计。当前只推进 Stage A，不开展Near–Far、安全轨迹重构、Strict-10mm、LeRobot转换或Student训练。

1. **L0T1 标准冷启动成功已达成**：标准 1000 步内 t=752 完成 `In(black_book_1, desk_caddy_1_back_contain_region)`；752 个成功路径动作及 18,800 个已监测子步可追溯。严格安全未通过。
2. **L0T0 公共逻辑回归已通过**：第二轮合法 t0 `--cold-start` 在标准 horizon=1000 的 t=499 完成 `On(akita_black_bowl_1, plate_1)`；499 个成功路径动作和 12,475 个子步可追溯。修复将容器紧容差限制为 `In`，`On` 保持基准容差；RRT 预算用 8×20 秒，未回放既有 505 步答案。严格安全未通过。
3. **L0T2 多目标原生成功已达成（demonstration-only）**：Pot1于t=776完成release/withdraw/settle；Stage 3/4抓姿窗口均筛查失败且回退排除。Stage 5从同一Snapshot真实抓起Pot2并于t=1390达成三项原生目标全true、reward=1、done=true。经审计修正后，成功Episode时间轴1390动作/34750子步完整连续，阶段Snapshot边界核验通过。标准1000步成功=false；最终稳定释放未验证。Strict-10mm、安全认证与训练资格均false。归档 `l0t2_native_multigoal_cold_start_04_h5000/v3_run`。
4. **L0T39 原生 Turnoff 已达成**：第26轮由合法t0冷启动，t=356、356个连续动作和8,900个完整监测MuJoCo子步，原生 `Turnoff(flat_stove_1)=true`、reward=1、done=true、`task_success=true`；独立审计通过。未证明双指同时接触，安全认证仍false。
5. **30项任务注册及矩阵已核实**：本地运行时 `get_benchmark()` 对 `obstacle_avoidance`、`obstacle_avoidance_human` 均注册15项，L0/L1/L2每级5项；30个BDDL和`.pruned_init`均存在。矩阵按用户指定优先级推进：先审计在途 `obstacle_avoidance/L0T3`，再运行 `L0T4`，接着同Suite L1/L2，最后覆盖`obstacle_avoidance_human`。每个配置记录合法t0及哈希、标准与扩展结果、动作数、候选搜索次数、失败原因、代码指纹和归档路径。完整矩阵见 [GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv)，注册核验见 [GS_OPD_SAFE_TRAIL_TASK_MATRIX_VERIFICATION.json](GS_OPD_SAFE_TRAIL_TASK_MATRIX_VERIFICATION.json)。
6. **固定协议评测与额外成果管理**：覆盖积累后冻结代码指纹和统一候选/RRT/物理分支预算，开展跨场景评测；分开报告开发中定向调试配置与未经针对性调试配置。L0T0/L0T1标准成功和L0T2 demonstration-only成功作为既有基线；L0T39关灶成功作为矩阵外额外结果保留；L0T6抽屉失败诊断与姿态改进保留；attempt14已归档并关闭后续轮次，不计入30项分母。

L0T1 最近修复的 `native_region_entry_alignment` 和L0T0紧容差作用域回归保持为历史通用能力与成功证据。现在按30项矩阵从 L0T3/L0T4 开始，通过失败归因补足通用几何搜索、物理执行或多阶段调度能力；已成功或失败结果都保留可追溯记录，不把少数任务推广成完整benchmark泛化。
---

# 当前总体路线：无示范任务搜索与安全重构（2026-10-28）

1. 从合法 t0 以无目标成功示范的协议运行基础 π0.5 和通用物理任务搜索，获得真实完整原生 Task Success；区分 RLinf LIBERO SFT 项目基座与未经 LIBERO SFT 的 `pi05_base`，并审计全程记录。
2. 对自主成功轨迹逐物理子步定位首次 Strict 风险与合法恢复状态，记录安全规则来源和完整任务里程碑。
3. 从合法 Snapshot 以当前几何生成局部安全替代动作，真实执行并在新状态重新规划剩余任务；持续修复后续风险直至任务成功。
4. 用现有验收与数据管线验证完整 Strict-Safe Episode，再以同一入口在此前未提供成功答案的不同任务或初态验证完整流程。
5. 扩展任务谓词与操作算子覆盖，按实际任务统计失败原因和成功率，不把有限场景结果推广到整个 benchmark。

当前第二轮无先验冷启动已取得 505 动作真实原生任务成功，分支边界、12,625 子步及 reward/done 已独立核验；安全重构、严格安全、数据验收和独立跨场景结果仍未证明。旧 attempt41 含同任务成功经验，只能作为对照。30 个 `obstacle_avoidance`/`obstacle_avoidance_human` 任务定义的覆盖矩阵、未支持的依赖及实验路径见 [无示范实验报告](GS_OPD_ZERO_SHOT_BOOTSTRAP.md)。以下为历史路线，不代表当前验收状态。

安全阶段首批 24 个几何候选均被严格抓姿筛查拒绝；下一步需要依据非授权接触及指垫夹持几何改进局部候选，或利用独立任务测试探索另一操作机制，不将单场景失败推广为通用不可行。当前还需获取并绑定真实 Teacher 规则，自动定位不能将缺失监测视为安全。

---

## 2026-10-28 11:23 Asia/Shanghai：切换到成功源运行时身份核验

L0T0 attempt41连续离散抓姿、单轴接触偏移及耦合XYZ小范围screen均未获得Strict候选。最新screen为16项、0可行（collision 7 / geometry 9），0 RRT、0动作。845步历史轨迹本身仍是有效原生Task Success源，但其报告只覆盖749步OSC运输段；源码重建模型SHA与保存几何SHA不同，且状态ID关联未证明。因此优先恢复来源导出运行时并核对模型/状态身份，成功后才进入抓姿提取与严格screen；不再扩展任意微扰搜索。若来源不可恢复，转向另一个独立L0T0成功轨迹或L0T2可完整验收的任务阶段。executor的identity-checked Snapshot fallback与OSC joint-limit recovery路径已有单测覆盖，物理/RRT候选预算有限（attempt41各1）；该次成功没有触发恢复，真实恢复执行仍未验证。当前Level B，Level A仍是未完成目标。

# 2026-10-28 10:17 总体路线状态：Level B由V3 attempt41复现；Level A严格抓姿离线筛选

## 2026-10-28 10:47：完成接触偏移筛查，切换到耦合构造

L0T0成功物理抓姿周围4个local-y偏移全部在approach/闭合几何筛查失败；L0T0 48个基姿与12个角度变体也无Strict解。L0T2的55 mm抓姿候选存在IK/碰撞阻塞，证据不足以启动预开夹执行。没有候选时不启动RRT或物理分支。下一阶段的可行路线是改进候选生成，让夹爪接触点、jaw orientation和approach jointly satisfy实测碰撞几何及非pad 10 mm要求；成功抓姿、失败归档和Strict阈值均作为输入约束。当前Level B，Level A未完成。


## 2026-10-28 10:43后续机械假设

候选宽度校准和attempt41物理稳定夹持间距差约4 μm；75项聚焦测试和4个subtests通过。一个L0T2几何候选需要55 mm aperture而合法t0当前pad gap为39.699 mm。若继续研究，应先实现并验证明确的最大开口动作，然后在该姿态下重做完整approach筛查；在这两项完成前不能做RRT或物理夹持。


## 2026-10-28 10:40验证状态

74项聚焦回归及4个subtests通过；最终校准screen各与合法t0 Snapshot指纹匹配。attempt41 posthoc物理动作审计覆盖抓姿与放置两个分支、669条记录、16,725子步。剩余Level A阻塞是严格抓姿候选为0、stable release未完成、没有安全Observation-Action训练trace。


## 2026-10-28 10:39校准：物理pad gap与关节间距差值

确认并修正 Panda screen qpos映射的1 mm模型偏移：活动开爪关节差40.699 mm，MuJoCo指垫Geom净距39.699 mm。候选物理宽度先加上活动场景实测`joint separation - pad gap`，再换算为双指对称qpos。缺少宽度的历史模板沿用已验证闭合姿态。

校准后完整筛选L0T0基础48项、L0T2基础16项，仍0 Strict可行。报告确认初始Snapshot未变、无RRT/动作。12项相关测试通过。此轮Level B仍依attempt41原生任务成功，L0T0和L0T2均未取得Strict抓姿。后续不再重跑同一宽度候选；如出现新的编译碰撞面或不同抓持接触解，再开启有界候选。


## 2026-10-28 10:35路线校准：夹爪宽度按候选复核

### 完成

- 修复`mujoco_jointspace_planning.py::screen_grasp_candidates`对不同几何候选共用一个历史夹爪闭合姿态的问题。
- 新增candidate width→Panda双指qpos映射、开口与限位校验、候选报告字段及回退路径；12项聚焦screen测试通过。
- 修复后复筛L0T0 48个基姿与L0T2 state0 16个基姿，均为0 Strict可行，0动作/0 RRT，live状态未变。报告在机器progress JSON中登记。

### 当前判定

旧的几何候选池结果在夹爪闭合阶段不再作为最终证据；同一候选池的修正复筛是有效结果。L0T0所有基础几何抓姿仍因完整机器人/目标几何、夹持接触或IK受阻。L0T2同样无基础Strict可行抓姿。attempt41的真实原生任务成功保持Level B，不升级为安全数据。

### 后续

不重复原宽度候选搜索，不启动不满足Strict筛选的物理候选。剩余工程时间用于归档、复现命令、运行证据与报告字段的独立复核；之后需要新的编译接触面/夹爪净距机制证据，才能设计与既有池不同的候选。安全阈值、BDDL、合法初态和Snapshot均保持原样。

V3 `experience_task_success`从合法L0T0/state0 t0 Snapshot完成抓取与原生放置，t669得到native task success，1000步时域成功。抓姿搜索167条+放置502条记录经后验聚合审计为669条动作、16,725个完整25子步监测；状态/动作字段、分支动作计数和子步计数均匹配。修复后的审计不修改原始V3报告，独立报告SHA `758dfaebd6deea1b5b958052e242d21d6524f65a461ed6b347eed10525b62e88`。

此结果保持Level B：669条诊断几何监测状态均为unsafe，Operational Safety未认证；stable release未验证，Strict-10mm为false/未认证，`safe_training_eligible=false`，该backend未导出完整Observation-Action训练轨迹。原始V3报告曾漏算放置记录且抓姿分支动作字段读取错误；相关实现已修复并增加回归测试。原始归档保持不变。

**后续顺序：**先以全新编译碰撞几何或不同抓姿候选族产生严格离线可行证据；精确核对L0T2任务/Snapshot/模型/BDDL身份；然后才允许一次有界物理验证。同步完成冻结Safety规则绑定和V3 Observation-Action导出审计。当前L0T2累计122项筛选均无Strict候选，旧变体族已停止；不启动重复物理分支，不更改BDDL或10 mm/2 mm阈值。Level A仍是开放目标，Level B为当前最高可证等级。

L0T0 attempt41成功抓姿在严格scratch screen中被拒绝：预抓取通过，双侧pad距离均在2 mm内，但非pad手指—黑碗穿透约0.153 mm、末端掌部接触；接近轴滚转±5/10°仍碰撞，±15°越过pad穿透许可。排除已测姿态后，活动生成池前16个不同基姿（15个新姿+该物理成功姿态）0可行；失败类collision 7、grasp geometry 8；全程动作0、live state不变。下一批将筛剩余32个不同基姿；如仍为0则停止此生成族并转向新的目标碰撞面/抓取候选方法。

随后完成完整48/48基姿screen，0可行（collision 23、grasp geometry 24、IK 1）；同一成功抓姿的6个approach roll和6个jaw-axis tilt也全部失败。显式candidate-ID筛选曾因经验模板截断后8项生成失败；当前代码修复为按请求ID从完整几何池挑选，2项回归通过。全程60项几何姿态screen没有物理动作、RRT或live-state变化。眼下只针对最近目标碰撞对测试4个local-y表面接触偏移；该批失败即停止本局部姿态族，转向新抓姿构造方法。

## 2026-10-28 09:30以前的路线记录

attempt27仍是本轮最高真实物理等级：669步L0T0原生任务成功，但没有稳定释放或Strict Safety认证。Pose对齐修复后，attempt33–38验证48个基姿、72个pose扰动及2个真实历史Episode抓姿候选，严格可行数为0；所有抓姿screen均零物理动作。

L0T2的1093步示范、1460步Operational-Safety重构数据与Keypose经验已完成身份核验。全部48个唯一基础候选、48个有限接触高度/jaw变体和2个哈希核验历史姿态共104次screen均无Strict可行项；额外12个接近轴滚转也全部失败。最近碰撞对已细化为非pad手指几何与壶体g11的干涉；±10°滚转穿入约1.8 mm，±20/30°更差。阈值、BDDL与物理动作仍为0。基于编译几何g13中心相对g12外移17.17 mm，下一步只筛6个转向相邻把手段的中心候选；如仍无Strict候选，停止L0T2抓姿搜索。V3逐字段核对Snapshot manifest身份、模型/BDDL和精确Snapshot SHA；并在engine启动前验证resume Snapshot envelope与t0 manifest哈希。错误backend/字段类型或Snapshot身份会持久化为零动作拒绝，正向可行候选会锁定并转发给executor。聚焦V3测试31 passed。1460来源Strict-10mm未通过，t329 Snapshot有约0.527 mm非pad finger穿透，未启动L0T2物理分支。

L1T2 run53区域局部释放可按当前Region重定位，但源撤离有22个Pot-hand接触子步，且阶段与L0T2 t329运输状态不符，不能继承Strict资格。t329 Runtime Snapshot payload SHA `34f213c63d87be1a26cd69a3e4d442f9df9f2dcf4d406230a32ecea1dcd00673`的task/model/BDDL/state身份已验证；配对t0 screen时因精确payload SHA不匹配而持久拒绝，report SHA `06ed14d701a40f6b3bf8d8518087738d86e5c72bfbfa142955bef4911df0f078`，engine/Bridge/actions均0。attempt40的g13接触筛选也失败，最近非pad间隙仅0.446–0.460 mm。当前按Level B回退，由V3从L0T0/state0合法t0运行一个有界物理分支，复核已验证黑碗抓姿及修复后的目标支撑映射。该执行不使用Strict screen gate；安全认证false、Strict-10mm与训练资格分开记录，BDDL和阈值不变。


## 2026-10-28：attempt27首个原生任务成功；修复终止后撤离动作

attempt27从attempt21的t167合法抬升持抓Snapshot精确恢复（Snapshot SHA `ba43889b…e943e088`，integration-state SHA `07b2def7…90a1edf`）。原生谓词`On(akita_black_bowl_1, plate_1)`首次在t669成立，reward=1、done=true；此前167步抓取前缀与502步运输/放置后缀形成669步连续Episode，满足1000步时域。释放条件已被控制器观测并进入`withdrawal`，但settle和撤离没有完成，故`stable_release_verified=false`。

终态审计发现源后缀还记录了t670的一个withdrawal动作，而环境在t669已终止。该动作从成功Episode中剔除；新review NPZ仅含截至首次done的669组7D动作与状态，不含双相机Observation，因此不是训练包。原始报告、Snapshot和帧归档未改写。归档：`/opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt27_obstacle_overflight/task_success_first_executor_obstacle_avoidance_L0T0_20261028_065344_974183682/expert_native_goal_stage_control/placement_branch_00_region_layout_12_direct_region_progress/review_episode_audit.json`，NPZ SHA-256 `8887dc6c…b1b6df48`。

有效669动作的16,725个物理子步记录完整，但几何状态仍有unsafe样本；`operational_safety=not_certified`、Strict-10mm=false、`safe_training_eligible=false`。attempt27越障路线避免了手部与香水瓶接触，但不能据此称安全成功。结果为验收Level B。

执行器现于native `done=true`后停止release/withdrawal/settle动作，并报告终态跳过撤离；新增终止边界回归测试。聚焦测试43 passed、6 subtests，py_compile和`git diff --check`通过。下一步先离线筛选不同的抓姿，要求非指垫—碗几何至少10 mm且指垫穿透不超过2 mm，并按活动BDDL约束预检路线；没有筛选和预检通过的候选时不启动物理分支，也不复跑同一路线。

## 2026-10-28：attempt26越障路线在零动作门禁失败，修复上下文接线

attempt26从attempt21 t167合法持抓Snapshot恢复；在物理动作前因`_native_place_single`未收到`target_context`而报告`NameError`。报告SHA `e9f92bab7cf4286d7defb69b12b8f070a58648fc10caac25ef4aad42eac3c7b7`，动作/子步均0、timestep保持167，原快照未改变。修复方式是在函数候选摆放调用显式传入已通过活动BDDL、model SHA核验的目标碰撞上下文；release续接路径不需要该上下文。聚焦验证42 passed、6 subtests；py_compile和diff检查通过。此为执行接线修复，attempt27按同一路线单分支重跑。

## 2026-10-28：attempt25证实香水瓶碰撞；attempt26转为编译几何越障路线

attempt25从attempt24的精确t353终态恢复，执行24个动作、600个MuJoCo子步后在t377停滞。每步25/25子步记录完整，双侧抓持保留，native Task Success仍为false。实际接触记录显示手部Geom与`bottle_of_perfume_1_g37/g45/g85`碰撞；最大接触力约12.45 N，几何最小clearance -0.563 mm。执行器报告SHA `f046d68f1660bb1a0fee89b28668e6683129ae0968b18877cefe51e6c1743bfa`，失败Snapshot SHA `5f2ed18f62800d6f06adcb9c5e03a0694aef589b33e5635955813723b6a052b3`。因此直接路线与两个水平侧绕方向不能视为已避开瓶子。

正式Bridge顺序（启动、显式reset、恢复快照）后，attempt25终态精确恢复成功，模型SHA `085817d128dd03511cfd14cd810a6c035e85258e0b8260881f3f0e0f3db2e840`，9/9活动对象与固定物碰撞包络齐全。根据这些活动几何，attempt21 t167合法持抓抬升快照的直达走廊中有6个包络相交，最高香水瓶顶为1.21847 m。目标碗根越障高度按当前碰撞Geom最低点加0.05 m间隙算为1.26697 m（plate区域局部z=0.34804 m），EEF预计约1.29881 m。

attempt26规划从attempt21精确抬升快照启动（SHA `ba43889b75039390c81dbda834b80a4ba51db8039af2dff57059b5e4e943e088`，t167），仅执行一个真实物理分支；原生目标路线先抬高越障再直达plate中心。此调整是基于碰撞包络的路线规划估算，实际碰撞、预算、Task Success、安全结果以新的物理记录为准；没有安全认证。

## 2026-10-28：attempt24 正向侧绕行在首个中间点停滞，attempt25镜像绕行

attempt24 从 attempt22 的timestep329终态快照恢复，以 `keypose_side_region_progress` 执行24个真实OSC动作、600个MuJoCo子步；每个动作的监测完整，双侧抓持未丢失。第一侧向中间点在24步内进展不足，原生目标仍为false。终态快照已保存，当前timestep353。

已新增镜像方向 `keypose_side_region_progress_reverse`，并用测试确认它跨越到直达线的另一侧。attempt25 从 attempt24 精确终态恢复，固定plate中心候选并仅运行一个反向侧绕行物理分支。仍保持原任务时域、BDDL、成功阈值和安全标注。聚焦pytest 41 passed、6 subtests；`py_compile`、`git diff --check`通过。

## 2026-10-28：attempt23 RRT预检无有界IK解，attempt24切换侧向绕行

attempt23 从 attempt22 的精确终态快照启动，task timestep保持329。一次12秒全几何放置RRT预检返回 `task_goal_has_no_bounded_current_scene_inverse_kinematics_solution`，因此没有执行物理动作，快照保持不变。

attempt24 从同一快照恢复，强制使用 `keypose_side_region_progress` 原生区域绕行，通过一个真实OSC物理分支绕开 attempt22 停滞的直达走廊。中心摆放目标、对象碰撞包络和BDDL谓词保持不变；预计进展与终态仍待实测。当前聚焦pytest为41 passed、6 subtests；代码编译和diff检查通过。

## 2026-10-28：attempt22 支撑面映射通过，直接OSC运输在走廊停滞

attempt22 精确恢复了 attempt21 的37.1 mm抬升快照，采用由10个编译碰撞Geom推导的plate_1支撑包络；居中候选位于支撑面局部[0,0]。单一路线实际执行162个动作、4050个MuJoCo子步，子步记录完整。双侧抓持持续确认；碗到目标中心的水平距离由49.1 cm降至31.6 cm，随后连续24步进展不足而安全停止。原生BDDL谓词仍为false，终态快照SHA-256 `c8c0e8cb575086c419a2b406abfd70027ce6da4bc03e0b9a85cc014736d3eca8`。direct OSC路线预算在该起点预估1278步，而当时预留的传输预算为497步，说明该路线在标准1000步任务时域内不适用。

已加入可选 `--native-region-route-variant`，可从中途精确快照恢复并强制使用侧向绕行。attempt23先从 attempt22 终态做一次12秒全几何RRT放置预检，最多执行一个物理分支；若无路线，可无损回到该终态快照尝试侧向绕行。当前聚焦pytest为41 passed、6 subtests；`py_compile`与`git diff --check`通过。安全认证、Strict10mm及任务成功仍未达到。

## 2026-10-28：attempt21 首次稳定抓取抬升，修复 plate_1 对象支撑区域映射

attempt21 使用候选 `geom113_jaw0_approach2_+1_flip+1__rrt0` 完成167个控制动作、4175个MuJoCo子步；子步记录完整，几何监测167步均记录为unsafe。抓取稳定性连续状态28/28通过，碗抬升37.1 mm，达到内部30 mm抓取抬升门槛。原生BDDL任务仍未成功：executor在摆放开始前报 `native_region_candidate_geometry_binding_unavailable`。审计确认 `plate_1` 是活动BDDL中的物体目标，不是 `object_sites_dict` site；site-only 查询错误地拒绝了合法目标。

已将 Bridge 的 native goal 查询扩展为：对site目标保持编译site读取；对对象目标从绑定的编译碰撞几何构造实时支撑包络。plate_1 当前包络由10个碰撞Geom测得，候选器已调整为在对象支撑面上优先居中，最终成功仍只由原生BDDL谓词确认。未改BDDL、成功阈值、场景对象或Safety Reconstruction。只读Bridge目标查询通过；候选中心核验通过；`py_compile`、`git diff --check`通过；聚焦pytest为40 passed、6 subtests。

attempt22 正从 attempt21 的精确抬升 Runtime Snapshot 继续，使用最多1条物理摆放分支和1次RRT预检。native task success、Strict10mm及安全训练资格仍未通过/未认证。

## 2026-10-28：attempt20四个新姿态RRT均超时，attempt21测试最后几何候选

attempt20从干净seed0/state0初态对候选索引10–13执行4次有界RRT预检：`geom112_jaw1_approach2_+1_flip+1`、`geom112_jaw0_approach2_+1_flip-1`、`geom111_jaw0_approach2_-1_flip+1`和`geom111_jaw1_approach2_-1_flip-1`均在约12秒达到规划预算且无路线。没有物理分支、动作或物理子步。V3报告SHA-256 `daa6179edcebacdac7e31553a2c0799c33d70e0beb19a6cdda39bc15410fd33a`，搜索报告SHA-256 `f688731a012678aefdf044e30688bc96fec5ffe046076690262c6000f9b4f401`。

attempt21仅剩未测的`geom113_jaw0_approach2_+1_flip+1`；编译Geom对应夹持轴宽约2.56 mm，路线和接触可行性未知。将使用同一哈希核验的t0 Snapshot，最多1个RRT预检和1个物理分支；Failure memory输入attempt8–14、17–20，排除先前失败的精确候选—路线对。当前代码聚焦验证70项测试+6 subtests、`py_compile`与`git diff --check`通过。attempt21准备启动。

## 2026-10-28：attempt19 geom113预检超时，attempt20合并四个有界预检

attempt19从哈希核验的seed0/state0初态筛选`geom113_jaw1_approach2_+1_flip+1`，其RRT预检在12.00秒时预算耗尽，没有生成路线，也没有物理分支或动作。V3报告SHA-256 `d4434ce41f83c554d0b81b0fc4950696170ef6ddc3e5c9fa2901eaf9aab35818`，搜索报告SHA-256 `e8cc709b83470dc5f696be8fa7ff963fb5c80062c1503171ce05efcc6ddb6d09`。归档保留，计入零物理动作预检。

attempt20仍从同一干净初态出发，使用`candidate_start_index=10`、最多4个12秒RRT预检，对候选10–13作一次批量筛选，最多仅对一个有路线的候选执行物理验证。Failure memory输入attempt8–14和17–19有效报告，跳过先前失败的精确候选—路线配对；attempt15/16元数据不完整零动作报告不作为failure memory。当前动态目标跟踪、候选位姿对齐门及其回归覆盖已通过70项测试+6 subtests、`py_compile`与`git diff --check`。attempt20准备启动。

## 2026-10-28：attempt18动态跟踪仍未到抓姿；attempt19换目标几何和干净初态

attempt18执行17步RRT预抓取与60步动态对象相对接近，共77动作/1,925物理子步；77/77动作25/25样本完整，几何监测77步均`unsafe`。动态跟踪把姿态误差从约17°降至0.2°，位置误差降至6.9 mm，但两侧指垫从未接触目标，且碗在接近阶段沿桌面滑移约53.7 mm，60步预算耗尽后停止，没有闭爪、稳定抓持、抬升或Task Success。该结果不支持继续延长这条同一目标路线。

新的位姿/接触记录显示：attempt11源Snapshot一开始存在夹爪—碗碰撞，恢复后首个预抓取动作就让接触消失；后续碗与炉灶燃烧器保持接触并移动。因此attempt19不再从这个扰动态续接，而从经哈希验证的seed0/state0原始Snapshot出发，换到另一个活动碰撞几何候选`geom113_jaw1_approach2_+1_flip+1`。最多一个RRT预检和一个物理分支；动态对象相对跟踪仍用于真实表面接近，抓姿误差仍须≤4 mm/0.045 rad才可闭爪。Failure memory导入attempt8–14、17–18。70项聚焦测试+6 subtests、`py_compile`和`git diff --check`通过，attempt19准备启动。

## 2026-10-28：attempt17固定世界坐标接近失配，attempt18改为实时对象相对跟踪

attempt17从attempt11经身份核验的末态Snapshot执行17步RRT/OSC预抓取、30步表面接近和18步闭爪，共65动作、1,625子步；65/65动作的25/25物理子步记录完整，65步几何监测均为`unsafe`。接近使目标碗移动并转动；现有实现把规划时的世界系抓姿固定下来，闭爪时EEF相对当前碗仍差16.6 mm、16.9°，并且没有左右指接触。无稳定抓持、抬升或原生Task Success，Safety未认证，Strict 10 mm=false，训练资格=false。失败Snapshot SHA-256 `e726dc1fde50048f8212c656ec2afacb34084e736f3453fef51300a526bafc73`；V3/executor/search报告哈希分别为`0ebe84a41336c69a98ce29271318345f549898825863281b634c8a2bab6c56ff`、`fd731cfe25194824ab2ed2a0f084608ac5542ddd7cd9076c07a015fef0311f4c`、`9aace512f2dd536692c23a37629759307627255c6ccdbe45d7f0c662dc1ab480`。

attempt18在同一经验证的接触快照上运行同一几何候选，但表面接近的每个OSC控制周期都会读取当前目标根位姿，并重算原有生成候选的`object_to_eef`目标。偏离候选时不作原位闭爪；闭爪必须先满足4 mm/0.045 rad跟踪容差。Failure memory导入attempt8–14和17，动态对象相对跟踪在route ID中单独标识；仅1个预检和1个物理分支。70项聚焦测试+6 subtests、`py_compile`和`git diff --check`通过，attempt18准备启动。

## 2026-10-28：attempt16无物理动作的输入校验失败，attempt17重跑

attempt16在Failure memory输入核验阶段拒绝了attempt15搜索报告：该报告源于前一轮矩阵索引错误，没有`native_goal_predicate`元数据。V3报告SHA-256 `335183fbf52bdf27fe951a8c309ccb6d67e6a4b8729f6a53a4e866a5ea003831`；动作审计确认物理分支0、动作0、子步0。该归档保留，不纳入后续failure memory。

attempt17重用attempt11身份核验的双接触Snapshot与精确候选；failure memory仅导入attempt8–14通过身份检查的报告。候选抓姿对齐判据为4 mm和0.045 rad；当前位姿不满足时按成功的RRT预检继续原生OSC路径与候选接近，最多执行1个物理分支。矩阵形状修正后69项聚焦测试+6 subtests通过，`py_compile`与`git diff --check`通过。attempt17准备启动。

## 2026-10-28：attempt15零动作形状故障已修复，attempt16继续候选接近

attempt15在当前接触Snapshot上完成了一个RRT预检，结果`offline_feasible`；随后姿态对齐检查把目标4×4矩阵误当成7D `[position, quaternion]`数组索引，导致预检/runner错误，没有真实物理分支、动作或子步。该0动作归档保留在`l0t0_state0_attempt15_contact_pose_alignment/`，V3/executor/search报告SHA-256分别为`e54325df7c1f69ca859f392c57d10733f5a368ad920a2fd5d3d6daa172c87fb3`、`a7f832b99b02f862887711e83f7441a9c29a14d1ed0c65d7732d40e5ea7b4a4a`、`eb53cd072a1dd684b8c2cb8d8fc66f2e88595026b30e4464898e5ab8a11c4b2e`。

矩阵输入现直接传入SE(3)对齐检查；使用同一attempt11接触Snapshot，闭爪仍须满足候选目标误差≤4 mm、≤0.045 rad，否则按规划结果继续对象相对接近。聚焦测试69项+6 subtests、`py_compile`和`git diff --check`通过。attempt16最多1次预检和1个真实分支，Failure memory包含attempt8–15报告，准备启动。

## 2026-10-28：attempt14验证闭爪错位；attempt15加候选抓姿对齐条件

attempt14从attempt11的精确Snapshot开始，两个中性等待动作期间双侧接触保留；随后首个闭爪动作（25%）即失去左指接触，50%时两侧均失去。20动作/500物理子步，20/20动作子步监测完整，几何监测20步均为`unsafe`，无稳定抓持或抬升；原生Task Success=false，Safety未认证，Strict 10 mm=false，训练资格=false。

离线由同一失败Snapshot和候选`object_to_eef`变换重算，发现EEF与候选抓姿相差103.2 mm、106.9°。旧接触探测只用宽松的位置距离≤约0.28 m，因而把“碰到目标物”误判成“已处在这个候选抓姿”，然后在错误位姿闭爪。这解释了attempt12和14的相同滑脱，也淘汰了继续增加等待时间的计划。attempt14失败Snapshot SHA-256 `bb5df0bb5f16b00f96494fdfaa8a52cbdbd7b9629f2124cd7fac1efb875090c1`；V3/executor/search报告哈希依次为`df4ac7fba7285f2558760f4fbd1bcdac2e49d53a5aa988c4d9d715f8f3a75963`、`f60339bc003baaf0a156f5d8622a8cd4c65bf3e1e768b551e32fb67f19cf351e`、`427abd34c5568762185a182b96d960e3869d3509657f56a644abcd42813ca29b`。

attempt15将原位闭爪条件收紧到已有OSC跟踪容差：位置误差≤4 mm且姿态误差≤0.045 rad。其余带目标接触但偏离候选抓姿的状态继续候选当前场景路线和OSC接近，不原位闭爪。该判据只是技能闭合准备度，不是安全门槛；BDDL和Safety阈值不变。failure memory导入attempt8–14报告。聚焦测试69项+6 subtests、`py_compile`与`git diff --check`通过，attempt15准备启动。

## 2026-10-28：attempt13等待过久丢失接触，attempt14收紧等待步数

attempt13在attempt11精确接触Snapshot上只执行了3个原生中性动作，75个物理子步均有完整记录。前两步后左右目标指接触仍在，碗线速度约7.4 mm/s、角速度约0.09 rad/s；第三步后左侧接触丢失，闭爪阶段没有启动。几何监测3步均为`unsafe`，Task Success、稳定抓持、抬升和安全训练资格均false。失败态Snapshot SHA-256 `78f06cc6475bfafe4174436682f0a4aa22fe8b43818647432a671d7969970cc4`；V3/executor/search报告哈希依次为`092077d1cd104248758106a65f996b3ec14bfd5a71b7ddb6753a69992a561460`、`850e264e8fad9aa5bc75f9845c3479ecf0823630e73dd006a11a0ae35434e1c5`、`36d43a19e7c5cd04181b779d74b05305cc27767a2a187923a24d62ba85449883`。

attempt14将中性等待上限缩至2步，接触仍在时立即闭爪；更新路线签名为`live_target_contact_settle2_close_probe`，并导入attempt8–13报告，让Failure memory跳过先前精确失败的接近/立即闭爪/长等待组合。只运行同一精确几何抓姿、一个预检和一个物理分支。聚焦测试68项+6 subtests、`py_compile`与`git diff --check`通过。attempt14准备启动。

## 当前阶段更新（2026-10-28）：L0T0抓取末端闭环

几何候选`geom111_jaw1_approach2_-1_flip+1`已通过真实预抓取/接近建立双指接触，但第一次闭爪探测显示碗仍在运动并于75%闭合时失去左侧接触；完整闭合后两侧均无接触，因而不能据此抬升或判断任务成功。下一步是同一失败Snapshot上的有界接触稳定等待，再验证闭爪和抬升；attempt13限1个物理分支。该分支完成后按原生BDDL谓词判断任务成功，安全认证与训练资格仍分别审核。聚焦套件68项+6 subtests通过，禁止因这次等待改动而放宽安全规则。

# 当前路线阶段：V3经验先验轨迹生成器与L0T0闭环修正（2026-10-28）

研究当前只推进`obstacle_avoidance/L0T0/state0`，验收仍是新鲜真实MuJoCo回合完成原生Task Success。参考为同任务845步成功Episode，动作不回放。attempt2/5/6在不同B-Spline路线约OSC第240步重现0.20 rad抓持旋转门限；attempt3实时姿态跟随和attempt4右绕分别因3.012 rad物体旋转和碗—木柜基座接触失败。attempt7加入等待后仍触发同一门限。attempt8经验抓姿接近分支执行194动作后耗尽预算；attempt9两方向共386动作，attempt10_retry1三方向共719动作，均未形成稳定抓持。attempt11从活动黑碗碰撞几何候选中，以90步预抓取+60步接近到达左右指垫接触（末态4个target contact），但夹爪仍张开、Task Success=false。该末态Snapshot身份完整且可恢复。V3现支持identity-checked partial-state resume、精确抓姿候选过滤，以及双指已有接触时的原位闭爪探测；68项聚焦测试+6 subtests通过。attempt12从attempt11 Snapshot精确恢复，只对该抓姿执行闭爪/稳定性/抬升，最多1次RRT和1个物理分支，导入attempt8–11 failure memory以避免旧接近路线重跑。安全判定、阈值、BDDL、物体与场景保持不变。详细证据见[V3说明](GS_OPD_TRAJECTORY_GENERATOR_V3.md)和[进度记录](GS_OPD_PROGRESS.md)。

## 历史路线：OA/L0T0放置候选预算耗尽，先解决低位Place可达性（2026-10-27）

唯一目标继续为从candidate9真实持抓前缀取得`obstacle_avoidance/L0T0`原生Task Success。已从t=344真实Place前Snapshot按8组上限搜索盘内位置、放置朝向、接近高度、开爪时序与撤离方向；5组没有上方路线，3组真实执行到21/41/80动作。最长c08分支到t=424仍双指持碗，低位Place目标在三组seed下均0/8 IK且CuRobo `no_result`。本轮没有释放、撤离或Settling。

| 指标 | 状态 |
|---|---|
| Task Success（放置、松爪、撤离、Settling后原生谓词） | false |
| 完整稳定释放 | false |
| 候选方案 | 8/8，不再扩展本轮预算 |
| 主搜索实际物理运行 | 142动作 / 3550子步 |
| 完整25子步监测 | 142/142动作 |
| Safety Success | false；严格安全认证0动作，规则未更改 |
| 新增可训练Episode | 0 |

任务可行性探索与安全认证分开。没有修改BDDL、直接写qpos或搬动物体，也没有将不安全接触认证为安全。双相机视频、分阶段/终态Snapshot、碰撞监测与逐动作记录位于`oa_l0t0_candidate9_placement_branch_search_20261027_130938`归档。意外重复的candidate6进程另有27动作/675子步，已停止且未计为新方案；末态Snapshot损坏情况在进度文档中披露。

**唯一工程阻塞：**从真实t=424测得的碗—夹爪相对变换，构造既贴合盘面支持几何、又存在IK解且经全场景碰撞筛查通过的Place目标。后续先离线确定该Keypose并验证IK/碰撞；本轮8组物理候选额度已耗尽，不继续执行新分支。历史目标与完整证据见[当前局部目标](GS_OPD_CURRENT_GOAL.md)和[进度记录](GS_OPD_PROGRESS.md)。

---

## 历史路线：OA/L0T0释放清除/撤离受阻（attempt22，2026-10-27)

唯一场景仍为`obstacle_avoidance/L0T0`。candidate9已产生真实运输和盘面放置前缀；从精确t=372松爪Snapshot恢复后，完整开爪46步使原生`on(bowl, plate)`在t=418从true变false，因此已立即停止。撤离尝试未执行：CuRobo多seed路线通过不了活动MuJoCo碰撞审计；接触法向变体IK无解或出现非法碰撞。t=372的瞬时Task Success及兄弟分支t=373–375的低速状态不构成稳定释放或完整Episode。

| 验收项 | 当前状态 |
|---|---|
| 原生Task Success（终止Snapshot） | false；t=372曾为true，t=418丢失 |
| 稳定释放 | 兄弟分支t=373–375有3步低速度证据，但撤离审计发现指外壳重接触；最终失败分支不稳定 |
| Retreat / 最终 Settling | 未完成；所有MuJoCo路径审计拒绝 |
| 25子步监测 | 本分支46/46完整（1150子步），严格安全可认证动作0 |
| Safety Success | false，既有安全规则未放宽 |
| 新可训练Episode | 0 |

下一阶段只有一个工程任务：在真实放置几何下重新确定Place/Release Keypose与全开姿态，使打开夹爪不推动黑碗，且CuRobo撤离路线通过完整MuJoCo审计。先做有限离线几何/IK/碰撞筛查，再执行一个有界真实分支；若任务谓词丢失或控制停滞立即停机并保留Snapshot。保持Task feasibility与Safety certification分开；本试点完成前不推广其他场景。详细记录见[当前局部目标](GS_OPD_CURRENT_GOAL.md)和[进度文档](GS_OPD_PROGRESS.md)。

---

## 历史路线：LIBERO-Safety OA/L0T0 Keypose复现准备（2026-10-27）

单场景最小复现优先，试点为`obstacle_avoidance/L0T0`。已完成对象/Region局部SE(3)与每阶段五个固定种子变体（阶段A通过，17项测试）。真实完整MuJoCo障碍场景接入CuRobo后，两组Keypose Bundle均仅完成Pre-Grasp/Grasp规划；闭爪scratch审计发现非指垫Finger Geom与碗相交以及夹爪自碰撞，路线严格拒绝。

**目前阶段门：B未通过，C保持关闭。** 不做物理执行、不把关节轨迹标成π₀.₅ OSC标签、不把scratch接触结果称为Episode安全认证。双Bundle报告与只读初态证据链接见[进度记录](GS_OPD_PROGRESS.md)顶部。后续唯一工作是验证新的稀疏抓持Keypose能否满足“左右真实指垫—目标物接触、无非指垫碰撞、无夹爪自碰撞”；该门通过后再计划完整的分阶段CuRobo路线，完整路线通过后才考虑最多一次标准1000步物理Episode。

旧Human L0T2与此前RRT/Region-Progress/CuRobo局部修补路线在此目标下暂停；文档后续内容保留为历史记录，不改变当前试点。

---

## 2026-10-27：Human L0T2 CuRobo规划问题定位，物理执行止于松爪预算

- 修复并验证了MuJoCo实际base↔CuRobo FK、EEF-site↔hand工具变换和15个Pot1源Geom附着球表示；旧四球拟合对两个候选制造了与瓶体的假碰撞。全场景起点及21个IK可达目标通过碰撞检查。
- CuRobo MotionPlanner真实返回两条路线，分别到预放置与放置目标；两段均由MuJoCo执行。Pot1终态原生On=true，但3600动作中仅余2步开夹爪控制，未通过释放；Pot2 On=false，完整task_success=false。没有任务成功Episode。
- 所有动作、双相机Observation/视频、25子步接触、快照、规划原始结果及FK/几何审计已归档于`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_curobo_transport_20261027_physical_attempt/`。接触子步采样完整，clearance监测不可用，安全认证false；7D arm targets不是π₀.₅原生动作标签。
- 下一次若继续该任务，先从归档终态评估可恢复性并按放置路径长度预留完整松爪/退爪/settling预算；不要重新抓取、删除番茄酱瓶或重复规划相同全路径。跨场景成功采集主线不应把本次Pot1持抓放置段误记为完整Episode。

---

## 最新路线状态：Human L0T2精确恢复成功，CuRobo预放置规划无路线（2026-10-26）

- **Snapshot恢复：通过。** 按历史启动和额外reset顺序，t=215、SHA `633b8b…`、331维Integration State和Pot1双指持抓均精确复现；恢复后未reset。
- **场景与IK：通过。** 当前MuJoCo完整碰撞场景只读导出无状态变化；CuRobo对24个Region等价目标并行IK通过23个。包含当前碰撞启用场景Geom、两个精确合并网格、机器人自碰撞模型和Pot1四球几何拟合。
- **CuRobo运动规划：阻塞。** MotionPlanner对全部候选在10.373秒后返回`planner_returned_none`。具体阻塞Geom未识别，标为unknown；不猜测为瓶接触或物理失败。
- **执行与采集：未启动。** 真实动作/子步/物理分支均为0，native `task_success=false`，完整Episode为0；无视频或动作监督标签。本轮不继续相似搜索。
- **归档：** `/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_curobo_transport_20261026_235735/`。恢复预检和旧版独立尝试的历史目录均保留。官方数据、训练及安全认证均未使用。

---

## 最新更新：Human L0T2 t=215持抓Snapshot精确恢复已验证（2026-10-26）

- 根因已定位：仅启动Bridge后立刻恢复缺少历史executor额外`reset`。按`BridgeSession.start → reset → model-SHA gate → restore Snapshot`、seed=0/init-state=0、demo horizon=3600复现后，模型SHA从`3b942cf3…`变为Snapshot所需`633b8b09…`。
- `compiled_model_fingerprint_details`显示Reset前后唯一不同字段为`body_pos`（33×3），其余字段摘要相同；同BDDL、模型SHA门禁不绕过。
- 原t=215 Runtime Snapshot恢复后Integration State逐元素精确一致；左右指垫接触Pot1、夹爪闭合、状态只读验证不变。动作和物理子步均为0。
- 可复现入口`research/gs_opd/repair/verify_human_l0t2_t215_snapshot.py`及配置/报告在`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_t215_bridge_recovery_20261026_run3/`。
- 下次可从同一恢复序列直接接CuRobo。当前仅验证抓持前缀，未规划、未运输、未获得任务成功Episode。

---

## 最新更新：Human L0T2 CuRobo尝试因Snapshot模型指纹不匹配而停止（2026-10-26）

- CuRobo既有环境可用，版本`0.0.post1.dev1`/源码commit `78fd485`；没有重装或升级。
- 合法t=215 Pot1持抓Snapshot内部模型SHA `633b8b0956f2f38e7f1c62067608f42743d0bccdc6bfeaca11976b5c7f301606`，当前相同任务Bridge模型SHA `3b942cf306a7decd7dcf576e7970d7a4dcb5af3c663db26da508dd5dd0a56fc8`。BDDL相同但模型不匹配，安全恢复门禁拒绝。
- 活动t=0 Bridge完成723 Geom只读清点及Pot/番茄酱瓶/炉灶语义绑定，但这不能证明Snapshot状态下的几何。CuRobo世界模型、规划和控制执行均未启动；动作、物理子步、成功Episode均为0。
- 保持停止旧RRT、Region-Progress和OSC阈值调试；先解决精确模型/Snapshot来源匹配，再继续场景建模与规划。详细报告见归档目录`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_curobo_transport_20261026_run1/`。

---

## 2026-10-26 路线更新：L2T2单次闭环路线验证未触发OSC，转向T0经验迁移

- 从精确t=2589持抓Snapshot核验模型、BDDL、状态和双指抓持来源后，复用L1T2自有Keypose经验；路线预筛4个有界RRT候选，其中`route_01`预放置离线RRT可行，终端IK/RRT因预算耗尽为unknown。
- 唯一Snapshot物理分支执行阶段重新规划预放置路线，20.053秒/265项检查后`planning_time_budget_exhausted`。OSC未启动，动作0、真实控制子步0，故无法判断稠密路径反馈能否改善Round60的OSC跟踪。状态仍2/3原生谓词，L2T2暂时未解决；不再增加相似L2轮次。
- 执行器将Keypose RRT完整7D关节路径和EEF参考保留为跟踪输入，不再压缩EEF路径，并复用现有实时观察与小步OSC反馈。相关执行逻辑通过25项executor测试和Python编译检查；本轮没有物理控制执行验证。Strict Safety Reconstruction、BDDL、物体状态与训练均未修改。
- 下一优先级是T0跨Level迁移，复用经验库中GS-OPD自有L0T0黑碗稳定抓持/真实抬升36.3 mm的子技能。源L0T0任务本身未完整成功，因此只当抓持技能先验；跨Level任务仍须独立满足原生`task_success=true`。所有历史记录保留，Round63证据见`GS_OPD_PROGRESS.md`顶部链接。

---

## 2026-10-26路线更新：缓存路线可复用，优先补齐物理闭环续接

- **已完成：**增加完整7D RRT关节路径和原始EEF路径的持久缓存与执行器交接，并以模型、BDDL、精确起点、目标、策略、规划器配置和路径hash共同验证复用身份。改变的路径段才触发重新规划；任务等价的区域放置姿态继续由通用几何生成，不要求复现历史世界坐标。
- **真实验证：**L2T2精确复用了GS-OPD自有L1T2路线（65个对齐路径状态），但42步真实OSC后仍在预放置前停滞。随后Human L0T2复用自有L0T2/L1T2成功经验，取得稳定抓持、34.404 mm抬升和360步运输进展，离预放置约85 mm；任务谓词未完成，松爪和稳定放置未到达。没有新增完整成功Episode。
- **本轮成本：**L2T2 42动作/1物理分支；Human L0T2 575动作/2物理分支；首次稳定抓持454.416 s；端到端墙钟耗时缺少可靠持久记录。两场景新增任务专用代码、人工改姿态数均为0。官方示范与训练均未使用。
- **共性缺口：**RRT路径的精确缓存复用解决了重复规划耗时，却没有保证现有OSC持续执行到位；局部失败Snapshot的关节限位门禁也会阻止直接续接。必须建立对真实状态变化有响应的通用路径续接/重规划接口，保持MuJoCo真实执行、关节限位和BDDL成功判据。
- **下一步：**暂停L2T2和Human L0T2同类路线追加。优先把按路径段缓存、当前状态核验、闭环进度反馈及只续接变化段组织成可恢复通用执行接口，再转到T0跨Level迁移验证；不得用放宽限位、停滞阈值或安全认证条件替代任务进展。

详见[跨场景机器报告](GS_OPD_CROSS_SCENE_CAPTURE_REPORT_20261026.json)和[可读报告](GS_OPD_CROSS_SCENE_CAPTURE_REPORT_20261026.md)。

---

## 最新总体里程碑：L2T2 Keypose跨Level续接保留2/3原生前缀，第三条稳定成功未取得（2026-10-26）

- **经验迁移：** L0T2与已完成L1T2的GS-OPD自有关键姿态经验均按活动BDDL和当前区域几何重定位；源动作未回放。首轮仅L0来源路线通过预放置全几何RRT/末端IK门槛，说明在当前L2摆位上L0经验更适配。
- **物理闭环：** 一条预放置RRT路线真实规划11.190秒并由MuJoCo OSC执行41步；跟踪在预放置Waypoint连续停滞后停止。Pot2始终双指抓持，t=2630仍保留Pot1 On与Stove Turnon，Pot2 On=false。真实物理分支1个，Student动作0。
- **有限备用筛查：** 从失败物理Snapshot精确恢复后，4次有界RRT调用没有生成新的端到端可执行不同路线；物理分支0。暂时停止继续相似路线搜索，保留Round60终态Snapshot。
- **指标与边界：** 本轮新增41个7D动作、1025个实际物理子步、RRT候选预筛166.663秒、物理路线规划11.190秒。原生任务未成功，松爪/撤爪/Settling未执行；接触身份齐全但包括Hand–Pot2非指接触，不能作安全认证。标准horizon仍1000，本轮运行标记demo-only horizon=3600；官方示范、训练和状态写入均未发生。
- **证据：** 综合机器与Markdown报告、dual-camera视频、动作、逐子步接触和可恢复Snapshot路径见`docs/GS_OPD_PROGRESS.md`。

### 下一步

继续从t=2630的Pot2持抓前缀解决“RRT离线可达、OSC预放置跟踪停滞”的控制衔接问题。先以无动作的状态/目标复核和有界可达性比较区分Waypoint定位与跟踪问题，再执行一条不同方向的候选；没有通过末端IK和全几何RRT的路线就不交给OSC。达到原生3/3后还需真实松爪、撤爪与Settling验收；Task-Success与严格Safety Reconstruction继续分离。

---

## 最新里程碑：run53后缀实现稳定物理释放（2026-10-26）

- **初始失稳成因：** 从同一t=1335成功前Snapshot对照原撤爪、原地松爪等待、缓慢上撤；无撤爪动作的等待分支同样在t=1342失去Pot2 `On`，说明原位置的边界支撑与释放滑移是主要问题。严格Safety Reconstruction阈值不变。
- **末端后缀修复：** 自动读取活动Cook Region并将最后持抓放置目标向区域中心移动25 mm，实测Pot2移动约24.1 mm；完成真实开放夹爪、缓慢上撤及Settling。未修改抓取/运输前缀、原生谓词或物体状态。
- **验收结果：** 77步、1925个完整物理子步；末态`task_success=true`、三项原生谓词全真、最终无手壶接触。Settling覆盖500子步且全程无夹爪接触；Pot2位置跨度0、最大线速度约`2.13e-14 m/s`。上撤早期有22个子步记录到短暂手壶重接触，记录不隐去；Settling及末态确认已完全分离。
- **证据位置：** `/opt/data/private/gs_opd/rollouts/run53_release_local_placement_20261026_attempt1/`，摘要`run53_release_stable_release_acceptance.md/.json`。35 mm同样成功但动作更多；侧向分支动作更多且短暂接触更多；45 mm放置偏移未能在预算内到位。所有候选从同一精确Snapshot恢复，视频、动作、25子步状态/接触及可恢复Snapshot均留档。
- **数据标签：** 这是基于run53既有成功前缀的稳定释放后缀，不是新的reset Episode。原生任务成功=true、轨迹物理执行可恢复、安全认证=false；无官方示范或训练。

本地释放目标已完成。本轮之后保持原路线：Task-Success成功与Safety Reconstruction分离；未经严格重构验证不得将此数据标为安全Episode。暂不启动其他场景搜索，除非后续目标明确要求。

---

## 2026-10-26 里程碑：run53 成功后的物理稳定性与来源链复核

- **原生成功：** L1T2 run53原始终态t=1335满足全部3个原生BDDL谓词，`task_success=true/reward=1/done=true`。成功时Pot2仍由夹爪持有；这项结果保持原样。
- **独立释放诊断：** 从成功前精确Snapshot恢复并重放run53自身118步，额外执行松爪、12步撤爪和20步Settling。t=1338夹爪与Pot2首次无接触时两个Pot仍On；t=1342撤爪过程中Pot2 On变false；Settling末Pot2仍在运动且On=false，故释放后稳定性不通过。桌面与玩具球新接触已记录。
- **来源完整性：** t=0…1335由run19、run46、run47、run48、run52、run53六段连续动作/物理状态记录组成；全部Snapshot边界payload与恢复报告及Integration State匹配。run48根报告缺失，但耐久中断索引与物理证据存在。逐步物理状态和双相机图像存在，完整策略Observation张量缺失；不能称为完整原始Observation–Action轨迹。
- **认证与范围：** 释放诊断只在控制步末采集接触/状态，没有25子步覆盖；原生成功、释放稳定、轨迹完整性、安全认证分列保存，`safety_certified=false`。真实MuJoCo物理执行，无物体状态写入；原任务定义和run53成功报告均未修改。本轮未训练、未启动新搜索、未改Keypose/IK/RRT。

证据位于 `/opt/data/private/gs_opd/rollouts/run53_release_stability_diagnostic_20261026_150815_continued/`。后续先基于该失败证据改进通用释放、支撑确认与撤爪后的Settling判定，再决定是否恢复其他物理搜索；不能把run53成功末态直接当作释放稳定状态。

---

## 2026-10-26 更新：从暴力搜索转向Keypose经验迁移

当前研究门线是“Task-Success Search → Safety Reconstruction → Validated Episode → Student Training/Data Expansion”。旧Task-Success-First多场景暴力搜索暂停；严格安全认证、阈值和训练代码保持不变。

1. **经验源已建立：** 从GS-OPD自有L0T2完整原生成功轨迹提取抓持、抬升、绕行、对齐、预放置、放置、释放、撤离姿态，分别记录物体坐标与目标Region坐标关系。仅迁移位姿和阶段结构，不回放源动作。该源轨迹本身含风险，`safety_success=false`。
2. **L1T2迁移已验证到完整原生成功：** run48同一持抓Snapshot旧方法920动作/3支未成功；Keypose run52+53用478动作/2支完成全部BDDL谓词。该结果从既有部分前缀续接，不代表从reset零样本完成。五个Keypose waypoint规划共记录14.376秒；旧基线规划时长缺失。
3. **L2T2迁移仍有通用运动规划缺口：** run54/55累计236动作、两支真实物理验证，从Pot2 held状态到high-west附近，但未到达BDDL目标。两次有界当前场景IK未解，OSC后续分别在lift与high-west阶段停滞；末态Pot2仍抓持且2/3谓词。没有任务成功或安全成功。
4. **接下来：** 对Keypose重定位后的workspace/IK失败和阶段接纳做通用归因；增加基于当前真实状态的候选姿态/局部路由而非复用不可达的绝对目标。仅在单次有界验证方案完成后续做L2；L2原生成功前不恢复L0T0或其他暴力搜索。每次保存完整Observation–Action、25子步状态、Snapshot、视频及原生谓词。
5. **后续阶段：** L2完成后再评估跨level成功成本和同技能经验摊销；然后选择一个不同技能L0场景。只有通过严格Safety Reconstruction与物理验收的数据才进入Validated Episode与Student Training；当前搜索轨迹均`task_success candidate`且`safety_certified=false`。

机器报告与证据索引：[跨场景迁移JSON](GS_OPD_CROSS_SCENE_TRANSFER_20261026.json)、[跨场景人读总结](GS_OPD_CROSS_SCENE_TRANSFER_20261026.md)、[详细进度](GS_OPD_PROGRESS.md)。

## 最新路线更新：跨层经验迁移已达抓取子技能，完整任务摊销待证（2026-10-26）

- **经验复用已物理验证：** L0T2自有Moka Pot物体相对抓取`t248`和阶段序列`t239`迁移到L1T2/L2T2；两个新布局都真实双指抓起并抬升。L2的第一只Pot又达到原生`On=true`，炉灶原生`Turnon=true`。
- **全任务结果：** 新的原生完整任务成功Episode为0。L2第二只Pot在运输中受Place Box、继而受已放置Pot1阻挡；run33的88动作耐久记录后Bridge断管，当前终态不可查询，最后记录`task_success=false`。L1放置也未确认任务成功。不同技能L0T0/T1/T3/T4均0/4完整成功。
- **成本验收：** 首个迁移抓取技能耗时L1 158.77秒、L2 474.324秒；L2第二个同类抓取技能614.191秒。L2整条多对象续跑共1800专家动作、2抓取分支、4运输/放置续跑段、18次RRT预检。全任务首次成功耗时为空，因此未证明随经验积累而降低完整任务搜索成本。
- **安全边界：** 接触法线用于生成真实MuJoCo局部重规划路线，碰撞保持有效并记录；不得修改物体状态。所有搜索数据`safety_certified=false`，安全重构阈值和认证保持独立不变。

### 下一阶段顺序

1. 从最后可恢复的Pot2持抓Snapshot，解决把已放置Pot1纳入多对象运输路径规划的问题；增加被阻挡目标状态的接触记忆与备选目标姿态/绕行高度，有限Snapshot重试后显式耗尽预算。
2. 打通release、settle和原生BDDL复核，验证L2T2完整成功。Bridge中断只保留最后耐久前缀，不将其提升为成功结果。
3. 迁移完成后自动把经过真实验证的对象相对技能写入GS-OPD经验库；单独记录搜索前后尝试数、动作数、分支数、从启动到首个子技能/完整任务成功耗时。没有冷启动对照时只报告“技能迁移”，不声称成本下降。
4. 再跨不同技能物体L0T0/T1/T3/T4；仍共用通用搜索器、物体/容器绑定和任务阶段，不写场景专用完整脚本，不使用LIBERO-Safety官方示范数据。

详情和每次run的机器可读指标见[跨场景迁移报告](GS_OPD_CROSS_SCENE_TRANSFER_20261026.json)。

# GS-OPD 总体路线

# 当前路线阶段：构建V3经验先验轨迹生成器并验证L0T0真实回合（2026-10-28）

全局研究仍以“自有成功经验 → 当前场景轨迹生成 → 原生任务验收 → 独立严格安全重构 → 合格Episode/训练”为主线。本阶段只处理LIBERO-Safety `obstacle_avoidance/L0T0/state0`，通过一条旧成功轨迹的路线结构先验驱动当前状态的B-Spline重生成与原生OSC_POSE闭环，不重放历史动作。

候选来自845动作L0T0成功轨迹；749步OSC段为left、0.020 m/s。左绕旋转漂移与右绕碗—柜接触均已由真实MuJoCo记录；实时姿态跟随改动已撤回。下一次由已筛选的上绕候选trajectory-003继续，经验库抓持姿态仍未进入该后端的动作控制。

attempt2/3/4的原生task success均false；其物理动作分别291/256/196，25子步监测完整。attempt4接触门控在木柜碰撞后停止了动作，严格安全和训练资格仍false。下一候选只切换到上绕trajectory-003，速度及门限不变。后续仍分别验收BDDL task_success、稳定松爪/撤离/Settling、操作安全、Strict 10 mm、子步覆盖与轨迹完整性；历史成功或监测不会跨Episode继承。

## 当前试点优先级：LIBERO-Safety OA/L0T0 candidate9有限运输（2026-10-27）

此前自研Task-Success-First RRT、Region-Progress、CuRobo局部修补及Human L0T2路线继续暂停。本阶段只复现LIBERO-Safety论文式稀疏Keypose到真实MuJoCo执行功能，不启动19,664条批量生成或模型训练；唯一场景为`obstacle_avoidance/L0T0`。

candidate9的t=190 Snapshot已精确恢复，`plate_1`现通过活动语义Body绑定及10个MuJoCo碰撞盒定位。CuRobo得41个有效路点，41态完整MuJoCo scratch路径筛查无非法接触。真实闭环发出9个关节目标动作、观察225个物理子步后停止于t=199：持抓保持，但碗离上方目标390.100 mm。第10路点预演将使非指垫Geom 73与目标碗Geom 113/114接触，故没有发送。双相机视频、动作/状态记录和可精确恢复Snapshot已留存；原生task_success=false、稳定释放未尝试、新增可训练Episode=0。

任务可行性与安全认证分别记录。每步均留有25个原始接触样本和25个几何结果，但Bridge组合监测为incomplete且严格几何为unsafe；目标持抓局部间隙低于原有2 mm规则。安全认证false，不放宽阈值。当前唯一阻塞是解决t=199到waypoint 10的非指垫碰撞并证明闭环运输可行；认证阻塞仍单独记录，不能把CuRobo静态无碰撞路径当作安全成功Episode。详细数据见[进度记录](GS_OPD_PROGRESS.md)与[当前局部目标](GS_OPD_CURRENT_GOAL.md)。

---

## 最新里程碑：经验复用已验证到抓取技能，尚未降低完整任务成功成本（2026-10-26）

当前跨场景验收以 [GS_OPD_CROSS_SCENE_TRANSFER_20261026.md](GS_OPD_CROSS_SCENE_TRANSFER_20261026.md) 为准；其更新并取代下方先前的run1–run3成本统计。

- L0T2自身成功源：1460个物理动作，t=1459得到原生task_success/reward=1/done=true。对象相对抓取与操作阶段模板取自GS-OPD历史，不导入官方安全示范动作。
- L1T2：通用匹配选择L0完整成功t248抓取模板及t239操作结构，182步专家搜索后稳定抓持并抬升34.54 mm，158.77秒首次验证该抓取技能。两次放置续跑合计723步仍未确认稳定原生目标，完整任务用时不存在。
- L2T2：同样选择L0完整成功模板；当前BDDL首待目标四次预检中2次IK无解、2次规划时间耗尽，零物理动作。不可将预检失败扩写成全局不可达结论。
- L0T0/L0T1：Student各8步后没有经验命中，四次RRT预检均未到目标，完整成功为0。
- 迁移模板命中2/2，跨层物理子技能通过1/2，跨层原生完整成功0/2；不同技能L0成功0/2。新原生完整成功Episode总数为0，故成本摊销仍未通过最终验收。
- L1抓取模板、两条经物理验证的L2抓取技能及L2部分转移结构已由隔离run库原子合并到持久v2库；未完成完整任务的技能仍标记native task success=false与safety_certified=false，L0完整成功经验优先。多目标调度、失抓放置检查和phase失败归档是通用逻辑；任务专用脚本、代码和人工调整均为0。

验证46项相关测试通过。下一阶段聚焦：利用真实失败/终态Snapshot改进L1释放后稳定放置与局部重规划；L2切换到不同完整成功的抓取变换或通用几何候选，限制同类重试；然后回到L0T0/L0T1验证经验库对不同技能的迁移。保持Student原始checkpoint、真实MuJoCo执行和原生BDDL唯一成功判据；所有搜索轨迹留待独立严格Safety Reconstruction，不作为安全训练数据。


## 当前路线优先级：从成功经验摊销跨场景搜索成本（2026-10-26）

目标优先级调整为：利用自有L0T2成功轨迹的对象相对抓取模板、阶段顺序和历史接近参考点，优先迁移L1T2/L2T2；不可迁移段才调用Student、IK、RRT、OSC与Snapshot有限搜索；物理验证通过的新技能再进入经验库。避免逐任务从零暴力搜索或写完整任务专用脚本。所有迁移仍要经过真实MuJoCo动力学执行，并以原生BDDL `task_success=true`作为完整成功判据。task-success搜索中的真实碰撞只记为后续重构风险，不放宽物理动作接口、关节范围或仿真稳定性要求；Strict Safety Reconstruction验收不变。

源L0T2经验库由一条1460动作、native-success=true的GS-OPD自有Episode构建，含2个Moka Pot object-relative grasp模板和2个pick/transport/place/release操作阶段模板。迁移对象和目标从活动BDDL、运行时Geom/Body以及真实目标Region解析。历史OSC动作没有直接跨场景回放，亦不作为新的动作标签；未读官方安全示范数据。

截至目前，L1T2三次有限尝试的最近记录为2个物理Snapshot分支、376个专家OSC动作；完成部分是历史经验候选接近路径（207步后仍距预抓取约90 mm）和一个几何候选到达预抓取，随后24步表面接近预算耗尽。L2T2两次已完成尝试的最近记录为1分支、320个专家OSC动作；RRT无路线后通过通用高位OSC实际移动，最高距预抓取约70.6 mm。两场景均未获得稳定抓持/抬升或原生完整成功；首条成功用时均为空，经验检索命中率2/2但已物理验收的技能复用率0/2。L1与L2无场景专用完整脚本或人工坐标更改。

新加入的闭环预算按真实OSC速度估算路线动作数，具固定上限；物理预接近失败可保存当前Snapshot，并尝试单次局部RRT/OSC重规划。L2T2 run3正在验证，结果和监测完整性记入进度文档。回归测试25项通过。需要继续改善的核心不是增加相似候选，而是OSC对迁移目标的实际跟踪和有证据的局部恢复；跨独立运行的失败路线持久记忆及不同物体族（Book/Bowl/Cup）经验仍未完成。

验收指标继续按场景列出：native success、完成谓词、稳定抓持/抬升/搬运里程碑、首次成功用时、物理分支数、Student/专家动作数、复用技能、完整脚本与人工调整数量、碰撞与子步监测覆盖。无成功的场景明确 `time_to_first_success=null`，不把候选IK、部分谓词或抓持Snapshot报作完整成功。

## 路线状态：Task-Success-First框架已接入L0T1，完整任务成功仍未获得（2026-10-26）

主路线保持：原始π₀.₅ Student优先 → 通过停滞/技能失败触发有界专家分支 → 真实MuJoCo动作与碰撞记录 → 失败恢复Snapshot重试 → 阶段目标完成后允许Student接续 → 只以原生BDDL `task_success`作为完整任务成功。已新增通用执行入口，复用原Student模型适配器、Safety Bridge、Runtime Snapshot、几何抓取生成、IK、Task-Success专用record-only RRT、OSC_POSE和TaskPhaseController；严格安全重构、认证及训练代码未修改。

L0T1验证结果：32步Student段未抓书，自动恢复t=0后启动专家搜索；候选姿态与反向/长轴接近共验证12/16个，做了最多3分支的多轮真实OSC尝试。已执行分支都在预抓取阶段因真实Beer邻近碰撞/路线阻挡停滞；反向侧候选出现IK不可达。动作和接触日志、双相机视频、逐子步监测状态、t=0及分支Snapshot均在归档；不完整监测明确标注为未知，不能声称安全。尚无双指稳定抓持、30 mm抬升、Student阶段接续或原生目标放置成功，因此无完整任务Episode。报告与归档位置见 [GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md) 顶部。

下一阶段进入“物理可达性反馈驱动的候选多样化”：用已记录的Beer/Hand接触和预抓取位姿误差调整候选接近侧与有限RRT重采样；不对不安全接触设置统一的早期拒绝门槛，真实动作继续允许碰撞并逐步记录。之后依序验证专家抓取/抬升、Student接续、BDDL contain-region放置和release/withdrawal/settle。每次失败从该尝试的可恢复Snapshot重试；只有最终native BDDL成功成立的轨迹可标为task-success candidate，安全属性仍需走原有独立严格验证。

本版入口先支持一个由活动BDDL解析出的`in/on`目标对象；多对象的有序阶段组合及30任务配置仍是后续工作。

## 最新路线状态：L0T1 书本候选通过IK、未通过抓取安全几何筛选（2026-10-26）

修复了已有MuJoCo规划适配器把最终指垫接触放在开爪状态测量的问题。当前候选筛选把张爪接近、闭爪目标IK、两侧pad距离和完整机器人几何闭合路径分开核验；保留精确静态安装Geom `(90,14)`例外，不放宽Hand、Floor/Table、Pedestal、Beer、Place Box或目标物安全要求。

从原始L0T1 t=0 Snapshot对24个已归档书本候选完成有界IK和scratch几何筛选：预抓取与闭爪目标IK均24/24可达，左右指垫闭合采样及关节范围均有完整记录。六个候选的双指几何距离在现有±2 mm容差内，另外18个过度穿透；但所有24个闭爪候选均出现 `gripper0_hand_collision` Geom 79 对书本Geom 92的非法接触（约−59.758 mm）。候选集筛选成功数为0，故不进行RRT、OSC抓取或抬升。Beer/Place Box最小路径间隙76.370/13.553 mm，初始Snapshot在只读筛选前后保持完全一致。结果归档及逐候选证据见 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_t0_book_grasp_candidate_screen_20261026_attempt4/`。

总体采集路线不变：原始Student Rollout → 暴力搜索任务成功轨迹（允许不安全）→ 仅使用自身历史成功经验做几何重定位与安全重构 → MuJoCo完整验证 → GS-OPD安全Episode。当前L0T1阶段状态从“t=0安全门禁阻塞”推进到“24个候选IK均可达，但抓取姿态的Hand穿书/部分双指穿透阻断”；下一子目标是按现有安全规则生成并筛选不与书本相交的Hand姿态。该轮无物理抓取、RRT、Student Rollout、训练或LeRobot数据转换，官方安全示范未参与。

## 最新路线状态：L0T1 固定安装几何门禁已精准解除（2026-10-26）

独立scratch `MjData` 恢复并核验了原始 `obstacle_avoidance/L0T1`、seed 0、init-state 0 Runtime Snapshot。此前“world floor Geom 14”的描述已更正：Geom 14 是挂在 `study_table` Body 1 的未命名桌面Box；Geom 90 是固定 Panda 安装圆柱，挂在 `mount0_pedestal` Body 22。模型SHA精确匹配，双方祖先链无关节，7个Panda关节扰动下两者几何不移动；t=0 −99.460 mm是固定安装座与桌面边缘模型重叠。

Safety Bridge仅在模型SHA精确命中可信静态pair注册表后纳入 `(90,14)` 局部接触例外（显式空列表可关闭）；策略归一化还要求Geom/Body身份全匹配且双方祖先链固定。旧L0T1策略省略新字段时也能兼容此精确模型。全场景门禁修复前 `unsafe`，修复后 `complete`、剩余最小间距 +28.005 mm；Beer/Place Box、安全阈值、BDDL、其他Floor/Table与Pedestal Geom、自碰撞和移动连杆规则未放宽。原始Snapshot的Integration State在只读前后校验中哈希不变；没有物理步、动作或IK/RRT。

首个L0试点仍为书本放入Caddy后格，但当前初始安全门禁不再是阻塞。抓取候选筛选尚未开始；已知开夹爪状态下指垫末端距离检查问题仍待后续单独修复/适配，且不得放宽原安全阈值。回归测试覆盖精确Pair许可、同Floor非法移动连杆接触保留和jointed body拒绝，2项通过。修复前后门禁和几何证据归档在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_t0_safety_gate_static_contact_fix_20261026/`。

下一阶段路线仍是：原始 π₀.₅ Student Rollout → 暴力搜索任务成功轨迹（允许不安全）→ 仅用自身历史成功经验进行几何重定位与安全重构 → MuJoCo完整验证 → 生成GS-OPD安全Episode。官方安全示范只保留为独立Offline SFT Baseline，不进入当前搜索先验、冷启动、重定位或训练Episode。

## 修复前路线快照：L0T1 抓取冷启动被静态底座门禁阻断（2026-10-26）

## 修复前路线快照：L0T1 书本抓取冷启动被静态底座门禁阻断（2026-10-26）

已从原始Student的t=0 Runtime Snapshot核对并恢复L0T1；场景、BDDL、原生目标及编译模型指纹一致。按活动黑书碰撞盒与真实Panda指垫生成24个几何候选，但现有全几何Safety Bridge在任何候选筛选前就判定初态unsafe：Panda固定底座 geom 90 与 world floor geom 14 有 `-99.460 mm` signed-distance重叠。该项不是 Beer 或 Place Box 违规；两者对机器人最小距离为66.233/142.490 mm，书与两者t=0无对象接触。

本轮没有运行候选IK、RRT、OSC物理分支或抬升，故抓取Snapshot、双相机操作视频和实际动作/子步记录都未产生；24个候选只是几何生成结果，不代表可执行抓取。完整证据和失败分类保存在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_t0_book_cold_grasp_20261026_attempt1/`。

已有 `grasp_retargeting.py`、`jointspace_rrt.py`、`mujoco_jointspace_planning.py`、OSC_POSE、抓持验证与`task_phase_controller.py`可复用，不能再记作“缺少RRT/阶段控制器”。重启该试点前需为固定底座—floor基线重叠设计局部、可审计的几何处理；不得降低Beer/Place Box禁触余量、书本指垫接触容差、关节限位或BDDL规则。该处理本轮没有授权实施。无官方安全示范数据参与，既定Student rollout、暴力搜索、自有成功经验重定位、安全重构与MuJoCo验收路线不变。

解除初态门禁后还需处理一个未运行验证的接口风险：当前 `screen_grasp_candidates` 的最终对向指垫距离计算发生在开夹爪姿态；接触筛选需在不放宽容差的条件下与闭合几何一致。本轮未调用该筛选器，也未改规划/仿真源码。

## 本轮进度：L0T1 原始 Student 基线已采集，任务未成功（2026-10-25）

已完成新场景首轮原始 π₀.₅ Student rollout：`obstacle_avoidance/L0T1`、init-state 0、seed 0、最多1000步，原生目标 `In(black_book_1, desk_caddy_1_back_contain_region)`。模型不带LoRA。Student达到1000步上限，task_success全程false、reward=0、done=false；没有机器人—书接触或稳定抓持证据，书也没有进入caddy后格。失败原因集中在目标获取前的偏离与碰撞：机器人先后接触 Beer（52动作步）和 Place Box（29步），step128书与Beer接触；完整结果和碰撞记录保存在私有rollout归档。它是失败搜索历史，不是成功路径，也未获安全认证。

模型加载后的标准 Bridge reset 保持任务/BDDL以及书、Beer、mug起始状态一致，但重采样了caddy和Place Box fixture位置；归档同时保留reset前只读预检与reset后真实rollout起始Snapshot，以后者作为Episode场景的物理依据。

本轮完整采集了1000个真实7D动作、逐步 Observation/机器人物体状态和双相机轨迹，且每动作的25个MuJoCo子步监测均完整（合计25,000）；起始及终态Snapshot已在导出时的活动Bridge模型内恢复校验通过，跨进程恢复需匹配模型几何SHA和场景元数据。接触接口把各子步观测聚合为每动作覆盖和按Geom对统计的接触出现，不保留每个子步完整原始接触点列表；后续若需要逐点逐时刻回放，应先扩充记录格式，但本轮不得补跑第二个Student Episode来追填。

路线仍按原约束推进：原始Student rollout → 暴力搜索任务成功轨迹（可保留不安全碰撞）→ 只从GS-OPD自身历史成功经验做几何重定位和安全重构 → MuJoCo完整验证 → 形成GS-OPD安全训练Episode。当前 episode 只作为原始失败/碰撞经验；不提供正向成功先验、不进入安全训练集。官方安全示范未下载或参与任何当前采集。

下一阶段建议先离线复核本回合 step37–99 Beer 接触、step128书—Beer接触、step136–268 Place Box接触及书的状态变化，再制定书的抓取与caddy后格插入证据配置。已有 `jointspace_rrt.py`、`mujoco_jointspace_planning.py` 与 `task_phase_controller.py` 可在研究分支复用；本轮没有运行RRT或阶段控制器。仍需完善任务级接触策略映射、稳定抓持证据、插入/退爪阶段条件，并在后续成功候选上完成逐步MuJoCo认证。

归档目录：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_raw_student_seed0_init0_20261025_234627/`；人读摘要 `episode_summary.md`，完整机器报告 `episode_report.json`。本轮未训练、未转换LeRobot、未将任何未经安全验证的轨迹标作安全数据。

## 本轮补充：LIBERO-Safety 跨场景配置盘点与代码复用修订（2026-10-25）

本地只读盘点 `obstacle_avoidance` 与 `obstacle_avoidance_human` 的 L0/L1/L2 共 30 个任务，逐项整理 BDDL、符号初始状态、原生目标、接触代价对象、动态定义、技能需求与静态采集可行性。机器清单和 Markdown 表见 [task manifest](GS_OPD_LIBERO_SAFETY_TASK_MANIFEST.json) 与 [任务清单](GS_OPD_LIBERO_SAFETY_TASK_CHECKLIST.md)。本轮未读取/下载官方安全示范，未运行 Rollout、MuJoCo 或训练。

严格沿用：**原始 π₀.₅ Student Rollout → 暴力搜索成功轨迹（允许不安全）→ 仅用 GS-OPD 自身历史成功经验做几何重定位与安全重构 → MuJoCo 完整验证 → 生成 GS-OPD 安全训练 Episode。** 官方安全示范中的动作、关键帧、轨迹、专家标签不得作为当前搜索先验、冷启动经验、重定位来源或 GS-OPD Episode；仅保留未来独立 Offline SFT Baseline。

代码复核来源是实验分支 `gs-opd-overfit-recovery-08a0198`，审计提交 `2af2ef2d8651fce1196ba0ee8317fb21c7bc7360`。此前只查主分支导致的“缺少 RRT/阶段控制器”结论现已修正：`research/gs_opd/repair/jointspace_rrt.py` 中的 `JointSpaceRRTConnect`、`mujoco_jointspace_planning.py` 的 MuJoCo scratch-scene 规划适配器，以及 `task_phase_controller.py` 的配置化阶段推进均可复用。需要扩展的是任务级 forbidden/allowed contact 映射、目标 IK、动态障碍相位同步、阶段证据配置，以及 caddy/basket/microwave 插入、铰链门和动态人手障碍策略；RRT 规划结果不替代完整 MuJoCo 执行轨迹验证。

推荐首个新 L0 试点为 `obstacle_avoidance/L0T1`（书放入 caddy 后格），先核对 `bottle_of_beer__7_1`、`white_place_box_1` 的活动碰撞 Geom 绑定，并为现有阶段控制器配置插入与退爪证据。此准备记录与下方既有 L0T2 实验历史并存，不替换、不否定此前归档结果。

## 当前路线状态：LeRobot训练Episode视觉与Observation–Action审计完成（2026-10-25）

L0T2 Round 0仍有一条通过Native+Operational验收的1460步LeRobot v2.1训练Episode（Strict-10mm附加指标未通过）；本轮只读使用官方LeRobot 0.3.3 Reader逐帧检查数据，输出完整73秒双相机、20 FPS视频和机器可读审计。所有1460帧的索引、pre-action Observation、真实执行7D action及t→t+1时间对应通过；25个来源段及24个Snapshot接缝均在报告中逐项记录。t=1459是训练最后一帧，t=1460只保留为provenance终态证据。

视觉报告无双相机重复帧、错位或超过稳健阈值的大幅跳变。t=10与t=398有小幅相机差异，被标作来源接缝人工查看项；对应前后画面连续且Snapshot兼容证据齐全。PandaGripper动作符号核对为负值张开、正值闭合、零值保持；t=6/9两条略低于−1的张开源动作原值保留并显式标注。视频、关键帧、接缝对照、JSON报告和复现入口在`/opt/data/private/gs_opd/audits/l0t2_round0_dataset_visual_audit_20261025/`与`research/gs_opd/data/audit_and_render_l0t2_round0_lerobot.py`。

此前Fresh rank-32 LoRA 30k训练已自然完成；30k rollout 1800步任务未成功，且缺少全几何余量证书并出现hand–Pot 2接触，不作为安全样本。训练Checkpoint与失败rollout归档保留。本轮未修改数据集、源轨迹、训练代码或权重，也未启动训练或环境Rollout。

## 前置路线状态：L0T2 Round 0 Fresh LoRA配置已完成单batch预检（2026-10-25）

训练数据仍是已验收的单个LeRobot v2.1 L0T2 Episode（1460步、1451个合法10步窗口）。训练入口现在按逐动作provenance中的真实12阶段分类窗口，默认每个唯一窗口采样一次，并为batch=3补齐一个transport阶段窗口；不再使用旧L0T0 845帧和三阶段名称/采样数。

π₀.₅原Student checkpoint与L0T2 rollout所用本地文件相同（SHA-256 `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`）。所有667个非LoRA权重键加载匹配；新建rank-32 Action Expert LoRA，基础模型冻结，未加载旧adapter/optimizer。官方OpenPI LIBERO图像、状态、7D动作与quantile normalization转换通过真实batch验证；loss有限、216个LoRA梯度有效且权重未变化。

新配置：`examples/sft/config/gs_opd_l0t2_round0_fresh_lora_20261025.yaml`；机器报告：`/opt/data/private/gs_opd/sft/round0_l0t2_fresh_lora_preflight_20261025/preflight_report.json`。单GPU RTX 4090、FSDP `no_shard`即可做batch 3；RLinf后端支持多卡数据并行，但GS-OPD provenance sampler目前限定world size=1，因此本配置不调多卡或模型切分。正式优化训练仍未启动；10,000步、rank-32、`1e-4` constant lr等为此前成功配置的起始参考，是否开始训练由后续单独授权决定。

## 最新路线状态：首条L0T2 Operational安全成功候选已生成单Episode LeRobot v2.1数据（2026-10-25）

统一验收通过的Round 0候选已转成独立LeRobot v2.1训练数据目录；25个原始分段在数据集中是**一个**20 Hz、1460步连续Episode。全部逐步真实7D动作、动作前双相机与8D状态已通过LeRobot 0.3.3 reader全量回读，1451个Action Horizon=10窗口均无跨终止填充。源段、阶段、24个Snapshot接缝、Native/Operational安全、逐动作重复记录的demo-only成功/标准1000失败/Strict-10mm附加失败项及源文件哈希均保存在数据manifest与逐动作provenance。

数据路径：`/opt/data/private/gs_opd/datasets/lerobot_l0t2_round0_operational_success_v21_20261025/`；机器验收：`round0_lerobot_acceptance.json`；转换入口：`research/gs_opd/data/convert_round0_acceptance_to_lerobot.py`。样本资格仍是1条Operational安全成功Episode：demo-only 1800成功、标准1000失败、Strict-10mm未通过。未启动SFT/LoRA，历史1093步示范及旧LeRobot数据未覆盖。

### 当前阶段结果与路线边界

1. **Round 0物理轨迹候选：已通过统一Operational验收。** t=0→1460共1460动作、36500子步、24个兼容Snapshot接缝，最终原生task_success/reward/done成立。
2. **LeRobot数据准备：已完成。** 一条Episode完整回读；不按分段拆分、去重或跨终止构造窗口。当前数据目录可供后续数据审查与训练配置使用。
3. **后续训练：未启动。** 本轮没有训练授权，也未执行SFT/LoRA；等用户明确要求后再单独确定训练配置与启动条件。

严格10 mm只是附加统计，当前候选仍标为Operational安全通过、Strict-10mm未通过。标准1000步Benchmark未成功，1800步demonstration-only任务成功。完整数据验收报告详见上述目录中的manifest与acceptance JSON。

## 最新L0T2 Round 0状态：完整t=0→1460 Operational安全成功候选已通过验收（2026-10-25）

当前首条Round 0安全成功数据候选已建立。为补齐最后缺口，从raw π₀.₅ seed/init-state重新物理执行前10个动作，所有250个子步通过当前Native/Operational规则，t=10 Integration State及控制器、夹爪、机器人缓存、任务bookkeeping和物体状态与Episode 9规范起点完全匹配。运行时Snapshot字节哈希不同，按状态级精确兼容处理。已验证的t=10→249回放接上t=249→1460归档链；除t=10外的23个Snapshot接缝字节SHA完全一致，OA时间索引连续。

全链有1,460个真实7D动作、36,500个25子步审计项、双相机Observation–Action及完整终态快照。Native Safety和Operational Safety通过；t=1459首次原生task_success=true，t=1460 Pot1/ Pot2 On与Stove Turnon均为true、reward=1、done=true。Standard 1000-step Benchmark=false；demonstration-only 1800-step=true。Strict-10mm统计false，明确只作附加统计、不覆盖Operational结果。历史动作未回放拼接，LoRA未启动，1093步示范与LeRobot数据未改。

完整机器报告和人读摘要位于`/opt/data/private/gs_opd/rollouts/round0_l0t2_complete_episode_readonly_acceptance_20261025/`；审计入口为`research/gs_opd/repair/audit_round0_complete_episode_archives.py`。旧版“t=0→10安全缺证、尚无Round 0候选”的进度段落由本次统一验收更新；未改变任何安全规则。

## 前一阶段L0T2前缀审计：t=10→249当前策略通过（后续已补齐t=0→10，2026-10-25）

> 以下为t=0→10当前策略安全重放完成前的阶段性状态；顶部统一验收已验证完整t=0→1460候选链。

已从原始π₀.₅ t=10已验证Snapshot，在隔离环境按Episode 9归档OA逐动作重放至t=249。239动作、5975子步的当前对象级Native/Operational结果全部通过，目标Pot 2非手指净距下限2 mm、Ginger Beer/Place Box各10 mm、仅四个指尖对Pot 2允许最多2 mm穿透、自碰撞及关节规则均保持。t=249终态与已有t=249→1460成功链源Runtime Snapshot字节完全一致；Integration State、OSC controller、gripper、robot buffers和task bookkeeping一致，Pot 2稳定双指抓持通过。实验没有执行t=249后缀，也未做轨迹搜索或LoRA。

归档及可复现入口：`/opt/data/private/gs_opd/rollouts/l0t2_episode9_prefix_t10_to_t249_current_safety_20261025/`、`research/gs_opd/repair/replay_episode9_t10_to_t249_current_safety.py`。本段t=10→249证据仍为完整连续后缀；当时尚未认证的t=0→10现已按顶部统一验收完成物理重放与安全复核。Round 0候选状态、完整接缝和最终谓词以顶部新报告为准；保留所有历史示范与LeRobot数据。

## 前一阶段L0T2状态：同Episode任务成功，t=1061→1460安全延续已审计（前缀复验前，2026-10-25）

以t=1061 Pot 1已抓持抬升Snapshot为入口，连续串接6个哈希核验的耐久归档段，共399个新真实动作至t=1460。RRT仅在旧运输路线终点姿态与实时抓持目标相差0.39274 rad时从当前真实状态重规划一次（33点、16.286 s）；此前t=1064→1374已有路线完成运输，t=1403进入precise placement。t=1453严格恢复后依据同Episode真实记录的阶段事件进入release；Pot 1在t=1459达成原生On，任务同时返回`task_success=true, reward=1, done=true`，t=1460实际松爪，原生Pot 1/Pot 2 On和Stove Turnon保持为真。最终物体真实接触burner，末五帧位姿跨度0.896 mm、线/角速度低。

t=1061→t=1460共有9975个记录MuJoCo子步，监测全部完整且没有硬安全违规；396个动作有最终预测/实执行一致证书，最早3个单步在提交后遇到日志/比较包装异常，仍保留完整动作、子步监测和逐段精确Snapshot连接，但没有最终提交比较证书。最终七步Native/Operational与逐子步证书齐全。Operational Pot 1目标间隙最低5.991 mm（规则下限2 mm），Ginger Beer携带间隙最低10.706 mm、Place Box最低91.736 mm，finger/support接触穿透均远低于2 mm；Strict-10mm整体为false。t=1460成功后运行器又用7.924 s生成一条33点withdrawal路线，但隔离预演预测On将丢失，故没有提交物理动作。源代码现已把“原生全任务成功+已释放”设为立即终止条件，避免后置RRT。

当时的结果是**同一Episode原生任务成功且t=1061→1460安全延续已审计**，前缀缺口使其尚不具备完整候选资格。此状态现由顶部完整t=0→1460 Operational安全验收取代。demonstration-only成功；标准1000步失败；未训练LoRA。轨迹、视频、逐子步证据、终态Snapshot及验收JSON位于 `/opt/data/private/gs_opd/rollouts/l0t1453_pot1_release_native_support_continue_20261025_attempt1/`。

## 最新总体状态：t=1061取得Pot 1安全抓持与32.4 mm真实抬升（2026-10-25）

Pot 1 Approach曾因10 mm目标物余量预演拒绝而停在t=1042。本轮审计了已验证Pot 2目标对象Operational策略，并将同一2 mm非指尖正间隙应用于当前Pot 1目标；四个指定指尖Geoms保留最多2 mm合法穿透，Palm/hand和其他机器人部件仍必须保持非穿透且满足2 mm目标余量。Ginger Beer/Place Box仍为10 mm，自碰撞、关节限位及原生BDDL不变。此前拒绝的原OSC动作在当前真实t=1042 Snapshot的25子步预演通过后获准执行。

同一连续物理状态链从t=1042到t=1061执行19步、475子步，全部Native/Operational检查通过。Pot 1双指真实稳定抓持，抬升32.438 mm并进入transport；Pot 2原生On=true全程保持。Pot 1非指尖最小间隙5.607 mm，Ginger Beer/Place Box最低47.329/68.368 mm；最大逐子步指尖对穿透0.694 mm。终点task_success=false、reward=0、done=false，完整双壶任务仍未完成；该结果是安全持物后缀，不是Round 0种子。全局Strict-10mm补充统计false（未映射环境几何记录0 mm），未将本段误标为Strict-10mm Safe。无新RRT、抓取搜索或LoRA。

归档及t=1061 Snapshot：`/opt/data/private/gs_opd/rollouts/l0t1042_pot1_grasp_lift_20261025_032402/`。下一阶段从该Snapshot严格恢复，利用现有控制和持物检查继续Pot 1搬运、放置、释放与原生On谓词验收；之后仍需完整核验前缀，才可声称完整Round 0安全种子。

## 最新总体状态：Pot 1 抓取停在 t=1042 的安全预演门槛（2026-10-25）

从已验证 t=1019 状态连续执行到 t=1042：23 个真实动作、575/575 子步安全通过，Pot 2 原生 `On=true` 保持。Pot 1 仍在 Approach，未抓持或抬升。下一候选在隔离预演中使 hand–Pot 1 净距从第5子步的9.996994 mm降至最低8.009800 mm，违反现有10 mm Operational余量，因此未提交；无实际碰撞或禁触违规。该阻塞不能用辅助轨迹误差或风险预警解释，也不能通过降低安全阈值处理。

执行与审计归档分别为 `/opt/data/private/gs_opd/rollouts/l0t1019_pot1_grasp_lift_20261025_030744/` 和 `/opt/data/private/gs_opd/rollouts/l0t1042_rejected_approach_preview_audit_20261025_031026/`。继续路线：取得满足冻结 Operational 余量的 Pot 1 Approach 接入动作/状态，再完成稳定双指抓持和真实≥30 mm抬升；之后才继续搬运放置及完整原生任务验收。保留原有禁触、碰撞、关节及BDDL标准，不重做此前已通过段，不训练 LoRA；所有局部后缀均不得冒充完整 Round 0 连续安全种子。

## 最新总体状态：t=1019抵达Pot 1预抓取并进入Approach（2026-10-25）

复核t=649→794阶段共3条接受的RRT路径（总计22,510构型检查、约427.06秒）与t=738/t=794两次Ginger Beer 49.581/49.204 mm风险预警后，本轮复用最后一条282点路径，不再调用规划器。预警无接触且大于Operational冻结的10 mm硬余量；50 mm边界仅作诊断，预警时以当前真实Snapshot的短段物理认证决定提交。没有重置历史同因保护或降低安全阈值。

由t=796精确续接执行至t=1019，223个真实OSC动作、5,575/5,575子步通过Native Safety和Operational Safety，Pot 2原生On一直为真。Ginger Beer/Place Box/Pot 1段内最小净距24.470/86.082/30.778 mm；全机器人几何、禁触、自碰撞、限位、Pot 2谓词均持续核验。Pot 1状态机在t=1019进入`approach`，预抓取误差10.296 mm及0.04608 rad。该进度解决了“停留在second_object_recovery且重复重规划”的阻塞；抓取、Pot 1放置和完整原生任务仍待完成。

当前状态和证据：主归档`/opt/data/private/gs_opd/rollouts/l0t796_pot1_execution_aware_pregrasp_20261025_021845/`，恢复入口`research/gs_opd/repair/continue_t794_pot1_execution_aware_pregrasp.py`。t=794→795→796两个单步恢复归档亦保留。全局Strict-10mm附加统计不通过（未映射环境组0 mm）；本段具体目标与禁触物均高于10 mm，Native/Operational通过。Pot1 On=false、Pot2 On=true、任务成功=false；标准1000步完整成功和demo-only 1800步完整成功均false。无完整Round 0种子，无LoRA训练。

### 接续路线

1. 从t=1019严格恢复，保持Pot 2 `On=true`，在同一条物理链上由Approach执行Pot 1闭爪、双指稳定抓持和真实抬升；逐动作继续使用当前安全Snapshot的短段MuJoCo认证，不把qpos参考差当成许可。
2. 抓持稳定后复用已有持物几何检查、任务阶段控制和已验证控制接口完成Pot 1搬运与放置，验证原生Pot 1/2 On和最终task_success/reward/done。只有确需路线且证据支持时才按用户授权使用既有有界规划。
3. 该t=1019后缀完成后仍需从原始安全t=10启动连续Episode并统一审计全物理前缀，才能命名Round 0安全种子；其后才考虑轨迹扩增至3条及LoRA训练。

## 历史总体状态：Pot 2放置后Pot 1阶段在t=794触发重复路线风险（已推进至t=1019）

Pot 2在t=649已通过稳定放置。t=649→794的旧后缀安全通过，但在t=738/t=794同一Ginger Beer 50 mm预警处重复重规划后停止；该预警未构成硬余量违规。该阶段的完整历史和诊断保留在下文，本轮已通过复用t=738第三条路线到达Pot 1预抓取并进入Approach。

本轮从已核验的 t=649 Pot 2稳定放置Runtime Snapshot开始，在同一Episode真实执行145步至t=794。先完成Pot 2释放退出，按Native谓词进入Pot 1 `second_object_recovery`；Pot 2 `On`全程保留。3,625个物理子步的Native Safety与Operational Safety均通过，禁触、自碰撞、关节限位违规为0。Pot 2非手指最小余量4.242 mm、Ginger Beer 32.097 mm、Place Box 47.198 mm；附加Strict-10mm未通过。

Pot 1恢复路径从t=699规划后，t=738首次因`gripper0_hand_collision`与Ginger Beer `bottle_of_ginger_beer_1_g102`间隙49.581 mm触发50/60 mm风险滞回并重规划；t=794同一Geom对的间隙49.204 mm，再次以相同`risk_margin_approaching`触发。距离仍高于10 mm硬余量，未发生接触；现有重复失败原因守卫应fail-closed，但其错误诊断调用因`reason`关键字重复而报TypeError。三次已接受RRT路径累计规划427.062秒、22,510次构型碰撞检查，qpos路径节点为49/417/282。该调用错误已在`research/gs_opd/repair/automatic_recovery.py`中做窄修复，没有更改RRT、安全阈值或物理状态。由于相同风险原因重复，不从t=794重新初始化后重试该路线，避免清除停滞保护并循环执行。

Pot 1尚未抓取：Pot1 On=false、Pot2 On=true、Stove Turnon=true，task_success=false、reward=0、done=false。终点t=794在标准1000 horizon内，但标准完整任务成功和demo-only 1800 horizon成功均未取得；无Round 0完整种子、无LoRA训练。全程数据及失败分类归档于`/opt/data/private/gs_opd/rollouts/l0t2_t649_pot1_task_suffix_20261025_005902/`，重点文件为`failure_diagnosis.json`、`observation_action_trace.npz`、`recovery_events.jsonl`、`snapshots/main_timestep_000794.pkl`和双相机视频。

后续路线：先用归档中t=699至794的真实几何、risk对象和RRT细节确定是否可提供不同Pot1接入路径；不复用相同风险边界失败路线，不拼接不同前缀。只有获得不同且有实据的可行路线后，才从严格恢复的同链Runtime Snapshot继续剩余Pot 1任务；Round 0完整种子仍需从安全t=10状态进行同Episode连续执行和全链统一复验。

## 前一总体状态：t=590→649 Pot 2局部安全放置成功；完整双壶任务未完成（2026-10-25）

从同一Episode的真实t=590连续状态继续，经严格t=642 Runtime Snapshot恢复、7个新OSC_POSE动作和每步25子步滚动认证，到t=649首次取得原生`On(moka_pot_2, flat_stove_1_cook_region)=true`，并通过5个连续样本的稳定放置验收。t=590→649共59个实际动作、1,475个完整物理子步，Native/Operational安全均通过，禁触、自碰撞、未解析/未许可接触及关节限制违规为0。Pot 2非手指间隙最低4.242 mm，机器人到Ginger Beer/Place Box最低32.678/52.850 mm；指尖最大合法穿透0.282 mm。Strict-10mm仅为附加统计且未通过，不影响本轮冻结的Operational门槛；没有安全阈值、qpos比较或BDDL成功标准更改，没有重复RRT。

本次诊断修复了预测终态几何策略缓存和release门控：预测状态按实时抓持重绑，恢复时回到精确源Snapshot；release阶段仅四个指尖Geom与Pot 2维持−2 mm允许对，其他机器人Geom和禁触对象持续受原约束。Native On由Pot 2与原生stove支撑接触后成立，而非释放前置条件。

验收边界：Pot 2局部放置及稳定成功；原生完整任务仍false（Pot 1 On=false、Stove Turnon=true、reward=0、done=false）。t=649处于标准1000步和demo-only 1800步内的是Pot 2局部里程碑，**完整Benchmark成功与完整扩展示范成功均false**。不构成Round 0完整安全种子，也没有Pot 1动作或LoRA训练。完整种子下一阶段仍要求从兼容的原始安全t=10 Snapshot开始，在同一Episode中连续执行和统一安全复验；不能拼接此次t=590局部安全轨迹补齐未知前缀。后续若先做Pot 1局部阶段，应严格恢复t=649并根据Native Pot 2谓词重建任务阶段。

入口：`research/gs_opd/repair/execute_t463_execution_aware_pot2_placement.py`。本轮归档：`/opt/data/private/gs_opd/rollouts/l0t2_execution_aware_pot2_release_continue_t642_20261025/`；总审计`posthoc_acceptance_audit.json`，原始视频、OA、逐子步安全和接触身份、Snapshot均在目录中。t=590→642连续前缀证据保存在相邻的同链运行归档中；59动作/1,475子步整体回读审计0项监测违规。1093步历史示范、Episode与LeRobot数据保持不变。

## 最新总体状态：t=463安全搬运到t=589；t=590提交比对失配后停止（2026-10-24）

复用已归档t=398的193点路径，以新执行感知入口从严格恢复的t=463 Snapshot做短段MuJoCo认证。无新RRT。126个动作连续到t=589，每动作25子步以及预测/实执行比较均通过；Native/Operational Safety成立，Pot 2双指持有，阶段控制器自然切换至precise placement。严格t=589 Snapshot恢复、Integration State和controller/gripper内部状态核对均精确通过；运行时状态、模型与当前策略attestation一致。

t=590只执行1个precise-placement动作。其25个物理子步Native/Operational监测通过：无禁触、无自碰撞、无关节限制违规，finger-only抓持仍成立。随后物理状态与分支预测在EEF位置的最大绝对分量上差`8.663630374972087e-05 m`，超过未变的`1e-8 m`严格提交比较限值。系统按fail-closed停止，没有后续动作或RRT。差异不是已检测安全违规，但阻止把t=463→590认定为已验证执行安全后缀。

t=590原生Pot 2 On仍false、reward=0、done=false；Pot 2已经移动1.343 mm但未放置。累计动作127/450，标准1000步及扩展1800步成功均false；Strict-10mm继续只做附加统计并失败，其他几何和安全阈值不变。Round 0种子与训练均未开始。归档含两段OA、双相机视频、真实Snapshot、实际journal、子步监测、阶段与谓词，并记录原runner未能在提交前持久化预测Snapshot这一证据缺口。执行入口现已在提交前保存预测数据供后续诊断；本次执行未因此重启。

**当前下一步：** 只读定位t=590分支与实执行的EEF状态差异；在解释偏差前不继续物理动作、不放宽1e-8比较、不运行RRT。后续任何续跑都从严格验证的真实Snapshot、按新的授权和剩余动作额度进行。即使后缀完成，也不得补接缺失的安全前缀当作完整Round 0种子；完整安全种子仍要求从已验证原始安全t=10开始同一Episode连续执行及统一安全验收。

完整结果与复现材料：`/opt/data/private/gs_opd/rollouts/l0t2_execution_aware_pot2_resume_t589_20261024_231708/t463_to_t590_stop_report.md`、`continuation_chain.json`。正式入口：`research/gs_opd/repair/execute_t463_execution_aware_pot2_placement.py`。历史示范、LeRobot数据、Episode归档保持不变。

## 前一总体状态：t=463偏差已归因；隔离OSC安全预演通过8步但不构成路线认证（2026-10-24）

7D RRT→OSC_POSE重复偏差已由104个历史实际状态定量分析：路径最近qpos点通常比`plan_cursor`落后4点，但最近点仅减少平均误差14.6–18.6%；t=398/t=463分别只解释末端0.0084/0.0061 rad。真实EEF仍贴近参考，而同末端附近的冗余关节qpos相差约0.174–0.176 rad。源码确认Bridge只把RRT点的EEF pose转成OSC动作、qpos只用于事后误差；robosuite nullspace项回归到controller构造时的`initial_joint`，Recovery cursor只按动作数递增。主要机制是7D关节目标与6D任务空间执行接口不一致，nullspace姿态回归促成冗余解差异，cursor滞后为次因。

隔离t=463 Snapshot严格恢复后，在独立Bridge以诊断模式运行8个OSC_POSE动作、200子步，Pot 2仍双指持有并移动15.914 mm；全子步Native/Operational安全、EEF跟踪、关节、禁触对象间隙和状态恢复均通过。qpos偏差从0.184203增加到0.192083 rad，正式0.18阈值仍冻结；这证明短段执行感知的MuJoCo分支安全预演值得作为下一步选项，但不证明余下全路线可执行。Native Safety和Operational Safety通过，Strict-10mm仍失败（Pot 2非手指间隙4.363 mm）。没有后续正式Episode动作、放置或任务成功。

下一阶段控制决策是有条件的：若7D RRT路径仍要求qpos参考限值，则需要成熟关节空间控制器；若沿用OSC，则必须在获得新明确预算后采用有限短段闭环与全状态MuJoCo隔离认证，不能把qpos阈值diag-only扩展到正式执行，也不能降低任何碰撞/关节/BDDL边界。当前继续暂停物理动作与RRT调用，先基于本轮结果评估执行模式。L0T2完整种子、Pot 2放置、Pot 1阶段及LoRA仍未开始。

诊断与复现入口：`research/gs_opd/repair/diagnose_t463_osc_path_reference.py`；归档位于`/opt/data/private/gs_opd/rollouts/l0t2_round0_t463_osc_tracking_diagnostic_20261024_223003/`。历史t=329→463物理轨迹与规划记录保留原归档，不修改1093步历史示范或LeRobot数据。

## 前一总体状态：单次重规划执行至t=463后关节路线偏差触发安全停止（2026-10-24）

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

## 最新里程碑（2026-10-26）：L1T2从reset到稳定释放的完整物理成功经验已重建

run19→run46→run47→run48→run52→run53的1335个原始动作按timestep连续接在新reset后执行，再续接77步经物理验证的稳定释放后缀。1412步、1413双相机Observation、35300个MuJoCo子步、各阶段源边界Snapshot与hash均归档。t=0 reset bitwise一致；六个动作段末Integration State和源Snapshot数值差最大`1.2862e-10`，按报告记录的`1e-7`容差通过，SHA不相同的情况如实保留。

原生任务在t=1335首次成功；done=true后同一实例继续77步demo-only horizon执行至t=1412，未改BDDL或done定义。这属于扩展示范/诊断协议，不应标作标准1000步Benchmark episode。释放、退爪、500子步Settling后Pot1/Pot2 On与Stove Turnon全真；Pot2没有可测位姿变化、末态无夹爪接触，Settling子步也全部无夹爪接触。释放后退爪期间仍有678个真实夹爪–Pot2接触子步，完整记录未过滤。

数据位于 `/opt/data/private/gs_opd/rollouts/l1t2_success_episode_reconstruction_20261026_attempt2/`，可作为后续LeRobot转换的Observation–Action物理成功Episode来源；`safety_certified=false`，不能直接用作安全训练数据，必须后续单独经过Safety Reconstruction及完整验证。本轮没有启动训练、转换或其他场景搜索。

### 当前建议路线

1. 保持已重建L1T2 Episode为只读来源，先完成其Observation/action对齐、接触标签和来源链的独立离线复核；转换时保留done后77步的扩展horizon协议元数据。
2. 仅把任务成功作为Task-Success候选；由既有严格Safety Reconstruction独立生成/拒绝安全后缀，不降低接触和安全余量规则。
3. 安全认证与数据格式复核完成后，再由用户安排LeRobot转换或下一跨场景迁移；不把本次成功解释为标准horizon成功或安全Episode。

## Round60 RRT→OSC执行接口（2026-10-26）

已修正Keypose刚体Pose Record在改写`position_m`后矩阵平移未同步的问题，并加入旋转/平移一致性回归。RRT→OSC执行链现在耐久归档完整7D qpos参考、原始EEF路径、压缩Waypoint及源索引，以及每一步OSC目标和动作前后qpos。

对Round60 t=2589精确Snapshot执行了一次18步真实短段到t=2607。因原始RRT `qpos_path`未归档，本次以同状态、seed和配置重新生成单条路线；新动作与原归档前18步完全一致，但不能逐元素比对原始RRT数组。短段出现EEF逐渐偏离最近RRT参考，t=2607请求与实位移近乎正交，但整体位移反向及后续停滞根因仍unknown。Hand–Pot2接触从起点存在，450个子步深度/接触力没有恶化。

下一步先实施局部OSC稠密路径跟踪与单调进度投影，记录实际7D状态相对规划路径的偏差；若该策略仍离开规划走廊，再评估基于真实动力学的关节路径执行。暂不扩展候选搜索、场景、训练或改变安全判据。详细机器证据位于`/opt/data/private/gs_opd/diagnostics/round60_rrt_osc_interface_20261026_185951_481254/`。
## Round60执行停滞诊断更新（2026-10-26）

L2T2 Round60的41步真实动作与1025个子步接触已完成只读诊断。Round60/61/62仍没有新的完整成功轨迹；本轮没有增加RRT候选、物理搜索或训练。首个明显控制失配在t=2607：执行器命令的平移与真实EEF运动方向开始冲突，随后24步的真实位移均值0.742 mm，23/24步方向相反。

优先级调整为先补齐“规划路线—物理执行”证据链：Round60代码路径会把RRT `qpos_path`转成压缩EEF waypoint并交给独立OSC跟踪，分支归档只保留规划摘要，没有保存选中关节路径或逐动作局部目标。关节限位违规为0；Hand Geom 74—Pot2 Geom 117非指垫接触从起始Snapshot持续存在，且停滞时穿透/法向负载未增加，不能据此判定它为直接根因。后续应先持久化关节与EEF参考、Waypoint映射和局部误差，再针对性验证路线跟踪；不放宽安全约束，不用重复增加同类搜索轮次替代诊断。

Round60诊断机器报告和逐步指标见归档`evidence/round60_stagnation_diagnostic.json`及`evidence/round60_stagnation_diagnostic.md`。准确局部Waypoint误差、实际qpos相对RRT路径偏差和Hand–Pot接触的因果影响仍标为unknown。L1T2完整稳定Episode继续保留为已验收的GS-OPD经验来源，且不视为Safety-Certified数据。

## Human L0T2执行状态更新（2026-10-26）

固定360步运输截断已由路线长度与剩余demo horizon/后续对象预算取代，并增加限位OSC恢复与合法Snapshot回退。单次Human L0T2续接未完成任务：失败t=575 Snapshot的joint7越限0.000206368 rad；8个真实OSC动作仍剩0.000127443 rad越限，随后准确回退至同场景合法Pot1持抓Snapshot。RRT接续规划12秒超时，没有物理运输动作；Pot1仍未放置，Pot2阶段未运行，原生任务完整成功为false。启动参数请求8秒但未传入主运输RRT，运行值为12秒；接口传播已修复，未为此重跑。未追加搜索。

后续实现应优先改进任务等价目标区域的RRT候选与失败分类/预算，避免将规划超时变成物理动作空转；关节状态必须始终来自真实合法状态，继续使用MuJoCo反馈与BDDL完成判据。保留本轮8步动作、200子步样本、接触日志、双相机视频和t=215最佳Snapshot作失败前缀。子步几何门禁标记incomplete，严格安全认证仍独立执行。定向测试37 passed及6 subtests；详见`GS_OPD_PROGRESS.md`及归档`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_transport_budget_recovery_20261026_retry2/`。


## Human L0T2任务区域进展运输尝试（2026-10-26）

本轮将固定高位Preplace/RRT前置门槛改为通用的物体坐标系运输路线：从Pot当前真实位姿与BDDL目标Region几何生成自适应抬升、区域内短段和最终放置目标；L0T2/L1T2历史Keypose只作软侧向参考。运输候选经真实OSC_POSE在MuJoCo中执行，不写入机器人或物体状态；RRT/IK不是终止门槛，完整成功仍由原生BDDL判定。最多两个物理分支均从已核验的合法t=215 Pot1持抓Snapshot开始。

结果：原生task_success=false，新增完整Task-Success Episode为0。Pot1仍双指稳定抓持但尚未放置；最佳上抬32.85 mm、XY位移2.34 mm，原生Pot1 On=false；Pot2阶段未开始；Stove Turnon=true。两分支共执行36个真实OSC动作、900个MuJoCo子步，无RRT调用。每支18步后都在自适应抬升Waypoint退出，原因为局部18步目标预算耗尽，而Pot和EEF仍实际上升约32.8/33.6 mm。两支共享初始抬升段，第二支的L1 Keypose侧向路线尚未物理执行，因此本轮没有证明替代路线有效。

物理分支后修正了Waypoint动作预算：现在使用运输路线剩余预算，允许实际OSC持续到子目标完成，并由真实停滞检测结束；第二候选路线也改为第一Waypoint同时上抬和侧移。该后置修正通过代码回归检查，但**本轮没有物理验证**。本轮双分支预算已耗尽，不追加相似搜索。Safety Bridge记录了36/36动作对应的900个子步样本，但几何监测均标记incomplete；不构成安全认证。

批处理入口曾在首个阶段启动前无物理动作地停滞，保留的t=0 Snapshot仍在归档中。后续代码复核发现剩余分支预算保护错误包住正常调度代码，现已纠正缩进并通过批处理单元回归；该控制流缺陷不能单独解释之前观察到的CPU停滞。

完整JSON及可读摘要：`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_region_progress_osc_20261026_run1/human_l0t2_region_progress_attempt_report.json`、`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_region_progress_osc_20261026_run1/human_l0t2_region_progress_attempt_report.md`。分支视频、动作/状态NPZ、逐步接触记录和终态Runtime Snapshot均按JSON清单保存。定向验证：39 passed、6 subtests passed，三模块py_compile通过。

**结论：本轮运输策略的物理尝试失败，主要已证实阻塞是过短的局部Waypoint动作分配；修正后的预算和侧向路线尚未在物理上验证。保留t=233可恢复Snapshot与失败证据，不启动Pot2，不训练。**


## Human L0T2 Region-Progress OSC端到端验证（2026-10-26）

从已验证合法的Pot1双指持抓Runtime Snapshot精确恢复：t=215，Snapshot SHA-256 `48b0c9faae531e1fd4654d2daaeb7f31fd315b86fb033f2ea8cbabe122bfa85a`（原始归档SHA为 `48b0c9faae531e1fd4654d2daaeb7f31fd315b86fb033f2ea8cbabe122bfa85a`），模型SHA `633b8b0956f2f38e7f1c62067608f42743d0bccdc6bfeaca11976b5c7f301606`，BDDL SHA `2c4010e44700b0d2069dcd47f489a7411ec364f31cfd69141801b49db3398af7`。启动参数使用seed=0、init-state=0、demo-only horizon=3600。批处理首阶段增加了精确Snapshot参数传递；运行证据确认以原有持抓状态接续，没有重做抓取、Student动作或RRT预检。

执行了两条实际不同的Snapshot运输路线，均从同一t=215状态分支：直接区域进展路线107个OSC动作至t=322；使用L1T2 Keypose侧向参考、第一Waypoint同时抬升和侧移的路线78个OSC动作至t=293。Pot1双指抓持在两支全程保持，共185个真实OSC动作、4625个MuJoCo子步样本，RRT调用0。两支终态均未满足Pot1 `On`，Pot2 `On`为false，Stove `Turnon`为true，原生谓词1/3、task_success=false。没有真实松爪、稳定释放或Pot2阶段；新增完整Task-Success Episode为0。

真正的物理阻塞是两条路线都进入番茄酱瓶障碍物的Hand接触：Geom 74 `gripper0_hand_collision`与`bottle_of_tomato_paste_2_with_hand_1`拆分几何发生碰撞。直接路线首次在t=307记录接触、持续到t=322，最深穿透约0.203 mm、接触力范数最高约7.30 N；侧向路线在t=282首次接触、持续到t=293，最深穿透约0.302 mm、力范数最高约3.70 N。接触后OSC移动减慢；各自停滞窗口末段Pot1到目标Region中心仍缩短约5.0 mm和3.2 mm，之后原12步EEF进展检测结束分支。接触因果与停止同时出现，不能据此证明唯一因果，但两支都存在明确真实障碍接触，不能把该结果描述为纯粹IK/RRT或任意跟踪误差失败。

结束后只做了最小执行门槛修正：Region-Progress OSC的现有停滞检测窗口从12步延长到24步，累计6 mm判断及其余限位、MuJoCo物理响应、动作预算和失稳保护保持不变。该代码修正未在本轮物理重跑；本轮两条分支预算已用尽，按要求停止。回归检查：批处理与执行器测试42 passed、6 subtests passed；两个修改模块py_compile通过。

机器摘要已按两个分支的耐久动作记录离线纠正：185个真实动作、4625个子步、6387条控制边界接触记录、185步的子步完整性标记均为incomplete；不构成安全认证。两分支各自的Observation/Action NPZ、双相机MP4、25子步接触记录和失败Runtime Snapshot均完整保留。综合报告：`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_region_progress_validation_20261026_attempt1/batch_report.json`、`batch_report.md`；任务级证据：`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_region_progress_validation_20261026_attempt1/obstacle_avoidance_human_L0T2/task_report.json`。最佳目标Region进展Snapshot在直接路线t=322终态；侧向路线t=293终态也单独保存。

**结论：任务失败并在2条分支后停止。首个实现修复确实消除了18步预算截断，但真实番茄酱瓶接触减缓了Pot1运输，12步停滞判断仍过早结束低速目标进展；24步门槛修正尚未物理验证。没有启动Pot2、没有完整成功Episode、没有安全认证或训练。**

## Human L0T2规划路线切换：完整MuJoCo几何RRT两候选超时（2026-10-26）

已把当前运输调用从无障碍Region-Progress路线切换到仓库已有MuJoCo 7D关节空间RRT。CuRobo现存封装用于抓取提案，并不直接承载LIBERO-Safety活动障碍场景，故本轮选用会查询当前编译模型全部机器人及场景Geom的MuJoCo适配器。任务搜索RRT采用零额外间隙的真实碰撞约束；原严格安全策略和认证标准未修改。持抓Pot几何通过活动语义绑定加入规划，固定安装接触只允许已核验的具体Geom对。

自合法t=215状态仅尝试两个不同原生Region位置/入口，各调用一次30秒RRT；二者均因`planning_time_budget_exhausted`停止，合计实际OSC/Student动作0，物理子步0，Pot1未运输/放置，原生`task_success=false`。两支起点与失败Snapshot字节一致。规划器详细迭代/配置检查计数未持久化，标注unknown。无成功Episode、动作视频或LeRobot可训练数据。因没有RRT路径，OSC跟踪兼容性仍未验证；直接关节执行会改变π₀.₅动作标签表示，本轮未尝试。

实现和报告修正包括：持抓Geom加入所有RRT运输路线、任务成功规划用零余量碰撞约束、MuJoCo规划器尊重审计过的固定安装接触对、保持完整RRT EEF路径并归档7D qpos与跟踪引用、两个Region候选都使用同一RRT规划器、批处理正确计数候选级RRT调用。测试53 passed、6 subtests passed。归档路径见`GS_OPD_PROGRESS.md`本节。

**状态：Human L0T2任务未解决。两候选预算已用完，本路线停止；下一次采集前应先获得规划器超时的可审计配置检查/阶段耗时证据，并解决搜索预算内规划不收敛问题。不能把规划失败解释为OSC控制失配，也不能宣称避开了番茄酱瓶。**


## 2026-10-27进展：CuRobo有效轨迹长度已修复，Human L0T2仍阻塞

**已完成：**CuRobo固定缓冲轨迹按`interpolated_last_tstep`有效点数执行；保存有效时间/位置/速度/加速度等计划状态；不抽样地逐点跟踪，有需要时按动态限制进行时间重采样；释放检测在不足10步的最终预算处也执行；为关键阶段保留3600步全任务预算。

**物理验证：**t=215旧预放置路线身份精确匹配并复用，1800缓冲仅21点有效，当前真实执行20步而非1799步。随后从真实t=235状态生成的新放置输入在规划前被门禁拒绝：attached-object sphere与番茄酱瓶主体mesh包络的代理重叠为6.735 mm；17/24目标终态无代理碰撞，但起点不清。500个物理子步未记录对应瓶体接触。Pot1未松爪、Pot2未启动，`task_success=false`，新增完整成功Episode为0。

**当前状态：**代码修复通过单次物理验证，但Human L0T2端到端目标未达成。具体运行、诊断、动作、双相机视频和Snapshot归档于`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_curobo_transport_20261027_020247/`。当前工作停在t=235的CuRobo预检起点碰撞，下一轮任何执行须先基于此归档判断代理重叠与真实状态差异；本轮不再运行搜索、不训练、不更改规划器或碰撞模型。

## 当前路线调整（2026-10-27）：L0T2代理误碰撞已排除，真实松爪接触仍阻挡退出

任务搜索优先级保持跨场景Episode采集；但Human L0T2此分支已停止。新证据将两类接触分开：t=235 Pot1盒体与番茄酱瓶真实间隙37.498 mm，旧单球代理误报6.735 mm；保守覆盖的8球表示有21.728 mm正间隙。该精确代理修正后CuRobo路径实际执行到Pot1原生On=true。

目前剩余阻塞不是瓶体运输，而是夹爪尚未离开Pot1。t=295逐步接触证明Hand Geom 74与Pot1 Geom 98最深穿透1.015 mm、最大力23.154 N，左右指也继续接触；CuRobo从此配置拒绝规划撤离。释放核验已修改为要求夹爪完全打开、左右指无接触且MuJoCo接触记录无机器人—目标物体接触。后续如重新打开该场景，优先从t=295合法Snapshot解决真实脱离与无碰撞撤离，不重新跑t=215抓取/运输，也不删除Pot1世界几何。

本轮物理状态：Pot1 On=true，Pot2 On=false，Stove Turnon=true，`task_success=false`；未稳定Settling，无完整Episode。80步来源链和2,000子步接触、双相机视频、t=295 Snapshot、CuRobo失败输入/输出与接触审计均在`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_curobo_t285_release_and_pot2_20261027_131000/`。关节扭矩动作与π₀.₅ OSC监督不兼容，安全认证独立待做。
# 2026-10-28 attempt27安全语义归因

attempt27达到新的原生任务成功（Level B），但没有安全认证。当前执行器的 `safety_policy_diagnostic_only.json` 将每个非目标对象都当作 forbidden；活动 BDDL 仅明确禁止机器人接触香水瓶/白色置物盒及黑碗接触这两者，并把炉灶支撑列为初始 `On`、盘子支撑列为目标 `On`。因此诊断策略的碗—炉灶与碗—盘子标记不是正式安全结论。Strict-10mm 已由实际非指垫接触失败：t168 finger1—碗穿透约0.183 mm，终态 finger2—碗间距约0.244 mm。路线未触碰香水瓶。下一步先做以真实手部碰撞几何筛选为准的离线不同抓姿候选，并为未来物理分支读取精确 BDDL 安全约束；不得将诊断记录升级为认证，也不得改变10 mm/2 mm规则。详见 attempt27 `safety_attribution.json` 与 [当前目标](GS_OPD_CURRENT_GOAL.md)。


## 2026-10-30 15:35 CST live Stage A checkpoint

当前优先级仍为恢复 OAH/L0/T0 的 reachability-biased placement 续跑：精确从已审计 t246 Snapshot 延伸到 t303，新增57动作，原生谓词暂未满足但目标距离下降约74 mm且持续抓持。无需改变全局搜索路线；此分支终结后优先修复 task-feasible 候选筛查的 pass-first/fallback 排序，再回到未覆盖任务。
