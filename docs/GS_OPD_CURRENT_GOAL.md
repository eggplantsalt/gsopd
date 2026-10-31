# SafeTRAIL Stage A 当前目标与局部进度（2026-10-31）

`GOAL_STATUS=ACTIVE`。唯一验收仍是30项任务都具备合法t0来源、真实MuJoCo连续动作、原生BDDL成功及可审计完整Episode。目前核对为 **18/30：12 Standard、6 Extended**。Stage A不以Stage B禁触约束阻断任务成功；超过1000动作的成功单列Extended。

## 当前局部目标：OA/L2T3 验证 Safety Bridge 后完成双物体 In

此前两次合法t0冷启动均在首个 `step_monitored` 请求收到 EOFError，未保存任何动作来源，物理效果未知，不能算成功或失败Episode。本轮从任务自己的合法t0 Snapshot `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_L2T3_cold_start_dev_02/t0_capture/stage_snapshots/t0` 重新启动当前执行器；当前输出根为 `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261031/obstacle_avoidance_L2T3_fresh_legal_t0_dev01`，horizon 1800，contact-only子步审计，跳过Student-first，最多一个物理分支。预期首先确认真实MuJoCo动作与审计记录可写；之后执行alphabet_soup与cream_cheese的原生In目标。t≤1000才标Standard，超过则单列Extended。运行中的报告未结束前不增加覆盖数。

### OAH/L1T4暂停记录：水平入口处抓持手掌顶住微波炉

任务仍未成功。围绕t921合法同Episode Snapshot开展了接触法线退让、入口高度、侧放、入口暂存与松爪试验。45mm抬升让杯子上沿撞到`microwave_1_g6`；90°侧放路线增大编译竖向碰撞包络并逼近腕关节6上限。此前入口暂存错误使用整个微波炉0.336m包围半径，导致外侧路点距离0.611m；已改用原生入口边界、物体足迹和实测手部外伸量，缩短至0.179m。20mm与8mm入口偏移都到达门口，但`t1045/t1047`的MuJoCo记录显示 `gripper0_hand_collision` 与 `microwave_1_g6` 接触约33N并阻止插入；32步真实松爪使杯子落在柜前台面，原生In仍false。当前顶抓EEF相对杯根高约89mm，入口中的手掌空间是主要障碍。dev16低位抓姿候选在卡住的当前姿态无可达IK。后续需更换抓持姿态或优先切换任务；所有这些分支均未计成功。归档根：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261031/obstacle_avoidance_human_L1T4_offset24_horizontal_aperture_dev01/continuations/`。

### OA/L2T4 微波炉路线暂停点（t884，尚未原生成功）

从原合法t0 Episode的t597抓持快照开展三次同链续接，当前串接前缀为t1–884：原选定t1–597 + dev06 t598–715（118步）+ dev07 t716–805（90步）+ dev08 t806–884（79步）。每个起点均精确Runtime Snapshot恢复；后3段使用contact-only物理审计，25/25接触子步/动作完整，几何全扫关闭，不作安全认证。终态t884 mug仍真实双指抓持，In/Close均false。dev05、以及与选定续接不兼容的分支尾部均不并入。

失败假设逐步收敛：dev05证实顶面支撑物应先垂直抬起（11步，杯子由1.109升至1.156m且保持双侧抓持），再水平移动；dev06按编译微波炉包络延长外侧点后，证实高位OSC路点可达性不足；dev07从t715以较低实时高度续接，将杯子推进至约y=0.589m，记录到`gripper0_right_gripper–microwave_1`接触；dev08从t805继续，终止于t884的`region_progress_waypoint_stagnated_for_24_actions`，末24步无fixture接触、无关节限位但EEF/关节不动。完整失败报告与Snapshot链在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261031/obstacle_avoidance_L2T4_t597_support_lift_first_dev05/`、`.../obstacle_avoidance_L2T4_t597_fixture_envelope_staging_dev06/`、`.../obstacle_avoidance_L2T4_t715_low_height_exterior_continue_dev07/`、`.../obstacle_avoidance_L2T4_t805_measured_hand_escape_continue_dev08/`。暂停重复direct-region路线；保留dev08的t884真实Snapshot，后续须先换腕姿/局部可达控制，再续接。该段为旧检查点；当前局部目标已切换至OAH/L1T4合法t0有界物理抓取验证。


已暂停OA/L2T3 soup路线族：合法t432 Snapshot中的cream cheese原生In=true；geom77宽夹持候选高位路线在clean t432起点仍由robot0_link6碰到beer瓶，geom79到达目标但约3.54mm夹持宽度只形成右指擦碰。通用控制已改为表面路线遇到任一目标指垫接触即停，并在闭爪时反馈保持接触EEF姿态；该修正把一次分支从255步缩至29步、将闭爪漂移由约18mm降至约4mm，但geom79仍未能形成双侧夹持。t432合法部分Episode、失败报告与Contact Snapshot保留供后续更换接近侧时续接。OAH/L0T4的-90° yaw已在dev16实测到t1521仍In/Close=false；dev21关节余量路线t1533也失败，避免照旧矩阵计划重复这两支。

OAH/L0T4暂时停放。dev35从同一Episode t1640真实执行39步至t1679；mug双侧抓持保持，EEF与mug向微波炉方向移动约45mm，但最后14步mug仅移动约2mm，最终因`region_progress_waypoint_stagnated_for_24_actions`停止，In/Close均false。接触记录只见指垫—mug，没有微波炉接触。该路由族已尝试多种yaw、入口、绕行和低位模式，本轮转向新任务，不增加同一控制预算。

## 前一突破：OA/L2T2 t1370 Extended 原生成功

合法t0同一Episode完整执行1,370动作、34,250个受监测MuJoCo接触子步；pot1/pot2均`On=true`，stove`Turnon=true`，BDDL`task_success=true`、reward=1、done=true。t1367 Snapshot中的pot2仍单侧接触；物理闭合/重抓后三步成功。成功轨迹及每段hash和Snapshot边界：[manifest](/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261031/obstacle_avoidance_L2T2_t1367_live_contact_regrasp_low_clearance_dev_17/task_success_first_executor_obstacle_avoidance_L2T2_20261031_090038_778150614/selected_success_episode_manifest.json)。超过1000动作，只计Extended。

## 历史检查点：OAH/L0T2 Extended 成功

`obstacle_avoidance_human/L0T2`在t1547原生成功：两只moka pot均满足On，炉灶满足Turnon，`task_success=true`、reward=1、done=true。合法t0 Episode的t667 Snapshot后精确续接，成功路径覆盖t1–1547，共1,547动作，超过Standard上限，只计Extended。冷启动、原生终态、逐段动作记录哈希及t667/t901/t991/t1016/t1105/t1312/t1355/t1446/t1547 Snapshot链见[success_episode_manifest.json](/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_human_L0T2_cold_start_dev_01/v3_run/continuation_stages/stage_17_t1446_reachability_yaw90/task_success_first_executor_obstacle_avoidance_human_L0T2_20261031_063716_085548457/success_episode_manifest.json)。Stage17在原生成功后记录抓持丢失异常；t1547 BDDL终态仍成功，终端Snapshot保留旧`failed_runtime_snapshot`文件名，清单中有说明。

这次成功使用严格预抓末端、首个真实目标指垫接触后的有界闭合探针、碰撞证据驱动的可达净空、目标x/y显式对齐后分段下降，以及On目标的yaw冗余搜索。之后在相似任务复用这些改动；不再重复OAH/L0T2已失败的geom117/116预抓路线。

## 已停放任务：OAH/L0T4 从 t=1490 Snapshot 解除过高清障与腕关节停滞

同Episode t1490 Runtime Snapshot：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T4_t1380_joint7_unwind_entry_dev_12/v3_run/task_success_first_executor_obstacle_avoidance_human_L0T4_20261030_212116_436021926/expert_native_goal_stage_control/placement_branch_00_region_layout_15_direct_region_progress/native_goal_transport_failed_runtime_snapshot.pkl`，SHA `29221ef221ae6c7ba4a3be4630e843cb7fdda30e2496dcf0f9dd0ca43ba2a3af`；原合法t0根在同一V3 run 的 `t0_capture/stage_snapshots/t0`。mug仍被真实双指夹持；In/Close均false。t1379附近joint7在下限外仅0.00034rad，先用真实OSC动作恢复；旧路随后估算25° aperture tilt所需清障高度为1.283m，把mug抬得远离目标，至入口姿态 waypoint停滞，终态joint7又回到下限附近。

dev13低位upright/+90°路线未生成1.283m清障抬升，但t1524停滞。dev14收紧死区后，真实OSC动作将joint7的0.000141rad超限恢复至范围内；下降仍在t1523将其推回下限并停滞。原因是水平aperture路点顺序先下降、后转yaw。现已改通用In路线：非staged低位路线上，先在当前位置真实转至目标aperture朝向，再下降/横移。将从同一t1490 Snapshot单分支验证该顺序。若原位转姿不跟踪或下降仍回到腕限位则暂停微波炉任务。后续成功只计Extended。OA/L1T4已暂停：t1239附近反向/upright三次24步测试均未生成预期escape waypoint，mug与原生谓词状态未变；该容器接触发现保留为待修通用问题，不继续在同一状态重复路线。

### OA/L1T3 本轮结论：真实目标接触，但未形成抓持

初始24候选筛查0通过后，Stage A仍把IK可达但碰撞筛查拒绝的候选送进MuJoCo。geom82顶部路线在t=154产生真实左指垫接触，闭合时两侧曾短暂同时接触，但 soup 随闭合横移约7mm，最后只剩单侧。精确 Snapshot 分支确认单侧接触时 EEF 比抓持帧高约26mm；继续向下12步仅移动约2mm并停滞。随后geom82/geom92侧向候选的预抓路线也出现低位/OSC waypoint stagnation；geom92局部Y上移20mm分支未恢复运动。当前L1T3仍是两个In谓词false，不计成功。代码现区分近位单侧闭合与远位接触后继续表面接近；冷启动同Episode continuation可用当前编译几何生成±20mm局部Y位移，不引入历史动作/示范。

OA/L2T2继续暂停：已有pot2稳定抓持Snapshot，但已试路线分别碰倒炉面物体、失去抓持或在侧向waypoint停滞；回访需先形成不同的障碍绕行路径。

## 最近新增：obstacle_avoidance_human/L1T1 Standard成功

L1T1现有完整t1–981 Episode：981动作、24,525/24,525接触子步；t981原生`In(black_book_1,desk_caddy_1_back_contain_region)=true`、task_success=true、reward=1、done=true。t578/t798/t904 Snapshot分支边界精确匹配；从t904修改Stage A路线预算后77步完成任务。offset1/yaw +120°解除joint7下限附近停滞，取消已被最终插入路线覆盖的固定96步末端保留，修复预算饥饿。终端Snapshot与t981最后物理记录integration state哈希一致。未用官方演示、历史成功动作或直接状态写入；safety_certified=false。审计清单：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L1T1_contact_fast_fallback_dev_04/v3_run/selected_success_episode_manifest.json`。

## 最近新增：obstacle_avoidance_human/L0T1 Standard成功

L0T1现有完整t1–760 Episode：760动作、19,000/19,000接触子步；t760原生`In(black_book_1,desk_caddy_1_back_contain_region)=true`、`task_success=true`、reward=1、done=true。t545合法连续前缀通过精确Snapshot边界续跑215动作；最终已通过1000动作Standard上限审核。未使用官方演示、Student动作或直接状态写入，`safety_certified=false`。

审计清单：[selected_success_episode_manifest.json](/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T1_single_geometry_physics_dev_01/v3_run/selected_success_episode_manifest.json)。完整有序物理记录：[selected_success_episode_physical_step_records.jsonl.gz](/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T1_single_geometry_physics_dev_01/v3_run/selected_success_episode_physical_step_records.jsonl.gz)。过程突破是跳过8秒超时的RRT门槛，改用原生Region Progress OSC物理执行ranked布局offset1（+125°）朝向；完整Snapshot与动作审计见清单。

## 最近新增：obstacle_avoidance_human/L0T0 Standard成功

L0T0已获得完整连续Episode：428个动作、10,700个接触审计子步；t=428原生`on(akita_black_bowl_1, plate_1)=true`、`task_success=true`、reward=1、done=true。成功Episode由合法t0前缀和同一物理Episode的精确Runtime Snapshot续接组成；使用过的step-size探针只选取其t340–356范围，明确排除了不兼容的分支尾段。最终终态Snapshot与成功记录末状态哈希一致。未使用官方演示或直接模拟器状态写入；`safety_certified=false`。

成功清单：[selected_success_episode_t428.json](/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T0_t425_near_goal_release_probe_01/task_success_engine/task_success_first_executor_obstacle_avoidance_human_L0T0_20261030_161106_801186037/selected_success_episode_t428.json)。原生Executor报告：[run_report.json](/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_human_L0T0_t425_near_goal_release_probe_01/task_success_engine/task_success_first_executor_obstacle_avoidance_human_L0T0_20261030_161106_801186037/run_report.json)。

## 搜索效率改进

主要耗时根因是`step_monitored`对每次控制动作的25个MuJoCo物理子步都执行全场景Geom间隙扫描，约18,360个几何对排序；该RPC实测均值11.64秒/动作。Stage A contact-only profile 保留每个子步的真实接触样本、状态/动作、抓持和原生BDDL检查，跳过全几何扫描后实测约0.445秒/动作（约26倍提速）。JSON/fsync约0.062秒/动作，不是主要开销。该profile用于任务可行性搜索，不提供任何安全结论。

目前通用改进包括：每子步接触审计保留、跳过Stage A不需要的全Geom clearance扫掠；endpoint-IK可达的几何候选即使被筛查碰撞拒绝，也可进入有界物理分支；精确Runtime Snapshot续接避免重复抓取前缀；Student被跳过时不再调用空模型handoff；放置候选可从排序offset直接开始；单个native-region路点有界为128动作，避免一个姿态/跟踪失败吞掉剩余Episode。OAH/L0T1已验证：screen拒绝的抓姿仍可真实抓取，替代容器yaw加原生区域OSC可从腕关节停滞前缀续至BDDL成功。当前覆盖15/30（12项Standard、3项Extended），只有完整原生BDDL成功Episode才增加计数。

## 持续验收

每条计数轨迹必须同时通过原生BDDL、1000动作Standard上限、完整连续同Episode动作/状态/接触记录和可恢复Snapshot链审核。Snapshot分支只在身份和Integration State精确匹配时并回同一时间轴；安全违规可接受但必须如实记录，绝不伪造成功。

## 历史运行与任务细节

## obstacle_avoidance_human/L0T0 首轮结果：搜索预算耗尽

`obstacle_avoidance_human/L0T0`首轮V3以3600秒wall-time budget结束，`engine_time_budget_exhausted`。原生最终状态和V3 `physical_action_audit`不可用；没有证明On目标或task_success成立。只读逐源检查确认所选候选路径的214 grasp + 32 handoff + 22 placement动作均为有效7D记录且均标记完整子步监测；轨迹中无task_success=true记录。Grasp skill达到22个连续稳定双侧样本并抬升36.457 mm；placement最后记录timestep 268，任务仍未成功。32动作Student-first前缀已隔离保存。下一覆盖项为`obstacle_avoidance_human/L0T2`。

## 最近完成：obstacle_avoidance_human/L1T0

L1T0 Standard 1000步原生成功已确认：t=484时On与`task_success`均true、reward=1、done=true；V3完整审计484个连续动作、12,100个完整监测子步、阶段Snapshot边界通过。安全认证和Strict-10mm未授予。成功轨迹与冷启动指纹见本文件覆盖状态、`GS_OPD_PROGRESS.md`及机器矩阵。

## 最近完成：obstacle_avoidance_human/L2T0

`pick_up_the_akita_black_bowl_on_the_stove_and_place_it_on_the_plate`从fresh legal t0完成Standard 1000步V3冷启动原生成功：t=517时On与`task_success`均true、reward=1、done=true。V3完整审计517个连续动作和12,925个完整监测子步，7D动作/状态有效、Snapshot阶段边界通过；成功路径为146步专家抓取、32步Student handoff和339步原生放置。几何监测114 complete/403 unsafe，Strict-10mm与安全训练资格未通过，安全认证未授予；release完成后没有withdrawal/settling评估。身份指纹、executor与V3路径见`GS_OPD_PROGRESS.md`顶部和机器矩阵。

## 最近首轮失败：obstacle_avoidance_human/L2T1

L2T1从fresh legal t0完成Standard冷启动，但24个抓取候选0可行（15碰撞约束失败、9 IK不可达），未进入RRT或专家物理执行；Student-first 15动作分支排除且t0恢复，原生目标保持false。V3物理动作审计无专家动作源，详见`GS_OPD_PROGRESS.md`与机器矩阵。

## 最近首轮失败：obstacle_avoidance_human/L2T2

双moka pot任务从fresh legal t0执行到t=634，Pot1和Pot2的On目标仍false。V3物理动作审计完整（634动作、15,850完整监测子步），但多个原生运输分支中失去双侧抓持，未形成成功Episode；几何unsafe 482，安全认证未授予。完整终态、身份与分支审计见`GS_OPD_PROGRESS.md`顶部及任务矩阵。

## 最近首轮失败：obstacle_avoidance_human/L2T3

首目标alphabet soup抓取候选24项均未通过（13碰撞、11 IK不可达），t0精确恢复，无专家物理动作；两项原生In目标保持false。详情见`GS_OPD_PROGRESS.md`顶部与矩阵。

## 当前公共瓶颈诊断：Native Region Progress运输与放置

30个任务配置均已有首轮真实冷启动证据或完整成功结果；当前共有9/30配置至少有一条成功轨迹（7 Standard、2 Extended）。近期失败中，OA/L1T4与OAH/L0T4/L2T4出现native-region waypoint stagnation；OAH/L2T2在运输中失去双侧抓持。L1T4已在当前world-up代码下完成一次重试，三条原生放置分支仍停滞；暂按覆盖优先转向`obstacle_avoidance/L2T1`，再结合L1T4及OAH/L0T4/L2T4、L2T2的动作、目标路点、抓持监测和分支恢复报告继续修复公共瓶颈。避免为单个场景写死抓姿、坐标或路线；多分支探索不构成成功Episode。详见`GS_OPD_PROGRESS.md`顶部、任务矩阵和各自V3归档。

## obstacle_avoidance_human/L0T2结果：Pot2候选池耗尽

`put_both_moka_pots_on_the_stove`从fresh legal t0（init-state 0/seed 0、source=`fresh_reset_no_prior_experience`）启动Standard V3 cold-start。Stage1由187步专家抓取、32步Student handoff和448步原生放置组成，共667动作/16,675子步；t=667时Pot1 On=true、Pot2 On=false、Stove Turnon=true、整体`task_success=false`。V3接受的Stage1 partial Episode审计为667/667动作、16,675/16,675子步，连续7D状态—动作记录与Snapshot边界核验通过；几何监测191步complete、476步unsafe，安全认证未授予。

从Stage1精确settled Snapshot续接Pot2，payload SHA `314dc362ab41a56fb8718072e683ce72693f900424f3eee9210d2f5a34d32dee`、integration-state SHA `81e95a72c570bd05fc3ea980d6d6766a0ac36f19a4eb2e181198392ffc95c75d`；两次恢复均exact verified。Pot2两个互不重叠的24候选池全部筛完：offset 0为0可行（20 collision constraint failure、4 IK unreachable），offset 24为0可行（19 collision constraint failure、5 IK unreachable）；无RRT或专家物理动作。V3将两次失败尝试排除，分别含28和29步Student-first前缀；原始t0尝试的18步Student-first也排除。Pot2搜索共48个候选无可行解，整体V3状态`task_incomplete`，`multi_goal_episode`为stage_count=1/attempt_count=3，不能计整任务成功。

按30任务覆盖优先策略暂停L0T2，保留Pot1部分Episode、筛查与恢复证据；待公共候选生成或多目标顺序能力改善后再回访。未使用官方示范、历史动作或外部Snapshot。归档：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_human_L0T2_cold_start_dev_01/v3_run`；V3报告：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_human_L0T2_cold_start_dev_01/v3_run/v3_report.json`；Stage02/03筛查分别归档在`continuation_stages/stage_02`与`stage_03`。Strict-10mm与训练资格未通过，安全认证未授予。
## obstacle_avoidance_human/L0T1 首轮结果：无抓取候选通过接近筛查

`obstacle_avoidance_human/L0T1`首轮V3最终`task_incomplete`，原因是24个抓取候选均未通过open-surface接近筛查（15碰撞约束失败、9 IK不可达）。没有RRT或expert物理执行。32个Student-first动作被排除且t0精确恢复；`native_task_success=false`，V3物理动作审计因没有选定expert来源而incomplete。下一覆盖项为`obstacle_avoidance_human/L0T2`。

## L2T4 Standard冷启动未成功：placement waypoint停滞，审计完整

`obstacle_avoidance/L2T4`从新鲜合法init-state 0/seed 0 t0开始，source=`fresh_reset_no_prior_experience`；BDDL `41ebfec9f85dd3df136d94f96a901c7a3c7d7bf866a381a3f25d4e3c00773a8a`，payload `fe57a5c60029d2e10f694a9c99c7e3bd6874f8ed9f07527be2f3e70b1b384db5`，integration `a2f30db5c13b2ff76b9237f1ee37f2ca78fd63e62d5871fd870d3bb09e81a290`，model `f1569ee73898506c17c2283a9ccb8555bb25bde00df2d35390cd41616415d7b8`。V3最终`task_incomplete`，原因是`transport_region_progress_native_region_placement_waypoint_stagnated_for_24_actions`；timestep 597时In与Close均false、reward=0、done=false、`task_success=false`。所选路径由188 grasp、32 handoff、377 branch01 placement动作组成；597动作/14,925子步的V3物理审计完整，来源连续、action/state记录有效、子步监测完整、snapshot边界通过。Branch00的271动作另存为替代轨迹且不计入所选路径，32步Student-first段已排除。Strict-10mm未通过，`safety_certified=false`、`safe_training_eligible=false`。下一覆盖项为`obstacle_avoidance_human/L0T0`。

## L2T3 Standard冷启动两次基础设施中断；无可审计轨迹

L2T3 run01与run02都从独立新鲜合法init-state 0/seed 0/no-prior t0启动；两次都在首个`step_monitored`请求时发生Safety Bridge `EOFError`，V3 `status=incomplete`且physical archive sources=0。t0/reset身份核验通过；两个In目标初始均false。请求的物理效果未知，不计成功或任务失败轨迹；暂不重复L2T3，保留两份V3报告待共享桥接器诊断。下一覆盖项为L2T4。

## L2T2 Standard冷启动首轮未成功：原生放置阶段未完成

`obstacle_avoidance/L2T2`从新鲜合法t0冷启动，无目标场景历史经验。V3 `task_incomplete`、return code 2；t=664时两项Pot On均false、Stove Turnon=true、reward=0、done=false、`task_success=false`。失败原因是`transport_region_progress_native_region_placement_did_not_reach_pose_within_149_actions`。抓取技能本身已通过：geom99分支185动作/4,625子步、23个连续稳定双指样本、Pot1抬升35.940 mm；Student handoff 32/800。V3最终物理动作审计为664动作/16,600完整监测子步，7D/state-action有效、记录源连续且Snapshot边界已核验。所选源包含185抓取、32 handoff、447 placement branch01动作；placement branch00另447动作/11,175子步为替代分支，Student-first 30/750已排除。没有成功Episode；安全认证与训练资格均未通过。详细失败证据见[GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md)及机器矩阵。

## L2T1 Standard首轮失败：抓取筛查耗尽

`obstacle_avoidance/L2T1`首轮从新鲜合法init-state 0/seed 0 t0开始，冷启动校验通过。24个task-approach候选全部失败（15碰撞/提前指垫接触、9 IK不可达），未进入RRT或专家物理分支。dev02使用当前共享代码从同一合法t0独立冷启动复验，结果仍为24候选0可行（15实际碰撞、9 IK不可达），screen live state未变化，start-state SHA与合法t0完全一致。12步Student-first前缀的12/12动作和300/300子步监测完整，按流程回滚并排除；无RRT、无接受专家物理动作，原生In和`task_success`均false。V3 `physical_action_audit=incomplete`仅因没有接受专家动作来源。两轮筛查均失败后，按覆盖优先切换至`obstacle_avoidance/L2T2`；待公共抓取候选族实质改进后再回访。

## L2T0 Standard 1000步原生任务成功

`obstacle_avoidance/L2T0`从新鲜合法init-state 0/seed 0 t0冷启动，`no_prior_experience=true`；未使用L1T0或其他目标任务历史成功动作、抓姿或示范。BDDL SHA-256 `c72a1fd35d6de09eff90b217b7aa6f990743faa8be8a84af54694545d187599d`；t0 payload SHA-256 `0bd50d95927a60e56d7ceb0e184ced0da8666e961061a25ecc0b707743111156`；integration-state SHA-256 `056d32f4b945cdd676a59cc7b8e778bf484ba4e68e4bf1eaea5c6c06fd97a329`；model geometry SHA-256 `4f9b8695c2bc660d6df163727b651f7c3410dd1589036d8acd4e7dbaf50109d2`。t0初始`On(akita_black_bowl_1,plate_1)=false`。

24个任务接近候选中11个通过筛查。专家候选`geom112_jaw0_approach2_+1_flip+1__approach_axis2_-11.2mm__rrt1`验证23个连续稳定双指接触样本并抬升36.973 mm。成功Episode包含142步专家抓取/抬升、32步Student handoff和323步专家native region放置，覆盖t1–497；失败Student首段32步已精确排除。V3独立审计确认497/497动作、12,425/12,425物理子步完整，7D动作和状态—动作字段有效，子步监测全部完整，时间轴连续且Snapshot边界核验通过。t=497原生`On(akita_black_bowl_1,plate_1)=true`、reward=1、done=true，故Standard 1000步成功。

`planning_time_budget_exhausted`保留为搜索诊断，不改变V3 `complete`/return code 0及原生成功结果。383步几何监测为unsafe；安全认证、Strict-10mm和训练资格均未通过/未授予，稳定释放未评估。完整归档：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_L2T0_cold_start_dev_01/v3_run`。覆盖后的下一项为`obstacle_avoidance/L2T1`。

## L1T0 标准任务成功

`obstacle_avoidance/L1T0` 从合法 init-state 0、seed 0 的新鲜 t0 开始。初始 `On(akita_black_bowl_1,plate_1)=false`，终态在t=402变为true，原生 `task_success=true`、reward=1、done=true；因此计为Standard 1000步成功。t0 payload SHA-256为`959474a6b0ead1c2d9cd12694dfd0e80f02f12c6f176e42816ab41e7a5362185`，MuJoCo integration-state SHA-256为`9f1623b303f4d23b13e25e12c99b89b107689b65c280f543d18dbdeee2687d87`，模型几何SHA-256为`9563488607b6d99838d870f1ff980ca6bfd7630595f5504662f26cd70233cc86`。BDDL SHA-256为`85f4396e3152c27323182ac2f2a0191371e917a412667e5123e53490b33f2e7b`。

最终成功Episode由专家抓取145步、经已核验抓持抬升Snapshot继续的Student handoff 32步、专家原生目标放置225步组成；三段分别覆盖t=1–145、146–177、178–402，顺序连续。较早的32步Student首分支已精确恢复t0并丢弃，不计入成功轨迹；最终成功不能归为Student-only。V3独立审计为402/402动作行、10,050/10,050 MuJoCo子步，动作维度及状态—动作字段有效，Snapshot阶段边界核验通过。候选为`geom112_jaw0_approach2_+1_flip+1__approach_axis2_-11.2mm__rrt0`，8次RRT预检后执行；报告另保留`planning_time_budget_exhausted`搜索诊断，但不影响已报告的原生成功与完整动作审计。

这只是Stage A任务成功。285步几何监测状态为unsafe，安全认证与Strict-10mm均未通过；稳定释放未评估，`safe_training_eligible=false`。完整归档位于`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_L1T0_cold_start_dev_01/v3_run`。

## L1T1 Standard 1000步冷启动成功（dev06）

L1T1使用init-state 0/seed 0合法t0，初始`In(black_book_1,desk_caddy_1_back_contain_region)=false`；BDDL SHA-256为`a84967cdb8bd66e5e42e4504e3dd89414923b215bf0c8a97cdc0a7fb909df0f3`，t0 payload SHA-256为`1bb657a9f1d1ec1233197ae7dc2f76014ae5655bbcdcec15f4ddfee97365f961`，integration-state SHA-256为`a3f00d570ffac8f5275abbdc323738ad1f24ef203a0c7808f515dec2aade77bb`，模型几何SHA-256为`d7d12c6d41f7f6f8184e8ec4f947d7bf03c4b299478257b9f1b43295483cbc53`。dev06冷启动未加载目标成功示范、关键姿态、历史动作、成功参考或旧搜索报告。

任务接近筛查24候选中3个通过；抓姿`geom92_jaw0_approach2_+1_flip-1__approach_axis2_-40.0mm__rrt0`通过RRT预检并真实执行稳定双侧抓取及30 mm抬升，共226动作。Student handoff为32动作。放置布局15直接路线30动作后停滞并被排除；布局36侧向绕行路线真实执行355动作，t=613时`In(black_book_1,desk_caddy_1_back_contain_region)=true`、reward=1、done=true、`task_success=true`。因此这是Standard 1000步任务成功。

V3状态为`complete`、engine return code 0。成功Episode由226抓取/抬升、32 handoff和355 placement动作组成，613/613动作记录、15,325/15,325 MuJoCo子步监测完整且时间轴连续；失败Student-first 32动作及直接放置备选路线不计入成功轨迹。geometry monitoring为178 complete/435 unsafe；`strict_10mm_pass=false`、`safety_certified=false`、`safe_training_eligible=false`，Stage A成功不代表安全认证。executor记录`post_success_terminal_exception=environment_done_during_grasp_only_branch`，但最终原生状态仍为成功并已由V3提交。

dev06归档：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_L1T1_world_vertical_clearance_cold_start_dev_06/v3_run`；V3报告：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_L1T1_world_vertical_clearance_cold_start_dev_06/v3_run/v3_report.json`；executor报告位于同归档的`task_success_engine/task_success_first_executor_obstacle_avoidance_L1T1_20261030_074500_342483841/run_report.json`。此前dev05在`t=364`失败；其归档中的`native_task_success_timestep=364`是旧报告字段错误（final task_success=false），V3现已修正为仅在终态成功时写入该字段。L1T3 dev02已完成且筛查失败；下一配置为`obstacle_avoidance/L1T4`。

## L1T2 Extended 5000步任务成功；Standard 1000步仍失败

`obstacle_avoidance/L1T2` 从新鲜合法init-state 0、seed 0 t0开始；reset/export未执行任务动作且`no_prior_experience=true`，init-state count=50。初始`On(moka_pot_1,flat_stove_1_cook_region)=false`、`On(moka_pot_2,flat_stove_1_cook_region)=false`、`Turnon(flat_stove_1)=true`、`task_success=false`。BDDL SHA-256 `88c64be300fa5934a096c6ef545514e1f541cc67c7c1f09bed1bc4f7e6583bc6`；t0 payload SHA-256 `b8137dca32defb9b1d0c2cde8c91fc132d66991129f823a180974765e561b456`；integration-state SHA-256 `6494373b0f5670cfd2e33e096b35c5f85f732a8767b85241c660f36f5563ae2a`；模型几何SHA-256 `6996aaa940b011861510ebd1911ae19f7db66c54e189ca4c107d372b06033edf`。未将目标场景旧成功动作、关键姿态或示范作为输入。

Standard 1000配置在t=595提交同run状态：pot1 `On=true`、pot2 `On=false`、炉灶`Turnon=true`、总体`task_success=false`；前缀595动作/14,875子步连续完整，失败Student前缀19动作精确丢弃。随后从合法同Episode t=595 snapshot续接。Stage 2–5没有增加原生目标；Stage 6的抓取/Student handoff在放置代码异常后整体弃置；Stage 7修复有界stage retry cap后从同一t595 snapshot继续。

Stage 7完成pot2专家抓取/抬升237动作、Student handoff 32动作、placement branch 319动作，共在t=1183满足全部目标：`On(pot1)=true`、`On(pot2)=true`、`Turnon(stove)=true`，原生`task_success=true`、reward=1、done=true。V3报告为`status=complete`、engine return code 0、`stage_count=2`、`stage_attempt_count=7`，并确认Stage 7边界提交。独立动作审计为1183/1183连续7D动作、29,575/29,575 MuJoCo子步监测完整；标准1000步结果仍失败，因此只计为Extended Demonstration Success。Stage 7 placement executor在任务成功后记录`environment_done_during_grasp_only_branch`异常，但其最终native状态为complete，V3已提交该成功阶段。executor报告：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_L1T2_budgetfix_01/v3_run/continuation_stages/stage_07/task_success_engine/task_success_first_executor_obstacle_avoidance_L1T2_20261029_135352_585622114/run_report.json`；V3报告：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_L1T2_budgetfix_01/v3_run/v3_report.json`。

这是一条Stage A任务可行轨迹，不是安全认证结果：报告中的`safe_training_eligible=false`、`strict_10mm_pass=false`保留为后续Stage B相关信息；Stage A不以安全认证为成功前提。最终稳定释放未评估。源码bundle指纹为`bac297d3a3ecd7d8dcd8b7a2c1bcc3ba8b6e225492e42981d349cf78eab63fe6a`。

## Task-Feasible筛查边界修正

对L0T3/L0T4归档及共享筛查代码的复核发现，`task_approach_only`原先仍将默认10mm clearance应用到BDDL禁触对象，形成隐含Strict余量筛选。逐候选记录未发现仅因正间隙小于10mm而拒绝的候选：L0T3为21项实际碰撞、3项IK失败；L0T4为24项实际碰撞；L1T1 dev02历史筛查为15项实际碰撞、9项IK失败且无余量门槛拒绝；dev06已在修复后完成原生任务成功。因此该实现偏差不解释已归档候选失败。

已在[候选筛查实现](../research/gs_opd/repair/mujoco_jointspace_planning.py)中修正：task-approach模式以实际碰撞/穿透和IK可达性评估任务可行性，不再继承10mm安全余量；Strict-10mm与2mm授权穿透边界的定义和安全认证逻辑保持不变。该修正发生在L1T0归档完成之后；L1T0运行仍对应当时的源码指纹`d0ea082cf6d31e6c7b4e5f6a55a80a95a8efe8e3f8a157c634b3106e4ad12e2b`。Task-approach筛查修正后的阶段指纹为`6b75e613b78f6662ad650d3a51d892bfcbbe6150120847aeea927ee788706371`；Stage 6运行源码bundle指纹为`16bf4da59356db6af8b3d0b9501d94616e4c232b386b5f7339d46278dba889e8`；修复后的当前bundle指纹为`bac297d3a3ecd7d8cd8b7a2c1bcc3ba8b6e225492e42981d349cf78eab63fe6a`，均按verification JSON记录的有序相对路径/文件SHA-256规则计算。

## 权威记录与下一步

30项逐任务状态唯一来源为[任务矩阵CSV](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv)；本地注册核验见[verification JSON](GS_OPD_SAFE_TRAIL_TASK_MATRIX_VERIFICATION.json)，协议见[矩阵说明](GS_OPD_SAFE_TRAIL_TASK_MATRIX.md)，过程及历史见[实验进度](GS_OPD_PROGRESS.md)与[路线图](GS_OPD_ROADMAP.md)。L1T2 Standard 1000在t=595仅完成pot1 On；同Episode的horizon=5000续接已在t=1183原生成功，计入Extended Demonstration Success而非Standard Success。当前共9/30配置至少有一条原生成功轨迹，其中7项Standard、2项Extended；L1T1已于t=613 Standard成功，L1T3 dev02筛查失败，L1T4 dev02放置失败，L2T1 dev02复验筛查失败；下一项为`obstacle_avoidance/L2T2`。`obstacle_avoidance_human/L1T0`已于t=484 Standard成功。统一冻结版本评测尚未开始；开发中定向调试结果与后续冻结版本评测分开统计。


## Active checkpoint — 2026-10-30 15:35 CST

当前仍在执行 OAH/L0/T0 Standard 续跑。原 246 步合法连续前缀加同快照续跑57步，共303动作/7,575完整监测子步；t303 native `On=false`，不计成功。branch00 bilateral grasp 持续，碗到 plate 目标平移距离已由约0.288 m降至约0.213 m。继续观察该物理分支到明确成功或失败，再依据终态状态决定下一任务/路线；覆盖率仍为9/30。
