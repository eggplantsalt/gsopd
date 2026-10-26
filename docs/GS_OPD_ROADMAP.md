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
