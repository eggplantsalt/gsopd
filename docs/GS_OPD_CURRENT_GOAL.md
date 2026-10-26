## 最新局部目标：停止L2T2和Human L0T2相似追加搜索，修复可执行路径续接共性缺口（2026-10-26）

Round63后已完成一次通用RRT结果缓存交接实现并做真实物理验证：L2T2的65点完整7D/EEF路线经过模型、BDDL、起点、目标、策略及规划配置指纹核验后直接复用，没有第二次RRT；42个真实OSC动作后在预放置前停滞。按预算转向Human/L0T2并用GS-OPD自有L0T2/L1T2成功关键姿态经验：215步实现Pot1稳定抓持和34.404 mm抬升，360步真实运输后距预放置约85 mm，但尚未松爪；一次局部重规划因物理失败状态在控制器关节限位外而无动作退出。两个场景完整原生成功均未取得。

本轮采集数量：L2T2 42个物理动作/1分支；Human L0T2 575个动作/2物理分支；新完整Task-Success Episode为0。首次抓持耗时454.416秒；端到端墙钟时长未被执行报告可靠保存。监测的记录数与25子步期望计数见跨场景报告，但整体子步监测标志不完整，因此数据仅为真实物理Task-Success搜索证据，不是已认证安全Episode。

保持`task-success→safety reconstruction`边界、BDDL、物理约束和控制器关节限位不变。暂不对这两条路线追加同类搜索。后续主线应解决“可复用规划路径如何在部分物理进展后由闭环控制持续执行或仅续接变化段”，同时确保可达状态才进入OSC，且失败Snapshot仍可恢复。报告：[`GS_OPD_CROSS_SCENE_CAPTURE_REPORT_20261026.md`](GS_OPD_CROSS_SCENE_CAPTURE_REPORT_20261026.md)。

---

## 最新局部目标：暂停L2T2相似搜索，优先验证T0跨Level技能迁移（2026-10-26）

Round63从t=2589精确Pot2持抓Snapshot出发，使用L1T2自有Keypose经验。唯一被选择的路线预放置RRT预筛可行，但执行阶段同状态RRT重规划在20.053秒/265项配置检查后超时，尚未发出OSC动作；因此本轮动作数为0，原生任务仍2/3且`task_success=false`。起点与失败终态Snapshot字节一致；闭环执行路径尚无物理验证证据。Round63证据和hash见`GS_OPD_PROGRESS.md`顶部机器/人读报告。

本轮关闭L2T2连续专项搜索，不追加候选或物理分支。下一步转到T0跨Level迁移：使用经验库中L0T0自有的黑碗稳定抓持技能验证同任务族重定位。该技能记录了36.3 mm真实抬升，但源任务未完整成功；只作为抓持技能先验，最终仍以目标场景原生BDDL `task_success=true`为唯一完整任务成功判据。跨Level执行继续使用真实MuJoCo动作、Snapshot与碰撞记录，危险接触只做Task-Success风险记录，安全认证保持独立。

---

## 最新局部目标：L2T2 Keypose续接已取得可恢复持抓前缀，完整成功仍未取得（2026-10-26）

从run55 Pot2双指持抓Snapshot（t=2589）严格恢复，核对模型SHA `718ba2a082c0…`、BDDL SHA `58b3db563524…`、payload SHA `9b0a9942…`、Integration State SHA `836981a4…`及2/3初始原生谓词。只用GS-OPD自有L0T2与L1T2成功关键姿态经验，不回放历史动作。L0来源路线唯一通过首轮整机/持物预放置RRT和末端IK筛选；L1来源候选受RRT预算阻挡。

Round60在一个真实分支执行41个OSC动作（1025个MuJoCo子步）后，真实预放置跟踪停滞。Pot2继续双指抓持并移动约80 mm，Pot1 On和Stove Turnon仍为真、Pot2 On仍为假；reward=0、done=false。失败终态可精确恢复，Snapshot SHA `e489a1e9…`、Integration State SHA `61a428f0…`。Round62从该状态精确恢复，4次RRT调用没有筛出不同的可执行路线，未开启第二个物理分支。Round61错误快照根在模型指纹门禁处停止，0动作。

本轮新增动作41；RRT候选筛选累计166.663秒，物理路线RRT 11.190秒；完整任务成功0，稳定释放未到达。1025/1025子步接触采样齐全；52,275条接触身份记录包含每步重复的Hand–Pot2非指接触，因此只记录为Task-Success候选，安全认证=false。视频、OA动作/状态、Snapshot与机器摘要链接见`GS_OPD_PROGRESS.md`和跨场景报告。

**当前最优状态：**Round60 t=2630失败终态Snapshot。下一次只继续L2T2，优先诊断为何可达RRT路线在真实OSC的预放置Waypoint停滞；不重做抓取，不重复原路线，不转去其他场景或训练。

---

## 最新局部目标：完成run53最后几厘米的稳定释放（2026-10-26）

L1T2 run53原生成功前缀t=1335的精确Snapshot SHA为`32bdc4296106e794df472092698fece507e697e7169ff4771b1a9bf0c72c9563`。原地完全松爪但不撤离时，Pot2仍于t=1342失去Cook Region的原生`On`，确认Pot2原放置中心靠近区域边缘，释放后存在自身滑移风险。

通过仅修正最后持抓放置位置，25 mm Region相对移动（实测`[15.797,18.213] mm`）后真实松爪、缓慢上撤并Settling，取得run53成功后缀的稳定原生终态。77个真实控制动作、1925个子步完整记录，夹爪最终完全脱离；Settling 500/500子步Pot1/Pot2 On和Stove Turnon为真，Pot2位姿跨度0、速度近零。退爪途中22个子步存在短暂手壶重接触，后续完全分离且Settling阶段无接触，已在报告中逐项标注。35 mm候选也成功但更耗动作；侧向先行策略更慢且发生更长短暂接触，故不选作最佳后缀。

证据与报告：`/opt/data/private/gs_opd/rollouts/run53_release_local_placement_20261026_attempt1/`，侧向对照在`.../run53_release_local_placement_lateral_20261026_attempt2/`。该结果从已成功前缀续接，非从初始化冷启动Episode；原生完整成功为真，严格安全认证仍为假。当前目标已完成；下一步仅按用户后续指示处理安全重构或其他工作，不继续扩大本次局部释放分支。

---

## 最新局部目标：run53 原生成功后的 Release/Retreat/Settling 诊断（2026-10-26）

本轮暂停新 L1/L2 搜索、训练及 Keypose/IK/RRT 算法改动。run53 的原生 L1T2 成功仍成立：t=1335 `Pot1 On=true`、`Pot2 On=true`、`Stove Turnon=true`，原生 `task_success=true/reward=1/done=true`。独立分支从 run53 成功前最近的物理 Snapshot 精确恢复并重放118步后执行真实松爪、撤爪和20步 Settling。指尖接触在t=1338完全消失；该时刻两个Pot的On谓词仍成立。Pot2在撤爪中t=1342失去On，之后接触桌面/玩具球；Settling后Pot2位姿跨度40.174 mm、最大线速度0.119 m/s、角速度1.876 rad/s，稳定性失败。诊断分支终态不能覆盖原run53成功结论。

run19→46→47→48→52→53的t=0…1335物理Snapshot/动作/机器人与物体状态来源连续，边界Snapshot payload SHA及Integration State与恢复报告匹配；run48根报告缺失，但中断索引、动作状态记录及Snapshot仍在。全链每步双相机视频与机器人/物体物理状态可追溯，但未保存完整逐步原始策略Observation张量。因此物理来源链可精确恢复，完整原始Observation–Action数据不完整。四项判定分别为：原生任务成功（原run53：true）、释放后稳定（false）、轨迹完整性（物理状态/动作连续，策略Observation不完整）、安全认证（false；本诊断仅记录控制步末状态，没有25子步安全监测）。

主诊断归档：`/opt/data/private/gs_opd/rollouts/run53_release_stability_diagnostic_20261026_150815_continued/`；摘要、逐动作/接触JSONL、NPZ、双相机视频、Snapshot和来源审计均已保存。没有改动原成功报告、原生任务判据、安全标准或训练数据。下一步先离线分析Pot2离开Cook Region的物理时序及Release/Retreat证据；没有新的场景搜索授权时不启动L1/L2搜索。

---

## 当前局部目标：Keypose迁移的通用可达性与阶段续接（2026-10-26）

旧版跨场景Task-Success-First暴力轮次已暂停。唯一最新成功证据是L1T2从已有Pot2持抓Snapshot经L0T2自有Keypose经验真实MuJoCo接续后取得原生`task_success=true`（3/3谓词），不是reset冷启动；478个专家OSC动作、两个精确Snapshot分支，安全认证false；L1终态时Pot2仍双指夹持，release/retreat没有执行（原生done提前结束），当前只认定原生BDDL成功，不认定释放后稳定放置。run48旧路线同一持抓起点为920动作/3分支且未成功。L2T2用同一关键姿态经验测试两次局部续接，合计236个真实动作后到高位西侧持抓状态，仍2/3谓词；主要阻塞是当前场景有界IK无解及OSC在已记录关键姿态附近停滞。

下一目标是针对关键姿态迁移的**通用可达性与失败段续接**：检查retarget变换、IK seed portfolio、路由/关键姿态转移和阶段完成阈值，提出可跨场景的改进并用Snapshot/真实物理验证；不得靠增加同一路线重试、毫米级安全余量或任务坐标硬编码。再次物理执行先用同一L2 Snapshot链的一次有界验证；L2达到原生完整成功前，不扩展L0T0及其他新场景，不启动训练。

验收仍是活动BDDL `task_success=true`。源动作不可直接回放；安全重构独立进行且标准不变；所有搜索候选`safety_certified=false`。当前没有活动物理进程。归档与对照见 [跨场景JSON](GS_OPD_CROSS_SCENE_TRANSFER_20261026.json)、[进度记录](GS_OPD_PROGRESS.md)。进程墙钟成功耗时未可靠记录，保持unknown。

## 当前局部目标：验证迁移技能、打通多目标第二物体放置（2026-10-26）

L0T2自有成功经验`t248`抓取模板与`t239`阶段序列已跨层用于L1T2、L2T2。两者均实际获得双指抓持并抬升；L1首次该技能用时158.77秒，L2首/次Moka Pot技能分别为474.324/614.191秒。L2先完成Pot1原生`On`谓词，当前物理证据链最后可用的BDDL状态为2/3谓词；Pot2在运输中先遇Place Box、后遇已放置Pot1接触。run33已存88个真实动作后Bridge断管，未取得最终BDDL查询。

完整任务成功仍为0条，首次完整成功耗时为空。跨场景抓取/阶段迁移得到验证，但降低完整任务首成功成本尚无证据；L0T0/T1/T3/T4经验检索及任务成功均为0/4。无任务专用完整脚本、无场景人工位姿调整，搜寻动作均未安全认证。官方示范未参与。

执行归档、技能时间、动作数、分支和复现链接见[机器报告](GS_OPD_CROSS_SCENE_TRANSFER_20261026.json)与[进度记录](GS_OPD_PROGRESS.md)。下一步应基于Pot2持抓Snapshot泛化处理先前目标对象阻挡，完成真实放置、释放、settle及BDDL复核；Bridge断开时保存最后耐久动作但必须把native终态标成不可用。成功后再验证不同物体族，仍以原生`task_success=true`唯一认定完整成功，严格安全认证继续独立执行。

# GS-OPD 当前局部目标

## 最新局部目标状态：跨层迁移已测，放置与L2可达性仍未闭环（2026-10-26）

L1T2和L2T2均使用L0T2完整成功源模板，场景专用脚本与人工坐标调整均为0。L1通过真实MuJoCo抓稳并抬升34.54 mm，首次抓取技能成功用时158.77秒；905个专家动作累计后，完整原生目标仍未稳定成立。L2按BDDL顺序选择moka_pot_1，4次预检中2次有界IK无解、2次规划超时，没有物理动作。L0T0/L0T1的8步Student试探后各做4次RRT预检，目标均未触达。

经验检索排序现优先完整原生成功来源；L1抓取模板、两条此前验证的L2抓取技能及L2部分转移技能已原子合并至持久v2经验库。所有新增条目仍标记为非完整任务成功、非安全认证；L0完整成功来源继续优先。合并前备份位于 /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2_pretransfer_merge_20261026.json。Executor多对象目标自动遵循BDDL待办顺序；抓持丢失时继续物理释放与目标复查。L1 run21 settle预算异常后的归档缺口已在代码中修复，后续错误会保存终态Snapshot和原生状态。

本轮没有新完整原生成功Episode，不能报告time-to-first-native-success，也没有证明任务成功搜索成本随经验累积而下降。所有搜索轨迹仍非安全认证数据。接下来先用失败Snapshot证据定位L1稳定放置失效与Settling判据，再只做一次有界的当前状态放置重规划；L2下一次应从另一个L0成功抓取变换或通用几何候选开始，不能重复同一个失败终端姿态的超时预检。随后用同一经验库入口尝试不同技能L0任务。详见 [跨场景迁移报告](GS_OPD_CROSS_SCENE_TRANSFER_20261026.md)。


## 最新优先级：L0T2经验跨层迁移与失败段成本摊销（2026-10-26）

以已成功的 GS-OPD L0T2 Episode 为源，先迁移到 `obstacle_avoidance/L1T2` 和 `L2T2`，再评估不同技能的L0任务。优先顺序是对象相对经验检索 → 物体坐标系下历史接近参考点 → 实时几何绑定与IK/RRT → OSC/MuJoCo有界验证；命中可迁移技能时不为无操作收益加载Student模型。放置从活动BDDL和实时目标Region解析。失败分支只可通过真实Snapshot回滚/局部重规划恢复，不能写qpos、瞬移物体或人工满足谓词。

现有自有经验库包含2个Moka Pot抓取模板和2个通用物体转移阶段模板（历史阶段顺序及阶段动作预算）。匹配的模板仅是搜索先验，跨层有效复用必须由实时抓持稳定性验证，原生完整任务成功仍只看 `task_success=true`。安全余量继续留在严格安全重构/认证阶段；搜索轨迹的 `safety_certified` 保持false。官方安全示范不参与搜索或训练，本轮不训练。

截至最新归档，L1T2有3次有限搜索、L2T2有2次有限搜索：两场景历史技能检索命中2/2，物理稳定抓持和完整原生成功均0；L1最高为到预抓取再因24步表面接近预算耗尽，L2最高为到预抓取约70.6 mm范围内但跟踪预算耗尽。没有首次成功，因此跨场景time-to-first-success均为null。下一次L2T2运行正在测试“失败段Runtime Snapshot → 从当前真实状态重规划”的通用逻辑，并将预检数减至2，以优先把预算放在历史技能的实际物理验证上。试验根目录为 `/opt/data/private/gs_opd/rollouts/task_success_first_cross_level_L2T2_20261026_run3/`。

跨独立运行的失败路线记忆还未持久化；局部重规划仅在最新L2T2 run3中首次执行，等待真实结果。L1/L2尚无成功经验可传播，因此暂不扩展L0T0/L0T1新物理搜索。新场景专用完整动作脚本与人工任务坐标调整数量均为0。25项相关回归测试通过；其他历史目标与记录保留在本文件下文。

## 当前目标：完成L0T1 Task-Success-First端到端轨迹（2026-10-26）

已实现 `research/gs_opd/repair/task_success_first_executor.py` 的通用Student优先调度：只用原始π₀.₅基础权重；Student未达阶段/原生任务目标时，从精确Runtime Snapshot触发有界候选搜索；抓取/抬升成功后恢复并核验Snapshot，可把当前状态交回Student；再未成功则由 `TaskPhaseController` 驱动原生目标区域运输、放置、松爪、退出和稳定验证。专家RRT只在此搜索模式显式使用record-only配置；碰撞与余量作为实际结果记录，不作为统一提前拒绝门槛。最终完整成功只接受活动BDDL `task_success=true`。无动作、状态或目标物体位置直接写入；无官方示范、训练、LeRobot或安全标签。

**当前结果：闭环可运行但未得到完整成功Episode。** 32步原始Student段没有抓书（书位移0.000059 mm、原生`In`为false），程序精确恢复t=0 Snapshot后自动搜索。多个抓取姿态的真实OSC路线撞到/受阻于Beer附近，预抓取超时；反向接近方向IK无解。最长有界验证覆盖12/16个几何候选方向、最多3个真实Snapshot分支/轮；所有失败分支均还原初始状态。无稳定抓持、30 mm抬升或完整BDDL成功快照。完整结果和分段失败证据见 [GS_OPD_PROGRESS.md](GS_OPD_PROGRESS.md) 顶部及私有rollout归档。

下一局部目标是基于真实失败接触优化任务成功候选的接近方向与搜索多样性，避免把Safety Reconstruction余量变成前置过滤；继续用物理分支确认可抓取性，失败自动恢复。取得真实双指抓持/抬升后，才验证Student接续与原生contain-region放置；只有原生BDDL task_success成立才结束Episode。Safety Reconstruction/训练实现保持不变，Task-Success候选绝不自动升级为安全数据。

当前入口以单个活动`in/on`目标完成L0T1试点；多对象目标链和跨30任务的阶段配置尚未验证。

## 最新局部状态：L0T1 24个书本抓取候选scratch筛选完成（2026-10-26）

在研究分支修复 `research/gs_opd/repair/mujoco_jointspace_planning.py::screen_grasp_candidates`：接近路段仍在t=0张爪姿态检查；抓取阶段在目标姿态按几何推导的闭爪IK，独立scratch `MjData` 中从开爪到闭爪做9个限位内采样，每个采样测左右真实pad碰撞Geom到书的有符号距离，并运行完整机器人/场景几何、接触及自碰撞策略。最终双侧均须在原有±2 mm接触容差内。经过模型SHA、Geom/Body身份和固定祖先链核验的唯一静态安装Pair `(robot Geom 90 mount0_pedestal_col, scene Geom 14 study_table)`继续保留；未豁免其他Floor/Table或Pedestal几何，也未改变10 mm禁触余量、Hand接触、Beer、Place Box、关节或BDDL规则。

从原始π₀.₅ t=0 Snapshot精确恢复：Snapshot SHA `41e9846c4d6cc983816d866d3c24aa554740ff427cd3982bed9be8102342103d`、模型SHA `9e2f374b19a09a123e715dcefc0cebaa11d647b4029d4546bc6b52a2944d451e`、Integration State SHA `41a6cd314d41cda632876ed7a889958bf3ccd99484196de55c81fb54daca2522` 均匹配。原有24候选均完成预抓取与闭爪目标IK（24/24在关节限位内），24/24执行10点张爪接近几何检查和9点闭爪几何检查。夹爪pad间真实几何开口从39.699 mm降至27.680 mm，闭合采样全程在关节范围内。

**筛选结果：0/24完整可行候选，无候选排名。** 六个候选两侧pad—书距离各约−0.500 mm，满足双指±2 mm几何容差；另18个为双侧过度穿透（12个约−3.042 mm、6个约−6.354 mm）。但24/24的闭爪姿态和全部9个闭合样本都触发书本非法Hand接触：`black_book_1_g1` Geom 92 与 `gripper0_hand_collision` Geom 79（Body 12）的signed distance约−59.758 mm；该接触不属于合法指垫Pair，故所有候选均按原安全规则拒绝。另有12个预抓取点的书本几何余量低于10 mm，另12个张爪接近路径低于10 mm；筛选没有因早期失败而跳过独立目标闭爪诊断。Beer/Place Box对候选机器人路径的最小已测余量分别为76.370 mm和13.553 mm，均未触发其禁触门槛。IK不可达0个；按失败类别并集计，碰撞约束失败24个、抓取几何/双指穿透失败18个（后者同时伴随Hand/路径碰撞）。

全筛选期间没有动作或MuJoCo物理子步；Safety Bridge前后Integration State逐元素相同，hash仍为 `41a6cd…2522`。未执行RRT、OSC抓取、抬升/退爪、训练或LeRobot转换；没有新增抓取Snapshot、真实动作或视频。未加载/读取官方安全示范。该结果是候选假设被筛除，不代表任务或抓取路线整体不可行；下一局部目标需在原禁触规则下调整书本候选几何以避免Hand穿书并满足双侧指垫容差，之后再做独立scratch筛选。

完整机器报告和人读摘要：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_t0_book_grasp_candidate_screen_20261026_attempt4/candidate_screening.json` 与同目录 `candidate_screening_summary.md`。复现入口：`/workspace/zy_workspace/LIBERO-Safety/.venv/bin/python research/gs_opd/repair/screen_l0t1_book_grasp_candidates.py --output-dir <new-empty-dir> --bridge-port <unused-port>`。新增目标回归 `research/gs_opd/repair/test_mujoco_jointspace_grasp_screen.py` 覆盖Hand触书、其他桌面碰撞、单指假抓、闭爪关节采样及唯一固定安装Pair；连同静态安装策略回归共7项通过。

## 最新局部状态：L0T1 t=0 固定安装几何门禁已解除（2026-10-26）

对原始 Student 的 L0T1 t=0 Runtime Snapshot 在独立 MuJoCo scratch `MjData` 中恢复并核验：Snapshot SHA-256 `41e9846c4d6cc983816d866d3c24aa554740ff427cd3982bed9be8102342103d`、模型几何 SHA-256 `9e2f374b19a09a123e715dcefc0cebaa11d647b4029d4546bc6b52a2944d451e` 和 Integration State SHA-256 `41a6cd314d41cda632876ed7a889958bf3ccd99484196de55c81fb54daca2522` 均匹配。几何身份复核纠正了先前描述：Geom 14 不是 world Body 0 的 floor，而是未命名 `mjGEOM_BOX`，挂在 `study_table` Body 1；Geom 90 是 `mount0_pedestal_col` 圆柱，挂在 `mount0_pedestal` Body 22。两者的 Body 祖先链都没有关节；扰动 7 个 Panda 关节后两个 Geom 的世界位姿变化均为 0。模型指纹、精确名称/ID和固定关系共同表明，这是固定机器人安装座与桌面碰撞盒边缘的模型几何重叠，signed distance 为 −99.460 mm。

只在 `safety_bridge.py` 增加可信静态安装pair注册表及可选 `allowed_static_installation_contact_pairs` 策略字段。为兼容已有L0T1调用方，只有Bridge元数据中的模型SHA精确匹配时才自动纳入该注册pair；显式空列表可关闭许可，其他模型不会获得默认许可。归一化器仍重新计算并核验指纹、Geom/Body身份及双方无关节祖先链。唯一被排除的是 `(robot geom 90, scene geom 14)` 这一对对 clearance 和 scene-contact 身份判定的影响。Floor/table其余几何、Pedestal其他 Geom、移动连杆、Beer、Place Box、目标物、自碰撞、关节限制和原 clearance 阈值仍照常检查。

在独立 Safety Bridge 进程恢复同一 Snapshot 后重跑全几何只读门禁：修复前唯一违规为 `unmapped_scene_environment` 的 −0.099460 m；修复后状态 `complete`，最小剩余间距 +0.028005 m，Beer/Place Box间距仍为66.233/142.490 mm，禁止接触、自碰撞与关节违规均为空。两次门禁前后 Integration State SHA 保持不变，原生目标仍为 false、reward=0、done=false；未发动作、未执行MuJoCo物理子步、IK、RRT、抓取或训练。新增回归测试2项通过，其中不同移动连杆对同一Floor的负距离仍被保留并判为非法。

证据归档：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_t0_safety_gate_static_contact_fix_20261026/`（`geometry_evidence.json`、修复前后Safety Bridge结果、Snapshot只读校验、汇总及artifact hashes）。旧冷启动失败归档保留在 `obstacle_avoidance_L0T1_t0_book_cold_grasp_20261026_attempt1/`。初态全几何门禁这一唯一阻塞已解除；本轮未继续IK/RRT或抓取。已知开夹爪指垫筛选问题仍按此前记录延期，未在本轮处理；后续开始候选筛选前需单独解决并保持安全阈值不变。

严格保持原始 Student rollout → 暴力搜索任务成功轨迹（允许不安全）→ 自身历史成功经验几何重定位与安全重构 → MuJoCo完整验证 → GS-OPD安全Episode路线。未使用LIBERO-Safety官方安全示范或其动作/关键帧/轨迹/专家标签。

## 修复前记录：L0T1 书本冷启动停在 t=0 全几何门禁（2026-10-26）

**勘误：**本节保留修复前的搜索状态；当时将 Geom 14 描述成 world floor/Body 0 是错误的身份标注。上方独立scratch模型核验及本节后的审计记录已更正为未命名桌面碰撞盒、`study_table` Body 1。原负距离与修复前 unsafe 结论仍是有效历史结果。

## 修复前记录：L0T1 书本冷启动停在 t=0 全几何门禁（2026-10-26）

已从原始 Student 的 `initial_runtime_snapshot.pkl` 严格恢复 `obstacle_avoidance/L0T1`、seed 0、init-state 0；Snapshot SHA-256 为 `41e9846c4d6cc983816d866d3c24aa554740ff427cd3982bed9be8102342103d`，模型几何 SHA-256 为 `9e2f374b19a09a123e715dcefc0cebaa11d647b4029d4546bc6b52a2944d451e`，Integration State SHA-256 与 manifest 一致（`41a6cd314d41cda632876ed7a889958bf3ccd99484196de55c81fb54daca2522`）。指令、BDDL 哈希与原生目标均匹配；书、Caddy、Beer、Place Box 的真实 t=0 位姿及碰撞绑定已归档。按书本 group-0 碰撞盒和活动 Panda 指垫几何生成了24个不同候选。

候选 IK/RRT 与物理分支尚未运行。初始全几何门禁报告 `unsafe`：唯一几何余量违规是固定 Panda 底座 `mount0_pedestal_col`（robot geom 90）与 world body 0 的未命名 floor geom 14 的基线 signed distance `-99.460 mm`。Beer、Place Box 对机器人的最小距离分别为 `66.233 mm`、`142.490 mm`；t=0 未见书—Beer或书—Place Box接触、机器人自碰撞或关节越限。由于既有 Safety Bridge 会把这个不随任务动作变化的底座—地面重叠判为 unsafe，本轮没有放宽阈值，也没有启动 RRT OSC 预演或物理动作。因此本轮没有书本抓取 Snapshot、动作 trace 或视频；不能把24个几何姿态称为IK可行或可抓取候选。

搜索预算实际消耗：几何候选生成24/24；IK与完整候选筛选0，RRT预检0/4，Snapshot物理分支0/2，真实控制动作0，MuJoCo动作子步0。代码盘点确认 `grasp_retargeting.py`、7D `JointSpaceRRTConnect`、`mujoco_jointspace_planning.py` 适配器、OSC_POSE、抓持稳定验证器和 `task_phase_controller.py` 均已存在；当前阻塞是基线静态几何处理，不是缺少RRT或阶段控制器。需要先为已验证的固定底座—地面支撑重叠补充明确且局部的几何处理，并保留 Beer/Place Box、书本合法指垫接触、其他机器人几何、关节与BDDL门槛，再从原始 t=0 重新做筛选。该扩展本轮未实施。

另一个静态代码审查发现：`screen_grasp_candidates` 在开夹爪姿态完成接近检查后，末端对向指垫距离检查仍沿用开夹爪 qpos；本轮未运行该筛选，也未修改规划代码。解除底座—地面门禁后，还需先修正/适配这一筛选接口，再将抓持接触判定交给物理分支核验。

归档：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_t0_book_cold_grasp_20261026_attempt1/`；关键文件为 `run_report.json`、`scene_start.json`、`geometry_candidates.json`、`initial_safety_geometry.json`、初始 Snapshot 副本及 `source/`。没有运行 Student Rollout、加载LoRA、使用官方安全示范、执行完整任务、训练或LeRobot转换；安全认证为 false。既定原始 Student → 暴力搜索（允许不安全）→ 自有历史成功经验几何重定位与安全重构 → MuJoCo完整验证 → GS-OPD安全Episode路线不变。

## 本轮结果：obstacle_avoidance/L0T1 原始 Student 基线（2026-10-25）

复用现有 Safety Bridge 与 `research/gs_opd/smoke/pi05_rollout.py`，在 `obstacle_avoidance/L0T1`、init-state 0、seed 0 核实场景后，只运行一个最多 1000 步的原始 π₀.₅ Student Episode。实际指令为 `pick up the book and place it in the back compartment of the caddy`，BDDL SHA-256 为 `b14943e935d04f7fe07dabce8c097dac3721d2c689965c0abb407df96dfcc844`，原生目标是 `In(black_book_1, desk_caddy_1_back_contain_region)`。Student 使用 `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft/model.safetensors`，未加载 LoRA；任务配置无需适配。

本回合执行 1000 个真实 7D 动作、200 个决策块；原生成功从未为真，reward sum=0、`done=false`，到达步数上限。未观察到稳定抓书：1000 个动作步均没有机器人—书接触，书始终与桌面接触；最高书体 z 相对首个动作后只变化 17.0 mm。书未进入 caddy 后格，终态原生谓词为 false。Student 与 Beer 在 52 个动作步发生接触（步 37–99），与 Place Box 在 29 步接触（步 136–268）；书与 Beer 在第128步接触，未观察到书与 Place Box 或 caddy 的接触。失败轨迹和碰撞历史完整保留，不作为安全数据。

全部 1000 个动作步均确认捕获 25/25 个 MuJoCo 子步，共 25,000 个；轨迹记录真实动作、200 个决策 Observation、每步双相机 Observation、机器人/物体状态、原生 reward/done/success 和被动接触监测。Safety Bridge 将接触按 Geom 对聚合为 occurrence count、首次子步和每动作覆盖计数，没有保存每个子步的原始接触点列表。起始与终止 Runtime Snapshot 均在活动 Bridge 模型内恢复校验通过；跨进程恢复还需匹配manifest中的模型几何 SHA 与场景元数据。视频为 1001 帧、20 FPS、agentview/wrist 并排 MP4。

归档目录：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_raw_student_seed0_init0_20261025_234627/`；任务摘要 `episode_summary.md`，机器报告 `episode_report.json`，原始 NPZ `raw_student_trace.npz`，双相机视频 `dual_camera_agentview_wrist_20fps.mp4`，Snapshot 位于 `snapshots/`。本轮没有使用官方安全示范、RRT、训练或 LeRobot 转换；安全 verdict 为 `not_evaluated`。下一步应基于该失败 trace 做离线阶段与碰撞归因，再按既定路线搜索任务成功轨迹；不得将它伪装为成功经验或安全 Episode。

`pi05_rollout.py` 在模型加载后执行既有 Bridge reset。reset前后任务、BDDL、目标书/Beer/mug起始状态一致；caddy与Place Box的随机fixture根位置不同。实际Episode起始物理状态以 `raw_student_trace.npz` metadata中的 `initial_physical_state` 为准，已在归档报告中记录该差异。

## 本轮补充：LIBERO-Safety 跨场景配置盘点与代码复用修订（2026-10-25）

本地只读盘点 `obstacle_avoidance` 与 `obstacle_avoidance_human` 的 L0/L1/L2 共 30 个任务，逐项整理 BDDL、符号初始状态、原生目标、接触代价对象、动态定义、技能需求与静态采集可行性。机器清单和 Markdown 表见 [task manifest](GS_OPD_LIBERO_SAFETY_TASK_MANIFEST.json) 与 [任务清单](GS_OPD_LIBERO_SAFETY_TASK_CHECKLIST.md)。本轮未读取/下载官方安全示范，未运行 Rollout、MuJoCo 或训练。

严格沿用：**原始 π₀.₅ Student Rollout → 暴力搜索成功轨迹（允许不安全）→ 仅用 GS-OPD 自身历史成功经验做几何重定位与安全重构 → MuJoCo 完整验证 → 生成 GS-OPD 安全训练 Episode。** 官方安全示范中的动作、关键帧、轨迹、专家标签不得作为当前搜索先验、冷启动经验、重定位来源或 GS-OPD Episode；仅保留未来独立 Offline SFT Baseline。

代码复核来源是实验分支 `gs-opd-overfit-recovery-08a0198`，审计提交 `2af2ef2d8651fce1196ba0ee8317fb21c7bc7360`。此前只查主分支导致的“缺少 RRT/阶段控制器”结论现已修正：`research/gs_opd/repair/jointspace_rrt.py` 中的 `JointSpaceRRTConnect`、`mujoco_jointspace_planning.py` 的 MuJoCo scratch-scene 规划适配器，以及 `task_phase_controller.py` 的配置化阶段推进均可复用。需要扩展的是任务级 forbidden/allowed contact 映射、目标 IK、动态障碍相位同步、阶段证据配置，以及 caddy/basket/microwave 插入、铰链门和动态人手障碍策略；RRT 规划结果不替代完整 MuJoCo 执行轨迹验证。

推荐首个新 L0 试点为 `obstacle_avoidance/L0T1`（书放入 caddy 后格），先核对 `bottle_of_beer__7_1`、`white_place_box_1` 的活动碰撞 Geom 绑定，并为现有阶段控制器配置插入与退爪证据。此准备记录与下方既有 L0T2 实验历史并存，不替换、不否定此前归档结果。

## 当前局部目标：已验收L0T2 LeRobot Episode视觉审计完成（2026-10-25）

只读通过LeRobot 0.3.3官方Reader逐帧检查已验收1460帧单Episode，输出20 FPS、73秒并排双相机MP4、8个关键时间步截图、接缝复核图、机器JSON和人读摘要。图像直接读取LeRobot存储内容，无额外180°旋转。全行动作前Observation与t→t+1真实7D动作对齐、source segment与phase一致；25段、24接缝、t=1459训练末帧与t=1460终态证据明确区分。

产物目录：`/opt/data/private/gs_opd/audits/l0t2_round0_dataset_visual_audit_20261025/`。视频为`l0t2_round0_expert_dual_camera_20fps.mp4`；关键帧总览`keyframes_contact_sheet.jpg`；可疑来源接缝前后对照`snapshot_seam_review_contact_sheet.jpg`；完整报告`l0t2_round0_lerobot_visual_audit.json`。复现入口：`research/gs_opd/data/audit_and_render_l0t2_round0_lerobot.py`。

没有发现重复帧、错位标记或稳健阈值下的大幅图像跳变候选。t=10腕部图像变化、t=398 agentview变化仅作为已核对的来源接缝复查项；关键帧视觉顺序大体连贯。夹爪符号按robosuite PandaGripper核实为−开、+闭、0保持；t=6和t=9有两条轻微低于−1的原始张开命令，保留原值并在JSON标记。终态t=1460只引用provenance里的成功证据，没有加入视频。数据集、模型和训练代码均未更改，未运行训练或Rollout。

## 前置局部结果：L0T2 Round 0 Fresh LoRA配置与batch预检通过（2026-10-25）

已新增配置`examples/sft/config/gs_opd_l0t2_round0_fresh_lora_20261025.yaml`，并将`research/gs_opd/data/run_single_episode_sft.py`阶段窗口改为从训练集逐动作provenance读取，不再依赖845帧、固定三阶段名称或旧phase平衡策略。L0T2使用`task_phase`标签的12个实际阶段、1451个唯一Action Horizon=10合法窗口；每epoch对transport重复1窗以凑足batch=3，执行1452次曝光、484个完整batch。稀少抓取/释放阶段不会被过度复制。

起点是L0T2 rollout既有Student checkpoint：`/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft/model.safetensors`，SHA-256 `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`。OpenPI审计确认667/667 base权重键覆盖且抽检权重差为0；训练时冻结base、新初始化rank-32 Action Expert LoRA（216 tensors、13,860,864参数），adapter及resume均为null。LIBERO norm stats SHA-256 `dae37d79a22108af83df9189c6710a3ec8e077d65b28e34bdf1da724e5ae30f1`。

真实官方FSDP/OpenPI单batch forward/backward通过：actions=`[3,10,32]`、state=`[3,32]`；two real camera views map to base/left wrist slots, the official third right-wrist slot is zero-padded and masked false. Loss `0.05735647` finite，216个LoRA梯度均有限、108个非零；backward前后trainable权重digest不变，optimizer step=false。单RTX 4090峰值分配18.94 GB，峰值保留23.996 GB。当前3×4090，RLinf支持FSDP world-size数据并行，但单卡已满足batch资源需求，因此不用额外卡或模型切分。

机器预检报告：`/opt/data/private/gs_opd/sft/round0_l0t2_fresh_lora_preflight_20261025/preflight_report.json`。主机有3×RTX 4090；官方RLinf FSDP可做数据并行，但当前GS-OPD provenance sampler约束world size=1，因此配置只用GPU 0的`no_shard`单卡方案。完整训练命令见`docs/GS_OPD_PROGRESS.md`本轮顶部或报告`resolved_config`。配置里的`1e-4`、constant schedule、0 warmup、batch 3、10,000 step预算取自此前成功fresh rank-32训练作为起始参考；正式LoRA训练仍未开始，下一目标是等用户明确开始训练后照该配置启动。

## 最新局部结果：Round 0 L0T2已导出并验证为一个LeRobot v2.1 Episode（2026-10-25）

已将统一验收报告接受的25个连续物理归档段转换成一个1460步LeRobot v2.1 Episode，20 Hz、单任务`put both moka pots on the stove`。动作使用原始实际执行的7D float32标签，不归一化、不使用预测动作；图像与8D机器人状态均取动作前观察。沿用现有RGB 180°相机方向变换和LeRobot 8D状态表示。

LeRobot 0.3.3 reader验证单Episode边界`[0,1460)`；1460/1460动作、状态逐项精确回读，两路共2920帧像素精确回读。有效Action Horizon=10窗口共1451个，逐个采样核验且没有越过终止动作；140个窗口跨越至少一个已验收快照接缝，接缝仍属于同一Episode。最大时间戳float32舍入误差3.052 μs。1460行provenance逐行携带Native/Operational验收、demo-only/标准Horizon终态资格及Strict-10mm未通过状态。

数据：`/opt/data/private/gs_opd/datasets/lerobot_l0t2_round0_operational_success_v21_20261025/`。Manifest、1460行逐动作provenance/源文件哈希及机器报告分别为`data_manifest.json`、`per_action_provenance.jsonl`、`round0_lerobot_acceptance.json`。Safety仍按源验收报告定义为Native+Operational pass、Strict-10mm未通过附加统计；1800 demo-only成功、1000标准Horizon未成功。这是一条成功Episode，不是25条样本。没有训练、修改旧数据或增加轨迹。

本轮数据转换入口：`research/gs_opd/data/convert_round0_acceptance_to_lerobot.py`。下一步目标只在用户另行启动训练后确定；当前保持不训练LoRA。

## 最新局部结果：t=0→1460统一安全验收通过，建立Round 0成功数据候选（2026-10-25）

本轮唯一新物理实验是从raw π₀.₅ seed/init-state重建t=0并重放前10个7D动作。正式reset序列下模型、BDDL、初始Integration State与归档完全一致；10/10动作与原始π₀.₅逐元素一致，250/250子步Native/Operational通过。得到的t=10完整Integration State、物体位置/姿态/关节、OSC控制器、夹爪当前命令、机器人缓存及task bookkeeping与Episode 9 t=10起点逐项相同；Runtime Snapshot字节哈希不同，已如实记为状态级精确兼容。

其后只读审计t=10→249和t=249→1460归档：239步前缀、各段完整7D OA索引、双相机、25子步覆盖和原生谓词均通过；t=249之后的所有Snapshot边界SHA精确相同。完整链共1,460动作、36,500子步，无记录的硬安全违规。终态t=1460原生Pot1 On、Pot2 On、Stove Turnon均true，`task_success=true, reward=1, done=true`，首次成功t=1459。Demo-only horizon成功，标准1000步Benchmark不成功。

结论：首条**Round 0 Operational安全成功数据候选**已建立；Strict-10mm附加统计仍为false，不能标注为Strict-10mm Safe。未重放后缀、未运行RRT、未训练LoRA，全部历史数据保留。

验收材料：`/opt/data/private/gs_opd/rollouts/round0_l0t2_complete_episode_readonly_acceptance_20261025/round0_complete_episode_acceptance.json`及同目录Markdown摘要；只读入口`research/gs_opd/repair/audit_round0_complete_episode_archives.py`。本轮物理重放归档为`l0t2_round0_raw_pi05_t0_to_t10_current_safety_20261025_attempt4/`。

## 历史局部结果：t=10→249前缀复验通过（后续完整验收已补齐t=0→10，2026-10-25）

> 以下记录生成于t=0→10当前策略重放之前。其“t=0→249完整证书尚未成立”状态已被本文件顶部的统一验收结果取代。

从已验证原始π₀.₅ t=10 Snapshot在隔离MuJoCo中重放Episode 9归档239动作至t=249。每个25子步均已重新检查当前对象级Native/Operational约束：5975/5975子步完整，Native与Operational全通过；自碰撞、禁触、未授权接触、未解析接触、Operational关节限位违规均为0。Pot 2非指手几何最小净距4.033 mm（规则2 mm），Ginger Beer 10.191 mm、Place Box 47.480 mm（均高于10 mm），四个白名单手指与Pot 2实际接触最深穿透0.624 mm（规则2 mm）。严格关节数值检查发现夹爪slide在1e-8 m raw判据下有微小越界，但最大7.375 μm，均低于既有、历史正常仿真校准的8 μm Operational容差；MJCF与关节标准没有调整。Strict-10mm附加统计未通过，不作为本策略失败。

t=249重放终态Snapshot与t=249→1460成功链源Snapshot文件SHA完全相同（`3e142c23aa2e0cad30569b350b6a4a4e83e2152ec34891f0ecd580d30c3d81c0`）；Integration State、OSC controller、机器人控制缓存、夹爪内部命令和task bookkeeping均完全一致，Pot 2稳定双指抓持验证通过。可复用该t=249接缝进行下一次连续Episode复验；后续TaskPhaseController应由恢复的真实状态新初始化，不能伪造历史Python计数。

归档：`/opt/data/private/gs_opd/rollouts/l0t2_episode9_prefix_t10_to_t249_current_safety_20261025/`；复现：`research/gs_opd/repair/replay_episode9_t10_to_t249_current_safety.py`。本段的t=10→249结果仍有效；t=0→10旧字段不足这一临时缺口已经由顶部记录的attempt4当前策略重放及t=0→1460统一验收补齐。

## 历史局部结果：t=1061→1460完成同Episode原生任务成功（后续完整验收已补齐前缀，2026-10-25）

Pot 1在t=1459首次原生On=true且task_success/reward=1/done=true；t=1460真实松爪后Pot 1仍On=true、Pot 2 On=true、Stove Turnon=true，物理释放由同Episode阶段事件确认。末五个Pot 1位置跨度0.896 mm，末态线/角速度低，存在真实burner支撑接触。最终状态t=1460，demonstration-only成功（horizon 1800），标准1000步Benchmark不成功。

从t=1061至t=1460的399个续接动作通过完整25子步监测，无禁触、自碰撞、未解析接触、关节越限或冻结Operational间隙违规；各耐久Snapshot段的接缝哈希精确一致。最终t=1453→1460七步有逐动作预测/实执行完整匹配。Pot 1目标物Operational 2 mm下限通过，Ginger Beer与Place Box保持10 mm以上；Strict-10mm统计不通过（主动Pot 1目标最低非手指间隙5.991 mm）。

归档 `/opt/data/private/gs_opd/rollouts/l0t1453_pot1_release_native_support_continue_20261025_attempt1/`；机器验收 `native_success_safety_acceptance.json`，终态Snapshot `snapshots/main_timestep_001460.pkl`。运行器在t=1460成功后多做了一次33点withdrawal RRT及隔离预演，预演因预计Pot 1 On丢失而被fail-closed拒绝，未提交任何额外动作；程序现已修复为在原生任务完成、物体已释放时立刻停止。

**当时本结果还不是Round 0安全种子**：t=0→1061前缀尚未按当前规则统一复验。这个状态现已由顶部t=0→1460验收更新；完整轨迹候选成立，未启动LoRA，原始LeRobot与示范归档保留。

## 最新局部结果：Pot 1稳定抓持并真实抬升32.4 mm；终点t=1061（2026-10-25）

从严格恢复的t=1042 Snapshot沿用Pot 2已验证的对象级Operational策略：当前操作目标Pot 1的非指尖最小间隙2 mm；四个指定指尖Geom（76/77/79/80）可与Pot 1合法接触，穿透最多2 mm；Ginger Beer/Place Box保持10 mm，Palm/hand、其他机器人Geom、自碰撞、关节限制及Native BDDL约束不变。t=1042原拒绝动作先隔离预演25子步通过，随后19个真实动作、475子步均通过Native/Operational监测。

达到`approach`、`close_gripper`、稳定左右双指抓持、真实lift后进入`transport`。lift锚点以上升32.438 mm；Pot 1非指尖最小间隙5.607 mm，Ginger Beer/Place Box最低47.329/68.368 mm，指尖对最深子步穿透0.694 mm，实际接触记录最深0.328 mm。Pot 2 `On=true`全程保持。终点Snapshot位于`/opt/data/private/gs_opd/rollouts/l0t1042_pot1_grasp_lift_20261025_032402/snapshots/main_timestep_001061.pkl`。完整OA、双相机、Snapshot、逐子步记录、视频与谓词归档于同目录。

任务尚未完成：Pot 1 `On=false`，Pot 2 `On=true`，task_success=false、reward=0、done=false；当前只取得可续接的Pot 1持物transport状态，不是完整Round 0种子。全局Strict-10mm附加统计因未映射桌面环境组为0 mm仍false；本段实际禁触物及Pot 1非指尖均符合各自Operational门槛。下一目标是从t=1061严格恢复并完成Pot 1安全搬运、放置和释放，持续保持Pot 2 `On=true`。

续接入口：`research/gs_opd/repair/continue_t1042_pot1_grasp_lift_target_policy.py`。不启动LoRA，不重复抓取搜索。

## 最新局部结果：Pot 1 Approach 候选被10 mm余量预演拒绝（t=1042，2026-10-25）

t=1019 Runtime Snapshot 严格恢复通过。真实执行至 t=1042 共 23 个动作、575 个 MuJoCo 子步，全部 Native/Operational 安全通过，Pot 2 原生 `On=true` 持续成立。Pot 1 尚未抓持。

下一候选动作的隔离25子步预演从第5子步起低于冻结的 Pot 1 非手指机器人几何 10 mm 余量：`gripper0_hand_collision`–`moka_pot_1_g12` 首次 9.996994 mm，最低 8.009800 mm；该动作未提交，没有实际接触。Ginger Beer/Place Box禁触、全机器人、自碰撞及限位无异常。已 fail-closed，不降阈值、不重复该动作。本轮离目标仍差 Pot 1 安全 Approach、双指稳定抓持和至少30 mm真实抬升。

证据归档：执行 `/opt/data/private/gs_opd/rollouts/l0t1019_pot1_grasp_lift_20261025_030744/`；拒绝预演 `/opt/data/private/gs_opd/rollouts/l0t1042_rejected_approach_preview_audit_20261025_031026/`。下一步需从兼容的真实状态寻找符合既有 Operational 约束的 Approach 接入，再继续闭爪、抬升；当前证据不支持盲目执行现有候选。

## 最新局部结果：复用t=738路径抵达Pot 1预抓取；本轮停在Approach入口（2026-10-25）

从同一物理链严格验证的t=796 Runtime Snapshot复用已接受的t=738第三条RRT路线，没有第四次搜索。路线有282个qpos/EEF参考点，起始游标58；旧qpos路径偏差只保留为诊断。风险预警50 mm不再单独阻止动作：进入预警带时，当前真实Snapshot上的短段分支必须先通过全25子步MuJoCo认证，随后恢复原状态并提交认证动作。t=892处预警经分支验证后继续，Ginger Beer段内最小净距24.470 mm；Ginger Beer/Place Box硬余量仍为10 mm，未放宽任何安全规则。

t=796→1019连续真实执行223个动作、5,575子步，Native/Operational Safety全通过，Pot 2 `On=true`持续成立。t=1019阶段自然进入`approach`，预抓取误差10.296 mm/0.04608 rad，在既有阶段切换容差内。尚未闭爪或抓取Pot 1；Pot 1 On=false，task_success=false、reward=0、done=false。故本轮达到预抓取/Approach里程碑，但未获得完整任务或Round 0安全种子。Strict-10mm全局附加项由于未映射环境几何组为0 mm而仍为false；机器人与Pot 1、Ginger Beer、Place Box的本段实测最小间隙均超过10 mm。

本轮入口：`research/gs_opd/repair/continue_t794_pot1_execution_aware_pregrasp.py`。主归档：`/opt/data/private/gs_opd/rollouts/l0t796_pot1_execution_aware_pregrasp_20261025_021845/`；从t=794开始的前两次单动作续接归档也保留于`l0t794_pot1_execution_aware_pregrasp_20261025_021358/`及`l0t795_pot1_execution_aware_pregrasp_20261025_021653/`。各段有独立Snapshot和完整OA/双相机/25子步证据，不作为彼此独立Episode拼接。

下一局部目标是在用户下一轮授权后，从t=1019真实Snapshot完成Pot 1 Approach、闭爪、稳定抓持和至少必要的真实抬升，保留Pot 2 On并延续相同滚动认证。当前在Approach入口停止，不自动开始Pot 1接触或完整任务续跑；不训练LoRA、不更改安全规则。

## 历史局部目标：t=794风险预警后确认第三条路线并停止等待诊断（已完成）

上一轮从 t=649 严格恢复后在同一 Episode 连续执行至 t=794，完成 Pot 2 释放退出，原生状态机进入 Pot 1 `second_object_recovery`。145个真实动作、3,625个MuJoCo子步均通过Native/Operational Safety；Pot 2 On持续为真。t=738及t=794同阶段先后两次触发 `gripper0_hand_collision`–Ginger Beer `bottle_of_ginger_beer_1_g102`约49.58/49.20 mm的 `risk_margin_approaching`。二者都高于10 mm硬余量且没有接触，但路线未退出同一50 mm重规划风险带。该历史停止点现已由t=738路径安全续接至t=1019，不再是当前目标。

本轮从 t=649 严格恢复后在同一 Episode 连续执行至 t=794，完成 Pot 2 释放退出，原生状态机进入 Pot 1 `second_object_recovery`。145个真实动作、3,625个MuJoCo子步均通过Native/Operational Safety；Pot 2 On持续为真。t=738及t=794同阶段先后两次触发 `gripper0_hand_collision`–Ginger Beer `bottle_of_ginger_beer_1_g102`约49.58/49.20 mm的 `risk_margin_approaching`。二者都高于10 mm硬余量且没有接触，但路线未退出同一50 mm重规划风险带；Recovery Engine的重复原因保护应停止继续执行。不得重置恢复计数后重放同一条路径。

停止是路线/控制阶段的重复风险触发，不是已发生的物理安全违规。t=794 EEF仍距Pot 1恢复目标362.661 mm、姿态差2.6714 rad，尚未发起Pot 1抓取。Pot 1 On=false、task_success=false、reward=0、done=false；标准完整任务和1800步扩展示范完整成功均未取得。Strict-10mm附加统计未通过；Native及Operational验收通过。t=794 Runtime Snapshot、双相机OA、每25子步监测、事件与视频已保存。

`automatic_recovery.py` 两处重复参数名已最小修复为 `replan_reason`，使下次运行能记录原定 fail-closed 分类；本轮不从 t=794 重启，也不再次跑同因路径。后续先复核 t=699→794 的几何风险对象与规划返回路径，确认是否存在实质不同且可从真实状态验证的Pot 1接入段；在该证据明确前不继续真实动作、不重新搜索相同候选、不拼接Round 0前缀、不训练LoRA。

证据：`/opt/data/private/gs_opd/rollouts/l0t2_t649_pot1_task_suffix_20261025_005902/failure_diagnosis.json`；Runtime Snapshot `snapshots/main_timestep_000794.pkl`；连续轨迹 `observation_action_trace.npz`。详细结果见 `docs/GS_OPD_PROGRESS.md` 本轮顶部记录。

## 前一局部目标：Pot 2局部安全放置已完成；完整任务继续留待后续阶段（2026-10-25）

本轮从严格恢复的真实t=642 Snapshot续接到t=649。源模型SHA`1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752`、Integration State SHA`728279cabf061d58d4cbc8994a1f8c6809e55fd30d3dd0a035a3e5712ac7e906`、源Snapshot SHA`21e1f64cdbe53e0dbeb52e5893231b766edbf9230230e086258f03ede67b7534`已核验。t=642的真实阶段事件与双指抓持通过后，TaskPhaseController只重建已归档证明的release阶段。

此前约0.087 mm EEF派生位姿差异现在保留为诊断，不再单独阻止安全动作；只有Runtime Snapshot和Integration State精确相同、每子步安全以及Native/接触证据均通过时才接受。数值门限、安全阈值、关节限位和BDDL谓词均未放宽。

已连续提交7个新OSC_POSE动作（t=642→649）；Pot 2在t=644满足原生`On(moka_pot_2, flat_stove_1_cook_region)=true`并解除抓持，t=649通过5样本稳定检查。新段175/175物理子步完整；与t=590→642的同Episode链合计59动作、1,475/1,475子步，Native Safety和Operational Safety均通过，违规0。合法指尖最大穿透0.282 mm；全链robot–Pot 2、Ginger Beer、Place Box最小间隙分别4.242、32.678、52.850 mm，Strict-10mm附加统计不通过。

当前状态是**已验证的Pot 2局部安全成功后缀**，不是完整任务：Pot 1 On=false，task_success=false，reward=0、done=false；t=649局部里程碑在标准1000步与demo-only 1800步内，但完整标准Benchmark/demo任务成功均为false。完整Round 0种子仍未取得，未训练LoRA。

归档目录：`/opt/data/private/gs_opd/rollouts/l0t2_execution_aware_pot2_release_continue_t642_20261025/`；机器可读总审计`posthoc_acceptance_audit.json`，续跑结果`suffix_result.json`，Snapshot `snapshots/main_timestep_000649.pkl`。后续待办是在用户授权后从t=649真实状态继续Pot 1阶段，或为Round 0种子从t=10启动一条连续全任务Episode；不把局部后缀拼成完整种子。历史数据均保留。

## 当前局部目标：prediction/commit失配后停止在t=590（2026-10-24）

本轮按获批方式复用t=398的193点路线，从严格恢复的t=463状态开始滚动预演和真实提交。t=463→589共126动作、3,150子步，逐动作预测/实执行认证通过，Pot 2保持双指抓持，阶段进入`precise_placement`。从完全匹配的同Episode t=589 Runtime Snapshot恢复后，TaskPhaseController根据真实状态重建，游标191，安全策略阈值及禁触对未改变。

t=590提交的第1个precise-placement动作本身完成25/25子步监测：Native/Operational动作安全检查通过，0禁触/自碰撞/未许可接触/关节违规，finger-only抓持仍稳定，Pot 2非手指间隙4.378 mm；Strict-10mm附加项失败。随后严格状态比对报`prediction_commit_eef_pos_mismatch:8.663630374972087e-05`（EEF位置最大绝对分量差，比较限值仍为1e-8 m），立即停止。此动作改变后的Pot 2仍未满足原生On谓词，reward=0、done=false。

最终预算127/450，未运行RRT；任务成功、Pot 2局部放置、标准1000步、demo-only 1800步、verified safe suffix、Round 0种子均未取得。该链条不进入Pot 1，不训练LoRA。停止是执行预测与真实状态的一致性失败，并非观测到碰撞或关节限位违规。归档路径为`/opt/data/private/gs_opd/rollouts/l0t2_execution_aware_pot2_resume_t589_20261024_231708/`；完整人读/机器报告见`t463_to_t590_stop_report.md`、`continuation_chain.json`和`prediction_commit_mismatch_stop.json`。

正式入口`research/gs_opd/repair/execute_t463_execution_aware_pot2_placement.py`已补上预测终态Snapshot在动作提交前持久化，避免下一次有授权的运行再次丢失预测侧证据；原0.18 rad qpos参考值、Native/Operational判据和比较容差均未修改。本次停止后不从t=590继续；下一项是只读诊断该EEF状态失配，之后再讨论续跑预算与起点。历史数据保持不变。


## 前一局部目标：从t=463真实持物状态重建Pot 2放置路线；按跟踪偏差fail-closed停止（2026-10-24）


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

---

# 当前局部目标更新（2026-10-26）：L1T2成功Episode来源链重建完成

本轮已从L1T2 seed=0/init-state=0 fresh reset连续重放run19→46→47→48→52→53的1335步真实动作，再在同一个MuJoCo实例中接续77步已验证稳定释放后缀。完成1412个控制动作、1413组双相机Observation、35300个25子步完整监测样本；源Snapshot的6个段边界最大Integration State差为`1.2862e-10`，都通过明确的`1e-7`绝对容差核验，t=0 reset则与归档状态逐位一致。接缝bitwise不等的事实保留在机器报告中。

最终`t=1412`原生Pot1 On、Pot2 On、Stove Turnon均真，reward=1、done=true。77步后缀跨越t=1335的原生done：使用demo-only horizon=1600，同一环境继续执行真实动作，未改BDDL/成功判据、未写入物体状态；因此它不是标准1000步Benchmark协议。Settling 500个物理子步谓词全真、Pot2零位姿跨度，末态和整个Settling均无手爪–Pot2接触。后成功阶段仍记录到678个手爪–Pot2接触子步（t=1336–1364与1366–1367），不做安全认证。

完整数据及hash清单位于 `/opt/data/private/gs_opd/rollouts/l1t2_success_episode_reconstruction_20261026_attempt2/`。可用于后续LeRobot转换的Action/Observation素材已保存，本轮没有转换或训练。当前局部任务完成；下一步如继续GS-OPD，应先对该数据执行独立Safety Reconstruction/接触分层审计，再由用户安排LeRobot转换或其它场景任务，不把task-success样本自动并入安全训练集。

## 当前局部目标更新（2026-10-26）：Round60 RRT→OSC短段诊断完成

上节的只读阶段性结论由一次18步短段验证补充：从Round60 t=2589精确Snapshot开始，唯一真实MuJoCo验证执行到t=2607，共450/450子步，模型、BDDL、指令和初始Integration State均匹配。由于Round60原始选中`qpos_path`未归档，本次以相同起始状态、seed和规划配置重新生成一条RRT路线：65个7D qpos/EEF参考点，压缩索引为0/15/29/41/51/59/64；完整重生成路径、每步OSC局部目标和真实qpos均已归档。新旧前18步动作及EEF状态逐步完全一致，但原始RRT数组无法逐元素比对。

t=2607命令位移`[-1.35,-2.45,-7.50] mm`与实位移`[+2.52,+0.73,-0.81] mm`几乎正交（余弦0.040），只有X轴分量方向相反，尚不能称整体位移反向。RRT最近点qpos偏差为0.010–0.191 rad，EEF路径偏差1.4–16.1 mm；这些最近点距离不是时间同步误差。持续Geom74 Hand–Geom117 Pot2接触有450条子步证据，深度0.157–0.179 mm、法向力1.851–2.154 N，t=2589已存在且到t=2607未加重。其后续阻挡因果和t=2607后的完整反向原因unknown。

位姿`position_m`更新现在会同步4×4矩阵，并已有刚体回归。执行器会持久化RRT完整7D参考、原始EEF路径、压缩映射及逐步实际OSC目标/关节状态。当前推荐下一步是局部增强OSC稠密路径跟踪与路径进度投影，暂不引入新关节控制器；如果改进后仍偏离规划走廊，再评估动力学感知的关节路径执行。

机器与可读报告：`/opt/data/private/gs_opd/diagnostics/round60_rrt_osc_interface_20261026_185951_481254/diagnostic_report.json`、`round60_rrt_osc_interface_diagnostic.md`；最佳物理Snapshot为`t=2607`的`best_progress_runtime_snapshot.pkl`。相关31项回归通过。本轮未扩展到其他任务、未训练、未放宽安全规则。
## 当前局部目标更新（2026-10-26）：Round60停滞只读诊断完成，暂停新搜索

Round60的41步/1025子步归档已完成离线对照；Round60/61/62不因本次报告而增加新的RRT候选或物理搜索。首个可观察到的OSC命令与实际EEF运动失配发生在t=2607，后24步平均EEF位移0.742 mm，23/24步实际平移方向与命令相反。目标RRT局部Waypoint和选中7D `qpos_path`未归档，所以精确局部误差及实际关节轨迹相对RRT轨迹均为unknown。

代码审计发现执行器将RRT输出降为压缩EEF路径后以OSC重新跟踪，未保留关节构型参考；这是优先排查项，不是已经证明的唯一失败原因。Geom 74 Hand–Pot2 Geom 117接触从t=2589起始Snapshot持续贯穿1025子步，最大穿透0.179 mm；停滞期间穿透与法向力没有恶化。接触可能限制运动，因果仍unknown；安全策略保持原样。

诊断JSON与Markdown保存在Round60归档`evidence/round60_stagnation_diagnostic.{json,md}`。后续如继续，先在不执行物理分支的工作中补齐RRT qpos/EEF参考、压缩Waypoint索引和逐步局部目标记录，并修复keypose位姿matrix同步；得到可复核实现后，再由用户安排下一次物理验证。保留L1T2完整稳定Episode为只读经验源；不启动训练。

## 当前局部目标更新（2026-10-26）：Human L0T2运输预算已修复，单次续接止于RRT规划失败

运输执行不再受固定360步截断：预算按完整路线长度、任务尚未完成的后续对象和demo horizon余量计算。Human L0T2的t=575失败Snapshot已精确恢复；Pot1关节7初始越限0.000206368 rad。8个真实OSC抬升动作把越限降至0.000127443 rad，仍未恢复合法，因此按策略恢复到SHA `48b0c9faae531e1fd4654d2daaeb7f31fd315b86fb033f2ea8cbabe122bfa85a`的合法双指持Pot1 Snapshot；终态与该Snapshot Integration State相同，关节均在限位内。恢复动作中的双指抓持持续有效。

在该合法Snapshot上只进行了一次放置/运输规划，实际RRT在12秒后超时；尽管本次CLI请求8秒，执行器未将该上限传入主运输RRT，现已修复传播。未开始真实运输、Pot1放置/释放或Pot2阶段，原生task_success为false，完整Episode新增数为0。没有追加相似搜索。8步真实动作、200个子步样本、接触记录和双相机视频在`/opt/data/private/gs_opd/rollouts/gs_opd_human_l0t2_transport_budget_recovery_20261026_retry2/`。每步25子步已采样，但全几何监测状态为incomplete，故不宣称该段安全认证。最可能的当前物理阻塞是从合法持抓姿态到目标区域的RRT规划超时；单独限位OSC恢复不足，但合法Snapshot回退有效。后续可复用预算逻辑、限位诊断和t=215持抓前缀，不能把这次搜索称为运输或任务成功。


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
