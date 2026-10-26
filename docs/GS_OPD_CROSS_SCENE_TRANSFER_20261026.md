# GS-OPD 跨场景经验迁移与成本摊销报告（2026-10-26）

## L2T2跨Level续接（run55→Round60/62）：保留Pot2持抓前缀，尚未完成任务

从run55的t=2589 Pot2双指稳定持抓Snapshot恢复，模型与BDDL SHA匹配。比较GS-OPD自有L0T2完整成功关键姿态和L1T2完整稳定成功关键姿态，不回放源轨迹动作。首轮4个候选中，L0来源的`region_layout_20 / yaw+180 / h=0.38 m`唯一路线通过预放置整机/持物RRT与终端IK；L1来源路线未通过RRT预算。真实RRT物理重规划11.190秒后执行41个OSC动作，t=2630因预放置Waypoint连续24步停滞而停止。Pot2仍双指抓持并移动约80.3 mm，原生状态仍2/3（Pot1 On、Stove Turnon真，Pot2 On假），task_success=false，reward=0、done=false。

| 搜索指标 | 结果 |
| --- | ---: |
| 新增Student / 专家7D动作 | 0 / 41 |
| 真实物理Snapshot分支 | 1 |
| Round60 RRT实际调用 / 可执行候选 | 8 / 1 of 4 |
| Round62备用RRT实际调用 / 新增可执行不同路线 | 4 / 0 |
| RRT调用返回可行 / 实际筛过姿态有IK解 | 2/12（同路线、不同起点）/ 6/6 |
| RRT预筛累计规划时长 | 166.663 s |
| 实际路线RRT规划时长 | 11.190 s |
| 完整3/3成功 / 稳定释放 | 0 / 未到达 |

Round62从Round60失败Snapshot精确恢复；由于固定预算内没有新的不同路线同时通过预放置RRT与末端IK，未启动第二个物理分支。Round61误用默认L0T1快照根，在模型SHA门禁处停止，执行0动作；Round62显式指定L2T2初始化根后恢复通过。

每个Round60动作均保存25/25子步接触样本和Geom身份（1025/1025子步、52,275条原始接触身份记录、19组Geom对）；其中Hand–Pot2非指接触出现于1025个子步，必须保留为风险证据。安全认证false。完整指标与快照哈希见[机器报告](/opt/data/private/gs_opd/rollouts/l2t2_crosslevel_keypose_20261026_round60/task_success_first_executor_obstacle_avoidance_L2T2_20261026_175558_280342129/evidence/l2t2_keypose_continuation_summary.json)和[Markdown报告](/opt/data/private/gs_opd/rollouts/l2t2_crosslevel_keypose_20261026_round60/task_success_first_executor_obstacle_avoidance_L2T2_20261026_175558_280342129/evidence/l2t2_keypose_continuation_summary.md)。最佳可恢复状态为Round60 t=2630 Snapshot（SHA-256 `e489a1e990b41876ced9432cdcaa94f409474d56b1f4292db89018fa33cfeefb`）。

---

## 历史汇总：2026-10-26 Keypose-Driven Planning（L1成功，L2阶段性迁移）

### 验证结论

仅用GS-OPD自有L0T2完整成功Episode生成的关键姿态经验（source `moka_native_success_keypose_v1`，真实源动作不回放），实现了物体坐标系与目标Region坐标系关键姿态的提取、重定位，并用现有RRT/OSC/Runtime Snapshot闭环执行。没有使用官方示范、CuRobo或训练；严格Safety Reconstruction保持不变。

L1T2从run48已验证Pot2持抓Snapshot精确恢复，分两段接续运行run52→run53，合计478个真实专家OSC动作、2个物理分支、5个关键姿态RRT规划尝试（累计记录规划时长14.376秒）。最终活动BDDL的Pot1 On、Pot2 On、Stove Turnon均为true；`task_success=true`, reward=1, done=true。它是**从已有部分任务前缀恢复的完整原生成功**，不是从reset冷启动的完整Episode。相同run48持抓起点的旧方法消耗920动作、3个分支但未成功：Keypose路径减少47.99%动作和33.33%分支，首次把此L1前缀完成到原生成功。旧run48规划墙钟时间未保存，无法做规划耗时对照；本次L1完整任务的进程墙钟时间也未有单调计时记录，保持unknown。安全认证为false，子步监测不完整。终止动作发生在`transport_place`：Pot2仍被双指夹持，Bridge因原生`done=true`结束；release与retreat关键姿态已经提取但本次没有执行或验证。

L2T2从同布局、同模型/BDDL身份且Pot2双指抓持、Pot1 On的run44分支Snapshot恢复。run54运行125步，抬升目标RRT因当前场景IK无解，真实OSC fallback后24步停滞；run55从run54终态Snapshot跳过近似完成的lift，再运行111步到高位西侧关键姿态附近，随后24步停滞。两段合计236个专家动作、2个分支、2次有界IK/RRT尝试；末态Pot2仍双指抓持，Pot1 On与Stove Turnon为true、Pot2 On为false，2/3谓词，reward=0、done=false、`task_success=false`。所有阶段仅使用真实MuJoCo动作，安全监测不完整，安全认证为false。

### 基线与复现

| 场景/方法 | 原生任务成功 | 专家动作 | 物理分支 | RRT尝试/规划时间 | 复用经验 | 最大进度 |
| --- | --- | ---: | ---: | --- | --- | --- |
| L1T2 run48旧运输 | 否 | 920 | 3 | 规划时间未归档 | L0抓持/阶段经验 | 持抓运输受阻 |
| L1T2 Keypose run52+53 | **是（恢复前缀）** | 478 | 2 | 5次；14.376秒累计规划时长 | L0T2物体与Region相对关键姿态 | 3/3原生谓词，reward=1/done |
| L2T2 Keypose run54+55 | 否 | 236 | 2 | 2次；均IK无解，OSC fallback | 同一L0T2关键姿态经验 | Pot2持抓到高位西侧，2/3谓词 |

L1逐步证据：`/opt/data/private/gs_opd/rollouts/keypose_driven_L1T2_20261026_run52/` 与 `.../keypose_driven_L1T2_20261026_run53_suffix_replan/`。run53终端`t=1335`的NPZ、压缩日志及双相机视频都含118步成功动作；另补齐耐久JSONL末步与终端视频帧，并保留修复前run_report副本和恢复sidecar。原终止快照按字节另存为`native_task_success_terminal_runtime_snapshot.pkl`；原失败命名文件未删除。

L2逐步证据：`/opt/data/private/gs_opd/rollouts/keypose_driven_L2T2_20261026_run54/` 与 `.../keypose_driven_L2T2_20261026_run55_suffix_replan/`。run48完整性索引：`/opt/data/private/gs_opd/rollouts/task_success_first_cross_level_L1T2_20261026_run48_second_object_from_partial_prefix/task_success_first_executor_obstacle_avoidance_L1T2_20261026_134731_258088424/run48_archive_index.json`；其root报告未写出，interruption report显示3个分支、920动作、最终native查询不可用，第三支缺最终Snapshot/NPZ/视频，均显式标注缺失。

旧Task-Success-First暴力搜索已暂停且当前没有活动物理进程。L0T2自身源轨迹安全成功为false；所有新搜索轨迹`safety_certified=false`。相关机器统计追加在[JSON报告](GS_OPD_CROSS_SCENE_TRANSFER_20261026.json)的`keypose_driven_transfer_update`。

## 验收结论

本轮从GS-OPD自有 obstacle_avoidance/L0T2 原生成功Episode检索并迁移Moka Pot经验。L1T2、L2T2均命中L0抓取/阶段模板；L1真实验证稳定双指抓持并抬升34.54 mm，L2在4次IK/RRT预检后没有进入物理分支。**本轮新搜索完整原生成功Episode为0条。** L1抓取冷启动实现了子技能迁移，但完整任务搜索成本随经验积累下降尚无证据；L1放置仍耗费较多动作而未稳定满足目标。

官方LIBERO-Safety示范数据未下载、读取或作为先验；没有训练、Safety Reconstruction、LeRobot转换、任务专用完整脚本或任务专用人工调整。所有搜索Episode均 safety_certified=false。详见[机器报告](GS_OPD_CROSS_SCENE_TRANSFER_20261026.json)。

## 场景指标

| 场景 | 原生完整成功 | 首次原生成功耗时 | Student / 专家动作 | 物理分支 / RRT | 复用历史技能 | 结果 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| L0T2源Episode | 历史成功（t=1459，reward=1，done=true） | 旧归档未测墙钟时间 | 1460条源动作 | 不计作本轮搜索 | 经验来源 | GS-OPD自有成功轨迹，不算新迁移成功 |
| L0T0 | 否 | 无 | 8 / 0 | 0 / 4 | 无 | 规划预算耗尽，目标未触达 |
| L0T1 | 否 | 无 | 8 / 0 | 0 / 4 | 无 | 3次规划超时、1次当前场景有界IK无解 |
| L0T3 | 否（0/2） | 无 | 8 / 0 | 0 / 4 | 无 | 3次规划超时、1次有界IK无解，未进入物理分支 |
| L0T4 | 否（0/2） | 无 | 8 / 440 | 1 / 4 | 无 | 局部表面接近重规划201动作预算耗尽；未抓稳，343/440动作子步监测不完整 |
| L1T2 | 否 | 无；首次迁移抓取成功158.77 s | 0 / 905 | 1抓取分支；4抓取预检+2放置RRT | L0 t248抓取、t239阶段序列 | 抓取抬升成功；两次放置续跑未确认稳定原生目标 |
| L2T2 | 否 | 无 | 0 / 0 | 0 / 4 | L0 t248抓取、t239阶段序列 | 2次有界IK无解、2次规划预算耗尽，未执行物理动作 |

time-to-first-success只对原生完整任务计数；L1的158.77秒是抓取技能成功时间，不是任务成功时间。L1的905专家动作由182步抓取搜索、run19耐久日志中的318步放置动作、run21的405步放置续跑组成。run19摘要漏计了318步，故以逐动作耐久JSONL计数纠正。run21在settle阶段预算异常退出，最后归档动作task_success=false，且当时没有导出终态Runtime Snapshot。run21精确程序耗时未记录；目录创建到报告写出间隔为975秒，是归档级近似值，不是精确进程计时。

L2新试次原生状态为1/3谓词满足（Stove已开，两个Pot On目标均未满足），调度器按BDDL顺序选择 On(moka_pot_1, flat_stove_1_cook_region)。搜索尝试复用L0成功模板和阶段序列，但2/4次没有有界当前场景IK，另2/4次耗尽8秒规划预算；没有发生物理动作，因此只能说明规划预检未取得路线，不能据此宣称整个L2目标不可达。

## 经验库和搜索器改进

- 经验检索改为优先完整原生成功来源，其次对象原生目标成功和阶段完整度，再参考跨场景验证数。L1/L2都选中L0 moka_pot:stable_grasp:t248 与 object_transfer_cycle:moka_pot:t239。
- L1真实稳定抓取模板自动加入run19隔离经验库副本并标记 source_native_task_success=false；本轮又将两条此前验证的L2抓取技能及L2部分转移技能，与L1模板原子合并到持久v2经验库。L0完整成功抓取/阶段模板均保留并优先排序；所有新条目仍 safety_certified=false。合并前备份位于 /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2_pretransfer_merge_20261026.json。默认经验库已指向该v2文件。
- 单任务Executor现在与批处理器一致，按原生BDDL顺序选择首个未满足对象谓词，多目标Snapshot恢复无需手工指定目标。
- 精确放置时若抓持丢失，控制器继续真实释放与原生谓词复核；release/settle步数异常会归档实际终态、Snapshot和物理前缀。
- 任务专用脚本、代码改动、场景人工位姿调整均为0；新增改动是通用调度、经验排序和异常归档。

## 证据和下一步

- L0T0/L0T1：/opt/data/private/gs_opd/rollouts/task_success_first_cross_skill_L0T0_L0T1_20261026_run18/batch_report.json
- L0T3/L0T4：/opt/data/private/gs_opd/rollouts/task_success_first_cross_skill_L0T3_L0T4_20261026_run23/batch_report.json
- L1迁移与成功抓取：/opt/data/private/gs_opd/rollouts/task_success_first_cross_level_L1T2_20261026_run19_L0_success_migration/
- L1精确Snapshot放置续跑：/opt/data/private/gs_opd/rollouts/task_success_first_cross_level_L1T2_20261026_run21_L0_success_migration_auto_goal/
- L2迁移预检：/opt/data/private/gs_opd/rollouts/task_success_first_cross_level_L2T2_20261026_run22_L0_success_migration/

先修正并回归验证L1放置后的稳定判据与terminal Snapshot留存，再从可验证L1抓取前缀做有界放置重规划；L2下一试次应换用库中另一条完整成功抓取变换或通用几何候选，不扩展同一路线超时重试。随后再选L0T0/L0T1验证不同操作技能，仍以原生 task_success=true 唯一判定完整成功。

验证：48项GS-OPD RRT/阶段控制/技能库/搜索器回归全部通过；相关Python模块py_compile通过，git diff --check通过。
