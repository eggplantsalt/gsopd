# GS-OPD 跨场景经验迁移与成本摊销报告（2026-10-26）

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
