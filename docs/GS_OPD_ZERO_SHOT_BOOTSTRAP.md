# 无示范任务搜索与安全重构：实现和实验记录

## 目标和当前状态

目标是从合法初态、当前任务定义和物理几何出发，在没有目标任务成功答案的条件下完成原生任务，再对真实成功轨迹定位风险、局部重构并继续任务，直至完整 Strict-Safe Success，最后在独立场景验证。当前 `GOAL_STATUS=INCOMPLETE`；L0T0 已实现无历史经验的 505 步原生 Task Success，尚无 Strict-Safe Episode、训练资格或跨场景闭环。

此前 attempt41 在 L0T0 达到原生成功，但其 manifest 指向同任务经验库，匹配了两个成功抓取模板且跳过 Student。因此它只可作为经验复用的历史对照，不能作为无示范 Bootstrap 证据。其后严格抓姿筛查也使用了历史抓持先验，不能用于证明冷启动。

## 冷启动实现

`trajectory_generator_v3.py`、`task_success_first_executor.py` 和 `task_success_first_grasp_search.py` 的 `--cold-start` 禁止成功经验库、Keypose、历史成功参考、先前搜索报告、部分任务恢复输入、手工指定候选与 LoRA；候选从零索引开始，并校验合法 t0 Snapshot。执行器和搜索器现在在恢复之前把快照完整 integration state 与同一任务、seed、state 的新鲜 reset 逐项比较，并要求时间步为零。报告记录 `target_success_demonstration_used=false`、`target_success_keyposes_used=false`、`official_demonstration_used=false`、`historical_actions_replayed=false`、`initial_policy=base_pi05`；这些标志描述运行时输入。

本地 Hugging Face 缓存中 `RLinf/RLinf-Pi05-LIBERO-SFT` 的 `refs/main` 与 checkpoint 下载元数据均为 revision `45ccfcc4e28634f1576ebf78cab0fbe2fd82432d`，下载权重的 metadata SHA 与实际 checkpoint SHA 一致。该模型是项目原有的 LIBERO SFT 基座，运行时未加载本项目 LoRA；它不等同于未经 LIBERO SFT 的 `pi05_base`。若验收将“原始预训练”严格解释为后者，当前两轮只能作为 SFT 基座上的机制实验，必须使用符合该解释的 checkpoint 重跑后才能作为验收证据。

Student 先进行真实动作并检测停滞，专家搜索从合法状态读取活动 BDDL、物体碰撞几何与指垫几何，生成对象相对抓姿并做 IK 和 RRT。第一轮暴露了 RRT 超时后没有通用候选可执行路线的问题。现在仅对当前 BDDL 几何来源、存在 IK 且 scratch 预检未改变 live state 的候选增加有限高位 OSC 接近路线。此路线只服务 Stage A 的任务可行性搜索，实际动作与子步仍须记录，不能继承安全认证。连续失败需根据实际失败原因更换接触面、路线或操作阶段，不重复相同搜索答案。

## 当前安装任务的覆盖矩阵

下面列出 `obstacle_avoidance` 和 `obstacle_avoidance_human` 两套、各 L0–L2 的 30 个 BDDL 文件：每种任务结构各 6 个场景。此表不代表其他 benchmark 的技能或物理验证已完成。目标表达式在六个场景中相同，场景几何和约束仍需逐场景验证。

| 任务族 | 原生目标 | 多对象 / 设施操作 | 当前代码覆盖与物理证据 | 无示范完整成功 / 安全重构 |
|---|---|---|---|---|
| 黑碗到盘 | `On(bowl, plate)` | 单物体 | `on` 几何抓取、运输与放置实现；L0T0 第二轮 505 步原生成功 | 无历史经验 Task Success 已验证 / 安全重构未验证 |
| 书到收纳盒 | `In(book, caddy_back)` | 单物体 | `in` 放置路径存在；当前目标场景无示范物理结果未验证 | 未验证 / 未验证 |
| 两个摩卡壶到炉灶 | `On(pot1, cook_region) ∧ On(pot2, cook_region) ∧ Turnon(stove)` | 双物体、开关 | 未满足 `on/in` 目标按 BDDL 顺序选择；新接入设施原生关节范围的开关算子，整个依赖链尚无冷启动物理验证 | 未验证 / 未验证 |
| 汤和奶酪到篮子 | `In(soup, basket) ∧ In(cheese, basket)` | 双物体 | 对象目标顺序选择存在；重规划及保持已完成目标尚无冷启动物理验证 | 未验证 / 未验证 |
| 杯子进微波炉并关闭 | `In(mug, heating_region) ∧ Close(microwave)` | 容器可达性、关门 | 有基于活动设施几何的 OSC 推动函数；打开前置条件、关门和完整依赖链尚无冷启动物理验证 | 未验证 / 未验证 |

此集合实际目标运算符为 `And`、`On`、`In`、`Turnon`、`Close`。当前 `pending_object_goals` 选择未满足的 `on/in`，随后执行器可调度设施谓词；一般逻辑依赖和同一冷启动多阶段物理执行尚未证明覆盖，不得把未支持谓词静默当作普通放置。表中“代码存在”不代表技能已完成物理验证。

另用 `research/gs_opd/repair/task_coverage_audit.py` 对安装目录全部 16,405 个 `.bddl` 文件进行只读统计（其中包含大量生成变体，并非 16,405 个独立任务）。目标谓词出现次数为 `On` 9,818、`In` 8,190、`Turnon` 1,230、`Close` 829、`Open` 412、`Turnoff` 1、`CheckGripperContactPart` 15。`affordance` 的 15 个文件使用含嵌套接触部位参数的 `CheckGripperContactPart`，当前没有该操作算子；`reasoning_safety` 的 15 个文件中有 8 个没有 `:goal`，需要单独确定安全推理或拒绝的成功语义。解析错误为 0，缺失目标与未实现算子都保留在机器报告 `/opt/data/private/gs_opd/overnight/gs_opd_zero_shot_bootstrap_20261028/installed_bddl_goal_coverage.json` 中。这是文件定义统计，不是任务成功率。

新增 `inspect_native_goal_coverage` 逐项记录完成状态、谓词和算子覆盖，校验谓词元数及显式布尔值。执行器现可调度原生 `open/close/turnon/turnoff` 设施目标：复用设施关节 scratch 规划与 OSC 推动，开关必须有设施声明的目标范围，否则报错；未知算子仍报告覆盖边界。L0T2 新鲜初态只读探针正在核查开关几何。容器打开前置条件、多阶段同一冷启动链、开关技能及后续物理效果仍未完成；纯逻辑分类和选择不构成这些技能的物理验证。

## 实验归档与复现

独立实验根目录：`/opt/data/private/gs_opd/overnight/gs_opd_zero_shot_bootstrap_20261028`。基准为 `obstacle_avoidance/L0T0`、state0、seed0，t0 Snapshot SHA-256 为 `bc86043308518e857703b319b2c309e4e9ca62b332fc939be20235991610cba7`；模型几何 SHA 为 `085817d128dd03511cfd14cd810a6c035e85258e0b8260881f3f0e0f3db2e840`，BDDL SHA 为 `b631d2f9be0884f5fb908abc0cf271e34a9c368085252dc48bbd5207d5b7fe12`。所加载 checkpoint 权重 SHA 为 `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`。

第一轮 `l0t0_seed0_cold_start_01`：Student 32 个真实动作、800 子步；3 个活动几何候选均有 IK，但每个 RRT 在 12 秒预算耗尽，旧代码没有允许通用几何高位兜底，专家物理分支为 0。任务未完成。这一结果引出上述路线机制改动。

第二轮 `l0t0_seed0_cold_start_02`：通用几何候选 `geom112_jaw0_approach2_+1_flip+1` 使用高位 OSC 路线，真实双指稳定抓持并抬升 0.036302 m。完整保留轨迹由抓取 163、Student 交接 32、放置 310 步组成，共 505 步和 12,625 物理子步，原生 reward=1、done=true；此前失败并回滚的 Student 32 步不计入成功轨迹。独立审计为 `native_success_episode_audit.json`。独立新鲜 reset 核验的 integration state 与输入 t0 完全相同，报告为 `independent_fresh_reset_t0_verification.json`。这是 No-Prior-Experience 和 No-Target-Demonstration 的物理 Task Success，不是 Strict Safety 成功。

两种设置分别报告：No-Prior-Experience：L0T0 505 步成功，L0T1 正在验证；No-Target-Demonstration：同一个 L0T0 无先验运行也满足该条件。允许使用可追溯跨场景经验、但禁止目标场景答案的独立运行尚未实现或验收，不能把前者说成已经验证经验迁移。

完整成功轨迹的事后 BDDL 风险扫描见 `observed_bddl_risk_windows.json`：首次观察到的风险为动作 137 的子步 9，非授权手指与碗距离 9.941 mm；窗口延续到动作 505。全部 12,625 子步都缺少禁触物体对距离证据，Teacher 规则未验证，因此不能声称该观察是形式上的最早风险。V3 冷启动成功后现在自动调用定位器；自接触、其他场景接触及关节限制尚未独立覆盖，定位结果必须保持 incomplete。

独立 L0T1 书到收纳盒的 No-Prior-Experience 验收归档为 `l0t1_no_prior_01`；其输入 Snapshot 从活动场景新鲜 reset 捕获，物理动作数为零，原生目标初始为 false，未加载目标场景历史答案。attempt41 的隔离物理回归也在运行；原历史归档的 669 步和本次 505 步已分别重新审计。

L0T1 第一轮现已结束，原生任务失败。Student 32 动作回退后，专家执行 249 动作/6,225 子步，接近时夹爪掌部接触书，书相对初态移动约 23 mm，60 步内未到达动态目标。3 次预检仅生成 1 个可执行物理候选，其他预检受 RRT 超时或 IK 失败限制；完整记录位于 `l0t1_no_prior_01/v3_run/v3_report.json`。冷启动接近现增加 15 mm 未抓持目标位移早停，第二轮 `l0t1_no_prior_02` 在相同合法初态从零生成候选，扩大到 8 次预检及最多 3 物理分支，不读取首轮报告或轨迹；其结果尚未完成。

抓取记录中的 Stage A 几何策略把所有非目标场景对象作为诊断禁触对象，并把目标与这些对象全部组成禁触对。活动 BDDL 的实际约束仅指定机器人与香水瓶/白盒以及目标碗与香水瓶/白盒的禁触；因此第一子步碗与炉灶的初始支撑接触被诊断策略误分类为 forbidden，并非该 BDDL 规则证据。只读新鲜 MuJoCo 初态对照中，原诊断策略为 `unsafe`、包含 20 条碗—炉灶 forbidden-object 接触；活动 BDDL 策略状态 `complete` 且 forbidden-object 接触 0，0 物理动作，报告为 `l0t0_seed0_cold_start_02/t0_bddl_policy_comparison.json`。该对照未包含 Teacher，不能证明完整严格安全。

旧成功轨迹的子步记录中 `carried_object` 为 null：它记录对象对的实际接触，但没有完整记录目标物体到禁触对象的逐子步 10 mm 距离。这些记录无法直接证明完整 Strict Safety，也不能从笼统 `unsafe` 状态直接确定首个正式风险。新增 `object_pair_clearance.py` 并接入 Bridge，对显式 BDDL 禁触对逐物理子步测量完整几何集合，10 mm 以内标违规、缺失测量标不完整；冷启动使用 BDDL 来源的诊断规则，经验复用基线仍使用原策略。Stage A 规划预检关闭这一额外严格距离门禁，真实执行的记录仍保持 10 mm 监测。

新归档 `bddl_pair_monitor_probe/probe_report.json` 用新鲜合法 t0 执行 1 个真实 MuJoCo 动作，25 个子步均完整测得碗—香水瓶和碗—白盒的碰撞几何最短距离及对应 Geom；初始支撑没有被判为 BDDL 禁触。此探针仅覆盖 BDDL 规则，Teacher 语义、完整成功 Episode 与严格安全验收仍未证明。

第二轮入口（输出目录必须为新目录）：

```bash
/workspace/zy_workspace/RLinf/.venv/bin/python -u research/gs_opd/repair/trajectory_generator_v3.py \
  --backend auto --cold-start --output-dir "$NEW_RUN_DIR" \
  --benchmark obstacle_avoidance --level 0 --task-id 0 --init-state 0 --seed 0 \
  --instruction 'pick the akita black bowl on the stove and place it on the plate' \
  --snapshot-root /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt8_snapshot_root \
  --demonstration-horizon 1000 --maximum-physical-branches 3 \
  --maximum-rrt-preflights 3 --rrt-time-budget-s 12.0 --candidate-start-index 0
```

V3 现为引擎进程组设置可配置的墙钟上限（默认 3600 秒）；超时终止引擎及其子进程并保留已写出的记录。引擎启动后的异常报告将物理动作数标为未知，等待根据归档核算，不能错误报告为零。当前第二轮由修改前的 V3 启动，故另启动独立 watchdog，以原进程启动时刻计 3600 秒并核验 PID、进程组及命令身份；超时写入 `watchdog_timeout.json` 后终止该进程组，已有 durable 记录保留。

五组相关测试合计 126 项及 15 个子测试通过，修改模块 `py_compile` 通过。上述测试不能代替完整物理验收。

保护基线固定为已 push 的 `1d63574e5ce1aa60b63f093da09665c4d8c741d5` 与 attempt41 历史归档。重新读取原动作记录的回归审计确认 attempt41 为 669 动作/16,725 子步；修复 V3 漏计 Student 接续后，新无先验轨迹为 505 动作/12,625 子步，证据为 `v3_baseline_and_cold_start_regression.json`。此外，新归档 `baseline_attempt41_regression_01` 正从合法 t0 物理复跑原候选与相同 SHA 的经验库副本；它是经验复用对照，最终结果尚待报告，不计为无目标示范。

根据仓库 `docs/source-en/rst_source/examples/embodied/sft_openpi.rst`，官方原始预训练权重位于 `gs://openpi-assets/checkpoints/pi05_base/`。该前缀共 29 个对象、12,441,749,581 bytes，当前正通过 Google Cloud Storage Python 客户端下载至 `/opt/data/private/gs_opd/checkpoints/pi05_base_jax`，逐对象核验 generation、大小和 CRC32C，外层时限一小时；记录为 `/opt/data/private/gs_opd/checkpoints/pi05_base_download.log`。完成后仍须按官方流程转换并核验模型、动作适配和 norm stats，才可用来进行严格预训练口径冷启动。

## 待完成

先取得第二轮的原生任务终态、动作连续性、冷启动来源和监测证据，按实际最早失败定位下一技术假设；如果任务成功，立即审计首次严格风险及可恢复 Snapshot，并接入安全重构和真实任务续接。之后独立验证真实危险轨迹的局部修正、完整严格安全成功、成熟数据验收和另一场景。完整搜索成本、哈希与失败归因随实验完成继续补充。
