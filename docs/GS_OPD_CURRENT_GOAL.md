# GS-OPD 当前局部目标

## 当前状态：L0 T2/T3/T4 原始 π₀.₅ Base 跨任务筛选完成，待复核（2026-10-22）

本轮暂停 L0T1 的 OSC/B-Spline 调试，不运行 L0T0，也没有重做 attempt6 抓取。用原始 Checkpoint `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft` 各独立运行 `obstacle_avoidance/L0` T2、T3、T4：init-state 0、环境 seed 0、VLA RNG seed 20261022、最多 400 个 VLA action。三次均执行 400 步、reward=0、`done=false`，不据预算耗尽推断模型完全不会任务。

- 三次均为 π₀.₅ Base-only，Checkpoint 文件 SHA-256 `4616e4c1966a5fb063b442ea5d69857f45e873222d839b794634d87d252ab9e7`；未加载 π₁/π₂ LoRA。模型仅收到 agentview/wrist RGB、EEF 位姿、夹爪 qpos 和原任务指令。BDDL、Body/Geom 及 MuJoCo 物理状态/接触只用于离线评价。未调用 OSC、Teacher、SAM3、轨迹修复或训练。
- **T2** `put both moka pots on the stove`：Moka pot 2 于第 119 步出现真实指尖接触，第 134 步观察到双指闭合接触，第 141 步满足连续三状态标准；稳定双指抓持持续 156 个动作状态（139–294）。抓持区间中物体根净位移 0.580 m、路径长 0.785 m；对象到 Stove Body 根部的最小距离由初始 0.527 m 降至 0.230 m，仍不是 BDDL `On` 完成。Moka pot 1 未移动；episode 最大位移 0.663 m。另观察到机器人与非目标 ginger-beer bottle 接触 36 个控制步，Bridge 子步记录峰值力 12.10 N。因此这是部分任务进展和唯一自主稳定抓取，不是安全成功轨迹或完整任务成功。
- **T3** `put both the alphabet soup and the cream cheese box in the basket`：两个目标均未出现双指接触或可测目标位移，EEF 到两个目标根部的最近距离分别为 0.247 m 和 0.223 m。观测到机器人与 ketchup 接触 368 步（峰值 31.74 N），与 carrot-juice bottle 接触 86 步（峰值 20.27 N）。
- **T4** `put the white yellow mug in the microwave and close it`：mug 根位移仅 0.000003 m，EEF 最近到 mug 根部为 0.248 m；microwave door joint 初始 `-1.6520 rad`，400 步变化 `0 rad`，与 microwave 无机器人接触。BDDL 初态已满足 `Open(microwave_1)`，本回合没有新的开门或关门进展。机器人与 dish-soap bottle 接触 307 步，峰值 48.05 N。
- 三个任务每个控制步均有 25/25 个 MuJoCo 子步接触记录，覆盖 10,000/10,000 个物理子步；NPZ 含实际 7D 动作、两相机帧、EEF/夹爪、物体根状态及逐步接触，视频各 401 帧、30 fps。接触数据是实测，不作 Safety Compiler 判定。
- 本轮只为元数据准确性让 Rollout 从 Bridge scene metadata 记录 task ID，并通过现有只读 `observe` 保存初始物理状态；VLA observation builder 和控制路径未变。T2 初次启动的未认证 TCP readiness probe 导致 Bridge 认证 EOF，模型虽加载但执行动作 0 步；已将此作为运维失败独立留档，随后 attempt2 完成有效 T2 rollout。
- **下一场景建议：T2**，因它是唯一观察到连续三状态真实目标双指抓持及较长真实搬运的场景。必须先针对已有 ginger-beer 接触进行安全筛选；本次原始回合不可直接标作安全轨迹。
- 总报告：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T234_base_pi05_rollout_400_20261022_attempt1/experiment_report.md`。T2/T3/T4 独立归档及 `experiment_report.json` 分别位于该报告末尾列出的运行目录。当前停止并等待复核。

### 已暂停的上一局部目标：L0T1/state0 Snapshot B-Spline 物理探索（2026-10-22）

这是此前 L0T1/state0 的 Snapshot B-Spline 局部探索记录：其拟从 attempt6 抓取快照继续 OSC 搬运并最终完成 `In(black_book_1, desk_caddy_1_back_contain_region)`。该分支目前暂停；既有 attempt6 抓取快照与 B-Spline 结果均保留。本轮跨任务筛选没有运行 L0T0、没有恢复 attempt6、没有做 OSC/B-Spline/搬运/放置或训练。

- 现有 18 条候选（Left/Right/Above 各 6）几何通过 0 条，全部因 `curve_clearance_bound_inconclusive:object_3` 被拒：证明不充分，不是已证明安全或碰撞。仅选择最短的 `trajectory-001/002/003`，每方向 1 条；候选 `passes_geometric_screen` 原值保持 false，未修改绕障曲线生成算法、障碍物半径或 Safety Compiler。
- 三个 Snapshot OSC 分支均未到绕障终点：Left 请求 604 步、执行 5 步，Book 与 study table（Body 1）碰撞后停止，Book 根位移 3.36 mm；Right 请求 604 步、执行 17 步，夹爪与 `white_place_box_1_main`（Body 27）碰撞后停止，Book 根位移 29.91 mm；Above 请求 604 步、执行 139 步，`robot0_joint7=2.900247` 超过上限 2.8973 后停止。没有任何候选被报告为安全成功。
- 已执行控制步的物理子步接触采样完整，分别 125/125、425/425、3475/3475；每步动作、Book 位姿、接触和 integration state，以及每候选视频均已保存。分支后 Restore 的 Integration State 与起始快照最大误差为 0。Safety Compiler 三条均返回 `unknown`（接触事件身份不完整），未解释为 `safe`。
- Teacher `rule_2`（夹爪–Caddy）仍为 `unknown`。本次隔离分支保留原始 Teacher，并将 Caddy Body 26 接触设为停止/拒绝条件；编译器输入只保留明确规则，完整 Teacher 未修改。
- 独立归档：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_snapshot_exploratory_bspline_physics_20261022_attempt2/`；含全部候选、18 行几何筛选 CSV、三条物理摘要、逐步 NPZ/JSONL、子步接触证据、视频和 SHA-256 清单。attempt1 在未知规则预检处阻塞，物理动作 0 步，单独保留。

### 已确认的抓取起点（attempt6）

- 当前实验目标是 `obstacle_avoidance/L0T1/state0` 的完整成功轨迹采集。本轮仅处理抓取阶段，不运行 L0T0、不执行 B-Spline 搬运/放置或训练。
- 原始 π₀.₅ Base 的 200 步抓取仍是失败历史（未接触 Body 23），但这不再阻塞当前路线：独立脚本化 OSC_POSE waypoint 辅助控制已实际抓住 reviewed Book Body 23 / Geom 91。抓取点沿 Book 长轴远离 place box 偏移 25 mm，接触判据保持原样。
- attempt6 共 408 个 OSC 控制动作，全部控制步具有完整子步接触监测；没有机器人–非目标物体接触。原有 `GraspStabilityMonitor` 的 3 个连续不同状态通过；另存首个稳定快照，在新 Bridge 进程恢复后 Integration State 最大绝对差为 0，并在恢复状态上再次通过 4 个连续不同状态。Book root 位移 8.34 mm，未通过放宽阈值认定成功。
- 可恢复快照：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_osc_assisted_book_grasp_20261022_attempt6/recovered_grasp_runtime_snapshot.pkl`。完整实际动作、双相机帧、物理状态、接触证据、视频、初次/恢复后快照及哈希清单均在同目录。尝试历史和索引见 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_osc_assisted_book_grasp_20261022_review_packet/`。
- 该分支曾计划在复核后从此快照继续真实 OSC 搬运并完成 Caddy 后侧隔间 `In` 放置；目前按最新指示暂停，是否恢复待后续复核决定。绕障 waypoint 不代表最终放置完成。

### L0T0 trajectory-001 Snapshot（历史实验，非当前任务）

2026-10-22 的 0.25 rad 显式阈值对照运行 108 个 Snapshot 动作后，因保持不变的 EEF 位置误差上限失败；较早一步还记录到夹爪手掌–炉灶 burner 接触，整体物理安全未通过。它不是当前 L0T1 目标，不再继续该分支。历史报告位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T0_round0_trajectory001_snapshot_rotlimit025_20261022_attempt1/`；0.20 rad 参考在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T0_round0_trajectory001_snapshot_speed0020_cap1200_20261022_attempt1/`。

### 上一 L0T1 C2 离线诊断（保留）

- 归档的 200 步动作序列中，EEF 曾接近并碰动 Body 25 绿色瓶子（89 条机器人–瓶接触），但没有接触 Body 23 Book；Book root 几乎未移动。40 个预测动作块与 NPZ 实际动作完全一致。目标方向/接近失败有外部证据，VLA 内部视觉目标错误仍是 unknown。
- 抓持代码按 Body 子树动态纳入 collision Geom，不限于语义 Geom 91；但运行时完整 Geom ID 清单未落盘，精确成员仍 unknown。默认关闭 Smoke flag 的 Student 推理和动作发送路径经静态审阅，没有发现被改变；没有运行正常 Rollout A/B。
- 报告、对照图、视频关键帧、轨迹/动作 CSV 和哈希清单在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_base_pi05_grasp_offline_diagnostic_20261022_attempt1/`。该诊断只描述原始 Base 行为；后续已获用户指示开展的 OSC 辅助抓取见本文件顶部。

## 前次 C2 抓取 Smoke 记录（2026-10-22）

### 运行结果与产物

- 一次有效 Smoke 使用原始 SFT Base（未加载 π₁/π₂ Adapter），针对 reviewed `object_1→Body 23/Geom 91` 抓取，最多 200 步；本次执行 200 步、reward 0、`done=false`。
- 未达到 3 个连续不同状态的稳定抓取条件。20 个状态确认夹爪闭合，但左右手指均无目标接触；另 181 个状态未满足闭合判据，状态查询不可用数为 0。失败分类 `gripper_contact_insufficient`；Book 基本未动，未记录 robot–book 接触。该诊断不是完整任务或整体安全判定。
- 视频、完整 NPZ、每步物理状态/接触、抓取监测报告及清单位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_state0_base_pi05_grasp_smoke_20261022_attempt1/`。
- C2 归档保留原始 Base 一次零动作启动错误及一次有效采集；该记录当时没有启动后续阶段。用户随后明确将目标转为 L0T1 OSC 辅助抓取，结果见本文件顶部。

## 前序局部状态记录（保留）

## 当前状态：L0T1 reviewed semantic binding 与 Compiler 输入静态验证已完成，等待复核（2026-10-22）

- 已按审核指示生成 object_1→Body 23/Geom 91、object_3→Body 25/Geom 133、object_2→Body 26/Geom 265 的独立 reviewed semantic binding 和 Smoke 可读的 per-object JSON。object_1 的 Teacher “white book” 与黑色实物外观冲突保留；object_2 低 SAM score 0.0986328125 保留；object_4–6 unresolved。
- 原始 Safety IR 被逐字节保留。Compiler 核心未改。完整 Inventory 为 288 Geoms；9 个空 geom_name 仅在单独兼容副本中替为可追溯的 ID 占位名。现有 `_resolve_target_binding` 对原 Inventory 按预期拒绝，对兼容副本的三条审核映射通过，并保留三条 unresolved。
- 规则绑定就绪情况与接触结论分开记录：object_1/object_2/object_3 绑定输入可用；rule_2 本身仍是 Teacher unknown；rule_3 与 ocr_1 还需完整运行时接触记录。没有任何轨迹 safe/violation 结论。
- 19 项静态检查通过；未连接 Bridge、未启动 MuJoCo、未运行完整 Compiler Smoke、VLM、SAM3、Student、规划、Rollout 或训练。结果目录：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_reviewed_semantic_compiler_input_20261022_attempt1/`。当前等待复核，不自动接入 Compiler 执行链。


## 当前状态：L0T1 三对象候选关联审计完成，待人工复核（2026-10-22）

- 归档证据支持 Book 原生 1024 与 Bottle/Caddy 使用的 512 图像来自同一物理初始状态：配套 512 RGB 与 Geom ID 哈希/数组完全一致；旧归档和原生分辨率采集的 289 项 integration state、机器人状态及所有物体姿态逐项一致。已验证 Safety IR 的运行清单也指向同一 raw RGB。
- 离线生成 `candidate_semantic_physical_bindings.json` 与 `manual_review_report.md`。候选关联：`object_1`→Candidate 1/Body 23/Geom 91（IoU 0.916855，Teacher “white book” 与黑色外观冲突，继续 unresolved）；`object_3`→Candidate 0/Body 25/Geom 133（0.984974）；`object_2`→Candidate 0/Body 26/Geom 265（0.958812，SAM score 0.098633，低分限制保留）。`object_4`–`object_6` 保持 unresolved。
- 完整源证据和 SHA-256 清单：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_candidate_binding_evidence_audit_20261022_attempt1/`。Safety IR 和 Compiler 未修改，候选 JSON 仅供审核；等待复核后决定后续语义确认。

## 当前状态：object_2 Caddy Task-Conditioned Candidate Verification 完成，等待复核（2026-10-22）

- 用同状态归档 512×512 RGB 和最新 `brown caddy`/threshold 0.05 SAM3 候选，仅对整只 Caddy 执行一次 Task-Conditioned Candidate Verification。原始 L0T1 指令不变，`--target-description caddy`；请求包含原图及全部 1 个自动裁剪候选，Prompt 允许 unresolved，不按单候选强制选择。
- 冻结 Qwen3.8 一次选择 Candidate 0。模型回答和选择落盘后再读取 IoU：Candidate 0 与 Body 26 `desk_caddy_1_main` / Geom 265 的 IoU 为 0.958812；SAM score 仅 0.098633，保留低分限制。
- 输入未含 Teacher 颜色描述、SAM score、Body/Geom/IoU 或预期答案。原 Safety IR 未修改，没有建立正式语义绑定。完整报告及请求、回答、候选裁剪、输入源副本与哈希清单位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object2_caddy_task_conditioned_candidate_verification_20261022_attempt1/`。服务已关闭；等待复核后再继续语义核验。

## 当前状态：object_2 caddy SAM3/Body IoU 低阈值诊断完成，等待复核（2026-10-22）

- 使用 Teacher 同状态的归档 512×512 RGB、Geom ID 和完整 Inventory；SAM 检索词 `brown caddy`、置信阈值 0.05，原有 Body-level IoU 阈值 0.1。没有 Body 预筛。
- SAM3 返回 1 个低分候选（0.098633），与 Body 26 `desk_caddy_1_main` / Geom 265 的 IoU 为 0.958812；对最高非目标 Body 27 的 IoU 仅 0.00008255。候选 Mask/BBox、全部 Body IoU 和可视化已归档。保留低 SAM 分数 caveat，不创建语义绑定。
- 结果在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object2_brown_caddy_sam3_conf005_iou010_20261022_attempt1/`。本轮未调用 VLM/Teacher、未运行环境动作或训练，等待复核。

## 当前状态：修正 target-description 优先级并完成一次 bottle 验证，等待复核（2026-10-22）

- Smoke Prompt 现规定：有 `--target-description` 时识别该类别，原任务指令只作上下文；无该参数时仍找任务操作目标。支持容器、障碍物等非操作目标，并保留 unresolved 和单候选不强选。
- 离线输入检查通过：book 指令无目标描述时仍按操作目标处理；`bottle` 描述时明确识别瓶子，不因指令提 book 而拒绝。实际请求未包含 Teacher 颜色描述、Body/Geom/IoU、SAM 分数或预期候选。
- 冻结 Qwen3.8 一次选择 Candidate 0；其归档 Mask 后验匹配 Body 25 / Geom 133，IoU 0.98497358。该值只作几何对照，无 Safety IR 修改或正式语义绑定。完整证据在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object3_task_conditioned_candidate_verification_promptfix_20261022_attempt1/`；服务已关闭，等待复核。

## 当前状态：object_3 bottle 候选验证已完成，保持 unresolved，等待复核（2026-10-22）

- 使用同状态归档 RGB、`green bottle`/0.5 SAM3 全部候选和原始 L0T1 指令，对 object_3 仅调用冻结 Qwen3.8 一次；目标描述为中性 `bottle`。RGB/Mask/Geom ID 的 512×512 坐标与采集状态核验通过。
- VLM 返回 unresolved，未选择唯一候选，理由是指令需要书而候选看起来是瓶子。回答保存后，归档 Candidate 0 Mask 与 Body 25 `bottle_of_beer__7_1_main` / Geom 133 的 IoU 为 0.98497358。保留为几何对应诊断，不创建语义绑定或修改 Safety IR。
- 审计限制：调用前的宽泛预检打印了包含 SAM 分数和 Body IoU 的归档摘要；这些数据没有进入提示或候选筛选，但操作者当时已看到摘要。Smoke 在 raw/selection 落盘后才读取详细 IoU 文件；遵守一次推理限制，不重跑，结果按非盲单次选择复核。
- 完整请求、原始输出、裁剪、输入和状态哈希、post-hoc 对照在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object3_task_conditioned_candidate_verification_20261022_attempt1/`。Qwen 服务已关闭；本轮不继续下游执行，等待人工复核。

## 当前状态：Task-Conditioned Candidate Verification 通用化与离线回归已完成，等待复核（2026-10-22）

- Smoke 现在按输入 RGB 尺寸、SAM 实际候选列表和任意候选索引工作；移除了固定 1024、固定五候选/索引 0–4、默认 L0T1 指令及 Body 23 专项代码。任务指令改为必填。
- 可选 `--target-description` 接受中性对象类别（例如 `book`、`bottle`、`caddy`）；拒绝常见颜色描述和仿真身份元数据。提示仅发送指令、可选目标描述、原图、全部自动裁剪图与候选索引。
- IoU 只在响应落盘后读取；输出通用的最佳 Body/Geom overlap，并只在 IoU 接受状态下填 `verified_body_geom_match`。不写 Safety IR 或语义绑定，unresolved 状态保留。
- 用已归档 book 数据与既有 VLM 回答进行离线回归：Candidate 1、Body 23 / Geom 91、IoU 0.916855；新推理请求数为 0。另完成 13×17 RGB、稀疏索引 `[2,9]`、中性 `bottle` 描述和颜色拒绝的无网络检查。结果在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_task_conditioned_candidate_verification_offline_regression_20261022_attempt1/`。
- `py_compile`、离线回归和 `git diff --check` 通过。没有运行 VLM、SAM3、MuJoCo、Student、规划或训练；当前等待复核。

## 前次真实执行：L0T1/state0 全候选 Task-Conditioned Smoke（2026-10-22）

- 新增并运行 `research/gs_opd/grounding/task_conditioned_candidate_verification_smoke.py`：从原生 1024 RGB 与全部 `book`/0.1 SAM3 Mask/Box 自动生成 Candidate 0–4 裁剪，将原图、五张裁剪和原任务指令一次性输入冻结 Qwen3.8。无 Body/Geom/IoU 预筛、无 Teacher 颜色描述或 SAM 分数输入。
- 唯一一次 VLM 请求 HTTP 200，选择 Candidate 1。Raw 输出已保存后才读取 Body IoU：Candidate 1→Body 23 IoU 0.916855，为最佳 Body 匹配；Body 27 IoU 0。没有改 Safety IR 或写入语义绑定，`object_1` 语义身份仍等待复核。
- 顺序限制：Smoke 运行时在保存原始回答后才解析候选 IoU，但前置准备曾解析 IoU JSON 确认结构（终端只展示开头的通用物理对象记录，未展示或使用候选/Body 23/27 IoU）。因此严格的“任何操作都在 VLM 后才读取 IoU 文件”顺序没有完全满足；未重复调用 VLM。
- 请求原文及图像 data URL、全部裁剪图、裁剪元数据、模型服务日志、原始回答、IoU 对照和 SHA 清单位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_task_conditioned_candidate_verification_1024_20261022_attempt1/`；完整结果见其 `experiment_report.md`。Qwen 服务已关闭，本轮不启动 Student/规划/训练。

## 前置状态：L0T1/state0 原生分辨率 SAM3 对照已完成（2026-10-22）

- 同一物理状态的 512×512/1024×1024 RGB 与 Geom ID 均以 180° forward 方向配对；同一 SAM3 checkpoint、`book` 检索词、阈值 0.1。512 返回 4 个候选，1024 返回 5 个；两组最高分分别为 0.238281 与 0.390625，均未达到 0.5。
- 原有 Body-level IoU 脚本无需修改，按 `min_iou=0.1` 运行通过。Body 23 IoU：512 Candidate 0 为 0.920192；1024 Candidate 0/1 分别为 0.327668/0.916855。Body 27 在两种分辨率全部候选上 IoU 均为 0。此几何结果不建立 `white book` 语义绑定，身份保持 unresolved。
- 完整结果、逐候选表、全部 Mask、Body 23/27 IoU 可视化和哈希清单：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object1_sam3_native_512_1024_20261022_attempt1/experiment_report.md`。没有重新采图、调用 Teacher/VLM、启动 Bridge 或执行动作；当前等待复核。

## 前序采集：L0T1/state0 原生 512/1024 RGB+Geom ID 对照（2026-10-22）

- 同一次 obstacle_avoidance/L0T1/state0、seed 0 Bridge 初始化中，完成 512×512 和 1024×1024 `agentview` MuJoCo 原生 RGB 与 Geom ID 采集；1024 使用 MuJoCo 渲染器直接输出，没有插值或放大。
- 两个分辨率各自的 RGB/Geom ID 像素坐标及方向检查通过。渲染前后完整 integration state、机器人/物体诊断和缓存 agentview 图像哈希完全相同；期间没有 reset 或 step。无参数 `render_highres` 仍返回默认 512×512，并与 512 segmentation RGB 逐像素一致。
- 已保存 raw 与 180° 同步旋转 RGB/Geom ID、28 Body/288 Geom Inventory、逐 Geom overlay、book/bottle/caddy 原生像素裁剪对照图、请求与状态记录、运行脚本、源码快照和哈希清单。主报告：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_native_512_1024_resolution_comparison_20261022_attempt1/experiment_report.md`。
- 512/1024 分辨率验证完成；未运行 VLM、SAM3、Student、动作、规划或训练。当前等待复核，不启动后续下游实验。

## 已有参考：Task-Conditioned 复核选 Region A，但语义身份仍 unresolved（2026-10-21）

- 使用最新 Safety IR v2、归档 `book`/0.1 四候选与 Body IoU、同帧 180° RGB、任务指令和完整 288-Geom Inventory，对 `object_1` 调用冻结 Qwen3.8 一次。复用原 VLM Prompt；返回 `unresolved`、`candidate_index=null`，未创建语义绑定。
- 本轮补做静态人工对照图：Candidate 0 与 Body 23 的真实可见 Mask IoU 为 0.920192、与 Body 27 为 0；编号“0”位于 Mask 左上边缘，文字影响 Mask 内 27 px（0.876%）。遮挡可能影响少量局部细节，但 VLM 判断因果未证实。完整 PNG 与报告位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object1_candidate_body_mask_review_20261021_attempt1/`；保持 unresolved 并等待复核。
- 新的独立 Task-Conditioned Reference Verification 只调用冻结 Qwen3.8 一次：用原始任务指令、原 RGB 和无语义标签的 A/B 放大图，模型选 A，称其像有可见页片的黑色书状物。旧 VLM 则把 Candidate 0 看作餐具类并返回 unresolved；Teacher 仍描述 white book。单次判断有差异，不能确证语义或物理身份；A/B 裁剪共享部分画面且 A 先呈现，顺序影响未排除。请求和离线对照在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_task_conditioned_reference_verification_20261021_attempt1/`；Safety IR 不变，绑定继续 unresolved。
- Smoke 按其原有筛选逻辑剔除了 IoU unresolved 的 Candidate 1–3，VLM 仅看到 Candidate 0。该候选与 Body 23 `black_book_1_main` / Geom 91 的 IoU 为 0.9202；VLM 将它描述为带叉子的黑色物体/餐具盒，并认为浅色矩形物体是盒子。此结果与 `white book` 的 Teacher 描述不能建立充分一致的身份对应；保持 unresolved，不依据高 IoU、唯一候选或任务文字强行绑定。
- 完整 Inventory 中 9 个 `geom_name=null` 项只在单独兼容副本中替换为按 ID 可追溯的占位名，原始 288 项清单不变。原始 VLM 回答、编号图、最终判断、启动与请求记录、输入副本和 SHA-256 清单位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_object1_vlm_semantic_disambiguation_20261021_attempt1/`。唯一有效 VLM 请求 HTTP 200、耗时 4.545 秒；临时服务已关闭。
- 本轮未重跑 SAM3/Teacher/MuJoCo，未处理其他物体、写入绑定、规划、运行 Student 或训练；当前等待复核，不进入轨迹规划。

## 已完成：L0T1 SAM3 三组检索诊断（2026-10-21）

- Safety Bridge 已支持 benchmark、level、task、init-state、seed 和相机分辨率参数；默认行为仍是 obstacle_avoidance/L0T0/state0、seed 0、256×256。`reset` 使用同一套场景参数。
- 本地核验后选用不同 BDDL 的 obstacle_avoidance/L0T1/state0：`pick up the book and place it in the back compartment of the caddy`，50 个 init states。只执行初始化/观察采集，没有 Student、动作 step、规划、Repair 或训练。
- 新增只读 `observe` 后，正式 Grounding 从 Bridge 启动的初始化观测开始，随后读取 `geom_segmentation` 和 `geometry_inventory`，采集客户端不 reset/step。两相机 512×512 RGB、Geom segmentation 512×512 int32；同状态前后 Snapshot diagnostics 与双相机缓存图像一致，agentview RGB 与 segmentation RGB 原始方向逐像素匹配。
- 完整 Geometry Inventory 含 28 Bodies、288 Geoms；book/caddy/bottle 的 BDDL 实体名命中三个不同 Body，且各有可见 Geom 像素。Teacher “white book” 与 BDDL `black_book_1` 的颜色/身份关系仍未确认；brown caddy、green bottle 也尚无 Teacher 语义绑定。
- 前次 Teacher 实际输入图像另存为 `teacher_input_reference/` 并仅作对照。本次与之像素不同，虽 task/init-state 配置相同，不能确认实际物理状态一致，因此不跨 reset 混配。完整 RGB、Geom ID、overlay、Body/Geom 清单和对齐报告在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_init0_geom_grounding_20261021/`。
- 两相机 512×512 RGB PNG、原始 RGB NPZ、场景元信息、BDDL 对比、冻结 Qwen3.8 Teacher 原始与校验 JSON、Schema 结果和 SHA-256 清单保存在 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_init0_scene_observation_20261021/`。Safety IR v2 输出 6 个语义物体、6 条机器人接触规则、1 条被搬运 book→bottle 禁止接触规则；后侧隔间可见性和 bottle 相对位置仍有不确定性。无任何旧场景 Teacher JSON 或物体绑定参与。
- 已从同一状态的 Geom 对齐 RGB 离线调用一次冻结 Qwen3.8 Teacher，没有连接/reset Bridge。复用原 Prompt/Schema、temperature 0、non-thinking、4096-token 配置；新 IR v2 通过 Schema，含 6 个对象、3 条机器人接触规则、1 条 book→bottle 规则和 3 条全局不确定性。与前次输出相比，cup/utensil/plant 名称及部分 robot-contact 规则不同，证据分别保存，没有覆盖旧结果。
- 新旧 Teacher 输出均称目标为 “white book”；当前 RGB 有浅色矩形区域，BDDL 同时列出 `black_book_1` 和 `white_place_box_1`。物理身份仍未确认，未修改 Teacher 输出或创建 Body/Geom 绑定。
- `white book` 和 `book` 在 0.5 阈值下此前均返回零候选；本轮仅将 `book` 阈值降到 0.1 后获得 4 个低置信候选，其中 Candidate 0 与 Body 23 `black_book_1_main` 的 IoU 为 0.9202、对 Body 27 `white_place_box_1_main` 为 0。低分几何重合不作为 Teacher 身份绑定，`object_1` 仍 unresolved。
- 另外两组独立运行中，`green bottle`/0.5 得到一个分数 0.9688 的候选，与 Body 25 `bottle_of_beer__7_1_main` IoU 0.9850；`brown caddy`/0.5 无候选，raw combined confidence 最高 0.0986、低于阈值，Body 26 没有候选 IoU 可比。Bottle 的几何匹配同样未自动写成语义绑定。
- 每次加载本地 SAM3 checkpoint 成功；输入预处理、200-proposal raw output、阈值过滤后的 shape/dtype、所有 Mask/分数/Body IoU 均已分别存档。三组报告和源证据副本位于 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T1_sam3_prompt_threshold_diagnostics_20261021_attempt1/`。未运行 VLM、Bridge、Student、动作、规划或训练；等待复核。

## 已完成：π₂ Batch Size 3 纯 Student 闭环评测（2026-10-21）

- 在原三 Episode π₂ 累计数据集上，从原始 SFT Base 新建 rank-32 LoRA，完成一次 10,000 步训练。相较既有 π₂ 配置只把 `actor.micro_batch_size` 和 `actor.global_batch_size` 从 2 改为 3，并使用独立实验标识；学习率、数据采样、模型与 Flow Matching 设置保持一致。
- 训练前确认官方训练 DataLoader 的 6,897 个窗口覆盖三个 Episode，batch size 为 3，三个阶段各曝光 2,299 次；实际 Batch 3 前向/反向通过，峰值 allocated/reserved 显存约 17.64/22.35 GiB。正式训练完成 10,000/10,000 步，保存 5k/10k Checkpoint、阶段固定评估和可独立重载的 Adapter。
- 独立从两个训练的原始 `step_metrics.jsonl` 重算后，Batch 2 首/末 100 步 Loss 均值为 `0.0256858893` / `0.0012088185`；Batch 3 为 `0.0253006398` / `0.0011320326`。Batch 3 进度报告中误抄的 Batch 2 数值已更正，训练日志与权重未改动。
- 使用 Batch 3 Adapter 做了一次纯 Student 闭环 Episode：848 步、reward `1.0`、`done=True`。完整子步监测下显式 Teacher 接触规则编译结果为 `safe`，但整体物理安全仍为 `unknown`。前 60 步 EEF 路线与 π₁ 对齐点平均距离 3.83 mm；Batch 2 失败轨迹对应值为 8.97 mm。
- 训练产物位于 `/opt/data/private/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_bs3_20261021_attempt1/`；闭环与对比证据位于 `/opt/data/private/gs_opd/eval/pi05_pi2_cumulative_fresh_bs3_obstacle_avoidance_l0t0_state0_seed20261020_20261021_attempt1/`。详情见 `docs/GS_OPD_PROGRESS.md`；等待复核，不继续 Rollout 或训练。

## π₂ 原始累计经验训练与存储整理基线（2026-10-21）

- 使用累计 LeRobot v2.1 三 Episode 数据集和已验证的三阶段均衡采样，完成 **10,000 optimizer steps**。训练始终从原始 SFT Checkpoint `/workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft` 加载冻结基座，并新建 rank-32 Action Expert LoRA；`initial_adapter_path: null`、`resume_dir: null`，没有加载或合并 attempt3 π₁ Adapter。
- 基座非 LoRA 权重 667/667 键完整覆盖，抽样权重最大差为 0；667 个基座参数张量冻结，13,860,864 个 LoRA 参数独立训练。AdamW 与学习率调度器从头初始化，固定 LR `1e-4`，未改变 OpenPI Flow Matching、归一化或模型结构。
- Batch Size 2 训练 loss 首 100 步均值 `0.025686`、末 100 步均值 `0.001209`。三阶段固定评估覆盖 step 0、2,000、4,000、6,000、10,000；π₂ Adapter 与完整训练权重在相同 Observation 上推理差为 0。
- 完整训练报告、逐步 Loss、配置、预检、step 0 与阶段评估、5,000/10,000 步 Checkpoint、独立 π₂ Adapter 及导出验证均保存在 `logs/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_20261021_attempt1/`。
- 本轮没有 MuJoCo Rollout、物理安全判定或新任务成功结论；等待复核训练与评估结果后再决定下一步。
- π₂ 训练目录已完整校验并迁至 `/opt/data/private/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_20261021_attempt1/`；原 workspace 路径保留为软链接。后续 SFT、评测、Rollout 和新数据集默认输出已分别指向数据盘 `sft/eval/rollouts/datasets`。旧 SFT 基座、π₁ Adapter、成功 Left 和累计数据集仍从原路径读取正常。
- 存储迁移与路径读取验证已完成；不启动训练或 Rollout，等待复核后再做 π₂ 与 π₁ 闭环评测。

累计数据来源与训练评估详见 `docs/GS_OPD_PROGRESS.md`。总体路线见 `docs/GS_OPD_ROADMAP.md`。

## 已完成：attempt3 LoRA 训练与纯 Student 闭环评测（2026-10-20）

- 从原始 Student 初始化 rank-32 Action Expert LoRA，完成 10,000 optimizer steps。冻结范围、完整 845 帧 Episode、三阶段均衡采样、官方 OpenPI DataLoader、RLinf `SFTRunner`、原生 Flow Matching Loss 和归一化保持不变；搬运阶段固定示范 Observation 的 Flow MSE 从 0.152026 降至 0.000982，10×7 Action MAE 从 0.183005 降至 0.009061。
- attempt3 Adapter 与原始 Student Checkpoint 组合后，在 LIBERO-Safety obstacle_avoidance L0T0 初始状态 0 完成一次纯 Student 回合：711 个环境动作步，reward 1.0，`done=True`；Adapter runtime 日志确认 rank 32、216 张量、13,860,864 参数已加载。未使用 B-Spline/Mink 接管或在线修正。
- 原生 20 Hz OSC_POSE、action horizon/chunk 10、7D 执行动作和原 norm stats 保持；保存了完整 MP4、NPZ（七维动作、双相机 Observation、EEF 与黑碗物理状态、接触记录）、Safety Compiler 审计和路线对比图。针对 `object_1→object_4` forbid 规则，Compiler 对 17,775 个子步完成审计并返回 `safe`；保守包围球余量仅约 0.19 mm，且该结论依赖现有 `object_4` 语义绑定（IoU 0.1427），不得外推到其他物体或规则。
- 本次产物位于 `logs/gs_opd/eval/pi05_libero_full_trajectory_lora_constant_lr_attempt3_obstacle_avoidance_l0t0_20261020/`；详细结果见 `docs/GS_OPD_PROGRESS.md` 和 `evaluation_report.json`。之前 LoRA Student 路径已按历史清理记录删除，因此图中明确标缺失。

attempt3 训练与单回合评测基线已完成；其后数据准备状态见本页开头。初始 Student、成功示范、LeRobot 数据、attempt3 Adapter 及评测原始证据保留。
