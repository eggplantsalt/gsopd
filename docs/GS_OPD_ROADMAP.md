# GS-OPD 总体路线

## 总体目标

在 π₀.₅ 原生动作空间和现有安全约束下，让 Student 从真实成功执行数据学习避障操作，并在 MuJoCo 中完成可复现的 Student 闭环任务。数据来源、动作来源和安全判定要可追溯；Compiler 的 `unknown` 不得伪装成 `safe`，普通行为克隆动作不得误标为安全修正。

## 阶段与当前状态

1. **安全输入与物体语义**：L0T1/state0 的 512×512 双相机 RGB、Geom segmentation 和完整 Body/Geom Inventory 已从同一次 Bridge 初始化采集并确认像素/状态对齐；book/caddy/bottle 的候选 BDDL 实体各自在独立可见 Body。基于这组 RGB 的一次新 Teacher v2 输出已通过 Schema；与前次输出相比，后 3 个物体命名和 robot-contact 规则数量有差异。两次图像来自不同 capture/reset，差异原因未判定。SAM3 对原始 Teacher `white book` 在阈值 0.5 下无候选；`book` 降至 0.1 时出现 4 个低置信候选，其中一个对 Body 23 `black_book_1_main` IoU 0.9202、对 Body 27 `white_place_box_1_main` IoU 0。随后对 `object_1` 用现有冻结 Qwen3.8 VLM 消歧一次，VLM 返回 unresolved；因原实现将另 3 个 IoU-unresolved 候选过滤，VLM 只见 Candidate 0，并把它描述为黑色带叉子的物体。该描述与 IoU 的 Body 23 `black_book_1_main` 和 Teacher 的 `white book` 均未形成足够一致证据，故不建立绑定。`green bottle`/0.5 候选与 Body 25 几何 IoU 0.9850；`brown caddy`/0.5 无候选，raw combined score max 0.0986。2026-10-22 对 object_3 进行一次真实 Task-Conditioned Verification：原指令 + 中性 `bottle` 下，Qwen 返回 unresolved（唯一候选看起来是瓶子而任务要求书）；回答保存后确认 Candidate 0 与 Body 25 / Geom 133 的几何 IoU 0.98497358，但没有候选被模型选择，不建立 object_3 语义绑定，Safety IR 未修改。SAM/Geom 与 VLM 结论均保留各自证据；`object_1` 与 `object_3` 安全使用在确立语义绑定前继续保持 `unknown`。Candidate 0/Body 23/Body 27 的同坐标静态人工对照图已归档；Candidate 0 与 Body 23 的 IoU 为 0.9202，但 Teacher `white book` 的语义关联仍 unresolved。一次独立 Task-Conditioned VLM 复核在原指令与 A/B 无语义标签局部视图下选择 A；旧 VLM 仍 unresolved，Teacher `white book` 的外观冲突未消解。当前等待人工复核，不据单次模型回答或几何重叠推进语义绑定。
   - 2026-10-22 在同一 L0T1/state0 Bridge 初始化中完成 `agentview` 512×512 与 1024×1024 原生 RGB+Geom ID 对照。两种尺寸的 RGB/标签同尺寸对齐与方向检查通过，完整 MuJoCo integration state 和缓存观察哈希保持一致；1024 没有通过缩放生成。book/bottle/caddy 的归一化位置一致，详细像素与文件哈希见 `docs/GS_OPD_PROGRESS.md` 和独立数据盘采集目录。当前等待复核。
   - 随后使用这组同状态 RGB/Geom ID 对 `object_1` 做 SAM3 配对：`book`/0.1，同一 checkpoint。512 为 4 候选、1024 为 5 候选，均无分数达到 0.5；Body 23 的最高 IoU 分别为 0.920192/0.916855，Body 27 全部为 0。Teacher `white book` 的语义绑定仍 unresolved；结果、Mask 和报告已归档，等待复核。
   - 2026-10-22 新增全候选 task-conditioned verification Smoke：全部 5 个 1024 Candidate 均通过自动 Mask/Box+背景裁剪进入单次冻结 Qwen3.8 请求，VLM 选 Candidate 1。仅在回答保存后读取 Body IoU；其 Body 23 IoU 为 0.916855、为最佳匹配，Body 27 IoU 为 0。单次候选选择不修改 Teacher IR 或创建语义绑定；完整输入、输出和裁剪归档于独立数据盘目录，当前等待复核。
   - 随后将 Smoke 通用化：分辨率、候选数量和索引、任务指令及目标描述均来自调用参数/输入；固定 Body 23 检查改为响应后的通用最佳 Body/Geom 重合，IoU 未通过时继续 unresolved。用前次 book 响应离线回归保持 Candidate 1 / Body 23 / Geom 91、IoU 0.916855，未发新 VLM 请求。目标描述禁止常见颜色属性，防止误带入 Teacher 的不可靠颜色；等待复核后再决定是否做其他物体的 VLM 调用。
   - 2026-10-22 修复 Task-Conditioned Prompt 的目标优先级：提供目标类别时按类别识别各类场景对象，任务指令仅作上下文；不提供时按操作目标定位，仍允许 unresolved。离线 book/bottle Prompt 检查通过；一次真实 bottle 请求选择 Candidate 0，其 Mask 与 Body 25 / Geom 133 的 IoU 为 0.98497358。Safety IR 与正式绑定未变；该证据不等同于绑定，当前等待复核。
   - 2026-10-22 对 `object_2` 使用 `brown caddy`、SAM 阈值 0.05 运行一组低阈值诊断，返回 1 个 score 0.098633 的候选；完整 Body 匹配中 Body 26 IoU 0.958812，最高非目标 IoU 0.00008255。该证据是几何候选核验，不创建正式绑定；等待复核。
2. **真实成功执行**：已有一次真实 Student→B-Spline→原生 OSC 左绕成功回合：749 步 B-Spline 左绕后由 π₀.₅ 接续 41 步，取得真实任务奖励 1.0；完整轨迹和视频保存在 `live_left_bspline_osc_adaptive_20261019_attempt1`。
3. **数据格式与读取**：已有 749 步 OSC 专用 LeRobot 数据集；现已另从原始成功 NPZ 生成完整 845 步单 Episode（55 抓取 / 749 搬运 / 41 放置），逐步验证 Observation–Action 对齐，保留 836 个 horizon-10 无 Padding 窗口及跨阶段窗口，继续使用初始 Student 对应 norm stats。
4. **单轨迹学习链路**：2026-10-20 从原始 Student 初始化新的 rank-32 Action Expert LoRA，使用恒定 `1e-4` LR、AdamW 重新初始化、官方 OpenPI DataLoader、`SFTRunner`、原生 Flow Matching Loss 和三阶段均衡采样，完成 10,000 optimizer steps。固定示范状态上的 OSC 搬运 Flow MSE 从 0.152026 降至 0.000982，真实 10×7 Action MAE 从 0.183005 降至 0.009061；独立 Adapter 已成功重载并与训练 checkpoint 推理一致。该结果确认完整轨迹监督拟合，不代表新闭环成功、安全泛化或真实执行能力。
5. **训练后 Student 评测与后续验证**：attempt3 新 Adapter 已与原始 Student 组合，在 LIBERO-Safety `obstacle_avoidance` L0T0 初始状态 0 完成一次纯 Student MuJoCo 回合：711 步、reward 1.0、`done=True`。完整 MP4、七维动作、双相机 Observation、EEF/黑碗物理状态和 17,775 个子步接触监测均已保存。针对现有 Safety IR 的 object_1/object_4 禁止接触规则，活动 Geometry Inventory 绑定通过核验，Compiler 在全子步记录上判为 `safe`；碗与该障碍的最近保守包围球间距约 0.19 mm，须保留这一很小余量以及 object_4 原语义匹配 IoU 0.1427 的置信限制。路径对比显示 Student 以更紧的 +World-Y 路径绕过障碍。此前 LoRA Student 物理路径文件已被空间清理，本次没有替代或重建该历史曲线。`object_3` 绑定继续保持 unknown，不从 reward 推导安全结论。

## 参数化任务语义采集进度（2026-10-21）

Safety Bridge 现可选择 benchmark、level、task、init-state 与相机分辨率，默认 L0T0 初始化保持原样。已选择不同 BDDL 的 obstacle_avoidance L0T1“pick up the book and place it in the back compartment of the caddy”，state 0；冻结 Qwen3.8 Teacher 已先后对两次独立 capture 生成 Safety IR v2。最近一次使用 Geom 对齐的同状态 RGB 离线调用一次，不连接 Bridge；输出通过 Schema，保留了输入哈希、原始 JSON 和与旧输出的对比。新旧响应对 book/caddy/bottle 语义保持一致，但对 cup/utensil/plant 的命名和 robot-contact 规则覆盖有差异；旧图与本次 RGB 来自不同 reset，不能把此差异归因于同状态重推理。Teacher `white book` 与 BDDL `black_book_1` 及 `white_place_box_1` 的关系仍未确认。本轮在相同归档输入和 Checkpoint 上独立运行 `book`/0.1、`green bottle`/0.5、`brown caddy`/0.5：book 的一个低分候选与 Body 23 IoU 0.9202；bottle 候选与 Body 25 IoU 0.9850；caddy 无候选且 raw combined score max 0.0986。候选与几何重叠仅作诊断，没有建立语义绑定。对 `object_1` 的一次 VLM 消歧已返回 unresolved；Candidate 0 与 Body 23/27 的同坐标人工对照图已归档，当前等待人工复核后再决定下一步。

## 主要记录

- 实验与历史阻塞：`docs/GS_OPD_PROGRESS.md`
- 当前局部目标：`docs/GS_OPD_CURRENT_GOAL.md`
- 原始 Student Checkpoint、成功示范和 LeRobot 数据集仍保留。attempt3 训练日志、报告、Adapter 和 RLinf checkpoint，以及本次 Student rollout 视频、NPZ、Compiler 审计和轨迹图，均保留于各自独立目录。详细记录见 `docs/GS_OPD_PROGRESS.md`。

## π₂ 累计训练状态（2026-10-21）

Round 0/1 累计 LeRobot v2.1 数据集包含原始 Student 845 步 Left，以及 attempt3 LoRA Student 起始状态上的 859/875 步 10%/20% 插值 Left；共 3 Episode、2,579 帧。已完成从原始 SFT Checkpoint 新初始化 rank-32 Action Expert LoRA 的 **10,000 optimizer steps** 训练；没有将 attempt3 π₁ Adapter 用作 π₂ 初始化。基座非 LoRA 权重 667/667 键完整覆盖且冻结，阶段均衡采样为每个阶段每 Epoch 2,299 次曝光。

π₂ 的 step 0、2,000、4,000、6,000、10,000 固定评估、5,000/10,000 步完整 Checkpoint 和独立 Adapter 已保留于 `logs/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_20261021_attempt1/`。Adapter 导出后重载验证通过，与最终完整权重在同一 Observation 的动作最大差为 0。该训练确认固定数据上的拟合变化；没有进行 MuJoCo Rollout，不构成物理安全或任务成功结论。

在同一三 Episode 累计数据集上，另建独立 Batch Size 3 训练：从原始 SFT Base 重新初始化 rank-32 LoRA，唯一训练超参数变更是 micro/global batch size 从 2 增至 3；数据采样、模型、Flow Matching、`1e-4` 学习率和 10,000 步预算保持不变。单 Batch 前向/反向预检通过后完成 10,000 步；5k/10k Checkpoint、step 0/2k/4k/6k/10k 阶段评估及独立 Adapter 保存在 `/opt/data/private/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_bs3_20261021_attempt1/`。

从 Batch 2/3 的原始逐步 Loss 日志分别重算首末 100 步均值后，Batch 2 为 `0.0256858893` / `0.0012088185`，Batch 3 为 `0.0253006398` / `0.0011320326`；Batch 3 进度报告的误抄值已更正，训练产物保持原样。随后 Batch 3 新 Adapter 在 LIBERO-Safety `obstacle_avoidance` L0T0、初始状态 0、seed `20261020` 的纯 Student 单回合中取得 848 步、reward `1.0`、`done=True`。完整子步监测的显式 Teacher 接触规则判为 `safe`，但整体物理安全仍为 `unknown`。首 60 步 EEF 与 π₁ 成功路线平均距离 3.83 mm（Batch 2 失败路线为 8.97 mm）。本次视频、完整 NPZ、Contact Compiler 审计及对比证据独立保存在 `/opt/data/private/gs_opd/eval/pi05_pi2_cumulative_fresh_bs3_obstacle_avoidance_l0t0_state0_seed20261020_20261021_attempt1/`。等待复核，不继续 Rollout 或训练。

π₂ 训练目录现已完整迁至 `/opt/data/private/gs_opd/sft/pi05_pi2_cumulative_fresh_rank32_lora_20261021_attempt1/`，workspace 原路径以软链接保留。新训练 Checkpoint、评测输出、Rollout 轨迹和数据集的默认路径分别使用数据盘的 `sft/eval/rollouts/datasets` 子目录；历史成果继续可从原路径访问。

下一步：复核 `docs/GS_OPD_PROGRESS.md` 中 Batch Size 2/3 Loss 重算、Batch 3 闭环结果与安全审计、路线比较、Checkpoint/Adapter 哈希及既有存储迁移记录；复核前不再启动 π₂ 或 π₁ 闭环 Rollout，也不开始下一轮训练。
