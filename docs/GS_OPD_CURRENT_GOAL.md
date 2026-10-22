# GS-OPD 当前局部目标

## L0T2 Pot 1 放置延续：本轮受 horizon 预算限制停止，等待复核（2026-10-22）

本轮直接从 Pot 1 Grasp Retargeting 成功快照恢复，起始 timestep 854 / horizon 1000。三条 Stove 几何候选中，两条只完成 clearance 抬升；`west_north_diagonal` 执行 112 步到 timestep 966。Pot 1 仍稳定抓持，离目标支撑高度约 133 mm；按最后 40 步实测下降速率，完成下降、释放、撤离与稳定估计需要 119 步，而只剩 34 步。因此按标准 horizon 约束停止，没有通过扩展 horizon 或放宽控制条件来冒充完成。

终态真实 reward=0、`done=false`；Pot 1 `On=false`，Pot 2 `On=true`，Stove `Turnon=true`。下降时 Pot 1–Pot 2 有真实接触（峰值 71.20 N），Pot 2 位移 0.303 mm 且原生谓词保持成立；未记录新机器人–非目标碰撞，112/112 关节限制检查通过。任务未成功，安全成功也为 false；原始轨迹已有的 Ginger Beer 接触记录继续保留。

独立证据：`/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_mokapot1_stove_placement_20261022_attempt1/README.md`。终态可恢复快照为 `evidence/pot1_partial_transport_t966_runtime_snapshot.pkl`；完整动作链、逐步 Observation/Action/接触、视频及候选报告见该归档。当前暂停物理执行并等待复核。复核后的下一阶段方向是从更早的 Pot 2 抓持状态压缩 Pot 1 搬运链，使完整放置、释放、撤离和稳定能在正常 horizon 内结束。

## 已完成前置：首版 Grasp Retargeting 抓持迁移验证

从 Pot 2 放置成功快照（timestep 719）恢复后，通用物体坐标抓取模板成功迁移至 Pot 1。闭爪前 EEF 误差 7.41 mm / 0.00054 rad，目标位移为 0；记录到 8 个连续稳定双指闭合样本，Pot 1 根部上升 30.657 mm。135 步 OSC 物理记录完整，Panda 限位 135/135 通过，命名抓取后 Runtime Snapshot 可精确恢复。抓取后任务 reward 仍为 0、`done=false`，未代表完整 T2 成功。

抓取组件与前置结果的详情见 `GS_OPD_PROGRESS.md` 和归档 `/opt/data/private/gs_opd/rollouts/grasp_retargeting_l0t2_20261022_attempt1/`。源 VLA 前缀含 Ginger Beer 接触，因此组合数据不能标注为安全成功。
