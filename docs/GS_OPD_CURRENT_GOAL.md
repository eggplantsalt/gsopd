# GS-OPD 当前局部目标

## L0T2 demonstration-only 快照恢复与 horizon 扩展已通过，等待复核（2026-10-22）

当前实验仍是 LIBERO-Safety `obstacle_avoidance/L0T2/state0`。t=966 的 Pot 1 抓持/部分搬运 Runtime Snapshot 已在与其完整模型指纹一致的独立 Bridge 中精确恢复。新实例初次编译 hash 不匹配；复现已验证的 reset/set-init-state 流程后恢复到快照 hash，并通过 scene metadata、BDDL、MuJoCo Integration State、机器人控制状态、timestep、Pot 1 双指抓持和原生任务谓词检查。

同一已恢复实例中的底层 horizon 已从1000扩展到1300，`ignore_done=false`；扩展前后模型和物理运行时状态完全不变。隔离实例当前空闲于 t=966/horizon=1300，Pot 1 尚未放置，reward=0、done=false。标准 Bridge 仍是 t=966/horizon=1000，标准评测默认值未改。

本轮执行了0个控制动作、0个VLA动作、0次重新抓取或搬运。完整验收与可复现启动/恢复步骤见 `/opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_demo_restore_horizon_20261022_attempt2/README.md` 和 `evidence/acceptance.json`；扩展后快照为 `evidence/extended_t966_runtime_snapshot.pkl`。

**当前阶段停止等待复核。** 复核后再从保存状态继续剩余 Pot 1 的 OSC 放置；任务成功和安全成功仍须分别判定。
