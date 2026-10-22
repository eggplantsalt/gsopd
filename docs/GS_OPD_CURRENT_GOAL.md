# GS-OPD 当前局部目标

## L0T2 完整任务示范：快照恢复被模型指纹拒绝，等待复核（2026-10-22）

当前原始任务仍是 LIBERO-Safety obstacle_avoidance/L0T2/state0 的完整任务成功示范。Pot 2 已放置，Pot 1 在 t=966 快照中保持稳定抓持；允许后续隔离的 demonstration-only 执行累计至多 1300 步，标准评测仍为 1000 步。

本轮已尝试创建隔离 Bridge：按 horizon=1000 编译后，仅将 Episode horizon 改为 1300，且 ignore_done=false。但新 Bridge 的模型指纹 75c8db457aa094e5efc43050a4f07192a9e8dca1379e84a67f4854800466d574 与 t=966 快照的 1a0c219cafe5ff4fa96429ca76e37e3885b2b0b390a8e76214a3e14f7ea9f752 不同，现有 restore_runtime_snapshot 正确拒绝恢复。原 Bridge 仍与快照指纹相同，但 horizon=1000 且只剩34步。

已执行0个物理控制步；没有轨迹候选、碰撞/关节结果或新视频。没有修改标准评测、BDDL、reward、物理位姿、关节限制或快照一致性校验。完整记录见 GS_OPD_PROGRESS.md 与独立归档 /opt/data/private/gs_opd/rollouts/obstacle_avoidance_L0T2_complete_demo_only_20261022_attempt1/。

当前停止等待复核。继续前需恢复快照生成时的模型/运行时版本，或找到并验证可作用于原模型兼容 Bridge 的 horizon 延长接口；只有完整模型指纹一致、积分状态精确恢复、t=966 谓词和抓持检查通过后，才可继续尝试 Pot 1 的 OSC 放置。不要重新运行 VLA、重新抓取两个 Pot、改变标准 horizon、改任务判据或训练。
