# GS-OPD 当前局部目标

## 首版 Grasp Retargeting 已完成真实抓持迁移验证（2026-10-22；等待复核）

目标为 LIBERO-Safety `obstacle_avoidance/L0T2/state0`。已从归档的原始 π₀.₅ Base Moka Pot 2 稳定抓取记录提取物体坐标系抓取位姿、接近方向/距离与夹爪时序，并新增可复用组件 `research/gs_opd/repair/grasp_retargeting.py`。运行时 Body/Geom 绑定由 Bridge 状态和 Geometry Inventory 提供，不在组件中固定 T2 身份或世界位置。

从 Pot 2 放置成功 Runtime Snapshot（timestep 719，horizon 1000）恢复后，单次 `source_pose` OSC_POSE 迁移抓取已通过：闭爪前 EEF 误差 7.41 mm / 0.00054 rad，目标位移 0；连续双指稳定监测为 8 个不同状态（门槛 3），Pot 1 根 Z 实际升高 30.657 mm。真实子步接触完整、Panda 限位 135/135 通过、无新机器人–非目标碰撞。保存的成功 Snapshot 再经 Bridge Restore，MuJoCo integration state 差为 0，实时左右指接触与闭合状态再次通过。

本轮只完成抓持迁移和抬升验证。环境 reward 0、`done=false`，Pot 1 尚未满足 Stove `On`；不代表完整 T2 成功。源 VLA 前缀存在 Ginger Beer 接触，合并历史不得标为安全成功示范。其余 3 个有限抓取候选未运行；未搬运、放置、训练或调用 Teacher/SAM3，也未修改 B-Spline 生成器、默认控制配置或模型。

完整逐步数据、接触、双相机视频、源抓取模板及可恢复抓取后快照独立归档于 `/opt/data/private/gs_opd/rollouts/grasp_retargeting_l0t2_20261022_attempt1/`。当前任务已完成，等待复核；复核前不启动后续 OSC 搬运。
