# SafeTRAIL Stage A 失败分类（2026-10-30）

GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv 的 stage_a_failure_category 按每项最新Standard尝试及其证据分类；Extended成功单独标注Standard horizon差距，不将其算作未解决任务。原因原文保留在 last_failure_type 和 last_failure_reason。

| 类别 | 含义 |
|---|---|
| A | 抓取候选集合全部不可行或筛查耗尽 |
| B | IK或RRT规划失败 |
| C | 抓取闭合、稳定性或抬升失败 |
| D | 抓持后运输丢失物体 |
| E | 几何路线可行但真实OSC跟踪失败 |
| F | Region Progress或容器入口进展停滞 |
| G | 到达目标附近但原生On/In未成立 |
| H | 释放、撤离、稳定或多目标续接失败 |
| I | Bridge、进程或记录基础设施错误 |
| J | 动作、墙钟或Standard horizon预算耗尽 |

## 当前分布

- A：11项——OA L0T3/L0T4/L1T3/L2T1；OAH L0T1/L0T3/L1T1/L1T3/L1T4/L2T1/L2T3。
- B：1项——OAH L2T4插入目标构型RRT不可行。
- D：1项——OAH L2T2运输中失去双侧抓持。
- F：4项——OA L1T4/L2T2/L2T4；OAH L0T4，均记录放置路点停滞或Region进展失败。
- H：1项——OAH L0T2首个目标后，第二目标Pot2候选集耗尽。
- I：1项——OA L2T3首个受监测动作遇Safety Bridge EOF。
- J：3项——OAH L1T2墙钟耗尽；OA L0T2和L1T2只差Standard horizon，均另有已审计Extended原生成功。
- none：7项Standard完整成功；in_progress：OAH L0T0 dev02。

当前未观察到有充分证据归入C、E或G的最新失败。不要为了填满分类而推断；后续实验若提供直接物理证据，再更新类别。A是最大的未解决根因组；当前OAH L0T0仍在Region Progress运行且物体距离持续下降，待终态后再决定是否先修复F组的通用运输算法。
