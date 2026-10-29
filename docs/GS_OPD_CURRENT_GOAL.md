# SafeTRAIL Stage A 当前目标与局部进度（2026-10-29）

`GOAL_STATUS=ACTIVE`。主目标是在本地注册的 LIBERO-Safety `obstacle_avoidance` 与 `obstacle_avoidance_human` 中，为 L0/L1/L2 × T0–T4 共30项任务分别生成至少一条从合法 t0 出发、真实 MuJoCo 连续执行并满足原生 `task_success=true` 的轨迹。当前仅推进 Task-Feasible Bootstrap（Stage A）；不开展安全重构、Teacher/SAM3、Student训练或 LIBERO-90 主线工作。

## 覆盖状态

- 30项中已有4项至少在一个步数口径下原生成功：L0T0、L0T1、L1T0为Standard 1000步成功；L0T2仅为horizon=5000的demonstration-only成功，Standard 1000步未成功。故Standard覆盖为3/30，扩展口径另有1项；L0T39关灶是矩阵外结果。
- `obstacle_avoidance/L0T3` 标准冷启动失败：24个候选中21个有实际碰撞、3个IK不可达；0个RRT或专家物理分支。31步Student首分支精确回滚并丢弃。
- `obstacle_avoidance/L0T4` 标准冷启动失败：24个候选均有实际碰撞；0个RRT或专家物理分支。32步Student首分支精确回滚并丢弃。
- 下一个覆盖项为 `obstacle_avoidance/L1T1`；L1T0成功已归档。之后继续覆盖其余未完成配置，并把矩阵作为唯一任务状态来源。

## L1T0 标准任务成功

`obstacle_avoidance/L1T0` 从合法 init-state 0、seed 0 的新鲜 t0 开始。初始 `On(akita_black_bowl_1,plate_1)=false`，终态在t=402变为true，原生 `task_success=true`、reward=1、done=true；因此计为Standard 1000步成功。t0 payload SHA-256为`959474a6b0ead1c2d9cd12694dfd0e80f02f12c6f176e42816ab41e7a5362185`，MuJoCo integration-state SHA-256为`9f1623b303f4d23b13e25e12c99b89b107689b65c280f543d18dbdeee2687d87`，模型几何SHA-256为`9563488607b6d99838d870f1ff980ca6bfd7630595f5504662f26cd70233cc86`。BDDL SHA-256为`85f4396e3152c27323182ac2f2a0191371e917a412667e5123e53490b33f2e7b`。

最终成功Episode由专家抓取145步、经已核验抓持抬升Snapshot继续的Student handoff 32步、专家原生目标放置225步组成；三段分别覆盖t=1–145、146–177、178–402，顺序连续。较早的32步Student首分支已精确恢复t0并丢弃，不计入成功轨迹；最终成功不能归为Student-only。V3独立审计为402/402动作行、10,050/10,050 MuJoCo子步，动作维度及状态—动作字段有效，Snapshot阶段边界核验通过。候选为`geom112_jaw0_approach2_+1_flip+1__approach_axis2_-11.2mm__rrt0`，8次RRT预检后执行；报告另保留`planning_time_budget_exhausted`搜索诊断，但不影响已报告的原生成功与完整动作审计。

这只是Stage A任务成功。285步几何监测状态为unsafe，安全认证与Strict-10mm均未通过；稳定释放未评估，`safe_training_eligible=false`。完整归档位于`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261029/obstacle_avoidance_L1T0_cold_start_dev_01/v3_run`。

## Task-Feasible筛查边界修正

对L0T3/L0T4归档及共享筛查代码的复核发现，`task_approach_only`原先仍将默认10mm clearance应用到BDDL禁触对象，形成隐含Strict余量筛选。两轮的逐候选记录均未出现仅因正间隙小于10mm而拒绝的候选：L0T3的21项拒绝含实际碰撞、3项为IK失败；L0T4的24项均含实际碰撞。因此该实现偏差不是这两轮失败的原因。

已在[候选筛查实现](../research/gs_opd/repair/mujoco_jointspace_planning.py)中修正：task-approach模式以实际碰撞/穿透和IK可达性评估任务可行性，不再继承10mm安全余量；Strict-10mm与2mm授权穿透边界的定义和安全认证逻辑保持不变。该修正发生在L1T0归档完成之后；L1T0运行仍对应当时的源码指纹`d0ea082cf6d31e6c7b4e5f6a55a80a95a8efe8e3f8a157c634b3106e4ad12e2b`。当前源码bundle指纹为`6b75e613b78f6662ad650d3a51d892bfcbbe6150120847aeea927ee788706371`，按verification JSON中记录的有序相对路径/文件SHA-256规则计算。

## 权威记录与下一步

30项逐任务状态唯一来源为[任务矩阵CSV](GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv)；本地注册核验见[verification JSON](GS_OPD_SAFE_TRAIL_TASK_MATRIX_VERIFICATION.json)，协议见[矩阵说明](GS_OPD_SAFE_TRAIL_TASK_MATRIX.md)，过程及历史见[实验进度](GS_OPD_PROGRESS.md)与[路线图](GS_OPD_ROADMAP.md)。下一步以修正后的现有V3流程为`obstacle_avoidance/L1T1`生成合法t0并执行标准1000步冷启动，记录运行源码指纹、动作及子步证据，再继续未覆盖配置。统一冻结版本评测尚未开始；开发中定向调通结果与后续冻结评测分开统计。
