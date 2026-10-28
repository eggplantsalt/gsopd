## 2026-10-28 11:23 Asia/Shanghai更新：耦合候选支持与当前物理验证边界

抓姿搜索器增加了单一方向族与多对象轴偏移的耦合候选生成接口，用于把接近轴roll与object-local x/y/z接触偏移绑定后送入既有严格MuJoCo scratch screen。attempt41 L0T0/state0上本次screen构造16项，结果0项Strict可行；失败为7 collision、9 grasp geometry，0动作、0 RRT，live Integration State未变。screen SHA、完整候选参数和Snapshot身份见`GS_OPD_PROGRESS.md`以及机器`progress.json`。使用RLinf虚拟环境复跑抓姿screen、候选搜索、执行器与V3聚焦回归：122 passed、10 subtests passed；8个相关模块py_compile、两个CLI `--help`及`git diff --check`通过。

新的来源审计确认L0T0历史NPZ有845个7D transition，terminal row 844 reward=1/done=true。伴随JSON是749步B-Spline/native OSC运输段报告，不是整Episode报告。其保存规划几何与scratch重建的模型指纹不一致，且state ID映射未证实，因此V3目前不会把这份历史EEF姿态作为可迁移Strict抓姿。这个阻塞不影响attempt41真实完整任务成功证据，也不改变其未获Strict Safety、稳定释放或训练资格的分类。

当前验收为Level B。executor保留identity-checked Snapshot fallback和OSC joint-limit recovery路径；39项恢复相关测试及3个子测试通过。attempt41的物理分支与RRT预算各为1，但任务正常成功，没有实际触发恢复，因此不能把代码存在或单测通过说成真实恢复成功。四组核心套件与三组恢复套件合计161 passed、13 subtests passed。下一项工程任务是复现历史来源的准确导出运行时及模型指纹；只有状态、模型与当前合法Snapshot身份全部一致且Strict scratch screen通过后，才进入路线预检或物理执行。

## 2026-10-28 10:47：接触偏移复核与屏幕边界

以attempt41合法L0T0/state0 Snapshot和真实稳定抓姿为先验，对object-local-y偏移−8/−6/−4/−2 mm做4项MuJoCo scratch筛查。4项均有预抓取及抓取IK，但approach、闭合及pad-only接触条件失败；3项归入collision constraint，1项归入grasp geometry。screen report SHA `5ee2db723e6c549e108b2d8614749089173d23a4c0fe2c15d78754684cc1f62c`；任务主状态不变、没有RRT或实体动作。故该候选家族不产生Strict可执行抓姿。

同步复核L0T2 55 mm候选：一方向变体IK不可达，另一方向虽IK可达但预抓取碰撞且接近/闭合检查失败。14.2 mm变体也因碰撞失败。当前没有证据支持单独增加预开夹动作能解决该失败，因此没有变更执行器；若将来加入此动作，必须在明确开夹状态重做完整路径筛查，并让物理执行动作与被筛查姿态一致。

## 2026-10-28 10:43闭合校准交叉核验与开口边界

2.560 mm candidate width加active MuJoCo 1.000 mm pad-gap偏移后得到3.560 mm关节间距，与attempt41物理稳定抓持源qpos的3.5639967 mm间距相差约4 μm。对应回归测试已加入；抓姿screen、搜索和V3相关测试共75 passed、4 subtests passed。

Screen按当前live开口姿态生成闭合扫描。L0T2一项目标宽度55 mm的候选超过t0当前39.699 mm pad gap，虽未超过声明的87 mm开口上限，仍需先显式开夹。任何后续接受都必须实现匹配的原生开夹控制、在最大开口下重新检查完整接近几何，并再评估物理分支；本轮未执行此动作。

## 2026-10-28 10:40回归与归档验收

抓姿screen、搜索器与V3入口相关测试合并为74 passed和4 subtests passed。attempt41原始V3报告保持原样，其独立后验审计合并抓姿/放置归档核对669条物理动作记录、16,725个子步。状态动作完整性已验证，但unsafe几何样本、未完成stable release与未导出的Observation-Action trace意味着没有安全训练数据。

## 2026-10-28 10:39：活动pad-gap校准候选闭合姿态

候选几何提供目标实体的物理夹宽，但Panda指垫之间的MuJoCo几何净距不等于两关节qpos之差。本轮L0T0 t0实测两者偏差为1.000 mm。Strict screen现从活动开爪scratch状态精确测pad geomDistance，并对每个带宽度候选计算`joint separation = estimated_clamp_width_m + (open joint separation - open pad gap)`，随后对称映射到双指关节qpos。每候选屏幕和屏幕汇总都写入开放间距、实测pad gap、偏移及最终qpos。无夹宽的技能模板使用核验过的历史闭合qpos。

12项单测、py_compile与diff检查通过。校准后实际screen覆盖L0T0 48项基础几何候选和L0T2 state0 16项基础几何候选，0 Strict可行；所有screen均完整、live状态不变、没有RRT或物理动作。严格规则、BDDL与支撑语义未变。Screen通过/失败只提供离线抓姿几何证据，不等于物理安全认证。

## 2026-10-28 10:35：Strict抓姿screen按候选夹距闭合

Strict scratch-screen现在读取每个候选自己的`estimated_clamp_width_m`，为两指Panda生成`[width/2,-width/2]`闭合关节姿态；验证正有限宽度、声明的最大开口与活动关节限位。只在候选缺少夹距元数据时，才使用来源已核验的历史稳定抓持qpos。候选报告包括宽度、实际qpos、其来源和每个闭合采样状态。9点闭合扫描、10 mm非pad间距、2 mm指垫穿入阈值及scratch-only语义均未改变。

回归：12项`test_mujoco_jointspace_grasp_screen`通过，含18.68 mm→`[9.34,-9.34] mm`、缺省宽度回退与非法宽度/超限候选拒绝；py_compile、`git diff --check`通过。

实际复核：L0T0候选0–47分两批、L0T2 state0基础候选0–15各一批，均精确恢复合法t0 Snapshot，仅对独立MuJoCo scratch状态做几何与IK计算。L0T0 48项、L0T2 16项均0 Strict可行；所有live状态未变、无物理动作、无RRT。screen本身仍不是物理安全认证或任务成功证明。此前attempt41 Level B归档与Strict否定结论维持独立。

## 2026-10-28 10:07：attempt41抓姿Strict gate暴露非pad目标干涉

将attempt41同场景严格抓姿screen结果与已完成V3物理结果关联后，确认该真实任务成功抓姿也不满足Strict。其预抓取可达，双侧pad到碗的有符号距离约-0.295/-0.087 mm，符合最多2 mm指垫—目标穿透；但最后接近/闭合阶段非pad `gripper0_finger1_collision` 对黑碗几何穿透约0.153 mm，掌部也碰碗。安全候选筛选报告`/opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_attempt41_strict_screen_followup/task_success_first_grasp_obstacle_avoidance_L0T0_20261028_100423_300214570/strict_grasp_geometry_screen.json`。

对该姿态绕接近轴筛±5/10/15°共6项，0可行；±5/10°仍有同类非pad碰撞，±15°又造成约2.378 mm pad穿透。由完整活动几何池取出未测过的其余15个首批姿态再筛，仍0可行，collision 7、grasp geometry 8。三批均为scratch-only、live state不变、动作0、RRT 0。Bowl目标接触按task-aware屏幕规则处理，Strict候选不允许非pad接触目标；外部禁触物clearance仍须满足10 mm。候选批次和阈值不变。

下一步筛剩余32个不同基姿；全池无严格可行候选时停止该离散生成族。即使另有基姿通过，也须先做预算内精确场景RRT预检和独立物理安全门，再选择一个Snapshot隔离分支验证；screen通过本身不代表安全任务成功。

## 2026-10-28 09:18：接触面局部y候选无改善

新增`--candidate-offset-object-y-mm`只用于Strict screen且必须绑定已生成candidate ID；沿活动对象local-y平移候选EEF，不改姿态，仍遵守24项/批次与现有严格接触规则。几何驱动的两个handle基姿×3个外侧面位置共6项，0 feasible。候选pad距离约0.078/0.343 mm，但闭合阶段nonpad clearance最多0.446 mm（要求10 mm），报告SHA `f63ca8664f89c4f43ec537f94eea11248cbc5030754028cc5aa4bde1c7754d28`。

累计L0T2 104次严格离线评估、0可行，0物理动作、0路线预检。摘要`l0t2_strict_grasp_pool_summary.json` SHA `ae9d44090035338e70a74e235527fde0e31722262e1bbf4e139625e43a4518b7`。同一handle平移没有改善闭合干涉；后续只考虑基于编译碰撞面与非pad clearance的耦合姿态/接触求解，不重复高度/倾角/平移变体。

## 2026-10-28 09:12：resume Snapshot envelope与t0 manifest都在V3门前验证

Strict gate现复用grasp-search的Runtime Snapshot identity validator，在Bridge/engine启动前验证resume payload的task、model、BDDL、seed、初态、instruction和MuJoCo integration state。t0起点也核对Snapshot manifest中的payload/model/integration-state SHA。确切screen payload与选中初态仍需SHA一致。测试覆盖身份匹配resume、错误task_id持久拒绝和root t0 manifest载荷不符；V3测试31 passed。

L0T2 t329 Snapshot经只读验证为本任务合法stage：payload SHA `34f213c63d87be1a26cd69a3e4d442f9df9f2dcf4d406230a32ecea1dcd00673`，integration state SHA `a24835bf1b93399a78d2802e71f551984289b5f19a59b2e09f9de6b1fb270ed1`。把既有t0严格screen与t329 resume路径组合，V3在engine/Bridge前因Snapshot SHA不符归档拒绝，report SHA `06ed14d701a40f6b3bf8d8518087738d86e5c72bfbfa142955bef4911df0f078`。t329不能复用t0抓姿结果；其约0.527 mm非pad穿透和已有L0T2 strict零候选结论不变。

最新t0真实gate仍12/0并阻断，报告SHA `2d3c20e427699f48765e58d8e8f4469b2dfd6cf3b37ab37a0732b55ef13a190c`。下一步不扩大现有局部偏移族；任何新候选必须由新的编译碰撞面/指垫净距测量驱动，严格阈值与BDDL维持不变。

## 2026-10-28 09:08：异常输入归档、正向候选锁定和有限变体结果

L0T2全部严格抓姿screen现覆盖48个唯一基姿、24个先前局部变体、2个历史pose prior，以及12个把手jaw倾角和12个把手接触高度追加变体，共98项、0 Strict可行；所有screen零物理动作、没有RRT。完整路径和逐组SHA在`l0t2_state0_attempt39_strict_grasp_screen/l0t2_strict_grasp_pool_summary.json`，SHA `b82c4ad5ae56ae04c27cce2af2f258e9bdc28732d4859110debeef7c13011f21`。

V3严格门禁的错误backend拒绝和错误字段类型现在也写入`rejected_strict_grasp_screen` manifest/report。新增正向candidate pinning接线覆盖：排序可行ID作为executor `--candidate-id`；现有任务/manifest/Snapshot身份匹配保持不变。相关runner测试28 passed，`py_compile`与diff检查通过。最新真实screen 12/0仍阻断于engine/Bridge前；V3报告SHA `8bffd2cd95c393f6507ff79cc5d17fb6b62f58a738f39c9dd42461fcd9dc8d4d`。

# GS-OPD经验先验轨迹生成器V3

## 2026-10-28 10:02：attempt41 V3原生成功；双归档动作审计完成

V3 `experience_task_success`从精确合法L0T0/state0 t0 Snapshot（SHA `bc86043308518e857703b319b2c309e4e9ca62b332fc939be20235991610cba7`）执行，使用`geom113_jaw0_approach2_+1_flip+1`，最多1次RRT预检和1个物理分支。抓姿搜索观测28个连续稳定持抓状态、抬升37.119 mm；执行器随后完成原生放置。`on(akita_black_bowl_1, plate_1)`于t669为true，reward=1、done=true，1000步任务成功。没有使用Strict grasp screen gate。

原始`v3_report.json`（SHA `d6a39efc0f6665ae1caa88a8322e6a75e95c49891695657945bd1c44aec8f694`）分类任务成功，但其`physical_action_audit`仅聚合搜索报告，漏掉放置502条记录；并且搜索分支动作计数字段为`branch_control_actions`。代码现按搜索和放置两个归档源聚合，并保留各源的连续索引、分支计数和完整子步核验。独立后验文件`attempt41_v3_physical_action_audit_posthoc.json` SHA `758dfaebd6deea1b5b958052e242d21d6524f65a461ed6b347eed10525b62e88`：2分支，抓取167+放置502=669条动作，16,725个物理子步；669条均完成25子步监测，动作维度与state/action字段有效，动作数和放置子步摘要匹配。此报告引用原始V3与executor报告，不改写原始归档。

归档中669条geometry诊断状态为unsafe；这不是Safety pass。控制在原生done处终止，stable release/settle/withdrawal未完成，Operational Safety未认证，Strict-10mm=false/未认证，`safe_training_eligible=false`。V3 backend未输出经验证的Observation-Action训练轨迹，因此结果只作为Level B任务可行性数据。L0T2 attempt40的122项Strict筛选仍为0可行，现有抓姿变体族已停止；继续到Level A前先要求新的严格离线抓姿证据。

新增覆盖搜索+放置的回归测试；相关两项通过，`py_compile`和`git diff --check`通过。

## 2026-10-28 09:30：L0T2抓姿筛选停止；V3 L0T0物理集成复验进行中

`screen_grasp_candidates`现在在Strict pad-only模式的每个检查阶段保留与目标Pot最近的机器人/物体Geom ID、名称和世界最近点；诊断由几何距离本身提供，不改变筛选阈值或候选排序。V3 runner与search接线不变。单元测试验证最近距离及pair信息；抓姿screen/search聚焦测试37 passed，planner与测试文件`py_compile`通过，`git diff --check`通过。

L0T2 attempt40对两个g12把手基姿做±10/20/30°接近轴滚转共12项，并对相邻g13/g14段做6项局部接触中心筛选；全部0严格可行、0动作、live state unchanged、RRT 0。滚转最近对是非pad手指碰撞Geom与pot g11；移至g13后最近非pad间隙仍为0.446–0.460 mm，接触段切换至g13/g14但未靠近10 mm要求。汇总路径为`l0t2_state0_attempt40_roll_geom_diagnostics/l0t2_attempt40_geometry_diagnostic_summary.json`。L0T2抓姿变体搜索停止。

正在运行attempt41：V3 `experience_task_success`从合法L0T0/state0 t0 Snapshot，使用历史验证的`geom113_jaw0_approach2_+1_flip+1`抓姿，限制1次12秒RRT预检和1个物理分支。修复后的编译目标支撑映射与端点停止逻辑由V3入口统一调用。该回退用于验证新完整原生任务成功是否能通过V3外层分类；不使用Strict screen gate，不声称Strict Safety或训练资格。结果待实测。

## 2026-10-28 09:02：Snapshot场景身份逐字段门禁

Strict screen gate现在从活动`runtime_snapshot_manifest.json`核对benchmark、level、task_id、init_state_index和environment_seed，并继续校验model geometry/BDDL SHA及screen中的精确Snapshot payload SHA。这样相同模型或BDDL上的另一个初态/seed不会被误认为同一运行状态。不匹配时保存`rejected_strict_grasp_screen`，不启动checkpoint、engine、Bridge或动作。不同task manifest的回归通过；Adapter+V3聚焦测试28 passed。

## 2026-10-28 09:00：L0T2抓姿候选池有限耗尽结果

attempt39首批仅覆盖候选0–15；随后补筛16–39和40–47。三批共48个唯一基础姿态（9个经验模板、39个活动编译几何姿态），0个Strict可行；主失败类为碰撞20、抓姿几何14、IK 14。再加24个接触高度/jaw倾角变体和2个历史成功Episode抓姿，共74次screen评估，0可行。所有screen均为scratch-only，物理动作0、live state unchanged、RRT=0。阈值/BDDL未更改。机器可读汇总与每个子报告的路径和哈希见`l0t2_strict_grasp_pool_summary.json`。

最后8项screen接入V3 Strict gate后也返回`blocked_no_strict_grasp_candidate`，无engine、Bridge或物理动作。48项离散池已用完，不再用该生成族重复筛查；下一步集中在V3身份校验、candidate pinning、Snapshot恢复和持久失败报告。机器汇总位于`l0t2_state0_attempt39_strict_grasp_screen/l0t2_strict_grasp_pool_summary.json`。

## 2026-10-28 08:54：run53区域内释放经验的迁移边界与失败归档

只读复核L1T2 run53的`t1335`任务成功Snapshot与L0T2 t329合法持抓Snapshot：两个活动`flat_stove_1_cook_region` site尺寸相同、朝向相同，中心相差约11.995 mm。25 mm放置内移策略可作为Region局部技能重定位线索，应从目标场景的实时物体/Region相对位姿重算位移；源世界坐标和动作不重放。run53来源的77动作/1925子步分支实现真实松爪与settle期原生谓词保持，但来源`Safety certified=false`，撤离阶段仍有22个Pot-hand接触子步。该技能只能作为任务局部先验。

源run53 Snapshot处于On谓词已满足后的区域内放置/释放阶段；L0T2 t329 Snapshot处于抬升/运输阶段、`On=false`，当前严格抓姿筛选也没有可行抓姿。因此本轮未执行跨场景物理动作或路线预检。审计报告路径与SHA见`GS_OPD_PROGRESS.md`。

## 2026-10-28 08:54：Strict-screen身份错误的耐久拒绝

V3 runner此前会在严格screen身份校验异常时抛出`ValueError`，虽然不会加载checkpoint或启动engine，但没有写终态manifest。现在将缺失/格式错误/任务、阈值或Snapshot SHA不匹配报告为`rejected_strict_grasp_screen`，同时保存异常类型与原因、明确`engine_launched=false`、`bridge_started=false`、动作0。门禁也核对screen为完整scratch-only、没有预先Route preflight，以及候选行数、可行数和排序ID一致。五个回归覆盖错误Snapshot hash、JSON根类型、screen类型、route状态与候选计数/排序；聚焦测试27 passed。

使用实际L0T2屏幕报告复跑后仍以`blocked_no_strict_grasp_candidate`退出，候选16、可行0、无engine/Bridge/物理动作。最新报告SHA为`d9c2d640b6949ad6e0435ce4e385cd3fc060791588608c8bd2a4ce625a0d029a`。这个结果验证了runner的严格门禁与失败保存，不等于任务成功或安全认证。

下面的命令可复现V3门禁；输出目录带时间戳以避免覆盖已有实验。当前存档的零候选会让CLI按设计以退出码2返回`blocked_no_strict_grasp_candidate`。

```bash
RUN_DIR=/opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t2_v3_gate_replay_$(date +%Y%m%d_%H%M%S)
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety:/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4:/workspace/zy_workspace/RLinf-gs-opd-overfit-recovery MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/workspace/zy_workspace/.libero_safety_config /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python /workspace/zy_workspace/RLinf-gs-opd-overfit-recovery/research/gs_opd/repair/trajectory_generator_v3.py \
  --backend experience_task_success \
  --output-dir "$RUN_DIR" \
  --benchmark obstacle_avoidance --level 0 --task-id 2 --init-state 0 --seed 0 \
  --instruction 'put both moka pots on the stove' \
  --experience-object-id moka_pot_2 --target-relation on \
  --target-region flat_stove_1_cook_region \
  --target-predicate on moka_pot_2 flat_stove_1_cook_region \
  --snapshot-root /opt/data/private/gs_opd/rollouts/task_success_first_l0_batch_20261026_run1/obstacle_avoidance_L0T2/stage_snapshots/t0 \
  --resume-from-runtime-snapshot /opt/data/private/gs_opd/rollouts/task_success_first_l0_batch_20261026_run1/obstacle_avoidance_L0T2/stage_snapshots/t0/snapshots/initial_runtime_snapshot.pkl \
  --experience-library /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2.json \
  --keypose-experience /opt/data/private/gs_opd/experience/gs_opd_l0t2_keypose_experience_20261026_v1.json \
  --strict-grasp-screen-report /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t2_state0_attempt39_strict_grasp_screen/gate_screen/task_success_first_grasp_obstacle_avoidance_L0T2_20261028_084728_796132927/strict_grasp_geometry_screen.json \
  --rlinf-python /workspace/zy_workspace/RLinf/.venv/bin/python \
  --bridge-port 38718
```

## 2026-10-28 08:49：L0T2身份核验与Strict门禁验证

V3新增`--strict-grasp-screen-report`入口。执行经验任务后端前，runner核对严格screen JSON的schema/状态、零物理动作、live state未变、未获安全认证、任务谓词、模型/BDDL、屏幕阈值和确切起始Snapshot SHA。screen报告有严格可行抓姿时runner将候选锁定到报告中的已排序可行项；可行数为0时将写出`blocked_no_strict_grasp_candidate`的manifest/report，然后停止，不加载checkpoint、不启动experience engine/Bridge、不发送物理动作。真实L0T2 attempt39运行确认`candidate_count=16`、`feasible_count=0`、`engine_launched=false`、`bridge_started=false`、动作0。记录SHA见`GS_OPD_PROGRESS.md`。

L0T2经验Adapter将1460步Operational-Safety重构来源中的真实t249–329连续持抓窗口核验为Moka Pot2局部姿态/抬升先验；它保留原始变换值、不重放源动作，并将`strict_10mm_pass=false`和`training_label_valid=false`显式保留。L0T2 Keypose序列已离线重定位为7个剩余里程碑，但由于当前无严格合格抓姿，没有路线预检或物理阶段。

测试：Adapter及V3 runner相关测试22 passed；相关模块`py_compile`和`git diff --check`通过。此次验证的报告状态不是任务成功或Safety通过，而是证明严格screen结果可以约束V3是否启动物理后端。

## 2026-10-28 08:37：711/848-step成功Episode抓姿重定位验证

新增screen-only外部抓姿先验加载入口`--historical-grasp-pose-priors-json`。输入v1 JSON明确任务身份、source NPZ/result/evaluation/contact报告SHA、base/adapter身份、原生成功标记、无Safety资格继承、contiguous pose samples、pad contact与lift证据。loader重新核验source哈希和NPZ metadata/terminal `done`，从每帧MuJoCo物理pose重算object-relative transform，并从首个接触前的实际EEF运动重算object-frame approach方向；该模式必须screen-only，不能混用几何candidate ID或姿态偏移，也不重放历史动作。

attempt38使用两条来源各一个稳定窗口候选。两组历史Transform位置相差0.845 mm、转角4.82°，说明它们共享相近抓持模式。t0 Strict scratch筛选都因approach与非pad手指—碗碰撞失败；目标pad接触落在−0.36至−1.07 mm，但非pad最小净距为−0.97至−1.55 mm，远低于10 mm要求。没有RRT或物理动作。输入JSON和报告路径/哈希见`GS_OPD_PROGRESS.md`及progress.json。

审计发现两个源NPZ的`post_robot0_eef_quat`与同一步物理JSON的EEF quaternion相差约90°。对照robosuite记录和Safety Bridge实现后，确认前者是EEF body quaternion、后者是EEF site姿态；抽样step 0/27/39/46/51/74/160/700的body-to-site矩阵最大偏差0.000887。因此这不是序列化错位。抓姿转换使用物理state还原的site pose以匹配规划工具帧；两个姿态先验仍因严格几何门禁失败而只作拒绝证据，不导出训练数据。

L0T0先验抓姿方向已有限停止；下一步按照用户既定路线转到L0T2完整阶段经验重定位，沿用相同安全筛选与资格边界。

## 2026-10-28 08:19：pose-aligned抓姿筛选与候选姿态切换

修复`t0 EEF`与历史抬升碗Pose错帧后，attempt33/34在相同合法state0、活动BDDL和不变阈值下重新筛查48个基姿：0严格可行，25个虽进入双侧pad±2 mm区间，但非pad finger与碗仍有约0.095–0.111 mm穿透。继续比较jaw倾斜、绕approach轴roll与目标物体局部z接触中心偏移三个有限变体族，各24个、0可行。attempt37的最近pad间隙约±1 mm，finger2—碗仍穿透约0.805 mm。所有为隔离scratch几何筛选、物理动作0；Safety不认证，严格规则没有修改。

attempt37新增`generate_grasp_surface_offset_variants()`和屏幕CLI参数`--candidate-offset-object-z-mm`，仅限screen-only、候选ID显式指定、每批最多24个、偏移非零且≤20 mm。其坐标作用于当前目标物体局部z，旋转与approach方向保持不变。回归测试覆盖SE(3)位移和4×6=24候选预算；聚焦测试32 passed、py_compile与`git diff --check`通过。

attempt37归档：`/opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt37_object_z_offset_screen/task_success_first_grasp_obstacle_avoidance_L0T0_20261028_081707_204419156/run_report.json`，SHA-256 `d33e49a944f61d591fd4485dd15c9412ad99938608655b922f772403d68413ac`；Strict screen SHA `3be0c8ad7c08ccb0a15522d5710cebdb9f74c6fce447395bec18a0ed0fc30df6`。source Snapshot SHA `bc860433…610cba7`，模型几何SHA `085817d1…3b2e840`，BDDL SHA `b631d2f9…b7fe12`。

下一步按预设回退顺序审计711/848步成功轨迹，核验checkpoint和物体/EEF数据的帧约定，再提取有双指接触与抬升证据、且与已失败候选不同的object-relative抓取变换，重新执行同一严格scratch门禁。只有静态筛选与路线预检同时通过才允许物理分支。

## 2026-10-28 08:10：抓姿筛选发现对象Pose错位；attempt28–32接触结论暂时撤回

代码复核发现`screen_grasp_candidates()`在approach检查后，把目标碗切到attempt21闭合/抬升后的历史世界Pose（比t0高约37.4 mm），却让候选EEF留在当前t0抓姿，然后继续测闭爪pad距离和候选held transform。这样close接触双方不在同一世界时刻，可能伪造指垫间隙与碰撞。attempt28–32的几何screen虽保持live state不变、没有物理动作，但其闭爪和held阶段筛选结果无效；因此“48候选均不可行/候选池耗尽”的结论暂停，不作为最终研究证据。

已修复：历史抬升Pose只用于目标物体路径clearance prior；闭爪碰撞检查前把scratch目标恢复到活动t0 Pose；候选held transform从候选EEF抓姿和活动t0目标Pose共同计算。新增37.4 mm差异回归样例。当前聚焦pytest 28 passed，相关模块py_compile及diff检查通过。下一步以相同模型、BDDL、Snapshot、阈值重新筛48个候选（两个24批次），物理执行继续禁止直至有通过项。

## 2026-10-28 attempt28–32：严格抓姿筛选接口与结果

任务入口`research/gs_opd/repair/task_success_first_grasp_search.py`现在允许专用于scratch筛选的24候选分页和48候选有限生成池；正式物理候选上限维持16。screen-only模式可通过重复`--candidate-id`筛选最多48池中特定候选，Bridge每次仍最多评估24个。筛选结果返回显式`target_contact_policy`：只允许活动Panda指垫Geom接触目标；严格模式下非pad机器人—目标净距至少10 mm，pad穿透限于2 mm。完成筛选但0可行候选是正常的CLI退出状态；screen报告保留候选窗口、使用的pad Geom、阈值与live-state hash。

尝试从attempt8核验的L0T0/state0 t0 Snapshot开始；身份SHA：Snapshot `bc860433…610cba7`，模型/几何`085817d1…3b2e840`，活动BDDL `b631d2f9…b7fe12`。attempt21经哈希核验的真实稳定抓取报告只给出闭爪关节、闭合后碗位姿和IK seed；没有重放历史动作，也不继承源Episode安全标签。

attempt28筛16个（0可行）；attempt30筛24个（0）；attempt31从扩展池筛24个，其中与前批重复2个，因为路由多样性排序随候选池大小改变；attempt32通过指定ID评估剩下2个（0）。因此共66次评估、48个不同候选，0个达到双侧pad≤2 mm接触且非pad≥10 mm。attempt30较近的抓姿仍有约4.5 mm pad间距、约−12 mm非pad目标穿透；attempt32两个补筛抓姿在pregrasp即碰撞。主要风险Geom对包含`gripper0_hand_collision`/`gripper0_finger*_collision`与碗编译Geom。所有screen结果`live_state_unchanged=true`、physical actions=0，Safety certification=false。attempt29错误使用默认task_id=1并被模型指纹门禁拒绝，亦为0动作。

复现attempt32最后两个唯一姿态的命令：

```bash
PYTHONPATH=/workspace/zy_workspace/LIBERO-Safety:/workspace/zy_workspace/LIBERO-Safety/third_party/robosuite-1.4:/workspace/zy_workspace/RLinf-gs-opd-overfit-recovery MUJOCO_GL=egl PYOPENGL_PLATFORM=egl LIBERO_CONFIG_PATH=/workspace/zy_workspace/.libero_safety_config /workspace/zy_workspace/LIBERO-Safety/.venv/bin/python /workspace/zy_workspace/RLinf-gs-opd-overfit-recovery/research/gs_opd/repair/task_success_first_grasp_search.py --benchmark obstacle_avoidance --level 0 --task-id 0 --init-state 0 --seed 0 --snapshot-root /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt8_snapshot_root --output-root /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt32_strict_grasp_screen_remaining2 --experience-library /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2.json --maximum-candidates 24 --strict-grasp-screen-only --strict-screen-reference-report /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt21_geom113_jaw0/task_success_engine/task_success_first_executor_obstacle_avoidance_L0T0_20261028_061650_951211258/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T0_20261028_061705_120518766/run_report.json --candidate-id geom115_jaw2_approach1_-1_flip-1 --candidate-id geom115_jaw0_approach1_-1_flip+1 --port 38645
```

聚焦测试27 passed；四个相关Python模块`py_compile`通过，`--help`只解析CLI而没有启动Bridge。此离散编译Geom主轴池已经耗尽；这是候选生成覆盖范围的限制，不是修改安全判据的依据。下一步需要新的、由真实闭合pad几何约束的连续/局部表面参数化，然后继续同一scratch门禁；0可行之前不做物理抓取。

## 2026-10-28：attempt27首个原生任务成功；修复终止后撤离动作

attempt27从attempt21的t167合法抬升持抓Snapshot精确恢复（Snapshot SHA `ba43889b…e943e088`，integration-state SHA `07b2def7…90a1edf`）。原生谓词`On(akita_black_bowl_1, plate_1)`首次在t669成立，reward=1、done=true；此前167步抓取前缀与502步运输/放置后缀形成669步连续Episode，满足1000步时域。释放条件已被控制器观测并进入`withdrawal`，但settle和撤离没有完成，故`stable_release_verified=false`。

终态审计发现源后缀还记录了t670的一个withdrawal动作，而环境在t669已终止。该动作从成功Episode中剔除；新review NPZ仅含截至首次done的669组7D动作与状态，不含双相机Observation，因此不是训练包。原始报告、Snapshot和帧归档未改写。归档：`/opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt27_obstacle_overflight/task_success_first_executor_obstacle_avoidance_L0T0_20261028_065344_974183682/expert_native_goal_stage_control/placement_branch_00_region_layout_12_direct_region_progress/review_episode_audit.json`，NPZ SHA-256 `8887dc6c…b1b6df48`。

有效669动作的16,725个物理子步记录完整，但几何状态仍有unsafe样本；`operational_safety=not_certified`、Strict-10mm=false、`safe_training_eligible=false`。attempt27越障路线避免了手部与香水瓶接触，但不能据此称安全成功。结果为验收Level B。

执行器现于native `done=true`后停止release/withdrawal/settle动作，并报告终态跳过撤离；新增终止边界回归测试。聚焦测试43 passed、6 subtests，py_compile和`git diff --check`通过。下一步先离线筛选不同的抓姿，要求非指垫—碗几何至少10 mm且指垫穿透不超过2 mm，并按活动BDDL约束预检路线；没有筛选和预检通过的候选时不启动物理分支，也不复跑同一路线。

## 2026-10-28：attempt26越障路线在零动作门禁失败，修复上下文接线

attempt26从attempt21 t167合法持抓Snapshot恢复；在物理动作前因`_native_place_single`未收到`target_context`而报告`NameError`。报告SHA `e9f92bab7cf4286d7defb69b12b8f070a58648fc10caac25ef4aad42eac3c7b7`，动作/子步均0、timestep保持167，原快照未改变。修复方式是在函数候选摆放调用显式传入已通过活动BDDL、model SHA核验的目标碰撞上下文；release续接路径不需要该上下文。聚焦验证42 passed、6 subtests；py_compile和diff检查通过。此为执行接线修复，attempt27按同一路线单分支重跑。

## 2026-10-28：attempt25碰撞诊断与活动编译包络越障

最新改动新增只读Bridge `task_scene_object_collision_envelopes`，逐个活动BDDLObject/fixture返回编译碰撞Geom的root-frame包络并带模型几何SHA。executor用这些包络与持抓物体的横向包络、EEF偏置检测直达走廊上的障碍物，最高障碍+50 mm减去持抓物体最低碰撞点，形成原生region-local自适应运输高度。Bridge必须先显式reset、再恢复Snapshot，才能与正式executor模型fingerprint一致。

attempt25的反向水平侧绕物理分支仍在首个waypoint停滞：24动作/600子步，子步记录完整、持抓保留、task false。逐子步证据显示`gripper0_hand_collision`与香水瓶三个Geom接触，最高力约12.45 N，最小clearance -0.563 mm。此前正向侧路线同样停滞；具体障碍接触说明应从物理记录归因为瓶体碰撞，不是单独的IK推断。

只读恢复attempt21 t167持抓抬升Snapshot后，对直达路线查询到6个相交对象包络，香水瓶顶1.21847 m；活动碗几何推导的运输高度为区域局部z=0.34804 m、世界root z=1.26697 m，EEF约1.29881 m。attempt26从t167合法前缀启动以保留运输/释放预算，单分支先升高后中心放置。代码聚焦测试42 passed、6 subtests；相关模块`py_compile`和`git diff --check`已通过。Task Success尚未验证；unsafe子步几何、Strict10mm和安全训练资格继续独立判定。

# GS-OPD经验先验轨迹生成器V3

## 2026-10-28 attempt27安全语义归因

attempt27达到新的原生任务成功（Level B），但没有安全认证。当前执行器的 `safety_policy_diagnostic_only.json` 将每个非目标对象都当作 forbidden；活动 BDDL 仅明确禁止机器人接触香水瓶/白色置物盒及黑碗接触这两者，并把炉灶支撑列为初始 `On`、盘子支撑列为目标 `On`。因此诊断策略的碗—炉灶与碗—盘子标记不是正式安全结论。Strict-10mm 已由实际非指垫接触失败：t168 finger1—碗穿透约0.183 mm，终态 finger2—碗间距约0.244 mm。路线未触碰香水瓶。V3 下一步应在不改阈值的前提下，先筛出手部非指垫几何保持10 mm余量的不同抓姿，并在未来分支中按活动 BDDL 约束重建安全归因。详见 attempt27 `safety_attribution.json`。


## 2026-10-28：attempt24 正向侧绕行在首个中间点停滞，attempt25镜像绕行

attempt24 从 attempt22 的timestep329终态快照恢复，以 `keypose_side_region_progress` 执行24个真实OSC动作、600个MuJoCo子步；每个动作的监测完整，双侧抓持未丢失。第一侧向中间点在24步内进展不足，原生目标仍为false。终态快照已保存，当前timestep353。

已新增镜像方向 `keypose_side_region_progress_reverse`，并用测试确认它跨越到直达线的另一侧。attempt25 从 attempt24 精确终态恢复，固定plate中心候选并仅运行一个反向侧绕行物理分支。仍保持原任务时域、BDDL、成功阈值和安全标注。聚焦pytest 41 passed、6 subtests；`py_compile`、`git diff --check`通过。

## 2026-10-28：attempt23 RRT预检无有界IK解，attempt24切换侧向绕行

attempt23 从 attempt22 的精确终态快照启动，task timestep保持329。一次12秒全几何放置RRT预检返回 `task_goal_has_no_bounded_current_scene_inverse_kinematics_solution`，因此没有执行物理动作，快照保持不变。

attempt24 从同一快照恢复，强制使用 `keypose_side_region_progress` 原生区域绕行，通过一个真实OSC物理分支绕开 attempt22 停滞的直达走廊。中心摆放目标、对象碰撞包络和BDDL谓词保持不变；预计进展与终态仍待实测。当前聚焦pytest为41 passed、6 subtests；代码编译和diff检查通过。

## 2026-10-28：attempt22 支撑面映射通过，直接OSC运输在走廊停滞

attempt22 精确恢复了 attempt21 的37.1 mm抬升快照，采用由10个编译碰撞Geom推导的plate_1支撑包络；居中候选位于支撑面局部[0,0]。单一路线实际执行162个动作、4050个MuJoCo子步，子步记录完整。双侧抓持持续确认；碗到目标中心的水平距离由49.1 cm降至31.6 cm，随后连续24步进展不足而安全停止。原生BDDL谓词仍为false，终态快照SHA-256 `c8c0e8cb575086c419a2b406abfd70027ce6da4bc03e0b9a85cc014736d3eca8`。direct OSC路线预算在该起点预估1278步，而当时预留的传输预算为497步，说明该路线在标准1000步任务时域内不适用。

已加入可选 `--native-region-route-variant`，可从中途精确快照恢复并强制使用侧向绕行。attempt23先从 attempt22 终态做一次12秒全几何RRT放置预检，最多执行一个物理分支；若无路线，可无损回到该终态快照尝试侧向绕行。当前聚焦pytest为41 passed、6 subtests；`py_compile`与`git diff --check`通过。安全认证、Strict10mm及任务成功仍未达到。

## 2026-10-28：attempt21 首次稳定抓取抬升，修复 plate_1 对象支撑区域映射

attempt21 使用候选 `geom113_jaw0_approach2_+1_flip+1__rrt0` 完成167个控制动作、4175个MuJoCo子步；子步记录完整，几何监测167步均记录为unsafe。抓取稳定性连续状态28/28通过，碗抬升37.1 mm，达到内部30 mm抓取抬升门槛。原生BDDL任务仍未成功：executor在摆放开始前报 `native_region_candidate_geometry_binding_unavailable`。审计确认 `plate_1` 是活动BDDL中的物体目标，不是 `object_sites_dict` site；site-only 查询错误地拒绝了合法目标。

已将 Bridge 的 native goal 查询扩展为：对site目标保持编译site读取；对对象目标从绑定的编译碰撞几何构造实时支撑包络。plate_1 当前包络由10个碰撞Geom测得，候选器已调整为在对象支撑面上优先居中，最终成功仍只由原生BDDL谓词确认。未改BDDL、成功阈值、场景对象或Safety Reconstruction。只读Bridge目标查询通过；候选中心核验通过；`py_compile`、`git diff --check`通过；聚焦pytest为40 passed、6 subtests。

attempt22 正从 attempt21 的精确抬升 Runtime Snapshot 继续，使用最多1条物理摆放分支和1次RRT预检。native task success、Strict10mm及安全训练资格仍未通过/未认证。

## 2026-10-28：attempt20四个新姿态RRT均超时，attempt21测试最后几何候选

attempt20从干净seed0/state0初态对候选索引10–13执行4次有界RRT预检：`geom112_jaw1_approach2_+1_flip+1`、`geom112_jaw0_approach2_+1_flip-1`、`geom111_jaw0_approach2_-1_flip+1`和`geom111_jaw1_approach2_-1_flip-1`均在约12秒达到规划预算且无路线。没有物理分支、动作或物理子步。V3报告SHA-256 `daa6179edcebacdac7e31553a2c0799c33d70e0beb19a6cdda39bc15410fd33a`，搜索报告SHA-256 `f688731a012678aefdf044e30688bc96fec5ffe046076690262c6000f9b4f401`。

attempt21仅剩未测的`geom113_jaw0_approach2_+1_flip+1`；编译Geom对应夹持轴宽约2.56 mm，路线和接触可行性未知。将使用同一哈希核验的t0 Snapshot，最多1个RRT预检和1个物理分支；Failure memory输入attempt8–14、17–20，排除先前失败的精确候选—路线对。当前代码聚焦验证70项测试+6 subtests、`py_compile`与`git diff --check`通过。attempt21准备启动。


## 2026-10-28：attempt19 geom113预检超时，attempt20合并四个有界预检

attempt19从哈希核验的seed0/state0初态筛选`geom113_jaw1_approach2_+1_flip+1`，其RRT预检在12.00秒时预算耗尽，没有生成路线，也没有物理分支或动作。V3报告SHA-256 `d4434ce41f83c554d0b81b0fc4950696170ef6ddc3e5c9fa2901eaf9aab35818`，搜索报告SHA-256 `e8cc709b83470dc5f696be8fa7ff963fb5c80062c1503171ce05efcc6ddb6d09`。归档保留，计入零物理动作预检。

attempt20仍从同一干净初态出发，使用`candidate_start_index=10`、最多4个12秒RRT预检，对候选10–13作一次批量筛选，最多仅对一个有路线的候选执行物理验证。Failure memory输入attempt8–14和17–19有效报告，跳过先前失败的精确候选—路线配对；attempt15/16元数据不完整零动作报告不作为failure memory。当前动态目标跟踪、候选位姿对齐门及其回归覆盖已通过70项测试+6 subtests、`py_compile`与`git diff --check`。attempt20准备启动。


## 2026-10-28：attempt18动态跟踪仍未到抓姿；attempt19换目标几何和干净初态

attempt18执行17步RRT预抓取与60步动态对象相对接近，共77动作/1,925物理子步；77/77动作25/25样本完整，几何监测77步均`unsafe`。动态跟踪把姿态误差从约17°降至0.2°，位置误差降至6.9 mm，但两侧指垫从未接触目标，且碗在接近阶段沿桌面滑移约53.7 mm，60步预算耗尽后停止，没有闭爪、稳定抓持、抬升或Task Success。该结果不支持继续延长这条同一目标路线。

新的位姿/接触记录显示：attempt11源Snapshot一开始存在夹爪—碗碰撞，恢复后首个预抓取动作就让接触消失；后续碗与炉灶燃烧器保持接触并移动。因此attempt19不再从这个扰动态续接，而从经哈希验证的seed0/state0原始Snapshot出发，换到另一个活动碰撞几何候选`geom113_jaw1_approach2_+1_flip+1`。最多一个RRT预检和一个物理分支；动态对象相对跟踪仍用于真实表面接近，抓姿误差仍须≤4 mm/0.045 rad才可闭爪。Failure memory导入attempt8–14、17–18。70项聚焦测试+6 subtests、`py_compile`和`git diff --check`通过，attempt19准备启动。


## 2026-10-28：attempt17固定世界坐标接近失配，attempt18改为实时对象相对跟踪

attempt17从attempt11经身份核验的末态Snapshot执行17步RRT/OSC预抓取、30步表面接近和18步闭爪，共65动作、1,625子步；65/65动作的25/25物理子步记录完整，65步几何监测均为`unsafe`。接近使目标碗移动并转动；现有实现把规划时的世界系抓姿固定下来，闭爪时EEF相对当前碗仍差16.6 mm、16.9°，并且没有左右指接触。无稳定抓持、抬升或原生Task Success，Safety未认证，Strict 10 mm=false，训练资格=false。失败Snapshot SHA-256 `e726dc1fde50048f8212c656ec2afacb34084e736f3453fef51300a526bafc73`；V3/executor/search报告哈希分别为`0ebe84a41336c69a98ce29271318345f549898825863281b634c8a2bab6c56ff`、`fd731cfe25194824ab2ed2a0f084608ac5542ddd7cd9076c07a015fef0311f4c`、`9aace512f2dd536692c23a37629759307627255c6ccdbe45d7f0c662dc1ab480`。

attempt18在同一经验证的接触快照上运行同一几何候选，但表面接近的每个OSC控制周期都会读取当前目标根位姿，并重算原有生成候选的`object_to_eef`目标。偏离候选时不作原位闭爪；闭爪必须先满足4 mm/0.045 rad跟踪容差。Failure memory导入attempt8–14和17，动态对象相对跟踪在route ID中单独标识；仅1个预检和1个物理分支。70项聚焦测试+6 subtests、`py_compile`和`git diff --check`通过，attempt18准备启动。


## 2026-10-28：attempt16无物理动作的输入校验失败，attempt17重跑

attempt16在Failure memory输入核验阶段拒绝了attempt15搜索报告：该报告源于前一轮矩阵索引错误，没有`native_goal_predicate`元数据。V3报告SHA-256 `335183fbf52bdf27fe951a8c309ccb6d67e6a4b8729f6a53a4e866a5ea003831`；动作审计确认物理分支0、动作0、子步0。该归档保留，不纳入后续failure memory。

attempt17重用attempt11身份核验的双接触Snapshot与精确候选；failure memory仅导入attempt8–14通过身份检查的报告。候选抓姿对齐判据为4 mm和0.045 rad；当前位姿不满足时按成功的RRT预检继续原生OSC路径与候选接近，最多执行1个物理分支。矩阵形状修正后69项聚焦测试+6 subtests通过，`py_compile`与`git diff --check`通过。attempt17准备启动。


## 2026-10-28：attempt15零动作形状故障已修复，attempt16继续候选接近

attempt15在当前接触Snapshot上完成了一个RRT预检，结果`offline_feasible`；随后姿态对齐检查把目标4×4矩阵误当成7D `[position, quaternion]`数组索引，导致预检/runner错误，没有真实物理分支、动作或子步。该0动作归档保留在`l0t0_state0_attempt15_contact_pose_alignment/`，V3/executor/search报告SHA-256分别为`e54325df7c1f69ca859f392c57d10733f5a368ad920a2fd5d3d6daa172c87fb3`、`a7f832b99b02f862887711e83f7441a9c29a14d1ed0c65d7732d40e5ea7b4a4a`、`eb53cd072a1dd684b8c2cb8d8fc66f2e88595026b30e4464898e5ab8a11c4b2e`。

矩阵输入现直接传入SE(3)对齐检查；使用同一attempt11接触Snapshot，闭爪仍须满足候选目标误差≤4 mm、≤0.045 rad，否则按规划结果继续对象相对接近。聚焦测试69项+6 subtests、`py_compile`和`git diff --check`通过。attempt16最多1次预检和1个真实分支，Failure memory包含attempt8–15报告，准备启动。


## 2026-10-28：attempt14验证闭爪错位；attempt15加候选抓姿对齐条件

attempt14从attempt11的精确Snapshot开始，两个中性等待动作期间双侧接触保留；随后首个闭爪动作（25%）即失去左指接触，50%时两侧均失去。20动作/500物理子步，20/20动作子步监测完整，几何监测20步均为`unsafe`，无稳定抓持或抬升；原生Task Success=false，Safety未认证，Strict 10 mm=false，训练资格=false。

离线由同一失败Snapshot和候选`object_to_eef`变换重算，发现EEF与候选抓姿相差103.2 mm、106.9°。旧接触探测只用宽松的位置距离≤约0.28 m，因而把“碰到目标物”误判成“已处在这个候选抓姿”，然后在错误位姿闭爪。这解释了attempt12和14的相同滑脱，也淘汰了继续增加等待时间的计划。attempt14失败Snapshot SHA-256 `bb5df0bb5f16b00f96494fdfaa8a52cbdbd7b9629f2124cd7fac1efb875090c1`；V3/executor/search报告哈希依次为`df4ac7fba7285f2558760f4fbd1bcdac2e49d53a5aa988c4d9d715f8f3a75963`、`f60339bc003baaf0a156f5d8622a8cd4c65bf3e1e768b551e32fb67f19cf351e`、`427abd34c5568762185a182b96d960e3869d3509657f56a644abcd42813ca29b`。

attempt15将原位闭爪条件收紧到已有OSC跟踪容差：位置误差≤4 mm且姿态误差≤0.045 rad。其余带目标接触但偏离候选抓姿的状态继续候选当前场景路线和OSC接近，不原位闭爪。该判据只是技能闭合准备度，不是安全门槛；BDDL和Safety阈值不变。failure memory导入attempt8–14报告。聚焦测试69项+6 subtests、`py_compile`与`git diff --check`通过，attempt15准备启动。


## 2026-10-28：attempt13等待过久丢失接触，attempt14收紧等待步数

attempt13在attempt11精确接触Snapshot上只执行了3个原生中性动作，75个物理子步均有完整记录。前两步后左右目标指接触仍在，碗线速度约7.4 mm/s、角速度约0.09 rad/s；第三步后左侧接触丢失，闭爪阶段没有启动。几何监测3步均为`unsafe`，Task Success、稳定抓持、抬升和安全训练资格均false。失败态Snapshot SHA-256 `78f06cc6475bfafe4174436682f0a4aa22fe8b43818647432a671d7969970cc4`；V3/executor/search报告哈希依次为`092077d1cd104248758106a65f996b3ec14bfd5a71b7ddb6753a69992a561460`、`850e264e8fad9aa5bc75f9845c3479ecf0823630e73dd006a11a0ae35434e1c5`、`36d43a19e7c5cd04181b779d74b05305cc27767a2a187923a24d62ba85449883`。

attempt14将中性等待上限缩至2步，接触仍在时立即闭爪；更新路线签名为`live_target_contact_settle2_close_probe`，并导入attempt8–13报告，让Failure memory跳过先前精确失败的接近/立即闭爪/长等待组合。只运行同一精确几何抓姿、一个预检和一个物理分支。聚焦测试68项+6 subtests、`py_compile`与`git diff --check`通过。attempt14准备启动。

## 2026-10-28：attempt12原位闭爪失败，改为接触稳定后再闭爪

attempt12_retry1从attempt11完全相同的双指接触Snapshot恢复，精确执行1个抓姿、1个物理分支。夹爪到25%和50%闭合时左右指仍接触黑碗；75%时左指接触消失，完全闭合时两侧接触均消失。18个闭爪动作内碗移动约12.7 mm，没有稳定抓持或抬升。18/18动作均记录完整25/25物理子步（450子步），动作与状态记录有效；几何监测18步均为`unsafe`，Operational Safety未认证，Strict 10 mm=false，训练资格=false。

失败态Snapshot SHA-256 `8c57e312d5bd670ff3b1e8c473db2dc7a2abed0eaeb6c5f25bf0f3e3a0106861`；V3/executor/search报告SHA-256依次为`0e65c6216e1881958d6311a1c52ab35fb2d94d45f4bbd7cac805b757546e4bfc`、`53b1946ad29623b054654b93ccbb0d4b189e324d000692e870ec466f2b698c15`、`30590e08f35eb4e11acef3abb0fa9672a5b558c790792f73c7318cfe60ac4b72`。

针对闭合时物体仍有运动的证据，attempt13使用同一个经身份核验的attempt11接触Snapshot与同一抓姿，但先以原生零动作保持双侧接触；只有物体线速度低于0.015 m/s、角速度低于0.08 rad/s并连续满足3次才提前结束等待，最多等待20步，接着沿用原抓持监测闭爪与抬升。若20步内未达到速度条件但双侧接触仍在，则记录并继续一次闭爪验证；若接触先丢失即停止该分支。没有写入机器人或物体状态，也不改安全判据。改动后`py_compile`、`git diff --check`和聚焦测试68项+6 subtests通过。attempt13正在运行。


本文记录生成器V3的实现边界、运行入口、产物和验证结果。V3先用一条同任务成功轨迹提取路线结构，再从当前LIBERO-Safety状态重新生成并闭环执行，不回放成功轨迹动作。

## 当前试点

试点固定为`obstacle_avoidance/L0T0/state0`、seed 0。参考Episode共845个7D动作，原生终端reward为1且`done=true`；已验证的左侧B-Spline/OSC段为749个OSC动作，参考速度0.020 m/s。该Episode没有稳定松爪或Strict 10 mm安全证明，以上标签不迁移给新Episode。

经验库对`akita_black_bowl_1/on/plate_1`匹配到一个黑碗局部抓持模板，无放置操作阶段或keypose。`student_bspline_osc`后端只把命中结果写入来源清单，不消费该抓持变换；`experience_task_success`后端则从实时目标状态重定位`object_to_eef`、重新规划接近并经MuJoCo物理验证。经验模板仍只是抓持先验，不能迁移任务成功或安全标签。


## Attempt9结果与下一次有限搜索

attempt9沿用attempt8 identity-verified的seed0/state0初始Snapshot，并导入attempt8抓持搜索报告。failure memory跳过历史高位接近失败对；V3生成五个对象相对接近族。本轮RRT预检索引0–2后，实际物理执行两个alternate approach：`side_v_pos`执行149步预抓取和60步接近，60步内未到达目标；`side_v_neg`执行119步预抓取、40步接近及18步闭爪，但左右指没有同时接触黑碗。两分支均没有稳定抓持、抬升、释放或原生Task Success。

总动作386，物理子步9650；386/386动作有完整25/25接触与几何采样，7D动作和状态行有效。几何检查386/386为unsafe；这些记录只证明覆盖完整，不是安全通过。V3外层报告状态`task_incomplete`、场景身份核验true、原生Task Success=false、Operational Safety未认证、Strict 10 mm=false、safe training=false。报告位于`/opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt9/v3_report.json`。

已将`--candidate-start-index`加入V3入口并转发给task grasp search，默认值0保持原有行为。attempt10首次启动在中间task executor参数解析阶段以code 2退出：`--candidate-start-index`尚未从executor转发到grasp search；该归档没有Bridge或物理动作。现已贯通V3→executor→grasp-search并通过对应CLI与command-forwarding测试。attempt10_retry1现用新目录设置索引3以选择`side_u_pos`、`side_u_neg`和`opposite`候选；同时传入attempt8、attempt9搜索报告，按精确candidate-route失败记忆跳过重复路线。物理预算仍最多3分支、3次12秒预检，使用标准1000步horizon和同一初始Snapshot。新增转发覆盖测试后，RLinf venv更新后的V3+executor聚焦测试50项、6个subtests通过，`py_compile`、`git diff --check`通过。

## Attempt10_retry1结果与下一轮夹爪姿态搜索

三条经验接近变体共执行719个MuJoCo控制动作、17,975个物理子步。`side_u_pos`执行123步预抓取、50步接近和18步闭爪；`side_u_neg`执行134步预抓取、58步接近和18步闭爪；两路在闭爪末端均没有左右指与目标碗的指垫接触。`opposite`分支执行318步预抓取后超时，观测到炉面与右指约−0.535 mm穿透，以及碗与左指接触。failure memory记录两次`grasp_or_lift`、一次`contact_blocked`，没有任一稳定抓持或抬升。

V3审计确认动作记录719/719，7D动作和状态行有效，719/719动作覆盖25/25接触与几何样本，合计17,975子步。几何监测719步均为`unsafe`，因此不能将完整监控解释为安全通过。原生Task Success=false，Operational Safety未认证，Strict 10 mm=false，安全训练资格false。外层报告SHA-256 `3603a67ce60fdfd94ee047f7f9bb4485633a4c0e13606e796070424723185617`，搜索报告SHA-256 `b34a42cf84c3d7d33e972fc78936cfc52939eb1594543116989d6eb79fe305a1`；完整报告、逐步动作审计和失败Snapshot位于`l0t0_state0_attempt10_retry1/`。

下一轮attempt11从新鲜的同一初始Snapshot起跑，搜索活动黑碗碰撞几何生成的抓姿索引7–9。经验抓姿的五种接近方向都未完成双指抓持，而新窗口改变jaw axis和`object_to_eef`姿态。保持最多3次RRT预检、3个物理分支、标准1000步horizon；failure memory同时读取attempt8、attempt9和attempt10_retry1报告，安全阈值、BDDL和场景均不变。

## Attempt11结果：出现真实双指接触并保存可恢复末态

索引7和9的RRT预检均在12秒预算内无路线。索引8 `geom111_jaw1_approach2_-1_flip+1`以一条record-only RRT路线进入真实执行，完成90步预抓取和60步接近后未达到目标姿态。其末态grasp诊断记录到左侧指垫1组、右侧指垫3组对黑碗Geom的接触，但夹爪仍为打开命令`[0,0]`，所以状态为`not_grasped`，没有闭爪确认、稳定抓持或抬升。

150个动作的V3审计完整：150/150条7D动作/状态行有效，每条动作都有25/25接触和几何子步记录，共3,750子步；几何状态150/150为`unsafe`。Native Task Success=false，Operational Safety未认证，Strict 10 mm=false，safe training=false。末态Snapshot `geom111_jaw1_approach2_-1_flip_1__rrt0__failed_runtime_snapshot.pkl`已只读校验：task、model、BDDL、seed 0、state0匹配；payload SHA-256 `36572abebac8e85ad5227f6ad5b4b5b28c2c078508f940e6928663aafb08fbcc`，integration-state SHA-256 `0534c2a3202340d7f540f3ad09d1f9d65d4088f6ad6d4c21337b402d193c0899`。

根据此实测证据，V3加入`--resume-from-runtime-snapshot`与`--grasp-candidate-id`，将快照路径和SHA记录进输入清单，并贯通V3→executor→grasp-search。Grasp search现在保留左右目标指垫同时接触的状态；当夹爪仍张开时，安排原位闭爪探测而不退回预抓取路线。68项聚焦测试+6 subtests、`py_compile`与`git diff --check`通过。

attempt12从上面的exact failure Snapshot启动，只选同一个几何抓姿ID，最多一个RRT预检和一个真实分支，标准horizon 1000。它会导入attempt8、9、10_retry1和attempt11搜索报告：attempt11的旧接近candidate-route pair由failure memory阻断，而原位闭爪probe具有独立route identity。验收依次要求稳定双指抓持和至少30 mm实测抬升；之后才进入既有放置/原生Task Success路径。全程维持安全阈值、BDDL与场景不变，Unsafe诊断仍不能通过Safety门线。

## 执行链路

`research/gs_opd/repair/l0t0_success_reference.py`只读验证参考轨迹的任务身份、N×7有限动作、reward/done长度、终端成功证据、可选报告一致性和文件SHA-256。对直接B-Spline报告，它只导出方向、速度和路线步数等结构字段，不导出或重放动作数组。

`research/gs_opd/repair/trajectory_generator_v3.py`校验检查点及经验来源后，按后端调用现有物理执行器。`student_bspline_osc`启动LIBERO Safety Bridge，使用带authkey的multiprocessing协议只读核对活动指令、state/seed、BDDL谓词、语义绑定及编译Geom，随后由π₀.₅从当前观察启动并用原生OSC_POSE闭环执行；可选`--pre-route-hold-actions N`逐步检查子步监测、接触、双指抓持和既有姿态门限，之后重查场景几何并重建路线。

`experience_task_success`使用`task_success_first_executor.py`：从带manifest的L0T0 seed/state Runtime Snapshot恢复，经活动BDDL确认任务目标后，把命中的同任务`object_to_eef`变换作为抓持几何候选交给`task_success_first_grasp_search.py`。该搜索在当前MuJoCo场景重新生成接近路线、做有界RRT预检并逐分支真实执行；抓持验证后继续原生任务放置/释放闭环。嵌套executor报告作为物理明细来源；V3外层以原生Task Success独立分类，未聚合完整子步和Observation–Action记录时标为`not_aggregated`/`not_emitted`，不推断安全或训练资格。

默认验收分别输出：原生任务成功、1000步标准成功、稳定释放、操作安全、严格10 mm、安全监测覆盖、动作轨迹完整性和安全训练资格。Native success不推导安全；没有同Episode严格审计报告时，Strict 10 mm与训练资格为false。系统不改BDDL、场景对象或安全阈值。

## 运行入口

以下为attempt7预运输等待实验命令（历史运行记录）。它校验三个同场景成功参考和经验库，最多执行1000动作，并将全部运行文件写入新目录：

```bash
/workspace/zy_workspace/RLinf/.venv/bin/python -u research/gs_opd/repair/trajectory_generator_v3.py \
  --backend student_bspline_osc \
  --output-dir /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt7 \
  --benchmark obstacle_avoidance --level 0 --task-id 0 --init-state 0 --seed 0 \
  --instruction "pick up the akita black bowl on the stove and place it on the plate" \
  --experience-object-id akita_black_bowl_1 --semantic-object-id object_1 \
  --target-relation on --target-region plate_1 \
  --target-predicate on akita_black_bowl_1 plate_1 \
  --route-candidate-id trajectory-003 --pre-route-hold-actions 20 \
  --reference-trace logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/direct_bspline_osc_trajectory.npz \
  --reference-report logs/gs_opd/mink_panda_physical_v3/live_left_bspline_osc_adaptive_20261019_attempt1/live_left_bspline_osc_report.json \
  --additional-reference-trace logs/gs_opd/eval/pi05_libero_full_trajectory_lora_constant_lr_attempt3_obstacle_avoidance_l0t0_20261020/student_rollout.npz \
  --additional-reference-trace /opt/data/private/gs_opd/eval/pi05_pi2_cumulative_fresh_bs3_obstacle_avoidance_l0t0_state0_seed20261020_20261021_attempt1/student_rollout.npz \
  --checkpoint /workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft \
  --experience-library /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2.json \
  --teacher-json /workspace/zy_workspace/RLinf/logs/gs_opd/teacher/initial_center_distinction_teacher_raw.txt \
  --semantic-results-json /workspace/zy_workspace/RLinf/logs/gs_opd/semantic_bindings_l0t0.json \
  --max-steps 1000 --bridge-port 38567
```

脚本从仓库根目录运行；它会管理执行器生命周期、写出清单并在执行完成后生成V3结果报告。每次尝试使用此前不存在的`--output-dir`，以保留预检失败和物理执行产物。

attempt9沿用同一个经校验的seed0/state0 Snapshot，并输入attempt8的抓持搜索报告。V3会记录前次报告哈希；执行器会验证其任务、模型及BDDL身份，再将失败candidate/route对导入failure memory。运行命令：

```bash
/workspace/zy_workspace/RLinf/.venv/bin/python -u research/gs_opd/repair/trajectory_generator_v3.py \
  --backend experience_task_success \
  --output-dir /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt9 \
  --snapshot-root /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt8_snapshot_root \
  --previous-search-report /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt8/task_success_engine/task_success_first_executor_obstacle_avoidance_L0T0_20261028_041546_247139192/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T0_20261028_041559_914875719/run_report.json \
  --benchmark obstacle_avoidance --level 0 --task-id 0 --init-state 0 --seed 0 \
  --instruction "pick up the akita black bowl on the stove and place it on the plate" \
  --experience-object-id akita_black_bowl_1 --target-relation on --target-region plate_1 \
  --target-predicate on akita_black_bowl_1 plate_1 \
  --checkpoint /workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft \
  --experience-library /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2.json \
  --demonstration-horizon 1000 --maximum-physical-branches 3 \
  --maximum-rrt-preflights 3 --rrt-time-budget-s 12 \
  --transport-strategy native_region_progress_osc
```

attempt7静止等待没有改变抓持旋转失稳后，切换到已有的完整任务执行器。attempt8从attempt7的初始Snapshot构造带来源清单的独立Snapshot root；payload、BDDL、模型几何及MuJoCo Integration State哈希均从原始manifest核验。运行示例如下（Snapshot root provenance见同目录`source_provenance.json`）：

```bash
/workspace/zy_workspace/RLinf/.venv/bin/python -u research/gs_opd/repair/trajectory_generator_v3.py \
  --backend experience_task_success \
  --output-dir /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt8 \
  --snapshot-root /opt/data/private/gs_opd/overnight/gs_opd_trajectory_generator_v3_20261028_0255/l0t0_state0_attempt8_snapshot_root \
  --benchmark obstacle_avoidance --level 0 --task-id 0 --init-state 0 --seed 0 \
  --instruction "pick up the akita black bowl on the stove and place it on the plate" \
  --experience-object-id akita_black_bowl_1 --target-relation on --target-region plate_1 \
  --target-predicate on akita_black_bowl_1 plate_1 \
  --checkpoint /workspace/zy_workspace/checkpoints/gs_opd/pi05_rlinf_libero_sft \
  --experience-library /opt/data/private/gs_opd/experience/gs_opd_task_skill_experience_library_20261026_v2.json \
  --demonstration-horizon 1000 --maximum-physical-branches 3 \
  --maximum-rrt-preflights 3 --rrt-time-budget-s 12 \
  --transport-strategy native_region_progress_osc
```

## 结果文件

Student/B-Spline run包含`student_rollout.npz`、视频、子步接触记录和Runtime Snapshot；experience backend包含V3清单、`execution.log`及`task_success_engine/`下执行器和抓持搜索的机器报告、动作/接触记录与Snapshot。外层报告单独分类native Task Success；没有聚合到外层的监测或trace字段明确标为未聚合/未发出。native-success候选只写入本地`experience_update_candidate.json`，不会自动合并共享经验库。

## 验证与试验记录

RLinf venv下的`py_compile`和`git diff --check`通过，V3聚焦测试14项通过，CLI可在未设置仓库`PYTHONPATH`时直接显示帮助。新增覆盖包括experience executor退出与native success分类、前序failure report转发及身份导入，以及`unsafe`几何记录与子步监测覆盖分离。

首次预检在Bridge打印ready后失败，V3 manifest确认物理动作0。原因是原端口就绪轮询使用未认证的裸TCP `connect_ex`，被Bridge单连接认证循环当作无效客户端。代码改为用端口bind检查冲突、等待Bridge ready日志，并用authkey协议建立首次连接。

attempt2通过只读预检后执行291动作：Student 50步确认双指稳定抓持，当前状态左侧B-Spline/OSC再走241步。桥接场景identity和BDDL哈希匹配，520个编译碰撞Geom；动作Trace有效，291/291每步记录25/25物理子步。OSC在夹持相对旋转误差达到0.200040 rad时触发已存在的0.20 rad停机门限。Native reward=0、`done=false`、完整任务失败，未执行释放。安全未认证，Strict 10 mm false/未评估，训练资格false。V3聚合器曾将嵌套于`control_steps`中的完整子步数据误判为不完整，也曾把runner退出0当作任务完成；两处报告门已修正，attempt2报告重分类为`task_incomplete`。

attempt3使用实时物体姿态更新EEF目标，实际碗姿态相对起点转动3.012 rad，旋转门限0.213534 rad，较attempt2更早失败；该控制修改已撤回。attempt4右绕`trajectory-002`在总196动作、OSC146步时发现碗Geom106与木柜基座Geom454接触（第17/25子步，8个接触子步，最大力7.045 N），随即停止。attempt5上绕`trajectory-003`在OSC240步以0.200061 rad旋转误差停止，球心跟踪峰值9.729 mm且无非预期碰撞。attempt6左绕`trajectory-004`在OSC242步以0.200224 rad停止，球心跟踪峰值7.959 mm且无非预期碰撞。attempt5和6分别是290和292个真实动作，逐子步监测与Trace均完整；原生任务均未成功，稳定释放、操作安全、Strict 10 mm与安全训练资格均未通过。

attempt2/5/6在OSC第50、100、150、200步的抓持旋转误差几乎重合，证明仅切换当前Bundle路线没有改变最初的失效段。attempt6的Student抓持结束后，左右指接触记录分别有1和5个Geom Pair；trace测得碗线速度0.1046 m/s、角速度0.0755 rad/s、最大关节速度0.283 rad/s。同场景经验模板与当前物体到EEF变换相差23.13 mm平移和16.47°旋转。该模板已有双侧抓持与36.3 mm抬升验证，但来源`source_native_task_success=false`、`safety_certified=false`，只作为抓持先验，不能迁移安全标签。

attempt7在50步Student动作后加入20个原生OSC零位移/闭爪等待动作。黑碗线速度从0.104629降至0.002015 m/s、角速度从0.075531降至0.003916 m/s；等待段20/20完整监控、双指抓持保留且无非预期接触。重新查询几何后执行上绕`trajectory-003`，OSC第229步以0.200091 rad触发原有抓持旋转门限。总计299步，原生reward=0、done=false；299/299动作均完成25/25子步监测，trace有效。稳定释放、操作安全、Strict 10 mm及训练资格未通过/未认证。

attempt8结果：标准1000步horizon、真实物理分支上限3、RRT预检上限3且单次预算12秒。实际3个预检都报`planning_time_budget_exhausted`，随后现有experience高位接近fallback进入1个真实分支：预抓取路线102步、历史对象相对接近60步、一次local RRT超时后直接OSC recovery 32步。最终`migrated_object_relative_approach_action_budget_exhausted`；总194动作、4850子步，双指稳定抓持false、抬升0、原生Task Success=false，未释放。最终失败Snapshot SHA-256 `86fe67c17d25bfc8eb236401b547a53c945991989e2a3871f8774aa9c29652aa`，局部恢复Snapshot SHA-256 `9bba0392cfa54659681dc7f0649bc338e513cfa35148cc521878565bea548c56`。

监测复核发现attempt8旧`substep_monitoring_complete`布尔值把每个几何状态`unsafe`都视为监测缺失。原始行实际显示194/194动作各有25/25接触样本、25/25几何样本及有效7D action/state记录。已修正`task_success_first_grasp_search.py`的覆盖判定，允许几何样本`complete`或`unsafe`表示“已观测”；这只修复覆盖字段，不改变安全 verdict。新增的`attempt8_posthoc_monitoring_audit.json`保留源报告哈希并给出194/194完整覆盖。几何监测194/194为unsafe、Operational/Strict Safety均未认证，safe training=false。

**下一步：**不重复attempt8的高位候选路线。V3新增`--previous-search-report`，报告SHA写入run manifest，并传给既有执行器；grasp search会按任务/模型/BDDL身份核验failure memory，跳过精确失败的candidate-route pair，再尝试不同对象相对接近方向。attempt9继续用合法初态、标准1000步、3物理分支与3个12秒RRT预检；既有0.20 rad和Strict Safety规则、BDDL均不变。
## 2026-10-28 10:17：显式候选ID生成截断修复；L0T0基姿池耗尽

V3 strict screen复用attempt41真实稳定抓持和精确L0T0/state0 Snapshot，覆盖活动几何候选池48个唯一抓姿；0 Strict可行，失败类collision 23、grasp geometry 24、IK 1。物理成功基姿的接近轴roll和jaw-axis tilt各测试±5/10/15°，共12项，仍未消除非pad指部—目标接触/末端approach碰撞，0可行。总计60项scratch screen，物理动作0、RRT 0、live Snapshot未变。

最后8个显式ID曾被候选生成阶段拒绝：9个经验模板占用了48候选cap，未生成geometry ID46/47。`task_success_first_grasp_search.py`现在在显式ID模式完整生成48个几何候选后按ID选择；不指定ID时保留原经验优先cap。增加两项回归测试（2 passed），随后8项真实MuJoCo scratch screen全部运行并维持0 Strict候选。拒绝报告、修复后报告和逐组screen报告均保留，见machine progress JSON。

最近目标碰撞在已编译几何g29/g30/g32沿object local-y排列；当前仅追加4个由该实测布局推导的contact patch偏移（-2/-4/-6/-8 mm），不执行失败候选的RRT或物理分支。

## 2026-10-28 10:02：attempt41 V3原生成功；双归档动作审计完成
