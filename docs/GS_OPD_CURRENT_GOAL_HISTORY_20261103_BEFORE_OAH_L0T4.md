# SafeTRAIL Stage A 当前目标

GOAL_STATUS=ACTIVE。唯一验收为30个OA/OAH任务均拥有合法Episode出发、完整连续真实MuJoCo执行且原生BDDL `task_success=true`的轨迹。

## 已核实覆盖

任务矩阵与原始报告核实覆盖22/30（12 Standard、10 Extended）；剩余OA/L0T4、L1T3、L1T4、L2T4及OAH/L0T3、L0T4、L1T4、L2T4。证据索引`/opt/data/private/gs_opd/safetrail_stage_a_exploration/takeover_20261102_native_evidence_audit.json`。

## 当前局部目标：OA/L1T3

dev332–335已将Cheese带至篮口外，native local z≈0.0705。dev336在z仍位于开口外时沿local x正向下降约30.7mm至x=-0.149，16动作/400子步；末端双侧抓持连续16样本稳定，RRT净空82.5mm。Soup/cheese In仍false。

当前dev337从合法t6192续跑，再沿local x正向下降约30mm到x=-0.12，z≈0.07保持在开口外。继续逐段下降至篮口高度，随后尝试沿local z负向进入Basket；每段由fresh native region/实时抓持变换计算，观察真实接触和抓持。最终必须完整原生BDDL task_success。

## 已结案机制与审计限制

- dev257分段IK关节PD真实抓杯成功；dev258/259 OSC外撤存在高度下沉与水平停滞，停止原样路线。
- dev261线性关节参考把q3推到2.896753rad但末速仍+.395399rad/s；dev262固定腕姿四端点无有界IK；dev263/264首动作限位guard中止，发生未保存物理动作，终态Snapshot不能用于完整成功链。dev265smoothstep将近终点qvel降至.04146rad/s但最后仍guard，亦不进入成功链。dev266改变腕姿+端点限位余量后成功完成局部转运。
- OA/L2T4 t1007 dev246/247已完成：g5掌面接触阻止下探，分别上漂约25mm/仅下降约2.5mm，In/Close=false；暂停该表面滑移家族。不能把dev13/14旧计划当成未启动。
- OAH/L0T3 dev216–220已结案，低位/上侧推手无IK；OAH/L1T4 t1181双pad持杯源仍可恢复，适合后续短边界插入，详见矩阵。

代码：`research/gs_opd/repair/run_snapshot_ik_joint_push_chain.py`新增可选smoothstep时间参考与IK限位内缩，并修正native_goal_query读取predicate_value、执行前保存segment诊断、报告timestep与已存动作数量差。`mujoco_jointspace_planning.py`只读IK支持端点限位余量；默认行为保持原1e-7rad边界。py_compile与17项现有关节RRT/抓取screen单测通过；物理有效性以原始MuJoCo记录为准。

探索日志 `/opt/data/private/gs_opd/safetrail_stage_a_exploration/experiments.md`；经验库同目录EXPERIENCE.md。先前局部目标历史已完整归档至 `docs/GS_OPD_CURRENT_GOAL_HISTORY_20261102_TAKEOVER.md`。

最新可用源dev268 t2979：杯根[.039505,.469493,1.111183]m、EEF[.009170,.481996,1.219039]m、双pad，native In/Close=false；Snapshot与报告位于数据盘 obstacle_avoidance_human_L0T4_exterior_object_pivot_search_dev268/run/，SHA见最新progress。


## OAH/L0T3 dev310结果与下一步

dev310没有成功。24个活动几何候选预筛全部被IK或碰撞约束拒绝；唯一物理分支仍尝试geom76向下倾斜抓姿，执行57个真实动作后末端停滞，Soup只移动约2mm，没有双pad抓持或抬升。link6与tomato_sauce接触解释了该路线为何无法继续。保持原生In状态未破坏（Soup=false、Cheese=true），覆盖22/30。dev311首个命令因冷启动禁止非零candidate-start-index在参数校验阶段退出（未启动MuJoCo，0动作）；已改为精确指定由当前活动几何重新生成的geom87候选ID，限制一个RRT预检和一个物理分支；仍以原生BDDL task_success为唯一成功判定，来源t2372，归类Extended。dev310完整审计见progress和数据盘报告。


## OAH/L0T3 dev311结论与dev312计划

dev311精确指定当前活动几何geom87后仍复现dev310同一EEF停滞位置与47.275mm误差（57动作、1425完整子步）；Soup仍不在篮内，未抓持。冷启动不接受非零candidate-start-index的首次命令在环境启动前退出，后改成受支持的活动几何candidate_id并完成真实分支。停止当前抓姿ID搜索；dev312切为显式物理接触推送路线，先沿法线向下脱离hairbrush，再经远离tomato_sauce的南侧高位绕行，转换掌面姿态后向篮方向分块推。


## dev312结果与dev313路由调整

dev312的真实掌面推送尚未接触Soup：80动作/2000完整子步；向下逃离初始hairbrush附近位姿后，负Y退让使robot0_link6碰tomato_sauce并停滞，末端误差84.2mm。Soup/cheese的原生In仍为false/true。dev313把退让段改为+X/+Y，在Soup北侧保持间距并绕开tomato_sauce，再高位转腕进场；若第二种绕行仍使link6受阻，切换优先任务，不重复参数微调。


## dev313结果与dev314路点

dev313从dev312终态t2452尝试正X横移，54动作/1350完整子步后EEF只移动36mm，hairbrush g5接触持续，Soup不动。此终态没有清掉hairbrush。dev314将从合法t2372源先沿向下法线脱离，再保持低位向+X/+Y移动，避开tomato_sauce和Soup，成功后升高转腕绕行。


## dev314障碍定位与dev315止损

dev314确认t2372向下脱离阶段有效（末4步无robot contact），但低位正X/+Y绕行在EEF约[.1237,.0838,.1205]m处撞上link6/hairbrush并停滞；与dev312负Y方向的tomato_sauce接触不同。dev315仅测试一次西侧低位绕行（+X不超过起点，向-Y移至[.08,-.08,.096]），脱离两障碍后再高位抬升；若失败则暂停此task路线并换任务。


## dev315–316改用关节空间RRT脱障

基于dev312/314实测link6分别接触tomato_sauce/hairbrush，停止手工末端直线绕障猜点。dev315只执行已验证的下探脱离并保存真实Snapshot；dev316从无接触状态用现有关节空间RRT‑Connect（最小净空0m、无penetration allowance）规划并真实执行西南高位绕行，再去Soup后方。失败则离开此task路线。


## dev316失败点与dev317候选端点筛查

dev316从t2400无接触Snapshot做关节RRT，首个西南高位endpoint IK无解，0物理动作。dev317不重复该点；只读预检当前EEF上方及小幅横移的多个高位endpoint，RRT同时检查全场景碰撞，只有第一个有解且有无碰路线的候选才会真实执行。


## dev317端点碰撞定位与dev318腕姿筛查

dev317从t2400无接触源筛查四个高位位置，0动作。一个目标无IK，其余全部由gripper hand与hairbrush g10全几何碰撞拒绝（约-18.6至-55.5mm）。dev318固定近邻竖直高位，以多种腕部旋转姿态做只读IK+零净空RRT筛查，仅在完整路线可行时执行一个端点；若全拒绝就切任务。
