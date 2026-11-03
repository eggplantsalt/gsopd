# SafeTRAIL Stage A 当前目标：完成剩余8项原生BDDL成功轨迹

## OAH/L0T4 dev263/264失败与dev265减速改进
dev263/264从dev261 t2997分别以0/0.001rad限位检查容差执行腕姿改变；首个物理动作触发MuJoCo关节限位guard，报告0已保存动作但native timestep已到2998。明确存在未完整保存的物理动作，两支Snapshot不进入可审计成功链，保留原档。dev261终态实测q3=2.896753rad、qvel3=0.395399rad/s，距上限不足0.00055rad，线性关节参考在近限位端点未减速是具体原因。停止扩大容差，改从dev259 t2945完整源重做该转运，dev265使用三次smoothstep时间参考，维持每关节≤0.02rad采样边界且端点参考速度趋零；保持MuJoCo原有物理限制。采样界离线算例与py_compile通过，物理结果待报告。IK chain另修正native query键predicate_value，并在执行前保存segment诊断，以便失败报告包含已选IK/真实执行进度。覆盖仍22/30。

## OAH/L0T4 dev260–263 实验续记
dev260同高腕姿外移端点无有界IK，0动作。dev261降低40mm端点可达，52动作至t2997，杯根[.021511,.366026,1.159965]m，末端[-.008885,.423195,1.252409]m，双pad保持，原生In/Close=false。全部52动作接触子步完整；最大关节L2跟踪误差0.021773rad，微波炉实时接触统计{}。合法终态payload da5d25ac3a28c8de3f458f9e87c06c94b7dc51e0d6f1fa7dfd8511837e6b9275。dev262在同一精确源搜索向外8/25mm并保持/降低0–30mm的四端点，全无有界IK，0物理动作、状态哈希未变。dev263改变腕姿（世界XYZ ±15°、yaw ±30°）做只读端点筛选，首个可达解真实关节PD执行；不继续相同腕姿增加预算。原始报告/Snapshot位于safetrail_stage_a_20261102对应dev260–263目录。覆盖22/30。
## 2026-11-02 最新接管与OAH/L0T4 dev257–261

核实无运行中的旧仿真。矩阵22/30（12 Standard、10 Extended），22项均已找到包含原生task_success=true的原始报告；证据索引 `/opt/data/private/gs_opd/safetrail_stage_a_exploration/takeover_20261102_native_evidence_audit.json`。OA/L2T4 t1007 dev246/247已结案，g5掌面碰撞阻挡下探，不再复跑dev13/14历史计划。

OAH/L0T4 dev257已完成：分段endpoint IK+稠密关节PD真实80动作接近，随后双pad抓持并抬杯成功；后续OSC清空抬升190动作未达目标，终点t2780。dev258从该合法持杯Snapshot横移267动作至t3047，杯根降至z=1.10946m，水平路点残差42.1mm且停滞，In/Close=false。dev259同源、指定1.20m转运高度，165动作至t2945，杯根[.011254,.345943,1.200848]m，双pad保持；第二横移路点128动作未到。暂停原样OSC外撤。

dev260从dev259合法t2945持杯Snapshot测试保持实测腕姿/高度外移10mm X、20mm Y，endpoint IK无有界解，0物理动作，live state unchanged。dev261同源保持腕姿，read-only筛选降低20–120mm的短外移端点，首个可达者用关节PD真实执行；不增加相同OSC预算。该策略变化针对实测IK可达性与OSC高度漂移。仅原生完整task_success=true计成功，当前仍22/30。

原始报告和Snapshot：数据盘 `overnight/safetrail_stage_a_20261102/obstacle_avoidance_human_L0T4_{staged_ik_joint_route_dev257,low_clearance_aperture_dev258,fixed_staging_height_dev259,joint_height_hold_outward_dev260,joint_outward_reachable_height_dev261}/`。dev259源payload SHA256 `a20c7b4d6e5e71a7fbc264c670bb274afad8a2849c18c64d30cc4b23c5f7edbb`，integration SHA256 `5a4add66458fc1cca19eb53b4adacd1db91692c136a07999f41c7be6079331c3`；同Episode谱系由各报告parent/continuation字段记录。dev261结果待报告。


最新原生成功覆盖 **22/30**（12 Standard、10 Extended），剩余8项：OA/L0T4、L1T3、L1T4、L2T4；OAH/L0T3、L0T4、L1T4、L2T4。22条覆盖由任务矩阵与原生成功记录核实；OA/L2T3在t3075、OAH/L1T3在t5590为Extended，其余成功项见矩阵审计字段。唯一验收为合法t0来源、连续真实MuJoCo执行与原生`task_success=true`；1000动作内归Standard，超出归Extended。

**当前局部目标：OAH/L0T4。** 杯子由`microwave_1_g6`支撑，原生`In=false, Close=false`。合法t0连续Episode的dev251终端t2441 payload SHA256为`06504602bd70f9253a241e0ffe59a7823565672ec195ec5d39a04838cdaa0982`。dev255首支的开爪后t2457 Snapshot payload SHA256为`9055fed241de8e0cb74f88ec781415e4f3f9297f628fefa6e317122523141dd8`。dev256从该合法Snapshot精确恢复、native状态核验通过但0物理动作；单候选关节RRT在20.36秒和715次配置检查后只扩展1次即超时，预抓位IK可达。dev257将跳过无需的开夹动作（已测pad gap约86mm，候选宽约8mm），在当前姿态、抬高、目标上方旋转和预抓位等阶段分别求IK，再用稠密关节PD实际执行；现场MuJoCo接触与抓持状态决定是否继续。若形成稳定抓持，再沿水平微波开口送入并操作门达到原生Close。

OA/L2T4 t1007的dev246/247已验证：g5掌面接触阻止surface-slide下探，杯子只降低约2.5mm，暂停该接触家族。OAH/L0T3 dev216–220的低位/上侧pusher候选已暂停。所有覆盖仍按原生完整`task_success=true`确认；详细物理证据见数据盘探索日志和各dev归档。

## 历史记录



## 历史快照：OAH/L0T3 dev102（2026-11-01，已由dev130/dev131更新）

`GOAL_STATUS=ACTIVE`。目标为 obstacle_avoidance 与 obstacle_avoidance_human 共30项，每项至少一条真实MuJoCo连续轨迹通过原生BDDL。当前20/30（12 Standard、8 Extended），剩余10项。

**当前局部目标：OAH/L0T3完成Soup放篮。** dev99/100从t2082接触Snapshot闭爪均未形成稳定抓持；双pad只在t2088短暂接触，后续执行器运动使接触丢失。dev101侧向jaw0候选两端IK不可达、物理动作0。dev102侧向jaw1候选已通过真实MuJoCo双pad稳定抓持监控16态并抬升Soup 22.5mm，现进入同一连续Episode的原生篮内运输；全任务仍未成功。t2109仅观测到右指垫接触，Soup角速度约5.58rad/s，需立即跟踪是否滑脱。

dev102源为合法静置t1842 Snapshot（payload SHA256 `421a8353ddf54bb6eb825bb285a27fb3f0f524f6479aa136470cc0e1b07afb60`，integration SHA256 `a7c249ec0ddb34083ebd5d58f2a2ed9f504813ad92b0cd4e78a0cad456656a37`），候选`geom89_jaw1_approach0_+1_flip-1`。当前原生放置输出在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L0T3_geom89_jaw1_sideapproach_dev102/task_success_engine/task_success_first_executor_obstacle_avoidance_human_L0T3_20261101_213914_316425218/`。必须持续至原生BDDL判定成功或真实分支结案；覆盖仍20/30。

全局矩阵为`GS_OPD_SAFE_TRAIL_TASK_MATRIX.csv`；实验动作、原生状态及失败证据在`/opt/data/private/gs_opd/safetrail_stage_a_exploration/experiments.md`。

---

## 之前进度快照（供恢复历史上下文；当前状态以本文首段为准）

`GOAL_STATUS=ACTIVE`；真实原生成功覆盖19/30（12 Standard、7 Extended）；dev23/24已结案，dev25正在物理执行。

**当前目标：OA/L1T4。** dev24从合法t826持杯Snapshot执行185步至t1011。按实测g8掌面接触点和目标旋转计算94.2mm内扫，加入延伸和余量后实际撤离133.5mm；转腕、入口接近及下降段未触g8。t987开始entry alignment后掌面重新碰上g8，24步停滞；末端root=[.08118,.09762,.94527]m，目标误差.26049m，角误差.00769rad，双pad仍持物，原生In/Close/task_success=false。

dev25从精确t1011失败Runtime Snapshot续跑，保留合法t0 Episode链，请求相对当前live姿态再+45°并沿用接触旋转扫掠外撤，目标检验约+90°累计姿态是否能改善alignment手部净空。由于源timestep已超过1000，此分支若成功记为Extended。脚本：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L1T4_t1011_yaw90_aperture_dev25/launch.sh`；父Snapshot SHA256 `d2f35dc8eba399771a8ea2bea491743590551467f0cd2b872c087b8fb780a0e2`。

dev23、dev24报告与物理记录路径见数据盘`/opt/data/private/gs_opd/safetrail_stage_a_exploration/experiments.md`。通用扫掠外撤代码位于`research/gs_opd/repair/task_success_first_executor.py`，SHA256 `1d8f9116c758cb4c22a115fe5eb70823938d589dbe4e0b11d08e4a9b81c1ab16`。当前覆盖不变；下一次计数仅在完整原生BDDL task_success=true后增加。

---

## 之前进度快照（供恢复历史上下文；当前状态以本文首段为准）

`GOAL_STATUS=ACTIVE`；完整原生成功覆盖19/30（12 Standard、7 Extended）；dev22已结案，dev23进行中。

**当前局部目标：OA/L1T4**。dev22从合法t689持杯Snapshot执行137步至t826。入口root实际降到`.953m`，比dev21低约10mm；t803掌面仍接触`microwave_1_g8`于`[.182,.229,1.096]m`，接触法线g8→hand约`[+.055,−.998,0]`，随后停滞24步。mug双指保持，q7=2.405rad，In/Close=false。dev23从这个精确合法t826接触Snapshot续跑，将yaw目标从+30°改为+45°；利用已测robot/g8接触生成真实重定向和法线退离航点，再重试低入口。Horizon1800，按实际episode timestep区分Standard/Extended；原生BDDL与完整task_success仍是唯一成功判据。

dev23脚本：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L1T4_t826_contact_escape_yaw45_dev23/launch.sh`；源Snapshot SHA256 `64d11725bfef370be38b9f3049354a867b2c831a9a47fbf84a2f0a5d1e3c9eb2`。OA/L1T4 t689原始持杯源仍见dev17，SHA256 `261925b127fbb9d72f9692be937af43abf93da29cab0babe9a0328fc38fb4de5`。OAH/L0T3 dev66/67两次抓持失败已暂停同一抓姿族；A* adaptive预算仍无物理航点验证。



`GOAL_STATUS=ACTIVE`；Stage A仍19/30。OAH/L1T3 dev84已证明短surface-slide下降(contact gate)、实测高度重基准都实际执行：t2799取得Soup–floor向上接触，transit root z重定为`.03842m`；4mm横向动作前两步有支撑接触，随后root z逐渐升至`.0415m`，t2816滚转/丢失双pad，累计横移约13mm，native Soup In=false、cheese In=true。4mm控制步长只降低位移、没延长稳定时间。

当前dev85在实测支撑高度基础上增加1mm向下setpoint偏置，抵消已测到的水平移动抬升漂移，并恢复8mm动作步长以提高进度；下降仍要求真实向上支撑接触，偏置只在接触后启用，所有运动由MuJoCo实际执行。脚本：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_surface_contact_bias_dev85/launch.sh`；代码SHA256 `65ca5e35b86d61f743c82183dc08323992136a71d09bcb25eec09707a8840490`。


`GOAL_STATUS=ACTIVE`；Stage A真实覆盖19/30（12 Standard、7 Extended）。最新dev83从dev82 t2798真实Snapshot运行18动作，Soup只在首个横移动作短暂接触MuJoCo `floor`，随后因目标root z低于支撑实际高度约2mm逐步抬离；沿basket方向前进约27mm后突发约13°滚转并失去双指。原生Soup In仍false，cheese In=true。分析发现surface_slide在2.6mm下降差被通用12mm门限跳过，因此contact-gated下降没有机会执行。

当前dev84加入1mm surface-slide专用下降生成门限并保留短下降waypoint；支撑接触仍是完成条件，接触后用实测root z；水平route改成XY进度并将末端单步限幅降至4mm，以降低摩擦拖动引起的失抓。它从dev82精确Snapshot分支，结果待物理执行和原生判定。脚本：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_surface_slide_contact_aware_slow_progress_dev84/launch.sh`。


`GOAL_STATUS=ACTIVE`。最终验收仍是30/30个任务拥有连续、可审计且经原生BDDL判真的成功Episode。当前覆盖19/30（12 Standard、7 Extended）。OAH/L1T3 dev81通过真实MuJoCo下降建立支撑接触，但2mm root误差过严，继续压入几何支持高度后失指；dev82虽把容差调至5mm，却在t2798、物体距名义root目标3.4mm且还未接触支撑时提前结束。两轮均未进行横向滑移，Soup In=false、cream-cheese In=true。

当前dev83从dev82的精确t2798 Snapshot续跑。新增通用控制器完成回调：该下降段以水平XY误差判定目标附近，但必须有target-body与环境的向上物理接触记录，才能结束；随后才以实测root z作为surface-slide高度。这样不再逼root穿过支撑面，也不在空中提前开滑。启动脚本：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_surface_slide_support_contact_gate_dev83/launch.sh`。coverage未变。


`GOAL_STATUS=ACTIVE`。唯一验收仍为30/30个OA/OAH任务具备合法t0来源、连续可审计MuJoCo动作及原生BDDL `task_success=true`。最新核实覆盖19/30（12 Standard、7 Extended）。OAH/L1T3 dev81从dev76真实闭爪Snapshot执行26动作/650子步：Soup确实降至支持面，`floor`向上法向接触从t2799持续出现，root支撑高度约`.0384m`；名义下降目标`.036381m`差约2.8mm，2mm跟踪门限造成继续下压，最终t2813失去双指抓持，Soup In=false、cheese In=true。下降误差与接触已定位，但横向滑移尚未真正测试。

当前实验dev82收紧后的处理是5mm下降容差，并在确认目标物体与环境有向上支撑接触后，将后续surface-slide高度基准更新到实测root z；避免把碰撞几何不可达到的root目标继续当作压入命令。它从同一dev76合法held Snapshot运行，最终仍以完整Episode和原生BDDL判定成功。启动脚本：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_surface_slide_support_rebase_dev82/launch.sh`；覆盖数不变。


`GOAL_STATUS=ACTIVE`。唯一验收仍是30/30任务拥有合法t0出发、连续可审计MuJoCo动作、原生BDDL `task_success=true` 的完整轨迹。当前真实成功覆盖19/30（12 Standard、7 Extended）。OAH/L1T3 dev76重力对齐local-Y抓取中心真实抬升23.354mm，但按原z=0.192m清空路线升至root z≈0.121m后最后一步失去左pad；无外物碰撞，native Soup In=false、cream-cheese In=true。

dev81已启动：dev80的surface-slide route雖计划降至root z=.0364m，但`reachability_descent`被40mm容差以0动作跳过，汤罐未触桌即开始横移并失抓；这不是对支撑滑移本身的有效检验。现在下降waypoint要求物体root在2mm内真实到位后再横向移动，仍从同一dev76 held Snapshot恢复。dev81脚本在数据盘`launch.sh`。 dev76物理记录：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_gravity_aligned_contact_center_dev76/task_success_first_executor_obstacle_avoidance_human_L1T3_20261101_094435_713591643/expert_native_placement/placement_branch_00_region_layout_88_direct_region_progress/physical_step_records.jsonl.gz`；lift源Snapshot位于其expert_grasp_search child archive。

---

## 2026-11-01 当前局部目标：沿世界重力方向下移Soup接触中心

全局Stage A仍19/30（12 Standard、7 Extended）。dev72–74显示原geom80/jaw1抓姿能真实lift但首次承载滚转并丢pad；沉降/单步制动未修复。dev75因把物体局部Z误当竖直，实际只测试了侧向接触偏移：MuJoCo中的Soup root local-Y近世界+Z，local-Z近世界+Y。侧向偏移让闭合先左指触碰、汤罐横向推移约15mm、t2775短暂双pad，lift首步即失去左pad，最大lift低于0，原生Soup In=false。

dev76从同一合法t2712 Snapshot保留geom80/jaw1/approach2候选、只把pad center沿物体局部Y下移10mm（约世界−Z），使用真实碰撞与原生BDDL审计更低侧壁抓取。若仍单指推移或lift失败，基于接触力与floor净空改变抓轴/候选族或转向其他任务，不在错误的物体轴上重复试验。完整成功必须仍为原生Soup In、cream cheese In与`task_success=true`。

dev76输出根：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_gravity_aligned_contact_center_dev76/`。源Snapshot为dev71 t2712合法同Episode状态；只变`--candidate-offset-object-y-mm -10`，目标模型/BDDL/t0沿用冷启动身份。dev75的动作记录、报告和Snapshot位于数据盘`obstacle_avoidance_human_L1T3_lower_contact_center_dev75/`。

---

## 2026-11-01 当前局部目标：OAH/L1T3以IK筛选旋腕高度续跑

总目标仍是Stage A 30/30原生BDDL完整成功Episode；当前经证据审计覆盖19/30（12 Standard、7 Extended）。最新分支dev70已结束且失败：高位A*通过到近目标区域，但0.46085m旋腕姿态没有有界当前场景IK解，125动作/3,125子步内无目标接触，Soup In=false，cheese In=true。

dev71从dev69合法t2526 Runtime Snapshot分支，静态只读预检选择最高IK可达旋腕层，避免再次要求不可达的最高位旋腕；接着走真实A*/MuJoCo接触监控/抓持lift/原生BDDL。若此候选仍失败，则根据首个真实接触和位置误差暂停geom80/jaw1候选族，转到其他未完成任务或新的抓姿方法；不增加同路径预算，不将部分动作计作成功。

恢复起点：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_axislock_blended_feedback_dev69/task_success_first_executor_obstacle_avoidance_human_L1T3_20261101_091400_886246454/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_human_L1T3_20261101_091412_695877317/geom80_jaw1_approach2_+1_flip+1__approach_axis2_-22.6mm__rrtexplicit_geometry_high_approach_physical_validation_local_replan_1_runtime_snapshot.pkl`；源Episode t0及模型/BDDL身份仍以原cold-start目录清单核验。代码：`research/gs_opd/repair/task_success_first_grasp_search.py`，SHA256 `d8bae8223369ef6ad715b2f13ab1544539832ae54f363a8e92d4de9e7a8bf637`。dev71输出：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_rotation_height_ik_dev71/`。

---

## 2026-11-01 OAH/L1T3 dev70：从dev69高位失败Snapshot用全XYZ路点闭环续跑（已启动）

覆盖仍为19/30。dev69采用每步90%主轴+40%锁轴反馈后执行117动作/2,925子步，EEF升至约0.46m clearance；但实测XY最大偏离A*目标约28–41mm，达不到原预期的受限轴向路点。没有robot-scene/Soup接触，Soup不动，native In=false。该策略比dev68保留了Z进度但40%锁轴预算仍不足，停止继续从t2421重复抬升。

dev70从dev69合法连续分支t2526（高位EEF约`[0.2111,-0.0848,0.4577]m`）精确Runtime Snapshot恢复，Soup位置与原native谓词不变，跳过此前已完成的首次抬升；重算活动场景A*路线至同一geom80顶向候选预抓。代码现在对轴向waypoint直接闭环完整XYZ目标误差，合成位移单步限幅仍为20mm；waypoint序列本身继续由A*轴对齐路线提供，真实接触监视器保持启用。若从此高位姿仍无法走到candidate，不再延长相同姿态的控制尝试，将转向新的jaw/approach几何或任务。

源Snapshot：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_axislock_blended_feedback_dev69/task_success_first_executor_obstacle_avoidance_human_L1T3_20261101_091400_886246454/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_human_L1T3_20261101_091412_695877317/geom80_jaw1_approach2_+1_flip+1__approach_axis2_-22.6mm__rrtexplicit_geometry_high_approach_physical_validation_local_replan_1_runtime_snapshot.pkl`，属dev69报告中physical replan开始前的实际MuJoCo状态；continuation root是dev69完整run目录、t0为原始合法冷启动root。新代码SHA256 `714afe67f8db6e0a2bfb16ad23a8af2427957fa10c560a7acf1402737e218421`；dev70输出根`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_full_xyz_feedback_dev70/`。

---

## 2026-11-01 OAH/L1T3 dev69：并行锁轴反馈后的同候选物理验证（已启动）

覆盖仍19/30（12 Standard、7 Extended）。dev68验证了dev67控制问题的位置：硬优先锁轴回正(>1mm)使首次垂直清障只升约9mm便停滞，42动作/1,050子步；Soup root不动，无机器人场景接触、无Soup pad接触，原生In=false。EEF的1–2mm横向误差令恢复逻辑反复优先修锁轴，垂直进度被饿死；这次属于控制修复副作用，不是抓取物理失败。

当前通用控制器改为**并行、限幅反馈**：锁轴始终按实测误差闭环；每步允许轴的进度预算为单步上限的90%，锁轴纠偏预算40%，合成位移仍不超过原步长上限。所有轴仍参与到达/停滞判定。dev69使用同一合法t2421 Snapshot和geom80顶向−22.6mm/A*候选，只变控制策略，检验先垂直抬升和旋腕时能否兼顾路径进度与侧向误差。

代码当前SHA256 `f75f4b6a865c0e7ca514ecdf64a7ea0d11f7518c0b94b43abe00a427f3fb1db7`；dev68证据：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_axislock_feedback_dev68/task_success_first_executor_obstacle_avoidance_human_L1T3_20261101_091157_143481589/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_human_L1T3_20261101_091208_884952363/`。dev69输出`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_axislock_blended_feedback_dev69/`。成功或失败均由接触、EEF姿态/位置、物体root与原生BDDL决定，覆盖未增加。

---

## 2026-11-01 OAH/L1T3 dev68：修复轴向路点锁轴漂移后重验顶向候选（已启动）

覆盖仍**19/30（12 Standard、7 Extended）**。dev67单候选在活动几何静态筛选中预抓IK、开爪抓取IK均通过，预抓最小场景距离+46.9mm；但247动作/6,175子步无任何robot-scene接触、Soup无位移或目标接触。高位A*路线可用；旋腕后EEF离开锁定路点约38mm X，且清障高度从≈0.44m漂落到≈0.32m。MuJoCo腕姿最终接近目标（约2°残差），但二次XY纠偏时局部IK没有有界解，RRT耗时上限后无法恢复；原生Soup In=false、Cheese In=true。

**通用修复：** [`task_success_first_grasp_search.py`](/workspace/zy_workspace/RLinf-gs-opd-overfit-recovery/research/gs_opd/repair/task_success_first_grasp_search.py) 的`translation_axes`原先只对允许轴发命令，对其它锁定轴既不计入收敛/停滞也不反馈纠正，故旋腕/轴向A*时误差累积。现将全位置误差用于判定，并在锁定轴偏差>1mm时优先回归该轴，再推进允许轴。dev68从完全相同合法t2421 Runtime Snapshot、相同geom80顶向−22.6mm候选和A*方案做物理对照；仅闭环算法变化。改动SHA256 `ee83c128649faf06836b081fc667204351bc47b1651086cec52211bf0b827377`；不运行单测，成功与否由完整MuJoCo记录/原生BDDL报告决定。

来源dev67完整报告与物理证据：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_topdown_depthminus22p6_dev67/task_success_first_executor_obstacle_avoidance_human_L1T3_20261101_090726_122975449/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_human_L1T3_20261101_090737_771344783/`。dev68输出根`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_axislock_feedback_dev68/`。如复验仍失败，检查单轴恢复是否真正把position error压下，并停止堆叠相同大角度旋腕抓姿，优先生成与当前腕姿更近且不触桌的姿态或换有希望的任务。

---

## 2026-11-01 OAH/L1T3 dev67：测试geom80上半部顶向抓取（已启动）

当前真实覆盖仍为**19/30（12 Standard、7 Extended）**。无旧实验进程运行。dev66从同一任务合法t2421 Snapshot执行279动作/6,975子步；A*高位横移及姿态/XY纠偏到达，但低位侧夹下降阶段EEF停在z≈0.100m，连续与`floor|gripper0_hand_collision`接触，Soup没有任何pad接触、root未移动，原生`In(soup,basket)=false`，已有cream-cheese In保持true。静态预抓筛选曾提示unmapped clearance −35.1mm；物理证据确认这是桌面/手掌净空问题，不是Soup抓持失败。证据归档见dev66 child report、Snapshot、逐动作/子步记录。

**dev67假设与分支：** 将落点从geom81的低位侧夹改到Soup另一侧编译碰撞几何geom80的顶向jaw1/approach2/+1/flip+1，并由活动场景确定性再生成approach-axis2 −22.6mm深度变体。该变体腕姿差91.7°、预抓点距当前EEF约31.9mm，接触中心world z≈0.0566m；优先观察是否能避免hand/table接触并形成双pad后lift。保持0.45m障碍感知A*高位approach、单分支440动作、原生Soup In目标；t2421起的Episode若完成仍按Extended统计。输出根`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_geom80_topdown_depthminus22p6_dev67/`。仅MuJoCo动作、逐子步接触及原生BDDL完整task_success可增加覆盖。

---

## 2026-11-01 当前实验：OAH/L1T3腕姿匹配抓取分支dev66

原生完整覆盖**19/30（12 Standard、7 Extended）**。OA/L1T2已由stage_07 t1183原生成功报告确认为Extended并计入覆盖。本轮dev64暂停了OAH/L0T3的jaw0接触家族。OAH/L1T3 dev65从合法t2421 Snapshot完成300动作高位接近：A*平移无碰撞并达到旋腕前预抓XY误差≈3mm，但候选jaw1姿态与真实腕姿相差167.6°；旋腕后EEF横向偏离≈57mm，目标仍差6.7°时耗尽路线预算，局部RRT超时无安全路径；Soup无接触、任务未成功。

**dev66活动假设：** 高位路径失败核心是Snapshot当前腕姿与jaw1抓姿不匹配。根据当前真姿态和Soup编译碰撞几何离线枚举504个轴向组合，选择新的geom81 jaw2/approach1−1/flip−1候选，计算姿态差28.6°、估计夹宽75.2mm小于87mm开口上限。以上仅用于挑选实验候选，不代表真实抓持已验证。dev66从同一t2421 Runtime Snapshot重新开始，用0.45m A*障碍高位路线验证双指接触与lift；物理成功后再接入8mm obstacle-aware XY region progress完成Soup In，最后要求两个原生In同时true。因累计episode从t2421开始，成功标Extended。

dev65完整动作/接触/A*诊断与失败Snapshot：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_t2421_geom81_jaw1_obstacle_astar_dev65/task_success_first_executor_obstacle_avoidance_human_L1T3_20261101_083413_026497209/`。dev66输出根见数据盘实验日志。
---

## 2026-11-01 接管更新：OAH/L0T3从t441持物续跑转向新抓取家族

Stage A实际覆盖仍为19/30（12 Standard、7 Extended）；本次无新增原生任务成功。所有实验进程均已结束。

OAH/L0T3 dev52/53从dev38同一合法t441双侧持物Runtime Snapshot改用full-geometry关节空间RRT/OSC：dev52目标yaw+15°时Soup真实向篮子移动约0.14m，t514在末段腕转后脱抓；dev53固定相对当前对象yaw=0°仍在t513、相近位置脱抓。两支都没有碰撞到hairbrush/目标障碍，双侧pad力在失抓前约17.5–18N；故停止对同一t441握持几何继续做yaw/路线微调。t514汤罐处于明显下落/转动态，不作慢速重抓起点。

dev56已从原始合法t0冷启动并按当前几何重生成geom81 top-down候选；0.45m高位A*预抓产生真实双pad抓持和30mm lift。之后默认native direct route只完成128动作/3200子步的清障抬升，Soup root到z=.3163m，仍双侧持物且无障碍接触；在目标z=.3883m前因单航点128动作cap停滞，A*横向运输未请求。下一步从精确dev56 t361 held Snapshot恢复，不重放抓取，显式设`obstacle_aware_xy_region_progress`并相对live yaw 0°；完成剩余72mm实测抬升后真实执行围绕hairbrush的A*路线。只有Soup与cheese两个原生In都true才算完整任务；标准horizon≤1000，超出单独记Extended。

关键来源：dev38合法t441运输起始Snapshot、dev52/53报告和dev56精确t361持物失败Snapshot在数据盘`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/`；dev39候选物理记录/失败分支Snapshot在`obstacle_avoidance_human_L0T3_fresh_geom_pool16_astar_dev39/`。探索日志与经验库为`/opt/data/private/gs_opd/safetrail_stage_a_exploration/experiments.md`、`EXPERIENCE.md`。整体覆盖、路线与审计规则见下方历史记录及`GS_OPD_ROADMAP.md`。

## 历史局部目标：OA/L0T4闭爪Snapshot原生运输

Stage A当前覆盖19/30（12 Standard、7 Extended）。dev35 geom105/+25mm/flip−1在t1752形成稳定双pad，标准lift最多升9.383mm后滑脱；dev36证明2mm低速更差。dev37暴露精确预抓Snapshot被冗余高位路由耗尽预算，已在代码中增加严格的near-pregrasp旁路；dev38从t1729经14步物体相对surface approach、10步闭合、20步标准lift，+22.5mm形成稳定抓持，最高抬升10.933mm，随后finger2双接触丢失，In/Close仍false。

下一步从dev35真实稳定闭爪t1752 Runtime Snapshot直接启动原生任务执行器的transport/placement，不再要求先通过人工lift阈值。逐步检查grasp monitor和杯体root；若搬运中滑脱，则记录首次失联对应路线、机械碰撞与原生谓词，再决定是更换持物姿态还是靠环境支撑完成接触器任务。Extended continuation需明确标注。

## 2026-11-01 OA/L0T4 dev35：短时双pad抓持成立，但首段提升滑脱


## 2026-11-01 OA/L0T4 dev34：+15° approach-roll在旋腕/末端XY纠偏时触发关节限位

- **假设/方法：** 单独执行dev33未运行的geom105/jaw1/approach0/flip−1、Z+25mm、approach-roll +15°；核对exact candidate后继续high A*物理路径。
- **真实结果：** 一支440动作/11,000子步，完成高位XY绕行，但在clearance旋腕和终点XY alignment阶段未收敛。局部RRT报告`current_real_qpos_is_outside_controller_joint_limits`，直接OSC续接最终耗尽120动作；全过程无robot-scene接触、无杯接触、无pad抓持。原生目标false。
- **原因与下一步：** roll+15°候选的末端姿态/关节姿态不在当前路径控制器可达范围内；避免重复该角度及其限位续跑。转向仍可达的geom105/jaw1/approach0/零roll/+25mm，只将jaw flip从+1改为−1，直接比较实际稳定抓持与升杯。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_z25_flipminus_rollplus15_dev34/task_success_first_executor_obstacle_avoidance_L0T4_20261101_071242_140245956/run_report.json`；child候选路由诊断、NPZ、完整接触日志与t1871可恢复失败Snapshot位于`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_z25_flipminus_rollplus15_dev34/task_success_first_executor_obstacle_avoidance_L0T4_20261101_071242_140245956/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_071253_206088789`。

## 2026-11-01 当前局部目标：单独验证geom105+25mm/flip−1/roll+15°


## 2026-11-01 OA/L0T4 dev33：候选上限只执行−15° roll，单侧pad交替接触

- **假设/方法：** geom105/jaw1/approach0/flip−1，object-local Z +25mm，同时请求approach-roll ±15°两态；从合法t1431恢复并保留高位A*。
- **配置偏差与真实结果：** 命令把`--maximum-candidates`设为1，variant生成后候选上限仅保留第一项`roll −15°`，实际不是计划的双角比较。该branch执行322动作至t1753；左/右finger接触交替出现，未形成稳定bilateral，`bilateral_grasp_not_stable_within_close_action_budget`，不进入lift；In/Close=false。
- **下一步：** 单独pin同一flip−1/+25mm候选的`roll +15°`做真实物理验证；后续若要批量roll网格，把`maximum-candidates`至少设为变体数，避免候选截断。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_z25_flipminus_rollpm15_dev33/task_success_first_executor_obstacle_avoidance_L0T4_20261101_071036_896278500/run_report.json`；child report中的search_budget明确记录roll=[−15,+15]但实际`candidates`只含−15°；物理记录与NPZ同目录。

## 2026-11-01 当前局部目标：对geom105+25mm侧夹加roll±15°


## 2026-11-01 OA/L0T4 dev32：+20mm flip−1短暂confirmed，仍无法稳定

- **假设/方法：** 在dev31 +15mm失败和dev30 +25mm边缘滑脱之间测+20mm；geom105/jaw1/approach0，flip±1两个exact-t1431恢复分支，保留1.35m obstacle-aware A*。
- **真实结果：** flip−1 330动作，t1751仅出现一次`grasp_confirmed=true`并有3个非连续双pad同帧样本；flip+1 306动作，仅t1721–1722双侧接触、无confirmed。两支均在18-action close budget后`bilateral_grasp_not_stable`，都未进入lift阶段；杯root在夹闭过程中最大抬高约6.8mm，属于接触扰动而非稳定持物。原生In/Close仍false。
- **原因与下一步：** +20mm接近夹持阈值但闭合时左右finger仍难以稳定落在同一截面；仅调高度和180°flip不足。geom105的approach法线保留，下一支在+25mm（已实测短暂稳定）的基础上，对flip−1候选绕approach轴roll±15°，让finger jaw旋转适应圆弧杯壁；两种角度独立分支，仍要求真实提升后才继续。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_zplus20_flipgrid_dev32/task_success_first_executor_obstacle_avoidance_L0T4_20261101_070713_768701326/run_report.json`；同目录NPZ按candidate_id区分两支，physical_step_records完整保存每动作/子步状态与指垫接触。

## 2026-11-01 当前局部目标：geom105 +20mm 双flip侧夹折中验证


## 2026-11-01 OA/L0T4 dev31：+15mm下两个腕部flip均未形成稳定闭合

- **假设/方法：** 从合法t1431分支，对geom105/jaw1/approach0的+15mm杯身中段偏置做flip−1/+1两态独立MuJoCo执行；维持1.35m obstacle-aware A*。
- **真实结果：** 两支共640动作（flip+1 308，flip−1 332），接触子步完整。+1只有一次左右finger同帧接触，grasp未确认；−1出现2个confirmed样本/4个双侧接触样本，但未达到稳定条件。两支杯root在闭爪过程中移动约4–5mm，maximum lift均0；In/Close=false。末态在杯壁g20附近主要剩finger1单侧接触。
- **原因与下一步：** +15mm使预抓位姿接近杯体中段且避免dev29的明显link6桌碰撞，但目标夹持几何未能保持两侧同步。与+25mm已确认双侧、lift约5.95mm后滑脱的点相比，+15与+25形成清晰高度边界；下一轮做单点中间偏置+20mm、保留两个flip，寻找lower-body稳定和link/table净空的折中。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_zplus15_flipgrid_dev31/task_success_first_executor_obstacle_avoidance_L0T4_20261101_070155_370995787/run_report.json`；两个候选动作通过candidate_id分列于同一NPZ及physical_step_records；失败Runtime Snapshots分别在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_zplus15_flipgrid_dev31/task_success_first_executor_obstacle_avoidance_L0T4_20261101_070155_370995787/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_070206_849410121`。

## 2026-11-01 当前局部目标：geom105杯身中段抓取偏置+15mm，比较两个腕部flip


## 2026-11-01 OA/L0T4 dev30：geom105+25mm首次双pad夹住杯身，提升仅5.95mm后滑脱

- **假设/方法：** 针对dev29 link6–table接触，将geom105/jaw1/approach0抓取中心沿mug object-local Z上移25mm；从合法t1431以1.35m staged A*高位绕障路线物理执行。
- **真实结果：** t1432–1749共318动作、7,950/7,950子步。high-A*与terminal approach避开了soap/box/table碰撞；t1721闭爪时真实左右finger同时接触杯壁几何（left finger1–g21/g22，right finger2–g17），t1726 grasp monitor确认双侧抓持。随后21动作lift使mug最多升高5.949mm；t1732左右pad接触丢失并继续滑脱，终态未抓住；In=false、Close=false。
- **原因与下一步：** 高度偏置解决可达性且证明该侧壁几何可形成短暂双pad夹持，但+25mm令接触点逼近杯口上缘；平滑/收窄杯壁上缘不能维持提升。下一轮做高度与腕部翻转的二维小网格：同一geom105、jaw1/approach0，Z偏置+15mm，分别flip−1/+1两个独立分支；lower contact更靠杯身中段，flip变化检验link6桌面姿态及pad夹持稳定性。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_zplus25_astar_dev30/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065902_209358684/run_report.json`；318动作NPZ、逐子步接触记录、视频及可恢复失败Snapshot在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_zplus25_astar_dev30/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065902_209358684/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_065913_893978880`。失败Snapshot SHA256 `b492cb2d2fce7bc7467dbcbe0cacfb1b06b8e69202a3945cb5af96a0460afb48`。

## 2026-11-01 当前局部目标：geom105侧夹上移25mm以清除link6–桌面接触


## 2026-11-01 OA/L0T4 dev29：geom105无clutter碰撞到达杯侧，但link6碰桌阻止末端

- **假设/方法：** 从合法t1431恢复，使用geom105/jaw1/approach0把手对侧杯壁；1.35m obstacle-aware A*高位XY路线，1支物理验证。
- **真实结果：** 299动作、7,475/7,475子步，成功离开dish-soap与white-place-box通道，t1730到geom105预抓端点附近。EEF约`[-0.1098,-0.2552,0.9644]m`，目标预抓点`[-0.1129,-0.2540,0.9521]m`；目标杯未被触碰，双pad未接触。仅见`robot0_link6_collision–table_collision`在t1705–1730持续26条接触记录，局部RRT后仍在12mm残差处停滞。原生In/Close均false。
- **原因与下一步：** A*侧向路线和候选面避障均已改善，剩余瓶颈是低预抓高度下机器人link6碰桌。保持geom105、approach方向和路线，沿mug object-local Z将接触中心上移25mm（仍在compiled wall高段范围内），再次检查link6净空及真实pad夹持/抬升。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_astar_high_dev29/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065628_012199104/run_report.json`；299态NPZ、接触记录、局部重规划Snapshot与候选路由诊断位于`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom105_astar_high_dev29/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065628_012199104/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_065638_356541455`。

## 2026-11-01 当前局部目标：OA/L0T4测试geom105侧面抓取及预抓净空


## 2026-11-01 OA/L0T4 dev28：A*跨过洗洁精瓶，但geom99下探终点仍撞瓶

- **假设/方法：** 从干净合法t1431 Snapshot恢复，固定杯身背把手侧geom99径向jaw1抓取；采用1.35m staged clearance和compiled-obstacle high-plane A*，以避免t1552直线通道中的dish-soap/white-place-box。
- **真实结果：** A*规划成功，使用XY路点`(-0.004,-0.285)→(0.000,-0.131)→(0.004,-0.132)m`；经197动作/4,925子步到t1628的预抓下降阶段。杯未移动、无pad接触。腕部在`[0.012,-0.133,1.314]m`停滞，真实hand_collision碰到dish-soap g135，共24条接触记录；native In/Close=false。
- **原因与下一步：** 高位横向路由可跨越，但geom99预抓终点XY`[0.004,-0.132]m`仍在洗洁精瓶附近，最后下探未留出手掌净空。改同一A*和高度，换杯身对侧另一段geom105，径向jaw1/approach0，预抓点约`[-0.113,-0.254,0.952]m`，其点到soap/box根中心分别0.289/0.219m；直接测试实际接触。
- **证据：** `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom99_astar_high_dev28/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065342_198912276/run_report.json`；路线/停滞诊断、完整NPZ与子步接触记录在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_geom99_astar_high_dev28/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065342_198912276/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_065352_685471616`。失败Snapshot与物理记录可恢复/审计。

## 2026-11-01 当前局部目标：OA/L0T4从t1431以A*高位路由执行侧夹


## 2026-11-01 OA/L0T4 dev27：径向侧夹几何正确，但t1552直线路由穿过dish-soap/white-place-box

- **假设/方法：** 从dev25 exact t1552 Snapshot尝试杯身背把手侧geom99的径向侧夹，候选`geom99_jaw1_approach0_-1_flip+1`，direct geometry approach，1支×440动作预算。
- **真实结果：** MuJoCo执行31动作至t1583后`task_success_first_rrt_osc_pregrasp_route_waypoint_stagnated_for_12_actions`。EEF从`[-0.0118,-0.2884,0.9795]m`到`[-0.0026,-0.2389,1.0174]m`；未进入抓取面、无双pad抓持。t1553手/指仍触mug g14/g15/g3；t1563 robot link5真实接触dish-soap g74等；t1573 gripper hand碰white-place-box g2。杯在起始机器人干涉中被推移约厘米量级，原生In/Close仍false。Snapshot可恢复。
- **原因与下一步：** 候选方向绕开了handle侧，但t1552起点末端已紧贴杯口/把手；点到点预抓路径穿过洗洁精瓶和白色收纳盒。不要从受扰的t1552再走直线。改用更早、更干净的合法同Episode t1431 Snapshot，侧夹候选不变，在1.35m抬升后用active compiled obstacles做Stage A A*高位XY transit，再旋腕、下探并物理核验夹持。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_sidegrasp_geom99_dev27/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065205_874339072/run_report.json`；child报告、31态NPZ、接触记录与failed Snapshot在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_sidegrasp_geom99_dev27/task_success_first_executor_obstacle_avoidance_L0T4_20261101_065205_874339072/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_065216_702586362`。失败Snapshot SHA256 `ffcf2d6284e1acb2ec1c991fdf4b56fb83aa3991efac8e7aa35a5359ca04c565`。

## 2026-11-01 当前局部目标：OA/L0T4从t1552改为杯身侧夹


## 2026-11-01 OA/L0T4 dev26：jaw1仍从杯口上方接近，手掌碰handle后停滞

- **假设/方法：** 从dev25 t1552 exact Snapshot恢复，固定`geom90_jaw1_approach2_-1_flip-1`，direct OSC approach，单支最多440动作；验证旋转jaw轴是否能避开jaw0的手指/手掌干涉。
- **真实结果：** t1553–1646共94动作、2,350/2,350接触子步。EEF由预抓约`[-0.012,-0.288,0.980]m`下降至`[-0.010,-0.289,0.959]m`后12动作停滞；mug root只在数毫米内变化。未见左右finger pad目标接触，抓持为false，In=false、Close=false。终态记录到`gripper0_hand_collision–white_yellow_mug_1_g1`，候选位置/方向未到预期面。
- **原因与下一步：** compiled geom90是杯身环壁的长条box：approach2沿杯轴向下，视频可见手腕进入杯口，且发生handle附近手掌碰撞；继续只换jaw轴无效。下一支恢复t1552 exact Snapshot，改抓杯身背离把手的geom99侧面，`jaw1/approach0/approach_sign=-1`，以direct approach物理验证从杯侧夹持。
- **证据：** parent report `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_t1552_jaw1_direct_dev26/task_success_first_executor_obstacle_avoidance_L0T4_20261101_064803_542364324/run_report.json`；子报告、94态NPZ、全子步接触记录及可恢复t1646失败Snapshot在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_t1552_jaw1_direct_dev26/task_success_first_executor_obstacle_avoidance_L0T4_20261101_064803_542364324/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_064814_427517585`。Snapshot SHA256 `b5cbcf4b504c277cf287a79bc1f4818e45f774c2c9241e6d7b88ec4379857e3e`；来源t1552 SHA256 `96dd06fbd5f2ddf6a0c58ba83303b570b3757e186cccb08d2c80b0e4da80414e`。

## 2026-11-01 当前局部目标：OA/L0T4 t1552接近Snapshot换jaw轴对准杯身

dev25从t1431以direct approach执行两个有界候选，已靠近mug表面。geom92 jaw0 EEF `[0.0155,-0.2679,0.9687]m` 时hand碰杯体g5–g7；geom90 jaw0 EEF `[-0.0132,-0.2883,0.9777]m` 时finger1碰handle geoms g1/g3、hand碰g13–15。两者均无finger pad接触，杯只移动2mm，原生In/Close false。接管最新geom90 branch t1552 exact Snapshot：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_L0T4_t1431_geometrypool8_direct_dev25/task_success_first_executor_obstacle_avoidance_L0T4_20261101_064455_554301602/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261101_064505_908073263/geom90_jaw0_approach2_-1_flip-1__rrtexplicit_geometry_direct_approach_physical_validation__failed_runtime_snapshot.pkl`。固定同表面几何候选族、换为`geom90_jaw1_approach2_-1_flip-1`，强制direct approach，单支最多440动作；考察两个pad能否夹住杯身，若仍是外壳碰撞则依据真实接触点改object-local Z/Y offset或换approach axis，不能重复高位RRT失败支。

## 2026-11-01 当前局部目标：OA/L0T4从可恢复t1431 Snapshot解决mug预抓和dish-soap阻挡

OA/L0T3 Extended成功已完成并归档，原生BDDL在t2102判定Soup与Cheese均In=true、reward=1、done=true。总覆盖19/30。下一优先任务`obstacle_avoidance/L0/T4`：恢复矩阵记录的同Episode最新Runtime Snapshot `/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261030/obstacle_avoidance_L0T4_liftfirst_t991_continuation_dev_01/v3_run/task_success_engine/task_success_first_executor_obstacle_avoidance_L0T4_20261030_184148_182862700/expert_grasp_search/task_success_first_grasp_obstacle_avoidance_L0T4_20261030_184203_721216653/geom95_jaw0_approach2__1_flip_1__approach_axis2_-26.2mm__rrtgenerated_geometry_high_approach__failed_runtime_snapshot.pkl`，场景root用同任务t0快照。先从report和真实接触轨迹确认t1431位置、task predicates、EEF及最近碰撞；此前同一通道link5触及dish soap，先改变 approach/侧向路由或使用screen发现的新抓取候选，不重复320+120步旧路线。按Standard/Extended分别标记，成功必须完整原生BDDL判定。

## 2026-11-01 当前局部目标：避开篮侧入口接触，改用保持水平夹持姿态的篮顶插入

dev23测试horizontal_aperture得到t1819连续Episode Snapshot，但末端24动作阻滞。真实接触对为gripper finger2–basket_1_g5、cream-cheese–basket_1_g5；双pad抓持保持，物体root约`[0.0323,0.3904,0.1427]m`，原生Soup In=true、Cheese In=false。失败原因是手/物体侧向穿越篮壁，不再重复相同侧向进场。立即exact resume t1819 held Snapshot，固定目标yaw−60°，`direct_region_progress + over_rim`，horizon2400。检查开口上方抬升、末端与篮壁碰撞、native In；如高位脱壁成功，继续垂直定位并仅以物理释放后的原生BDDL全任务成功验收。总体18/30。

## 2026-11-01 当前局部目标：用direct horizontal-aperture真实验证OA/L0T3篮侧入口

dev22从dev21 t1643确切持物快照恢复，但`obstacle_aware_xy_region_progress + horizontal_aperture`0动作route preflight失败：状态`not_requested`来自planner仅给`over_rim`计算A*路径，障碍路由变体又要求该路径存在。无物理状态变化。下一支保持t1643 Snapshot和Soup In真谓词，固定dev22 screen已通过的目标姿态`native-region-yaw-delta-degrees=-60`，使用`direct_region_progress + horizontal_aperture`，horizon 2200。观察是否能先在篮外侧向移动、再实际进入，并据碰撞/抓持变化续跑。

## 2026-11-01 当前局部目标：OA/L0T3恢复已抬离地面的cream-cheese持物Snapshot，改走水平篮口

dev21解决了dev20识别的竖夹无底部支撑问题：从合法t1009 Snapshot pin`geom95_jaw1_approach0_+1_flip+1`，双pad稳定接触15态，cheese真实抬升23.93mm。连续转运至t1643后仍持物，Soup In保持true，但over-rim入口失败：目标碰撞包围投影超出basket半宽（region half extents约61mm），高位路线13个航点已完成，入口姿态128动作未收敛。精确terminal Snapshot SHA `10a0662fd6b6b3a22f399df91573465625ab377b7269e46656b3ed9f77bfa8d7`。立即以该Snapshot运行独立续跑，snapshot root固定同一OA/L0T3 t0；使用`obstacle_aware_xy_region_progress`绕过场景障碍、`horizontal_aperture`从篮侧进入，demo horizon 2200、单分支≤440动作。检查持物双pad、每阶段物体姿态和原生In谓词。成功则保存完整Episode并继续下一个未完成任务；若入口仍停滞，按物理接触/目标物碰撞投影/EEF动作误差选择tilt、entry-height或route-side的单变量下一支。总体覆盖18/30。

## 2026-11-01 当前局部目标：OA/L0T3将cheese夹持轴转为水平以抬离floor

dev20从同Episode t1009以Soup In=true、Cheese false启动，geom95 jaw2/approach0 A*进场可达；242动作内surface approach真实碰到cheese，闭爪期间左右pad接触均为true，但20mm? 实测40mm lift段18步最大物体上升仅0.645mm，t1251双pad已丢。cheese root停在z≈0.0089m并有floor接触/约1–2mm穿透；jaw2对应垂直夹持，未给底侧finger留下抬升支撑。避免同jaw重试。下一支回到精确t1009 snapshot，pin `geom95_jaw1_approach0_+1_flip+1`：仍从+X侧向接近，jaw夹轴转成水平Y；预抓和开爪抓端点IK都通过，白place-box净空约127mm。使用0.45m高位A*，做一支真实夹持+抬升，若持物抬升则接着将cheese放进basket并按原生BDDL验收task_success。

## 2026-11-01 当前局部目标：OA/L0T3用侧向候选绕开white-place-box

总体覆盖18/30。OA/L0T3 t1009同Episode中Soup In=true，Cheese false。dev18的pool offset24窗口为空、0动作；dev19首16窗10个候选有端点IK但自动排序误回到已知`geom95_jaw2_approach1+1_flip+1`。精确续跑129动作至t1138，出现26次link6/white-place-box g4接触，EEF几乎无进展，cheese无接触；已知route不重试。task-approach screen表明`geom95_jaw2_approach0_+1_flip+1`端点IK都通过、white-place-box最近通道净空约158mm。下一支从dev14 t1009 Snapshot固定该候选，0.45m A*避障高位路线、单分支≤440动作，接触审计继续确认Soup In=true以及新目标pad接触和完整BDDL。

## 2026-11-01 当前局部目标：接续OA/L0T3已有Soup谓词，完成cream-cheese In

OAH/L0T3 dev45已经结束且没有接触Soup。299动作从t16到t315，高空A*XY段可达，但旋转下降阶段EEF停在z≈0.232m，距Soup约0.193m；实接触只有robot0_link6与hairbrush g10（56条记录），没有pad/Soup接触，目标未动。暂停这一路线族以避免重复。当前转到`obstacle_avoidance/L0/T3`：合法同Episode t1009 Snapshot中alphabet soup的原生In=true、cream cheese In=false，已近Standard边界而属于Extended续跑；最新失败在同侧抓取遇white_place_box侧碰、反侧候选预抓IK不可达。下一支从t1009 exact Snapshot，活动几何候选pool offset24筛选cream_cheese，优先候选screen pass和可达endpoint IK，启用障碍感知高位通道并做单个≤440动作的真实MuJoCo分支，保留Soup谓词，成功必须是完整原生BDDL task_success。

## 2026-11-01 当前局部目标：OAH/L0T3改横向approach轴

dev44表明geom74/jaw0的object-Y −15mm偏置没有改变首次物理接触点：t285–297仍是单侧左pad接触Soup外壳，接触Z约73–75mm，EEF正向X偏移，右pad不接触；t297 soup root约`[0.10255,-0.17471,0.03900]m`，两项目标均false。不要重复同一个approach-axis2上方下降或仅继续增加相同的Y偏置。下一轮从合法t16 Snapshot pin `geom86_jaw1_approach0_+1_flip-1`，其screen两端IK可达、预抓brush净空约40.5mm、Soup净空46.5mm、接近通道brush净空31.1mm；方向改为object-X横向接近。将接触中心沿重力对齐object-Y上移10mm，减轻其预抓姿态6mm地面干涉；0.45m障碍感知high A*单分支、≤440真实动作，核验末端接触双pad/抬升/native BDDL。

## 2026-11-01 当前局部目标：OAH/L0T3把接触点下移到soup罐身

总体覆盖18/30。dev43修复后确实在t296单pad接触现场闭爪8动作，但只有left pad，Soup根部约沿−X移动4.9mm，stable grasp=false且无抬升。因此geom74/jaw0原高度属于侧缘单边推挤；不重复该接触高度或单纯增大probe。下一轮从同一合法t16 Snapshot pin `geom74_jaw0_approach2_+1_flip+1` 的active geometry contact offset，沿soup对象局部Y下移15mm（当前Y轴与重力方向对齐），stage-A 0.45m high A*到达后完成真实surface approach/close/lift；最多一个物理分支、440动作。验证更低罐身处是否能产生对向pads并持物抬升；若未成功，检查真实触点和物体移动，再换不同contact face/approach axis。

## 2026-11-01 当前局部目标：OAH/L0T3从t296单侧接触处触发真实闭爪

总体覆盖18/30。dev41从精确t16恢复，geom74/jaw0走高位A*后于t285–296出现左指垫对汤罐多个编译geom的真实接触，右pad没有接触；汤罐根部约移动+2.3mm X，未双指抓持，t296 Snapshot可恢复。名义grasp frame仍约49mm远，因此原20mm阈值没有启动单侧闭爪探针。下一轮复用`geom74_jaw0_approach2_+1_flip+1`并从dev41 t296 failed Runtime Snapshot恢复，显式将`--stage-a-unilateral-close-probe-position-m`设为0.08m，令闭爪在已测单侧接触处发生，再以MuJoCo双pad稳定抓持及真实抬升验证。动作预算约120、单支；若双侧夹住并抬升，从该Snapshot延伸完整basket原生目标；若不成立则记录接触力学后改搜其他approach side/height。

## 2026-11-01 当前局部目标：OAH/L0T3换接触面验证汤罐抓取

总体覆盖仍18/30。dev40从精确合法t16 Snapshot以geom74/jaw1实测223动作；0.45m高位A*横移可达且未扫到毛刷，最终下降却出现gripper手掌与毛刷/汤罐接触，双指pad未触汤罐，t239因12步停滞退出。当前分支已结束，未发现运行rollout。不要重跑geom74/jaw1。下一轮沿用同一t16 Runtime Snapshot（避免重放16步），固定`geom74_jaw0_approach2_+1_flip+1`，其筛选记录显示预抓时毛刷/汤罐净空+23.1/+24.5mm；末段接触发生较早，故当作Stage A物理抓取假设，而非通过或失败的先验判定。继续启用0.45m staged high approach/A*、逐子步接触和原生BDDL，检查pads、物体移动及双目标谓词。

# SafeTRAIL Stage A 当前目标与局部进度（2026-11-01）

`GOAL_STATUS=ACTIVE`。唯一验收为30/30任务各自具备合法初态、真实MuJoCo连续执行、LIBERO-Safety原生BDDL成功判定和完整可审计Episode。覆盖仍 **18/30（12 Standard、6 Extended）**。

## 接管点：OAH/L0T3 dev39物理接近前暴露通用变量错误，t16快照可接续

dev38从合法t441持物Snapshot用2mm目标执行76动作：汤与EEF一同抬高约28mm，双侧pad持续至t516，无hairbrush/robot-scene碰撞；t517汤约2mm偏移和约5°转动后丢失两侧接触，原生In为false。该结果排除了A*横向碰撞，当前geom81夹持位姿承载失败。

dev39从fresh legal t0以active geometry pool offset16开始；16动作/400子步仅是开爪和预抓等待，没有目标接触或任务状态变化。共享RRT路点执行遇到`staged_waypoint_translation_axes`未初始化。已修复为进入可选staged route前初始化中性路点轴注记。精确t16失败Snapshot保持可恢复，SHA `fc531b45dad345e00196c6e60a961864148253b1c540ae732080e5448dcecad5`。offset16生成的geom74夹宽约29mm，screen brush净空−23mm（仍有碰撞风险），比已反复失败的geom81夹宽13.9mm、净空−35mm更有希望。

下一分支从t16恢复，pin geom74，使用0.45m分阶段高位A*预抓路线，真实核验hairbrush碰撞和双侧抓持；不重复开爪等待动作。总体覆盖保持18/30，证据见数据盘dev39记录。

## 历史背景：截至dev45的OAH/L2T2高位Y路段问题

dev41从t763 held Snapshot执行36动作至t799（900/900接触子步），真实下降到灶面，MuJoCo记录`flat_stove_1_collision_burner`与pot1接触；终态原生`On(moka_pot_1,flat_stove_1_cook_region)=true`，pot1释放后抓持消失。Pot2仍在桌面、On=false，stove Turnon=true，所以完整`task_success=false`，不能计入30任务覆盖。t799精确Snapshot中的pot1 root约`[0.0103,-0.2568,1.0032]m`并已在灶面，pot2 root`[0.0702,0.2382,0.9661]m`。

dev43从t799以t1053候选做49个真实动作，斜向接触法线逃逸后跨轴漂移并触碰Pot1，Pot2未接触。dev44从t799恢复后固定t248候选，共86动作至t885：开爪时Pot1移约16mm，随后EEF真实上升约175mm且清除了机器人–Pot1接触，但负X偏置侧绕实际漂向负Y，侧清障残差约57mm。dev45从t885精确恢复并核验Pot1 On=true、Pot2 On=false、Turnon=true；179动作到t1064（4475/4475子步），无机器人–场景接触，EEF沿Y前进约346mm，但同时+X约165mm、+Z约107mm，距离Y目标仍约194mm，局部RRT超时后12步Fallback停滞。Pot2仍未接触/抓持，终态原生状态未再查询。完整任务与覆盖不变（18/30）；t799仍是已确认的谓词前缀，t1064为另一条可恢复分支。下一步从t1064恢复并先重查原生谓词，实现轴选择性OSC：当前高位段仅发送所选世界轴分量、按该轴误差推进，允许耦合漂移由后续轴段纠正。

## 2026-11-01 最新试验：dev40抵达Cooktop支撑上方；dev41验证真实下降

dev40从合法t0 Episode的精确t728持物Snapshot执行35个真实动作至t763（875/875接触子步）。水平On路线保留live物体朝向后，A*航点旋转命令为零，持Pot1先后到达逃逸/运输/原生目标对齐阶段；双侧抓持持续，记录中无lemon-bottle或robot-link接触。t763 Snapshot实际pot1 root为`[0.0083,-0.2493,1.1844]m`；cook_region局部XY约`[0.0560,-0.0515]m`，位于region范围内，距所选合法支撑点约7.4mm，仍高于目标支撑高度约0.20m。它因目标对齐默认5mm阈值在24步停滞，原生pot1/pot2 On仍false、stove Turnon=true。该物理证据表明可改用更宽但仍受原生BDDL约束的Stage A对齐容差推进下降。

dev41已从精确t763 held Snapshot启动，将水平On最终目标对齐容差设为15mm，并移除障碍A*对对象包围圆额外8mm膨胀；物理接触审计和原生On仍是唯一判定。Standard总horizon剩237动作，分支上限235。其结果需继续验证真实灶面接触、松爪稳定及Pot2。Snapshot和证据：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L2T2_preserve_on_orientation_dev40/task_success_first_executor_obstacle_avoidance_human_L2T2_20261101_034712_657881982/expert_native_goal_stage_control/placement_branch_00_region_layout_61_obstacle_aware_xy_region_progress/native_goal_transport_failed_runtime_snapshot.pkl`；覆盖仍18/30。

## 2026-11-01 接管更新：OAH/L2T2 A*持物运输到t698，dev38验证自由yaw容差

当前活动实验为dev38，唯一来源是合法t0连续Episode的dev37终态Snapshot（t698）。系统此前没有在运行的rollout；dev38已从该Snapshot启动，Standard总horizon为1000、剩余302动作，分支动作预算300。首次启动仅因漏设`PYTHONPATH`在Python import前退出、没有物理动作；已补环境变量重新启动。

dev37从dev34 t520双侧持物Snapshot分支执行178个真实MuJoCo动作至t698（4,450接触子步）。修复后的障碍A*扩展326个格点、仅将lemon bottle作为横向障碍，并在起点保守包络重叠时真实执行118.8mm径向逃逸。Pot1沿瓶子西侧/南侧绕行至炉灶目标区附近；全程左右指垫抓持为真，无bottle或robot-link接触。末端停在A*第11号高位航点前约12mm，触发24动作停滞监控；这是中间waypoint带插值yaw但On谓词不约束yaw导致的容差过严，不是丢抓或碰撞。终态pot root约`[-0.0281,-0.2527,1.1804]m`，pot1/pot2 On均false、stove Turnon=true。

dev38把`obstacle_aware_region_progress`在On任务中的位置容差设为15mm、姿态软容差设为0.65rad，并从t698精确恢复；30动作后在t728停滞。双侧抓持持续、无lemon-bottle或robot-link接触，但A*重规划的19mm逃逸点仍插值出约0.1rad/动作的yaw命令，EEF发生跨轴漂移，On仍false。原因是自由yaw还在作为OSC目标，不是单纯接纳容差太紧。

dev39的首次启动在Snapshot前置校验时因命令漏写`task_success_engine/`目录而以0物理动作退出，未改变状态。dev40已用完整路径从精确t728 Snapshot启动；水平On路线保留实时持物朝向，避免向任意支撑yaw旋转，分支最多260动作、Standard剩余272动作。若原生On判定显示保留姿态不能稳定支撑，再依据真实MuJoCo结果调整。t728 Snapshot：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L2T2_free_yaw_continue_dev38/task_success_engine/task_success_first_executor_obstacle_avoidance_human_L2T2_20261101_034226_926537946/expert_native_goal_stage_control/placement_branch_00_region_layout_61_obstacle_aware_xy_region_progress/native_goal_transport_failed_runtime_snapshot.pkl`。dev38报告和动作记录在同一分支目录。覆盖仍18/30。

## 历史路线背景：OAH/L2T2 t481 held-pot运输，dev34测试RRT超时后的接触容忍OSC关键姿态

活动进程审计无SafeTRAIL实验运行。OA/L2T3矩阵旧的fresh-t0 dev01“running”标记已过期；最新有效报告dev17在t356结束，alphabet soup与cream cheese原生In均false。OAH/L0T4的−90°腕姿已在dev16实际执行至t1619仍失败，不重跑该建议。

选择OAH/L2T2：合法t0 Episode中`Turnon(flat_stove_1)=true`，pot1有真实双指接触的可恢复Snapshot，pot2仍在桌面。dev23–26尝试低表面、低净空和yaw变化的native-region OSC；dev26从精确t297分支到t481，pot1仍与两侧finger pads接触，EEF/手体持续接触柠檬汁瓶，pot1与pot2的原生`On`均false。t481 Snapshot：
`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L2T2_t297_yaw90_low_clearance_dev26/task_success_first_executor_obstacle_avoidance_human_L2T2_20261101_015033_890164911/expert_native_goal_stage_control/placement_branch_00_region_layout_12_direct_region_progress/native_goal_transport_failed_runtime_snapshot.pkl`

dev28已验证该held-pot Snapshot无物理动作恢复损失，但legacy RRT唯一选中的cooktop中心位没有有界当前场景IK解，因此没有控制动作。dev30的16个候选中没有完整3/3代表航点IK；原排序让1/3高余量位压过2/3路线位。已改为先按覆盖航点数排序，再比较关节余量，并给无朝向约束的native On目标加入0/±90/180°支撑法线偏航选项，每个方向保留候选支撑位置。dev31的16候选中8个达到3/3代表航点IK；首选region_layout63的−90°姿态终点可达，但全几何RRT在14.3秒/166次配置检查后耗尽12秒预算，物理动作0。dev32两条30秒full-geometry RRT均超时；dev33使用已验证的L1T2自有成功keypose（只迁移目标姿态、不回放源动作），但首个lift RRT在13.42秒/165次配置检查/1轮后超时，执行0动作，实时双指抓持与源t481状态均未变。dev34首段OSC闭环从t482到t520执行39动作，EEF与pot1根部均真实抬升约74mm；双侧pad抓持持续，首步1次柠檬瓶接触后终段无该碰撞。但原生On仍false，且关键位姿因控制响应慢而未到：8mm指令的实测EEF响应约1.9mm/动作，方向误差趋近但剩余约106mm。dev35从t520精确恢复但未执行动作：同一L1T2 `lift`终点（EEF z≈1.341m）在当前已抬高臂姿下直接无有界IK，而pot1已经抬高73.7mm并脱离柠檬瓶；停止追该过高姿态。dev36实测A*返回`obstacle_aware_xy_path_not_found`（59×72网格仅扩展1节点），但旧实现错误地退回直线OSC；23动作后pot1接触lemon bottle并失去右指抓持，原生On仍false。根因是`_cook_region`目标stove owner未被解析/排除，start cell又在bottle保守二维膨胀区，A*无出口。dev37回到t520双指快照：修正region owner，选 cook region 内瓶体包络clearance较大的支撑点，给包络内起点生成真实径向逃逸航点，并把A*未找到路径改为物理前硬失败，禁止直线回退。20mm路线仍受双指实时检查和1000步horizon约束。任务未原生成功前不更新覆盖数。

实验轻量日志和复用经验在`/opt/data/private/gs_opd/safetrail_stage_a_exploration/`，整体路线见`docs/GS_OPD_ROADMAP.md`。


### dev66运行状态

dev66现从精确t2421 Snapshot执行geom81 jaw2/approach1−1/flip−1腕姿匹配变体；主假设为减少dev65观察到的167.6°大旋腕带来的轨迹偏离。等待真实物理动作、接触/抓持/lift和native result。运行输出根：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261101/obstacle_avoidance_human_L1T3_t2421_geom81_jaw2_wristmatch_dev66`。
