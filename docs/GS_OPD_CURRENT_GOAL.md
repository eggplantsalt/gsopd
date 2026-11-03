# SafeTRAIL Stage A 当前目标

GOAL_STATUS=ACTIVE。唯一验收为30个OA/OAH任务均拥有合法Episode出发、完整连续真实MuJoCo执行且原生BDDL task_success=true的轨迹。

## 已核实覆盖

根据30任务矩阵与原生成功索引，当前23/30：12个Standard、11个Extended。七个未完成配置：OA/L0T4、OA/L1T3、OA/L1T4、OA/L2T4、OAH/L0T3、OAH/L0T4、OAH/L2T4。成功索引为`/opt/data/private/gs_opd/safetrail_stage_a_exploration/successes/`中的三条Extended轨迹；矩阵另记录Standard成功。

## 当前局部目标：OAH/L0T4继续真实闭门与杯位验证

OAH/L0T4最初的闭门接触源为dev372 t4836 Snapshot：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261103/obstacle_avoidance_human_L0T4_close_panel_segment_dev372/run/terminal_runtime_snapshot.pkl`，SHA256 `c6d84618d5a2a689fa64fce94a61063cdf869b21aeeeeaa09cd9d1fb19dd9308`。当时双指pad接触门板g17，门角约−1.3964rad；杯根[-0.055416, 0.265579, 0.945079]m，距原生In区域world-y下界约0.89mm。当前最新连续分支已推进至dev399 t4958，后续具体结果见下方dev399记录。

dev377从该精确源尝试将已验证+0.20rad跟随姿态缩至+0.10rad；只读末端碰撞预检发现gripper手与cap几何重叠23.6mm，超过已用5mm余量，故0动作终止，integration state保持不变。dev378从原高度沿门板下移实际约11mm，执行16动作，双pad—g17接触持续且hand—cap接触清除，杯根基本不变；随后+0.10rad端点仍因cap overlap 13.6mm被拒绝。dev379/380均在规划阶段0动作退出。终审发现二者waypoint误用了observation `state.robot0_eef_quat`，与MuJoCo Runtime Snapshot姿态不同，造成1.50rad假性关节绕行；后续统一从`physical_step_records.snapshot_state.robot_states[0].eef_quat`取姿态。dev381按正确Snapshot姿态从t4852继续下移约12mm，再沿切向执行+0.05rad目标；32动作/800个完整子步将门角从−1.38816推进至−1.34283rad，pad—g17接触持续、杯根基本不动。native In/Close仍false。dev382从t4884精确源进行+0.05rad的碰撞预检，端点hand—cap overlap为7.116mm，超过5mm余量约2.1mm，故0动作；dev383把末端再下移10mm时与cap另一几何g87重叠8.55mm，0动作。dev384执行8个真实动作，EEF沿+Y移动约6mm，双pad—g17接触保持但门角从−1.34283轻微回弹至−1.34972rad；hand—cap仍近零接触，杯稳定。dev385从t4892以MuJoCo live腕姿沿+X推进3mm，2动作后门角由−1.349722到−1.332624rad（+0.017098rad），双pad—g17接触持续、杯稳定，原生目标仍false。dev386再沿+X推进3mm/2动作，门角−1.332624→−1.316204rad（+0.016420rad），双pad—g17接触保持，cap穿入约0.48mm且杯稳定。dev387前两个3mm目标完成，累计6真实动作，门角−1.316204→−1.298149rad（+0.018055rad），cap穿入最高约1.32mm、双pad—g17接触保持；第三目标因cap限制被预检拒绝。dev388从t4902单步+3mm推进2动作，门角−1.298149→−1.294681rad（+0.003468rad），pad—g17保持，cap穿入约1.53mm，杯稳定。dev389的+4mm段执行4动作，门角−1.294681→−1.291028rad（+0.003653rad），pad—g17保持、cap穿入约1.72mm且杯稳定。dev390的无穿透+6mm端点被cap g1预检以6.494mm拒绝（上限5mm），0动作。dev391在手—cap具名pair 7mm规划限额下物理执行4动作，门角−1.291028→−1.282528rad（+0.008500rad）；接触使cap根x移动约1.3mm，g17 pad接触持续，杯稳定。dev392从t4912执行6mm切向段4动作，门角−1.282528→−1.278026rad（+0.004502rad）；cap根x再物理移动约0.72mm，pad—g17保持且杯稳定。dev393从t4916提出的首个+6mm目标需hand—cap重叠7.132mm，较7mm上限超0.132mm，0动作。dev394的+5mm切向命令执行4动作，但门角净变化仅+0.00026rad、cap根几乎不动，手部被cap挡住，In/Close仍false。dev395的+Y25mm接触点端点被预检以9.51mm hand—cap穿入拒绝，0动作；继续横移会更靠近cap。最近点离线几何显示局部工具Z负向滚腕10°可把手部邻近cap的点抬高约16mm。dev396的局部Z−10°端点仍与cap重叠9.77mm，0动作。dev397真实执行+10°滚腕32动作，双pad—g17接触保持、cap根x实际−2.6mm、g1穿入降至约0.2mm；但门角回开0.0167rad，杯稳定，native目标false。dev398从新腕姿t4952沿闭门+X做3mm微推，确认方向性并监视cap/g17。成功必须由原生BDDL同时报告mug In=true、microwave Close=true、task_success=true；未成功前不计覆盖。该任务为Extended。

dev399结果：从dev398 t4954按live Runtime Snapshot腕姿做+6mm world-X切向目标，4个真实动作/100个接触子步到t4958。门角−1.287612→−1.280214rad（净闭合+0.007398rad）；门板接触点沿X约移动2.25mm，EEF实际移动约2.41mm，垫片最近间隙约0.47–0.88mm。hand—cap g87最大穿入约2.33mm，仍低于具名7mm规划限额；cap根x物理增加约1.22mm。杯根[-0.055476,0.265577,0.945116]m，稳定由g7支撑。原生In=false、Close=false、task_success=false。动作连续、所有100个接触子步均记录。合法终态Snapshot SHA256 `9b607f18c062e8c7508fa6d109377575bab2805a39fc5440d7ce2ac41a570504`，run在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261103/obstacle_avoidance_human_L0T4_live_tangent_after_roll_dev399/run/`。dev400从精确t4958源以Runtime Snapshot真实EEF姿态规划+3mm world-X，重算门板与cap接触净空，不提高7mm限额。

dev400结果：从dev399 t4958按当前Runtime WXYZ姿态规划+3mm world-X，4个真实动作/100个接触子步到t4962。虽然关节空间RRT端点预检可行（最小禁碰净空2.37mm；具名hand—cap限额仍为7mm），EEF最终只移动约−0.45mm并有3.48mm端点误差；门角逐步在−1.27964至−1.28313rad间回摆，终态−1.28193rad，净回开约0.00171rad。g17垫片间隙约0.47–0.88mm；hand—cap g87约2.33mm，杯根[-0.055480,0.265577,0.945094]m稳定由g7支撑；原生In/Close/task_success=false。合法终态Snapshot SHA256 `c52368734d060f18750195fdec37857faf0164ce9e5b24e3b960a527c001c5d3`，报告在`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261103/obstacle_avoidance_human_L0T4_live_tangent_after_roll_dev400/run/`。这验证当前切向腕姿已进入cap阻挡/无稳定铰链力矩区，暂停重复+X微推。

dev401尝试沿用OAH/L1T4 dev489链路前的+120mm竖直清障抬升，从dev400精确t4962 Snapshot启动真实OSC。60动作内未到达抬升目标，抛出`TimeoutError`；由于执行器在异常路径没有先写出内存动作记录或Runtime Snapshot，本次没有可审计物理轨迹，也不能可靠定位具体接触阻塞。运行报告为`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261103/obstacle_avoidance_human_L0T4_link6_articulated_close_dev401/task_success_first_executor_output/task_success_first_executor_obstacle_avoidance_human_L0T4_20261103_115645_886372592/run_report.json`，状态`executor_error`；没有终态Snapshot，dev401分支不用于续接。

dev402从dev400 t4962 Snapshot沿world −X退10mm，碰撞感知关节RRT可行，执行8动作/200接触子步到t4970。EEF实际位移约[-8.81,-2.42,+0.17]mm，端点误差2.71mm；门角−1.281928→−1.318969rad（因保留pad—g17接近而开约0.03704rad）。hand—cap g87穿入由约2.33降至1.58mm、g206降至0.26mm；cap根x由约.06773移动到.06166m，属于机器人真实接触导致；杯根稳定约[-.055483,.265576,.945094]m，native In/Close=false。终态Snapshot SHA256 `f76a1a66b29ff29b4c4621a834c58011bfb2fb478ac7ad08c7a36df447cbccaf`，报告/物理记录：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261103/obstacle_avoidance_human_L0T4_link6_access_retreat_dev402/run/`。

dev403从精确dev402 t4970状态尝试link6—g17闭门预接近，全几何RRT在物理前拒绝，0动作。转换后的高位预接近目标为[-0.232605,0.145679,1.354972]m；所有8个IK候选都因当前杯—微波炉物体对被报告为禁碰而判unsafe，其中六个候选本身无自碰，机器人至cap、门板、杯的端点净空分别约114mm、21.8mm、187mm。8个禁碰条目全部是mug body23—microwave body26在g7处约0.49–0.99mm的真实承托近接触，MuJoCo力约0.14–0.23N；没有出现link6预接近路径失败。报告：`/opt/data/private/gs_opd/overnight/safetrail_stage_a_20261103/obstacle_avoidance_human_L0T4_link6_articulated_close_dev403/task_success_first_executor_output/task_success_first_executor_obstacle_avoidance_human_L0T4_20261103_120142_489269693/run_report.json`，终态与源相同t4970、native In/Close=false。

已针对该测量证据修改articulated-fixture规划策略：仅当原生BDDL含`In(target, fixture_region)`、编译区域的parent等于当前hinge fixture、且Runtime实时接触检查确认该target与fixture collision body在2mm内并有至少0.01N测量力时，移除`forbidden_object_pairs`中的这一个target—fixture对。其余物体、机器人—物体、机器人自碰检查不变；该planner例外不写入MuJoCo、不移动杯/障碍，也不声称安全认证。dev404从dev402相同合法t4970 Snapshot重跑link6全几何预接近，以报告明确记录该例外是否被应用；若后续因机器人起点cap接触被拒，再做有记录的短物理脱离。当前门角−1.318969rad，杯In=false；闭门后如In仍false，再毫米级物理推杯补齐。

已验证方法：dev369–372使用测得的手腕或夹爪—microwave g17接触及分段关节RRT/PD；dev489在相同microwave几何上以`robot0_link6_collision`真实接触闭门，206动作把门从−0.194789推进至−0.003307rad并得到原生成功。dev401尝试的是这一已验证接触机制在L0当前铰链角下的几何重规划，不拼接历史轨迹。

## 暂停分支：OA/L2T4

dev263–292复核了高位RRT过门与外侧低位入口。dev278在真实物理执行中触发过原生In=true；更深插入后松夹无法稳定，偏置/抬高撤离会把杯拖出，dev291–292最终滑至微波炉前沿并未形成可靠再抓持。最新dev292 t2905杯根[-0.031683,0.229146,0.933311]m，In/Close均false，Snapshot SHA256 `98f7f7036ed68f47aa9e8bb0ffb18622ebbf40fc4522e0f54a0c1f8f656606fb`。停止该松夹/垂直撤手机制，保留其合法分支供未来不同物理路线使用。

## 最新已完成任务：OAH/L1T4 dev489（Extended，t4810）

dev489从dev488精确t4604 Runtime Snapshot连续执行206个真实动作至t4810；物理记录覆盖连续步4605–4810和5150个接触子步。link6–microwave g17实测接触将门铰链由−0.194789推进至−0.003307rad。原生BDDL给出mug In=true、microwave Close=true、task_success=true、reward=1、done=true。成功索引：`/opt/data/private/gs_opd/safetrail_stage_a_exploration/successes/OAH_L1T4_dev489_native_success.json`。归类Extended，不计作Standard 1000成功。

历史路线细节保留在下方；其中旧覆盖计数和当时的下一步只代表历史时间点，当前状态以上述矩阵与本节为准。

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
