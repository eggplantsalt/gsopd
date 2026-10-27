# OA L0T0 闭爪接触逐 Pair 诊断

本表按真实 Geom ID 合并接触点；每个旧 Bundle 只有一个强制 `[0,0]` 静态采样点。距离为有符号距离（负数表示穿透），并非电机闭爪结果。完整各接触点距离、Body/Link、指垫身份和开度见同目录 JSON。

| Bundle / Grasp变体 | Geom ID / 名称 | Body / MuJoCo Link | 分类 | 接触点记录数 | 出现采样点数 | 最深距离(mm) |
|---|---|---|---|---:|---:|---:|
| 20261027 / 0 | 73 `gripper0_finger1_collision`<br>77 `gripper0_finger2_pad_collision` | 14 `gripper0_leftfinger`<br>17 `gripper0_finger_joint2_tip` | 机器人自碰撞 | 1 | 1 | -0.623535 |
| 20261027 / 0 | 73 `gripper0_finger1_collision`<br>111 `akita_black_bowl_1_g28` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -0.016099 |
| 20261027 / 0 | 73 `gripper0_finger1_collision`<br>112 `akita_black_bowl_1_g29` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.090133 |
| 20261027 / 0 | 73 `gripper0_finger1_collision`<br>113 `akita_black_bowl_1_g30` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.839579 |
| 20261027 / 0 | 73 `gripper0_finger1_collision`<br>114 `akita_black_bowl_1_g31` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.839579 |
| 20261027 / 0 | 74 `gripper0_finger1_pad_collision`<br>76 `gripper0_finger2_collision` | 15 `gripper0_finger_joint1_tip`<br>16 `gripper0_rightfinger` | 机器人自碰撞 | 1 | 1 | -0.623535 |
| 20261027 / 0 | 74 `gripper0_finger1_pad_collision`<br>77 `gripper0_finger2_pad_collision` | 15 `gripper0_finger_joint1_tip`<br>17 `gripper0_finger_joint2_tip` | 机器人自碰撞 | 4 | 1 | -0.500000 |
| 20261027 / 0 | 76 `gripper0_finger2_collision`<br>112 `akita_black_bowl_1_g29` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.998160 |
| 20261027 / 0 | 76 `gripper0_finger2_collision`<br>113 `akita_black_bowl_1_g30` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -2.026090 |
| 20261027 / 0 | 76 `gripper0_finger2_collision`<br>114 `akita_black_bowl_1_g31` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -2.026090 |
| 20261028 / 3 | 73 `gripper0_finger1_collision`<br>77 `gripper0_finger2_pad_collision` | 14 `gripper0_leftfinger`<br>17 `gripper0_finger_joint2_tip` | 机器人自碰撞 | 1 | 1 | -0.623535 |
| 20261028 / 3 | 73 `gripper0_finger1_collision`<br>111 `akita_black_bowl_1_g28` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.516288 |
| 20261028 / 3 | 73 `gripper0_finger1_collision`<br>112 `akita_black_bowl_1_g29` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.639279 |
| 20261028 / 3 | 73 `gripper0_finger1_collision`<br>113 `akita_black_bowl_1_g30` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.618100 |
| 20261028 / 3 | 73 `gripper0_finger1_collision`<br>114 `akita_black_bowl_1_g31` | 14 `gripper0_leftfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.618100 |
| 20261028 / 3 | 74 `gripper0_finger1_pad_collision`<br>76 `gripper0_finger2_collision` | 15 `gripper0_finger_joint1_tip`<br>16 `gripper0_rightfinger` | 机器人自碰撞 | 1 | 1 | -0.623535 |
| 20261028 / 3 | 74 `gripper0_finger1_pad_collision`<br>77 `gripper0_finger2_pad_collision` | 15 `gripper0_finger_joint1_tip`<br>17 `gripper0_finger_joint2_tip` | 机器人自碰撞 | 4 | 1 | -0.500000 |
| 20261028 / 3 | 76 `gripper0_finger2_collision`<br>111 `akita_black_bowl_1_g28` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.507957 |
| 20261028 / 3 | 76 `gripper0_finger2_collision`<br>112 `akita_black_bowl_1_g29` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.426226 |
| 20261028 / 3 | 76 `gripper0_finger2_collision`<br>113 `akita_black_bowl_1_g30` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.493047 |
| 20261028 / 3 | 76 `gripper0_finger2_collision`<br>114 `akita_black_bowl_1_g31` | 16 `gripper0_rightfinger`<br>23 `akita_black_bowl_1_main` | 非指垫—目标碗 | 1 | 1 | -1.493047 |

两组分别是 **13 条记录 / 10 个唯一 Pair**、**14 条记录 / 11 个唯一 Pair**。其中均有 6 条自碰撞记录 / 3 个唯一 Pair；其余分别为 7 / 8 个非指垫—碗 Pair。没有 Hand 接触。

指垫身份：左指垫 ID 74 / Body 15 `gripper0_finger_joint1_tip`，右指垫 ID 77 / Body 17 `gripper0_finger_joint2_tip`。非指垫 ID 73 / Body 14 `gripper0_leftfinger` 和 ID 76 / Body 16 `gripper0_rightfinger` 为不同的 mesh 碰撞 Geom。CuRobo link 对应关系不由名称推测，JSON 中未核实字段为 null；这里 Robot Link 特指 MuJoCo 实际所属 Body。

## 无碗接触对照

同一真实模型执行 5 组 × 8 个静态开度检查，包含初始远离碗的手臂、两组原抓取构型，以及原构型下仅在 scratch 数据中将碗上移 2m 的对照。未改 live state / model / contact rules，动作和 timestep 均为 0。所有零开度状态重现相同三组自碰撞，最深距离分别约 -0.623535、-0.623535、-0.500000mm。证据支持“零开度几何本身存在重叠”，不支持“电机物理闭爪必然到达零开度”。

原 Bundle 1 的静态闭爪还使用了 planned EEF→Object 关系重新设置 scratch 碗位姿；新对照保持原始碗位姿，所以其最后一项为 12 条而旧归档为 13 条。两者不混记，唯一 self Pair 与穿透深度一致。边界附近微小姿态差异能改变接触流形点数。

## 历史真实轨迹核查

已找到自有完整成功 711 步轨迹。首次双真实指垫接触 frame=46，但存在非指垫—碗接触；全轨迹仅 frame=[473] 的一个 post-action 状态满足 pad-only，最长连续长度=1。它不是可靠的 pad-only 抓持前缀，也不能据单帧认证安全。其旧安全结论只覆盖绑定的特定禁止物对。

run50 为真实抓持/抬升前缀，原生完整任务未成功、逐子步安全监测不完整；旧 `left_finger/right_finger` 组包含 finger mesh 和 pad。新候选只借用其 `verified_object_to_eef_at_grasp` 实测稀疏姿态，不借用旧候选理论矩阵，也不重放动作。
