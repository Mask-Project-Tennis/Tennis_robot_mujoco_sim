# R1-A7-D 型号、模型与接口审计报告（Task 2.0）

- **日期**：2026-11-15
- **性质**：文档与资源核对（不运行运动控制、不写业务代码、不安装 SDK、不连硬件）
- **任务来源**：`docs/superpowers/plans/2026-11-15-unitree-stage2-construction.md` Task 2.0
- **验收方式**：证据审查。本报告所有断言均附来源（URL+commit+SHA256 或仓库文件路径）；未编造任何 pytest 记录。
- **结论速览**：固定 commit 的官方 URDF SHA256 校验通过；左右臂 7×2 关节名与官网整机索引 15–21/22–28 一一对齐（并获 unitree_sdk2 低层示例代码直接确认）；发现指南与 URDF 共 3 类限位冲突（腰、双侧腕俯仰、双侧腕滚转）——只记录不裁断；整机 SDK 走 DDS（hg IDL + `rt/lowcmd`/`rt/lowstate`），高层 `rt/arm_sdk` 封装只覆盖 A5 兼容 13 关节、不含 A7 特有的腕俯仰/腕偏航，A7 全 7 关节需低层接口。实物型号、击球侧、真机标定为未确认项。

---

## 1. 目标型号与版本证据表

证据等级说明：**[指南]** = 宇树官方文档中心页面事实；**[公开示例]** = 官方开源仓库代码事实；**[实机]** = 本台机器人实物验证结论（当前一项都没有，全部待真机核验）。

| 项目 | 结论 | 证据等级 | 来源 |
|---|---|---|---|
| 目标型号 | R1-A7-D（R1-A5-D/R1-A7-D 双版本系列中的 7 自由度单臂配置；用户口径为 R1-A7-D，**实物是否为 A7 版未确认**） | [指南] | 指南首页"部件名称"节：R1-A5-D 相对 R1-A7-D 减少手腕关节电机；单手臂自由度 5/7 可选 |
| 单臂自由度 | 7（肩俯仰/肩侧摆/肩偏航/肘/腕滚转/腕俯仰/腕偏航） | [指南]+[公开示例] | 指南关节限位表 15–21/22–28 各 7 项；URDF 每臂 7 个 revolute joint |
| 总自由度（带夹爪） | 22（指南口径；其中底盘 2、升降 1、腰 1、头 2、双臂 14，余 2 为夹爪计数，**夹爪计数口径未确认**，URDF 中 Dex1 为二指各 1 个 prismatic 关节、左右各一） | [指南] | 指南首页规格表；URDF `left/right_dex1_finger_joint_1/2` |
| 官方模型源 | `unitree_ros/robots/r1d_description`，commit `157ff950351e5f535e03c5d85a16b7d594b3dbcc` | [指南]+[公开示例] | 指南 get_sdk 页 URDF 链接指向 `unitree_ros/tree/master/robots/r1d_description`；本报告固定到上述 commit |
| r1d.urdf 完整性 | SHA256 = `39ad1a206c34c3deecd33745ccda0c0c94ece4195f674dfbaf950f567f08809c`，与计划记录一致，**校验通过** | [公开示例] | raw 下载后本地 sha256sum（暂存于 `.superpowers/sdd/2026-11-15-unitree-stage2-construction/r1d-src/r1d.urdf`） |
| 整机 URDF 结构 | 轮子×2（continuous）+ 被动脚轮×4 组（各 Yaw+Roll continuous）+ 双级升降（2×prismatic，mimic 1:1）+ 腰（revolute）+ 双臂（2×7 revolute）+ 头（2 revolute）+ 二指夹爪×2（4 prismatic）+ 腕部相机×2（fixed） | [公开示例] | r1d.urdf 全文解析（1139 行，robot name="R1D"） |
| 官方 SDK | `unitree_sdk2`，R1 专属示例与 wrapper 存在；本次核对基于 main HEAD `aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1`（2026-11-09，晚于最新 tag 2.0.2） | [指南]+[公开示例] | 指南 get_sdk 页；GitHub API commits/main |
| 通信架构 | 整机 DDS 中间层，IDL 兼容 ROS2（需适配 RMW）；开发接口 DDS（C++/Python）/ROS2/GST（仅图传）；二次开发仅在外置计算单元（192.168.123.164） | [指南] | 指南架构页第 4/6 条与"开发接口"节 |
| 独立 D2 arm_sdk 可用性 | **不能由上述证据推导**（D2 路线已作废，不作为 R1-A7-D 验收依据） | — | 计划"型号更正与执行边界"节 |
| 资产许可 | unitree_ros 仓库根 `LICENSE` 为 **BSD-3-Clause** 全文（版权人 HangZhou YuShu TECHNOLOGY CO.,LTD，2016-2022），SHA256 `84aac59fd3246e3ddc49d1387644e8fcf43b0def4f0b9fe687f372e90446df2d`；r1d_description 子目录**无包级 package.xml**，根许可按仓库惯例覆盖子目录属合理推断；BSD-3 允许再分发但须保留版权与免责声明。**保守起见本报告记 license_status=bsd-3-clause(仓库根，子目录无独立声明)，资产入库与对外分发前仍需主代理/用户复核** | [公开示例] | 仓库根 LICENSE 原文；`robots/r1d_description` 树仅含 `meshes/` 与 `r1d.urdf`（GitHub API tree，truncated=false） |
| unitree_sdk2 许可 | BSD-3-Clause（版权 2016-2023），仅作参考核对，本阶段不安装 | [公开示例] | GitHub API license + raw LICENSE 头部 |
| 实机验证 | **无**。本报告不含任何实机测量；真机零位、限位、能力值均未验证 | [实机] | — |

### r1d_description 资产清单（36 个 STL + 1 个 URDF）

URDF 引用网格全部位于 `meshes/`。下列 SHA256 为本次逐文件下载后本地计算，git blob SHA 与 GitHub API 树完全一致（双重校验，下载完整可信）。原始下载物暂存于 `.superpowers/sdd/2026-11-15-unitree-stage2-construction/r1d-src/`（git-ignored，不作为仓库运行依赖）。

| 文件 | 大小 (B) | SHA256 |
|---|---|---|
| r1d.urdf | 41014 | 39ad1a206c34c3deecd33745ccda0c0c94ece4195f674dfbaf950f567f08809c |
| meshes/base_link.STL | 4181384 | a9e6b7fd0a8c2c852ca3552bfa68097d46311f32896a9c7e2d17761467d9b98e |
| meshes/Camera.STL | 338084 | 9ac57a9bc1fe9f8a1aff2b679626f61bbb3a8e7c789cb330a07ae000be8a66b3 |
| meshes/Dex1_base_link.STL | 1261084 | a4e95f91e1b3a349580080bf01d34307221c6be1462c01900392e677dbf668e5 |
| meshes/Dex1_finger_link_1.STL | 478484 | f34639c388f4a0b14670b7076307c76aae927dd076b426887d4677adf82790a8 |
| meshes/Dex1_finger_link_2.STL | 478484 | ee154ad8f5209d23387d85bed5622319151615e31bec0bd12fc44eccfa29603d |
| meshes/front_left_caster_Roll_link.STL | 294484 | 8b6d7fd800039387dadbbe905f21b856a014641e304269d352531177e68e46d7 |
| meshes/front_left_caster_Yaw_link.STL | 301384 | 91c3190b59e945a40b0cddbbf721b6162d7443ae6b8b023a2c913fdc8f845661 |
| meshes/front_right_caster_Roll_link.STL | 294484 | 241853e788c8cfb09c6e02fc06b90c26caa000d734342f1b7715f8ad15e4f43b |
| meshes/front_right_caster_Yaw_link.STL | 301384 | e64c58331b721062fe119ef622ae975fe1b23e150d35b1f02fda114e752280c3 |
| meshes/head_pitch_Link.STL | 279084 | f2c64723273d93c2da9d5e204f27cd569f8fda0477dc0595760ce09d9d7f9db9 |
| meshes/head_yaw_Link.STL | 1122384 | 2ba30deb440e2584b7595d619ed3829c4ec52ae5c78a2e929b8be70577ceadf3 |
| meshes/left_elbow_link.STL | 373384 | e6b6584de6ec6c629da24fbebdc7b88cccd7ed24ad0d7547fa2d871fce4faa7c |
| meshes/left_shoulder_pitch_Link.STL | 420084 | df713bd42d5349868befd618a7e571815677c7f39d026f716a56c366a9521a00 |
| meshes/left_shoulder_roll_link.STL | 542884 | 7fcfd330aa78b58a9f2dcbf6f87d62b6314eae3678d36abbbc1632a29cf9d7b9 |
| meshes/left_shoulder_yaw_link.STL | 649584 | 6b09a1c14f9ef1a38fe4d658b4e51586f972b0bace28cd90caada7d80db50fb0 |
| meshes/left_wrist_pitch_link.STL | 527484 | 7b3c0568ebd0c864e2401d80436cb6ea0fdc4b83ac0892311d734e33b35a7bff |
| meshes/left_wrist_roll_link.STL | 165084 | 1cab8b3cf32fd7423b663d600dbc2666f5bcb3c78998c701335111bc8b96848b |
| meshes/left_wrist_yaw_link.STL | 657784 | e490c5848bfab23e4780789cb65e98fa1788f104e759c5647caa0b2eee10f8e2 |
| meshes/Left_Wheel_link.STL | 470084 | 09c88e244da0d5087d3313d5ed7f5b06162c98126a0383cc5e8e6d34579b4386 |
| meshes/lift_mount_link.STL | 12284 | 1bd33bf5a12005f2c7f4f0e580fc40b0ac6242898077afa8621429bce34b512c |
| meshes/lift_stage1_link.STL | 28884 | 63368dd170c97a4a4c8ee6fbd331352687aee04e129ca49e5bf187ddf82729bd |
| meshes/lift_stage2_link.STL | 69984 | dccf3ebd7390c90548cdb07e1acb0b3198a67f6351e3b3432a7b8f6971f3cd7e |
| meshes/pelvis_link.STL | 18884 | 17486b5bb0ee7b24608a87983a45bd690b87515345ebd20161a8d0287b45bfe6 |
| meshes/rear_left_caster_Roll_link.STL | 294484 | 75c1d1bb63ebc022a1f9f96ddf639d9773d7d731f8720880887dd0e5011dcc16 |
| meshes/rear_left_caster_Yaw_link.STL | 301384 | 04b8d6ae703f0bd3a051b99fd498e0350f2ac17367f9b1f3dc50ae68e9438dc6 |
| meshes/rear_right_caster_Roll_link.STL | 294484 | 40cc63c5af5a8b0c28c269c2fec5da81353e846edbf48b0f17b2b072a827f427 |
| meshes/rear_right_caster_Yaw_link.STL | 301384 | 9450b99657a58470d2989921c79fc21f098e457e492c37fa11d4f32063fb67d1 |
| meshes/right_elbow_link.STL | 365284 | 1bd3a60789fe4b35ce399d6134e53fa84010f2c6c3e3d888c5cfd91b095ae37e |
| meshes/right_shoulder_pitch_Link.STL | 425184 | beba627f582135b63c05d71e410d6bd5510f313d6fa0229aca67d06c05b92216 |
| meshes/right_shoulder_roll_link.STL | 535034 | 97e41a1b567ffda5019cdefe9bfe6b0c637b589d07ccaaece5aa4e6ed2b31d15 |
| meshes/right_shoulder_yaw_link.STL | 643184 | 37f814647983716e1049c18815a6d6f45eb616b9db1958469b4eb21914fc7e05 |
| meshes/right_wrist_pitch_link.STL | 527584 | 74974a903dd795cf43d19dd67d0e8b455433d38eeeab016b2873da35882160a9 |
| meshes/right_wrist_roll_link.STL | 164084 | c4f86872bfa9f72a19e331b7c3b2fb5bf1282e2b6b88b60642abfe3cb8ae4a7d |
| meshes/right_wrist_yaw_link.STL | 657284 | 87f1fcfa6bfbffb8478b7164652f6c9067f56ea47b52859b718d48b59b214050 |
| meshes/Right_Wheel_link.STL | 470084 | 3341865c7d2bc7f494b1037631443b7ffe550e49ec0abad5348b292d29c17ccb |
| meshes/waist_yaw_link.STL | 2065984 | 54b7f815c79e0b085e783286fdaa64796f0e0ce00e13410b100921390b4c1135 |

**左右臂/头部/夹爪网格复用关系**：`Dex1_base_link.STL`、`Dex1_finger_link_1/2.STL`、`Camera.STL` 为左右两侧共用（同名引用）；其余按 side 命名独立。

**指南附加资源（未纳入本阶段资产）**：get_sdk 页提供 R1-A7-D 智能版/旗舰版简化模型 STEP zip（oss-global-cdn 链接）。本阶段几何基准选定 URDF+STL 路线，STEP 仅记录为备选参考，不下载入库。

---

## 2. 左右臂 URDF 关节名与整机索引对照表

排列顺序按任务要求：肩俯仰 → 肩侧摆 → 肩偏航 → 肘 → 腕滚转 → 腕俯仰 → 腕偏航。
**两套地址体系严格分列**：

- **SDK 整机索引**：宇树 DDS `LowState_.motor_state[]` 数组下标（通信接口地址），来源=指南关节限位表，并经 unitree_sdk2 `example/r1/low_level/r1_A7_wrist_swing_example.cpp` 的 `joint_idx_in_idl{15..21, 22..28, 29, 30}` 直接确认。
- **MuJoCo 状态地址**：阶段 2 派生模型中由生成顺序决定，**必须经 `bind_robot_model` 按关节名解析**（`src/robot/model_binding.py`），不得把 SDK 索引当作 qpos/qvel 下标使用，也不得反过来。

### 左臂

| 语义 | URDF joint 名 | URDF 子 link | SDK 整机索引 | SDK 名称 | 指南限位 (rad) | URDF 限位 (rad) | URDF axis | effort (N·m) | velocity (rad/s) |
|---|---|---|---|---|---|---|---|---|---|
| 肩俯仰 | `left_shoulder_pitch_joint` | left_shoulder_pitch_Link | 15 | L_SHOULDER_PITCH | [-3.1416, 2.0944] | [-3.1416, 2.0944] | 0 1 0 | 60 | 18.8 |
| 肩侧摆 | `left_shoulder_roll_joint` | left_shoulder_roll_link | 16 | L_SHOULDER_ROLL | [-0.2269, 2.4784] | [-0.22689, 2.4784] | 1 0 0 | 60 | 18.8 |
| 肩偏航 | `left_shoulder_yaw_joint` | left_shoulder_yaw_link | 17 | L_SHOULDER_YAW | [-1.9199, 1.9199] | [-1.9199, 1.9199] | 0 0 1 | 33 | 33.4 |
| 肘 | `left_elbow_joint` | left_elbow_link | 18 | L_ELBOW | [-0.9757, 2.1850] | [-0.97564, 2.1852] | 0 1 0 | 33 | 33.4 |
| 腕滚转 | `left_wrist_roll_joint` | left_wrist_roll_link | 19 | L_WRIST_ROLL | [-2.0944, 2.0944] | **[-1.9199, 1.9199]** ★ | 1 0 0 | 33 | 33.4 |
| 腕俯仰 | `left_wrist_pitch_joint` | left_wrist_pitch_link | 20 | L_WRIST_PITCH | [-0.7854, 0.7854] | **[-0.5236, 0.5236]** ★ | 0 1 0 | 10 | 37.7 |
| 腕偏航 | `left_wrist_yaw_joint` | left_wrist_yaw_link | 21 | L_WRIST_YAW | [-1.5708, 1.5708] | [-1.5708, 1.5708] | 0 0 1 | 10 | 37.7 |

### 右臂

| 语义 | URDF joint 名 | URDF 子 link | SDK 整机索引 | SDK 名称 | 指南限位 (rad) | URDF 限位 (rad) | URDF axis | effort (N·m) | velocity (rad/s) |
|---|---|---|---|---|---|---|---|---|---|
| 肩俯仰 | `right_shoulder_pitch_joint` | right_shoulder_pitch_Link | 22 | R_SHOULDER_PITCH | [-3.1416, 2.0944] | [-3.1416, 2.0944] | 0 1 0 | 60 | 18.8 |
| 肩侧摆 | `right_shoulder_roll_joint` | right_shoulder_roll_link | 23 | R_SHOULDER_ROLL | [-2.4784, 0.2269] | [-2.47849, 0.2268] | 1 0 0 | 60 | 18.8 |
| 肩偏航 | `right_shoulder_yaw_joint` | right_shoulder_yaw_link | 24 | R_SHOULDER_YAW | [-1.9199, 1.9199] | [-1.9199, 1.9199] | 0 0 1 | 33 | 33.4 |
| 肘 | `right_elbow_joint` | right_elbow_link | 25 | R_ELBOW | [-0.9757, 2.1850] | [-0.97564, 2.1852] | 0 1 0 | 33 | 33.4 |
| 腕滚转 | `right_wrist_roll_joint` | right_wrist_roll_link | 26 | R_WRIST_ROLL | [-2.0944, 2.0944] | **[-1.9199, 1.9199]** ★ | 1 0 0 | 33 | 33.4 |
| 腕俯仰 | `right_wrist_pitch_joint` | right_wrist_pitch_link | 27 | R_WRIST_PITCH | [-0.7854, 0.7854] | **[-0.5236, 0.5326]** ★（且与左侧不对称） | 0 1 0 | 10 | 37.7 |
| 腕偏航 | `right_wrist_yaw_joint` | right_wrist_yaw_link | 28 | R_WRIST_YAW | [-1.5708, 1.5708] | [-1.5708, 1.5708] | 0 0 1 | 10 | 37.7 |

标注 ★ 的行进入第 5 节冲突清单。非 ★ 差异（-0.22689 vs -0.2269、2.1852 vs 2.1850 等）为指南 4 位小数舍入，不视为冲突，但**同形数组不视为同义数组**：派生模型与测试必须逐字采用 URDF 数值，不得以指南舍入值替代。

**其他部件 SDK 索引**（供阶段 4 参考）：0=L_CHASSIS_WHEEL、1=R_CHASSIS_WHEEL、2–11 空（被动脚轮未编址）、12=LIFTING_COLUMN、13=WAIST_YAW、14 空、29=HEAD_PITCH、30=HEAD_YAW。

**SDK 枚举警告**：`unitree_sdk2/include/unitree/dds_wrapper/robots/r1/defines.h` 的 `JointIndex` 枚举是**带双腿的 R1 布局**（0–11 腿、12=WaistRoll、13=WaistYaw、左臂 15–19 止于 WristRoll、右臂 22–26、头 29–30），与 R1-A-D 指南表在**臂段起点一致但枚举缺 A7 特有的 20/21/27/28，且 12 号索引含义不同**（defines.h=WaistRoll vs 指南=LIFTING_COLUMN）。**底盘/腰/升降段索引不能从 defines.h 推导**；A7 腕俯仰/腕偏航必须按指南表+低层示例的裸索引使用。

---

## 3. 轴、限位与零位约定核对

- **轴向约定**：URDF axis 均为所在 link 系的标准基向量（无斜轴）。肩俯仰/肘/腕俯仰绕 y，肩侧摆/腕滚转绕 x，肩偏航/腕偏航绕 z。左右臂轴向相同（不做镜像翻转），侧别差异只体现在 origin 的 y 分量符号与 roll 限位区间翻转。
- **肩部安装倾角**：`left_shoulder_pitch_joint` origin rpy=(+0.26187, 0, 0) rad（+15°），随后 `left_shoulder_roll_joint` origin rpy=(-0.26187, 0, 0) 反向补偿（右侧符号相反）。零关节角时两 rpy 绕同一 x 轴先后作用相互抵消，中间连杆存在固定 Z 形偏置——**零位姿态不是简单的垂直下垂**，FK 对照必须逐字使用 origin xyz+rpy。
- **零位定义**：URDF 无 `<safety_controller>`、无 zero-offset 声明，URDF 零位 = 全部关节变量取 0。SDK 侧 q=0 姿态与 URDF 零位是否同一物理姿态**未确认**（官方低层示例把 0 作为可用目标姿态，但没有官方图文确认两侧零位重合；见第 7 节）。
- **限位来源优先级**：派生模型逐字采用 URDF `<limit lower/upper>`（rad）；指南表数值只作交叉核对。两侧冲突见第 5 节，**只记录，不裁断，不扩限，不修正疑似笔误**。
- **能力值（effort/velocity）**：与 D2 URDF 全 0 不同，R1D URDF 的臂关节 effort/velocity 均非 0（见上两表；肩 60 N·m 与指南规格表"肩膀最大扭矩 60Nm"一致）。这些是官方 URDF 声明值，**未经实机验证，不得直接写入安全限值**；按现有参数层约定进入配置时仍须与真机辨识/官方确认对齐（阶段 3/4 决定）。
- **升降/轮/夹爪/头能力值**：升降 effort=3000 N、velocity=0.04 m/s（每级）；轮 effort=31.7 N·m、velocity=27.9 rad/s；手指 effort=20 N、velocity=0.2 m/s；头 pitch 33 N·m / yaw 0.86 N·m；被动脚轮 joint 无 `<limit>` 标签（URDF continuous 关节宽松写法，解析时不当作能力 0，也不当作已声明能力）。

---

## 4. 世界系 → 底盘 → 升降 → 腰 → 肩 → 腕 → 球拍 变换链

以下全部 origin 数值逐字取自固定 commit 的 r1d.urdf（位姿均为父系到子系变换）。

```
world
 └─ base_link                      [整机根，11.100 kg；冻结：整机固定于地面]
     ├─ Left_Wheel_joint           continuous，origin (-0.037284, 0.1968, -0.0925)，axis (0,-1,0)  → 冻结
     ├─ Right_Wheel_joint          continuous，origin (-0.037284,-0.1968,-0.0925)，axis (0,-1,0)  → 冻结
     ├─ 4×脚轮 Yaw/Roll            continuous，无 limit 标签 → 冻结（被动）
     └─ lift_mount_joint           fixed，origin (-0.13328, 0, 0.3288)
         └─ lift_mount_link        [7.8424 kg]
             └─ lift_stage1_joint  prismatic z，origin (0.0523, 0, 0.0427)，range [0, 0.425] m → 冻结为实验前设定值 q_L
                 └─ lift_stage1_link [3.7976 kg]
                     └─ lift_stage2_joint prismatic z，origin (0.1402, 0, -0.2115)，range [0,0.425] m，
                        mimic lift_stage1_joint multiplier=1 offset=0 → 冻结为同一 q_L
                         └─ lift_stage2_link [0.7701 kg]
                             └─ upper_body_mount_joint fixed，origin (0,0,0)
                                 └─ pelvis_link [0.4434 kg]
                                     └─ waist_yaw_joint revolute z，origin (0,0,0)，URDF ±2.618 rad → 冻结 0
                                         └─ waist_yaw_link [6.024 kg]（躯干上部，头与双臂的父级）
                                             ├─ head_pitch_joint  revolute，origin (-0.006, 0.03155, 0.255) → 冻结 0
                                             │   └─ head_yaw_joint revolute → 冻结 0
                                             ├─ 左臂 7 关节链（§2 左臂表；肩俯仰 origin (0, 0.085688, 0.19749) rpy(+0.26187,0,0)）
                                             │   └─ left_wrist_yaw_link [末端连杆，0.1091 kg]
                                             │       └─ left_dex1_base_joint fixed，origin (0.0485, -0.003, 0.02) → 工具安装参考
                                             │           ├─ 左 Dex1 二指夹爪（prismatic ×2）→ 移除/冻结，拍面占位替代
                                             │           └─ left_dex1_camera_joint fixed（腕部相机）→ 移除
                                             └─ 右臂 7 关节链（同构，y 分量取反；肩俯仰 origin (0,-0.085688,0.19749) rpy(-0.26187,0,0)）
                                                 └─ … → 右工具安装参考 right_dex1_base_joint origin (0.0485, 0.003, 0.02)
```

**冻结策略（阶段 2 仿真与规划一致）**：轮、脚轮、双级升降（两级同值 q_L）、腰、头、对侧臂全部冻结；仅选定击球臂 7 关节主动。q_L 与腰 0 位姿必须在真机运行时以同样方式保持（升降由实验前设定、腰锁定为 0 或等效保持），**真机上腰/升降能否机械锁定未确认**（见第 7 节）。

**双级升降 mimic 位移关系（运动学推导，非官方文字声明）**：
`lift_stage2_joint` mimic `lift_stage1_joint`（multiplier=1, offset=0），两级同为 z 轴 prismatic。设两级关节值同为 q，则
- lift_stage2_link 原点在 lift_mount_link 系中的 z 坐标 = 0.0427（stage1 origin z）+ q − 0.2115（stage2 origin z）+ q = **2q − 0.1688**（m）
- 即上段总行程 = 2 × 0.425 = **0.85 m**；总速度 = 2 × 0.04 = 0.08 m/s = **80 mm/s**，与指南"升降柱上升/下降 最大 80mm/s"**相互印证**。
- 指南整机高度"收缩 683mm / 升起 1600mm"差 917mm，与 0.85m 行程不同口径（外廓尺寸含底盘/头部，无法精确对账，记录不裁断）。
- **MuJoCo 移植注意**：MuJoCo 的 URDF 导入不保留 mimic 等式；因升降在规划中冻结为常数，派生模型应把两级 prismatic 一并固定为同一 q_L 值（或转为固定段），不引入运行时 mimic。

**基座高度语义**：阶段 2 的固定基座位姿（挂在 lift 冻结后的某一臂安装父级之上）表达"实验前设定的固定高度"，与 Global Constraint"升降不参与运动规划、仿真与规划一致"对齐；该数值是虚拟固定基座的刚体平移，不证明真实升降机构动力学。

---

## 5. 指南与 URDF 限位冲突清单（只记录，不裁断）

| # | 部件 | 指南值 (rad) | URDF 值 (rad) | 备注 |
|---|---|---|---|---|
| 1 | 腰 WAIST_YAW（索引 13） | ±0.6109（=±35°，与规格表"腰关节 Yaw±35°"一致） | ±2.618（=±150°） | 差 4.3 倍。可能分别是控制安全限位与机械极限，官方无说明。阶段 2 冻结腰为 0（两集合公共值），不选边 |
| 2 | 左腕俯仰 L_WRIST_PITCH（20） | ±0.7854 | [-0.5236, 0.5236] | 指南宽于 URDF；派生模型采用 URDF 值 |
| 3 | 右腕俯仰 R_WRIST_PITCH（27） | ±0.7854 | [-0.5236, 0.5326] | 同上；且 URDF 右侧自身不对称（+0.5326 vs 左 +0.5236），疑似官方笔误，**不修正** |
| 4 | 左/右腕滚转 WRIST_ROLL（19/26） | ±2.0944（=±120°） | ±1.9199（=±110°） | 指南宽于 URDF 10°；派生模型采用 URDF 值 |

**处理边界**：
- 阶段 2 派生模型与测试逐字采用 URDF 数值（模型是可审计的唯一几何事实来源）；指南数值仅存档对照。
- 缺少真机确认**不阻止**只读 FK 检查与场景组装；但**不能**据此验收真实安全范围、高速运动能力或宣称"官方模型已等于实机标定模型"。
- 上述冲突未上真机前无解；阶段 4 真机接入时必须以实机回读限位为准并回报修订。

另有指南内部不一致（非 URDF 冲突，记录备查）：正文"2-4kg 手臂负载" vs 规格表"3~5kg"；get_sdk 页 STEP 表第二列表头误写"R1-A7-D智能版"两列（应为智能版/旗舰版）。

---

## 6. 整机 SDK 话题 / IDL / 机型模式 / 控制权（只读核对，阶段 4 输入）

核对对象：unitree_sdk2 main@`aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1` 下 `include/unitree/dds_wrapper/robots/r1/*`、`include/unitree/idl/hg/*`、`example/r1/**`。**本阶段不安装 SDK、不连硬件**；以下仅为公开代码事实，具体控制频率、停止/失联行为、未选中部件保持策略留待阶段 4 实机验证。

### 6.1 话题与 IDL

| 话题 | 方向 | IDL（unitree_hg::msg::dds_） | 用途与证据 |
|---|---|---|---|
| `rt/lowcmd` | 发布 | LowCmd_ | 低层电机命令；`r1_A7_wrist_swing_example.cpp` HG_CMD_TOPIC |
| `rt/lowstate` | 订阅 | LowState_ | 低层全量状态（motor_state[] 按 §2 整机索引寻址、imu、无线遥控、mode_machine）；同示例 HG_STATE_TOPIC |
| `rt/secondary_imu` | 订阅 | IMUState_ | 躯干第二 IMU；同示例 HG_IMU_TORSO |
| `rt/arm_sdk` | 发布 | LowCmd_（复用） | 高层臂 SDK；`robots/r1/r1_pub.h` ArmSdk 构造默认话题 |
| `rt/api/arm/request` / `rt/api/arm/response` | RPC | — | 机载臂动作服务（预设动作/示教回放）；`r1_arm_action_example.cpp`（API id 7106–7113） |

注意：低层走 **hg** 命名空间 IDL（非 go2）；R1 复用 g1 wrapper 的 LowState 订阅类（`g1_sub.h` 中 LowState 默认话题同为 `rt/lowstate`）。

### 6.2 电机索引寻址（与 §2 对照表闭环）

`r1_A7_wrist_swing_example.cpp` 定义本地序（0–15：左臂 7、右臂 7、头 2）→ `joint_idx_in_idl{15,16,17,18,19,20,21, 22,23,24,25,26,27,28, 29,30}` → IDL 数组下标。**这是"左 15–21、右 22–28、头 29–30"的官方代码级确认**，与指南表完全一致。

### 6.3 机型模式与控制权

- **机型模式**：`LowState_.mode_machine()` 字段即机型标识（示例打印 "R1 type"）。具体取值枚举未见公开文档，阶段 4 需实机回读确认（含区分 R1-A5-D/R1-A7-D 与带腿 R1）。
- **控制权（低层）**：先经 `MotionSwitcherClient`（b2 命名空间）`CheckMode`/`ReleaseMode` 释放机载运动控制服务，再发 `rt/lowcmd`；命令需 CRC32（多项式 0x04c11db7）校验，低层周期 2 ms（500 Hz）。
- **控制权（高层臂）**：`rt/arm_sdk` 以 `mode_pr = weight×100`（0–100）平滑接管/交还；enable 时把目标位置初始化为当前实测位置防跳变；release 线性退权。示例发布 ~100 Hz。
- **命令语义**：`motor_cmd[i] = {mode(1=Enable), q, dq, kp, kd, tau}`（阻抗型），官方示例 Kp：肩 [100,100,100,100]→腕 [50,50,50]、Kd=2（低层腕摆示例）；高层 KP [50,50,40,40,30 | 腰 50 | 头 15,15]。这些是**示例值不是安全参数**。
- **占用互斥**：臂动作服务错误码 7400 = "`rt/arm_sdk` is occupied"；7401 = holding 须先 release。官方源码注释明确警告：**不要用 G1 的 `g1_arm_action_client.hpp` 替换 R1 版**（同名服务但动作 ID 表与错误码集不同）——直接证据支持"不套用 G1/D2 示例"。

### 6.4 高层臂 SDK 的 A5 兼容限制（关键结论）

`r1_pub.h` 中 `ArmSdk::JOINTS` 固定为 13 关节：左臂 5（15–19，止于 WristRoll）+ 右臂 5（22–26）+ WaistYaw + HeadPitch/HeadYaw，**不含 A7 特有的腕俯仰/腕偏航（20/21/27/28）**。因此：
- 仅用高层 `rt/arm_sdk` 封装**无法**驱动 7 自由度臂的全部关节；
- R1-A7-D 的 7 关节控制需走低层 `rt/lowcmd`（hg LowCmd_，按整机索引直接寻址 15–21/22–28），并承担 ReleaseMode + CRC + 500 Hz 周期的实时义务；
- 该结论影响阶段 4 接口选型（高层 vs 低层），本报告不替阶段 4 做决定。

### 6.5 与现有仓库接口的关系

- 仓库 `src/robot/robot_profile.py` / `model_binding.py` / `runtime_parameters.py` 为模型侧名称绑定与参数层，与 SDK 整机索引无耦合；阶段 4 若需要"关节名 ↔ SDK 索引"映射，应新建显式映射表（数据源=本报告 §2 表），不得让 ModelBinding 携带 SDK 索引语义。
- 现有 `d2_runtime.yaml`、`parked-2.5/d2.yaml` 等配置不作为 R1-A7-D 配置（原切片处理表裁定）。

---

## 7. 未确认项清单

| # | 项 | 现状 | 阻塞与否 |
|---|---|---|---|
| 1 | **实物型号**（R1-A5-D vs R1-A7-D；智能版 vs 旗舰版） | 用户口径 R1-A7-D；无实物铭牌/序列号证据 | 不阻塞阶段 2 模型准备（按 A7 建模）；阻塞真机验收 |
| 2 | **击球侧（左/右）** | 完全未确认。**不从 RM65 的右臂选择推导宇树侧别** | 不阻塞：阶段 2 同步产出左右两套关节名/限位/网格与 FK 基准，配置层留 side 参数；派发实现时不预设默认侧 |
| 3 | **真机标定**（URDF 是否等于这台实机的几何/零位/限位） | 未验证；指南与 URDF 已有 4 处冲突（§5） | 不阻塞只读 FK；阻塞真机安全范围与高速运动验收 |
| 4 | SDK q=0 与 URDF 零位是否同一物理姿态 | 无官方图文；低层示例仅把 0 当可用目标 | 不阻塞（模型侧以 URDF 零位为准并显式记录该假设）；阶段 4 上机首步核对 |
| 5 | 腰/升降在真机上的锁定方式（机械锁 vs 控制保持） | 未知 | 阶段 4 输入；仿真侧按"冻结"假设记录 |
| 6 | 低层停止/失联/未选中部件保持策略（DDS 断链后关节行为） | 指南/代码未公开说明 | 阶段 4 实机验证输入 |
| 7 | mode_machine 机型枚举取值 | 未公开 | 阶段 4 实机回读 |
| 8 | 指南"总自由度 22"的夹爪计数口径 | URDF 为双 Dex1 二指（4 prismatic）；指南另述 Dex3-1 灵巧手（7 DOF） | 仅记录；不影响臂建模 |
| 9 | 官方"简化模型 STEP"与 URDF 的几何一致性 | 未比对（未下载） | 不阻塞；记录为备选几何来源 |
| 10 | unitree_sdk2 固定版本策略（main@aaf69cb5 vs tag 2.0.2） | 阶段 4 依赖尚未引入 | 阶段 4 开工前由主代理固定 |

---

## 8. 证据链接索引

1. 官方指南首页（部件名称/关节顺序名称与关节限位/机器人规格）：https://support.unitree.com/home/zh/R1-A-D_SDK_Development_Guide/R1-A-D_SDK_Development_Guide （更新时间 2026-11-13，本次全文抓取成功）
2. 获取 SDK：https://support.unitree.com/home/zh/R1-A-D_SDK_Development_Guide/get_sdk （更新时间 2026-11-12，抓取成功）
3. 软件架构说明：https://support.unitree.com/home/zh/R1-A-D_SDK_Development_Guide/architecture_description （更新时间 2026-11-12，抓取成功）
4. 固定 URDF：https://github.com/unitreerobotics/unitree_ros/blob/157ff950351e5f535e03c5d85a16b7d594b3dbcc/robots/r1d_description/r1d.urdf （raw SHA256 = 39ad1a20…8089c，校验通过）
5. r1d_description 树（36 STL 清单与 git blob SHA）：https://api.github.com/repos/unitreerobotics/unitree_ros/git/trees/157ff950351e5f535e03c5d85a16b7d594b3dbcc:robots/r1d_description 及其 meshes 子树（truncated=false）
6. unitree_ros 根 LICENSE（BSD-3-Clause 全文）：https://raw.githubusercontent.com/unitreerobotics/unitree_ros/157ff950351e5f535e03c5d85a16b7d594b3dbcc/LICENSE
7. unitree_sdk2 R1 低层示例（索引映射/话题/CRC/ReleaseMode）：https://github.com/unitreerobotics/unitree_sdk2/blob/aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1/example/r1/low_level/r1_A7_wrist_swing_example.cpp
8. unitree_sdk2 R1 wrapper（defines.h 枚举 / r1_pub.h ArmSdk JOINTS）：https://github.com/unitreerobotics/unitree_sdk2/tree/aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1/include/unitree/dds_wrapper/robots/r1
9. unitree_sdk2 高层臂示例（weight 接管/100Hz/KP-KD）：https://github.com/unitreerobotics/unitree_sdk2/blob/aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1/example/r1/high_level/r1_arm_sdk_dds_example.cpp
10. unitree_sdk2 臂动作服务示例（rt/api/arm、错误码、勿用 G1 客户端警告）：https://github.com/unitreerobotics/unitree_sdk2/blob/aaf69cb56c9d5ed63d8b284008ec3a8ab999bec1/example/r1/high_level/r1_arm_action_example.cpp
11. 仓库先例（D2 资产清单格式）：`assets/unitree_d2/source_manifest.yaml`；加载器资源集合：`src/utils/mujoco_loader.py::_SIGNATURE_RESOURCE_DIRS`；`.gitattributes` 现有 `-text` 行
12. 本任务暂存下载物（git-ignored）：`.superpowers/sdd/2026-11-15-unitree-stage2-construction/r1d-src/`（r1d.urdf、meshes/×36、REPO_ROOT_LICENSE、sdk2-ref/×6）

---

## 9. 对新版模型切片的输入摘要（详细施工方案见计划修订版）

1. **新资产目录**：`assets/unitree_r1d/`（原始 URDF + 36 STL + source_manifest.yaml + README + LICENSE 拷贝），格式沿用 D2 manifest 先例；许可节记 bsd-3-clause（仓库根）并保留"再分发前复核"提示。
2. **导出流程**：从整机 URDF 提取选定臂链（waist_yaw_link 之下 7 关节 + 末端）+ 冻结环境几何（base/lift/pelvis/waist 或其简化包络）→ 生成 `<mujocoinclude>` 裸臂片段与资产片段；**不得**把独立 D2 臂改名顶替；**不得**令完整模型维数等于规划维数（环境几何冻结段不计入主动 n=7）。
3. **`.gitattributes` 需补**：`assets/unitree_r1d/** -text` 与 `src/robot/r1d_*.xml -text`（先例：assets/unitree_d2/** 与 src/robot/d2_*.xml）。
4. **`_SIGNATURE_RESOURCE_DIRS` 需补**：`("assets/unitree_r1d", "*")` 一行（先例：assets/unitree_d2）。
5. **parked-2.5 可回捞的通用机械**：ElementTree 模板合并与参数注入（`_build_template_element`/`_apply_scene_parameters` 的生成与加载共用单一入口防漂移）、跨目录 mesh 路径重定位（`_retarget_mesh_paths`）、`jnt_range` → q_lower/q_upper 回填（`_populate_joint_limits`）、维数/初态校验（`_validate_model_shape`/`_check_initial_state`）、圆盘惯量公式与相邻对 exclude 表、六组核心测试思路（合同/刚体变换/惯量/独立实例/限位回填/穿透诊断 + 防漂移三件套）。所有 D2 专属常量（d2_base/link_7/joint_1..7/FLANGE_POS）不可复用。
6. **SDK 索引映射表**：以本报告 §2 表为唯一数据源，阶段 4 需要时新建显式映射，不进 ModelBinding。

---

## 10. 自审（本次核对的局限）

- 指南三链接首次静态抓取为空壳（SPA 动态渲染），后经渲染抓取成功获取全文；抓取内容以页面"更新时间"（2026-11-12/13）为准，若官方后续改版，限位表与架构描述需重新核对。
- r1d.urdf 为静态快照；`unitree_ros` master 后续变更不影响本报告结论（固定 commit），但阶段 4 前建议复查 master 是否有 r1d_description 修订。
- unitree_sdk2 核对基于 main HEAD（无对应 release tag），版本浮动风险已列入未确认项 #10。
- 升降"总位移 2q−0.1688、总速度 80mm/s"为 URDF 运动学推导 + 指南速度交叉印证，非官方文字声明；已在 §4 标注推导性质。
- 本报告未运行任何 FK 数值验证（该工作属新版 2.6 切片的独立 URDF FK 对照测试）；本报告只固定名称/数值/结构的对照事实。
