# tennis_rbt 自制机器人建模指南

> 目标：把 MPC + iLQR + Tube 框架部署到「RM-65B 搭建的自制双臂机器人 + 自制底盘」上。
> 本指南说明如何从实物测量出发，构建一份**仿真与真机共用的机器人参照文件**，
> 并完成 Sim2Real 标定。

## 目录

- [1. 总体策略与算法耦合点](#1-总体策略与算法耦合点)
- [2. 交付物与文件位置](#2-交付物与文件位置)
- [3. 阶段 1：实物测量](#3-阶段-1实物测量)
- [4. 阶段 2：构建模型文件](#4-阶段-2构建模型文件)
- [5. 阶段 3：仿真侧集成与验证](#5-阶段-3仿真侧集成与验证)
- [6. 阶段 4：位置模式适配（Dynamixel/总线舵机）](#6-阶段-4位置模式适配dynamixel总线舵机)
- [7. 阶段 5：Sim2Real 标定](#7-阶段-5sim2real-标定)
- [8. 完成度检查清单](#8-完成度检查清单)
- [附录 A：quat ↔ rpy 转换脚本](#附录-aquat--rpy-转换脚本)
- [附录 B：MuJoCo 常见坑](#附录-bmujoco-常见坑)

---

## 1. 总体策略与算法耦合点

### 1.1 为什么工作量比预想小

因为**臂本身是 RM-65B**，现有项目 `src/robot/rm65_model.xml` 已经是同一款臂的精确模型。
所以这是一个**手术式替换**任务，而非从零建模：

- 双臂的 link 几何/惯量/关节参数 → **直接复用**
- 现有 STL 网格（`r_link*.STL`/`l_link*.STL`）→ **直接复用**
- C++ 加速模块的 `kNQ/kNX/kNU = 6/12/6` → **不用改**
- Python 侧 `NQ/NX/NU`、关节名查找、所有 iLQR/MPC/Tube 算法 → **不用改**
- 真正要做的：**自制底盘建模 + 双臂安装接口标定**

### 1.2 算法对模型的硬耦合点（不可改名）

下表是代码按 `name` 字符串查找的对象，**新模型里必须同名存在**，否则运行时报错：

| 类型 | 名称 | 用途 | 查找位置 |
|------|------|------|---------|
| joint | `r_joint1`~`r_joint6` | 右臂驱动关节 | qpos 索引 0~5 |
| joint | `l_joint1`~`l_joint6` | 左臂关节 | qpos 索引 6~11 |
| site | `racket_center` | 末端执行器（球拍面中心） | iLQR 代价函数的 p_ee |
| site | `hit_target` | 击打目标标记 | 可视化 |
| body | `r_racket_body` | 球拍体（法向量来源） | get_ee_normal() |
| body | `ball` | 网球 | 仿真/回放 |
| actuator | `torque_r_joint1`~`6` | 右臂执行器 | ctrl 索引 0~5 |
| actuator | `torque_l_joint1`~`6` | 左臂执行器 | ctrl 索引 6~11 |

> 此外，qpos 布局必须是 `右臂(6) + 左臂(6) + 球freejoint(7) = nq 19`，
> 这由「右臂在前、左臂次之、球最后」的 body 声明顺序保证。

### 1.3 自由区（可随意修改）

| 内容 | 影响 | 精度要求 |
|------|------|---------|
| 底盘外形/质量/惯量 | 仅视觉/碰撞/动力学线性化质量 | 粗估即可（位置模式 PD 主导） |
| 立柱尺寸/高度 | 决定臂的安装高度 → **影响 FK** | 需精确（±5mm） |
| 双臂安装 pos/quat | **最关键**，决定 racket_center 的工作空间 | 需精确（±5mm / ±1°） |
| 头部/相机 | 算法不使用 | 可删除 |
| 球拍几何 | 决定拍面位置/法向量 | 需与实物一致 |

---

## 2. 交付物与文件位置

```
mujoco_sim/
├── src/robot/tennis_rbt.xml              # ★ MJCF 模型（规划管线加载用）
├── assets/tennis_rbt/
│   ├── urdf/tennis_rbt.urdf              # ★ URDF 模型（ROS/RViz canonical 描述）
│   └── robot_measurements.yaml           # ★ 测量数据填空表
└── docs/tennis_rbt_modeling_guide.md     # 本文件
```

### URDF vs MJCF 的关系

两者**运动学等价**（已验证，端点 FK 误差 < 1e-5 m），分工不同：

| 格式 | 角色 | 用途 |
|------|------|------|
| **URDF** | canonical 机器人描述 | ROS / RViz / MoveIt / tf2，人类可读事实来源 |
| **MJCF** | MuJoCo 优化版 | 规划管线（PlanningEnv）加载，额外含 contact/exclude/actuator/site |

**维护原则**：修改其中之一的运动学参数（安装位姿/连杆长度/关节范围），**必须同步另一份**。
MJCF 用 `pos`+`quat`(w,x,y,z)，URDF 用 `origin xyz`+`rpy`（弧度，绕 XYZ 固定轴）。

---

## 3. 阶段 1：实物测量

> 填入 `assets/tennis_rbt/robot_measurements.yaml`，再同步到两份模型。

### 3.A 关键运动学参数（必须精确）

**工具**：卷尺（±1mm）、角尺/量角器（±0.5°）、铅垂线。

#### A1. 双臂底座法兰安装位姿（全表最关键的 12 个数）

测量 `r_base_link1` / `l_base_link1` 在**底盘坐标系**（原点在底盘中心地面投影，X 前 Y 左 Z 上）中的位置与朝向：

```
right_arm_mount:
  xyz: [x, y, z]          # 法兰中心相对底盘原点的位移（米）
  rpy_deg: [roll, pitch, yaw]  # 法兰朝向（度，绕 X,Y,Z 固定轴）
```

**测量方法**：
1. 把机器人摆正，用铅垂线确认 Z 轴竖直
2. 找右臂第一个关节（肩部偏航，`r_joint1`）的旋转中心作为法兰原点
3. 量它到底盘中心（X 方向投影）的 (x, y) 平面距离
4. 量它到地面的高度 z
5. 朝向：法兰底面法线方向。RM-65B 标准安装是绕 Y 轴 -45°（让臂斜向前伸出）

#### A2. 立柱顶面离地高度

```yaml
arm_base_height_z: 1.271   # 平台（双臂安装面）离地高度
```

这决定 MJCF 里 `platform_base_link` 的 `pos` z 分量，和 URDF 里 `platform_joint` 的 `origin xyz` z 分量。

#### A3. 球拍安装确认

确认球拍是否与 RM-65B 一致：
- 手柄从法兰中心沿 Z 轴引出
- 拍面相对法兰绕 Z 轴旋转 90°（法线沿法兰局部 X）
- 拍面中心距法兰 0.25m

若一致 → 不用改模型。若不同 → 测拍面法线相对法兰的朝向，更新 `r_racket_body` 的 `quat`（MJCF）/ `r_racket_joint` 的 `rpy`（URDF）。

### 3.B 次要参数（可粗估）

**工具**：秤（称重）、卷尺。

#### B1. 底盘外形与惯量

```yaml
chassis:
  size_xyz: [长, 宽, 高]      # 米，用 box 建模，不需要 STL
  mass: 8.0                   # kg，称重
  inertia_diag: [Ixx, Iyy, Izz]  # 按均匀长方体估：I=m(b²+c²)/12 等
```

惯量公式（a=长, b=宽, c=高）：`Ixx=m(b²+c²)/12`, `Iyy=m(a²+c²)/12`, `Izz=m(a²+b²)/12`

#### B2. 立柱几何

同上，量截面尺寸和高度。位置模式下惯量只影响动力学线性化质量（A/B 矩阵），不影响 FK 正确性，粗估即可。

#### B3. 底盘世界位姿（仅真机用）

机器人在球场坐标系中的摆放位置，写入 `configs/real_robot.yaml`。

---

## 4. 阶段 2：构建模型文件

### 4.1 MJCF（`src/robot/tennis_rbt.xml`）

已克隆自 `rm65_model.xml`，需要你填的 `TODO_DIY` 标记处：

| TODO_DIY 位置 | 填什么 | 数据来源 |
|--------------|--------|---------|
| `chassis_base` 的 `size`/`mass`/`diaginertia` | 底盘外形/质量/惯量 | `[B].chassis` |
| `pillar_base` 的 capsule `fromto`/`size`/`mass` | 立柱几何/质量 | `[B].pillar` |
| `platform_base_link` 的 `pos` z 分量 | 立柱顶面离地高度 | `[A].arm_base_height_z` |
| `r_base_link1` 的 `pos`/`quat` | 右臂安装位姿 | `[A].right_arm_mount` |
| `l_base_link1` 的 `pos`/`quat` | 左臂安装位姿 | `[A].left_arm_mount` |

**rpy → quat 转换**：MJCF 用 quaternion（w,x,y,z）。见[附录 A](#附录-aquat--rpy-转换脚本)。

**修改示例**（底盘 box 尺寸）：
```xml
<!-- 原（占位） -->
<geom name="col_chassis" type="box" size="0.20 0.20 0.10" .../>
<!-- 改为实测：底盘 0.50×0.45×0.18 → MuJoCo box size 是半尺寸 -->
<geom name="col_chassis" type="box" size="0.25 0.225 0.09" .../>
```

> 注意 MuJoCo `box` 的 `size` 是**半尺寸**（half-extent），实测长 0.50 则填 0.25。

### 4.2 URDF（`assets/tennis_rbt/urdf/tennis_rbt.urdf`）

同样的 `TODO_DIY` 标记，参数等价（URDF 用 `rpy` 弧度，不用 quat）。

**注意事项**：
- URDF 关节限位用**弧度**（不是度）。骨架已预填，改限位时注意转换
- 臂段 visual mesh 路径 `../../rm_65/urdf/meshes/r_link*.STL`（相对 URDF 所在目录）
- MuJoCo 加载 URDF 时会**合并 fixed-joint link**（chassis/pillar/flange/racket_body 会并入父 link），这是正常现象，不影响 FK

### 4.3 验证模型可加载

```bash
cd mujoco_sim
python -c "
import mujoco
from src.utils.mujoco_loader import load_mujoco_model
m = load_mujoco_model('src/robot/tennis_rbt.xml')
print(f'nq={m.nq} nv={m.nv} nu={m.nu}')  # 应为 19/18/12
"
```

---

## 5. 阶段 3：仿真侧集成与验证

### 5.1 用新模型启动 PlanningEnv

`PlanningEnv` 已支持 `model_path` 参数注入，无需改默认值：

```python
from pathlib import Path
from src.ilqt.planning_env import PlanningEnv

env = PlanningEnv(model_path=Path("src/robot/tennis_rbt.xml"))
```

真机入口脚本（`scripts/run_real_robot.py`）若硬编码了模型路径，改为读取该参数。

### 5.2 FK 验证（关键）

零位下末端位置应与**实物测量的拍面中心位置**吻合：

```python
env.reset(np.zeros(6))
p = env.get_ee_pos()
# 对比：用卷尺量机器人零位时拍面中心在底盘坐标系的位置
print(f"仿真零位 racket_center = {p}")
print(f"实测零位 racket_center = [实测值]")
# 偏差应 < 20mm
```

若偏差大 → 检查 A1 安装位姿和 A2 立柱高度是否填对。

### 5.3 可视化检查

```bash
# 用查看器确认底盘/双臂姿态合理
python scripts/tools/rm65_joint_viewer.py  # 改模型路径后
# 或直接：
python -c "
import mujoco, mujoco.viewer
from src.utils.mujoco_loader import load_mujoco_model
m = load_mujoco_model('src/robot/tennis_rbt.xml')
mujoco.viewer.launch_passive(m, mujoco.MjData(m))
"
```

---

## 6. 阶段 4：位置模式适配（Dynamixel/总线舵机）

### 6.1 模型里保留 motor actuator

**不要改** MJCF 里的 `<motor>` 执行器。`PlanningEnv.configure_actuator_mode("position", kp, kd)` 会在运行时把 motor 动态转成 PD 位置执行器（见 `planning_env.py:344`）。这样：

- **仿真规划**：MuJoCo PD 执行器模拟 Dynamixel 的内部 PD
- **真机执行**：Dynamixel 内部 PD + 你的角度指令（`rm_movej_follow` 或等价 API）

### 6.2 需要确认的舵机属性

| 属性 | 说明 | 处理 |
|------|------|------|
| 减速比 | 舵机输出端到关节。多数总线舵机已在输出端 | `robot_measurements.yaml [C].gear_ratio`，通常全 1 |
| 角度方向 | 舵机正方向是否与 URDF 一致 | `[C].sign`，反向填 -1 |
| 零点 | 舵机零位是否对应 URDF 零位 | `[C].zero_offset_rad` |

若方向反向：在真机接口层（`src/real/robot_interface.py` 的对应物）对读到的角度乘 sign、对写出的角度乘 sign。

---

## 7. 阶段 5：Sim2Real 标定

更新 `configs/real_robot.yaml`（可基于现有模板）：

### 7.1 关节零位偏移

```yaml
joint_zero_offset: [0, 0, 0, 0, 0, 0]  # 弧度
```

**标定方法**：让真机摆到一个已知姿态（如零位），读舵机角度 `q_real`，
对比仿真同姿态的 `q_sim`，`offset = q_real - q_sim`。

### 7.2 底座世界位姿

```yaml
# 新增节：机器人在球场坐标系中的摆放
world_pose:
  xyz: [0.0, 0.0, 0.0]      # 底盘中心位置
  rpy_deg: [0.0, 0.0, 0.0]   # 底盘朝向
```

### 7.3 PD 增益匹配

```yaml
position_mode:
  kp: [200, 200, 100, 50, 50, 20]   # 匹配 Dynamixel 硬度
  kd: [20, 20, 10, 5, 5, 2]
```

调参顺序：先调 `kd` 消除震荡，再调 `kp` 提高精度。首次保守（低 kp），逐步提高。

### 7.4 Sim2Real 对齐验证

让真机摆几个已知关节角（如 `q=[0, -45°, 90°, 0, 0, 0]`），对比：
1. 仿真 `env.get_ee_pos()` 的 racket_center 位置
2. 实测拍面中心位置（用尺量）

偏差 < 30mm 即可接受（考虑球拍安装公差）。偏差大 → 回到阶段 1 复查 A1。

---

## 8. 完成度检查清单

- [ ] **测量**：A1 双臂安装位姿（12 个数）已实测
- [ ] **测量**：A2 立柱顶面离地高度已实测
- [ ] **测量**：B1 底盘尺寸/质量已测
- [ ] **MJCF**：所有 `TODO_DIY` 已填入实测值
- [ ] **URDF**：所有 `TODO_DIY` 已同步
- [ ] **验证**：模型可加载，`nq=19 nv=18 nu=12`
- [ ] **验证**：零位 FK 与实测拍面位置偏差 < 20mm
- [ ] **验证**：PlanningEnv 集成测试通过（FK/雅可比/step/IK）
- [ ] **标定**：`joint_zero_offset` 已标定
- [ ] **标定**：`configs/real_robot.yaml` 底座世界位姿已填
- [ ] **标定**：PD 增益已匹配 Dynamixel
- [ ] **Sim2Real**：多姿态拍面位置偏差 < 30mm

---

## 附录 A：quat ↔ rpy 转换脚本

MJCF 用 quaternion（w,x,y,z），URDF 用 rpy（弧度，绕 XYZ 固定轴）。互转：

```bash
cd mujoco_sim
python -c "
from scipy.spatial.transform import Rotation as R
import numpy as np

# quat(w,x,y,z) → rpy(度)
def q2rpy(w,x,y,z):
    r = R.from_quat([x,y,z,w])
    return r.as_euler('xyz', degrees=True)

# rpy(度) → quat(w,x,y,z)
def rpy2q(roll,pitch,yaw):
    r = R.from_euler('xyz', [roll,pitch,yaw], degrees=True)
    x,y,z,w = r.as_quat()
    return [w,x,y,z]

# 示例：右臂安装 quat → rpy
print('quat[0.92388,0,-0.382683,0] →', q2rpy(0.92388,0,-0.382683,0))
# 输出: [0, -45, 0]

# 示例：rpy(0,-45,180) → quat
print('rpy(0,-45,180) →', rpy2q(0,-45,180))
"
```

骨架文件里的 quat↔rpy 已由该脚本精确转换，只有你改了安装角度时才需要重算。

---

## 附录 B：MuJoCo 常见坑

1. **`box` size 是半尺寸**：实测长 0.50m 填 `size="0.25 ..."`，不是 0.50
2. **`range` 用度**：MuJoCo 关节 `range="-178 178"` 是度（自动转弧度），URDF `<limit>` 用弧度
3. **URDF 加载会合并 fixed-joint link**：chassis/pillar/flange/racket_body 在 MuJoCo 里看不到是正常的，不影响 FK
4. **中文路径**：始终用 `load_mujoco_model()`（`src/utils/mujoco_loader.py`），不要直接 `from_xml_path()`
5. **关节顺序 = body 声明顺序**：右臂必须在左臂之前声明，球必须在最后，保证 qpos 布局为 `[右臂6, 左臂6, 球7]`
6. **碰撞位掩码**：底盘用 `class="col_body"`（contype=2），臂段用 `class="col_arm"`（contype=4），不要混用以免自碰撞阻止运动
