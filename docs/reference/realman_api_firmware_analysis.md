# 睿尔曼 RM-65B API 与固件版本分析报告

> 分析日期：2026-06-25
> 分析对象：`reference/Tennis-Robot/Realman_Script/notes/` 下的三方信息源
> 关联代码：`src/real/robot_interface.py`

---

## 1. 概述

### 1.1 三方信息源

| 源 | 位置 | 版本/日期 | 说明 |
|---|---|---|---|
| **PDF V1.9** | `notes/睿尔曼机械臂接口函数说明(Python)V1.9.pdf` | 2024-11-05 | 厂家随机械臂交付的纸质文档电子版 |
| **Notes (Web Docs)** | `notes/00_*.md` ~ `notes/14_*.md`（15 个文件） | 提取自官方开发者中心 | 从 `develop.realman-robotics.com` 逐页提取的 Markdown |
| **GitHub SDK** | [RealManRobot/RM_API2](https://github.com/RealManRobot/RM_API2) `Python/Robotic_Arm/` | 最新主干 | 官方 Python 二次开发包源码 |

### 1.2 核心结论

| 源 | 准确性 | 结论 |
|---|---|---|
| **PDF V1.9** | ❌ **完全过时** | 记录的是**第一代 API**（CamelCase 命名，`robotic_arm_package.Arm`），与当前 SDK 属于不同代际，不可作为开发参考 |
| **Notes (00-14)** | ✅ **准确** | 使用正确的**第二代 API2**（snake_case，`Robotic_Arm.RoboticArm`），与 GitHub SDK 一致 |
| **项目代码** | ✅ **正确** | `robot_interface.py` 已使用 API2，与当前 SDK 匹配，无需修改 |

---

## 2. API 代际差异

### 2.1 总览

| 维度 | PDF V1.9（旧 API） | 当前 API2（新） |
|---|---|---|
| **pip 包名** | 无（手动安装 `robotic_arm_package`） | `Robotic_Arm` |
| **导入路径** | `from robotic_arm_package.robotic_arm import *` | `from Robotic_Arm.rm_robot_interface import *` |
| **主类名** | `Arm` | `RoboticArm` |
| **连接模型** | 构造器直连：`Arm(RM65, "192.168.1.18")` | 工厂方法：`RoboticArm(mode)` + `.rm_create_robot_arm(ip, port)` |
| **函数命名** | CamelCase：`Movej_CANFD`、`Get_Joint_Degree` | snake_case：`rm_movej_canfd`、`rm_get_joint_degree` |
| **断开连接** | `Arm_Socket_Close()` | `rm_delete_robot_arm()` |
| **线程模式** | 无选择 | `rm_thread_mode_e` 枚举（单/双/三线程） |

### 2.2 包与导入差异

```python
# ❌ PDF V1.9（旧 API — 已废弃）
from robotic_arm_package.robotic_arm import *
robot = Arm(RM65, "192.168.1.18")

# ✅ 当前 API2（项目使用 — 正确）
from Robotic_Arm.rm_robot_interface import *
arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)
```

### 2.3 连接模型差异

| 方面 | PDF V1.9 | API2 |
|---|---|---|
| 连接方式 | 构造器参数 `(型号, IP)` | 构造器仅接受线程模式，连接由 `rm_create_robot_arm` 完成 |
| 返回值 | `Arm` 实例 | `rm_robot_handle`（含 `id` 字段标识连接） |
| 多机械臂 | 需要多个 `Arm` 实例 | 单 `RoboticArm` 可管理多个 `rm_robot_handle` |
| 日志级别 | 无参数 | `rm_create_robot_arm(ip, port, level=3)` 可设日志等级 |

---

## 3. 项目使用的 14 个 API 映射表

项目 `src/real/robot_interface.py` 使用以下 14 个 SDK 函数。以下是旧→新完整对照：

### 3.1 改名 API（PDF V1.9 → API2）

| # | 项目代码（API2 ✅） | PDF V1.9 旧名 | 变更类型 |
|---|---|---|---|
| 1 | `RoboticArm(mode)` | `Arm(model, ip)` | 类名 + 构造器完全不同 |
| 2 | `rm_create_robot_arm(ip, port)` | （构造器内完成） | 新增独立连接方法 |
| 3 | `rm_delete_robot_arm()` | `Arm_Socket_Close()` | 改名 |
| 4 | `rm_get_joint_degree()` | `Get_Joint_Degree()` | 改名（snake_case + `rm_` 前缀） |
| 5 | `rm_movej_follow(joint)` | `Movej_Follow(joint)` | 改名 |
| 6 | `rm_movej_canfd(joint, follow, expand, trajectory_mode, radio)` | `Movej_CANFD(joint, follow, expand)` | 改名 **+ 2 个新参数** |
| 7 | `rm_set_collision_state(stage)` | `Set_Collision_Stage(stage)` | 改名 |
| 8 | `rm_set_self_collision_enable(enable)` | `Set_Self_Collision_Enable(enable)` | 改名 |
| 9 | `rm_set_arm_max_line_speed(speed)` | `Set_Arm_Line_Speed(speed)` | 改名 |
| 10 | `rm_set_arm_max_line_acc(acc)` | `Set_Arm_Line_Acc(acc)` | 改名 |
| 11 | `rm_set_arm_stop()` | `Move_Stop_Cmd()` | 改名 |

### 3.2 V1.9 后新增 API（PDF 中不存在）

| # | 项目代码（API2 ✅） | 说明 | PDF V1.9 状态 |
|---|---|---|---|
| 12 | `rm_set_arm_slow_stop()` | 缓停（可恢复） | **不存在** — V1.9 后新增 |
| 13 | `rm_set_controller_torque_limit(limit)` | 控制器力矩限制 | **不存在** — V1.9 后新增 |
| 14 | `rm_set_avoid_singularity_mode(mode)` | 奇异点规避模式 | **不存在** — V1.9 后新增 |

### 3.3 `rm_movej_canfd` 参数变更详情

这是项目使用的最核心控制接口，参数从 3 个扩展到 5 个：

```python
# ❌ PDF V1.9（3 参数）
Movej_CANFD(joint, follow, expand)

# ✅ 当前 API2（5 参数）
rm_movej_canfd(joint, follow, expand=0, trajectory_mode=0, radio=0)
```

| 参数 | 类型 | V1.9 | API2 | 说明 |
|---|---|---|---|---|
| `joint` | `list[float]` | ✅ | ✅ | 关节目标角度（°） |
| `follow` | `bool` | ✅ | ✅ | 是否高跟随 |
| `expand` | `float` | ✅ | ✅ | 扩展关节位置（默认 0） |
| `trajectory_mode` | `int` | ❌ | ✅ 新增 | 轨迹模式（0=默认） |
| `radio` | `int` | ❌ | ✅ 新增 | 平滑系数（0-100） |

> 项目 `robot_interface.py:155-160` 已使用 5 参数版本（含 `trajectory_mode` 和 `radio`），正确。

---

## 4. 固件版本分析

### 4.1 当前固件信息（实测）

来源：`notes/机械臂查询函数测试结果.md` 中 `rm_get_arm_software_info()` 返回值。

| 组件 | 版本 | 编译日期 |
|---|---|---|
| 产品型号 | RM65-BI | — |
| **控制器 (ctrl)** | **V1.6.4** | 2024/10/15 20:30:10 |
| **规划层 (plan)** | **V1.6.4** | 2024/10/15 20:30:10 |
| 算法库 | 1.4.8 | — |
| C API | 1.1.3 | — |
| 动力学模型 | 2 | — |

### 4.2 最新固件版本

来源：[官方版本变更说明](https://develop.realman-robotics.com/robot/releaseNotes/releaseNotes/)。

| 组件 | 最新版本 | 更新时间 |
|---|---|---|
| **控制器** | **V1.7.5** | 2026/04/29 |
| API2 | V1.1.5 | 2026/04/29 |
| 关节驱动器（常规） | Vd5.1.0 | — |
| 关节驱动器（多圈） | Ve5.1.0 | — |

### 4.3 API 失败诊断

来源：`notes/机械臂查询函数测试结果.md` + `notes/机械臂查询函数测试结果（旧版API）.md`。

实测结果（共 10 个函数，7 成功 / 2 失败 / 1 不存在）：

| API | 状态码 | 症状 | 诊断 |
|---|---|---|---|
| `rm_get_current_arm_state()` | **163 / 165** | `parsing error`，返回全零数据 | ❌ **固件-SDK 协议不匹配** |
| `rm_get_arm_all_state()` | **-3** | `parsing error`，返回全零数据 | ❌ **固件-SDK 协议不匹配** |
| `rm_get_arm_plan_num()` | — | 函数不存在于当前 SDK | ⊘ 已移除 |
| `rm_get_joint_degree()` | 0 | 返回正确关节角度 | ✅ 正常 |
| `rm_get_current_joint_temperature()` | 0 | 返回正确温度 | ✅ 正常 |
| `rm_get_current_joint_current()` | 0 | 返回正确电流 | ✅ 正常 |
| `rm_get_current_joint_voltage()` | 0 | 返回正确电压 | ✅ 正常 |
| `rm_get_init_pose()` | 0 | 返回正确初始位姿 | ✅ 正常 |
| `rm_get_install_pose()` | 0 | 返回正确安装角度 | ✅ 正常 |
| `rm_get_arm_software_info()` | 0 | 返回正确版本信息 | ✅ 正常 |

**根因分析**：

两个失败的函数（`rm_get_current_arm_state` 和 `rm_get_arm_all_state`）均报 `parsing error`，说明 SDK 的 ctypes wrapper 期望的数据包结构与固件实际发送的不一致。这是典型的**固件版本落后于 SDK 版本**导致的协议格式差异。

- 固件 V1.6.4 编译于 2024/10/15
- 当前 SDK（API2 V1.1.x）适配的最低固件版本为 V1.7.x
- V1.7.0 固件变更日志中明确提到"适配控制器 1.6.0 主动上报接口"和"运动接口优化"——数据包格式在 V1.6.x → V1.7.x 之间发生了变更

**影响范围**：
- `rm_get_current_arm_state()` 失败 → 项目无法通过此接口一次性获取关节角度+位姿+错误码
- `rm_get_arm_all_state()` 失败 → 项目无法通过此接口获取全状态（温度/电流/电压/错误码合并）
- **替代方案**：项目当前用 `rm_get_joint_degree()` 独立读取关节角度，不受影响

### 4.4 V1.6.4 → V1.7.5 关键变更

从官方版本变更说明中提取与项目相关的变更：

**V1.7.4**：
- 新增手动关闭碰撞解除模式（影响 `rm_set_collision_state` 行为）
- 新增电流环拖动示教功能
- 修复实时调速接口暂停后关节掉使能问题
- 修复 `movep_follow` 四元数位姿异常
- 奇异点保护功能从管理员权限开放至常规安全配置

**V1.7.5**：
- 新增 RM65-B-V 等视觉臂型号适配
- 新增力控示教安全校验（错误码 `0x1017`）
- 一键升级优化（控制器/关节/末端固件一次重启完成）
- WiFi 模块默认关闭（V1.7.5 起）
- RM65 关节 4 硬限位统一调整为 -183° ~ 183°
- 通讯丢帧检测机制优化

---

## 5. 固件升级指南

### 5.1 前置条件

1. **联系厂家技术支持**获取 `.realman` 固件文件
   - 说明当前版本：控制器 V1.6.4，C API 1.1.3
   - 目标版本：V1.7.5（或厂家推荐的最新稳定版）
   - 机械臂型号：**RM65-BI**
2. **确认驱动器版本兼容性**：V1.7.5 要求驱动器 Vd5.1.0（常规）/ Ve5.1.0（多圈），升级前由厂家确认
3. **备份当前配置**：记录关节限位、坐标系、碰撞参数等自定义设置（升级可能重置）
4. **机械臂状态**：确保机械臂处于零位，无负载，电源稳定

### 5.2 公开资源调查结果

对 [RealManRobot GitHub 组织](https://github.com/RealManRobot)全部 27 个公开仓库进行了扫描，**确认固件文件不通过 GitHub 分发**。

**仓库分类**：

| 类别 | 仓库 | 内容 | 含固件？ |
|---|---|---|---|
| SDK | `RM_API2`、`RM_API` | Python/C/C++ 二次开发包源码 | ❌ |
| 文档 | `Dev_Center` | 用户手册 PDF（`RobotGen3/download/manual/RM/`） | ❌ |
| 模型 | `rm_models`、`rm_joints_model` | 3D 模型 / URDF / MATLAB | ❌ |
| ROS | `rm_robot`、`ros2_rm_robot` | ROS / ROS2 驱动包 | ❌ |
| AI/应用 | 其余 20 个 | 视觉 / 抓取 / 标定等应用 | ❌ |

**官方下载页声明**（[中文](https://develop.realman-robotics.com/robot/download/redevelopment/) / [英文](https://develop.realman-robotics.com/en/robot/download/redevelopment/)）：

> 需要升级至最新版控制器，**请联系技术支持提供帮助**。

`.realman` 固件文件**不通过 GitHub / PyPI / 官网下载页公开分发**，必须联系厂家技术支持获取。

**Dev_Center 仓库可用资源**：

| 路径 | 内容 | 链接 |
|---|---|---|
| `RobotGen3/download/manual/RM/` | RM65 & RM75 系列用户使用说明书 V1.2.0（PDF, 8MB） | [GitHub](https://github.com/RealManRobot/Dev_Center/tree/main/RobotGen3/download/manual/RM) |
| `RobotGen4/download/manual/` | 第四代机械臂用户手册 | [GitHub](https://github.com/RealManRobot/Dev_Center/tree/main/RobotGen4/download/manual) |

### 5.3 示教器说明

睿尔曼示教器为 **WEB 内置式**，是控制器固件的一部分，**无独立软件需要下载安装**。

- **访问方式**：浏览器打开 `http://<机械臂IP>`（如 `http://192.168.1.18`）
- **升级方式**：随控制器固件一同升级，无需单独更新
- **升级后刷新**：按 `Ctrl+F5` 强制刷新浏览器，清除旧版页面缓存

### 5.4 升级步骤（示教器）

> 来源：[官方系统升级文档](https://develop.realman-robotics.com/robot/teachingPendant/systemUpgrade/)

1. 将厂家提供的 `.realman` 文件下载到本地电脑
2. 打开 WEB 示教器 → **配置 → 机械臂配置 → 版本信息**
3. 点击 `选择文件` 按钮，选择 `.realman` 固件文件
4. 点击 `开始升级`，等待升级进度条完成（文件较大时约 4-5 分钟）
5. 升级成功后，示教器弹窗提示，控制器发出连续提示音
6. **重启控制器**
7. 重新打开 WEB 示教器，按 **Ctrl+F5** 强制刷新浏览器清缓存

> ⚠️ **注意**：升级过程中切勿断电或关闭示教器，否则可能导致控制器损坏。

### 5.5 升级后验证

升级完成后，运行以下验证脚本确认 API 恢复正常：

```python
# 验证脚本 — 可通过 scripts/tools/test_real_robot/ 运行
from Robotic_Arm.rm_robot_interface import *

arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
handle = arm.rm_create_robot_arm("192.168.1.18", 8080)

# 1. 确认固件版本已更新
info = arm.rm_get_arm_software_info()
print(f"控制器版本: {info[1]['ctrl_info']['version']}")  # 应显示 V1.7.x

# 2. 测试之前失败的 API
state = arm.rm_get_current_arm_state()
print(f"rm_get_current_arm_state 状态码: {state[0]}")  # 应为 0
print(f"关节角度: {state[1]['joint']}")  # 应有实际数值

all_state = arm.rm_get_arm_all_state()
print(f"rm_get_arm_all_state 状态码: {all_state[0]}")  # 应为 0

# 3. 确认项目核心 API 仍正常
joint = arm.rm_get_joint_degree()
print(f"rm_get_joint_degree 状态码: {joint[0]}")  # 应为 0

arm.rm_delete_robot_arm()
```

**验证清单**：

| 检查项 | 预期结果 |
|---|---|
| `rm_get_arm_software_info()` 版本 | ctrl_info.version ≥ V1.7.0 |
| `rm_get_current_arm_state()` | 状态码 0，返回非零关节数据 |
| `rm_get_arm_all_state()` | 状态码 0，返回非零状态数据 |
| `rm_get_joint_degree()` | 状态码 0（升级前后均正常） |
| `rm_movej_follow()` | 状态码 0，机械臂正常运动 |
| 关节限位设置 | 确认未被重置（RM65 关节 4 可能变为 -183°~183°） |

### 5.6 版本兼容性对照表

| 组件 | 项目所需版本 | 机器人当前 | 最新 | 升级后预期 |
|---|---|---|---|---|
| 控制器固件 | ≥ V1.7.0（API2 要求） | V1.6.4 ❌ | V1.7.5 | V1.7.5 ✅ |
| C API / API2 库 | V1.1.5 | V1.1.3 | V1.1.5 | V1.1.5 |
| Python SDK | 最新 (`pip install Robotic_Arm`) | 未安装 | 最新 | 安装最新 |
| 关节驱动器 | Vd5.1.0 / Ve5.1.0 | 未知 | Vd5.1.0 | 由厂家确认 |

### 5.7 联系技术支持清单

联系睿尔曼技术支持获取 `.realman` 固件文件时，提供以下信息以便厂家匹配合适的固件版本：

| 信息项 | 值 | 获取方式 |
|---|---|---|
| 产品型号 | `RM65-BI` | `rm_get_arm_software_info()` → `product_version` |
| 当前控制器版本 | `V1.6.4`（2024/10/15 编译） | `ctrl_info.version` / `ctrl_info.build_time` |
| 当前规划层版本 | `V1.6.4`（2024/10/15 编译） | `plan_info.version` / `plan_info.build_time` |
| 算法库版本 | `1.4.8` | `algorithm_info.version` |
| C API 版本 | `1.1.3` | 连接时控制台输出 `current c api version` |
| 动力学模型 | `2` | `dynamic_info.model_version` |
| 目标固件版本 | `V1.7.5`（或厂家推荐最新稳定版） | [版本变更说明](https://develop.realman-robotics.com/robot/releaseNotes/releaseNotes/) |
| 问题描述 | `rm_get_current_arm_state` / `rm_get_arm_all_state` 返回 parsing error（错误码 163/165/-3），疑似固件-SDK 协议不匹配 | `notes/机械臂查询函数测试结果.md` |

**联系渠道**：

| 渠道 | 链接 |
|---|---|
| 开发者中心 | [develop.realman-robotics.com](https://develop.realman-robotics.com) |
| 官方技术论坛 | [bbs.realman-robotics.cn](https://bbs.realman-robotics.cn) |
| GitHub（SDK 源码 + 示例） | [github.com/RealManRobot](https://github.com/RealManRobot) |

---

## 6. 结论与建议

### 6.1 PDF V1.9 文档

**建议归档**。PDF V1.9 记录的是第一代 API（`robotic_arm_package.Arm`，CamelCase 命名），与项目使用的第二代 API2（`Robotic_Arm.RoboticArm`，snake_case 命名）完全不兼容。保留仅作为历史参考，不用于开发。

### 6.2 Notes 文档

**准确但覆盖不全**。Notes（00-14 共 15 个文件）准确反映了当前 API2，与 GitHub SDK 一致。但官方 Web 文档现有 30+ 个 API 分类，Notes 缺失的包括：

| 缺失分类 | 项目是否使用 |
|---|---|
| 系统安装方式配置 (`installPos`) | 否 |
| 通用扩展关节配置 (`expandControl`) | 否 |
| 升降机构配置 (`liftControl`) | 否 |
| 末端六维力配置 (`force`) | 否 |
| 电子围栏和虚拟墙 (`electronicFenceConfig`) | 否 |
| 透传力位混合控制 (`forcePositionControl`) | 否 |
| 全局路点管理 (`globalWaypointManage`) | 否 |
| 末端工具夹爪配置 (`gripperControl`) | 否 |
| 五指灵巧手配置 (`handControl`) | 否 |
| 末端生态协议 (`rmPlus`) | 否 |
| 控制器/末端 IO 配置 (`controllerIOConfig` / `effectorIOConfig`) | 否 |
| 关节配置 (`jointsConfig` / `jointsConfigQuery`) | 否 |
| Modbus 配置 (`modbusConfig`) | 否 |
| 在线编程文件管理 (`projectManagement`) | 否 |
| UDP 主动上报 (`udpConfig`) | 否 |

> 项目仅使用基础运动控制 + 状态查询 + 安全配置 API，上述缺失分类均未使用，**不影响当前开发**。如后续需要夹爪/力控/IO 等功能，可按需从 [官方 Web 文档](https://develop.realman-robotics.com/robot/apipython/) 补充。

### 6.3 固件升级

**建议尽快升级**。当前固件 V1.6.4 导致 `rm_get_current_arm_state()` 和 `rm_get_arm_all_state()` 两个状态查询函数不可用（parsing error）。虽然项目当前用 `rm_get_joint_degree()` 绕过了此问题，但升级后可获得：

1. 修复状态查询 API（`rm_get_current_arm_state` / `rm_get_arm_all_state`）
2. 支持力矩限制 API（`rm_set_controller_torque_limit`）
3. 支持奇异点规避 API（`rm_set_avoid_singularity_mode`）
4. 支持缓停 API（`rm_set_arm_slow_stop`）
5. WiFi 默认关闭（减少干扰，适合工业环境）
6. 通讯丢帧检测优化

### 6.4 项目代码

**无需修改**。`src/real/robot_interface.py` 已正确使用 API2 接口，与当前 GitHub SDK 和 Web 文档一致。待固件升级后，所有 14 个 API 调用均可正常工作。

---

## 7. 固件升级后验证记录

> 验证日期：2026-06-26
> 测试脚本：`test_real_robot/01_connect_disconnect.py`
> 配置修复：`config.py` `_flatten()` 别名映射（`robot.ip` → `robot_ip`）

### 7.1 环境

| 项目 | 升级前 | 升级后 |
|---|---|---|
| 产品型号 | RM65-BI | RM65-BI（未变） |
| 控制器版本 | V1.6.4 | **V1.7.5-b570c1e** |
| 算法库版本 | 1.4.8 | **1.6.0-c64b4fd9** |
| 规划层版本 | V1.6.4 | **V1.7.5-05590ea** |
| 动力学模型 | 2 | 2（未变） |
| C API 版本 | 1.1.3 | **v1.1.5** |
| 连接方式 | — | 网口 192.168.1.19:8080（右臂） |

### 7.2 验证结果

测试脚本：`01_connect_disconnect.py` + `01b_firmware_api_verify.py`

| 测试项 | 结果 | 说明 |
|---|---|---|
| TCP 连接 | ✅ | 连接成功 |
| 固件版本读取 | ✅ | 5 字段全部读取，3 项已升级 |
| 关节角度/温度/电流/电压 | ✅ | 4 个 API 均正常 |
| 初始/安装位姿 | ✅ | 2 个 API 均正常 |
| `rm_get_current_arm_state` | ✅ **已修复** | 升级前 ret=163/165（parsing error） |
| `rm_get_arm_all_state` | ✅ **已修复** | 升级前 ret=-3（parsing error） |
| `rm_get_arm_plan_num` | ⊘ 跳过 | SDK 中不存在此函数 |
| 控制器安全配置 | ⚠️ | `rm_set_controller_torque_limit` 缺失 |

查询 API 汇总：9 成功 / 0 失败 / 1 跳过

### 7.3 待解决：rm_set_controller_torque_limit 缺失

`_configure_safety()` 调用此 API 时报 `'RoboticArm' object has no attribute`。
连接不受影响（被 try/except 捕获），但 Layer 1 力矩硬限制未生效。

可能原因：PyPI 包版本未导出此方法 / API 名称在 API2 中已变更。
修复前避免高速/大负载运动。

### 7.4 下一步

1. ~~运行 `01b_firmware_api_verify.py` 全面验证~~ ✅ 完成
2. ~~确认 `rm_get_current_arm_state` / `rm_get_arm_all_state` 修复~~ ✅ 已修复
3. 解决 `rm_set_controller_torque_limit` 缺失
4. ~~运行 `06_safety_config_verify.py` 确认安全参数回读~~ ✅ 3/4 通过（力矩限制 API 缺失）

### 7.6 流式运动测试

测试脚本：`04_send_zero_pose.py`
测试方式：100Hz 流式插值，`rm_movej_follow` 关节空间指令

| 项目 | 值 |
|---|---|
| 起始角度（度） | [-38.78, 12.34, -10.88, 22.34, 27.56, -0.86] |
| 目标角度（度） | [0, 0, 30, 0, 0, 0]（J3=30° 安全位姿） |
| 最大变化 | 38.8° |
| 运动时长 | 1.0s（100 步 × 100Hz） |
| 最终角度（度） | [-0.06, -0.05, 29.9, 0.02, 0.01, 0.02] |
| 最大跟踪误差 | **0.10°**（J3） |

rm_movej_follow 流式关节跟踪精度优秀（< 0.1°）。

**测试过程中发现并修复的问题**:
1. `pre_motion_check` 单点检查 → 15 点路径采样
2. 球拍包络球 radius 0.13 → 0.07（原值 2× 过大导致误报）
3. 奇异性检测从默认行为中移除（关节空间运动无需 IK，奇异性不影响）
4. 默认目标位姿从全零改为 J3=30°（避开肘部奇异区）
5. YES 确认兼容大小写（4 个脚本共 6 处）

### 7.7 多关节同时运动测试

测试脚本：`05_send_joint_command.py`
测试方式：交互式输入 6 关节目标角度，100Hz 流式插值

| 项目 | 值 |
|---|---|
| 起始角度（度） | [0, 0, 30, 0, 0, 0]（安全位姿） |
| 目标角度（度） | [30, 30, 30, 30, 30, 30] |
| 最大变化 | 30.0° |
| 运动时长 | 1.0s（100 步 × 100Hz） |
| 最终角度（度） | [29.99, 29.97, 30.06, 30.02, 30.05, 29.94] |
| 最大跟踪误差 | **0.06°**（J6） |

6 关节同时运动，rm_movej_follow 跟踪精度稳定 < 0.1°。

### 7.8 缓停与急停测试

测试脚本：`07_emergency_stop.py`
测试方式：J6 ±5° 正弦摆动 (0.2Hz)，运动中触发停止

| 测试项 | 接口 | 运动时长 | 结果 |
|---|---|---|---|
| 缓停 | `rm_set_arm_slow_stop` | 5s | ✅ 平滑停止 |
| 急停 | `rm_set_arm_stop` | 3s | ✅ 立即停止 |

缓停后机械臂可继续运动（回起点插值正常）；急停后机械臂锁定，需示教器手动复位或重新上电。

### 7.9 连续正弦波运动测试

测试脚本：`08_full_motion_test.py`
测试方式：J1 ±5° 正弦波，周期 2s，持续 10s，100Hz 流式指令

| 项目 | 值 |
|---|---|
| 起始角度（度） | [22.89, 12.9, 29.92, 29.85, 39.01, 1.83] |
| 振幅 | ±5.0°（J1） |
| 最大跟踪误差 | ~1.0°（首周期瞬态） |
| 稳态跟踪误差 | 0.2~0.7° |
| 峰值速度 | ±26°/s（0.45 rad/s） |

rm_movej_follow 在连续正弦跟踪中存在~0.5° 相位滞后（控制器内部规划平滑所致）。
误差在 ±5° 振幅下约 10%，MPC 规划模型已包含此跟随动态。

### 7.5 查询频率压测

测试脚本：`02_read_joints.py --benchmark 500`
测试方式：500 次无 sleep 紧密循环 `rm_get_joint_degree()`

| 指标 | 值 |
|---|---|
| 平均吞吐 | **109 Hz** |
| min | 2.85 ms |
| median | 8.55 ms |
| mean | 9.19 ms |
| p95 | 16.65 ms |
| p99 | 20.07 ms |
| max | 34.22 ms |
| std | 3.87 ms |

min 2.85ms 接近文档标称 2ms 硬件极限；median 8.55ms 比 01b 单次冷调用 16ms 快一倍（TCP/Python 热路径）。
109 Hz 吞吐远超 MPC 控制需求（exp9 已证明 10Hz 观测无命中率退化）。

---

## 参考链接

- [官方开发者中心](https://develop.realman-robotics.com/)
- [Python API 文档](https://develop.realman-robotics.com/robot/apipython/getStarted/)
- [版本变更说明](https://develop.realman-robotics.com/robot/releaseNotes/releaseNotes/)
- [系统升级指南](https://develop.realman-robotics.com/robot/teachingPendant/systemUpgrade/)
- [GitHub SDK 仓库](https://github.com/RealManRobot/RM_API2)
- [PyPI 包](https://pypi.org/project/robotic-arm/)（`pip install Robotic_Arm`）

---

## 8. Algo FK vs MuJoCo PlanningEnv 差异记录

> 测试脚本：`scripts/tools/test_real_robot/09_algo_fk_ik_verify.py`
> 测试日期：2026-07-02
> 分支：`feature/test-new-api`

### 8.1 Algo DH 参数（真机内部模型）

从 `rm_algo_get_dh()` 读取的算法层 DH 参数：

| 参数 | 值 | 单位 |
|------|-----|------|
| d | [0.2405, 0, 0, 0.21, 0, 0.144] | m |
| a | [0, 0, 0.256, 0, 0, 0] | m |
| alpha | [0, 90, 0, 90, -90, 90] | ° |
| offset | [0, 90, 90, 0, 0, 0] | ° |

### 8.2 FK 位置残差

| 关节角组（°） | 位置残差 | 姿态残差 | 原因 |
|--------------|---------|---------|------|
| [0,0,0,0,0,0] 零位 | 0mm | 0° | 基准一致 |
| [0,20,70,0,90,0] 文档示例 | 25mm | 0° | TCP 定义差异 |
| [0,-30,90,0,90,0] 高位 | 24mm | 0° | TCP 定义差异 |
| [30,0,45,0,45,0] 侧伸 | 299mm | 60° | J1 大角度暴露基座旋转差异 |
| [0,0,-90,0,-90,0] IK示例 | 25mm | 0° | TCP 定义差异 |

### 8.3 关键证据

- **相对旋转角全部一致**：5 组姿态的 algo/mj 相对旋转角完全匹配 → 运动学结构同构
- **IK 回环全部通过**：最大关节角差 < 0.00006° → Algo 内部自洽
- **IK 全解 8 个**，最优解选择正确
- **坐标变换往返误差 < 3e-8**

差异仅在参考点位置（25mm TCP 偏移）和基座朝向定义（J1 大角度时暴露），不影响关节空间控制。

### 8.4 对 MPC 部署的影响

- **25mm 沿法兰 Z 轴（球拍方向）**：racket_center site 的世界坐标可能偏移 25mm
- 网球有效接触区 30~40mm，25mm 偏移占 60~80% → 可能影响击球时机和接触点
- **J1=30° 的 299mm 异常**：网球挥拍 J1 通常在 0°~20°，影响较小
- **关节空间控制不受影响**：MPC 规划关节角 → rm_movej_follow 下发 → 控制器用自己的 FK，关节角本身准确

### 8.5 标定计划

真机部署前的标定步骤：

1. **实测 racket_center 位置**：用真机示教到已知关节角，测量球拍面中心实际位置
2. **对比 MuJoCo FK**：计算 MuJoCo racket_center 在相同关节角下的世界坐标
3. **调整 MuJoCo 模型**：根据偏差修改 `rm65_model.xml` 中 `racket_center` site 的 Z 偏移
4. **或在 PlanningEnv 补偿**：添加常量偏移修正（简单但不够通用）
5. **重新运行 09 脚本**：验证标定后 FK 残差 < 1mm

---

## 9. _connect.py 包络球构造 bug

> 发现脚本：`10_algo_safety_verify.py`
> 测试日期：2026-07-02

**问题**：`_connect.py init_algo()` 用 `rm_tool_sphere_t(x=, y=, z=, radius=)` 构造包络球，
SDK 静默忽略 x/y/z 参数 → centrePoint 始终为 [0,0,0]。

**正确用法**：`rm_tool_sphere_t(centrePoint=(x, y, z), radius=)`

**影响**：3 个球拍包络球全部堆在法兰中心，球拍柄(z=0.12)和拍面(z=0.25)未建模。
自碰撞检测只保护法兰区域 → 拍面/拍柄碰机身不会报警（假阴性）。

**当前风险**：低（测试脚本运动幅度小，控制器 Layer 1 碰撞检测独立生效）。
**MPC 部署前必须修**：挥拍幅度大，假阴性有安全隐患。

**修复方案**（`_connect.py:138-141`，改 3 行）：

```python
# 修复前（buggy）：
rm_tool_sphere_t(x=0, y=0, z=0.0, radius=0.03)
rm_tool_sphere_t(x=0, y=0, z=0.12, radius=0.02)
rm_tool_sphere_t(x=0, y=0, z=0.25, radius=0.07)
# 修复后：
rm_tool_sphere_t(centrePoint=(0, 0, 0.0), radius=0.03)
rm_tool_sphere_t(centrePoint=(0, 0, 0.12), radius=0.02)
rm_tool_sphere_t(centrePoint=(0, 0, 0.25), radius=0.07)
```

> ✅ **已修复（2026-07-03）**：`centrePoint=` 构造已应用，同时将 `except Exception: pass`
> 改为 `logging.warning` 防止未来类似 bug 被静默吞掉。

---

## 10. 关节级限位与 config 不一致

> 测试脚本：`11_joint_config_query.py`
> 测试日期：2026-07-02

### 10.1 真机关节级限位快照

| 关节 | 关节级 max_pos(°) | 驱动级 max_pos(°) | 关节级 max_speed(RPM) | 关节级 max_acc(RPM/s) |
|------|-------------------|-------------------|-----------------------|-----------------------|
| J1 | 178 | 183 | 180 | 600 |
| J2 | 130 | 135 | 180 | 600 |
| J3 | 135 | 140 | 225 | 600 |
| J4 | 178 | 360 | 225 | 600 |
| J5 | 128 | 133 | 225 | 600 |
| J6 | 360 | 365 | 225 | 600 |

### 10.2 位置限位层级异常（config > 关节级）

| 关节 | 关节级(°) | config q_upper(°) | 差值 | 影响 |
|------|----------|------------------|------|------|
| J1 | 178 | 180 | +2° | MPC 规划到 179° 时控制器强制停止 |
| J3 | 135 | 150 | +15° | 差距最大，挥拍中 J3 活跃 |
| J4 | 178 | 180 | +2° | 同 J1 |

**修复**：收紧 `configs/real_robot_test.yaml` 和 `real_robot.yaml` 中 J1/J3/J4 的 `q_upper`
至 ≤ 关节级限位（J1/J4→175°，J3→130°，留 3~5° 安全余量）。

> ✅ **已修复（2026-07-03）**：`real_robot.yaml` / `real_robot_test.yaml` / `src/real/config.py`
> 三处同步收紧。J1/J4 ±180→±175，J3 ±150→±130（q_lower/q_upper 对称收紧）。

### 10.3 其他发现

- **运行模式 mode=1**（非 0 空闲）：上次运动残留状态，非阻塞异常
- **速度限位合理**：关节级 18~24 rad/s，config 1.0 rad/s（测试保守值）✅
- **里程计**：J1 = 2634万°，J4 = 2515万°（磨损最大，符合挥拍特征）
- **使能全 1、错误全 0** ✅
- **D2 dict 结构**：`{return_code, err_flag[6], brake_state[6]}`

---

## 11. plan_speed 规划速度测试

> 测试脚本：`scripts/tools/test_real_robot/13_plan_speed_test.py`
> 测试日期：2026-07-02

### 11.1 三次测试数据

| 测试 | plan_speed=100 v=20 | plan_speed=50 v=20 | plan_speed=20 v=20 | plan_speed=100 v=50 | follow 20 vs 100 |
|------|---------------------|---------------------|---------------------|---------------------|-------------------|
| J1 ±10° | 0.240s* | 1.043s | 1.868s | 0.465s | 1.05 |
| J6 ±40° | 1.485s | 2.390s (1.61x) | 5.062s (3.41x) | 0.924s (0.62x) | 0.99 |
| J6 ±90° | 2.601s | 4.612s (1.77x) | 10.614s (4.08x) | 1.401s (0.54x) | 1.00 |

*J1 ±10° 基线 ret=1 可能无效

### 11.2 plan_speed 缩放分析

plan_speed 和 v 均有效影响规划运动耗时，缩放为亚线性：

| 行程 | T(50)/T(100) | T(20)/T(100) | 线性预期 |
|------|-------------|-------------|---------|
| 40° | 1.61x | 3.41x | 2.0x / 5.0x |
| 90° | 1.77x | 4.08x | 2.0x / 5.0x |

行程越大越接近线性（匀速段占比增加，加减速开销占比减小）。

### 11.3 结论

- **plan_speed 和 v 均有效**：数值越大运动越快
- **缩放亚线性**：短行程以加减速三角波为主，时间 ∝ 1/√加速度 而非 1/速度
- **`rm_movej_follow` 三次验证不受 plan_speed 影响**（比值 1.00~1.05）
- **MPC 部署结论**：连接时设 `plan_speed=100`，流式控制完全独立，无需动态调整

---

## 12. 速度透传测试（rm_movev_canfd）

> 测试脚本：`scripts/tools/test_real_robot/14_velocity_passthrough.py`
> 测试日期：2026-07-03

### 12.1 三次测试数据

| 运行 # | 速度 | 基线位移精度 | 方向验证 X 位移 | 停止延迟 follow=True |
|--------|------|-------------|----------------|---------------------|
| 第 1 次 | 0.01 m/s | 5.2mm（16%）| -1.8mm（反方向！）| 43ms |
| 第 2 次 | 0.01 m/s | 19.1mm（96%）| +14.6mm（正确） | 40ms |
| 第 3 次 | 0.05 m/s | 97.6mm（98%）| -10.4mm（反方向！）| 143ms |

### 12.2 速度大小跟踪

线性度优秀（第 2/3 次）：

| 指令速度 | 第 2 次位移 | 第 3 次位移 | 达到率 |
|---------|-----------|-----------|-------|
| 0.005/0.025 | 5.6mm | 23.8mm | ~95% |
| 0.010/0.050 | 9.7mm | 46.5mm | ~93-97% |
| 0.020/0.100 | 20.5mm | 96.4mm | ~96% |

### 12.3 方向精度问题（核心缺陷）

命令 +X 运动，实际 Y 方向位移远大于 X：

| 运行 | ΔX(mm) | ΔY(mm) | Y/X 比值 |
|------|--------|--------|---------|
| 第 2 次 | +4.3 | -18.6 | 4.3 |
| 第 3 次 | +20.4 | -94.6 | 4.6 |

Y/X 比值一致 ~4.5:1 → 基坐标系旋转 ≈ arctan(4.5) ≈ 77°。SDK "X 轴"几乎指向物理 Y 方向。
方向验证跨运行不一致（时正时负）。

### 12.4 停止延迟随速度恶化

| 速度 | follow=True | follow=False |
|------|------------|-------------|
| 0.01 m/s | 40ms | 66-99ms |
| 0.05 m/s | 143ms | 105ms |

MPC 挥拍速度（~1-2 m/s）的停止延迟会更长，安全不可接受。

### 12.5 结论

| 维度 | 评价 |
|------|------|
| 速度大小跟踪 | 好（93~98%），线性度优秀 |
| 方向精度 | **系统性偏移**（基坐标系旋转 ~77°），跨运行不一致 |
| 停止延迟 | 高速恶化（40→143ms） |
| MPC 可用性 | **不可用** |

**`rm_movev_canfd` 排除**。`rm_movej_follow`（关节空间控制）仍是唯一可行的流式控制接口。
速度透传作为备选方案记录在案，未来若需要可先标定基坐标系旋转再评估。

---

## 13. 修复记录 + Script 15 跳过决策

> 日期：2026-07-03

### 13.1 包络球构造修复（§9）

**状态**：✅ 已修复

`_connect.py init_algo()` 中 `rm_tool_sphere_t(x=, y=, z=, radius=)` 构造方式已修正为
`rm_tool_sphere_t(centrePoint=(x, y, z), radius=)`。SDK ctypes Structure 静默忽略未知 kwargs，
导致 centrePoint 始终为 [0,0,0]——3 个包络球全堆在法兰中心，拍柄/拍面无碰撞保护。

修复后 3 个球正确分布于法兰(0.03)、手柄(0.02)、拍面(0.07)，自碰撞检测覆盖完整球拍。
同时将 `except Exception: pass` 改为 `logging.warning`。

### 13.2 关节限位收紧（§10）

**状态**：✅ 已修复

| 关节 | 修复前(°) | 修复后(°) | 关节级限位(°) | 余量 |
|------|----------|----------|-------------|------|
| J1 ± | ±180 | ±175 | ±178 | 3° |
| J3 ± | ±150 | ±130 | ±135 | 5° |
| J4 ± | ±180 | ±175 | ±178 | 3° |

同步修改：`configs/real_robot.yaml`、`configs/real_robot_test.yaml`、`src/real/config.py`。
`configs/default.yaml`（MPC 规划用 `q_min_deg`/`q_max_deg`）原本已正确，无需改动。

### 13.3 Script 15 跳过决策

**状态**：⏭️ 跳过（MPC 部署不需要）

Script 15 (`15_joint_management_write.py`) 测试 6 个关节管理写操作：
清错误 / 清里程计 / 清系统错误 / 清运行时间 / 自动限位 / 设置安装姿态。

**跳过理由**：全部为维护/恢复类接口，不在 MPC 控制回路中。需要时可用示教器手动操作，
无需代码层面验证。脚本已归档至 main 分支供未来参考。
