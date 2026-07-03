# 真机接口测试工具

逐个验证 RM-65B 真机 API 是否符合预期，确保每个接口可靠后才集成到项目中。

## ⚠️ 安全须知

1. **首次运行**：必须先运行 `01~03`（只读脚本），确认硬件通信正常
2. **运动脚本**（04+）：物理急停按钮必须**在手边**
3. **Ctrl+C**：所有运动脚本中 `Ctrl+C` = **缓停**（`rm_set_arm_slow_stop`），不是直接退出
4. **确认机制**：运动类脚本需要手动输入 `YES` 才会执行
5. **安全预检**：默认启用限位+自碰撞+奇异性检查，`--no-algo-check` 可跳过后两项。TCP 速度由控制器固件强制限制
6. **渐进步进**：先低速验证 → 再逐步提高

## 快速开始

```bash
# 0. 配置文件（二选一）
#    测试专用（保守参数，推荐首次使用）: configs/real_robot_test.yaml
#    生产参数: configs/real_robot.yaml
#    详细说明见 CONFIG.md
vim configs/real_robot_test.yaml   # 至少修改 robot.ip 为你的机械臂 IP

# 1. 验证连接（零风险）
python scripts/tools/test_real_robot/01_connect_disconnect.py

# 1b. 验证固件版本 + 查询 API 可靠性（零风险，固件升级后必跑）
python scripts/tools/test_real_robot/01b_firmware_api_verify.py

# 2. 持续读关节角度（零风险）
python scripts/tools/test_real_robot/02_read_joints.py

# 3. 读温度/电压/电流（零风险）
python scripts/tools/test_real_robot/03_read_temperature.py

# 4. 回到零位（微风险，需确认）
python scripts/tools/test_real_robot/04_send_zero_pose.py

# 5. 发送任意角度（中风险）
python scripts/tools/test_real_robot/05_send_joint_command.py --deg 0 30 0 0 0 0

# 6. 验证安全参数已生效（零风险）
python scripts/tools/test_real_robot/06_safety_config_verify.py

# 7. 测试缓停（中风险，推荐首次只测缓停）
python scripts/tools/test_real_robot/07_emergency_stop.py

# 8. 正弦波运动（中风险）
python scripts/tools/test_real_robot/08_full_motion_test.py --joint 1 --amplitude 5
```

## 配置文件

测试脚本默认加载 `configs/real_robot_test.yaml`（保守参数）。
完整字段说明和调参建议见 **[CONFIG.md](./CONFIG.md)**。

| 文件 | 用途 | `max_tcp_speed` | `max_qdot` | `torque_limit` |
|------|------|-----------------|------------|-----------------|
| `real_robot_test.yaml` | 测试默认（保守） | 0.3 m/s | 1.0 rad/s | 30/30/30/20/20/20 |
| `real_robot.yaml` | MPC 生产部署 | 1.0 m/s | 3.14 rad/s | 50/50/50/30/30/30 |

通过 `--config` 参数可切换配置文件：
```bash
python scripts/tools/test_real_robot/04_send_zero_pose.py --config configs/real_robot.yaml
```

## 脚本清单

### 只读测试（零风险）

| 脚本 | 说明 | 关键 SDK API |
|------|------|-------------|
| `01_connect_disconnect.py` | 连接→安全配置→读角度→读固件版本→断开 | `rm_create_robot_arm` / `rm_get_joint_degree` / `rm_get_arm_software_info` / `rm_delete_robot_arm` |
| `02_read_joints.py` | 持续表格显示角度/速度（20Hz） | `rm_get_joint_degree`（数值微分速度） |
| `03_read_temperature.py` | 持续表格显示温度/电压/电流 | `rm_get_current_joint_temperature` / `_voltage` / `_current` |
| `06_safety_config_verify.py` | 读回安全参数，验证 `_configure_safety()` 是否生效 | `rm_get_collision_stage` / `rm_get_self_collision_enable` / `rm_get_avoid_singularity_mode` / `rm_get_controller_torque_limit` |
| `09_algo_fk_ik_verify.py` | Algo FK/IK 与 MuJoCo PlanningEnv 对比 | `rm_algo_forward_kinematics` / `inverse_kinematics` / DH 参数 |
| `10_algo_safety_verify.py` | Algo 自碰撞/奇异检测验证 | `rm_algo_safety_robot_self_collision_detection` / `rm_algo_kin_robot_singularity_analyse` |
| `11_joint_config_query.py` | 关节级/驱动级限位 + 使能/错误/里程计 | `rm_get_joint_max_pos/speed/acc` / `rm_get_joint_drive_max_*` / `rm_get_joint_en_state` / `rm_get_joint_err_flag` / `rm_get_joint_odom` |
| `12_system_info_query.py` | 运行模式/控制器/电源/安装姿态/版本 | `rm_get_arm_run_mode` / `rm_get_controller_state` / `rm_get_robot_info` / `rm_get_install_pose` |

### 运动测试

| 脚本 | 风险 | 说明 | 关键 SDK API |
|------|------|------|-------------|
| `04_send_zero_pose.py` | 微 | 流式插值回零位（YES确认+预检） | `rm_movej_follow` |
| `05_send_joint_command.py` | 中 | 发送任意角度（`--deg`/`--rad`/交互式） | `rm_movej_follow` |
| `07_emergency_stop.py` | 中 | 缓停+急停测试 | `rm_set_arm_slow_stop` / `rm_set_arm_stop` |
| `08_full_motion_test.py` | 中 | 小幅正弦波运动（可配置关节/幅度/周期） | `rm_movej_follow` 连续发送 |
| `13_plan_speed_test.py` | 微 | 规划速度缩放测试 | `rm_set_plan_speed` / `rm_movej` |
| `14_velocity_passthrough.py` | 中 | 速度透传方向/停止延迟测试（⚠️ 已排除，方向偏移 ~77°） | `rm_movev_canfd` |
| `15_joint_management_write.py` | 中 | 关节管理写操作（⏭️ 跳过测试，MPC 不需要） | `rm_set_joint_clear_err` / `rm_clear_joint_odom` / `rm_set_install_pose` |

### 公共参数

所有脚本通过 `_connect.py` 共享以下 CLI 参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config PATH` | 配置文件路径 | `configs/real_robot_test.yaml`（不存在时回退 `real_robot.yaml`） |
| `--no-algo-check` | 跳过 SDK Algo 自碰撞/奇异性检查 | False |
| `--hz FLOAT` | 发送/刷新频率 | 脚本特定（见下文） |

### 运动脚本参数详解

#### 04_send_zero_pose.py

```bash
python scripts/tools/test_real_robot/04_send_zero_pose.py                          # 默认 1s 到达
python scripts/tools/test_real_robot/04_send_zero_pose.py --duration 2.0           # 2秒到达
python scripts/tools/test_real_robot/04_send_zero_pose.py --hz 50                  # 50Hz 发送
python scripts/tools/test_real_robot/04_send_zero_pose.py --no-algo-check          # 跳过碰撞检查
```

#### 05_send_joint_command.py

```bash
# 方式 1: 度数（推荐日常使用）
python scripts/tools/test_real_robot/05_send_joint_command.py --deg 0 30 -15 0 5 0

# 方式 2: 弧度
python scripts/tools/test_real_robot/05_send_joint_command.py --rad 0.0 0.524 -0.262 0 0.087 0

# 方式 3: 交互式（不传参数）
python scripts/tools/test_real_robot/05_send_joint_command.py
# 逐个输入 J1~J6 角度（度），回车保持当前值

# 可选参数
python scripts/tools/test_real_robot/05_send_joint_command.py --deg 0 30 0 0 0 0 --duration 2.0  # 2秒到达
python scripts/tools/test_real_robot/05_send_joint_command.py --deg 0 30 0 0 0 0 --hz 50         # 50Hz 发送
python scripts/tools/test_real_robot/05_send_joint_command.py --deg 0 30 0 0 0 0 --no-algo-check # 跳过碰撞检查
```

#### 08_full_motion_test.py

```bash
# 默认: J1 ±5°, 周期2s, 持续10s, 100Hz
python scripts/tools/test_real_robot/08_full_motion_test.py

# J2 ±10°, 周期3s, 持续20s
python scripts/tools/test_real_robot/08_full_motion_test.py --joint 2 --amplitude 10 --period 3 --duration 20

# 跳过碰撞检查
python scripts/tools/test_real_robot/08_full_motion_test.py --no-algo-check
```

## 公共模块

### `_connect.py`

| 函数 | 功能 |
|------|------|
| `load_and_connect(config_path)` | 加载 YAML → 创建 RobotInterface → 连接 → 配置安全参数 |
| `load_config(config_path)` | 加载 YAML → 返回 RealRobotConfig |
| `pre_motion_check(ri, monitor, q_desired, arm_state, algo)` | 限位+自碰撞+奇异性预检 |
| `init_algo()` | SDK Algo 类初始化（球拍包络球配置） |
| `safe_disconnect(ri)` | 缓停 + 断开连接 |
| `add_config_arg(parser)` | argparse 添加 `--config` 参数 |
| `add_algo_check_arg(parser)` | argparse 添加 `--no-algo-check` 参数 |

### `pre_motion_check` 检查项

按顺序执行，任一失败即取消运动：

| # | 检查项 | 来源 | 可跳过 |
|---|--------|------|--------|
| 1 | SafetyMonitor 限位检查（关节位置 + 关节速度） | `safety_monitor.py` `is_safe()` | ❌ 不可跳过 |
| 2 | 自碰撞检测 | SDK Algo (`rm_algo_safety_robot_self_collision_detection`) | ✅ `--no-algo-check` |
| 3 | 奇异性检测 | SDK Algo (`rm_algo_kin_robot_singularity_analyse`) | ✅ `--no-algo-check` |

> TCP 速度由控制器固件 Layer 1（`rm_set_arm_max_line_speed`，连接时 `_configure_safety()` 下发）强制限制，不在 `pre_motion_check` 中重复检查。

## 常见问题

### 连接失败

```
❌ 连接失败
```

**排查**：
- 机械臂 IP 是否正确？检查 `configs/real_robot_test.yaml` 中 `robot.ip`
  - 左臂 `192.168.1.18` / 右臂 `192.168.1.19`
- 网线是否连接？`ping 192.168.1.18`
- Realman SDK 是否安装？`pip install Robotic_Arm`
- 机械臂是否已开机？控制器指示灯应为绿色

### `rm_get_joint_degree` 返回错误码

| 错误码 | 含义 | 解决 |
|--------|------|------|
| 0 | 成功 | — |
| -1 | 发送失败 | 检查网络连接 |
| -2 | 接收超时 | 检查网络延迟 |
| 162/163 | 数据解析错误 | SDK/固件版本不匹配 |
| 165 | API 版本不兼容 | 更新 SDK |

完整错误码见 `reference/Tennis-Robot/Realman_Script/notes/13_API错误代码.md`

### 安全参数配置失败

`_configure_safety` 中的 `try/except` 会跳过部分失败的配置项。
运行 `06_safety_config_verify.py` 读回参数验证是否生效。

### 温度超过 60°C

`03_read_temperature.py` 会标记 `⚠️`。持续高温需停机冷却。

### 急停后无法恢复

`07_emergency_stop.py --test-estop` 触发的 `rm_set_arm_stop` 不可软件恢复。
需要重新连接机械臂或手动复位（重新上电）。
