# 真机接口测试工具

逐个验证 RM-65B 真机 API 是否符合预期，确保每个接口可靠后才集成到项目中。

## ⚠️ 安全须知

1. **首次运行**：必须先运行 `01~03`（只读脚本），确认硬件通信正常
2. **运动脚本**（04+）：物理急停按钮必须**在手边**
3. **Ctrl+C**：所有运动脚本中 `Ctrl+C` = **缓停**（`rm_set_arm_slow_stop`），不是直接退出
4. **确认机制**：运动类脚本需要手动输入 `YES` 才会执行
5. **安全预检**：默认启用限位+自碰撞+奇异性检查，`--no-algo-check` 可跳过
6. **渐进步进**：先低速验证 → 再逐步提高

## 快速开始

```bash
# 0. 修改配置文件中的 IP 地址
vim configs/real_robot.yaml

# 1. 验证连接（零风险）
python scripts/tools/test_real_robot/01_connect_disconnect.py

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

## 脚本清单

### 只读测试（零风险）

| 脚本 | 说明 | 关键 SDK API |
|------|------|-------------|
| `01_connect_disconnect.py` | 连接→安全配置→读角度→断开 | `rm_create_robot_arm` / `rm_get_joint_degree` / `rm_delete_robot_arm` |
| `02_read_joints.py` | 持续表格显示角度/速度（20Hz） | `rm_get_joint_degree`（数值微分速度） |
| `03_read_temperature.py` | 持续表格显示温度/电压/电流 | `rm_get_current_joint_temperature` / `_voltage` / `_current` |
| `06_safety_config_verify.py` | 读回安全参数，验证 `_configure_safety()` 是否生效 | `rm_get_collision_stage` / `rm_get_self_collision_enable` / ... |

### 运动测试

| 脚本 | 风险 | 说明 | 关键 SDK API |
|------|------|------|-------------|
| `04_send_zero_pose.py` | 微 | 流式插值回零位（YES确认+预检） | `rm_movej_follow` |
| `05_send_joint_command.py` | 中 | 发送任意角度（`--deg`/`--rad`/交互式） | `rm_movej_follow` |
| `07_emergency_stop.py` | 中 | 缓停+急停测试 | `rm_set_arm_slow_stop` / `rm_set_arm_stop` |
| `08_full_motion_test.py` | 中 | 小幅正弦波运动（可配置关节/幅度/周期） | `rm_movej_follow` 连续发送 |

### 运动脚本参数详解

#### 05_send_joint_command.py

```bash
# 方式 1: 度数（推荐日常使用）
python 05_send_joint_command.py --deg 0 30 -15 0 5 0

# 方式 2: 弧度
python 05_send_joint_command.py --rad 0.0 0.524 -0.262 0 0.087 0

# 方式 3: 交互式（不传参数）
python 05_send_joint_command.py
# 逐个输入 J1~J6 角度（度），回车保持当前值

# 可选参数
python 05_send_joint_command.py --deg 0 30 0 0 0 0 --duration 2.0  # 2秒到达
python 05_send_joint_command.py --deg 0 30 0 0 0 0 --no-algo-check  # 跳过碰撞检查
```

#### 08_full_motion_test.py

```bash
# 默认: J1 ±5°, 周期2s, 持续10s
python 08_full_motion_test.py

# J2 ±10°, 周期3s, 持续20s
python 08_full_motion_test.py --joint 2 --amplitude 10 --period 3 --duration 20
```

## 公共模块

### `_connect.py`

| 函数 | 功能 |
|------|------|
| `load_and_connect(config_path)` | 加载 YAML → 创建 RobotInterface → 连接 → 配置安全参数 |
| `pre_motion_check(ri, monitor, q_desired, algo)` | 限位+自碰撞+奇异性预检 |
| `init_algo()` | SDK Algo 类初始化（球拍包络球配置） |
| `safe_disconnect(ri)` | 缓停 + 断开连接 |
| `add_config_arg(parser)` | argparse 添加 `--config` 参数 |
| `add_algo_check_arg(parser)` | argparse 添加 `--no-algo-check` 参数 |

### `pre_motion_check` 检查项

按顺序执行，任一失败即取消运动：

| # | 检查项 | 来源 | 可跳过 |
|---|--------|------|--------|
| 1 | 关节位置限位 `q_lower ≤ q ≤ q_upper` | SafetyMonitor | ❌ 不可跳过 |
| 2 | 关节速度限位 `\|qdot\| ≤ max_qdot` | SafetyMonitor | ❌ 不可跳过 |
| 3 | 自碰撞检测 | SDK Algo (`rm_algo_safety_robot_self_collision_detection`) | ✅ `--no-algo-check` |
| 4 | 奇异性检测 | SDK Algo (`rm_algo_kin_robot_singularity_analyse`) | ✅ `--no-algo-check` |

## 常见问题

### 连接失败

```
❌ 连接失败
```

**排查**：
- 机械臂 IP 是否正确？检查 `configs/real_robot.yaml` 中 `robot.ip`
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
