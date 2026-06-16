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
```

## 脚本清单

### 首批（已实现）

| 脚本 | 风险 | 说明 | 关键 SDK API |
|------|------|------|-------------|
| `01_connect_disconnect.py` | 零 | 连接→安全配置→读角度→断开 | `rm_create_robot_arm` / `rm_get_joint_degree` / `rm_delete_robot_arm` |
| `02_read_joints.py` | 零 | 持续表格显示角度/速度（20Hz） | `rm_get_joint_degree`（数值微分速度） |
| `03_read_temperature.py` | 零 | 持续表格显示温度/电压/电流 | `rm_get_current_joint_temperature` / `_voltage` / `_current` |
| `04_send_zero_pose.py` | 微 | 流式插值回零位（YES确认+预检） | `rm_movej_follow` |

### 第二批（待实现）

| 脚本 | 风险 | 说明 |
|------|------|------|
| `05_send_joint_command.py` | 中 | 发送任意关节角度（`--deg`/`--rad`/交互式） |
| `06_safety_config_verify.py` | 零 | 读回安全参数验证是否生效 |
| `07_emergency_stop.py` | 中 | 测试缓停+急停 |
| `08_full_motion_test.py` | 中 | 小幅正弦波运动测试 |

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
运行 `01_connect_disconnect.py` 查看日志中是否有 `控制器安全配置部分失败` 警告。

### 温度超过 60°C

`03_read_temperature.py` 会标记 `⚠️`。持续高温需停机冷却。
