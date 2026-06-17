# RM-65 网球机器人 — MPC+iLQR+Tube 击打仿真与真机部署

RM-65 双臂机器人网球击打项目。使用 **MPC（模型预测控制）** 作为外层闭环框架，**iLQR（迭代线性二次调节器）** 作为内层轨迹优化求解器，**Tube-based Robust Hitting** 实现时空鲁棒性，**多层安全滤波**保障关节/TCP 约束。

支持两种控制模式：
- **力矩模式（默认）**：MPC 输出关节力矩，MuJoCo 直接驱动
- **位置模式（`--position-mode`）**：MPC 输出关节角度，MuJoCo PD 执行器模拟真机控制器，iLQR 直接规划 q_desired 轨迹

真机部署使用 Realman SDK 角度控制（IP 通信 `rm_movej_follow`），三层安全架构（控制器固件 + 软件监控 + 紧急停止）。

---

## 环境安装

```bash
# 创建 conda 环境
conda create -n mujoco_tennis python=3.11
conda activate mujoco_tennis
pip install -r requirements.txt

# 编译 C++ 加速模块（iLQR 线性化 + 前向/后向传递，1.50× 加速）
python setup.py build_ext --inplace

# Linux 环境需要设置 MuJoCo 库路径
export LD_LIBRARY_PATH="$(python -c 'import mujoco, os; print(os.path.dirname(mujoco.__file__))'):$LD_LIBRARY_PATH"
```

依赖：`mujoco>=3.0`, `numpy>=1.24`, `scipy>=1.10`, `matplotlib>=3.7`, `pyyaml>=6.0`

---

## 快速开始

### 仿真

```bash
# ★ V11 最新版 — 力矩模式（默认）
python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --viewer

# V11 位置模式（模拟真机角度控制）
python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --position-mode --viewer

# 指定随机种子
python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --seed 42 --viewer

# 离线仿真（无渲染，更快）
python scripts/rm65_mpc_tube_constraint.py --serve-box --ball-speed 9 --no-plot

# 关节调节查看器（拖动滑条控制关节）
python scripts/tools/rm65_joint_viewer.py
```

### 真机部署

```bash
# 1. 修改真机配置（IP 地址、安全参数、PD 增益）
vim configs/real_robot.yaml

# 2. 逐个验证真机接口（先只读，后运动）
python scripts/tools/test_real_robot/01_connect_disconnect.py    # 连接测试
python scripts/tools/test_real_robot/02_read_joints.py           # 读关节角度
python scripts/tools/test_real_robot/04_send_zero_pose.py        # 回零位
# ...完整清单见 scripts/tools/test_real_robot/README.md
```

### 测试

```bash
# 运行全部测试（217 tests）
pytest tests/

# 代码检查
ruff check src/ tests/ scripts/
```

---

## 双模式执行器

项目支持力矩和位置两种控制模式，仿真和真机共用同一套 MPC+iLQR 框架。

### 力矩模式（默认）

MPC 输出关节力矩 `u = tau(6)`，MuJoCo `motor` 执行器直接输出力矩到关节。

```bash
python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7
```

### 位置模式（`--position-mode`）

MPC 输出期望关节角度 `u = q_desired(6)`，MuJoCo `general` 执行器内部 PD 闭环：
```
tau = Kp * (q_desired - q) - Kd * qdot
```
iLQR 在 PD 执行器动态下规划最优 q_desired 轨迹，前馈补偿重力+科氏力。

```bash
python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --position-mode
```

**真机使用位置模式** — Realman SDK `rm_movej_follow` 接受角度指令，PlanningEnv 用 MuJoCo PD 执行器模拟真机控制器。

### 关键差异

| 特性 | 力矩模式 | 位置模式 |
|------|---------|---------|
| 控制量 u | tau(6) 力矩 | q_desired(6) 期望角度 |
| B 矩阵 | `dt * M⁻¹` | `dt * M⁻¹ * diag(Kp)` |
| 安全滤波 β 缩放 | `beta * u` 力矩缩放 | `q + beta*(u - q)` 位置插值 |
| 紧急制动 | `u = -20·qdot` 阻尼力矩 | 保持当前角度 |
| 随挥 | `J^T * F` PD 控制器 | `solve_ik()` 逆运动学 |
| 真机部署 | ❌ SDK 无力矩 API | ✅ `rm_movej_follow` |

---

## 真机部署架构

```
                    真机控制管线
┌─────────────────────────────────────────────────┐
│  每个控制 tick:                                   │
│                                                  │
│  RobotInterface.get_arm_state()  ← 真机关节角度   │
│  BallPerceiver.get_latest_filtered() ← 动捕+KF   │
│           ↓                                      │
│  PlanningEnv (MuJoCo 纯计算)                     │
│    • set_arm_state(x_real)  ← 把真机状态搬进仿真  │
│    • iLQR 在仿真中规划最优轨迹                    │
│    • step_from_state(x, u)  ← 试不同的控制        │
│           ↓                                      │
│  RobotInterface.send_joint_command(q_desired)    │
│    → rm_movej_follow → 真机执行                   │
└─────────────────────────────────────────────────┘
```

| 模块 | 文件 | 职责 |
|------|------|------|
| **PlanningEnv** | `src/ilqt/planning_env.py` | MuJoCo 纯计算（FK/Jacobian/前向仿真），不接触真机 |
| **RobotInterface** | `src/real/robot_interface.py` | Realman SDK 封装（角度控制 + 连接时配置控制器安全） |
| **BallPerceiver** | `src/real/ball_perceiver.py` | 球感知（sensor → KF 滤波 → pos/vel） |
| **SafetyMonitor** | `src/real/safety_monitor.py` | 软件安全检查（关节/TCP 超限 → 急停） |
| **配置** | `configs/real_robot.yaml` | 7 节配置（连接/控制/安全/PD/感知 + 丰富注释） |

### 三层安全架构

```
Layer 1: 控制器固件（连接时自动配置）
  rm_set_collision_state / rm_set_self_collision_enable / rm_set_controller_torque_limit

Layer 2: SafetyMonitor（每 tick 软件检查）
  关节位置/速度/TCP 速度超限 → slow_stop()

Layer 3: 紧急停止（兜底）
  rm_set_arm_stop()（不可恢复）/ 硬件急停按钮
```

### 真机接口测试工具

逐个验证 SDK API，确保可靠后才集成到项目中。位于 `scripts/tools/test_real_robot/`：

| 脚本 | 风险 | 说明 |
|------|------|------|
| `01_connect_disconnect.py` | 零 | 连接→安全配置→读角度→断开 |
| `02_read_joints.py` | 零 | 持续表格显示角度/速度 |
| `03_read_temperature.py` | 零 | 持续读温度/电压/电流 |
| `04_send_zero_pose.py` | 微 | 流式插值回零位 |
| `05_send_joint_command.py` | 中 | 发送任意角度（`--deg`/`--rad`/交互式） |
| `06_safety_config_verify.py` | 零 | 读回安全参数验证 |
| `07_emergency_stop.py` | 中 | 缓停+急停测试 |
| `08_full_motion_test.py` | 中 | 正弦波运动测试 |

详见 [`scripts/tools/test_real_robot/README.md`](scripts/tools/test_real_robot/README.md)

---

## 核心算法

### MPC + iLQR 闭环架构

```
每 N 步重规划（由 replan-interval 控制）:
  1. 观测球当前位置和速度
  2. find_hitting_point_physics → 物理仿真预测击打点
  3. Softmin 多终端代价 → 允许在候选时间窗口内任意时刻击球
  4. 生成 Warm-start 控制序列（后摆轨迹 PD）
  5. solve_few_iters → iLQR 优化轨迹
  6. 安全滤波（关节/TCP/半空间约束）
  7. 执行第一个控制指令（力矩或角度）
  8. 下一时间步重复
```

### Tube-based Robust Hitting

不确定性管道建模球到达时间的偏差，通过空间走廊式代价提升鲁棒性：

```
σ(t) = σ₀ + σᵥ·t + σₐ·t²       （不确定性管道）
候选击球窗口：以 best_k 为中心，window_half_ms 为半宽

走廊代价（不绑定时间-空间对应）:
  1. 垂直偏离代价（hinge loss）
  2. 速度方向代价（球拍沿球轨迹线方向运动）
  3. 法向量代价（拍面朝向来球方向）
```

### 多层安全滤波

```
1. 关节约束：位置/速度/加速度/力矩 四重限制
2. TCP 速度硬限制
3. X 平面墙：臂不越过身体中线（X=0）
4. 逐步安全滤波：β = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
```

### 感知层：BallEstimator 6D 卡尔曼滤波器

在噪声观测下估计球的真实位置和速度：

```
状态向量: x = [px, py, pz, vx, vy, vz]  （6D 位置+速度）
过程模型: 匀速 + 重力（F 矩阵考虑 g=9.81 对 Vz 的衰减）
观测模型: H = I₆（全状态直接观测）
弹跳保护: Z < 0.01m 时 slam 位置、速度处理反弹/落地
```

三层感知架构：仿真真值 → 噪声注入（`add_observation_noise`）→ 卡尔曼滤波（`BallEstimator`）→ 规划消费。

---

## V11 命令行参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--serve-box` | flag | — | 使用长方体发球区模式 |
| `--ball-speed` | float | None | 球到达击打点时水平速度 (m/s) |
| `--position-mode` | flag | — | 位置模式（默认力矩模式） |
| `--seed` | int | None | 随机种子 |
| `--viewer` | flag | — | MuJoCo 查看器回放 |
| `--no-plot` | flag | — | 禁用 matplotlib 可视化 |
| `--horizon` | int | None | 短地平线步数 |
| `--iter` | int | None | 每次重规划迭代数 |
| `--replan-interval` | int | None | 重规划间隔步数 |
| `--window-ms` | float | 50.0 | Tube 候选窗口半宽 (ms) |
| `--softmin-beta` | float | 5.0 | Softmin 温度参数 |
| `--max-tcp` | float | None | TCP 线速度硬限制 (m/s) |
| `--terminal-exempt-steps` | int | None | 终段 qdot/TCP 豁免步数 |

> V5 参数表已过时，请使用 V11。完整参数列表运行 `python scripts/rm65_mpc_v11.py --help`

---

## 目录结构

```
mujoco_sim/
├── README.md                             # 本文件
├── AGENTS.md                             # 项目开发规范（详细）
├── requirements.txt
├── setup.py                              # C++ 扩展构建（pybind11）
│
├── configs/
│   ├── default.yaml                      # 基础仿真参数
│   ├── mpc.yaml                          # MPC 专用参数
│   ├── cost_hitting.yaml                 # 代价函数权重
│   └── real_robot.yaml                   # 真机配置（7节+丰富注释）
│
├── scripts/
│   ├── rm65_mpc_v11.py                   # ★ V11 最新仿真主脚本
│   ├── rm65_mpc_tube_constraint.py       # 离线仿真
│   ├── tools/
│   │   ├── rm65_joint_viewer.py          # 关节调节查看器
│   │   ├── test_real_robot/              # 真机接口测试工具（01-08）
│   │   │   ├── README.md                 # 安全须知 + 使用说明
│   │   │   ├── _connect.py               # 公共连接/预检模块
│   │   │   └── 01~08_*.py                # 逐个 API 测试脚本
│   │   └── ...
│   ├── exp/                              # 批量实验设施
│   ├── extract/                          # 结果提取
│   └── plot/                             # 论文图表
│
├── src/
│   ├── robot/
│   │   └── rm65_model.xml                # MuJoCo 模型（双臂12DOF + 球拍 + 球）
│   ├── sim/
│   │   └── rm65_env.py                   # RM65Env 仿真环境（力矩/位置双模式）
│   ├── ilqt/                             # iLQR 求解器 + 规划环境
│   │   ├── solver.py                     # iLQR 后向-前向迭代
│   │   ├── cost.py                       # 代价函数（Tube + Softmin）
│   │   ├── planning_env.py               # PlanningEnv 规划计算环境（MuJoCo 纯计算）
│   │   ├── robot_env_protocol.py         # RobotEnv Protocol
│   │   ├── async_replanner.py            # 异步重规划器
│   │   └── robot_limits.py               # 安全滤波
│   ├── real/                             # 真机部署模块（纯真机接口，不含 MuJoCo）
│   │   ├── config.py                     # RealRobotConfig
│   │   ├── robot_interface.py            # Realman SDK 封装
│   │   ├── ball_sensor.py                # BallSensor ABC + SimulatedBallSensor
│   │   ├── ball_perceiver.py             # BallPerceiver（KF 滤波）
│   │   ├── safety_monitor.py             # SafetyMonitor
│   │   ├── adaptive_timer.py             # 自适应频率控制
│   │   └── torque_to_position.py         # 力矩→位置积分器（备用）
│   ├── dynamics/                         # 动力学线性化
│   ├── perception/                       # 卡尔曼滤波器
│   ├── tennis/                           # 网球轨迹预测 + 击打点计算
│   ├── cpp/                              # C++ 加速（pybind11）
│   └── utils/                            # 工具（模型加载/噪声注入/数学）
│
├── tests/                                # 单元测试（217 tests）
├── docs/                                 # 技术文档
└── paper/                                # 论文 LaTeX 工程
```

> 详细模块说明见 [`src/README.md`](src/README.md)，脚本清单见 [`scripts/README.md`](scripts/README.md)

---

## 配置文件

### `configs/default.yaml` — 仿真参数

```yaml
sim:
  dt: 0.005              # 仿真步长 (s)

cost:
  Q_p: [50000, 50000, 50000]   # 终端位置代价权重
  Q_v: [200, 200, 200]         # 终端速度代价权重
  R: 0.0001                    # 控制代价权重

hitting:
  racket_speed: 1.8             # 期望击球速度 (m/s)
  workspace_radius: 0.85        # 工作空间半径 (m)
```

### `configs/real_robot.yaml` — 真机配置

```yaml
robot:
  ip: "192.168.1.18"             # 机械臂 IP 地址
  port: 8080

control:
  control_mode: "ip"             # "ip"(rm_movej_follow) | "canfd"(rm_movej_canfd)
  dt: 0.005                      # MPC 规划步长

safety:
  collision_stage: 5             # 碰撞灵敏度 0-8（⚠️ 首次5，稳定后降低）
  torque_limit: [50, 50, 50, 30, 30, 30]  # N·m
  max_tcp_speed: 1.0             # TCP 最大线速度 m/s（⚠️ 实验时调整）

position_mode:
  kp: [200, 200, 100, 50, 50, 20]   # PD 位置增益（⚠️ 实验时调整）
  kd: [20, 20, 10, 5, 5, 2]         # PD 速度增益
  enable_feedforward: true           # 重力+科氏力前馈补偿

perception:
  sensor_type: "simulated"       # "simulated" / "optitrack" / "realsense"
  pos_noise_std: 0.005           # 位置噪声 m（⚠️ 标定后修改）
```

> 完整配置说明和参数注释见 `configs/real_robot.yaml`，标记说明见 [`AGENTS.md`](AGENTS.md) § 真机部署架构
