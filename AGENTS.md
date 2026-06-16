# AGENTS.md - Tennis Robot 项目规范

## 项目概述
本项目使用 **MPC + iLQR + Tube** 框架，解决 **RM-65B 双臂机械臂**挥拍击打网球的场景。
机器人模型为 **RM-65B（12自由度双臂工业机械臂）**，安装在垂直桩柱上方，
双末端各连接垂直网球拍（物理上仅右臂装拍，左臂保持零位），
在给定网球飞来轨迹的情况下，计算最优挥拍轨迹，使末端执行器（球拍面）在正确的时间和位置以期望的速度击中网球。

- 状态向量: x = [q(12), qdot(12)] ∈ R^24（双臂关节位置 + 关节速度）; 球自由关节另计
- 控制向量: u = tau(6) ∈ R^6（力矩模式，默认）；或 u = q_desired(6)（位置模式，--position-mode）
- 末端执行器: 球拍面中心点（racket_center site）
- 实际关节: r_joint1~r_joint6（右臂）+ l_joint1~l_joint6（左臂）
- MuJoCo 模型: nq=19(双臂 12+球 7), nv=18(双臂 12+球 6), nu=12(右臂 motor 6 + 左臂 motor 6)

## 语言与注释规范
- **所有代码使用 Python 编写**
- **所有代码注释、docstring 必须使用中文**
- **所有深度思考、设计决策说明使用中文**
- 变量名、函数名、类名使用英文（遵循 Python 命名规范）
- 类型提示（type hints）必须标注在所有函数签名上
- 公有函数必须有中文 docstring（Google 风格）

## 技术栈
- **语言**: Python 3.11+ (conda 环境 `mujoco_tennis`)
- **仿真**: MuJoCo 3.9+（mujoco Python 包）— 跨平台 Windows/Ubuntu
- **数值计算**: NumPy, SciPy
- **C++ 加速**: pybind11 — `src/cpp/` 下的 iLQR 核心循环（linearize_analytical_batch, forward_pass, backward_pass），累计 1.50× 加速
- **可视化**: MuJoCo 内置查看器 + matplotlib（轨迹绘图）
- **包管理**: conda + pip + requirements.txt
- **构建**: `setup.py build_ext --inplace` 编译 C++ 扩展

## 机器人模型定义
RM-65B 12 自由度（双臂各 6），关节分配如下：

### 右臂关节（驱动）
| 关节编号 | MuJoCo 关节名 | qpos 索引 | 说明 |
|----------|--------------|----------|------|
| 0 | r_joint1 | qpos[0] | 右肩偏航 |
| 1 | r_joint2 | qpos[1] | 右肩俯仰 |
| 2 | r_joint3 | qpos[2] | 右肘 |
| 3 | r_joint4 | qpos[3] | 右腕 1 |
| 4 | r_joint5 | qpos[4] | 右腕 2 |
| 5 | r_joint6 | qpos[5] | 右腕 3 |

### 左臂关节（不驱动，保持零位）
| 关节编号 | MuJoCo 关节名 | qpos 索引 |
|----------|--------------|----------|
| 6 | l_joint1 | qpos[6] |
| 7 | l_joint2 | qpos[7] |
| 8 | l_joint3 | qpos[8] |
| 9 | l_joint4 | qpos[9] |
| 10 | l_joint5 | qpos[10] |
| 11 | l_joint6 | qpos[11] |

### 球自由关节
- qpos[12:19] (7维 quaternion + xyz)
- qvel[12:18] (6维)

- MuJoCo 模型为 `src/robot/rm65_model.xml`，是 DOF 数和关节顺序的唯一事实来源
- 球拍: 连杆沿法兰局部方向延伸，球拍面在连杆末端（racket_center site）
- 臂展: 约 850mm（单臂）
- 终端执行器: `racket_center` site（右臂 r_flange → r_racket_body → r_racket → 球拍面）

## 资产目录
```
assets/rm_65/
├── urdf/                        # RM-65B URDF 源文件 + 网格
│   ├── meshes/*.STL, *.dae     # 机械臂视觉网格
│   ├── visual/*.STL            # 灵巧手 visual 网格
│   └── dh_robotics_ag95.urdf   # 灵巧手 URDF
├── realmanControlNode.py       # 真实机械臂控制节点
├── config.py                   # RM-65B 硬件参数
└── ...
```

## 项目目录结构
```
mujoco_sim/
├── AGENTS.md                          # 本文件 — 项目规范与 Agent 指令
├── setup.py                           # C++ 扩展构建脚本（pybind11）
├── requirements.txt                   # Python 依赖
├── skills/                            # Skill 定义（8 个，详见下方 Skills 表）
│   ├── framework_design.md            # 代码框架设计
│   ├── file_management.md             # 文件管理规范
│   ├── sim_run.md                     # 仿真运行流程
│   ├── experiment_design.md           # 实验设计与数据管理
│   ├── figure_generation.md           # 论文图表生成
│   ├── paper_writing.md               # 论文撰写
│   └── paper_review.md                # 论文审稿与迭代
├── src/
│   ├── __init__.py
│   ├── README.md                     # 源代码模块索引与依赖说明
│   ├── robot/                         # 机器人模型定义
│   │   ├── __init__.py
│   │   ├── rm65_model.xml             # MuJoCo XML 模型（双臂12DOF + 球拍 + 球）
│   │   ├── model.xml                  # 旧版单臂模型（左臂装在右侧桩柱）
│   │   └── kinematics.py              # 正运动学 / 雅可比矩阵工具
│   ├── dynamics/                      # 动力学计算
│   │   ├── __init__.py
│   │   ├── linearize.py               # 动力学线性化（fx, fu），供 iLQR 使用
│   │   └── simulate.py                # 前向仿真 / rollout
│   ├── ilqt/                          # iLQR 求解器核心
│   │   ├── __init__.py
│   │   ├── solver.py                  # iLQR 后向-前向迭代主循环（solve / solve_few_iters）
│   │   ├── cost.py                    # 代价函数（终端击打点代价 + 控制代价 + Tube 代价）
│   │   ├── utils.py                   # 增益计算、线搜索、正则化辅助函数
│   │   ├── robot_limits.py            # 关节约束 + 安全滤波（RobotLimits, strict_braking_check）
│   │   ├── retiming.py                # 时间重映射工具
│   │   ├── async_replanner.py         # 异步重规划器（后台线程 iLQR）
│   │   ├── jt_init.py                 # 位置模式 JT 初始控制 + 后摆 warm-start
│   │   ├── robot_env_protocol.py      # RobotEnv Protocol（@runtime_checkable，RM65Env/PlanningEnv 共同接口）
│   │   ├── planning_env.py            # MPC 规划计算环境（MuJoCo 纯计算，无球/无左臂/无碰撞，供真机 iLQR 规划）
│   │   └── costs/                     # 模块化代价函数
│   │       ├── __init__.py
│   │       ├── base.py                # BaseCost 基类
│   │       └── hitting.py             # 击打场景专用代价
│   ├── cpp/                           # C++ 加速模块（pybind11）
│   │   ├── __init__.py
│   │   ├── solver_cpp.py              # Python 封装，桥接 C++ 和 Python（含 _backward_pass_numpy 参考实现）
│   │   ├── core_ext.cpp               # pybind11 模块入口（Unity Build: linearize + forward_pass + backward_pass）
│   │   ├── types.h                    # 常量 + 指针转换 + set_arm_forward
│   │   ├── mujoco_utils.h             # sim_step（含位置模式+FF+qfrc 管理）
│   │   ├── cost_params.h              # StepCheckParams + check_step 约束检查
│   │   ├── linearize.cpp              # 解析动力学线性化（批量）
│   │   ├── forward_pass.cpp           # 前向传递（含碰撞禁用+limits+check_step）
│   │   └── backward_pass.cpp          # 后向传递（纯代数 Riccati，栈上小矩阵高斯消元）
│   ├── sim/                           # MuJoCo 仿真封装
│   │   ├── __init__.py
│   │   ├── env.py                     # MujocoEnv 基类
│   │   ├── rm65_env.py                # RM65Env 双臂环境封装
│   │   └── viewer.py                  # 可视化工具
│   ├── perception/                    # 感知模块（噪声+滤波）
│   │   ├── __init__.py
│   │   └── ball_estimator.py          # 6D 卡尔曼滤波器（位置+速度，匀速+重力过程模型）
│   ├── tennis/                        # 网球场景相关
│   │   ├── __init__.py
│   │   ├── ball.py                    # 网球抛物线轨迹预测
│   │   └── hitting.py                 # 击打点计算 & 球拍-球接触判断
│   ├── real/                          # 真机部署模块（纯真机接口，不含 MuJoCo 仿真）
│   │   ├── __init__.py
│   │   ├── config.py                  # RealRobotConfig（控制器安全参数/PD增益/感知配置，from_yaml）
│   │   ├── robot_interface.py         # Realman SDK 封装（rm_movej_follow 角度控制 + 连接时配置控制器安全）
│   │   ├── torque_to_position.py      # 力矩→位置积分器（备用，位置模式不用）
│   │   ├── adaptive_timer.py          # 在线自适应频率控制（EMA 平滑）
│   │   ├── safety_monitor.py          # 软件层安全检查（关节位置/速度/TCP 超限）
│   │   ├── ball_sensor.py             # BallSensor ABC + SimulatedBallSensor（动捕/相机抽象接口）
│   │   └── ball_perceiver.py          # BallPerceiver（sensor → 有限差分速度 → KF 滤波 → pos/vel）
│   └── utils/
│       ├── __init__.py
│       ├── math_utils.py              # 通用数学工具
│       ├── mujoco_loader.py           # 跨平台安全模型加载器（处理中文路径）
│       └── noise.py                   # 噪声注入（观测/力矩/初始关节随机化，支持 per-axis std + Z clamp）
├── configs/
│   ├── default.yaml                   # 默认超参数（时间步长、iLQR 参数、关节约束）
│   ├── mpc.yaml                       # MPC 专用参数
│   ├── cost_hitting.yaml              # 代价函数权重
│   ├── v4_follow_through.yaml         # V4 随挥策略配置
│   ├── v5_active_hit.yaml             # V5 主动击球配置
│   └── real_robot.yaml                # 真实机器人配置（底座位姿、坐标系标定、控制频率）
├── scripts/
│   ├── rm65_mpc_tube_constraint.py               # 离线仿真（根，被 exp/ 包装 import）
│   ├── rm65_mpc_tube_constraint_realtime.py      # 实时 v1（根）
│   ├── rm65_mpc_tube_constraint_realtime_v2.py   # 实时 v2（根）
│   ├── rm65_mpc_tube.py / rm65_mpc_ilqr_5_5.py  # Tube/iLQR 基线（根）
│   ├── rm65_evaluate.py                          # 评估脚本（根）
│   ├── rm65_mpc_v6.py                            # V6 仿真主脚本（被 run_exp_* subprocess 调用）
│   ├── rm65_mpc_v7.py                            # V7 仿真主脚本（被 run_exp_* subprocess 调用）
│   ├── rm65_mpc_v8.py                            # V8 仿真主脚本（被 import + subprocess 调用）
│   ├── rm65_mpc_v9.py                            # V9 仿真主脚本（解耦 Tube/Softmin + ablation 模式）
│   ├── rm65_mpc_v10.py                           # V10 仿真主脚本（V9 去随挥 + 40cm 终端偏移）
│   ├── rm65_mpc_v11.py                           # ★ V11 仿真主脚本（最新迭代：bug修复 + sigmoid 权重调度）
│   ├── run_20hits_video.py                       # 连续 20 次击打视频生成脚本
│   ├── sim/            # 独立仿真（v4/v5/v8v9变体/fast/ilqt/train）
│   ├── exp/            # 实验设施 52 个（包装·批量·运行器）
│   ├── extract/        # 结果提取 9 个（日志→CSV）
│   ├── plot/           # 论文图表 14 个
│   ├── tools/          # 独立工具 10 个（查看器·扫描·诊断·可视化）
│   ├── test/           # 快速验证 10 个
│   └── README.md       # 完整清单与说明
├── tests/
│   ├── test_kinematics.py
│   ├── test_linearize.py
│   ├── test_mpc.py
│   ├── test_ball.py
│   ├── test_noise.py
│   ├── test_ball_estimator.py              # BallEstimator 单元+集成测试（16 tests）
│   ├── test_estimator_pipeline.py          # 感知 pipeline 端到端测试（7 tests）
│   ├── test_actuator_modes.py              # 双模式执行器测试（46 tests）
│   ├── test_jacobian_cache.py              # 雅可比缓存回归测试（8 tests）
│   ├── test_cpp_forward_pass.py            # C++ 前向传递等效性测试（14 tests）
│   └── test_cpp_backward_pass.py           # C++ 后向传递等效性测试（4 tests）
├── experiment_data/                  # 实验数据（按 exp1~exp12 组织）
│   └── README.md                     # 数据存储规范
├── paper/                            # 论文 LaTeX 工程
│   ├── main.tex
│   ├── references.bib
│   ├── sections/                     # 英文各节 .tex
│   ├── sections_zh/                  # 中文草稿 .md
│   └── figures/                      # 图表
├── docs/                             # 项目文档
│   ├── README.md                     # 文档索引与阅读推荐
│   ├── experiments/                  # 实验设计与报告
│   │   ├── README.md                 # 实验索引（exp1~exp12）
│   │   ├── design/                   # 实验设计文档（目的/假设/参数）
│   │   └── reports/                  # 实验报告（日期前缀，结果/分析）
│   ├── architecture_and_algorithm.md # 系统架构与算法详解
│   ├── cpp_acceleration.md           # C++ 加速模块构建指南
│   ├── rm65_evaluate_usage.md        # 评估脚本使用说明
│   ├── rm65_mpc_ilqr_core.md         # MPC+iLQR 核心机制
│   ├── rm65_mpc_ilqr_pipeline.md     # MPC+iLQR Pipeline
│   ├── rm65_tennis_report.md         # 项目技术总览
│   └── rm65_mpc_fast_group_report.md # 组会汇报材料
├── results/                          # 输出目录（日志、轨迹、视频）
└── requirements.txt
```

## Skills 参考

项目包含 8 个 Skill，位于 `skills/` 目录。Agent 在对应场景下应加载相应 Skill：

| Skill | 文件 | 用途 | 触发条件 |
|-------|------|------|---------|
| 代码框架设计 | `skills/framework_design.md` | 设计代码架构、模块归属、接口定义 | 创建/重构模块时 |
| 文件管理 | `skills/file_management.md` | 文件创建/移动/命名规范、目录结构映射 | 添加/移动文件时 |
| 仿真运行 | `skills/sim_run.md` | iLQT 训练、MuJoCo 评估、轨迹回放 | 启动仿真/训练时 |
| 实验设计与数据管理 | `skills/experiment_design.md` | 批量运行脚本模板、CSV/NPZ 数据规范、三层架构（包装/运行/提取） | 运行批量实验时 |
| 论文图表生成 | `skills/figure_generation.md` | 8 张 IEEE RAL 论文图（系统/算法/关节/命中率/Tube/实时/诊断）| 生成论文图表时 |
| 论文撰写 | `skills/paper_writing.md` | IEEE RAL 结构、中文草稿→英文翻译、符号表 | 撰写论文时 |
| 论文审稿与迭代 | `skills/paper_review.md` | 6 维自审查单、审稿报告模板、迭代工作流 | 审查论文草稿时 |
| 实验记录 | `skills/experiment_log.md` | 实验后自动生成记录：读取CSV→聚合统计→Agent 数据观察结论+人工分析决策→更新索引 | 运行批量实验后 / "记录实验" |

### 实验数据目录
- 所有实验数据存放在 `experiment_data/` 目录
- 按 `exp1~exp12` 编号组织，每组含 `config.yaml` + `results.csv` + `raw/`
- 实验设计文档: `docs/experiments/design/expN_*.md`
- 实验报告: `docs/experiments/reports/YYYY-MM-DD_expN_*.md`
- 详见 `experiment_data/README.md` 和 `docs/experiments/README.md`

## 核心算法说明

### MPC + iLQR + Tube 三层框架

- **MPC 外循环**: 每 `replan_interval` 步重规划，分 far/mid/near 三阶段自适应迭代次数
- **iLQR 内循环**: 后向传递计算增益 K_k, k_k + 前向传递线搜索更新轨迹
- **Tube 鲁棒层**: 空间走廊式代价（不绑定时间-空间对应），候选击球窗口以 best_k 为中心

### iLQR（迭代线性二次调节器）
- **后向传递**: Riccati 递推，从终端时刻到初始时刻计算增益矩阵 K_k, k_k
- **前向传递**: 用线搜索更新轨迹和控制序列（MPC 模式跳过线搜索，固定 alpha=0.5）
- **终端代价**: 惩罚末端执行器偏离期望击打点（位置 + 速度 + 法向量）
- **运行代价**: 惩罚过大的控制力矩 + 关节加速度 + 控制变化率
- **正则化**: Levenberg-Marquardt 风格（mu_min=1e-6, mu_max=1e10, delta_0=1.6）
- 终端代价形式：
  ```
  l_terminal(x) = ||p_ee(x) - p_hit||^2_Q_p + ||v_ee(x) - v_hit||^2_Q_v + (1 - n_racket·n_des) * Q_n
  ```
  其中 p_ee 为末端位置，p_hit 为击打点位置，v_ee 为末端速度，v_hit 为期望击打速度

### Tube-based Robust Hitting
- 不确定性管道：σ(t) = σ₀ + σᵥ·t + σₐ·t²
- 候选击球窗口：以 best_k 为中心，window_half_ms 为半宽（默认 50ms）
- 空间走廊式代价（不绑定时间-空间对应）：
  1. 垂直偏离代价（hinge loss）：球拍超出走廊半径即惩罚
  2. 速度方向代价（球拍沿球轨迹线方向运动）
  3. 法向量代价（拍面朝向来球方向）
- Softmin 终端聚合：多个候选终端代价加权，β 控制锐度
- 不确定性管道：σ(t) = σ₀ + σᵥ·t + σₐ·t²

### 多层安全滤波
- **X 平面墙预判**：臂不越过身体中线（X≥-0.1），越界 PD 推回
- **关节约束**：位置/速度/加速度/力矩四重限制
- **TCP 速度硬限制**：max_tcp_speed = 1.8 m/s
- **逐步安全滤波**：β = [0.8, 0.6, 0.4, 0.2, 0.0]，找到最大可行控制
- **终段豁免**：击球前 terminal_exempt_steps 步跳过速度检查（默认 20 步）
- **紧急制动**：所有 β 均失败时施加阻尼力矩 u = -20·qdot（力矩模式）；保持当前角度（位置模式）

### 双模式执行器（Stage -1，已集成至 V11）
- **力矩模式（默认）**：`u = tau`，MuJoCo `actuator` 直接输出力矩到关节。B 矩阵下半块 = `dt * M^{-1}`
- **位置模式（`--position-mode`）**：`u = q_desired`，MuJoCo `general` 执行器内部 PD 闭环 `tau = Kp*(u - q) - Kd*qdot`。B 矩阵下半块 = `dt * M^{-1} * diag(Kp)`，A 矩阵含额外 `-M^{-1}*Kp` 和 `-M^{-1}*Kd` 项
- `configs/default.yaml` 中 `actuator` 节管理 `kp`/`kd` 参数，V11 通过 `env.configure_actuator_mode("position", kp, kd)` 动态切换
- **beta 缩放→位置插值**：力矩模式 `beta * u` 缩放控制量 → 位置模式 `q + beta*(u - q)` 插值（beta=0 保持当前角度）
- **随挥 IK**：位置模式用 `env.solve_ik()` 替代力矩模式的 `J^T*F` PD 控制器
- **TCP 速度检查（位置模式）**：用 `step_from_state` 预测 + 位置插值限速（替代力矩模式的直接缩放）

### 动力学线性化
- 通过 MuJoCo 的 `mj_jac` 和 `mj_rne` 计算雅可比和动力学偏导
- 解析线性化（C++ `linearize_analytical_batch`）— MPC 默认
- 有限差分法数值线性化（开发初期，`--fd` 标志）— 较慢但更鲁棒
- 线性化结果：x_{k+1} ≈ f(x_k, u_k) = A_k δx + B_k δu + ...

### 网球轨迹预测
- 假设网球在重力作用下做抛物线运动（忽略空气阻力）
- 给定球的初始位置和速度，预测球在任意时刻的位置
- 计算击打时刻和击打点：球到达球拍可及范围内的时间点
- serve_box 模式：从 8m×0.2m×0.3m 范围内随机发球

### 噪声注入（`src/utils/noise.py`）
- **模块状态**：已开发，测试通过（17 tests），已通过 exp7（噪声×Tube 消融）和 exp8（KF 恢复）实验集成验证
- **三个纯函数**：
  - `add_observation_noise`：球位置/速度观测噪声，支持标量 std（向后兼容）和 per-axis std（各向异性，如深度方向误差更大），per-axis 优先；Z 坐标 clamp ≥ 0.01m 防止球在地下
  - `add_torque_noise`：力矩执行噪声（暂不在实验中使用）
  - `randomize_init_q`：初始关节角度随机化
- **接口设计**：
  ```python
  # 标量模式（向后兼容）
  add_observation_noise(pos, vel, rng, pos_std=0.05, vel_std=0.3)
  # per-axis 模式（Y 轴深度方向误差更大，模拟深度相机特性）
  add_observation_noise(pos, vel, rng, pos_std_xyz=(0.02, 0.05, 0.02))
  ```
- **噪声特性**：零均值高斯、独立同分布、seed 可复现、不修改输入数组
- **未建模的二阶效应**（经评估暂不修复）：位置/速度相关性、距离相关 std、速度裁剪
- **集成验证**：`scripts/test/test_noise_integration.py`（一次性工具，不进 git），σ_p=0.03/σ_v=0.3 下规划成功率下降 15%，p_hit 偏差均值 125mm

### 感知模块（`src/perception/ball_estimator.py`）
- **模块状态**：已开发，测试通过（16 单元+集成 tests + 7 端到端 pipeline tests），已永久集成到 `RM65Env`
- **架构**：6D 线性卡尔曼滤波器，状态 x = [px, py, pz, vx, vy, vz]，全状态观测
- **过程模型**：匀速 + 重力（g=9.81），F 矩阵考虑重力加速度对速度的衰减
- **观测模型**：全状态直接观测（位置+速度），H = I₆
- **弹跳保护**：观测 Z < 0.01m 时将位置 slam 为 min(z_obs, 0.01)，速度 Vz > 0 时保持（反弹），Vz ≤ 0 时置零（落地）
- **per-axis R 矩阵**：观测噪声协方差支持标量（各向同性）和 per-axis std（各向异性，精确匹配 exp8 五级噪声）
- **RM65Env 集成方式**（零侵入，默认关闭）：
  ```python
  env = RM65Env(estimator_config={"enabled": True, "obs_pos_std": 0.05, "obs_vel_std": 0.3})
  pos, vel = env.get_ball_state()   # 返回 KF 滤波后的估计值
  pos = env.get_ball_pos()          # 快捷方法
  vel = env.get_ball_vel()          # 快捷方法
  ```
  - `estimator_config=None`（默认）：直接读真值，零开销
  - `estimator_config` 启用时：`reset()` 自动清空 estimator，`get_ball_state()` 注入噪声→KF update→返回估计
- **dt 时序陷阱与修复**（exp8 墙钟时间 bug）：
  - 问题：`BallEstimator.update()` 用 `perf_counter()` 墙钟时间（~20ms）作为预测 dt，物理仅 5ms → 66mm 系统偏差
  - 修复：wrapper 中每次 `update` 前强制 `_last_update_time = perf_counter() - dt` → 偏差降至 7.5mm
  - 根本原因：KF 内部使用墙钟时间假设实时调用，但仿真中物理步长固定 5ms
- **exp8 核心结论**（10,000 runs, 2h27min）：
  - off+kf: 52.2-52.6%（性能税 -20~29pp，根因 7.5mm 残留偏差 + 每步 2 次 update）
  - lo+kf: 13.4-15.2%（绝对恢复 +12-14pp, 相对恢复 ~17%）
  - anis+kf: 8.0-10.8%（意外好，per-axis R 建模有效）
  - mid/hi+kf: 1.2-5.2%（恢复微弱）
  - Tube 无交互效应（与 exp7 一致）
- **三层感知架构**：仿真层真值 → 实验层噪声注入（`add_observation_noise`）→ 感知层滤波（`BallEstimator`）→ 规划层消费（`get_ball_state`）
- **测试文件**：
  - `tests/test_ball_estimator.py`：16 个单元+集成测试（初始化/预测/更新/弹跳/per-axis R/收敛性）
  - `tests/test_estimator_pipeline.py`：7 个端到端 pipeline 测试（噪声→KF→规划链路验证）

## 优化策略
- **雅可比转置初始控制（JT warm-start）**：使用 `J^T * (p_hit - p_ee)` 生成初始控制序列，远优于零/常数力矩初始猜测
- **分阶段迭代**：far 阶段仅 JT 控制（零 iLQR 开销），near 阶段减少迭代数 + 启用 hard_constraints
- **R 退火**：控制代价 R 从击球前逐步衰减（r_decay_ratio=0.40），关节1额外衰减 10×
- **后摆策略**：五次多项式后摆轨迹，增大挥拍行程达到更高末端速度
- **随挥（V5）**：击球后 60 步（300ms）内末端沿来球反方向加速，随挥长度 0.5m
- **权重调度**：far 阶段 Q_p×5, Q_v×3；near 阶段 Q_p×8, Q_v×120

## 构建与运行命令
- 创建 conda 环境: `conda create -n mujoco_tennis python=3.11 -y`
- 激活环境: `conda activate mujoco_tennis`
- 安装依赖: `pip install -r requirements.txt`
- 编译 C++ 扩展: `python setup.py build_ext --inplace`
- 运行 MPC 仿真（当前活跃版本）: `python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7`
- 位置模式仿真: `python scripts/rm65_mpc_v11.py --serve-box --ball-speed 7 --position-mode`
- 离线测试: `python scripts/rm65_mpc_tube_constraint.py --serve-box --ball-speed 9`
- 关节安全扫描: `python scripts/scan_joint_safety.py`
- 运行测试: `pytest tests/`
- 代码检查: `ruff check src/ tests/ scripts/`
- 类型检查: `mypy src/`
- 真实部署: `python scripts/run_real_robot.py`
- 底座标定: `python scripts/tools/calibrate_base.py`

## 批量实验架构

新增实验采用**主 Agent 创建脚本 + experiment-runner subagent 启动**的分工模式。
脚本模板参考见 `skills/experiment_design.md` § 新建实验工作流。

### 三层架构

| 层 | 文件模板 | 谁负责 | 职责 |
|----|---------|--------|------|
| 包装 | `scripts/exp/_run_expN_*.py` | 主 Agent | monkey-patch 约束 → 构建 sys.argv → 调主脚本 |
| 运行 | `scripts/exp/run_expN_batch.py` | 主 Agent | 遍历参数矩阵 → subprocess → UTF-8 日志 |
| 提取 | `scripts/extract/extract_expN_results.py` | 主 Agent | regex 解析日志 → results.csv + 命中率汇总 |
| 启动 | tmux 后台 | experiment-runner subagent | mkdir → tmux new-session → 立即返回 |

### 参考实现（复制即改）

| 实验类型 | 包装参考 | 运行参考 | 提取参考 |
|---------|---------|---------|---------|
| 豁免约束 + 离线 | `_run_exp1_v3_exempt.py` | `run_exp1_v3_batch.py` | `extract_exp1_v3_results.py` |
| 严格约束 + 离线 | `_run_exp2_v3_strict.py` | `run_exp2_v3_batch.py` | `extract_exp2_v3_results.py` |
| V11 + 实时（有 `__RESULT__`） | 不需要 | `run_exp11_batch.py` | `extract_exp11_results.py` |

### 新建实验流程

1. **主 Agent 建目录 + 脚本**：创建 `experiment_data/expN_<name>/raw/` + `config.yaml`，编写设计文档 `docs/experiments/design/expN_<name>.md`，编写包装/运行器/提取脚本（参考 `skills/experiment_design.md` § 新建实验工作流）
2. **主 Agent dispatch subagent**：传入 `experiment_id`、`data_dir`、`raw_dir`、`batch_script`、`extract_script`、`workers` 6 个参数
3. **subagent 启动 tmux**：执行 4 个 bash 命令（预检→mkdir→断点检查→tmux），立即返回确认信息
4. **主 Agent 后续检查**：`test -f <DATA_DIR>/_.COMPLETE && echo DONE || echo RUNNING`

### 常见坑

| 坑 | 现象 | 原因 | 解决 |
|----|------|------|------|
| UTF-16LE 日志 | regex 匹配不到中文 | 旧 PowerShell `Tee-Object` 编码 | 已迁移到 Python `subprocess` + `encoding="utf-8"`（V11 batch 脚本） |
| 离线脚本无 `__RESULT__` | 提取脚本报 KeyError | 离线脚本只输出 step log | 用离线专用提取脚本（解析 `球拍击球!` 行） |
| monkey-patch 不生效 | 约束未改变 | import 顺序错误 | patch 必须在 `import main_mod` **之前** |
| 并行跑崩 | MuJoCo segfault | 多进程共享 GL context | 确保 `--no-plot` 关掉所有渲染 |

## 编码规范
- **所有代码注释、docstring 使用中文**
- 使用 `numpy` 进行数组运算，禁止对数组使用原生 Python 循环
- MuJoCo 模型定义在 `src/robot/rm65_model.xml`，是 DOF 数和关节顺序的唯一事实来源
- 可调参数放在 `configs/*.yaml` 中，不要硬编码
- 日志使用 Python `logging` 模块，不要用 `print`
- 测试文件与 `src/` 结构对应，放在 `tests/` 下
- 导入使用绝对路径：`from src.ilqt.solver import ILQTSolver`
- 文件路径使用 `pathlib.Path`，不拼接字符串

## 跨平台注意事项
- **Windows**: MuJoCo 查看器原生支持，使用 `mujoco.viewer.launch_passive()`
- **Ubuntu**: 同样 API；无头服务器上设置 `MUJOCO_GL=osmesa` 或 `egl`
- 不要使用 `glx` 或平台特定的渲染调用
- 文件路径使用 `pathlib.Path`，不要拼接字符串
- **中文路径问题**：Windows 上 MuJoCo C 层 (`mj_loadXML`) 无法打开含非 ASCII 字符的路径。
  所有 `mujoco.MjModel.from_xml_path()` 调用必须替换为 `load_mujoco_model()`（位于 `src/utils/mujoco_loader.py`），
  该函数在 Win32 + 非 ASCII 路径下自动复制模型到临时 ASCII 目录加载，Linux 上直接加载零开销。

## MuJoCo 关键注意事项
- **`range` 属性使用角度（degrees）**：MuJoCo 3.8+ 的 `range` 属性以度为单位，会自动转换为弧度。例如 `range="-180 180"` 对应 ±π，而非 `range="-3.14 3.14"`（后者仅给出 ±3.14°）
- **`ctrlrange` 与力矩裁剪**：前向传递和 `env.step()` 中必须使用 `model.actuator_ctrlrange` 裁剪控制，不要硬编码
- **自碰撞**：手臂和躯干的 geom 需设置 `contype="0" conaffinity="0"` 以避免碰撞约束阻止运动
- **积分器**：使用 `integrator="implicitfast"` 可提高大扭矩下的仿真稳定性
- **模型加载**：始终通过 `src/utils/mujoco_loader.py` 的 `load_mujoco_model()` 加载模型，而非直接调用 `mujoco.MjModel.from_xml_path()`

## 核心算法脚本参考

本项目的核心算法实现在以下脚本中：

| 脚本 | 用途 | 关键特性 |
|------|------|---------|
| `scripts/rm65_mpc_tube_constraint.py` | 离线仿真主脚本 | MPC+iLQR+Tube+硬约束+X平面墙 |
| `scripts/rm65_mpc_v11.py` | ★ 最新版本（V11） | V9 基础 + X平面墙修复 + sigmoid 权重调度 + 远段轻量 iLQR + `--position-mode` 双模式 |
| `scripts/rm65_mpc_v10.py` | V10 仿真主脚本 | V9 去随挥 + 40cm 终端偏移，用于消融对比 |
| `scripts/rm65_mpc_v9.py` | V9 仿真主脚本 | 解耦 Tube 走廊 + Softmin 终端，`--ablation` 消融模式 |
| `scripts/rm65_mpc_v8.py` | V8 仿真主脚本 | 解耦 Tube 走廊 + Softmin 终端，`--no-tube`/`--no-softmin` |
| `scripts/rm65_mpc_v7.py` | V7 仿真主脚本 | V6 + 击球点终端 + TCP/关节硬约束 |
| `scripts/rm65_mpc_v6.py` | V6 仿真主脚本 | 满秩 Q_v + 来球反方向 + softmin + PD 随挥 |
| `scripts/rm65_mpc_tube_constraint_realtime_v5.py` | 实时 v5（sim/） | 主动击球+随挥+空间走廊Tube+多层安全滤波+异步重规划 |
| `scripts/rm65_mpc_tube_constraint_realtime.py` | 实时仿真 v1 | 异步重规划+buffer机制 |
| `scripts/exp/run_tcp_limit_experiment_v3.py` | TCP 限速实验 | monkey-patch 安全滤波器注入 TCP 检查 |
| `scripts/exp/_run_exp7_kf.py` | exp8 KF 过滤包装 | estimator 模块级变量 + dt 强制修正 + 噪声互斥 assert |
| `scripts/sim/rm65_mpc_ilqt.py` | 简化 MPC+iLQR | 无 Tube，基础两阶段 iLQR |
| `scripts/sim/train_ilqt.py` | 离线训练入口 | 单次 iLQR 优化 + 保存轨迹 |
| `scripts/tools/rm65_joint_viewer.py` | 关节调节查看器 | position 执行器，拖动滑条控制关节角 |

## 真实部署架构（Real Robot Deployment）

### 概述
将 MPC+iLQR+Tube 框架从 MuJoCo 仿真迁移到真实 RM-65B 双臂机械臂。
核心挑战：力矩→位置控制转换、动捕感知、坐标系标定、动力学差异。

- **控制模式**: 位置控制（角度指令），MPC 用 MuJoCo PD 执行器规划 q_desired 轨迹
- **感知**: 动捕/相机（待选定）追踪网球位置 → 卡尔曼滤波 → 抛物线轨迹预测
- **规划计算**: `PlanningEnv`（`src/ilqt/planning_env.py`）基于 MuJoCo 做 FK/Jacobian/前向仿真
- **坐标系**: 通过 `configs/real_robot.yaml` 标定真实底座位姿（位置+旋转）

### 模块结构
```
src/real/                              # 真机部署模块（纯真机接口，不含 MuJoCo 仿真）
├── __init__.py
├── config.py                          # RealRobotConfig（控制器安全参数/PD增益/感知配置，from_yaml）
├── robot_interface.py                 # Realman SDK 封装（rm_movej_follow 角度控制 + 连接时配置控制器安全）
├── torque_to_position.py             # 力矩→位置积分器（备用，位置模式不用）
├── adaptive_timer.py                 # 在线自适应频率控制（EMA 平滑）
├── safety_monitor.py                 # 软件层安全检查（关节位置/速度/TCP 超限）
├── ball_sensor.py                    # BallSensor ABC + SimulatedBallSensor（动捕/相机抽象接口）
└── ball_perceiver.py                  # BallPerceiver（sensor → 有限差分速度 → KF 滤波 → pos/vel）
src/ilqt/
└── planning_env.py                    # MPC 规划计算环境（MuJoCo 纯计算，不接触真机硬件）
configs/
└── real_robot.yaml                    # 真机配置（7节+丰富注释，实验时频繁调整）
```

### 各模块职责

#### `config.py` — RealRobotConfig
- 控制器安全参数：collision_stage / torque_limit / max_tcp_speed / max_line_acc
- 位置模式 PD：kp / kd / enable_feedforward
- 感知参数：sensor_type / pos_noise_std / vel_noise_std
- 关节零位偏移：仿真 vs 真实关节零位差异
- `from_yaml()` 支持 YAML 度→弧度自动转换（q_lower/q_upper）

#### `robot_interface.py` — RobotInterface
- 封装 Realman SDK：`rm_get_joint_degree()` 读角度，`rm_movej_follow()` / `rm_movej_canfd()` 写角度
- `connect()` 后自动配置控制器安全参数（碰撞灵敏度/自碰撞/力矩限制/TCP 速度）
- 内部处理弧度↔角度转换，对外统一弧度制
- 关节速度：数值微分（SDK 状态字典实测不可靠，错误码 165/-3）

#### `ball_sensor.py` — BallSensor(ABC) + SimulatedBallSensor
- ABC 接口：`start()` / `stop()` / `get_latest() → (pos, ts)` / `is_running`
- `SimulatedBallSensor`：push 模式（测试+MuJoCo 仿真用）
- 未来实现：`OptiTrackSensor` / `RealSenseSensor`

#### `ball_perceiver.py` — BallPerceiver
- Pipeline: `BallSensor.get_latest()` → 有限差分速度 → `BallEstimator` KF 滤波 → (pos, vel)
- 有限差分噪声自动缩放：σ_v = σ_p·√2/dt_obs
- 挂钟时间同步（修复 exp8 dt 陷阱）
- 过时数据短路（避免零速度注入 KF）

#### `safety_monitor.py` — SafetyMonitor
- 三重检查：关节位置超限 / 关节速度超限 / TCP 速度超限
- 失败时委托 `RobotInterface.slow_stop()` / `emergency_stop()`

#### `planning_env.py` — PlanningEnv（位于 `src/ilqt/`）
- MPC 规划计算环境，基于 MuJoCo 纯计算（不接触真机硬件）
- 实现 `RobotEnv` Protocol：FK / Jacobian / step_from_state / solve_ik
- 无球物理/无左臂PD/无碰撞（全禁用，碰撞由控制器固件负责）
- 支持力矩模式 + 位置模式（`configure_actuator_mode`）

### 三层安全架构
```
Layer 1: 控制器固件（实时，连接时配置）
  rm_set_collision_state / rm_set_self_collision_enable / rm_set_controller_torque_limit / rm_set_arm_max_line_speed

Layer 2: SafetyMonitor（软件，每 tick 检查）
  关节位置/速度/TCP 速度超限 → slow_stop()

Layer 3: 紧急停止（兜底）
  rm_set_arm_stop()（不可恢复）/ rm_set_arm_slow_stop()（缓停）/ 硬件急停按钮
```

### 安全注意事项
1. **紧急停止**：`RobotInterface.emergency_stop()` 调用 `rm_set_arm_stop()`
2. **关节位置限制**：`SafetyMonitor` 检查 q_desired 在 `q_lower`/`q_upper` 范围内
3. **速度限制**：真机比仿真更保守（max_tcp_speed=1.0 m/s）
4. **首次运行**：必须先用 `SimulatedBallSensor` + MuJoCo 端到端验证
5. **渐进步进**：先低速（ball_speed=3）验证，再逐步提高到 7 m/s

### 实施进度
| 阶段 | 状态 | 模块 |
|------|------|------|
| Stage 0 | ✅ | `config.py`（12字段）+ `real_robot.yaml`（7节注释） |
| Stage 1 | ✅ | `robot_env_protocol.py`（Protocol） |
| Stage 2+3 | ✅ | `planning_env.py`（规划计算环境，MuJoCo 纯计算） |
| Stage 4 | ✅ | `robot_interface` + `torque_to_position` + `adaptive_timer` + `safety_monitor` |
| Stage 5 | ✅ | `ball_sensor` + `ball_perceiver` |
| Stage 6 | ⬜ | ★ MPCController 提取（从 V11 重构为共享类） |
| Stage 7 | ⬜ | `real_runner.py`（主循环编排） |

### 待确认事项
- **真机 MuJoCo 模型**：当前使用占位模型（rm65_model.xml），真机模型开发中
- **动捕系统**：待选定，`BallSensor` 设计为抽象基类
- **球拍安装**：真机球拍安装方式需测量确认（与仿真中垂直安装是否一致）
- **底座位姿**：需要实际测量后填入 `real_robot.yaml`
