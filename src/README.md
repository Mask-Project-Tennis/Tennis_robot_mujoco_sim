# src/ — 源代码模块

## 模块概览

| 模块 | 职责 | 关键文件 | 符号数 |
|------|------|---------|--------|
| `robot/` | 机器人模型定义（MuJoCo XML + 正运动学） | `rm65_model.xml`, `kinematics.py` | 9 |
| `sim/` | MuJoCo 环境封装 + 轨迹回放 + 击打检测 | `rm65_env.py`, `env.py`, `viewer.py`, `replay.py`, `hit_detection.py` | 91 |
| `dynamics/` | 动力学线性化（解析/有限差分）+ 前向 rollout | `linearize.py`, `simulate.py` | 13 |
| `ilqt/` | iLQR 求解器 + MPCController + EpisodeRunner + 可插拔策略 + 可组合组件 | `solver.py`, `mpc_controller.py`, `episode_runner.py`, `strategies/`, `components/` | 160 |
| `cpp/` | C++ 加速模块（pybind11：线性化、前向传递、约束检查） | `solver_cpp.py`, `mujoco_utils.h`, `cost_params.h` | 91 |
| `perception/` | 球状态估计（6D 卡尔曼滤波 + 观测门控） | `ball_estimator.py`, `ball_obs_gate.py` | 31 |
| `tennis/` | 网球抛物线预测 + 击打点计算 + 球拍接触判断 | `ball.py`, `hitting.py` | 24 |
| `real/` | 真机部署模块（SDK 封装、安全监控、球感知，不接触 MuJoCo） | `robot_interface.py`, `config.py`, `safety_monitor.py`, `ball_sensor.py` | 35 |
| `utils/` | 通用工具（跨平台模型加载、噪声注入、数学） | `mujoco_loader.py`, `noise.py`, `math_utils.py` | 32 |

## 依赖方向

```
scripts/（调用方，不属于 src/）
  │
  ├── ilqt/ ←── cpp/（C++ 加速：线性化 + 前向传递 + check_step）
  │    │           ↑
  │    ├── dynamics/（A/B 矩阵 + rollout）
  │    │    └── sim/（MuJoCo step + FK + 雅可比）
  │    │         └── robot/（XML 模型 + 运动学）
  │    ├── cost.py → sim/（set_arm_state + get_ee_*）
  │    └── robot_limits.py（约束参数 + 安全滤波）
  │
  ├── perception/ ←── utils/（噪声注入 + 数学工具）
  │    └── ball_estimator.py → sim/（集成到 RM65Env）
  │
  └── tennis/（纯数学，无 MuJoCo 依赖）
       ├── ball.py（抛物线预测）
       └── hitting.py（击打点 + 接触判断）
```

核心依赖链：`ilqt/ → dynamics/ → sim/ → robot/`，横向依赖 `perception/` 和 `tennis/`。

## 关键入口

```python
from src.sim.rm65_env import RM65Env          # 创建仿真环境
from src.cpp.solver_cpp import ILQTSolver     # C++ 加速 iLQR 求解器
from src.ilqt.cost import HittingCost         # 代价函数（Tube + Softmin + 平滑项）
from src.ilqt.robot_limits import RobotLimits # 关节约束参数
from src.perception.ball_estimator import BallEstimator  # 卡尔曼滤波器
from src.tennis.ball import predict_trajectory            # 抛物线预测
from src.utils.mujoco_loader import load_mujoco_model     # 跨平台模型加载
```

## 模块详解

### robot/ — 机器人模型

- `constants.py`：共享常量单一事实来源（DT/INIT_Q/INIT_Q_REAL/INIT_Q_LEFT/SHOULDER_POS/WORKSPACE_RADIUS/KP/KD），仅依赖 numpy
- `rm65_model.xml`：MuJoCo XML 模型（双臂 12DOF + 球拍 + 球 freejoint），DOF 和关节顺序的唯一事实来源
- `kinematics.py`：正运动学 / 雅可比矩阵工具函数

### sim/ — 仿真环境

- `env.py`：`MujocoEnv` 基类（模型加载、状态读写、仿真步进）
- `rm65_env.py`：`RM65Env` 双臂环境（力矩/位置双模式、前馈补偿、雅可比缓存、球弹跳、碰撞控制、KF 集成）
- `viewer.py`：MuJoCo 可视化工具

### dynamics/ — 动力学

- `linearize.py`：解析线性化（`linearize_analytical_trajectory`）+ 有限差分 + 快速模式（跳过 H_q/H_qdot）
- `simulate.py`：前向 rollout 工具

### ilqt/ — iLQR 核心 + 管线架构

- `solver.py`：纯 Python iLQR 求解器（后向 Riccati + 前向线搜索）+ `build_solver` 工厂函数
- `cost.py`：`HittingCost` 代价函数（终端 Q_p/Q_v/Q_n + 运行 R/Q_p_running/平滑项/X 墙/body 规避/softmin）
- `robot_limits.py`：`RobotLimits` 约束参数 + `check_step_feasibility`（制动感知 qdot + 滑窗 qddot）
- `utils.py`：前向传递（含 alpha 回退）+ 轨迹指标 + 控制量缩放
- `async_replanner.py`：异步重规划器（后台线程 iLQR + buffer 机制）
- `jt_init.py`：位置模式 JT 初始控制 + 后摆 warm-start
- `robot_env_protocol.py`：`RobotEnv` Protocol（@runtime_checkable，RM65Env/PlanningEnv 共同接口）
- `planning_env.py`：`PlanningEnv` MPC 规划计算环境（MuJoCo 纯计算，无球/无左臂/无碰撞，供真机管线 iLQR 规划用）
- `tube_types.py`：Tube 数据结构（TubeConfig/HitWindow/HittingTube/BallTrajectoryTube/ReplanState）
- `tube_builder.py`：Tube 构建函数（search_hit_window/build_hitting_tube）
- `tube_cost.py`：代价包装器（TubeHittingCostWrapper/TubeOnlyCost/SoftminOnlyCost）
- `mpc_helpers.py`：JT 初始控制 dispatch + fix_joint5 + R 退火调度
- `replan_core.py`：`do_replan` 完整重规划编排（含 Tube 构建 + iLQR 求解 + warm-start；类型化签名 `do_replan(request, env_plan, state, config, robot_limits, solver)`）
- `ball_predictor.py`：`BallPredictor` 解析抛物线预测（无 MuJoCo 依赖）
- `mpc_controller.py`：★ `MPCController` 可组合规划模块（封装完整规划生命周期，含策略注入）+ `MPCConfig` + `MPCStepResult`
- `episode_runner.py`：★ `EpisodeRunner` 通用管线编排器（4 组件：mpc/perception/safety/executor + 5 hook 插入点，仿真/真机共用）
- `step_context.py`：`StepContext` 步骤上下文（pre_plan/post_plan/post_exec/on_unsafe/on_done hook 间数据传递容器）
- `strategy_config.py`：`StrategyConfig` 策略注入容器（聚合 follow_through/hit_refiner/phase_schedule/direction，None → 默认实现）
- `strategies/`：★ 可插拔策略模块
  - `follow_through.py`：`PlannedFollowThrough`（随挥策略，kp/kd 可通过 MPCConfig 配置）
  - `hit_point_refiner.py`：`HybridRefiner`（击球点后过滤，阈值可通过 MPCConfig 配置）
  - `replan_mode.py`：`ReplanMode`（Sync/Async 重规划模式）
  - `phase_schedule.py`：`DefaultPhaseSchedule`（far/mid/near 阶段调度）
  - `direction.py`：`ReflectDirection`（来球反方向计算）
- `components/`：★ 可组合管线组件
  - `protocols.py`：3 个 Protocol（Perception/Executor/Safety，`@runtime_checkable`；`get_metrics` 归入 ExecutorComponent）
  - `sim_component.py`：★ `SimComponent` 仿真执行+诊断一体化（共享模块，V11/V12 复用，实现 ExecutorComponent）
  - `sim_perception.py`：`SimPerception`（读 MuJoCo 球状态，可选 obs_gate 噪声/KF）
  - `predictive_safety.py`：`PredictiveSafetyFilter`（beta 递降 + X 墙，定义共享常量 `X_WALL_BODY_NAMES`）
  - `basic_safety.py`：`BasicSafetyFilter`（仅限位检查，无预测；构造时 emit `RuntimeWarning`）

### cpp/ — C++ 加速

- `solver_cpp.py`：Python 封装，自动检测 C++ 模块可用性，回退到纯 Python
- `core_ext.cpp`：pybind11 模块入口（Unity Build，#include 各 .cpp）
- `types.h`：常量 + 指针转换 + `set_arm_forward`
- `mujoco_utils.h`：`sim_step`（含 mj_forward + 位置裁剪 + FF + qfrc 管理）
- `cost_params.h`：`StepCheckParams` + `check_step`（q/qdot 制动/u/qddot 滑窗）
- `forward_pass.cpp`：前向传递（含碰撞禁用 + limits 检查 + qdot_hist 环形 buffer）
- `linearize.cpp`：解析动力学线性化（批量，含双模式 B 矩阵）

### perception/ — 感知

- `ball_estimator.py`：6D 线性卡尔曼滤波器（匀速 + 重力过程模型，全状态观测，per-axis R 矩阵，弹跳保护）
- `ball_obs_gate.py`：观测门控（频率控制 + 异常值剔除）

### tennis/ — 网球场景

- `ball.py`：抛物线轨迹预测（无空气阻力）+ serve_box 随机发球
- `hitting.py`：击打点/时刻计算 + 球拍-球接触判断

### real/ — 真机部署模块

纯真机接口模块，不含 MuJoCo 仿真。规划计算由 `ilqt/planning_env.py` 提供。

- `config.py`：`RealRobotConfig` 配置数据类（从 YAML 加载，含控制器安全参数/PD 增益/感知配置）
- `robot_interface.py`：`RobotInterface` Realman SDK 封装（角度控制 IP/CANFD 模式 + 连接时自动配置控制器安全参数）
- `torque_to_position.py`：`TorqueToPositionIntegrator` 力矩→位置积分器（备用，位置模式不用）
- `adaptive_timer.py`：`AdaptiveTimer` 在线自适应频率控制（EMA 平滑）
- `safety_monitor.py`：`SafetyMonitor` 软件层安全检查（关节位置/速度/TCP 超限 → 委托 RobotInterface 急停）
- `ball_sensor.py`：`BallSensor` ABC + `SimulatedBallSensor`（动捕/相机抽象接口）
- `ball_perceiver.py`：`BallPerceiver` 球感知器（sensor → 有限差分速度 → KF 滤波 → pos/vel）
- `robot_arm_protocol.py`：`RobotArmInterface` Protocol（@runtime_checkable，真机/Mock 共同接口）
- `fake_robot.py`：`FakeRobot` Mock 实现（简单一阶动力学，测试用）
- `real_runner.py`：`RealRunner` 真机部署主循环（start/step/stop 分步 + run_episode EpisodeRunner 编排）
- `runner_factory.py`：工厂函数（build_robot_limits + build_solver 重导出 + build_real_robot_mpc_config + 共享常量）
- `robot_executor.py`：`RobotExecutor` 适配器（RobotArmInterface → ExecutorComponent）
- `perception_adapter.py`：`PerceptionAdapter` 适配器（BallPerceiver → PerceptionComponent）
- `safety_adapter.py`：`SafetyAdapter` 适配器（SafetyMonitor → SafetyComponent）

### utils/ — 工具

- `mujoco_loader.py`：跨平台安全模型加载（Windows 中文路径自动复制到临时 ASCII 目录）
- `noise.py`：噪声注入（观测/力矩/初始关节，per-axis std + Z clamp）
- `math_utils.py`：通用数学工具
