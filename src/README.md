# src/ — 源代码模块

## 模块概览

| 模块 | 职责 | 关键文件 | 符号数 |
|------|------|---------|--------|
| `robot/` | 机器人模型定义（描述/参数/绑定/依赖闭包/装配 + MuJoCo XML + 运动学） | `model_assets/`, `model_building/`, `model_factory.py`, `robot_profile.py`, `racket_profile.py` | 10+ |
| `sim/` | MuJoCo 环境封装 + 轨迹回放 + 击打检测 + R1D 单臂仿真栈 + V12 配置/回合/产物（旧入口 facade） | `configuration/`, `episode/`, `artifacts/`, `rm65_env.py`, `env.py`, `viewer.py`, `replay.py`, `hit_detection.py`, `v12_robot_adapter.py` | 120+ |
| `dynamics/` | 动力学线性化（解析/有限差分） | `linearize.py` | 5 |
| `ilqt/` | iLQR 求解器 + MPCController + EpisodeRunner + 可插拔策略 + 可组合组件 + R1D 规划环境 | `solver.py`, `mpc_controller.py`, `episode_runner.py`, `profile_planning_env.py`, `strategies/`, `components/` | 160 |
| `cpp/` | C++ 加速模块（pybind11：线性化、前向传递、约束检查） | `solver_cpp.py`, `mujoco_utils.h`, `cost_params.h` | 91 |
| `perception/` | 球状态估计（6D 卡尔曼滤波 + 观测门控） | `ball_estimator.py`, `ball_obs_gate.py` | 31 |
| `tennis/` | 网球抛物线预测 + 击打点计算 + 球拍接触判断 | `ball.py`, `hitting.py` | 24 |
| `real/` | 真机部署模块（SDK 封装、安全监控、球感知 + R1-A7-D 单臂驱动栈，不接触 MuJoCo） | `robot_interface.py`, `config.py`, `safety_monitor.py`, `ball_sensor.py`, `r1d_types.py`, `unitree_r1d_interface.py`, `r1d_factory.py` | 35（阶段 4 前口径；新增 9 个 `r1d_*` 驱动模块未计入） |
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
from src.ilqt.cost import CompositeCost        # 组合代价聚合器（Flyweight + Protocol 委托）
from src.ilqt.robot_limits import RobotLimits # 关节约束参数
from src.perception.ball_estimator import BallEstimator  # 卡尔曼滤波器
from src.tennis.ball import predict_trajectory            # 抛物线预测
from src.utils.mujoco_loader import load_mujoco_model     # 跨平台模型加载
from src.robot.model_factory import load_robot_model      # 兼容 facade（新入口 model_building）
from src.robot.model_assets import collect_model_file_dependencies  # 模型文件依赖闭包
from src.robot.model_factory import collect_model_dependencies  # 兼容 facade（profile→条目）
from src.robot.model_building import RacketAssembly, RobotModelBundle  # 共享结果类型
from src.robot.racket_profile import load_racket_profile   # 球拍档案（几何/质量/COM/惯量/安装/来源，SI）
from src.robot.model_kinematics import ModelKinematics    # 名称映射运动学探针（TCP 位姿/雅可比）
from src.sim.sim_config import load_sim_task_config, validate_simulation_inputs  # R1D 仿真任务配置（阶段 3）
from src.sim.env_factory import build_r1d_environments     # R1D 执行/规划环境组装（阶段 3）
from src.sim.r1d_runner import run_r1d_episode             # R1D 旧单臂闭环入口（阶段 3，兼容）
from src.sim.configuration import resolve_v12_config          # V12 配置解析（新入口）
from src.sim.v12_robot_adapter import build_v12_robot      # V12RobotBundle/SimulationEnv 双机器人适配（T4）
from src.sim.episode import V12EpisodeResult  # 结果类型（轻量）
from src.sim.v12_episode import run_v12_episode  # 兼容 facade（实现已拆到 episode/）
from src.sim.artifacts import write_episode_artifacts  # v2 产物/快照/回放（新入口）
from src.ilqt.profile_planning_env import ProfilePlanningEnv  # R1D 规划侧环境（虚拟球，阶段 3）
from src.real.r1d_factory import build_r1d_arm              # R1D 单臂驱动构造工厂（阶段 4，待实机验收）
```

## 模块详解

### robot/ — 机器人模型

- `constants.py`：共享常量单一事实来源（DT/INIT_Q/INIT_Q_REAL/INIT_Q_LEFT/SHOULDER_POS/WORKSPACE_RADIUS/KP/KD），仅依赖 numpy
- `robot_profile.py`：`RobotProfile` 机器人描述加载器（configs/robots/*.yaml → 身份/模型路径/关节语义名称，纯配置层，不加载 MuJoCo/SDK、不含数值参数；校验 model_path 存在性）
- `racket_profile.py`（T1）：`RacketProfile` 球拍档案加载器（`load_racket_profile(path)`；冻结 dataclass，几何/质量/COM/惯量/法兰安装/连接件/来源，SI 单位，柄底原点 +Z 长轴 +X 法向，TCP=拍面中心；非法输入 ValueError 不回落）
- `model_binding.py`：`ModelBinding` 模型名称绑定器（按名称解析关节/执行器/TCP 的 qpos/qvel 地址，只接受标量关节，校验执行器传动目标与描述维度/重名，只读不修改模型）
- `runtime_parameters.py`：`RobotRuntimeParameters`/`SceneParameters` 运行参数纯参数层（configs/robots/*_runtime.yaml 严格解析 + 维数/单位/能力校验；未知限值显式 null 不补值，不导入 MuJoCo/real/ilqt）
- `model_assets/`：模型文件级依赖扫描（`collect_model_file_dependencies`；引用角色/路径归一化/闭包/缺失清单；不依赖 MuJoCo/NumPy）
- `model_building/`：模型装配（`types.py` 定义 `RacketAssembly`/`RobotModelBundle`；`template.py`/`racket.py`/`scene.py`/`loading.py`；旧工厂 facade 同对象）
- `model_factory.py`：兼容 facade（`write_r1d_model_template`/`load_robot_model`/`collect_model_dependencies`；实现已拆到 `model_assets` + `model_building`）
- `model_kinematics.py`：`ModelKinematics` 名称映射运动学探针（按 binding 地址读写 [q, qdot] 状态，TCP 位姿/解析雅可比；每实例自有 MjData，无动力学 step/控制器/硬件生命周期）
- `rm65_model.xml`：MuJoCo XML 模型（双臂 12DOF + 球拍 + 球 freejoint），DOF 和关节顺序的唯一事实来源
- `r1d_arm_left.xml` / `r1d_arm_right.xml` / `r1d_env.xml` / `r1d_assets.xml`：R1-A7-D 官方整机 URDF 导出片段（`scripts/tools/build_r1d_model.py` 生成，勿手改；环境片段含 bake 的升降 q_L）
- `r1d_model_left.xml` / `r1d_model_right.xml`：R1D 双侧单臂场景模板（`model_factory.py` 生成，勿手改；nq=nv=nu=7）
- `d2_arm.xml` / `d2_assets.xml`：D2 历史资产片段（阶段 2 型号更正后保留，不作 R1-A7-D 验收依据）
- `kinematics.py`：正运动学 / 雅可比矩阵工具函数

### sim/ — 仿真环境

- `env.py`：`MujocoEnv` 基类（模型加载、状态读写、仿真步进）
- `rm65_env.py`：`RM65Env` 双臂环境（力矩/位置双模式、前馈补偿、雅可比缓存、球弹跳、碰撞控制、KF 集成）
- `viewer.py`：MuJoCo 可视化工具
- `hit_detection.py`：击打检测共享模块（`classify_hit_result` 主动/被动/未命中判定）；接触语义（球—拍 geom 对且 `contact.dist<=0`）与主动阈值 `ACTIVE_HIT_FACE_SPEED_THRESHOLD=0.3` 由 `src/ilqt/components/sim_component.py:29,301-332` 定义
- `sim_config.py`（阶段 3）：`SimTaskConfig` + `load_sim_task_config(path, profile)` + `validate_simulation_inputs(profile, parameters, task)`（仿真任务 YAML 严格解析/侧别与工件契约校验，纯配置层）
- `tennis_scene.py`（阶段 3）：`TennisSceneBundle` + `build_tennis_scene(profile, parameters, task)`（球 freejoint/球—拍 pair/球拍与机身碰撞掩码/torso 场景装配）
- `bound_robot_env.py`（阶段 3）：`BoundRobotEnv` 名称绑定单臂环境（`NQ/NX/NU`、`control_range`、`set_arm_state/update_kinematics/get_ee_*`、`step_from_state` 私有 scratch 隔离、力矩裁剪到 `torque_max`）
- `profile_sim_env.py`（阶段 3）：`ProfileSimEnv` 带球执行环境（`step_full` + 球状态读写 + `set_arm_collision` 只切球—拍 pair_margin + `get_racket_face_speed`）
- `env_factory.py`（阶段 3）：`RobotEnvironments` + `build_r1d_environments(profile, parameters, task)`（执行/规划双环境组装、位置模式 PD 同步、球初态注入）
- `collision_checks.py`（阶段 3）：`has_forbidden_contact` + `ForbiddenContactChecker.from_env/has_contact_at_q`（禁止机身—臂接触检查，独立 MjData 零污染）
- `r1d_runner.py`（阶段 3）：`run_r1d_episode(profile_path, runtime_path, task_path, output_dir, *, seed=None)` 旧 R1D 单臂闭环入口（保留兼容；T5 后主路径改走 `v12_episode`）
- `configuration/`（T3）：V12 配置解析（types/validation/selection/defaults/overrides/resolution）；`resolve_v12_config(args, explicit_options)` → `ResolvedV12Config`；RM65 默认→v5→mpc→CLI 顺序，R1D 显式注册表 + 稀疏覆盖
- `v12_config.py`（T3）：兼容 facade（`resolve_v12_config`/`R1D_V12_REGISTRY`；实现已拆到 `configuration/`）
- `v12_robot_adapter.py`（T4）：`SimulationEnv` Protocol + `V12RobotBundle`（execution/planning/limits/init_q/joint_names/几何/保护接触/X 墙/planning_env_factory/backend/racket）+ `build_v12_robot(config)`；R1D 七维地址、异步工厂、estimated joint damping/armature 注入
- `episode/`（T5）：`V12EpisodeResult` + `run_v12_episode(config, ...)` 共享 V12 流程，拆为 preparation/assembly/hooks/post_hit/results/execution 阶段函数；按 bundle 名映射不硬编码 6/7
- `v12_episode.py`（T5）：兼容 facade（`run_v12_episode`/`V12EpisodeResult`；实现已拆到 `episode/`）
- `artifacts/`（T6）：`write_episode_artifacts/load_episode_artifacts/build_replay_bundle/replay_episode_artifacts`；`v12_episode_v2` JSON/NPZ、ContactEvent、源配置/模型快照、资源哈希与回放校验
- `episode_artifacts.py`（T6）：兼容 facade（产物/快照/回放入口；实现已拆到 `artifacts/`）

### dynamics/ — 动力学

- `linearize.py`：解析线性化（`linearize_analytical_trajectory`）+ 有限差分 + 快速模式（跳过 H_q/H_qdot）

### ilqt/ — iLQR 核心 + 管线架构

- `solver.py`：纯 Python iLQR 求解器（后向 Riccati + 前向线搜索）+ `build_solver` 工厂函数
- `cost.py`：`CompositeCost` 组合代价聚合器 + `FKContext` FK 缓存 + `build_production_cost` 工厂
- `cost_terms.py`：8 个独立代价项（ControlEffort/Smoothness/QdotLimit/TcpSoft/TerminalHit/JointLimit/BodyAvoidance/XWall）
- `robot_limits.py`：`RobotLimits` 约束参数 + `check_step_feasibility`（制动感知 qdot + 滑窗 qddot）
- `utils.py`：前向传递（含 alpha 回退）+ 轨迹指标 + 控制量缩放
- `async_replanner.py`：异步重规划器（后台线程 iLQR + buffer 机制）
- `jt_init.py`：位置模式 JT 初始控制 + 后摆 warm-start
- `robot_env_protocol.py`：`RobotEnv` Protocol（@runtime_checkable，RM65Env/PlanningEnv 共同接口）
- `planning_env.py`：`PlanningEnv` MPC 规划计算环境（MuJoCo 纯计算，无球/无左臂/无碰撞，供真机管线 iLQR 规划用）
- `profile_planning_env.py`（阶段 3）：`ProfilePlanningEnv`（`BoundRobotEnv` 子类）R1D 规划侧环境——裸模型 + `BallPredictor` 纯内存虚拟球（`set_ball_state/predict_ball_trajectory` 等，不承担真实碰撞）
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
- `runner_factory.py`：工厂函数（build_robot_limits + build_solver 重导出 + build_real_robot_mpc_config + validate_rm65_profile 纯描述校验 + validate_rm65_runtime 运行能力校验 + load_validated_rm65_binding 加载绑定校验 helper + build_robot_arm 驱动工厂（可选 binding 复用） + 共享常量）
- `robot_executor.py`：`RobotExecutor` 适配器（RobotArmInterface → ExecutorComponent）
- `perception_adapter.py`：`PerceptionAdapter` 适配器（BallPerceiver → PerceptionComponent）
- `safety_adapter.py`：`SafetyAdapter` 适配器（SafetyMonitor → SafetyComponent）

#### R1-A7-D 单臂驱动栈（阶段 4，待实机验收——G3-G5 未启动，未实机验证）

协议事实来源 `configs/robots/r1d_protocol_manifest.yaml`（G1 冻结）；硬件数值载体 `configs/robots/r1d_hardware_{left,right}.yaml`（全 null 待 G4 现场冻结）。验收证据见 `docs/experiments/reports/2026-11-17-unitree-stage4.md`。

- `r1d_types.py`（4.2）：R1D 整机协议数据类型——`R1DStateFrame`/`R1DCommandFrame`/`R1DStatus` + Transport Protocol + hg `LowCmd_` 35 槽位常量（纯数据层，import 零副作用）
- `r1d_mapping.py`（4.2）：`R1DJointMapping` 七关节名称映射——`q_sdk = sign·q_model + offset` 逐关节换算（模型序 = `profile.active_joint_names`，左 15-21/右 22-28），`extract_velocities` 速度换算唯一实现点（速度仅变符号）
- `r1d_config.py`（4.2）：`R1DHardwareConfig` 硬件配置加载/结构校验/purpose 门槛（offline→read_only→motion 梯度收紧；校准/限值/timing null 禁代填）
- `r1d_command_policy.py`（4.3）：`R1DCommandPolicy` 全机命令构造——首帧新鲜实测锚定、变化约束过界明确拒绝不静默截断、非目标槽位（对侧臂/头/腰/升降/底盘/夹爪/空槽）保持策略须全 28 槽显式给出（R5 修订，评审 2026-11-17：显式 > 拒绝——manifest 示例数值不作运行时默认，官方示例 q=0 保持自带 2s 插值防跳变，回退会拉非零姿态跳变）
- `r1d_state_buffer.py`（4.4）：`R1DStateBuffer` 线程安全状态缓存——时效过期 + 会话隔离 + 序号保守语义（重复丢弃/严格递增/≥2^31 回绕/小幅回退丢弃）
- `r1d_transport.py`（4.5）：`SdkR1DTransport` SDK 传输适配——SDK 懒加载（模块 import 零副作用）、open 只读不 acquire/publish、close 幂等 + 世代门；真实路径 unitree_sdk2py + CycloneDDS（本机 11.1.0 离线验证 / 部署目标 0.10.2 锁定）
- `unitree_r1d_interface.py`（4.6）：★ `UnitreeR1DInterface` 驱动接口——生命周期状态机（CLOSED→READ_ONLY→ACTIVE→FAULT 终态）、单一发送线程、五类看门狗（命令过期/状态过期/周期超时/控制权丢失/未确认停止）、发布节奏每控制周期至多一帧（期限自实际发布时刻顺延、不追赶）、发布许可与停止登记同锁复查（未发布帧可撤销）、停止 = 有限帧序列 + 反馈确认（未确认 → FAULT 不退权，退权窗口急停升级并入终态）、发布记录与 capability_notes
- `r1d_factory.py`（4.7）：`build_r1d_arm` 独立构造工厂（R13 冻结签名：必填 `acquire_timeout_s` + `clock` + 驱动 kwargs 透传；构造零传输副作用、失败先于传输创建；RM65 工厂拒绝 R1D profile，FakeRobot 不用于 R1D）
- `r1d_diagnostics.py`（4.8）：`run_r1d_diagnostic` 只读核对/单关节低速正弦 tracking 诊断——JSON verdict + provenance 可追踪指标（含 `frame_aligned`）、tracking 验收阈值必填（缺失→fail 并注明仅完成采集）、误差按帧身份（`received_at`+`sequence`）消费制配对（同一帧重复采样只为一条命令作证）、发布超期判据 = 间隔 > 1.5×期限（半拍容差吸收发布开销）、非整周期回程受限速多拍、缺共享 clock/非目标状态时降级路径如实记录

### utils/ — 工具

- `mujoco_loader.py`：跨平台安全模型加载（Windows 中文路径自动复制到临时 ASCII 目录）
- `noise.py`：噪声注入（观测/力矩/初始关节，per-axis std + Z clamp）
- `math_utils.py`：通用数学工具
