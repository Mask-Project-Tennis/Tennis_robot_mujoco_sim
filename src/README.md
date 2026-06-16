# src/ — 源代码模块

## 模块概览

| 模块 | 职责 | 关键文件 | 符号数 |
|------|------|---------|--------|
| `robot/` | 机器人模型定义（MuJoCo XML + 正运动学） | `rm65_model.xml`, `kinematics.py` | 9 |
| `sim/` | MuJoCo 环境封装（状态读写、仿真步进、雅可比缓存） | `rm65_env.py`, `env.py`, `viewer.py` | 91 |
| `dynamics/` | 动力学线性化（解析/有限差分）+ 前向 rollout | `linearize.py`, `simulate.py` | 13 |
| `ilqt/` | iLQR 求解器 + 代价函数 + 约束 + 异步重规划 | `solver.py`, `cost.py`, `robot_limits.py`, `utils.py` | 144 |
| `cpp/` | C++ 加速模块（pybind11：线性化、前向传递、约束检查） | `solver_cpp.py`, `mujoco_utils.h`, `cost_params.h` | 91 |
| `perception/` | 球状态估计（6D 卡尔曼滤波 + 观测门控） | `ball_estimator.py`, `ball_obs_gate.py` | 31 |
| `tennis/` | 网球抛物线预测 + 击打点计算 + 球拍接触判断 | `ball.py`, `hitting.py` | 24 |
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

- `rm65_model.xml`：MuJoCo XML 模型（双臂 12DOF + 球拍 + 球 freejoint），DOF 和关节顺序的唯一事实来源
- `kinematics.py`：正运动学 / 雅可比矩阵工具函数

### sim/ — 仿真环境

- `env.py`：`MujocoEnv` 基类（模型加载、状态读写、仿真步进）
- `rm65_env.py`：`RM65Env` 双臂环境（力矩/位置双模式、前馈补偿、雅可比缓存、球弹跳、碰撞控制、KF 集成）
- `viewer.py`：MuJoCo 可视化工具

### dynamics/ — 动力学

- `linearize.py`：解析线性化（`linearize_analytical_trajectory`）+ 有限差分 + 快速模式（跳过 H_q/H_qdot）
- `simulate.py`：前向 rollout 工具

### ilqt/ — iLQR 核心

- `solver.py`：纯 Python iLQR 求解器（后向 Riccati + 前向线搜索）
- `cost.py`：`HittingCost` 代价函数（终端 Q_p/Q_v/Q_n + 运行 R/Q_p_running/平滑项/X 墙/body 规避/softmin）
- `robot_limits.py`：`RobotLimits` 约束参数 + `check_step_feasibility`（制动感知 qdot + 滑窗 qddot）
- `utils.py`：前向传递（含 alpha 回退）+ 轨迹指标 + 控制量缩放
- `async_replanner.py`：异步重规划器（后台线程 iLQR + buffer 机制）
- `jt_init.py`：位置模式 JT 初始控制 + 后摆 warm-start
- `costs/`：模块化代价函数基类（`BaseCost` + `HittingCost`）

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

### utils/ — 工具

- `mujoco_loader.py`：跨平台安全模型加载（Windows 中文路径自动复制到临时 ASCII 目录）
- `noise.py`：噪声注入（观测/力矩/初始关节，per-axis std + Z clamp）
- `math_utils.py`：通用数学工具
