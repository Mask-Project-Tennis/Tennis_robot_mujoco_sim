"""RobotEnv Protocol — 仿真/规划环境的共同接口。

iLQT 求解器、代价函数、安全滤波器仅依赖此协议，
实现仿真（RM65Env）与规划计算（PlanningEnv）的零耦合替换。

阶段 3 Task 3.3 起，协议从 RM65 专属假设中解放为维度无关：
- 维度常量 ``NQ/NX/NU`` 允许实现为类变量（旧 RM65 实现，n=6）或
  实例属性（``src/sim/bound_robot_env.py::BoundRobotEnv``，按
  ``ModelBinding`` 维数确定，n=6/7 通吃）；
- ``control_range``：每执行器 (n,2) 控制范围**只读副本**，行序与
  环境的执行器绑定顺序一致（旧实现为前缀 ``[:NU]``，绑定实现为
  ``binding.actuator_ids``）；求解器与限位层按它裁剪控制；
- ``LEFT_ARM_NQ`` / ``init_q_left`` 为 **RM65 双臂旧实现专属属性**
  （RM65Env/PlanningEnv 继续提供），仅供旧 C++ 线性化/回放管线
  （``src/cpp/solver_cpp.py``、回放脚本）消费；它们不属于通用规划
  消费者的必需接口，新实现（BoundRobotEnv）不提供。旧路径保持
  原样，不做迁移。
"""

from typing import Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


@runtime_checkable
class RobotEnv(Protocol):
    """机器人环境协议 — RM65Env、PlanningEnv 与 BoundRobotEnv 的共同接口。

    属性:
        model: MuJoCo 模型（MjModel），用于读取关节范围、actuator 等静态信息。
        data: MuJoCo 数据（MjData），用于读取/写入当前状态及 FK 结果。
        NQ: 主动关节数 n（类变量或实例属性）。
        NX: 状态维数 2n。
        NU: 控制维数 n。
        control_range: (n,2) 每执行器控制范围只读副本（随执行器模式切换）。
    """

    NQ: int
    NX: int
    NU: int

    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    control_range: np.ndarray

    @property
    def dt(self) -> float: ...

    def get_arm_state(self) -> np.ndarray: ...
    def set_arm_state(self, x: np.ndarray) -> None: ...
    def step(self, u: np.ndarray) -> np.ndarray: ...
    def step_from_state(self, x: np.ndarray, u: np.ndarray) -> np.ndarray: ...

    def get_ee_pos(self) -> np.ndarray: ...
    def get_ee_vel(self) -> np.ndarray: ...
    def get_ee_jacp(self) -> np.ndarray: ...
    def get_ee_jacr(self) -> np.ndarray: ...
    def get_ee_normal(self) -> np.ndarray: ...

    def update_kinematics(self) -> None: ...

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_init: np.ndarray | None = ...,
        max_iter: int = ...,
        eps: float = ...,
        damp: float = ...,
        step_size: float = ...,
    ) -> np.ndarray: ...
