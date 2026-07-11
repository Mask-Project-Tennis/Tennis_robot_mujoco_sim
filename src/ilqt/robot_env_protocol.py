"""RobotEnv Protocol — 仿真/规划环境的共同接口。

iLQT 求解器、代价函数、安全滤波器仅依赖此协议，
实现仿真（RM65Env）与规划计算（PlanningEnv）的零耦合替换。
"""

from typing import Protocol, runtime_checkable, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import mujoco


@runtime_checkable
class RobotEnv(Protocol):
    """机器人环境协议 — RM65Env 和 PlanningEnv 的共同接口。

    维度常量 NQ/NX/NU/LEFT_ARM_NQ 为类变量（非 property），
    model/data 为 MuJoCo 模型实例。

    属性:
        model: MuJoCo 模型（MjModel），用于读取关节范围、actuator 等静态信息。
        data: MuJoCo 数据（MjData），用于读取/写入当前状态及 FK 结果。
        init_q_left: 左臂初始关节角度，用于双臂模型中设置左臂零位。
        LEFT_ARM_NQ: 左臂自由度数（6），用于 qpos 索引偏移计算。
    """

    NQ: int
    NX: int
    NU: int
    LEFT_ARM_NQ: int
    init_q_left: np.ndarray

    model: "mujoco.MjModel"
    data: "mujoco.MjData"

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
