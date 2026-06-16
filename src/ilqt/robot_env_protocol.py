"""RobotEnv Protocol — 仿真/规划环境的共同接口。

iLQR 求解器、代价函数、安全滤波器仅依赖此协议，
实现仿真（RM65Env）与规划计算（PlanningEnv）的零耦合替换。
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class RobotEnv(Protocol):
    """机器人环境协议 — RM65Env 和 PlanningEnv 的共同接口。

    维度常量 NQ/NX/NU 为类变量（非 property），
    与 MujocoEnv 基类定义一致。
    """

    NQ: int
    NX: int
    NU: int

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
