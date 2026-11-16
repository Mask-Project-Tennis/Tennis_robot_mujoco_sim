"""ProfilePlanningEnv — 规划副本环境 + 球预测适配层（阶段 3 · Task 3.4）。

职责（施工计划 §共享接口契约 RobotEnvironments 条目）：
- 仅封装 ``BoundRobotEnv``（3.3 冻结接口）+ ``BallPredictor``：规划计算
  环境继承 3.3 的全部状态/推进/FK/IK/执行器模式行为，**新增的只有球
  预测绑定**；
- 虚拟球（解析预测存储）：``set_ball_state`` / ``get_ball_state`` /
  ``get_ball_pos`` / ``get_ball_vel`` / ``set_ball_vel`` 与
  ``predict_ball_trajectory``，与 ``PlanningEnv`` 的球状态委托完全同构
  （鸭子兼容，供 do_replan / 3.8 规划链复用）；
- **球预测与物理球存储分离**：本类持有的球状态是 ``BallPredictor``
  的纯内存状态，与执行环境（ProfileSimEnv）的 MuJoCo 物理球无任何
  共享；调用本类球接口不触碰任何 MjData。

规划路径的碰撞语义：工厂交付的 ``ProfilePlanningEnv`` 基于裸模型
（无球 free joint、无显式 pair、几何掩码全 0）—— 禁用球接触、不承担
真实碰撞判定（碰撞保护由执行路径与 3.7 保护对检查负责）。
"""

from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from src.ilqt.ball_predictor import BallPredictor
from src.robot.model_factory import RobotModelBundle
from src.sim.bound_robot_env import BoundRobotEnv

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]


class ProfilePlanningEnv(BoundRobotEnv):
    """MPC 规划副本环境 — BoundRobotEnv + BallPredictor 虚拟球绑定。

    用法::

        planning = ProfilePlanningEnv(bare_bundle)
        planning.set_ball_state(pos, vel)      # 虚拟球（解析预测存储）
        traj = planning.predict_ball_trajectory(pos, vel, n)

    Attributes:
        NQ/NX/NU: 主动臂维数（实例属性）。
        model: 裸模型（无球；由工厂交付时几何碰撞掩码全 0）。
        data: live MuJoCo 数据（仅臂状态/运动学计算）。
    """

    def __init__(self, bundle: RobotModelBundle) -> None:
        """初始化规划副本环境并绑定球预测器。

        Args:
            bundle: 裸模型同源组装结果（load_robot_model 产物；
                无球 free joint，规划不承担球物理）。
        """
        super().__init__(bundle)
        # 球轨迹预测器（解析抛物线；与执行环境物理球分离）
        self._ball = BallPredictor(dt=self.dt)
        logger.debug(
            "ProfilePlanningEnv 初始化: n=%d dt=%.6f（robot_id: %s）",
            self.NQ,
            self.dt,
            bundle.parameters.robot_id,
        )

    # ------------------------------------------------------------ 虚拟球委托

    def set_ball_state(self, pos: Array, vel: Array) -> None:
        """设置虚拟球状态（解析预测用，不触碰任何 MjData）。

        Args:
            pos: 球的当前位置 (3,)。
            vel: 球的当前速度 (3,)。
        """
        self._ball.set_state(pos, vel)

    def get_ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        """返回虚拟球 (pos, vel)（新数组副本）。"""
        return self._ball.get_state()

    def get_ball_pos(self) -> np.ndarray:
        """返回虚拟球位置 (3,)。"""
        return self._ball.get_state()[0]

    def get_ball_vel(self) -> np.ndarray:
        """返回虚拟球速度 (3,)。"""
        return self._ball.get_state()[1]

    def set_ball_vel(self, vel: Array) -> None:
        """仅设置虚拟球速度（不改变位置）。

        Args:
            vel: 球的新速度 (3,)。
        """
        pos, _ = self._ball.get_state()
        self._ball.set_state(pos, vel)

    def predict_ball_trajectory(
        self, ball_pos: Array, ball_vel: Array, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """解析抛物线预测球轨迹（含地面弹跳），不改规划环境 MuJoCo 状态。

        委托给 ``BallPredictor.predict_from``。

        Args:
            ball_pos: 初始位置 (3,)。
            ball_vel: 初始速度 (3,)。
            n_steps: 预测步数。

        Returns:
            (positions (n,3), velocities (n,3))。
        """
        return self._ball.predict_from(ball_pos, ball_vel, n_steps)
