"""球轨迹预测器 — 解析抛物线 + 地面弹跳。

纯数学单元，不依赖 MuJoCo。与 RM65Env._handle_ball_bounce 行为一致。
"""

import numpy as np


class BallPredictor:
    """球轨迹预测器 — 解析抛物线 + 地面弹跳。

    纯数学单元，不依赖 MuJoCo。与 RM65Env._handle_ball_bounce 行为一致。
    """

    BALL_RADIUS: float = 0.033       # 网球半径 (m)
    BOUNCE_RESTITUTION: float = 0.8  # 弹跳恢复系数

    def __init__(self, dt: float, g: float = 9.81) -> None:
        """初始化球轨迹预测器。

        Args:
            dt: 仿真时间步长 (s)。
            g: 重力加速度 (m/s²)，向下为正。
        """
        self._dt = dt
        self._g = g
        # 默认状态：原点静止
        self._pos = np.zeros(3)
        self._vel = np.zeros(3)

    def set_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """设置球状态（位置 + 速度）。

        Args:
            pos: 球的当前位置，形状 (3,)。
            vel: 球的当前速度，形状 (3,)。
        """
        self._pos = np.asarray(pos, dtype=float).copy()
        self._vel = np.asarray(vel, dtype=float).copy()

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """返回当前球状态。

        Returns:
            (pos, vel) 元组，均为 (3,) 数组的拷贝。
        """
        return self._pos.copy(), self._vel.copy()

    def predict(self, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        """从当前状态预测 n_steps 步轨迹。

        使用 set_state 设置的状态。
        Returns: (positions (n,3), velocities (n,3))。
        """
        return self._simulate(self._pos, self._vel, n_steps)

    def predict_from(
        self, p0: np.ndarray, v0: np.ndarray, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """从指定初始状态预测（不依赖 set_state）。

        供 PlanningEnv.predict_ball_trajectory(p0, v0, n) 委托调用。

        Args:
            p0: 初始位置 (3,)。
            v0: 初始速度 (3,)。
            n_steps: 预测步数。

        Returns:
            (positions (n,3), velocities (n,3))。
        """
        return self._simulate(p0, v0, n_steps)

    def _simulate(
        self, p0: np.ndarray, v0: np.ndarray, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """从指定初始状态解析积分 n_steps 步。

        每步用解析运动学公式（单步精确）更新：
          v_new = v + a*dt
          p_new = p + v*dt + 0.5*a*dt^2
        其中 a = [0,0,-g]。该更新在多步上严格等价于解析抛物线
        p(t) = p0 + v0*t + 0.5*a*t^2。途中检测地面弹跳。

        Args:
            p0: 初始位置 (3,)。
            v0: 初始速度 (3,)。
            n_steps: 预测步数。

        Returns:
            (positions (n,3), velocities (n,3))。
        """
        dt = self._dt
        # 加速度向量（仅 z 方向，向下）
        a = np.array([0.0, 0.0, -self._g])

        positions = np.empty((n_steps, 3))
        velocities = np.empty((n_steps, 3))

        pos = np.asarray(p0, dtype=float).copy()
        vel = np.asarray(v0, dtype=float).copy()

        for k in range(n_steps):
            # 解析抛物线：用当前速度 + 半加速度积分位置，随后更新速度
            pos = pos + vel * dt + 0.5 * a * (dt ** 2)
            vel = vel + a * dt

            # 弹跳检测：与 RM65Env._handle_ball_bounce 一致
            if pos[2] < self.BALL_RADIUS and vel[2] < 0:
                pos[2] = self.BALL_RADIUS
                vel[2] = -vel[2] * self.BOUNCE_RESTITUTION
                vel[0] *= 0.95
                vel[1] *= 0.95

            positions[k] = pos
            velocities[k] = vel

        return positions, velocities

