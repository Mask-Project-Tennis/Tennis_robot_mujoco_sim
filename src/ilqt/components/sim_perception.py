"""仿真感知组件 — 从 RM65Env / PlanningEnv 读球状态。

实现 ``PerceptionComponent`` Protocol，封装 V11 主循环中读取球状态的逻辑。
支持可选的观测门控（obs_gate），用于噪声注入等外部处理。
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

# 观测门控签名: (pos, vel) -> (pos, vel)
ObsGate = Callable[[NDArray[np.floating], NDArray[np.floating]], "tuple[NDArray[np.floating], NDArray[np.floating]]"]


class SimPerception:
    """仿真感知组件 — 从 MuJoCo 环境读球状态。

    实现 ``PerceptionComponent`` Protocol。
    封装 V11 的球状态读取逻辑：

      - 无 ``obs_gate`` 时直接读环境真值（``env.get_ball_state()``）。
      - 有 ``obs_gate`` 时读原始 ``pos``/``vel`` 后过门控（用于噪声注入、KF 等）。

    Args:
        env: RM65Env 或 PlanningEnv 实例（须提供 ``get_ball_state`` /
            ``get_ball_pos`` / ``get_ball_vel`` 接口）。
        obs_gate: 观测门控，两种形态:
            - ``BallObservationGate`` 实例（有 ``get_state(step, pos, vel)``），
              按物理步驱动门控（观测频率/噪声/KF）；
            - 可调用 ``(pos, vel) -> (pos, vel)``（旧语义，无步数上下文）。
            None 时直接读真值。
    """

    def __init__(
        self,
        env: object,
        obs_gate: Optional[ObsGate] = None,
    ) -> None:
        """初始化感知组件。

        Args:
            env: 含球状态接口的环境实例。
            obs_gate: 可选观测门控回调。
        """
        self._env = env
        self._obs_gate = obs_gate

    def get_ball_state(
        self, step: int = 0,
    ) -> Optional[tuple[NDArray[np.floating], NDArray[np.floating]]]:
        """从环境读球状态。

        Args:
            step: 当前物理仿真步数（透传给 BallObservationGate 做
                观测间隔判定；callable 形态的 gate 不使用该参数）。

        Returns:
            (pos(3,), vel(3,)) 元组；环境无数据时返回 None。
        """
        if self._obs_gate is not None:
            # 有门控：读原始分量 → 过门控
            pos = self._env.get_ball_pos()  # type: ignore[attr-defined]
            vel = self._env.get_ball_vel()  # type: ignore[attr-defined]
            if hasattr(self._obs_gate, "get_state"):
                # BallObservationGate: 需要 step 驱动观测间隔/抛物线预测
                return self._obs_gate.get_state(step, pos, vel)
            return self._obs_gate(pos, vel)
        # 无门控：直接读完整球状态（RM65Env 会走缓存/真值）
        return self._env.get_ball_state()  # type: ignore[attr-defined]

    def reset(self) -> None:
        """重置门控状态（新 episode 时调用，无门控则为空操作）。"""
        if self._obs_gate is not None and hasattr(self._obs_gate, "reset"):
            self._obs_gate.reset()
