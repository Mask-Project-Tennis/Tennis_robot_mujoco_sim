"""仿真诊断组件 — tube 指标 + 末端-击球点距离 + history。

实现 ``DiagnosticsComponent`` Protocol。可选组件，由 EpisodeRunner 在每拍
``execute`` 后调用 ``record``，在 episode 结束时通过 ``get_metrics`` 汇总。

记录内容：
  - 末端到当前击球点 ``p_hit`` 的欧氏距离序列；
  - 累计步数 ``total_steps``；
  - 最小距离 ``min_dist``（末端最接近击球点的程度）。

设计为轻量、无副作用：仅读取 ``env.get_ee_pos()`` 与 ``result.p_hit``。
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class SimDiagnostics:
    """仿真诊断组件 — 记录末端-击球点距离与累计计数。

    实现 ``DiagnosticsComponent`` Protocol。

    Attributes:
        distances: 每拍的末端-击球点距离序列 (list[float])。
        total_steps: 累计记录步数。
        ball_near_count: 球进入近场范围的步数（预留，当前未累加）。
        tube_ready_count: Tube 就绪步数（预留，当前未累加）。

    Args:
        env: RM65Env / PlanningEnv 实例（须提供 ``get_ee_pos``）。
    """

    def __init__(self, env: object) -> None:
        """初始化诊断组件。

        Args:
            env: 含 ``get_ee_pos`` 的环境实例。
        """
        self._env = env
        # history 列表
        self.distances: list[float] = []
        self.normal_align: list[float] = []
        self.ball_near_count: int = 0
        self.tube_ready_count: int = 0
        self.total_steps: int = 0

    def record(self, result: object, arm_state: NDArray[np.floating]) -> None:
        """记录一步诊断数据。

        读取末端位置与 ``result.p_hit``，追加到距离序列。``result.p_hit``
        为 None 时跳过距离记录但仍计入步数。

        Args:
            result: MPCStepResult 或任意带 ``p_hit`` 属性的对象。
            arm_state: 当前臂状态（预留，当前未使用）。
        """
        self.total_steps += 1
        p_hit = getattr(result, "p_hit", None)
        if p_hit is not None:
            p_ee = self._env.get_ee_pos()  # type: ignore[attr-defined]
            dist = float(np.linalg.norm(p_ee - np.asarray(p_hit)))
            self.distances.append(dist)

    def get_metrics(self) -> dict:
        """返回汇总指标。

        Returns:
            含 total_steps / min_dist / distances / ball_near_count /
            tube_ready_count 的字典。
        """
        return {
            "total_steps": self.total_steps,
            "min_dist": min(self.distances) if self.distances else float("inf"),
            "distances": self.distances.copy(),
            "ball_near_count": self.ball_near_count,
            "tube_ready_count": self.tube_ready_count,
        }
