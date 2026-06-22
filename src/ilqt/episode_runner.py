"""通用管线编排器 — 组合感知/规划/安全/执行组件。

仿真和真机共用此类，差异仅在注入的组件不同。
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ilqt.mpc_controller import MPCController
    from src.ilqt.components.protocols import (
        PerceptionComponent,
        ExecutorComponent,
        SafetyComponent,
    )

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """通用管线编排器 — 组合 4 个组件。

    管线流程（每步）:
        1. perception.get_ball_state() → (ball_pos, ball_vel)
        2. executor.get_arm_state() → arm_state
        3. mpc.step(ball_pos, ball_vel, arm_state) → MPCStepResult
        4. safety.filter(u_cmd, arm_state) → (safe_u, is_safe)
        5. executor.execute(safe_u)
    """

    def __init__(
        self,
        mpc: "MPCController",
        perception: "PerceptionComponent",
        safety: "SafetyComponent",
        executor: "ExecutorComponent",
    ) -> None:
        """注入 4 个组件。

        Args:
            mpc: MPC 规划控制器。
            perception: 球感知组件。
            safety: 安全滤波组件。
            executor: 执行组件（含指标汇总）。
        """
        self._mpc = mpc
        self._perception = perception
        self._safety = safety
        self._executor = executor

    def run(self, max_steps: int = 500) -> dict:
        """运行完整 episode。

        Args:
            max_steps: 最大运行步数。

        Returns:
            metrics dict，含 total_steps/safe_steps/mpc_done 等。
        """
        # 1. 读取初始状态
        arm_state = self._executor.get_arm_state()
        ball = self._perception.get_ball_state()
        if ball is None:
            logger.error("EpisodeRunner: 初始球状态不可用")
            return {"total_steps": 0, "safe_steps": 0, "error": "no_ball"}

        # 2. 启动 MPC（首次同步规划）
        self._mpc.start(ball[0], ball[1], arm_state)

        # 3. 主循环
        safe_steps = 0
        step_count = 0

        while not self._mpc.done and step_count < max_steps:
            # 感知
            ball = self._perception.get_ball_state()
            arm_state = self._executor.get_arm_state()

            if ball is None:
                logger.warning(f"步 {step_count}: 球状态不可用，退出")
                break

            # 规划
            result = self._mpc.step(ball[0], ball[1], arm_state)

            # 安全滤波
            safe_u, is_safe = self._safety.filter(result.u_cmd, arm_state)
            if not is_safe:
                logger.warning(f"步 {step_count}: 安全检查失败，停止 episode")
                break

            # 执行
            self._executor.execute(safe_u)

            safe_steps += 1
            step_count += 1

        # 4. 清理
        self._mpc.stop()

        # 5. 汇总指标（从 executor 获取）
        metrics: dict = {
            "total_steps": step_count,
            "safe_steps": safe_steps,
            "mpc_done": self._mpc.done,
        }
        if hasattr(self._executor, "get_metrics"):
            metrics.update(self._executor.get_metrics())

        # 关键标量（INFO）+ 完整 metrics（DEBUG，避免大数组刷屏）
        logger.info(
            "EpisodeRunner 完成: total_steps=%s safe_steps=%s mpc_done=%s",
            metrics.get("total_steps"),
            metrics.get("safe_steps"),
            metrics.get("mpc_done"),
        )
        logger.debug("EpisodeRunner metrics 详情: %s", metrics)
        return metrics
