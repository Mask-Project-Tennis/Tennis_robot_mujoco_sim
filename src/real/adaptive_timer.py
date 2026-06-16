"""在线自适应频率控制器。

每个 tick 测量实际耗时，EMA 平滑，动态调整目标 dt。
目标利用率 80%（留 20% 余量）。
"""

import time


class AdaptiveTimer:
    """在线自适应频率控制器。

    Args:
        target_hz: 初始目标频率（Hz）。
        utilization: CPU 利用率目标（0-1），低于此值时 sleep 补偿。
    """

    def __init__(
        self,
        target_hz: float = 100.0,
        utilization: float = 0.8,
    ) -> None:
        self._target_dt = 1.0 / target_hz
        self._utilization = utilization
        self._ema_alpha = 0.1
        self._avg_elapsed = self._target_dt
        self._tick_start: float = 0.0

    def tick_start(self) -> None:
        """tick 开始，记录起始时间。"""
        self._tick_start = time.perf_counter()

    def tick_end(self) -> float:
        """tick 结束，返回应 sleep 的秒数。

        EMA 平滑后，若 avg_elapsed / utilization > target_dt，
        说明持续超时，不 sleep（降频运行）。

        Returns:
            应 sleep 的秒数（≥0）。
        """
        elapsed = time.perf_counter() - self._tick_start
        self._avg_elapsed = (
            self._ema_alpha * elapsed
            + (1 - self._ema_alpha) * self._avg_elapsed
        )
        adjusted_dt = max(
            self._avg_elapsed / self._utilization,
            self._target_dt,
        )
        return max(0.0, adjusted_dt - elapsed)

    @property
    def current_hz(self) -> float:
        """当前自适应频率估计值。"""
        return 1.0 / max(self._avg_elapsed, 1e-9)
