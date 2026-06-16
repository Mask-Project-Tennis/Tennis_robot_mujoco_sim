"""AdaptiveTimer 单元测试。

测试自适应频率控制器的时序逻辑。
"""

import pytest

from src.real.adaptive_timer import AdaptiveTimer


class TestAdaptiveTimer:
    """自适应频率控制器测试。"""

    def test_over_budget_returns_zero_sleep(self, monkeypatch):
        """耗时 > target_dt → tick_end 返回 0（不 sleep）。"""
        times = iter([0.0, 0.02])
        monkeypatch.setattr(
            "src.real.adaptive_timer.time.perf_counter",
            lambda: next(times),
        )
        timer = AdaptiveTimer(target_hz=100.0)  # target_dt=0.01s
        timer.tick_start()
        sleep_time = timer.tick_end()
        assert sleep_time == 0.0

    def test_under_budget_returns_positive_sleep(self, monkeypatch):
        """耗时 < target_dt → tick_end 返回正值（sleep 补偿）。"""
        times = iter([0.0, 0.005])
        monkeypatch.setattr(
            "src.real.adaptive_timer.time.perf_counter",
            lambda: next(times),
        )
        timer = AdaptiveTimer(target_hz=100.0)
        timer.tick_start()
        sleep_time = timer.tick_end()
        assert sleep_time > 0.0

    def test_ema_smooths_elapsed(self, monkeypatch):
        """多次 tick 后 current_hz 趋于稳定（差异递减）。"""
        time_seq = [
            val
            for i in range(5)
            for val in (i * 0.01, i * 0.01 + 0.005)
        ]
        iterator = iter(time_seq)
        monkeypatch.setattr(
            "src.real.adaptive_timer.time.perf_counter",
            lambda: next(iterator),
        )
        timer = AdaptiveTimer(target_hz=100.0)
        hz_values = []
        for _ in range(5):
            timer.tick_start()
            timer.tick_end()
            hz_values.append(timer.current_hz)
        assert abs(hz_values[-1] - hz_values[-2]) < abs(
            hz_values[0] - hz_values[1]
        )
