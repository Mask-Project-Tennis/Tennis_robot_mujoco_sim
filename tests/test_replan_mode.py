"""重规划模式策略测试 — ReplanMode（同步/异步统一）。

测试 ReplanMode Protocol 的两个实现：
- SyncReplanMode: submit 后结果立即可用
- AsyncReplanMode: 通过 AsyncReplanner 异步轮询
"""

from __future__ import annotations

import numpy as np

from src.ilqt.async_replanner import PlanRequest, PlanResult
from src.ilqt.strategies.replan_mode import (
    AsyncReplanMode,
    ReplanMode,
    SyncReplanMode,
)


def _make_request(step: int = 10) -> PlanRequest:
    """构建测试用 PlanRequest。"""
    return PlanRequest(
        x_current=np.zeros(12),
        ball_pos=np.array([0.0, -1.0, 1.5]),
        ball_vel=np.array([0.0, 2.0, 0.0]),
        step=step,
        k_hit_current=200,
        U_prev=np.zeros((0, 6)),
        p_hit_current=np.array([0.0, -0.5, 1.2]),
        v_hit_desired=np.array([0.0, 1.8, 0.0]),
        n_des_current=np.array([0.0, 1.0, 0.0]),
        is_first_plan=False,
    )


def _make_result(k_hit: int = 150) -> PlanResult:
    """构建测试用 PlanResult。"""
    return PlanResult(
        U_buffer=np.zeros((20, 6)),
        U_prev=np.zeros((10, 6)),
        U_mpc_full=np.zeros((200, 6)),
        k_hit_new=k_hit,
        p_hit_new=np.array([0.1, -0.5, 1.2]),
        v_ball_hit_new=np.array([0.0, -1.0, 0.0]),
        n_des_new=np.array([0.0, 1.0, 0.0]),
        request_step=10,
    )


# ──────────────────────────────────────────────────────────────────
# A3 轮 1: SyncReplanMode
# ──────────────────────────────────────────────────────────────────


class TestSyncReplanMode:
    """同步重规划模式测试。"""

    def test_sync_submit_makes_result_available(self) -> None:
        """submit 后 has_result=True（同步立即完成）。"""
        result = _make_result()
        mode = SyncReplanMode(replan_fn=lambda req: result)
        assert not mode.has_result()
        mode.submit(_make_request())
        assert mode.has_result() is True

    def test_sync_get_result_returns_planresult(self) -> None:
        """get_result 返回 do_replan 的结果。"""
        expected = _make_result(k_hit=123)
        mode = SyncReplanMode(replan_fn=lambda req: expected)
        mode.submit(_make_request())
        result = mode.get_result()
        assert result is not None
        assert result.k_hit_new == 123

    def test_sync_is_busy_always_false(self) -> None:
        """同步模式永不忙（is_busy 始终 False）。"""
        mode = SyncReplanMode(replan_fn=lambda req: _make_result())
        assert mode.is_busy() is False
        mode.submit(_make_request())
        assert mode.is_busy() is False

    def test_sync_get_result_clears_after_read(self) -> None:
        """get_result 后 has_result 变 False（一次性消费）。"""
        mode = SyncReplanMode(replan_fn=lambda req: _make_result())
        mode.submit(_make_request())
        assert mode.has_result()
        mode.get_result()
        assert not mode.has_result()

    def test_sync_is_protocol(self) -> None:
        """SyncReplanMode 实现 ReplanMode Protocol。"""
        mode = SyncReplanMode(replan_fn=lambda req: _make_result())
        assert isinstance(mode, ReplanMode)


# ──────────────────────────────────────────────────────────────────
# A3 轮 2: AsyncReplanMode（mock AsyncReplanner）
# ──────────────────────────────────────────────────────────────────


class _MockReplanner:
    """模拟 AsyncReplanner（用于测试 AsyncReplanMode）。"""

    def __init__(self) -> None:
        self._planning: bool = False
        self._has_new: bool = False
        self._result: PlanResult | None = None
        self.submit_count: int = 0

    def submit(self, request: PlanRequest) -> bool:
        self.submit_count += 1
        self._planning = True
        return True

    def has_new_plan(self) -> bool:
        return self._has_new

    def apply_new_plan(self) -> PlanResult | None:
        if not self._has_new:
            return None
        self._has_new = False
        self._planning = False
        return self._result

    def is_planning(self) -> bool:
        return self._planning

    def _set_result(self, result: PlanResult) -> None:
        """测试辅助：设置规划结果。"""
        self._result = result
        self._has_new = True
        self._planning = False


class TestAsyncReplanMode:
    """异步重规划模式测试。"""

    def test_async_submit_sets_busy(self) -> None:
        """submit 后 is_busy=True（正在规划）。"""
        mock = _MockReplanner()
        mode = AsyncReplanMode(replanner=mock)
        assert not mode.is_busy()
        mode.submit(_make_request())
        assert mode.is_busy() is True

    def test_async_has_result_when_new_plan(self) -> None:
        """replanner 有新结果时 has_result=True。"""
        mock = _MockReplanner()
        mode = AsyncReplanMode(replanner=mock)
        assert not mode.has_result()
        mock._set_result(_make_result())
        assert mode.has_result() is True

    def test_async_get_result_returns_planresult(self) -> None:
        """get_result 返回新规划结果。"""
        mock = _MockReplanner()
        mode = AsyncReplanMode(replanner=mock)
        mock._set_result(_make_result(k_hit=99))
        result = mode.get_result()
        assert result is not None
        assert result.k_hit_new == 99

    def test_async_get_result_clears_busy(self) -> None:
        """get_result 后 is_busy=False。"""
        mock = _MockReplanner()
        mode = AsyncReplanMode(replanner=mock)
        mode.submit(_make_request())
        assert mode.is_busy()
        mock._set_result(_make_result())
        mode.get_result()
        assert not mode.is_busy()

    def test_async_is_protocol(self) -> None:
        """AsyncReplanMode 实现 ReplanMode Protocol。"""
        mock = _MockReplanner()
        mode = AsyncReplanMode(replanner=mock)
        assert isinstance(mode, ReplanMode)

    def test_async_no_submit_when_busy(self) -> None:
        """is_busy 时 submit 不重复提交。"""
        mock = _MockReplanner()
        mode = AsyncReplanMode(replanner=mock)
        mode.submit(_make_request())  # submit once
        mode.submit(_make_request())  # busy, should skip
        assert mock.submit_count == 1
