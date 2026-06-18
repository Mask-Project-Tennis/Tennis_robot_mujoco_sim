"""重规划模式策略 — 同步/异步统一接口。

将同步（do_replan 直接调用）和异步（AsyncReplanner 后台线程）两条路径
统一为 ReplanMode 接口，消除 MPCController 中 ~50 行重复的后处理逻辑。

两种模式的核心差异：
- SyncReplanMode: submit() 立即执行 do_replan，结果即时可用
- AsyncReplanMode: submit() 提交到后台线程，结果异步轮询
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Protocol, runtime_checkable

from src.ilqt.async_replanner import PlanRequest, PlanResult

logger = logging.getLogger(__name__)


@runtime_checkable
class ReplanMode(Protocol):
    """重规划模式接口 — 统一同步/异步调度。"""

    def submit(self, request: PlanRequest) -> None:
        """提交规划请求。

        Args:
            request: 规划请求（球状态/臂状态/当前击球点等）。
        """
        ...

    def has_result(self) -> bool:
        """是否有可用的新规划结果。

        Returns:
            True 表示 get_result() 可返回非 None 结果。
        """
        ...

    def get_result(self) -> PlanResult | None:
        """获取并消费规划结果（一次性）。

        Returns:
            PlanResult 或 None（无结果时）。
        """
        ...

    def is_busy(self) -> bool:
        """是否正在规划（不可重复提交）。

        Returns:
            True 表示后台正在规划，应跳过本次 submit。
        """
        ...


class SyncReplanMode:
    """同步重规划模式 — submit 时立即调用 do_replan。

    replan_fn 在 submit() 内同步执行，完成后结果立即可用。
    is_busy() 始终返回 False（同步无等待期）。
    """

    def __init__(self, replan_fn: Callable[[PlanRequest], PlanResult]) -> None:
        """初始化同步模式。

        Args:
            replan_fn: 规划函数 (PlanRequest) → PlanResult。
                       调用方负责绑定 env_plan/replan_state/replan_cfg。
        """
        self._replan_fn: Callable[[PlanRequest], PlanResult] = replan_fn
        self._result: PlanResult | None = None

    def submit(self, request: PlanRequest) -> None:
        """同步执行 do_replan 并存储结果。

        Args:
            request: 规划请求。
        """
        self._result = self._replan_fn(request)

    def has_result(self) -> bool:
        """是否有结果（submit 后 True，get_result 后 False）。"""
        return self._result is not None

    def get_result(self) -> PlanResult | None:
        """获取并清空结果。

        Returns:
            PlanResult（或 None）。
        """
        result = self._result
        self._result = None
        return result

    def is_busy(self) -> bool:
        """同步模式永不忙。"""
        return False


class AsyncReplanMode:
    """异步重规划模式 — 通过 AsyncReplanner 后台线程。

    submit() 提交请求到后台线程，get_result() 轮询结果。
    is_busy() 在规划期间返回 True，防止重复提交。
    """

    def __init__(self, replanner: Any) -> None:
        """初始化异步模式。

        Args:
            replanner: AsyncReplanner 实例（鸭子类型：
                       submit/has_new_plan/apply_new_plan/is_planning）。
        """
        self._replanner: Any = replanner
        self._submitted: bool = False

    def submit(self, request: PlanRequest) -> None:
        """提交请求到后台线程（busy 时不重复提交）。

        Args:
            request: 规划请求。
        """
        if self._submitted or self._replanner.is_planning():
            return
        if self._replanner.submit(request):
            self._submitted = True

    def has_result(self) -> bool:
        """后台是否有新结果。"""
        return self._replanner.has_new_plan()

    def get_result(self) -> PlanResult | None:
        """获取并消费新结果（清空 submitted 标志）。

        Returns:
            PlanResult（或 None）。
        """
        result = self._replanner.apply_new_plan()
        if result is not None:
            self._submitted = False
        return result

    def is_busy(self) -> bool:
        """是否正在规划（已提交且未完成）。"""
        return self._submitted or self._replanner.is_planning()
