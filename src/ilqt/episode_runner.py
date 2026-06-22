"""通用管线编排器 — 组合感知/规划/安全/执行组件。

仿真和真机共用此类，差异仅在注入的组件不同。
支持 5 个 hook 插入点，允许在不动管线代码的情况下插入自定义逻辑。
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from src.ilqt.mpc_controller import MPCController
    from src.ilqt.components.protocols import (
        PerceptionComponent,
        ExecutorComponent,
        SafetyComponent,
    )
    from src.ilqt.step_context import StepContext

logger = logging.getLogger(__name__)


class EpisodeRunner:
    """通用管线编排器 — 组合 4 个组件 + 5 个可选 hook。

    管线流程（每步）:
        1. perception.get_ball_state() → (ball_pos, ball_vel)
        2. executor.get_arm_state() → arm_state
        → [pre_plan_hooks]         ← 可插入在线估计/自适应参数
        3. mpc.step(ball_pos, ball_vel, arm_state) → MPCStepResult
        → [post_plan_hooks]        ← 可修改 u_cmd / 插入外部扰动
        4. safety.filter(u_cmd, arm_state) → (safe_u, is_safe)
        → if not is_safe: [on_unsafe_hooks]; break
        5. executor.execute(safe_u)
        → [post_exec_hooks]        ← 可插入自定义诊断/数据收集

    episode 结束时:
        → [on_done_hooks]
    """

    def __init__(
        self,
        mpc: "MPCController",
        perception: "PerceptionComponent",
        safety: "SafetyComponent",
        executor: "ExecutorComponent",
        pre_plan_hooks: list[Callable] | None = None,
        post_plan_hooks: list[Callable] | None = None,
        post_exec_hooks: list[Callable] | None = None,
        on_unsafe_hooks: list[Callable] | None = None,
        on_done_hooks: list[Callable] | None = None,
    ) -> None:
        """注入 4 个组件 + 5 个可选 hook 列表。

        Args:
            mpc: MPC 规划控制器。
            perception: 球感知组件。
            safety: 安全滤波组件。
            executor: 执行组件（含指标汇总）。
            pre_plan_hooks: 规划前调用（感知后），可读取/修改臂状态和球状态。
            post_plan_hooks: 规划后调用（安全前），可修改 u_cmd。
            post_exec_hooks: 执行后调用，可追加自定义指标到 ctx.metrics。
            on_unsafe_hooks: 安全检查失败时调用。
            on_done_hooks: episode 结束时调用。
        """
        self._mpc = mpc
        self._perception = perception
        self._safety = safety
        self._executor = executor
        self._pre_plan_hooks = pre_plan_hooks or []
        self._post_plan_hooks = post_plan_hooks or []
        self._post_exec_hooks = post_exec_hooks or []
        self._on_unsafe_hooks = on_unsafe_hooks or []
        self._on_done_hooks = on_done_hooks or []

    def run(self, max_steps: int = 500) -> dict:
        """运行完整 episode。

        Args:
            max_steps: 最大运行步数。

        Returns:
            metrics dict，含 total_steps/safe_steps/mpc_done + executor 指标 + hook 追加。
        """
        from src.ilqt.step_context import StepContext

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
        hook_metrics: dict = {}  # hook 追加的持久化指标

        while not self._mpc.done and step_count < max_steps:
            # 感知
            ball = self._perception.get_ball_state()
            arm_state = self._executor.get_arm_state()

            if ball is None:
                logger.warning(f"步 {step_count}: 球状态不可用，退出")
                break

            # 构建上下文（metrics 指向持久化 dict，跨步累积）
            ctx = StepContext(
                step_count=step_count,
                arm_state=arm_state,
                ball_pos=ball[0],
                ball_vel=ball[1],
                metrics=hook_metrics,
            )

            # pre_plan hooks
            for hook in self._pre_plan_hooks:
                hook(ctx)

            # 规划（ball 已确认非 None，arm_state 可被 hook 修改）
            result = self._mpc.step(ball[0], ball[1], ctx.arm_state)
            ctx.mpc_result = result

            # post_plan hooks
            for hook in self._post_plan_hooks:
                hook(ctx)

            # 安全滤波
            u_for_safety = ctx.u_cmd if ctx.u_cmd is not None else result.u_cmd
            safe_u, is_safe = self._safety.filter(u_for_safety, ctx.arm_state)
            ctx.u_cmd = safe_u
            ctx.is_safe = is_safe

            if not is_safe:
                logger.warning(f"步 {step_count}: 安全检查失败，停止 episode")
                for hook in self._on_unsafe_hooks:
                    hook(ctx)
                break

            # 执行
            self._executor.execute(safe_u)

            # post_exec hooks
            for hook in self._post_exec_hooks:
                hook(ctx)

            safe_steps += 1
            step_count += 1

        # 4. 清理
        self._mpc.stop()

        # on_done hooks
        final_ctx = StepContext(step_count=step_count, arm_state=arm_state, metrics=hook_metrics)
        for hook in self._on_done_hooks:
            hook(final_ctx)

        # 5. 汇总指标（从 executor 获取 + hook 追加）
        metrics: dict = {
            "total_steps": step_count,
            "safe_steps": safe_steps,
            "mpc_done": self._mpc.done,
        }
        if hasattr(self._executor, "get_metrics"):
            metrics.update(self._executor.get_metrics())
        if hook_metrics:
            metrics.update(hook_metrics)

        logger.info(
            "EpisodeRunner 完成: total_steps=%s safe_steps=%s mpc_done=%s",
            metrics.get("total_steps"),
            metrics.get("safe_steps"),
            metrics.get("mpc_done"),
        )
        logger.debug("EpisodeRunner metrics 详情: %s", metrics)
        return metrics
