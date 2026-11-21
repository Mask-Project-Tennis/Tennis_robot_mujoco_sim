"""V12 回合的击打后保持阶段（R01）。

本模块只负责「主评估段结束之后是否继续推进仿真、以及如何推进」这一
单一职责：

* :data:`_POST_HIT_STEPS` / :data:`_POST_HIT_ALLOWED_STOP_REASONS` 是
  该阶段仅有的两个事实常量；
* :func:`_apply_post_hit_phase` 按 ``stop_reason`` 门控，只有 MPC 正常
  完成或达到步数上限才允许进入未滤波的保持控制；
* :func:`_run_post_hit_hold` 以位置/力矩两种模式保持当前臂形，并逐帧
  追加 ``SimComponent`` 的 history 与 ``TrajectoryRecorder`` 记录；
* :func:`run_post_hit` 是阶段编排入口，显式接收准备阶段/组装阶段的
  结果，不读取任何模块级可变状态。

状态所有权：``sim_component``（history 宿主）与 ``execution``（物理执行
环境）都由调用方（``execution.run_v12_episode_impl``）持有，本模块只原地
追加数据，不创建也不释放这两个对象；``recorder`` 由 ``assembly`` 阶段
创建并同样由调用方持有，这里只调用其 ``record`` 方法。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from src.ilqt.components.sim_component import SimComponent

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查使用，避免运行期循环导入
    from src.sim.episode.assembly import EpisodeAssembly
    from src.sim.episode.preparation import EpisodePreparation

logger = logging.getLogger(__name__)

# 击打后继续仿真步数（旧脚本 L1003，V11 语义）
_POST_HIT_STEPS = 20

# 允许执行击打后保持段的终止原因（R01）：只有 MPC 正常完成或达到步数上限
# 才继续推进仿真；安全拒绝/球状态不可用时立即截断，不再执行未滤波控制。
_POST_HIT_ALLOWED_STOP_REASONS = frozenset({"mpc_done", "max_steps"})


def _apply_post_hit_phase(
    *,
    run_metrics: Mapping[str, Any],
    execution: Any,
    sim_component: SimComponent,
    is_position_mode: bool,
    dt: float,
    total_horizon: int,
    follow_through_steps: int,
    recorder: Any,
    ball_unreachable: bool,
) -> tuple[int, str | None]:
    """按 ``stop_reason`` 门控击打后保持段（R01）。

    只有 ``mpc_done``/``max_steps`` 且非 ``ball_unreachable`` 才执行未滤波的
    保持控制；``unsafe``/``no_ball``/不可达（及未来任何非正常终止）直接截断，
    返回 0 步与跳过原因。不可达时即使 ``mpc_done`` 为 True 也不得继续甩动。

    Args:
        run_metrics: ``EpisodeRunner.run`` 返回的指标（读 ``stop_reason``）。
        execution: 执行环境。
        sim_component: 轨迹 history 宿主。
        is_position_mode: 位置模式/力矩模式。
        dt: 控制周期（秒）。
        total_horizon: 主评估段步数。
        follow_through_steps: 随挥步数。
        recorder: 轨迹记录器或 ``None``。
        ball_unreachable: 规划侧是否判定来球不可达（``mpc.ball_unreachable``）。

    Returns:
        ``(post_hit_steps, post_hit_skipped_reason)``；执行时为
        ``(_POST_HIT_STEPS, None)``，截断时为 ``(0, 跳过原因)``。
    """
    stop_reason = str(run_metrics.get("stop_reason") or "")
    if ball_unreachable:
        logger.warning("击打后保持段跳过（ball_unreachable=True）：不执行未滤波控制")
        return 0, "ball_unreachable"
    if stop_reason not in _POST_HIT_ALLOWED_STOP_REASONS:
        skipped_reason = stop_reason or "unknown"
        logger.warning(
            "击打后保持段跳过（stop_reason=%s）：不执行未滤波控制",
            skipped_reason,
        )
        return 0, skipped_reason
    steps = _run_post_hit_hold(
        execution=execution,
        sim_component=sim_component,
        is_position_mode=is_position_mode,
        dt=dt,
        total_horizon=total_horizon,
        follow_through_steps=follow_through_steps,
        recorder=recorder,
    )
    return steps, None


def _run_post_hit_hold(
    *,
    execution: Any,
    sim_component: SimComponent,
    is_position_mode: bool,
    dt: float,
    total_horizon: int,
    follow_through_steps: int,
    recorder: Any,
) -> int:
    """击打后保持段：PD 保持当前臂形并推进 ``_POST_HIT_STEPS`` 步（R01）。

    调用方必须先按 ``stop_reason`` 门控：只有 MPC 正常完成/达到步数上限
    才允许进入本函数；安全拒绝与球状态不可用时不得用 ``step_full``
    补未滤波制动。本函数内的控制量与 append 顺序与旧内联循环逐字一致。

    Args:
        execution: 执行环境（需 ``NQ``/``NU``/``step_full``/``get_arm_state``）。
        sim_component: 轨迹 history 宿主（X/U/ball_pos/ball_vel）。
        is_position_mode: 位置模式用关节角保持，否则用力矩 PD。
        dt: 控制周期（秒）。
        total_horizon: 主评估段步数（用于 recorder 时间戳）。
        follow_through_steps: 随挥步数（用于 recorder 时间戳）。
        recorder: 轨迹记录器或 ``None``。

    Returns:
        实际执行步数（未截断时为 ``_POST_HIT_STEPS``）。
    """
    nq = int(execution.NQ)
    nu = int(execution.NU)
    ctrl_range = execution.model.actuator_ctrlrange
    x_current = np.asarray(execution.get_arm_state(), dtype=np.float64).copy()
    for i in range(_POST_HIT_STEPS):
        q_hold = x_current[:nq].copy()
        if is_position_mode:
            u_hold = q_hold.copy()
        else:
            u_hold = 100.0 * (q_hold - x_current[:nq]) \
                - 10.0 * x_current[nq:]
            u_hold = np.clip(
                u_hold,
                ctrl_range[:nu, 0],
                ctrl_range[:nu, 1],
            )
        x_current, ball_pos_post, _ = execution.step_full(u_hold)
        x_current = np.asarray(x_current, dtype=np.float64)
        sim_component.X_history.append(x_current.copy())
        sim_component.U_history.append(np.asarray(u_hold, dtype=np.float64).copy())
        sim_component.ball_pos_history.append(
            np.asarray(ball_pos_post, dtype=np.float64).copy()
        )
        sim_component.ball_vel_history.append(
            np.asarray(execution.get_ball_vel(), dtype=np.float64).copy()
        )
        if recorder is not None:
            step_idx = total_horizon + follow_through_steps + i
            recorder.record(
                q_desired=q_hold.copy() if is_position_mode else None,
                q_actual=x_current[:nq].copy(),
                timestamp=step_idx * dt,
                tcp_pos=np.asarray(execution.get_ee_pos(), dtype=np.float64),
                ball_pos=np.asarray(ball_pos_post, dtype=np.float64),
                u=np.asarray(u_hold, dtype=np.float64).copy(),
            )
    return _POST_HIT_STEPS


def run_post_hit(
    *,
    prep: EpisodePreparation,
    assembly: EpisodeAssembly,
    run_metrics: Mapping[str, Any],
) -> tuple[int, str | None]:
    """击打后阶段编排入口：从准备/组装结果中取出门控所需事实。

    本函数不改变 ``_apply_post_hit_phase`` 的任何语义，只负责把阶段对象
    中的单一所有者字段显式摊开，使 ``execution.run_v12_episode_impl`` 的
    调用顺序可被单测替换（patch 本函数）而不触达具体实现。

    Args:
        prep: 准备阶段结果（提供执行环境/时间步/段长度等事实）。
        assembly: 组装阶段结果（提供 SimComponent/recorder/MPC）。
        run_metrics: ``EpisodeRunner.run`` 的返回指标。

    Returns:
        ``(post_hit_steps, post_hit_skipped_reason)``。
    """
    return _apply_post_hit_phase(
        run_metrics=run_metrics,
        execution=prep.execution,
        sim_component=assembly.sim_component,
        is_position_mode=prep.is_position_mode,
        dt=prep.dt,
        total_horizon=prep.total_horizon,
        follow_through_steps=prep.follow_through_steps,
        recorder=assembly.recorder,
        ball_unreachable=bool(assembly.mpc.ball_unreachable),
    )
