"""V12 回合的阶段编排（prepare → assemble → run → post_hit → result）。

本模块是 ``src.sim.v12_episode.run_v12_episode`` facade 的唯一实现体。它
**只**做五件事：

1. 校验配置类型（保持旧 ``TypeError`` 文案）；
2. 调用 :func:`~src.sim.episode.preparation.prepare_episode` 得到准备快照；
3. 球不可达时直接转交 ``build_unreachable_result``（不进入组装/运行）；
4. 调用 :func:`~src.sim.episode.assembly.assemble_episode` 组装并运行
   ``EpisodeRunner``；
5. 调用击打后阶段与结果构造，返回 ``V12EpisodeResult``。

阶段实现不在这里：任何一行准备/组装/指标逻辑都不应复制进本文件。阶段之间
通过 ``EpisodePreparation`` / ``EpisodeAssembly`` 两个不可变(outcome)对象与
``run_metrics`` 显式传递状态；本模块是这些对象生命周期的唯一持有者。
"""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from src.ilqt.components.sim_component import SimComponent
from src.ilqt.mpc_controller import MPCController
from src.sim.configuration.types import ResolvedV12Config
from src.sim.episode.assembly import EpisodeAssembly, assemble_episode
from src.sim.episode.post_hit import run_post_hit
from src.sim.episode.preparation import prepare_episode
from src.sim.episode.results import build_result, build_unreachable_result
from src.sim.episode.types import V12EpisodeResult
from src.sim.episode_runtime import UNSET, UnsetType
from src.sim.v12_robot_adapter import V12RobotBundle, build_v12_robot


def run_v12_episode_impl(
    config: ResolvedV12Config,
    *,
    dump_trajectory: Path | None = None,
    first_plan_flight: bool | UnsetType = UNSET,
    always_collision: bool | UnsetType = UNSET,
    ball_speed_perturb_pct: float | UnsetType = UNSET,
    fixed_direction: bool | UnsetType = UNSET,
    hit_direction: tuple[float, float, float] | None | UnsetType = UNSET,
    post_exec_hooks: tuple[Callable[[object], None], ...] = (),
    limits_config: Path | None | UnsetType = UNSET,
    max_tcp: float | None | UnsetType = UNSET,
    terminal_exempt_steps: int | None | UnsetType = UNSET,
    dq_max_fraction: float | None | UnsetType = UNSET,
    robot_builder: Callable[[ResolvedV12Config], V12RobotBundle] = build_v12_robot,
    mpc_controller_cls: Callable[..., Any] = MPCController,
    sim_component_cls: Callable[..., Any] = SimComponent,
) -> V12EpisodeResult:
    """执行一次 V12 共享闭环回合（阶段编排版）。

    参数与 :func:`src.sim.v12_episode.run_v12_episode` 完全一致；额外的
    ``robot_builder`` / ``mpc_controller_cls`` / ``sim_component_cls`` 是
    facade 注入点，用于让旧路径上的 monkeypatch 仍然生效（调用时读取模块
    属性，而不是绑定本模块的 import）。

    Args:
        config: 解析后的唯一配置来源。
        dump_trajectory: 轨迹 npz 落盘路径（None 不记录/不落盘）。
        first_plan_flight: 首个规划期间是否让球飞行。
        always_collision: 击球前球-拍碰撞常开。
        ball_speed_perturb_pct: 球速耦合扰动百分比。
        fixed_direction: 随挥方向是否用配置固定方向。
        hit_direction: 固定击打方向覆盖。
        post_exec_hooks: 额外 post_exec hook。
        limits_config: 显式限位文件覆盖。
        max_tcp: TCP 速度上限覆盖。
        terminal_exempt_steps: 终段豁免步数覆盖。
        dq_max_fraction: 关节位置增量上限分数覆盖。
        robot_builder: 机器人构建函数（facade 注入，默认真实实现）。
        mpc_controller_cls: MPC 控制器类（facade 注入）。
        sim_component_cls: SimComponent 类（facade 注入）。

    Returns:
        :class:`V12EpisodeResult`。

    Raises:
        TypeError: 传入的配置不是 :class:`ResolvedV12Config`。
        ValueError: 机器人/配置组合不受支持，或固定击打方向非法。
    """
    if not isinstance(config, ResolvedV12Config):
        raise TypeError(
            "run_v12_episode 只接受 ResolvedV12Config（请用 "
            "resolve_v12_config 解析），收到 "
            f"{type(config).__name__}"
        )

    prep = prepare_episode(
        config,
        first_plan_flight=first_plan_flight,
        always_collision=always_collision,
        ball_speed_perturb_pct=ball_speed_perturb_pct,
        fixed_direction=fixed_direction,
        hit_direction=hit_direction,
        limits_config=limits_config,
        max_tcp=max_tcp,
        terminal_exempt_steps=terminal_exempt_steps,
        dq_max_fraction=dq_max_fraction,
        robot_builder=robot_builder,
    )
    if prep.hit_info is None:
        return build_unreachable_result(prep)

    assembly: EpisodeAssembly = assemble_episode(
        prep,
        mpc_controller_cls=mpc_controller_cls,
        sim_component_cls=sim_component_cls,
        dump_trajectory=dump_trajectory,
        post_exec_hooks=post_exec_hooks,
    )

    # ── 9. 运行回合（旧 L984-998）──────────────────────────────────────
    t_total_start = time.perf_counter()
    run_metrics = assembly.runner.run(
        max_steps=prep.total_steps,
        first_plan_flight=prep.first_plan_flight_value,
    )
    t_mpc_end = time.perf_counter()

    # ── 10. 击打后继续仿真（旧 L1000-1031）──────────────────────────────
    post_hit_steps, post_hit_skipped_reason = run_post_hit(
        prep=prep,
        assembly=assembly,
        run_metrics=run_metrics,
    )
    x_current = np.asarray(prep.execution.get_arm_state(), dtype=np.float64).copy()

    return build_result(
        prep=prep,
        assembly=assembly,
        run_metrics=run_metrics,
        x_current=x_current,
        t_total_start=t_total_start,
        t_mpc_end=t_mpc_end,
        post_hit_steps=post_hit_steps,
        post_hit_skipped_reason=post_hit_skipped_reason,
        dump_trajectory=dump_trajectory,
    )
