"""V12 回合的 hook / 轨迹记录装配辅助。

本模块只做两件事：按需构造 :class:`TrajectoryRecorder` 并生成其 post_exec
hook；把调用方额外传入的 hook 与记录 hook 合并成 ``EpisodeRunner`` 需要的
列表。另外保留 R07 要求的球 history 首帧播种函数（``_seed_ball_history``）。

状态所有权：``recorder`` 由本模块创建后交给 ``EpisodeAssembly``，之后只由
``execution.run_v12_episode_impl`` 在击打后阶段/落盘阶段使用；本模块不缓存
也不释放它。``sim_component`` 由 ``assembly`` 创建，本模块只覆盖其 history
首帧。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.ilqt.components.sim_component import SimComponent
from src.robot.constants import INIT_Q_LEFT

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查使用，避免运行期循环导入
    from src.sim.episode.preparation import EpisodePreparation

logger = logging.getLogger(__name__)


def _seed_ball_history(
    sim_component: SimComponent,
    *,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
) -> None:
    """把球位置/速度 history 的首帧设为环境初态（R07）。

    调用方必须传入写入 ``execution.set_ball_state`` 的同一对初态
    （``p0_real``/``v0_real``，已含 ``ball_speed_perturb_pct`` 扰动），
    保证产物 ``ball_initial`` 与轨迹首帧同源。

    Args:
        sim_component: 轨迹 history 宿主。
        ball_pos: 环境实际初态位置（3,）。
        ball_vel: 环境实际初态速度（3,）。
    """
    sim_component.ball_pos_history = [
        np.asarray(ball_pos, dtype=np.float64).copy()
    ]
    sim_component.ball_vel_history = [
        np.asarray(ball_vel, dtype=np.float64).copy()
    ]


def build_post_exec_hooks(
    prep: EpisodePreparation,
    *,
    dump_trajectory: Path | None,
    post_exec_hooks: tuple[Callable[[object], None], ...],
) -> tuple[Any, list[Callable[[object], None]]]:
    """构造轨迹记录器并合并 post_exec hook 列表。

    仅当 ``dump_trajectory`` 非 ``None`` 时创建 :class:`TrajectoryRecorder`
    （与旧实现一致：不落盘就不记录）。记录器 metadata 的键集合逐字保持
    ``p0``/``v0``/``ball_speed``/``is_position_mode``，避免改变 T0 逐位
    基线产物。

    Args:
        prep: 准备阶段结果（提供执行环境/初始关节角/初态/机器人事实）。
        dump_trajectory: 轨迹 npz 落盘路径；``None`` 表示不记录。
        post_exec_hooks: 调用方额外 hook，追加在记录 hook 之后。

    Returns:
        ``(recorder_or_none, hooks)``；``hooks`` 为可直接传入
        ``EpisodeRunner(post_exec_hooks=...)`` 的列表。
    """
    recorder: Any = None
    hooks: list[Callable[[object], None]] = []
    if dump_trajectory is not None:
        from src.real.trajectory_recorder import TrajectoryRecorder

        recorder = TrajectoryRecorder(
            env=prep.execution,
            init_q=prep.init_q,
            init_q_left=(
                np.asarray(INIT_Q_LEFT, dtype=np.float64)
                if prep.robot_id == "rm65" else np.zeros(0, dtype=np.float64)
            ),
            dt=prep.dt,
            metadata={
                "p0": prep.p0_real.tolist(),
                "v0": prep.v0_real.tolist(),
                "ball_speed": prep.serve_cfg.get("ball_speed"),
                "is_position_mode": prep.is_position_mode,
            },
            is_position_mode=prep.is_position_mode,
        )
        hooks.append(recorder.make_hook())
    hooks.extend(post_exec_hooks)
    return recorder, hooks
