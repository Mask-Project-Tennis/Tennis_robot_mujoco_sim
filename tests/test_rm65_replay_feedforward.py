"""R13 回归：RM65 回放环境必须继承 ``--no-feedforward``。

审查 R13 实测：``_build_rm65_plot_env`` 在位置模式下只调用
``configure_actuator_mode("position", kp, kd)``，而该方法默认打开前馈，
于是同初态同控制下回放环境与执行环境 20 步后状态偏差 0.54–0.63；修复后
回放环境按 ``config.feedforward`` 显式配置，两侧逐位一致。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT))

import rm65_mpc_v12 as v12  # noqa: E402
from src.robot.constants import INIT_Q  # noqa: E402
from src.sim.v12_config import resolve_v12_config  # noqa: E402
from src.sim.v12_robot_adapter import build_v12_robot  # noqa: E402

#: 同初态同控制的比对步数（审查用 20 步）
_N_STEPS = 20
#: 逐位一致容差（MuJoCo 同模型同参数应为 0，留浮点余量）
_PARITY_TOL = 1e-9


def _resolved(*, position_mode: bool, no_feedforward: bool = False) -> object:
    """构造真实 :class:`ResolvedV12Config`（T0 基线 serve-box）。

    Args:
        position_mode: 是否位置模式（R13 只发生在该模式）。
        no_feedforward: 是否显式 ``--no-feedforward``。

    Returns:
        解析后的 V12 配置。
    """
    args: dict[str, object] = {
        "serve_box": True,
        "ball_speed": 7.0,
        "seed": 42,
        "no_plot": True,
    }
    flags = {"--serve-box", "--ball-speed", "--seed", "--no-plot"}
    if position_mode:
        args["position_mode"] = True
        flags.add("--position-mode")
    if no_feedforward:
        args["no_feedforward"] = True
        flags.add("--no-feedforward")
    return resolve_v12_config(argparse.Namespace(**args), frozenset(flags))


def _envs(config: object) -> tuple[object, object]:
    """构造回放环境（plot）与执行环境（V12 adapter）并断言同源设置。

    Args:
        config: 已解析配置。

    Returns:
        ``(plot_env, execution_env)``。
    """
    plot_env = v12._build_rm65_plot_env(config)  # noqa: SLF001
    execution_env = build_v12_robot(config).execution  # type: ignore[arg-type]
    return plot_env, execution_env


@pytest.mark.parametrize("enabled", [False, True])
def test_plot_env_feedforward_matches_execution(enabled: bool) -> None:
    """位置模式下回放/执行环境的 feedforward 开关同值。"""
    config = _resolved(position_mode=True, no_feedforward=not enabled)
    plot_env, execution_env = _envs(config)
    assert bool(config.feedforward) is enabled  # type: ignore[attr-defined]
    assert plot_env.use_feedforward is enabled
    assert execution_env.use_feedforward is enabled


def test_position_default_feedforward_enabled() -> None:
    """未显式关闭时位置模式默认仍启用前馈（不改变默认行为）。"""
    config = _resolved(position_mode=True)
    assert bool(config.feedforward) is True  # type: ignore[attr-defined]
    plot_env, execution_env = _envs(config)
    assert plot_env.use_feedforward is True
    assert execution_env.use_feedforward is True


@pytest.mark.parametrize("feedforward", [False, True])
def test_replay_and_execution_trajectories_identical(
    feedforward: bool,
) -> None:
    """同初态同控制 20 步：回放环境与执行环境状态最大差 < 1e-9。"""
    config = _resolved(position_mode=True, no_feedforward=not feedforward)
    plot_env, execution_env = _envs(config)

    x_plot = np.asarray(plot_env.reset(INIT_Q), dtype=float)
    x_exec = np.asarray(execution_env.reset(INIT_Q), dtype=float)
    assert np.max(np.abs(x_plot - x_exec)) < _PARITY_TOL

    rng = np.random.default_rng(7)
    controls = INIT_Q[None, :] + rng.normal(0.0, 0.08, size=(_N_STEPS, 6))
    max_diff = 0.0
    for u in controls:
        x_plot = np.asarray(plot_env.step(u), dtype=float)
        x_exec = np.asarray(execution_env.step(u), dtype=float)
        max_diff = max(max_diff, float(np.max(np.abs(x_plot - x_exec))))
    assert max_diff < _PARITY_TOL, (
        f"feedforward={feedforward} 时回放/执行环境 {_N_STEPS} 步后偏差 "
        f"{max_diff:.6g}（应逐位一致）"
    )
