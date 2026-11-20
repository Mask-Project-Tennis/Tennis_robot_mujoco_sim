"""V12 复审修复回归：固定击球方向优先级 + 求解器 effective 后端。

覆盖第三轮复审两项 P2：

1. ``--fixed-direction`` 时 ``hitting.hit_direction`` 配置被忽略、回退硬编码
   ``[0, -1, 0.3]``。修复后优先级为：显式 runtime/CLI > ``hitting`` 配置 >
   ``_DEFAULT_HIT_DIRECTION``；非 fixed 仍用来球反方向，不误用配置。
2. C++ 求解器被记录为 Python（只看环境兼容性）。修复后 ``backend.effective``
   按实际构造的 solver 记录：``cpp`` / ``python-analytic`` / ``python-fd``。

最后两个用例真实跑 horizon=60 的 RM65 短回合（不跑验收长回合）。
"""

from __future__ import annotations

import argparse
import dataclasses
import types

import numpy as np
import pytest

from src.ilqt.mpc_controller import MPCController, describe_solver_backend
from src.sim.v12_config import ResolvedV12Config, resolve_v12_config
from src.sim.v12_episode import (
    _DEFAULT_HIT_DIRECTION,
    _effective_backend,
    _normalize_hit_direction,
    _predict_effective_backend,
    _resolve_hit_direction,
    _select_follow_direction,
    run_v12_episode,
)


def _unit(vector: object) -> np.ndarray:
    """返回向量的单位化副本（测试辅助）。"""
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def _cpp_available() -> bool:
    """C++ 扩展是否在本机可用（未编译时相关断言 skip）。"""
    from src.cpp import solver_cpp

    return bool(getattr(solver_cpp, "_CPP_AVAILABLE", False))


# ---------------------------------------------------------------------------
# 问题 1：固定击球方向优先级
# ---------------------------------------------------------------------------


def test_default_hit_direction_unchanged() -> None:
    """无 runtime、无配置时仍用 ``[0, -1, 0.3]``（归一化后生效）。"""
    vector, source = _resolve_hit_direction(None, {})

    assert source == "default"
    assert _DEFAULT_HIT_DIRECTION == (0.0, -1.0, 0.3)
    np.testing.assert_allclose(vector, _unit(_DEFAULT_HIT_DIRECTION), atol=1e-12)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_hitting_config_direction_is_used() -> None:
    """``hitting.hit_direction``（YAML/--task-overrides）优先于默认值。"""
    vector, source = _resolve_hit_direction(None, {"hit_direction": [1.0, 0.0, 0.0]})

    assert source == "config:hitting.hit_direction"
    np.testing.assert_allclose(vector, [1.0, 0.0, 0.0], atol=1e-12)


def test_hitting_config_direction_is_normalized() -> None:
    """非单位配置向量按单位长度生效（[3, 4, 0] → [0.6, 0.8, 0]）。"""
    vector, source = _resolve_hit_direction(None, {"hit_direction": [3.0, 4.0, 0.0]})

    assert source == "config:hitting.hit_direction"
    np.testing.assert_allclose(vector, [0.6, 0.8, 0.0], atol=1e-12)
    assert np.linalg.norm(vector) == pytest.approx(1.0)


def test_runtime_direction_overrides_hitting_config() -> None:
    """显式 runtime 方向优先于 ``hitting`` 配置。"""
    vector, source = _resolve_hit_direction(
        [0.0, 2.0, 0.0],
        {"hit_direction": [1.0, 0.0, 0.0]},
    )

    assert source == "runtime"
    np.testing.assert_allclose(vector, [0.0, 1.0, 0.0], atol=1e-12)


@pytest.mark.parametrize(
    "bad",
    [
        [1.0, 0.0],  # 长度非 3
        [0.0, 0.0, 0.0],  # 零向量
        [float("nan"), 0.0, 1.0],  # 非有限
        [float("inf"), 0.0, 1.0],  # 非有限
        "reflect",  # 非数值序列
    ],
)
def test_invalid_hit_direction_raises(bad: object) -> None:
    """非法方向配置必须明确报错，不得静默回退默认值。"""
    with pytest.raises(ValueError):
        _normalize_hit_direction(bad, source="config:hitting.hit_direction")


def test_select_follow_direction_fixed_uses_config() -> None:
    """fixed 分支返回配置方向与来源标签。"""
    direction, source = _select_follow_direction(
        fixed_direction=True,
        configured_direction=np.array([1.0, 0.0, 0.0]),
        configured_source="config:hitting.hit_direction",
        v_ball_hit=np.array([0.0, -7.0, 0.0]),
    )

    assert source == "config:hitting.hit_direction"
    np.testing.assert_allclose(direction, [1.0, 0.0, 0.0], atol=1e-12)


def test_select_follow_direction_non_fixed_reflects_ball() -> None:
    """非 fixed 分支始终用来球反方向，不误用配置方向。"""
    direction, source = _select_follow_direction(
        fixed_direction=False,
        configured_direction=np.array([1.0, 0.0, 0.0]),
        configured_source="config:hitting.hit_direction",
        v_ball_hit=np.array([0.0, -7.0, 1.0]),
    )

    assert source == "reflect"
    np.testing.assert_allclose(
        direction, _unit([0.0, 7.0, -1.0]), atol=1e-12,
    )


# ---------------------------------------------------------------------------
# 问题 5：求解器 effective 后端按实际实例记录
# ---------------------------------------------------------------------------


class _FakePythonSolver:
    """模拟 ``ILQTSolver``（仅暴露 use_analytical）。"""

    def __init__(self, use_analytical: bool) -> None:
        self.use_analytical = use_analytical


class _FakeCompatibleEnv:
    """模拟 RM65 规划环境（满足 C++ 内核六维前缀布局）。"""

    NQ = 6
    NX = 12
    NU = 6
    init_q_left = np.zeros(6)
    _torque_ctrlrange = np.array([[-1.0, 1.0]] * 6)


class _FakeR1DEnv:
    """模拟 R1D 规划环境（n=7，非 C++ 兼容）。"""

    NQ = 7
    NX = 14
    NU = 7


@pytest.mark.parametrize(
    ("use_analytical", "expected"),
    [(True, "python-analytic"), (False, "python-fd")],
)
def test_describe_python_solver(use_analytical: bool, expected: str) -> None:
    """Python 求解器按 use_analytical 区分解析/FD。"""
    assert describe_solver_backend(_FakePythonSolver(use_analytical)) == expected


def test_describe_cpp_solver_wrapper() -> None:
    """C++ 包装类（模块为 src.cpp.solver_cpp 且扩展可用）记为 cpp。"""
    if not _cpp_available():
        pytest.skip("C++ 扩展不可用")

    class _FakeCppSolver:
        use_analytical = True

    _FakeCppSolver.__module__ = "src.cpp.solver_cpp"
    assert describe_solver_backend(_FakeCppSolver()) == "cpp"


def test_describe_unknown_solver_raises() -> None:
    """无法判定的实例明确报错，不静默归类为 Python。"""
    with pytest.raises(ValueError, match="无法判定"):
        describe_solver_backend(object())


def test_mpc_controller_effective_backend_reads_solver() -> None:
    """``MPCController.effective_backend`` 读取实际 solver，而非请求值。"""
    controller = object.__new__(MPCController)
    controller._solver = _FakePythonSolver(False)

    assert controller.effective_backend == "python-fd"


def test_effective_backend_uses_constructed_solver() -> None:
    """构造出 MPC 后，effective 取实际 solver（requested 保持请求值）。"""
    bundle = types.SimpleNamespace(backend="auto")
    fake_mpc = types.SimpleNamespace(effective_backend="cpp")

    backend = _effective_backend(_FakeR1DEnv(), bundle, fake_mpc)

    assert backend == {"requested": "auto", "effective": "cpp"}


def test_effective_backend_prediction_without_mpc() -> None:
    """早退路径（无 MPC）按 build_solver 规则推断，区分 C++ 可用性。"""
    bundle = types.SimpleNamespace(backend="auto")
    compatible_expected = "cpp" if _cpp_available() else "python-analytic"

    assert _effective_backend(_FakeCompatibleEnv(), bundle)["effective"] == (
        compatible_expected
    )
    assert _effective_backend(_FakeR1DEnv(), bundle)["effective"] == "python-fd"
    assert _predict_effective_backend(_FakeCompatibleEnv(), "python") == (
        "python-analytic"
    )
    assert _predict_effective_backend(_FakeCompatibleEnv(), "cpp") == "cpp"
    assert _predict_effective_backend(_FakeR1DEnv(), "python") == "python-fd"


def test_real_rm65_solver_backend() -> None:
    """真实 RM65 环境构造的 solver：扩展可用记 cpp，否则 python-analytic。"""
    from src.ilqt.planning_env import PlanningEnv
    from src.ilqt.solver import build_solver

    env = PlanningEnv(dt=0.005)
    solver = build_solver(env=env, backend="auto")
    expected = "cpp" if _cpp_available() else "python-analytic"

    assert describe_solver_backend(solver) == expected


# ---------------------------------------------------------------------------
# 端到端：方向配置进入真实短回合 metrics
# ---------------------------------------------------------------------------


def _args(**overrides: object) -> argparse.Namespace:
    """构造只含所需字段的局部 Namespace（其余走 resolver 缺省）。"""
    return argparse.Namespace(**overrides)


def _resolve_rm65_short() -> ResolvedV12Config:
    """解析短回合 RM65 配置（horizon=60/2 次迭代，供一次真实闭环用）。"""
    return resolve_v12_config(
        _args(
            serve_box=True,
            ball_speed=7.0,
            seed=42,
            no_plot=True,
            horizon=60,
            iter=2,
            near_iters=2,
            first_plan_iters=2,
            near_threshold=30,
            replan_interval=20,
        ),
        frozenset({"--serve-box", "--ball-speed", "--seed", "--no-plot"}),
    )


def _with_configured_direction(
    config: ResolvedV12Config, direction: list[float],
) -> ResolvedV12Config:
    """返回替换 ``hitting.hit_direction`` 后的配置副本。"""
    return dataclasses.replace(
        config,
        hitting_config={**config.hitting_config, "hit_direction": direction},
    )


def test_run_v12_episode_fixed_direction_honors_hitting_config() -> None:
    """回归：``--fixed-direction`` 时配置 ``[1,0,0]`` 必须生效。"""
    config = _with_configured_direction(_resolve_rm65_short(), [1.0, 0.0, 0.0])
    result = run_v12_episode(config, fixed_direction=True)

    assert result.metrics["hit_direction_source"] == (
        "config:hitting.hit_direction"
    )
    np.testing.assert_allclose(
        result.metrics["hit_direction"], [1.0, 0.0, 0.0], atol=1e-9,
    )
    v_hit = np.asarray(result.metrics["v_hit_at_contact"], dtype=np.float64)
    np.testing.assert_allclose(v_hit / np.linalg.norm(v_hit), [1.0, 0.0, 0.0], atol=1e-9)
    # 问题 5：真实 RM65 短回合的 effective 必须是实际 solver（C++ 可用则 cpp）。
    expected_backend = "cpp" if _cpp_available() else "python-analytic"
    assert result.metrics["backend"]["effective"] == expected_backend


def test_run_v12_episode_runtime_direction_overrides_hitting_config() -> None:
    """回归：显式 ``hit_direction`` 覆盖配置方向，并记录来源 runtime。"""
    config = _with_configured_direction(_resolve_rm65_short(), [1.0, 0.0, 0.0])
    result = run_v12_episode(
        config,
        fixed_direction=True,
        hit_direction=(0.0, 1.0, 0.0),
    )

    assert result.metrics["hit_direction_source"] == "runtime"
    np.testing.assert_allclose(
        result.metrics["hit_direction"], [0.0, 1.0, 0.0], atol=1e-9,
    )
