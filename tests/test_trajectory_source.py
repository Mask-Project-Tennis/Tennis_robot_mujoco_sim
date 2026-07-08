"""TrajectorySource 装饰器链单元测试。

测试 FileSource / ResampledSource / TcpSpeedLimiter 的协议符合性、
基本行为、装饰器组合和异常安全性。
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from src.real.resample_strategy import InterpolatingResampler
from src.real.trajectory_source import (
    FileSource,
    ResampledSource,
    TcpSpeedControllable,
    TcpSpeedLimiter,
    TrajectorySource,
)


# ── 手写 Mock 类（不使用 MagicMock）──


class MockRobot:
    """模拟支持 TCP 速度控制的机器人，记录 set_max_tcp_speed 调用。"""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def set_max_tcp_speed(self, speed: float) -> None:
        """记录 TCP 速度设置调用。"""
        self.calls.append(speed)


class RaisingSource:
    """内部 source，迭代时抛出异常（测试 finally 恢复）。"""

    def __init__(self, n_before_raise: int = 1) -> None:
        self._n = n_before_raise

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        for i in range(self._n):
            yield np.zeros(6), float(i)
        raise ValueError("inner source error")


# ── 辅助函数 ──


def _save_test_npz(
    path: Path, n_steps: int = 10, dt: float = 0.005
) -> tuple[np.ndarray, np.ndarray]:
    """创建测试用 .npz 文件，返回 (q_desired, timestamps)。

    使用线性轨迹（q[i] = i*0.1），CubicSpline 对线性数据插值精确。

    Args:
        path: 保存路径。
        n_steps: 步数。
        dt: 时间步长（秒）。

    Returns:
        (q_desired (n_steps, 6), timestamps (n_steps,)) 元组。
    """
    q_desired = np.tile(
        (np.arange(n_steps, dtype=float) * 0.1).reshape(-1, 1), (1, 6)
    )
    timestamps = np.arange(n_steps, dtype=float) * dt
    np.savez(
        path,
        q_desired=q_desired,
        q_actual=q_desired.copy(),
        timestamps=timestamps,
        tcp_pos=np.zeros((n_steps, 3)),
        ball_pos=np.zeros((n_steps, 3)),
        init_q=np.zeros(6),
        init_q_left=np.zeros(6),
        dt=dt,
        hit_step=-1,
        metadata=json.dumps({}),
    )
    return q_desired, timestamps


def _save_old_pickle(path: Path, n_steps: int = 5) -> np.ndarray:
    """创建旧格式 pickle 文件，返回 U_history 数组。

    Args:
        path: 保存路径。
        n_steps: 步数。

    Returns:
        U_history (n_steps, 6) 数组。
    """
    X_history = [np.zeros(24) for _ in range(n_steps + 1)]
    for i in range(n_steps + 1):
        X_history[i][:6] = float(i) * 0.1
    U_history = [np.full(6, float(i) * 0.2) for i in range(n_steps)]
    old_data = {
        "X_history": X_history,
        "U_history": U_history,
        "init_q": np.zeros(6),
        "init_q_left": np.zeros(6),
        "hit_step": -1,
    }
    with open(path, "wb") as f:
        pickle.dump(old_data, f)
    return np.array(U_history)


# ── 协议符合性测试 ──


class TestTrajectorySourceProtocol:
    """验证各 Source 类满足 TrajectorySource Protocol。"""

    def test_file_source_satisfies_protocol(self, tmp_path):
        """FileSource 实例是 TrajectorySource 的运行时实例。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path)
        assert isinstance(FileSource(path), TrajectorySource)

    def test_resampled_source_satisfies_protocol(self, tmp_path):
        """ResampledSource 实例是 TrajectorySource 的运行时实例。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path)
        inner = FileSource(path)
        outer = ResampledSource(inner, InterpolatingResampler(), speed_factor=0.1)
        assert isinstance(outer, TrajectorySource)

    def test_tcp_speed_limiter_satisfies_protocol(self, tmp_path):
        """TcpSpeedLimiter 实例是 TrajectorySource 的运行时实例。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path)
        inner = FileSource(path)
        limiter = TcpSpeedLimiter(inner, MockRobot(), max_tcp_speed=0.2)
        assert isinstance(limiter, TrajectorySource)

    def test_mock_robot_satisfies_tcp_speed_controllable(self):
        """MockRobot 满足 TcpSpeedControllable Protocol。"""
        assert isinstance(MockRobot(), TcpSpeedControllable)


# ── FileSource 测试 ──


class TestFileSource:
    """FileSource 文件加载与迭代测试。"""

    def test_load_npz_iterate_correct_count(self, tmp_path):
        """加载 .npz 文件 → 迭代 → 点数与原始数据一致。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=10)
        source = FileSource(path)
        points = list(source)
        assert len(points) == 10

    def test_points_match_original_data(self, tmp_path):
        """迭代产出的 (q, t) 与原始 q_desired + timestamps 一致。"""
        path = tmp_path / "traj.npz"
        q_desired, timestamps = _save_test_npz(path, n_steps=10, dt=0.005)
        source = FileSource(path)
        points = list(source)
        assert len(points) == 10
        for i, (q, t) in enumerate(points):
            np.testing.assert_allclose(q, q_desired[i])
            assert t == pytest.approx(timestamps[i])

    def test_copy_prevents_aliasing(self, tmp_path):
        """__iter__ 产出 q 的 copy()，修改产出不影响内部数据。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=5)
        source = FileSource(path)
        points = list(source)
        # 修改产出的 q（in-place 修改数组元素，不能重新赋值 tuple 元素）
        points[0][0][0] = 999.0
        # 重新迭代，验证内部数据未被修改
        points2 = list(source)
        assert points2[0][0][0] != 999.0
        np.testing.assert_allclose(points2[0][0], np.zeros(6))

    def test_load_old_pickle_format(self, tmp_path):
        """加载旧 pickle 格式 → 迭代产出正确的 q_desired 序列。"""
        path = tmp_path / "old.pkl"
        u_history = _save_old_pickle(path, n_steps=5)
        source = FileSource(path)
        points = list(source)
        assert len(points) == 5
        for i, (q, t) in enumerate(points):
            np.testing.assert_allclose(q, u_history[i])

    def test_accepts_string_path(self, tmp_path):
        """FileSource 接受字符串路径（不仅限 Path）。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=3)
        source = FileSource(str(path))
        points = list(source)
        assert len(points) == 3

    def test_yields_numpy_array_and_float(self, tmp_path):
        """迭代产出的 q 是 np.ndarray，t 是 float。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=3)
        source = FileSource(path)
        q, t = next(iter(source))
        assert isinstance(q, np.ndarray)
        assert q.shape == (6,)
        assert isinstance(t, float)


# ── ResampledSource 测试 ──


class TestResampledSource:
    """ResampledSource 重采样装饰器测试。"""

    def test_speed_factor_0_1_more_points(self, tmp_path):
        """speed_factor=0.1 → 约 10 倍点数。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=100, dt=0.005)
        source = ResampledSource(
            FileSource(path), InterpolatingResampler(), speed_factor=0.1
        )
        points = list(source)
        # M = 10*(N-1)+1 = 991
        assert len(points) == 10 * (100 - 1) + 1

    def test_speed_factor_1_same_points(self, tmp_path):
        """speed_factor=1.0 → 点数不变。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=100, dt=0.005)
        source = ResampledSource(
            FileSource(path), InterpolatingResampler(), speed_factor=1.0
        )
        points = list(source)
        assert len(points) == 100

    def test_points_at_original_timestamps_match(self, tmp_path):
        """speed_factor=0.1 时每 10 个采样点对应一个原始点，值精确重现。"""
        path = tmp_path / "traj.npz"
        q_desired, _ = _save_test_npz(path, n_steps=100, dt=0.005)
        source = ResampledSource(
            FileSource(path), InterpolatingResampler(), speed_factor=0.1
        )
        points = list(source)
        for i in range(100):
            np.testing.assert_allclose(points[10 * i][0], q_desired[i], atol=1e-9)

    def test_target_dt_controls_spacing(self, tmp_path):
        """target_dt 控制输出时间戳间隔。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=100, dt=0.005)
        source = ResampledSource(
            FileSource(path),
            InterpolatingResampler(),
            speed_factor=1.0,
            target_dt=0.01,
        )
        points = list(source)
        ts = [t for _, t in points]
        diffs = np.diff(ts)
        np.testing.assert_allclose(diffs, 0.01, atol=1e-15)

    def test_nested_decorator_double_resampling(self, tmp_path):
        """嵌套装饰器：两层 ResampledSource 各 0.1 → 约 100 倍点数。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=100, dt=0.005)
        inner = ResampledSource(
            FileSource(path), InterpolatingResampler(), speed_factor=0.1
        )
        outer = ResampledSource(inner, InterpolatingResampler(), speed_factor=0.1)
        points = list(outer)
        # 两层 0.1 → 10*(10*(N-1))+1 = 9901
        assert len(points) == 10 * (10 * (100 - 1)) + 1

    def test_nested_decorator_endpoints_match(self, tmp_path):
        """嵌套重采样首末点与原始首末点一致（线性数据插值精确）。"""
        path = tmp_path / "traj.npz"
        q_desired, _ = _save_test_npz(path, n_steps=100, dt=0.005)
        inner = ResampledSource(
            FileSource(path), InterpolatingResampler(), speed_factor=0.1
        )
        outer = ResampledSource(inner, InterpolatingResampler(), speed_factor=0.1)
        points = list(outer)
        np.testing.assert_allclose(points[0][0], q_desired[0], atol=1e-9)
        np.testing.assert_allclose(points[-1][0], q_desired[-1], atol=1e-9)

    def test_copy_prevents_aliasing(self, tmp_path):
        """ResampledSource 产出 q 的 copy()，修改不影响重采样结果。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=10, dt=0.005)
        source = ResampledSource(
            FileSource(path), InterpolatingResampler(), speed_factor=0.1
        )
        points = list(source)
        original_first = points[0][0].copy()
        points[0][0][0] = 999.0
        points2 = list(source)
        np.testing.assert_allclose(points2[0][0], original_first, atol=1e-9)


# ── TcpSpeedLimiter 测试 ──


class TestTcpSpeedLimiter:
    """TcpSpeedLimiter TCP 速度限制装饰器测试。"""

    def test_set_max_tcp_speed_called_before_iteration(self, tmp_path):
        """迭代前设置 max_tcp_speed（首个 next 触发设置）。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=5)
        robot = MockRobot()
        limiter = TcpSpeedLimiter(
            FileSource(path), robot, max_tcp_speed=0.2, restore_speed=1.0
        )
        it = iter(limiter)
        # 生成器创建但未启动 → 无调用
        assert robot.calls == []
        next(it)  # 启动 → 设置 max_tcp_speed
        assert robot.calls == [0.2]

    def test_restore_speed_called_after_iteration(self, tmp_path):
        """迭代完成后恢复 restore_speed。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=5)
        robot = MockRobot()
        limiter = TcpSpeedLimiter(
            FileSource(path), robot, max_tcp_speed=0.2, restore_speed=1.0
        )
        list(limiter)
        assert robot.calls == [0.2, 1.0]

    def test_restore_on_exception(self):
        """内部 source 抛异常时 restore_speed 仍被调用（try/finally）。"""
        robot = MockRobot()
        limiter = TcpSpeedLimiter(
            RaisingSource(n_before_raise=1),
            robot,
            max_tcp_speed=0.2,
            restore_speed=1.0,
        )
        with pytest.raises(ValueError, match="inner source error"):
            list(limiter)
        assert robot.calls == [0.2, 1.0]

    def test_restore_speed_none_no_restore(self, tmp_path):
        """restore_speed=None → 迭代后不调用恢复。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=5)
        robot = MockRobot()
        limiter = TcpSpeedLimiter(
            FileSource(path), robot, max_tcp_speed=0.2, restore_speed=None
        )
        list(limiter)
        assert robot.calls == [0.2]

    def test_points_pass_through_unchanged(self, tmp_path):
        """TcpSpeedLimiter 透传内部 source 的点（不修改数据）。"""
        path = tmp_path / "traj.npz"
        q_desired, timestamps = _save_test_npz(path, n_steps=5)
        robot = MockRobot()
        limiter = TcpSpeedLimiter(
            FileSource(path), robot, max_tcp_speed=0.2, restore_speed=1.0
        )
        points = list(limiter)
        assert len(points) == 5
        for i, (q, t) in enumerate(points):
            np.testing.assert_allclose(q, q_desired[i])
            assert t == pytest.approx(timestamps[i])

    def test_default_restore_speed_none(self, tmp_path):
        """restore_speed 默认 None → 仅设置一次 max_tcp_speed。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=3)
        robot = MockRobot()
        limiter = TcpSpeedLimiter(FileSource(path), robot, max_tcp_speed=0.15)
        list(limiter)
        assert robot.calls == [0.15]


# ── 装饰器链组合测试 ──


class TestDecoratorChain:
    """装饰器链组合测试。"""

    def test_full_chain_works(self, tmp_path):
        """FileSource → ResampledSource → TcpSpeedLimiter 全链工作。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=50, dt=0.005)
        robot = MockRobot()
        source = TcpSpeedLimiter(
            ResampledSource(
                FileSource(path),
                InterpolatingResampler(),
                speed_factor=0.1,
            ),
            robot,
            max_tcp_speed=0.2,
            restore_speed=1.0,
        )
        assert isinstance(source, TrajectorySource)
        points = list(source)
        # 50 点 × 0.1 速度 → 10*(50-1)+1 = 491 点
        assert len(points) == 10 * (50 - 1) + 1
        # TCP 速度设置和恢复
        assert robot.calls == [0.2, 1.0]

    def test_tcp_wraps_resampled_wraps_file_endpoints(self, tmp_path):
        """验证装饰层次：TcpSpeedLimiter(ResampledSource(FileSource)) 首末点正确。"""
        path = tmp_path / "traj.npz"
        q_desired, _ = _save_test_npz(path, n_steps=50, dt=0.005)
        robot = MockRobot()
        file_source = FileSource(path)
        resampled = ResampledSource(
            file_source, InterpolatingResampler(), speed_factor=0.1
        )
        limiter = TcpSpeedLimiter(
            resampled, robot, max_tcp_speed=0.2, restore_speed=1.0
        )
        points = list(limiter)
        # 首点应与原始首点一致（speed_factor=0.1 时 new_qs[0] == qs[0]）
        np.testing.assert_allclose(points[0][0], q_desired[0], atol=1e-9)
        # 末点应与原始末点一致
        np.testing.assert_allclose(points[-1][0], q_desired[-1], atol=1e-9)

    def test_full_chain_restore_on_exception(self, tmp_path):
        """全链中 ResampledSource 内部异常时 TCP 速度仍恢复。"""
        path = tmp_path / "traj.npz"
        _save_test_npz(path, n_steps=5)
        robot = MockRobot()

        class RaisingFileSource(FileSource):
            """FileSource 子类，迭代末尾抛异常。"""

            def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
                yield from super().__iter__()
                raise RuntimeError("chain failure")

        source = TcpSpeedLimiter(
            ResampledSource(
                RaisingFileSource(path),
                InterpolatingResampler(),
                speed_factor=1.0,
            ),
            robot,
            max_tcp_speed=0.2,
            restore_speed=1.0,
        )
        with pytest.raises(RuntimeError, match="chain failure"):
            list(source)
        # try/finally 保证恢复
        assert robot.calls == [0.2, 1.0]
