"""轨迹源 — 装饰器链产出 (q_desired, timestamp) 序列。

TrajectorySource: 轨迹源接口（Protocol，运行时可检查）
FileSource: 从文件加载轨迹并迭代产出
ResampledSource: 重采样装饰器（时间轴拉伸 + 插值）
TcpSpeedLimiter: TCP 速度限制装饰器（可选安全层）

装饰器链用法::

    source = TcpSpeedLimiter(
        ResampledSource(FileSource(path), InterpolatingResampler(), 0.1),
        robot, max_tcp_speed=0.2, restore_speed=1.0
    )
    for q, t in source:
        robot.send_joint_command(q)
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np

from src.real.resample_strategy import ResampleStrategy
from src.real.trajectory_recorder import TrajectoryRecorder


@runtime_checkable
class TrajectorySource(Protocol):
    """轨迹源：迭代产出 (q_desired, timestamp) 序列。"""

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """迭代产出 (q_desired (6,), timestamp (float)) 元组。"""
        ...


class FileSource:
    """从 .npz 或 .pkl 文件加载轨迹，迭代产出 (q, t) 序列。

    内部用 TrajectoryRecorder.load() 加载，支持新旧格式。
    """

    def __init__(self, path: Path | str) -> None:
        """加载轨迹文件。

        Args:
            path: 轨迹文件路径（.npz 或 .pkl）。
        """
        traj = TrajectoryRecorder.load(Path(path))
        self._q_desired = np.asarray(traj.q_desired, dtype=float)
        self._timestamps = np.asarray(traj.timestamps, dtype=float)

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """迭代产出 (q_desired.copy(), timestamp)。

        copy() 防止外部修改内部数据。
        """
        for q, t in zip(self._q_desired, self._timestamps):
            yield q.copy(), float(t)


class ResampledSource:
    """重采样装饰器：包装内部 Source，应用重采样策略。

    收集内部 source 全部点 → 调用 ResampleStrategy.resample() → 产出重采样后的点。
    这是装饰器模式：可包装任意 TrajectorySource（FileSource 或其他装饰器）。

    核心场景：speed_factor=0.1 → 慢速 1/10，点数 ×10。
    """

    def __init__(
        self,
        inner: TrajectorySource,
        resampler: ResampleStrategy,
        speed_factor: float,
        target_dt: float | None = None,
    ) -> None:
        """初始化重采样装饰器。

        Args:
            inner: 内部轨迹源。
            resampler: 重采样策略（如 InterpolatingResampler）。
            speed_factor: 速度因子（0.1 = 1/10 速度）。
            target_dt: 目标采样间隔（秒），None 用原始 dt。
        """
        self._inner = inner
        self._resampler = resampler
        self._speed_factor = speed_factor
        self._target_dt = target_dt

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """收集内部全部点 → resample → 产出。"""
        points = list(self._inner)
        qs = np.array([p[0] for p in points])
        ts = np.array([p[1] for p in points])
        new_qs, new_ts = self._resampler.resample(
            qs, ts, self._speed_factor, self._target_dt
        )
        for q, t in zip(new_qs, new_ts):
            yield q.copy(), float(t)


@runtime_checkable
class TcpSpeedControllable(Protocol):
    """支持 TCP 速度控制的机器人接口（供 TcpSpeedLimiter 依赖）。"""

    def set_max_tcp_speed(self, speed: float) -> None:
        """设置 TCP 最大线速度（m/s）。"""
        ...


class TcpSpeedLimiter:
    """TCP 速度限制装饰器（可选安全层）。

    迭代前设置控制器 max_tcp_speed，迭代后恢复原值。
    这是装饰器模式：可包装任意 TrajectorySource。

    设计理由：plan_speed 对 rm_movej_follow 无效（固件分析 §11.3），
    但 max_tcp_speed 对 rm_movej_follow 有效（控制器固件层 TCP 速度限制）。
    """

    def __init__(
        self,
        inner: TrajectorySource,
        robot: TcpSpeedControllable,
        max_tcp_speed: float,
        restore_speed: float | None = None,
    ) -> None:
        """初始化 TCP 速度限制装饰器。

        Args:
            inner: 内部轨迹源。
            robot: 支持 TCP 速度控制的机器人。
            max_tcp_speed: 限制的 TCP 速度（m/s）。
            restore_speed: 迭代后恢复的速度，None 表示不恢复。
        """
        self._inner = inner
        self._robot = robot
        self._max_tcp_speed = max_tcp_speed
        self._restore_speed = restore_speed

    def __iter__(self) -> Iterator[tuple[np.ndarray, float]]:
        """设置 max_tcp_speed → yield from inner → 恢复（try/finally）。"""
        self._robot.set_max_tcp_speed(self._max_tcp_speed)
        try:
            yield from self._inner
        finally:
            if self._restore_speed is not None:
                self._robot.set_max_tcp_speed(self._restore_speed)
