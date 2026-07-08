"""轨迹重采样策略 — 时间轴拉伸 + 三次样条插值。

ResampleStrategy: 重采样策略接口（Protocol，运行时可检查）
InterpolatingResampler: 插值重采样策略（B 方案核心，scipy CubicSpline）

核心思想:
    原始轨迹在 [ts[0], ts[-1]] 上定义。speed_factor<1 表示慢放（运动时长拉伸
    1/speed_factor 倍），speed_factor>1 表示快放。在拉伸后的播放时间轴上以
    target_dt 间隔均匀采样，每次采样映射回原始时间轴求 CubicSpline 值。

时间映射:
    设新（播放）时间 t'，对应原始时间 t = ts[0] + (t' - ts[0]) * speed_factor。
    该线性映射保证 t' ∈ [ts[0], ts[0]+T/speed_factor] 时 t ∈ [ts[0], ts[-1]]，
    因此 CubicSpline 始终在定义域内求值，无需外推。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.interpolate import CubicSpline


@runtime_checkable
class ResampleStrategy(Protocol):
    """重采样策略接口。"""

    def resample(
        self,
        qs: np.ndarray,
        ts: np.ndarray,
        speed_factor: float,
        target_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """对轨迹进行重采样。

        Args:
            qs: (N, 6) 原始关节角度序列。
            ts: (N,) 原始时间戳（秒）。
            speed_factor: 速度因子（0.1 = 1/10 速度，运动时长 ×10）。
            target_dt: 目标采样间隔（秒）。None 表示用原始 dt。

        Returns:
            new_qs: (M, 6) 重采样后的关节角度。
            new_ts: (M,) 重采样后的时间戳。
        """
        ...


class InterpolatingResampler:
    """插值重采样策略（B 方案核心）。

    使用 scipy.interpolate.CubicSpline 进行 C² 连续插值。
    时间轴拉伸 1/speed_factor 倍，在拉伸后的时间轴上以 target_dt 间隔均匀采样，
    再将采样点线性映射回原始时间轴求 CubicSpline 值。
    """

    def resample(
        self,
        qs: np.ndarray,
        ts: np.ndarray,
        speed_factor: float,
        target_dt: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """重采样实现。

        算法步骤:
            1. 边界检查：空输入报错，单点原样返回，speed_factor<=0 报错。
            2. 原始 dt = ``np.median(np.diff(ts))``（鲁棒处理非均匀时间戳）。
            3. target_dt: 参数传入或用原始 dt。
            4. 新时间轴:
               - 原始时长 T = ts[-1] - ts[0]
               - 新时长 T_new = T / speed_factor
               - 新点数 M = int(T_new / target_dt) + 1
               - new_ts = ts[0] + np.arange(M) * target_dt
            5. CubicSpline 拟合 qs 关于 ts（axis=0，沿时间轴插值）。
            6. 映射回原始时间轴求值:
               sample_t = ts[0] + (new_ts - ts[0]) * speed_factor
               new_qs = cs(sample_t)
               clamp 到 [ts[0], ts[-1]] 防浮点漂移外推。

        Args:
            qs: (N, 6) 原始关节角度序列。
            ts: (N,) 原始时间戳（秒，须严格递增）。
            speed_factor: 速度因子，必须 > 0。
            target_dt: 目标采样间隔（秒）。None 表示用原始 dt。

        Returns:
            (new_qs, new_ts) 重采样后的关节角度 (M,6) 和时间戳 (M,)。

        Raises:
            ValueError: 输入序列为空、或 speed_factor <= 0。
        """
        # 1. 边界检查
        n = qs.shape[0]
        if n == 0 or ts.shape[0] == 0:
            raise ValueError("输入序列为空，无法重采样")
        if speed_factor <= 0:
            raise ValueError(f"speed_factor 必须 > 0，得到 {speed_factor}")
        if n == 1 or ts.shape[0] == 1:
            # 单点无法插值，原样返回（拷贝避免别名）
            return np.asarray(qs, dtype=np.float64).copy(), np.asarray(
                ts, dtype=np.float64
            ).copy()

        # 2. 原始 dt（中位数，鲁棒处理时间戳抖动）
        raw_dt = float(np.median(np.diff(ts)))
        dt = float(target_dt) if target_dt is not None else raw_dt
        if dt <= 0:
            raise ValueError(f"target_dt 必须 > 0，得到 {dt}")

        # 3. 新时间轴（拉伸后的播放时间轴）
        t0 = float(ts[0])
        t_end = float(ts[-1])
        T = t_end - t0
        T_new = T / speed_factor
        # 加小 epsilon 防浮点截断（如 99.0 被表示为 98.9999... → int 截成 98）
        # 1e-9 远大于 float64 累积误差（~1e-13），远小于真实非整数的小数部分（≥0.01）
        m = int(T_new / dt + 1e-9) + 1
        new_ts = t0 + np.arange(m, dtype=np.float64) * dt

        # 4. CubicSpline 拟合（沿 axis=0，即时间轴插值）
        cs = CubicSpline(ts, qs, axis=0)

        # 5. 映射回原始时间轴求值，clamp 防浮点漂移外推
        sample_t = t0 + (new_ts - t0) * speed_factor
        sample_t = np.clip(sample_t, t0, t_end)
        new_qs = np.asarray(cs(sample_t), dtype=np.float64)

        return new_qs, new_ts
