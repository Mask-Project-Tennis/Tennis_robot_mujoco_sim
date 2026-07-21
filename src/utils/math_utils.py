"""通用数学工具函数。"""

from __future__ import annotations

from typing import Union

import numpy as np
from numpy.typing import NDArray

# 支持标量或 NDArray 输入（quintic_smoothstep 等通用插值原语）
ScalarOrArray = Union[float, NDArray[np.floating]]


def normalize(v: NDArray[np.floating]) -> NDArray[np.floating]:
    """归一化向量。

    Args:
        v: 输入向量。

    Returns:
        单位向量。若输入为零向量，返回零向量。
    """
    n = np.linalg.norm(v)
    if n < 1e-10:
        return v.copy()
    return v / n


def rotation_matrix_x(angle: float) -> NDArray[np.float64]:
    """绕 x 轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rotation_matrix_y(angle: float) -> NDArray[np.float64]:
    """绕 y 轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotation_matrix_z(angle: float) -> NDArray[np.float64]:
    """绕 z 轴的旋转矩阵。"""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# 五次多项式平滑插值在 t=0.5 处的峰值速度因子。
# 推导: f(t) = 10t³-15t⁴+6t⁵ → f'(t)=30t²(1-t)² → f'(0.5)=30/16=1.875。
QUINTIC_SMOOTHSTEP_PEAK_FACTOR: float = 1.875


def quintic_smoothstep(t: ScalarOrArray) -> ScalarOrArray:
    """五次多项式平滑插值 s(t) = 10t³ - 15t⁴ + 6t⁵。

    经典 smoothstep 公式。在 t∈[0, 1] 区间内单调递增，
    起点和终点的位置/速度/加速度均为零（C² 连续，无 jerk 突变）。
    峰值速度因子 1.875 在 t=0.5 处取到（见 QUINTIC_SMOOTHSTEP_PEAK_FACTOR）。

    Args:
        t: 插值参数。标量或 NDArray。建议 ∈ [0, 1]，超出范围行为由多项式外插决定。

    Returns:
        插值结果，与输入同形。t=0 时返回 0，t=1 时返回 1。

    用途:
        - 真机 pre_motion 过渡（replay_pipeline.py）
        - 后摆轨迹生成 / 随挥过渡（任何需要平滑加减速的轨迹段）

    Examples:
        >>> quintic_smoothstep(0.0)
        0.0
        >>> quintic_smoothstep(1.0)
        1.0
        >>> quintic_smoothstep(0.5)  # = 10*0.125 - 15*0.0625 + 6*0.03125
        0.5
    """
    return 10 * t**3 - 15 * t**4 + 6 * t**5
