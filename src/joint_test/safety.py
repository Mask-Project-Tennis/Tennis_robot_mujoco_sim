"""真机模式软件层安全防护。"""
from __future__ import annotations

import logging

import numpy as np

from src.joint_test.types import WaveformConfig, WaveformType

logger = logging.getLogger(__name__)


class JointSafetyGuard:
    """真机模式下的指令裁剪与预检。

    仿真模式不使用（传 None 即可）。

    三层职责:
        1. 预检查 (check_preconditions): 运行前返回警告列表
        2. 指令裁剪 (clip_command): 每步裁剪 q_des 到关节限位
        3. 默认参数保守: max_amplitude_rad=0.05, max_frequency_hz=1.0

    Args:
        q_lower: 关节角度下限 (6,) [rad]。
        q_upper: 关节角度上限 (6,) [rad]。
        qdot_max: 关节速度上限 (6,) [rad/s]。
        max_amplitude_rad: 单关节实验幅值硬上限 [rad]，默认 0.05（≈2.86°）。
        max_frequency_hz: 真机模式频率硬上限 [Hz]，默认 1.0。
    """

    def __init__(
        self,
        q_lower: np.ndarray,
        q_upper: np.ndarray,
        qdot_max: np.ndarray,
        max_amplitude_rad: float = 0.05,
        max_frequency_hz: float = 1.0,
    ) -> None:
        """初始化安全防护器。"""
        self._q_lo = np.asarray(q_lower, dtype=float)
        self._q_hi = np.asarray(q_upper, dtype=float)
        self._qdot_max = np.asarray(qdot_max, dtype=float)
        self._max_A = float(max_amplitude_rad)
        self._max_f = float(max_frequency_hz)

    def check_preconditions(self, config: WaveformConfig) -> list[str]:
        """运行前检查，返回警告列表（空列表=通过）。

        检查项:
            1. 幅值是否超过 max_amplitude_rad
            2. 频率是否超过 max_frequency_hz（step 波形跳过；chirp 取 f0 与 f1 的最大值）
            3. 波形范围 [offset-A, offset+A] 是否超出关节限位

        Args:
            config: 波形配置。

        Returns:
            警告字符串列表，空列表表示全部通过。
        """
        warnings: list[str] = []
        j = config.joint_idx

        # 检查 1: 幅值
        if config.amplitude_rad > self._max_A:
            warnings.append(
                f"幅值 {config.amplitude_rad:.3f} rad > 真机安全上限 {self._max_A:.3f} rad"
            )

        # 检查 2: 频率（step 波形不需要频率检查）
        if config.waveform != WaveformType.STEP:
            # chirp: 取起始频率与终止频率的最大值参与比较，防止 end_freq 绕过安全上限
            if config.waveform == WaveformType.CHIRP:
                effective_freq = max(
                    config.frequency_hz,
                    config.end_frequency_hz or config.frequency_hz,
                )
            else:
                effective_freq = config.frequency_hz
            if effective_freq > self._max_f:
                warnings.append(
                    f"频率 {effective_freq:.2f} Hz > 真机安全上限 {self._max_f:.2f} Hz"
                )

        # 检查 3: 波形范围是否超关节限位
        offset = config.offset_rad
        A = config.amplitude_rad
        wave_lo = offset - A
        wave_hi = offset + A
        if wave_lo < self._q_lo[j] or wave_hi > self._q_hi[j]:
            warnings.append(
                f"波形范围 [{wave_lo:.3f}, {wave_hi:.3f}] rad "
                f"超出关节 {j} 限位 "
                f"[{self._q_lo[j]:.3f}, {self._q_hi[j]:.3f}] rad"
            )

        return warnings

    def check_runtime_state(
        self,
        q_current: np.ndarray,
        qdot_current: np.ndarray,
    ) -> tuple[bool, str]:
        """运行时逐关节状态检查（每步调用）。

        检查实际关节位置是否在 [q_lo, q_hi]、实际速度模是否在 qdot_max 以内。
        任一关节越界即返回 (False, reason)；全部通过返回 (True, "")。

        Args:
            q_current: 实际关节角度 (6,)，弧度。
            qdot_current: 实际关节速度 (6,)，弧度/秒。

        Returns:
            (is_safe, reason)，is_safe=False 时 reason 给出首个违规关节的描述。
        """
        q_current = np.asarray(q_current, dtype=float)
        qdot_current = np.asarray(qdot_current, dtype=float)

        # NaN/Inf 输入直接判违规
        if not np.all(np.isfinite(q_current)):
            bad = np.where(~np.isfinite(q_current))[0]
            return False, f"q_current 含 NaN/Inf，关节 {bad.tolist()}"
        if not np.all(np.isfinite(qdot_current)):
            bad = np.where(~np.isfinite(qdot_current))[0]
            return False, f"qdot_current 含 NaN/Inf，关节 {bad.tolist()}"

        # 关节位置超限检查
        too_high = np.where(q_current > self._q_hi)[0]
        if len(too_high) > 0:
            j = int(too_high[0])
            return False, (
                f"关节 {j} 位置 {q_current[j]:.3f} rad 超上限 "
                f"{self._q_hi[j]:.3f} rad"
            )
        too_low = np.where(q_current < self._q_lo)[0]
        if len(too_low) > 0:
            j = int(too_low[0])
            return False, (
                f"关节 {j} 位置 {q_current[j]:.3f} rad 超下限 "
                f"{self._q_lo[j]:.3f} rad"
            )

        # 关节速度超限检查
        overspeed = np.where(np.abs(qdot_current) > self._qdot_max)[0]
        if len(overspeed) > 0:
            j = int(overspeed[0])
            return False, (
                f"关节 {j} 速度 {abs(qdot_current[j]):.3f} rad/s 超上限 "
                f"{self._qdot_max[j]:.3f} rad/s"
            )

        return True, ""

    def clip_command(self, q_des: np.ndarray, q_current: np.ndarray) -> np.ndarray:
        """裁剪 q_des 到关节限位范围。

        处理 NaN：替换为 q_current 对应值（安全降级）。
        处理 Inf：同上。
        处理 q_current 中的 NaN/Inf：用零值兜底（防止降级时把 NaN 透传给 clip）。

        Args:
            q_des: 期望关节角度 (6,)。
            q_current: 当前关节角度 (6,)，用于 NaN/Inf 降级。

        Returns:
            裁剪后的关节角度 (6,)。
        """
        q_des = np.asarray(q_des, dtype=float).copy()
        q_current = np.asarray(q_current, dtype=float)

        # q_current NaN/Inf 兜底：替换为 0（防止降级时把 NaN 透传）
        if not np.all(np.isfinite(q_current)):
            logger.error(
                "q_current 含 NaN/Inf，已用 0 兜底: indices=%s",
                np.where(~np.isfinite(q_current))[0],
            )
            q_current = np.where(np.isfinite(q_current), q_current, 0.0)

        # NaN/Inf 安全降级：替换为当前值
        invalid_mask = ~np.isfinite(q_des)
        if np.any(invalid_mask):
            logger.warning(
                "q_des 含 NaN/Inf，已降级为 q_current: indices=%s",
                np.where(invalid_mask)[0],
            )
            q_des[invalid_mask] = q_current[invalid_mask]

        return np.clip(q_des, self._q_lo, self._q_hi)
