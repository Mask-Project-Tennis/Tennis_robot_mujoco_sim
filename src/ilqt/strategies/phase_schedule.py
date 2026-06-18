"""阶段调度策略 — 根据剩余击球步数判定 far/mid/near 阶段。

阶段判定驱动 iLQR 迭代数和代价权重的阶段自适应（V11 行 989-1003）。
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class PhaseSchedule(Protocol):
    """阶段调度接口 — far/mid/near 分类 + 阶段参数查询。"""

    def classify(self, k_hit: int) -> str:
        """根据剩余击球步数判定阶段。

        Args:
            k_hit: 距击球时刻的剩余步数。

        Returns:
            "far" / "mid" / "near"。
        """
        ...


class DefaultPhaseSchedule:
    """默认阶段调度 — 与 V11 `_classify_phase` 完全一致。

    - k_hit > far_threshold（默认 50）→ "far"
    - k_hit > near_threshold（默认 20）→ "mid"
    - 其它 → "near"
    """

    def __init__(self, far_threshold: int = 50, near_threshold: int = 20) -> None:
        """初始化阶段阈值。

        Args:
            far_threshold: far/mid 分界阈值（步数）。
            near_threshold: mid/near 分界阈值（步数）。
        """
        self._far_threshold: int = far_threshold
        self._near_threshold: int = near_threshold

    def classify(self, k_hit: int) -> str:
        """阶段分类（与 V11 `_classify_phase` 逻辑一致）。

        Args:
            k_hit: 剩余击球步数。

        Returns:
            "far" / "mid" / "near"。
        """
        if k_hit > self._far_threshold:
            return "far"
        if k_hit > self._near_threshold:
            return "mid"
        return "near"
