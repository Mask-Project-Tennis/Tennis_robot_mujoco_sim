"""击球结果判定模块。

统一管理击球结果的分类和判定逻辑，消除 V12 控制台消息和提取脚本中
基于 pos_error 阈值的假阳性问题。

核心原则：命中判定基于真实物理接触（hit_type），而非规划精度（pos_error）。
"""
from __future__ import annotations


def classify_hit_result(
    active_contact: bool,
    passive_contact: bool,
    pos_error: float,
) -> str:
    """根据接触类型和规划精度分类击球结果。

    替代旧版 pos_error 阈值逻辑，消除无接触时的小 pos_error 假阳性。

    Args:
        active_contact: 是否主动击球（球拍速度 > 0.3 m/s 时触发）。
        passive_contact: 是否被动接触（球拍速度 ≤ 0.3 m/s 时触发）。
        pos_error: 末端到球（或规划点）的距离（米），作为辅助信息。

    Returns:
        分类消息字符串：
        - "击打成功" — 主动击球
        - "被动命中" — 被动接触
        - "未命中" — 无接触（即使 pos_error 小）
    """
    if active_contact:
        return f"RM-65 击打成功！（主动击球，pos_error={pos_error:.4f}m）"
    if passive_contact:
        return f"RM-65 被动命中！（球拍速度较低，pos_error={pos_error:.4f}m）"
    return (
        f"RM-65 未命中（无物理接触，规划距离 {pos_error:.4f}m，"
        f"球拍到达了规划点但球不在该位置）"
    )


def determine_hit_from_type(hit_type: str) -> bool:
    """根据 hit_type 判定是否命中（统一提取脚本逻辑）。

    替代旧版 pos_error < 0.153 阈值逻辑，消除假阳性。

    Args:
        hit_type: __RESULT__ 行的 hit_type 字段（"active" / "passive" / "miss"）。

    Returns:
        True 表示命中（active 或 passive），False 表示未命中。
    """
    return hit_type in ("active", "passive")
