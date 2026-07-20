"""轨迹安全检查纯函数（无 IO，可跨模块复用）。

从 scripts/tools/inspect_trajectory.py 抽取，供:
  - inspect_trajectory CLI（脚本层 IO + 绘图）
  - replay_pipeline.run_replay（重演前预检）
共享，消除限位检查逻辑的 DRY 风险。

设计原则: 仅依赖 numpy，不依赖 MuJoCo / 真机接口 / 文件系统。
"""

from __future__ import annotations

import numpy as np

# 关节名称（与 RM-65B 右臂 r_joint1~r_joint6 一一对应）
_JOINT_NAMES = ["J1(r_肩偏航)", "J2(r_肩俯仰)", "J3(r_肘)",
                "J4(r_腕1)", "J5(r_腕2)", "J6(r_腕3)"]


def check_joint_limits(
    q_desired: np.ndarray,
    q_lower_deg: np.ndarray,
    q_upper_deg: np.ndarray,
    margin_deg: float = 10.0,
) -> list[str]:
    """检查关节限位，返回违规/接近边界的告警列表。

    Args:
        q_desired: (N, 6) 目标关节角度（弧度）。
        q_lower_deg: (6,) 关节下限（度）。
        q_upper_deg: (6,) 关节上限（度）。
        margin_deg: 安全裕度（度），距边界小于此值告警。
            run_replay 预检建议传 0.0（仅看硬超限）；
            inspect_trajectory 工具用默认 10.0（含裕度告警）。
            margin_deg=0 时裕度分支自动短路（dist<x 且 dist>=0 不可同时成立）。

    Returns:
        告警字符串列表，空列表表示全部安全。
        每条告警以 "[超限]" 或 "[裕度]" 前缀标识类型，便于消费方按前缀过滤。
    """
    warnings: list[str] = []
    q_lower_rad = np.radians(q_lower_deg)
    q_upper_rad = np.radians(q_upper_deg)
    margin_rad = np.radians(margin_deg)

    for j in range(6):
        col = q_desired[:, j]
        q_min = float(col.min())
        q_max = float(col.max())

        # 硬超限检查
        if q_min < q_lower_rad[j]:
            warnings.append(
                f"  [超限] {_JOINT_NAMES[j]}: min={np.degrees(q_min):.1f}° < "
                f"lower={q_lower_deg[j]:.0f}°"
            )
        if q_max > q_upper_rad[j]:
            warnings.append(
                f"  [超限] {_JOINT_NAMES[j]}: max={np.degrees(q_max):.1f}° > "
                f"upper={q_upper_deg[j]:.0f}°"
            )

        # 裕度检查（未超限但接近边界）；margin_deg=0 时条件不可同时成立，自动跳过
        dist_lower = q_min - q_lower_rad[j]
        dist_upper = q_upper_rad[j] - q_max
        if dist_lower < margin_rad and dist_lower >= 0:
            warnings.append(
                f"  [裕度] {_JOINT_NAMES[j]}: 距下限 {np.degrees(dist_lower):.1f}° < {margin_deg}°"
            )
        if dist_upper < margin_rad and dist_upper >= 0:
            warnings.append(
                f"  [裕度] {_JOINT_NAMES[j]}: 距上限 {np.degrees(dist_upper):.1f}° < {margin_deg}°"
            )

    return warnings
