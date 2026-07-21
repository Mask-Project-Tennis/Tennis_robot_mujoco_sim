#!/usr/bin/env python
"""轨迹预检工具 — 在真机重演前验证轨迹安全性和合理性。

加载 .npz 轨迹文件，检查:
  1. 关节限位: q_desired 范围 vs 真机限位，裕度 <10 度标注
  2. 平滑性: 相邻步角速度 max(delta_q/dt)，突跳 >30 度/step 标注
  3. TCP 速度估计: tcp_pos 差分速度，标注峰值
  4. 绘图: 6 关节 q_desired 时间序列 + TCP xyz 轨迹

用法:
    python scripts/tools/inspect_trajectory.py results/traj.npz
    python scripts/tools/inspect_trajectory.py results/traj.npz --config configs/real_robot.yaml
    python scripts/tools/inspect_trajectory.py results/traj.npz --no-plot
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 支持从 scripts/ 目录直接运行，确保 from src.xxx 可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.real.trajectory_recorder import TrajectoryRecorder
from src.real.trajectory_safety import _JOINT_NAMES, check_joint_limits
from src.real.trajectory_types import ReplayTrajectory


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CONFIG_PATH = _PROJECT_ROOT / "configs" / "real_robot.yaml"


def _load_limits(
    config_path: str | None,
    no_config: bool = False,
) -> tuple[np.ndarray, np.ndarray, float]:
    """加载关节限位与固件 TCP 限制。

    优先级（M5 修复）：
        1. config_path 显式给定 → 加载该 YAML
        2. no_config=True → 用 RealRobotConfig dataclass 默认（不读 YAML）
        3. 默认 → 加载 _DEFAULT_CONFIG_PATH（configs/real_robot.yaml）

    原 M5 问题：config_path=None 时直接走 dataclass 默认，
    操作员若在 real_robot.yaml 中收紧限位（如 J2 改为 85°），运行
    inspect_trajectory 不带 --config 时拿到的是 dataclass 默认（90°），
    导致预检结果与实际部署限位脱节。

    Args:
        config_path: YAML 路径或 None。
        no_config: True 时强制使用 dataclass 默认（逃生口，测试用）。

    Returns:
        (q_lower_deg, q_upper_deg, firmware_tcp) 三元组。
    """
    from src.real.config import RealRobotConfig

    # M5: 默认加载项目 YAML，让操作员的 YAML 编辑生效
    effective_path = config_path
    if effective_path is None and not no_config:
        effective_path = str(_DEFAULT_CONFIG_PATH)

    cfg = (
        RealRobotConfig.from_yaml(effective_path)
        if effective_path
        else RealRobotConfig()
    )
    return np.degrees(cfg.q_lower), np.degrees(cfg.q_upper), float(cfg.max_tcp_speed)


def check_smoothness(
    q_desired: np.ndarray,
    dt: float,
    threshold_deg: float = 30.0,
) -> list[str]:
    """检查轨迹平滑性，返回突跳点告警列表。

    Args:
        q_desired: (N, 6) 目标关节角度（弧度）。
        dt: 时间步长（秒）。
        threshold_deg: 单步角变化阈值（度），超过此值告警。

    Returns:
        告警字符串列表，空列表表示平滑。
    """
    if len(q_desired) < 2:
        return []

    threshold_rad = np.radians(threshold_deg)
    delta_q = np.abs(np.diff(q_desired, axis=0))  # (N-1, 6)
    # 向量化：argwhere 一次性找出所有超阈值 [step, joint] 索引
    exceed = np.argwhere(delta_q > threshold_rad)  # (K, 2)
    return [
        f"  [突跳] step={int(i)+1} {_JOINT_NAMES[int(j)]}: "
        f"Δ={np.degrees(delta_q[i, j]):.1f}° > {threshold_deg}°"
        for i, j in exceed
    ]


def estimate_tcp_speed(tcp_pos: np.ndarray, dt: float) -> np.ndarray:
    """估计 TCP 速度（差分法）。

    Args:
        tcp_pos: (N, 3) TCP 位置序列（米）。
        dt: 时间步长（秒）。

    Returns:
        (N,) TCP 速度幅值序列（m/s），首末步用单边差分。
    """
    if len(tcp_pos) < 2:
        return np.zeros(len(tcp_pos))
    diff = np.diff(tcp_pos, axis=0)  # (N-1, 3)
    step_speeds = np.linalg.norm(diff, axis=1) / dt  # (N-1,)
    # 首末步用单边差分，中间用双边差分平均
    speeds = np.empty(len(tcp_pos))
    speeds[0] = step_speeds[0]
    speeds[-1] = step_speeds[-1]
    if len(speeds) > 2:
        speeds[1:-1] = (step_speeds[:-1] + step_speeds[1:]) / 2.0
    return speeds


def check_tcp_speed(
    tcp_speeds: np.ndarray,
    max_tcp: float = 2.0,
    firmware_tcp: float = 1.0,
) -> tuple[list[str], float]:
    """校验 TCP 峰值速度并给出推荐重演 speed 因子。

    设计文档 Stage 0.5 通过条件: 原始轨迹 TCP 峰值 < max_tcp (默认 2.0 m/s)。
    重演时实际 TCP ≈ 原始峰值 × speed_factor，须 < firmware_tcp（真机固件限制）。

    Args:
        tcp_speeds: (N,) TCP 速度幅值序列（m/s）。
        max_tcp: 原始轨迹峰值上限（m/s）。
        firmware_tcp: 真机固件 TCP 限制（m/s），用于反推推荐 speed。

    Returns:
        (warnings, rec_speed): 告警列表 + 推荐最小安全 speed 因子。
    """
    if len(tcp_speeds) == 0:
        return [], 1.0
    peak_tcp = float(tcp_speeds.max())
    warnings: list[str] = []
    if peak_tcp >= max_tcp:
        warnings.append(
            f"  [超限] TCP 峰值 {peak_tcp:.2f} m/s ≥ 阈值 {max_tcp:.2f}"
        )
    rec_speed = firmware_tcp / peak_tcp if peak_tcp > 0 else 1.0
    return warnings, rec_speed


def plot_trajectory(
    traj: ReplayTrajectory,
    q_check: np.ndarray,
    tcp_speeds: np.ndarray,
    title: str = "",
    overlay: tuple[ReplayTrajectory, np.ndarray, np.ndarray] | None = None,
) -> None:
    """绘制关节角度 + TCP 位置/速度图表，可选叠加第二条轨迹。

    Args:
        traj: 主轨迹数据（含 timestamps/tcp_pos/hit_step）。
        q_check: 主轨迹的关节角度序列 (N,6)，弧度。
        tcp_speeds: 主轨迹的 TCP 速度序列 (N,)，m/s。
        title: 图表标题。
        overlay: 对比模式叠加轨迹，三元组 (traj2, q2, speeds2)。
            traj2 为 ReplayTrajectory，q2 为关节角度，speeds2 为 TCP 速度。
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 2]}
    )

    # 上图: 6 关节角度
    ax1 = axes[0]
    t = traj.timestamps
    for j in range(6):
        ax1.plot(t, np.degrees(q_check[:, j]), label=_JOINT_NAMES[j])
    if overlay is not None:
        traj2, q2, _ = overlay
        t2 = traj2.timestamps
        for j in range(6):
            ax1.plot(t2, np.degrees(q2[:, j]), "--", alpha=0.6)
    if traj.hit_step >= 0:
        ax1.axvline(
            traj.timestamps[traj.hit_step],
            color="r", linestyle="--", alpha=0.5, label="hit_step",
        )
    ax1.set_xlabel("时间 (s)")
    ax1.set_ylabel("关节角度 (°)")
    ax1.set_title(title or "关节轨迹")
    ax1.legend(fontsize=7, loc="upper right")
    ax1.grid(True, alpha=0.3)

    # 下图: TCP 位置 + 速度
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    for j, label in enumerate(["x", "y", "z"]):
        ax2.plot(t, traj.tcp_pos[:, j], label=f"TCP_{label}", alpha=0.7)
    if overlay is not None:
        traj2, _, speeds2 = overlay
        t2 = traj2.timestamps
        for j, label in enumerate(["x", "y", "z"]):
            ax2.plot(t2, traj2.tcp_pos[:, j], "--", alpha=0.5)
        ax2_twin.plot(t2, speeds2, color="orange", linestyle=":", alpha=0.4)
    ax2_twin.plot(
        t, tcp_speeds, color="k", linestyle="--", alpha=0.5, label="speed (m/s)"
    )
    ax2.set_xlabel("时间 (s)")
    ax2.set_ylabel("TCP 位置 (m)")
    ax2_twin.set_ylabel("TCP 速度 (m/s)")
    ax2.legend(fontsize=7, loc="upper left")
    ax2_twin.legend(fontsize=7, loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="轨迹预检 — 在真机重演前验证轨迹安全性和合理性",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "trajectory",
        type=str,
        help="轨迹文件路径 (.npz 或 .pkl)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="真机配置 YAML（读取 q_lower/q_upper）。默认加载 configs/real_robot.yaml",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="强制使用 dataclass 内置限位（不读 YAML），测试或快速检查用",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="跳过绘图（仅终端报告）",
    )
    parser.add_argument(
        "--margin-deg",
        type=float,
        default=10.0,
        help="关节限位安全裕度（度）",
    )
    parser.add_argument(
        "--max-tcp",
        type=float,
        default=2.0,
        help="原始轨迹 TCP 峰值速度上限（m/s），超阈值告警。设计文档 Stage 0.5 标准 = 2.0",
    )
    parser.add_argument(
        "--use-actual",
        action="store_true",
        help="检查 q_actual（仿真实际角度）而非 q_desired（MPC 命令）。"
        "真机重演推荐检查 q_actual。",
    )
    return parser.parse_args()


def main() -> None:
    """入口主函数：加载轨迹 → 三项检查 → 终端报告 → 可选绘图。"""
    args = _parse_args()

    # 加载轨迹
    traj: ReplayTrajectory = TrajectoryRecorder.load(Path(args.trajectory))
    q_check = traj.q_actual if args.use_actual else traj.q_desired
    q_label = "q_actual" if args.use_actual else "q_desired"
    print(f"\n{'='*60}")
    print(f"轨迹预检: {args.trajectory}  [检查 {q_label}]")
    print(f"{'='*60}")
    print(f"步数: {len(q_check)}, dt: {traj.dt:.4f}s, "
          f"hit_step: {traj.hit_step}")
    print(f"init_q (度): {np.degrees(traj.init_q).round(1)}")
    for k, v in traj.metadata.items():
        print(f"metadata.{k}: {v}")

    # 关节限位 + 固件 TCP（统一从 RealRobotConfig 取，消除硬编码漂移）
    q_lower_deg, q_upper_deg, firmware_tcp = _load_limits(
        args.config, no_config=args.no_config
    )

    print(f"\n--- 1. 关节限位检查 (裕度 {args.margin_deg}°) ---")
    limit_warnings = check_joint_limits(
        q_check, q_lower_deg, q_upper_deg, margin_deg=args.margin_deg
    )
    if limit_warnings:
        for w in limit_warnings:
            print(w)
    else:
        print("  所有关节在限位内且有足够裕度")

    # 平滑性
    print("\n--- 2. 平滑性检查 (阈值 30°/step) ---")
    smoothness_warnings = check_smoothness(q_check, traj.dt)
    if smoothness_warnings:
        for w in smoothness_warnings:
            print(w)
    else:
        print("  无突跳点")

    # TCP 速度（原始轨迹峰值 + 阈值校验 + 推荐重演 speed 因子）
    print("\n--- 3. TCP 速度估计 ---")
    tcp_speeds = estimate_tcp_speed(traj.tcp_pos, traj.dt)
    peak_tcp = float(tcp_speeds.max())
    print(f"  峰值: {peak_tcp:.2f} m/s")
    print(f"  均值: {tcp_speeds.mean():.2f} m/s")
    peak_idx = int(np.argmax(tcp_speeds))
    print(f"  峰值位置: step {peak_idx} / {len(tcp_speeds)} "
          f"({traj.timestamps[peak_idx]:.3f}s)")

    # TCP 阈值校验 + 推荐 speed（设计文档 Stage 0.5: 原始峰值 < 2.0 m/s）
    tcp_warnings, rec_speed = check_tcp_speed(
        tcp_speeds, max_tcp=args.max_tcp, firmware_tcp=firmware_tcp,
    )
    print(f"  建议重演 --speed ≤ {rec_speed:.2f} "
          f"(固件 {firmware_tcp:.1f} / 原峰 {peak_tcp:.2f})")
    for w in tcp_warnings:
        print(w)

    # 汇总结论
    all_ok = not limit_warnings and not smoothness_warnings and not tcp_warnings
    print(f"\n{'='*60}")
    if all_ok:
        print("结论: 轨迹安全，可进入真机重演")
    else:
        n_total = len(limit_warnings) + len(smoothness_warnings) + len(tcp_warnings)
        print(f"结论: 发现 {n_total} 告警（限位 {len(limit_warnings)} + "
              f"突跳 {len(smoothness_warnings)} + TCP {len(tcp_warnings)}），请审查后再决定")
    print(f"{'='*60}\n")

    # 绘图
    if not args.no_plot:
        plot_trajectory(
            traj, q_check, tcp_speeds,
            title=f"轨迹预检 — {Path(args.trajectory).name}",
        )


if __name__ == "__main__":
    main()
