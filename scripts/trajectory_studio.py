#!/usr/bin/env python
"""轨迹工作台 — 交互式终端界面。

浏览 → 安全检查 → 可视化确认 → 真机重演，一站式完成。
纯 input() 菜单驱动，零新依赖。

用法:
    python scripts/trajectory_studio.py
    python scripts/trajectory_studio.py --results-dir results --use-actual
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from src.real.replay_pipeline import ReplayConfig, run_replay  # noqa: E402
from src.real.trajectory_recorder import TrajectoryRecorder  # noqa: E402
from src.real.trajectory_types import ReplayTrajectory  # noqa: E402
from src.sim.rm65_env import RM65Env  # noqa: E402

# 复用 inspect_trajectory 的检查函数
_SCRIPTS_TOOLS = str(Path(__file__).resolve().parent / "tools")
if _SCRIPTS_TOOLS not in sys.path:
    sys.path.insert(0, _SCRIPTS_TOOLS)
from inspect_trajectory import (  # noqa: E402
    _load_limits,
    check_joint_limits,
    check_smoothness,
    check_tcp_speed,
    estimate_tcp_speed,
    plot_trajectory,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _PROJECT_ROOT / "src" / "robot" / "rm65_model.xml"
_DEFAULT_CONFIG = _PROJECT_ROOT / "configs" / "real_robot.yaml"


# ============================================================================
# 纯函数（可测试）
# ============================================================================


def _scan_trajectories(
    results_dir: Path,
) -> tuple[list[tuple[str, Path]], list[tuple[str, Path]]]:
    """扫描 results/ 目录，分类规划轨迹和真机录制。

    Args:
        results_dir: results/ 目录路径。

    Returns:
        (planned, real): 各为 (display_name, path) 列表。
    """
    planned: list[tuple[str, Path]] = []
    real: list[tuple[str, Path]] = []
    if not results_dir.exists():
        return planned, real
    for p in sorted(results_dir.glob("*.npz")):
        name = p.name
        if "real" in name.lower() or "stage3" in name.lower():
            real.append((name, p))
        else:
            planned.append((name, p))
    return planned, real


def _path_length(tcp_pos: np.ndarray) -> float:
    """计算 TCP 路径总长度（米）。"""
    if len(tcp_pos) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(tcp_pos, axis=0), axis=1)))


def _format_list_line(
    name: str, traj: ReplayTrajectory, is_real: bool = False
) -> str:
    """格式化轨迹列表单行摘要。"""
    n = len(traj.q_desired)
    plen = _path_length(traj.tcp_pos)
    if is_real:
        speed = traj.metadata.get("speed", "?")
        return f"  {name:<42s} {n:>4d}步  speed={speed}"
    hit = f"hit@{traj.hit_step}" if traj.hit_step >= 0 else "no-hit"
    return f"  {name:<42s} {n:>4d}步  {plen:.2f}m  {hit}"


def _safety_card(
    traj: ReplayTrajectory,
    q_check: np.ndarray,
    config_path: Path | None = None,
    margin_deg: float = 10.0,
    max_tcp: float = 2.0,
) -> tuple[list[str], float, float, float]:
    """运行三项安全检查，返回文字卡片行列表和推荐 speed。

    Returns:
        (lines, rec_speed, peak_tcp, firmware_tcp):
            卡片文字行 + 推荐重演 speed 因子 + TCP 峰值 + 固件 TCP 上限。
            peak_tcp/firmware_tcp 供 _do_replay 的速度告警二次确认使用。
    """
    q_lower_deg, q_upper_deg, firmware_tcp = _load_limits(
        str(config_path) if config_path else None
    )

    limit_warnings = check_joint_limits(
        q_check, q_lower_deg, q_upper_deg, margin_deg=margin_deg
    )
    smoothness_warnings = check_smoothness(q_check, traj.dt)
    tcp_speeds = estimate_tcp_speed(traj.tcp_pos, traj.dt)
    tcp_warnings, rec_speed = check_tcp_speed(
        tcp_speeds, max_tcp=max_tcp, firmware_tcp=firmware_tcp
    )

    lines: list[str] = []
    if limit_warnings:
        for w in limit_warnings:
            lines.append(f"  {'✗' if '超限' in w else '⚠'} {w.strip()}")
    else:
        lines.append(f"  ✓ 关节限位: 全部在限位内 (裕度 ≥ {margin_deg:.0f}°)")

    if smoothness_warnings:
        for w in smoothness_warnings:
            lines.append(f"  ⚠ {w.strip()}")
    else:
        lines.append("  ✓ 平滑性: 无突跳点")

    peak_tcp = float(tcp_speeds.max()) if len(tcp_speeds) > 0 else 0.0
    if tcp_warnings:
        for w in tcp_warnings:
            lines.append(f"  ✗ {w.strip()}")
    else:
        lines.append(f"  ✓ TCP峰值: {peak_tcp:.2f} m/s")
    lines.append(f"  → 建议 speed ≤ {rec_speed:.2f}")

    return lines, rec_speed, peak_tcp, firmware_tcp


def _speed_warning_message(
    speed: float,
    rec_speed: float,
    peak_tcp: float,
    firmware_tcp: float,
) -> str | None:
    """速度超 rec_speed 时返回告警消息，否则 None。

    严格阈值 speed > rec_speed（用户决策：不要更宽松，固件 Layer-1 已是主防线，
    本告警为易用性二次确认，避免操作员困惑于轨迹跟踪滞后）。

    Args:
        speed: 操作员选择的播放速度因子。
        rec_speed: 推荐速度上限（firmware_tcp / peak_tcp）。
        peak_tcp: 轨迹 TCP 速度峰值（m/s）。
        firmware_tcp: 真机固件 TCP 上限（m/s）。

    Returns:
        告警消息字符串（含预测峰值与推荐值），无需告警时返回 None。
        rec_speed <= 0 时跳过（防 peak_tcp=0 导致 inf rec_speed 误判）。
    """
    if rec_speed <= 0 or speed <= rec_speed:
        return None
    predicted = peak_tcp * speed
    return (
        f"预测 TCP 峰值 {predicted:.2f} m/s > 固件 {firmware_tcp:.1f} m/s，"
        f"推荐 speed ≤ {rec_speed:.2f}"
    )


# ============================================================================
# 交互函数（涉及 IO，不测试）
# ============================================================================


def _input(prompt: str) -> str:
    """input() 包装，处理 EOFError。"""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def _preview_3d(
    traj: ReplayTrajectory,
    use_actual: bool,
    preview_speed: float = 0.1,
) -> None:
    """在 MuJoCo viewer 中播放轨迹动画（运动学回放）。

    不跑物理仿真，仅逐帧设置 qpos → mj_forward → viewer.sync。
    关闭 viewer 窗口或按 Ctrl-C 退出。

    Args:
        traj: 待预览的轨迹。
        use_actual: True 用 q_actual，False 用 q_desired。
        preview_speed: 播放倍速，语义同视频播放器。
            0.1 = 十分之一速（慢放 10 倍，默认）；1.0 = 原速。
            **只允许慢放或原速**（0 < preview_speed ≤ 1.0），快进对安全预览无意义。
    """
    if not 0 < preview_speed <= 1.0:
        raise ValueError(
            f"preview_speed 必须在 (0, 1.0] 范围内（只允许慢放或原速），得到 {preview_speed}"
        )

    env = RM65Env(_MODEL_PATH)
    q_data = traj.q_actual if (use_actual and len(traj.q_actual) > 0) else traj.q_desired
    env.reset(traj.init_q)
    env.init_q_left = traj.init_q_left.copy()
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = traj.init_q_left
    env.data.qpos[env.BALL_QPOS_START : env.BALL_QPOS_START + 3] = (
        traj.ball_pos[0] if len(traj.ball_pos) > 0 else np.array([5.0, 0.0, 2.0])
    )
    mujoco.mj_forward(env.model, env.data)

    has_ball = len(traj.ball_pos) > 0 and np.any(traj.ball_pos[:, 2] > 0.01)

    duration_real = len(q_data) * traj.dt
    duration_view = duration_real / preview_speed
    print(
        f"\n>>> MuJoCo 3D 预览启动 ({len(q_data)} 步, dt={traj.dt:.4f}s)"
        f"\n    实际时长 {duration_real:.2f}s → 播放 {duration_view:.2f}s ({preview_speed:.2f}x 慢放)"
    )
    print("    关闭 viewer 窗口返回菜单\n")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -15
        viewer.cam.azimuth = 135

        for i in range(len(q_data)):
            if not viewer.is_running():
                break
            env.data.qpos[: env.NQ] = q_data[i]
            if has_ball and i < len(traj.ball_pos):
                env.data.qpos[
                    env.BALL_QPOS_START : env.BALL_QPOS_START + 3
                ] = traj.ball_pos[i]
            mujoco.mj_forward(env.model, env.data)
            viewer.sync()
            time.sleep(traj.dt / preview_speed)

    print(">>> 预览结束\n")


def _show_2d(
    traj: ReplayTrajectory,
    use_actual: bool,
    overlay: ReplayTrajectory | None = None,
) -> None:
    """显示 2D 关节/TCP 图表。"""
    q_check = traj.q_actual if use_actual else traj.q_desired
    tcp_speeds = estimate_tcp_speed(traj.tcp_pos, traj.dt)

    overlay_tuple = None
    if overlay is not None:
        q2 = overlay.q_actual if use_actual else overlay.q_desired
        speeds2 = estimate_tcp_speed(overlay.tcp_pos, overlay.dt)
        overlay_tuple = (overlay, q2, speeds2)

    plot_trajectory(
        traj, q_check, tcp_speeds,
        title="轨迹对比（蓝=规划 橙=录制）" if overlay else "轨迹预览",
        overlay=overlay_tuple,
    )


def _do_replay(
    traj_path: Path,
    use_actual: bool,
    config_path: Path,
    rec_speed: float,
    peak_tcp: float,
    firmware_tcp: float,
) -> None:
    """执行真机重演子流程：选速度 → 超速告警 → 确认 → run_replay。

    Args:
        traj_path: 轨迹文件路径。
        use_actual: 是否用 q_actual。
        config_path: RealRobotConfig yaml 路径。
        rec_speed: 推荐速度上限（firmware_tcp / peak_tcp）。
        peak_tcp: 轨迹 TCP 速度峰值（m/s），用于超速告警预测。
        firmware_tcp: 真机固件 TCP 上限（m/s），用于超速告警显示。
    """
    print("\n=== 真机重演 ===")

    speeds = [0.05, 0.1, 0.2, 0.3, 0.5]
    # 预选最接近推荐值的速度
    default_idx = min(
        range(len(speeds)), key=lambda i: abs(speeds[i] - min(rec_speed, 0.5))
    )

    print("选择速度:")
    for i, s in enumerate(speeds):
        marker = " ← 推荐" if i == default_idx else ""
        print(f"  [{i + 1}] {s:.2f}{marker}")
    print("  [0] 自定义输入")

    choice = _input("\n选择: ")
    if not choice:
        return

    if choice == "0":
        try:
            speed = float(_input("输入速度因子 (0 < speed ≤ 1.0): "))
        except ValueError:
            print("无效输入")
            return
        if speed <= 0 or speed > 1.0:
            print("速度必须在 (0, 1.0] 范围内")
            return
    else:
        try:
            idx = int(choice) - 1
            speed = speeds[idx]
        except (ValueError, IndexError):
            print("无效选择")
            return

    # I1: 严格阈值超速告警（speed > rec_speed 即触发，用户决策"不要更宽松"）
    # 不阻断 — 固件 Layer-1 是主防线，此处仅易用性二次确认
    warning = _speed_warning_message(speed, rec_speed, peak_tcp, firmware_tcp)
    if warning is not None:
        print(f"\n⚠️  {warning}")
        print("    固件 Layer-1 将裁剪速度，挥拍跟踪可能不到位")
        if _input("\n确认超速重演? (y/N): ").lower() != "y":
            print("已取消")
            return

    mock = _input("使用 FakeRobot (mock)? (y/N): ").lower() == "y"

    print("\n⚠️  即将执行:")
    print(f"    轨迹: {traj_path.name}")
    print(f"    速度: {speed:.2f} ({1.0 / speed:.0f}/{'原速' if speed < 1 else '原速'})")
    print(f"    Mock: {mock}")
    confirm = _input("\n确认执行? (y/N): ")
    if confirm.lower() != "y":
        print("已取消")
        return

    cfg = ReplayConfig(
        trajectory_path=traj_path,
        speed=speed,
        use_actual=use_actual,
        mock=mock,
        config_path=config_path,
    )
    result = run_replay(cfg)
    if result.success:
        print(f"\n重演结束: {result.steps} 步")
    else:
        print(f"\n重演失败 [{result.status}]: {result.reason}")

    # M7: 仅在 pre_motion 执行后提示（机械臂已移动）
    if result.status in ("ok", "safety_abort", "pre_motion_aborted"):
        print("\n⚠️  机械臂已移动，请手动复位后再进行下一次重演。")


def _trajectory_detail(
    name: str,
    path: Path,
    traj: ReplayTrajectory,
    use_actual: bool,
    config_path: Path,
    real_list: list[tuple[str, Path]],
) -> None:
    """单条轨迹详情菜单：安全卡片 → 2D → 3D → 对比 → 重演。"""
    q_check = traj.q_actual if (use_actual and len(traj.q_actual) > 0) else traj.q_desired
    q_label = "q_actual" if (use_actual and len(traj.q_actual) > 0) else "q_desired"

    while True:
        plen = _path_length(traj.tcp_pos)
        mode = traj.metadata.get("is_position_mode", "?")
        hit_type = traj.metadata.get("hit_type", "?")

        print(f"\n┌─ {name} ──────────────────────────")
        print(f"│ 步数: {len(q_check)}    路径: {plen:.2f}m    "
              f"击球步: {traj.hit_step}")
        print(f"│ 模式: {mode}    命中: {hit_type}    检查: {q_label}")

        lines, rec_speed, peak_tcp, firmware_tcp = _safety_card(traj, q_check, config_path)
        for line in lines:
            print(f"│ {line}")
        print("└──────────────────────────────────")

        print("\n  [1] 2D 图表 (关节角度 + TCP 位置/速度)")
        print("  [2] 3D 仿真预览 (MuJoCo 挥拍动画)")
        if real_list:
            print("  [3] 对比真机录制")
        print("  [4] 真机重演")
        print("  [5] 返回列表")

        choice = _input("\n选择: ")
        if choice == "1":
            _show_2d(traj, use_actual and len(traj.q_actual) > 0)
        elif choice == "2":
            _preview_3d(traj, use_actual and len(traj.q_actual) > 0)
        elif choice == "3" and real_list:
            _compare_submenu(traj, use_actual, real_list)
        elif choice == "4":
            _do_replay(path, use_actual, config_path, rec_speed, peak_tcp, firmware_tcp)
        elif choice == "5" or not choice:
            return


def _compare_submenu(
    planned: ReplayTrajectory,
    use_actual: bool,
    real_list: list[tuple[str, Path]],
) -> None:
    """对比子菜单：选择真机录制 → 叠加图表。"""
    if not real_list:
        print("  无可对比的真机录制")
        return

    print("\n选择真机录制:")
    for i, (name, _) in enumerate(real_list):
        print(f"  [{i + 1}] {name}")

    choice = _input("\n选择: ")
    if not choice:
        return
    try:
        idx = int(choice) - 1
        name, path = real_list[idx]
    except (ValueError, IndexError):
        print("无效选择")
        return

    real_traj = TrajectoryRecorder.load(path)
    _show_2d(planned, use_actual, overlay=real_traj)
    print(f"\n已叠加: {name}")


# ============================================================================
# 主入口
# ============================================================================


def main() -> None:
    """主入口：扫描 → 列表 → 详情菜单循环。"""
    import argparse

    # I3 修复：中文 Windows GBK 控制台无法编码 ✗⚠→╔ 等符号，
    # 强制 stdout 走 UTF-8（errors="replace" 容错）。
    # 修复方式与 tests/test_run_20hits_video.py 一致。
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace"
        )

    parser = argparse.ArgumentParser(
        description="轨迹工作台 — 交互式浏览/检查/预览/重演",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(_PROJECT_ROOT / "results"),
        help="轨迹文件目录",
    )
    parser.add_argument(
        "--config", type=str, default=str(_DEFAULT_CONFIG),
        help="真机配置文件（限位/TCP）",
    )
    parser.add_argument(
        "--use-actual", action="store_true", default=True,
        help="检查/重演 q_actual（推荐）",
    )
    parser.add_argument(
        "--use-desired", action="store_true",
        help="检查/重演 q_desired（覆盖 --use-actual）",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    results_dir = Path(args.results_dir)
    config_path = Path(args.config)
    use_actual = args.use_actual and not args.use_desired

    print("╔══════════════════════════════════════════╗")
    print("║     轨迹工作台 Trajectory Studio          ║")
    print("╠══════════════════════════════════════════╣")

    while True:
        planned, real = _scan_trajectories(results_dir)

        print(f"\n规划轨迹 ({len(planned)}):")
        # 预加载元数据用于列表显示
        planned_loaded: list[tuple[str, Path, ReplayTrajectory]] = []
        for name, path in planned:
            try:
                traj = TrajectoryRecorder.load(path)
                planned_loaded.append((name, path, traj))
                print(_format_list_line(name, traj))
            except Exception as e:
                print(f"  {name:<42s} [加载失败: {e}]")

        if real:
            print(f"\n真机录制 ({len(real)}):")
            for name, path in real:
                try:
                    traj = TrajectoryRecorder.load(path)
                    print(_format_list_line(name, traj, is_real=True))
                except Exception as e:
                    print(f"  {name:<42s} [加载失败: {e}]")

        choice = _input(
            f"\n选择规划轨迹编号 (1-{len(planned_loaded)})"
            " r=刷新 q=退出): "
        )
        if not choice or choice.lower() == "q":
            break
        if choice.lower() == "r":
            continue

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(planned_loaded):
                name, path, traj = planned_loaded[idx]
                _trajectory_detail(name, path, traj, use_actual, config_path, real)
            else:
                print("编号超出范围")
        except ValueError:
            print("无效输入")

    print("再见!")


if __name__ == "__main__":
    main()
