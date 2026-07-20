#!/usr/bin/env python
"""真机轨迹重演入口脚本（薄壳 CLI）。

核心逻辑已抽取到 src/real/replay_pipeline.py，本文件仅负责
argparse → ReplayConfig → run_replay()。

用法:
    # 真机重演（1/2 速度）
    python scripts/replay_trajectory.py --trajectory results/traj.npz \
        --speed 0.5 --use-actual

    # Mock 模式（无真机，用 FakeRobot 验证流程）
    python scripts/replay_trajectory.py --trajectory results/traj.npz \
        --speed 0.1 --use-actual --mock

⚠️ --use-actual 必须加: 位置模式 MPC 的 q_desired 可达 J2=130°/J6=±360°（超出
真机限位），必须用 q_actual（仿真实际执行的关节角度）重演。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.real.replay_pipeline import ReplayConfig, ReplayResult, run_replay  # noqa: E402

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "real_robot.yaml"
)


def _parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="真机轨迹重演 — 从文件加载仿真轨迹，以可调慢速在真机上重演",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--trajectory", type=str, required=True,
        help="轨迹文件路径 (.npz 或 .pkl)",
    )
    parser.add_argument(
        "--speed", type=float, default=0.1,
        help="速度因子 (0.1 = 1/10 速度)",
    )
    parser.add_argument(
        "--pre-motion-duration", type=float, default=10.0,
        help="预运动过渡时长（秒），实际取 max(此值, 最小安全值)",
    )
    parser.add_argument(
        "--max-tcp-speed", type=float, default=0.0,
        help="TCP 速度限制 (m/s), 0=不限制",
    )
    parser.add_argument(
        "--config", type=str, default=str(_DEFAULT_CONFIG_PATH),
        help="真机配置文件",
    )
    parser.add_argument("--mock", action="store_true", help="使用 FakeRobot (无真机)")
    parser.add_argument(
        "--record", type=str, default=None,
        help="记录真机实际轨迹到文件",
    )
    parser.add_argument(
        "--target-dt", type=float, default=None,
        help="目标采样间隔 (秒), 默认用原始 dt",
    )
    parser.add_argument(
        "--force-mode", action="store_true",
        help="跳过控制模式安全检查（危险！仅用于确认旧文件是位置模式生成）",
    )
    parser.add_argument(
        "--use-actual", action="store_true",
        help="使用 q_actual（仿真实际执行角度）替代 q_desired（MPC 命令）。"
        "位置模式 MPC 的 q_desired 可能超出真机限位，推荐启用。",
    )
    args = parser.parse_args()

    if args.speed > 1.0:
        parser.error(
            f"--speed={args.speed} > 1.0 会加速轨迹，真机不安全。"
            "仅支持减速（0 < speed ≤ 1.0）。"
        )

    return args


def main() -> None:
    """入口：argparse → ReplayConfig → run_replay。"""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )

    cfg = ReplayConfig(
        trajectory_path=Path(args.trajectory),
        speed=args.speed,
        use_actual=args.use_actual,
        mock=args.mock,
        record=Path(args.record) if args.record else None,
        pre_motion_duration=args.pre_motion_duration,
        max_tcp_speed=args.max_tcp_speed,
        target_dt=args.target_dt,
        force_mode=args.force_mode,
        config_path=Path(args.config),
    )

    result: ReplayResult = run_replay(cfg)
    if not result.success:
        logger.error("重演失败 [%s]: %s", result.status, result.reason)
        sys.exit(1)


if __name__ == "__main__":
    main()
