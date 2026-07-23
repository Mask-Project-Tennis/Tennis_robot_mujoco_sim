#!/usr/bin/env python3
"""04_send_zero_pose.py — 流式插值回到安全位姿。

从当前角度线性插值到安全位姿（J3=30°，避开肘部奇异区），逐步发送 rm_movej_follow 指令。
测试项目实际使用的流式接口，而非 SDK 规划运动。

用法:
    python 04_send_zero_pose.py
    python 04_send_zero_pose.py --duration 2.0      # 2秒到达（默认1秒）
    python 04_send_zero_pose.py --target 0 0 0 0 0 0  # 全零位姿（可能触发奇异性）
    python 04_send_zero_pose.py --no-algo-check      # 跳过碰撞/奇异性检查

微风险: 会有实际运动。YES 确认后执行，Ctrl+C 缓停。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import (
    add_algo_check_arg,
    add_config_arg,
    home_to_pose,
    init_algo,
    load_and_connect,
    safe_disconnect,
)
from src.real.safety_monitor import SafetyMonitor


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 流式插值回安全位姿（微风险）")
    add_config_arg(parser)
    add_algo_check_arg(parser)
    parser.add_argument("--duration", type=float, default=1.0, help="插值时长秒（默认 1.0）")
    parser.add_argument("--hz", type=float, default=100.0, help="发送频率 Hz（默认 100）")
    parser.add_argument(
        "--target",
        type=float,
        nargs=6,
        default=[0, 0, 30, 0, 0, 0],
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="目标位姿（度，默认 0 0 30 0 0 0，J3=30° 避开肘部奇异）",
    )
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)
    monitor = SafetyMonitor(config, ri)
    algo = None if args.no_algo_check else init_algo()

    # 1. 读当前角度
    state = ri.get_arm_state()
    q_current = state[:6].copy()
    q_target = np.radians(np.array(args.target))
    q_deg_current = np.degrees(q_current)

    print(f"\n当前关节角度（度）: {q_deg_current.round(2)}")
    print(f"目标关节角度（度）: {np.array(args.target).round(2)}")
    print(f"最大变化: {np.max(np.abs(q_deg_current)):.1f}°")
    print(f"插值时长: {args.duration:.1f}s ({args.hz:.0f}Hz)")

    # 2. YES 确认
    confirm = input("\n即将回到零位，输入 YES 确认: ")
    if confirm.strip().upper() != "YES":
        print("已取消")
        safe_disconnect(ri)
        return

    # 3. 流式插值回到目标位姿（内置预检）
    try:
        home_to_pose(ri, monitor, algo, q_target,
                     duration=args.duration, hz=args.hz)
    except SystemExit as e:
        print(f"\n{e}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Ctrl+C — 正在缓停...")
        ri.slow_stop()
        print("已缓停")
    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
