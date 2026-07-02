#!/usr/bin/env python3
"""08_full_motion_test.py — 小幅正弦波运动测试。

单个关节做小幅正弦波运动，实时显示目标角度 vs 实际角度 vs 速度。
综合验证 send_joint_command 连续发送 + 数值微分速度 + 跟踪精度。

用法:
    python 08_full_motion_test.py                           # 默认: J1 ±5°, 周期2s, 持续10s
    python 08_full_motion_test.py --joint 2 --amplitude 10  # J2 ±10°
    python 08_full_motion_test.py --duration 20 --period 3  # 20s, 周期3s
    python 08_full_motion_test.py --no-algo-check            # 跳过碰撞检查

中风险: 连续运动。YES 确认 + 预检 + Ctrl+C 缓停 + 回起点。
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import (
    add_algo_check_arg,
    add_config_arg,
    init_algo,
    load_and_connect,
    pre_motion_check,
    safe_disconnect,
)
from src.real.safety_monitor import SafetyMonitor


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 正弦波运动测试（中风险）")
    add_config_arg(parser)
    add_algo_check_arg(parser)
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7),
                        help="运动关节编号 1-6（默认 1）")
    parser.add_argument("--amplitude", type=float, default=5.0,
                        help="正弦幅度（度，默认 5）")
    parser.add_argument("--period", type=float, default=2.0,
                        help="正弦周期（秒，默认 2）")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="持续时间（秒，默认 10）")
    parser.add_argument("--hz", type=float, default=100.0,
                        help="发送频率 Hz（默认 100）")
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)
    monitor = SafetyMonitor(config, ri)
    algo = None if args.no_algo_check else init_algo()

    dt = 1.0 / args.hz
    joint_idx = args.joint - 1

    # 1. 读起始位置
    state = ri.get_arm_state()
    q_start = state[:6].copy()
    q_deg_start = np.degrees(q_start)

    # 计算运动峰值角度（用于预检）
    q_peak = q_start.copy()
    q_peak[joint_idx] += np.radians(args.amplitude)

    print(f"\n运动参数:")
    print(f"  关节: J{args.joint}")
    print(f"  幅度: ±{args.amplitude}°")
    print(f"  周期: {args.period}s")
    print(f"  持续: {args.duration}s")
    print(f"  频率: {args.hz:.0f}Hz")
    print(f"  起始角度（度）: {q_deg_start.round(2)}")
    print(f"  峰值角度 J{args.joint}: {(np.degrees(q_peak[joint_idx])):.1f}°")

    # 2. 安全预检（检查峰值）
    ok, msg = pre_motion_check(ri, monitor, q_peak, state, algo)
    print(f"\n预检结果: {msg}")
    if not ok:
        print("已取消")
        safe_disconnect(ri)
        return

    # 3. YES 确认
    confirm = input("\n即将开始正弦波运动，输入 YES 确认: ")
    if confirm.strip().upper() != "YES":
        print("已取消")
        safe_disconnect(ri)
        return

    # 4. 正弦波运动
    t0 = time.time()
    omega = 2 * np.pi / args.period

    # 表头
    header = f"{'时间':>6s} | {'目标(°)':>8s} | {'实际(°)':>8s} | {'误差(°)':>8s} | {'速度(°/s)':>10s}"
    separator = "-" * len(header)

    print(f"\n{header}")
    print(separator)

    try:
        while True:
            elapsed = time.time() - t0
            if elapsed >= args.duration:
                break

            # 目标角度
            offset = np.radians(args.amplitude) * np.sin(omega * elapsed)
            q_target = q_start.copy()
            q_target[joint_idx] += offset

            # 发送
            ri.send_joint_command(q_target)
            time.sleep(dt)

            # 读实际
            state = ri.get_arm_state()
            q_actual = state[:6]
            qdot = state[6:]

            # 显示（每 0.2s 打印一行，避免刷屏）
            if int(elapsed * 5) != int((elapsed - dt) * 5):
                target_deg = np.degrees(q_target[joint_idx])
                actual_deg = np.degrees(q_actual[joint_idx])
                error_deg = actual_deg - target_deg
                vel_deg = np.degrees(qdot[joint_idx])
                print(f"  {elapsed:6.1f} | {target_deg:8.2f} | {actual_deg:8.2f} | {error_deg:8.2f} | {vel_deg:10.2f}")

        print(separator)
        print(f"\n运动完成")

    except KeyboardInterrupt:
        print(f"\n\n⚠️ Ctrl+C — 正在缓停...")
        ri.slow_stop()
        time.sleep(0.5)
        print("已缓停")

    # 5. 回到起始位置
    print("\n回到起始位置...")
    try:
        state = ri.get_arm_state()
        q_now = state[:6].copy()
        n_return = 100
        for i in range(1, n_return + 1):
            alpha = i / n_return
            q = q_now * (1 - alpha) + q_start * alpha
            ri.send_joint_command(q)
            time.sleep(0.01)
        print("✅ 已回到起始位置")
    except Exception:
        print("⚠️ 回起点失败，请手动复位")

    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
