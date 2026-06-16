#!/usr/bin/env python3
"""04_send_zero_pose.py — 流式插值回到零位。

从当前角度线性插值到零位，逐步发送 rm_movej_follow 指令。
测试项目实际使用的流式接口，而非 SDK 规划运动。

用法:
    python 04_send_zero_pose.py
    python 04_send_zero_pose.py --duration 2.0      # 2秒到达（默认1秒）
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
    init_algo,
    load_and_connect,
    pre_motion_check,
    safe_disconnect,
)
from src.real.safety_monitor import SafetyMonitor


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 流式插值回零位（微风险）")
    add_config_arg(parser)
    add_algo_check_arg(parser)
    parser.add_argument("--duration", type=float, default=1.0, help="插值时长秒（默认 1.0）")
    parser.add_argument("--hz", type=float, default=100.0, help="发送频率 Hz（默认 100）")
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)
    monitor = SafetyMonitor(config, ri)
    algo = None if args.no_algo_check else init_algo()

    # 1. 读当前角度
    state = ri.get_arm_state()
    q_current = state[:6].copy()
    q_target = np.zeros(6)
    q_deg_current = np.degrees(q_current)

    print(f"\n当前关节角度（度）: {q_deg_current.round(2)}")
    print(f"目标关节角度（度）: {np.zeros(6).round(2)}")
    print(f"最大变化: {np.max(np.abs(q_deg_current)):.1f}°")
    print(f"插值时长: {args.duration:.1f}s ({args.hz:.0f}Hz)")

    # 2. 安全预检
    ok, msg = pre_motion_check(ri, monitor, q_target, state, algo)
    print(f"\n预检结果: {msg}")
    if not ok:
        print("已取消")
        safe_disconnect(ri)
        return

    # 3. YES 确认
    confirm = input("\n即将回到零位，输入 YES 确认: ")
    if confirm != "YES":
        print("已取消")
        safe_disconnect(ri)
        return

    # 4. 流式插值
    dt = 1.0 / args.hz
    n_steps = int(args.duration * args.hz)
    print(f"\n开始运动（{n_steps} 步）...")

    try:
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q = q_current * (1 - alpha) + q_target * alpha
            ri.send_joint_command(q)
            time.sleep(dt)

        # 保持目标位置
        ri.send_joint_command(q_target)
        time.sleep(0.2)

        # 读最终角度
        final_state = ri.get_arm_state()
        q_final = final_state[:6]
        error_deg = np.degrees(q_final - q_target)
        print(f"\n运动完成")
        print(f"最终角度（度）: {np.degrees(q_final).round(2)}")
        print(f"跟踪误差（度）: {error_deg.round(2)}")
        print(f"最大误差: {np.max(np.abs(error_deg)):.2f}°")

    except KeyboardInterrupt:
        print("\n\n⚠️ Ctrl+C — 正在缓停...")
        ri.slow_stop()
        print("已缓停")

    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
