#!/usr/bin/env python3
"""05_send_joint_command.py — 发送任意关节角度。

三种输入方式:
  --deg 0 30 -15 0 5 0       度数（直观，推荐日常使用）
  --rad 0.0 0.524 -0.262 0 0.087 0   弧度（程序化测试）
  无参数                       交互式逐个输入

每步都经过: YES 确认 → pre_motion_check → 流式插值发送。

用法:
    python 05_send_joint_command.py --deg 0 30 -15 0 5 0
    python 05_send_joint_command.py --rad 0.0 0.5 -0.3 0.0 0.1 0.0
    python 05_send_joint_command.py --no-algo-check
    python 05_send_joint_command.py               # 交互式

中风险: 有实际运动。YES 确认 + 预检 + Ctrl+C 缓停。
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


def interactive_input(q_current_deg: np.ndarray) -> np.ndarray:
    """交互式逐个输入关节角度（度）。回车保持当前值。"""
    print("\n逐个输入关节角度（度），回车保持当前值:")
    q_deg = q_current_deg.copy()
    for i in range(6):
        val = input(f"  J{i+1} [{q_deg[i]:.1f}°]: ").strip()
        if val:
            try:
                q_deg[i] = float(val)
            except ValueError:
                print(f"    无效输入，保持 {q_deg[i]:.1f}°")
    return np.radians(q_deg)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="RM-65B 发送任意关节角度（中风险）")
    add_config_arg(parser)
    add_algo_check_arg(parser)
    parser.add_argument("--deg", type=float, nargs=6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="目标关节角度（度）")
    parser.add_argument("--rad", type=float, nargs=6, metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
                        help="目标关节角度（弧度）")
    parser.add_argument("--duration", type=float, default=1.0, help="插值时长秒（默认 1.0）")
    parser.add_argument("--hz", type=float, default=100.0, help="发送频率 Hz（默认 100）")
    args = parser.parse_args()

    if args.deg is not None and args.rad is not None:
        print("❌ --deg 和 --rad 不能同时使用")
        return

    ri, config = load_and_connect(args.config)
    monitor = SafetyMonitor(config, ri)
    algo = None if args.no_algo_check else init_algo()

    # 1. 读当前角度
    state = ri.get_arm_state()
    q_current = state[:6].copy()
    q_deg_current = np.degrees(q_current)

    # 2. 确定目标角度
    if args.deg is not None:
        q_target = np.radians(np.array(args.deg))
    elif args.rad is not None:
        q_target = np.array(args.rad)
    else:
        q_target = interactive_input(q_deg_current)

    q_deg_target = np.degrees(q_target)
    delta_deg = q_deg_target - q_deg_current

    print(f"\n当前角度（度）: {q_deg_current.round(2)}")
    print(f"目标角度（度）: {q_deg_target.round(2)}")
    print(f"变化量  （度）: {delta_deg.round(2)}")
    print(f"最大变化: {np.max(np.abs(delta_deg)):.1f}°")

    # 3. 安全预检
    ok, msg = pre_motion_check(ri, monitor, q_target, state, algo)
    print(f"\n预检结果: {msg}")
    if not ok:
        print("已取消")
        safe_disconnect(ri)
        return

    # 4. YES 确认
    confirm = input("\n即将发送关节角度，输入 YES 确认: ")
    if confirm.strip().upper() != "YES":
        print("已取消")
        safe_disconnect(ri)
        return

    # 5. 流式插值发送
    dt = 1.0 / args.hz
    n_steps = int(args.duration * args.hz)
    print(f"\n开始运动（{n_steps} 步）...")

    try:
        for i in range(1, n_steps + 1):
            alpha = i / n_steps
            q = q_current * (1 - alpha) + q_target * alpha
            ri.send_joint_command(q)
            time.sleep(dt)

        # 保持目标
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
