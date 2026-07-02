#!/usr/bin/env python3
"""13_plan_speed_test.py — 规划速度测试。

测试 rm_set_plan_speed 对 rm_movej 运动时间的影响，
以及与 movej 的 v 参数的关系、对跟随运动的影响。

微风险: 规划运动，低速小幅，YES确认后执行。
Ctrl+C = 缓停。

用法:
    python 13_plan_speed_test.py                          # J1 +10°, v=20
    python 13_plan_speed_test.py --offset -10             # J1 -10°
    python 13_plan_speed_test.py --joint 2 --offset 10    # J2 +10°
    python 13_plan_speed_test.py --v 30                   # movej 速度 30%
    python 13_plan_speed_test.py --no-algo-check          # 跳过碰撞检查
"""

import sys
import time
from pathlib import Path

import numpy as np

# Windows GBK 控制台兼容：强制 stdout/stderr 用 UTF-8，避免 emoji 输出报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

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

# 运动结束确认轮询参数
_IDLE_POLL_INTERVAL = 0.05   # 秒
_IDLE_POLL_TIMEOUT = 5.0     # 秒
_FOLLOW_POLL_INTERVAL = 0.02  # 秒
_FOLLOW_TIMEOUT = 10.0       # 秒
_FOLLOW_TOL_DEG = 1.0        # 跟随运动到位容差（度）
# 回起点参数（快速返回，不计入测量）
_RETURN_PLAN_SPEED = 100
_RETURN_V = 50


def _wait_idle(arm) -> bool:
    """轮询 rm_get_arm_run_mode 直到返回空闲(mode==0)。

    用于确认 block=1 的 rm_movej 运动真正结束，
    防止 block 因超时提前返回而计时偏小。

    Args:
        arm: SDK RoboticArm 实例。

    Returns:
        是否在超时内进入空闲。
    """
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < _IDLE_POLL_TIMEOUT:
        try:
            ret, mode = arm.rm_get_arm_run_mode()
            if ret == 0 and mode == 0:
                return True
        except Exception:
            pass
        time.sleep(_IDLE_POLL_INTERVAL)
    return False


def _timed_movej(arm, q_deg: list[float], v: int, plan_speed: int) -> tuple[float, int, bool]:
    """设置 plan_speed 后阻塞执行 rm_movej 并计时。

    流程:
      1. rm_set_plan_speed(plan_speed)
      2. 计时 rm_movej(q_deg, v, r=0, connect=0, block=1)
      3. 轮询 run_mode 确认运动真正结束（防止 block 提前返回）

    Args:
        arm: SDK RoboticArm 实例。
        q_deg: 目标关节角度列表（度，长度 6）。
        v: movej 速度比例 1-100。
        plan_speed: 规划速度百分比 1-100。

    Returns:
        (运动耗时秒, SDK返回码, 是否确认空闲)。
    """
    arm.rm_set_plan_speed(plan_speed)
    t0 = time.perf_counter()
    ret = arm.rm_movej(q_deg, v, 0, 0, 1)
    elapsed = time.perf_counter() - t0
    idle = _wait_idle(arm)
    return elapsed, ret, idle


def _return_home(arm, q_deg: list[float]) -> None:
    """快速回起点（plan_speed=100, v=50, 阻塞）。

    回程不计时，使用全速以缩短测试总时长。

    Args:
        arm: SDK RoboticArm 实例。
        q_deg: 起点关节角度列表（度）。
    """
    arm.rm_set_plan_speed(_RETURN_PLAN_SPEED)
    arm.rm_movej(q_deg, _RETURN_V, 0, 0, 1)
    _wait_idle(arm)


def _timed_follow(ri, arm, q_target_deg: np.ndarray, plan_speed: int) -> tuple[float, int]:
    """跟随运动计时（非阻塞 rm_movej_follow，轮询到位）。

    rm_movej_follow 无 block 模式，单次发送后机械臂插值趋近目标，
    通过轮询关节角度误差判断是否到位。

    Args:
        ri: RobotInterface 实例（用于 get_arm_state）。
        arm: SDK RoboticArm 实例。
        q_target_deg: 目标关节角度（度，(6,)）。
        plan_speed: 发送前设置的规划速度百分比。

    Returns:
        (到位耗时秒, SDK返回码)。发送失败返回 (-1.0, ret)，轮询超时返回 (耗时, -1)。
    """
    arm.rm_set_plan_speed(plan_speed)
    t0 = time.perf_counter()
    ret = arm.rm_movej_follow(q_target_deg.tolist())
    if ret != 0:
        return -1.0, ret
    while time.perf_counter() - t0 < _FOLLOW_TIMEOUT:
        state = ri.get_arm_state()
        err = float(np.max(np.abs(np.degrees(state[:6]) - q_target_deg)))
        if err < _FOLLOW_TOL_DEG:
            return time.perf_counter() - t0, 0
        time.sleep(_FOLLOW_POLL_INTERVAL)
    return time.perf_counter() - t0, -1


def _print_table(plan_results: list[dict], follow_results: list[dict]) -> None:
    """打印运动时间对比表。

    Args:
        plan_results: 规划运动测量结果，每项 {plan_speed, v, time, ret, idle, note}。
        follow_results: 跟随运动测量结果，每项 {plan_speed, time, ret, note}。
    """
    print("\n" + "=" * 64)
    print("测量结果汇总")
    print("=" * 64)

    # 规划运动表
    if plan_results:
        baseline = plan_results[0]["time"]
        print("\n[规划运动 rm_movej]")
        header = f"{'plan_speed':>11s} | {'v':>4s} | {'耗时(s)':>9s} | {'比例':>8s} | 备注"
        print(header)
        print("-" * len(header))
        for r in plan_results:
            ratio = r["time"] / baseline if baseline > 0 else float("nan")
            note = r.get("note", "")
            if r.get("ret", 0) != 0:
                note = f"ret={r['ret']} " + note
            if not r.get("idle", True):
                note = "⚠️未确认空闲 " + note
            print(
                f"{r['plan_speed']:>11d} | {r['v']:>4d} | {r['time']:>9.3f} | "
                f"{ratio:>7.2f}x | {note}"
            )

    # 跟随运动表
    if follow_results:
        print("\n[跟随运动 rm_movej_follow]")
        header = f"{'plan_speed':>11s} | {'耗时(s)':>9s} | 备注"
        print(header)
        print("-" * len(header))
        for r in follow_results:
            note = r.get("note", "")
            if r.get("ret", 0) != 0:
                note = f"ret={r['ret']} " + note
            print(f"{r['plan_speed']:>11d} | {r['time']:>9.3f} | {note}")

    # plan_speed 缩放分析
    if len(plan_results) >= 3:
        t100 = plan_results[0]["time"]
        t50 = plan_results[1]["time"]
        t20 = plan_results[2]["time"]
        print("\n[plan_speed 缩放分析]")
        print(f"  T(plan_speed=50) / T(plan_speed=100) = {t50 / t100:.2f}  (期望 ≈ 2.0)")
        print(f"  T(plan_speed=20) / T(plan_speed=100) = {t20 / t100:.2f}  (期望 ≈ 5.0)")
        if len(plan_results) >= 4:
            tv50 = plan_results[3]["time"]
            v_base = plan_results[0]["v"]
            ratio_v = tv50 / t100 if t100 > 0 else float("nan")
            print("\n[v 参数对比]")
            print(
                f"  T(plan_speed=100, v=50) / T(plan_speed=100, v={v_base}) "
                f"= {ratio_v:.2f}  (v 更大应更快, <1.0)"
            )

    # 跟随运动影响判定
    if len(follow_results) >= 2:
        f20 = follow_results[0]["time"]
        f100 = follow_results[1]["time"]
        ratio = f20 / f100 if f100 > 0 else float("nan")
        print("\n[跟随运动受 plan_speed 影响？]")
        print(f"  T_follow(plan_speed=20) / T_follow(plan_speed=100) = {ratio:.2f}")
        if abs(ratio - 1.0) < 0.25:
            verdict = "不受影响（比例≈1.0）→ MPC 透传模式可放心设置 plan_speed"
        else:
            verdict = f"疑似受影响（比例={ratio:.2f}）→ MPC 场景需注意 plan_speed 副作用"
        print(f"  结论: {verdict}")

    print("=" * 64)


def main() -> None:
    """规划速度测试主流程。"""
    import argparse

    # Windows UTF-8 垫片（emoji 输出兼容）
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RM-65B 规划速度测试（微风险）")
    add_config_arg(parser)
    add_algo_check_arg(parser)
    parser.add_argument("--joint", type=int, default=1, choices=range(1, 7),
                        help="偏移关节编号 1-6（默认 1，即 J1）")
    parser.add_argument("--offset", type=float, default=10.0,
                        help="关节偏移角度（度，默认 +10；预检不过可改 -10）")
    parser.add_argument("--v", type=int, default=20, choices=range(1, 101),
                        help="rm_movej 速度比例 1-100（默认 20）")
    args = parser.parse_args()

    ri, config = load_and_connect(args.config)
    monitor = SafetyMonitor(config, ri)
    algo = None if args.no_algo_check else init_algo()
    arm = ri._arm  # 访问底层 SDK 以调用 rm_movej / rm_set_plan_speed / rm_get_arm_run_mode

    joint_idx = args.joint - 1

    # 1. 读起始关节角
    state = ri.get_arm_state()
    q_start_rad = state[:6].copy()
    q_start_deg = np.degrees(q_start_rad)

    # 2. 计算目标关节角（指定关节偏移）
    q_target_deg = q_start_deg.copy()
    q_target_deg[joint_idx] += args.offset
    q_target_rad = np.radians(q_target_deg)

    print("\n运动参数:")
    print(f"  偏移关节: J{args.joint}")
    print(f"  偏移角度: {args.offset:+.1f}°")
    print(f"  movej 速度 v: {args.v}%")
    print(f"  起始角度（度）: {q_start_deg.round(2)}")
    print(f"  目标角度（度）: {q_target_deg.round(2)}")

    # 3. 预检（目标位姿）
    ok, msg = pre_motion_check(ri, monitor, q_target_rad, state, algo)
    print(f"\n目标预检: {msg}")
    if not ok:
        print("已取消（可尝试 --offset 取反）")
        safe_disconnect(ri)
        return

    # 4. YES 确认
    confirm = input("\n即将开始规划运动测试，输入 YES 确认: ")
    if confirm.strip().upper() != "YES":
        print("已取消")
        safe_disconnect(ri)
        return

    q_start_list = q_start_deg.tolist()
    q_target_list = q_target_deg.tolist()

    plan_results: list[dict] = []
    follow_results: list[dict] = []

    try:
        # ---- 步骤 1: 基线 plan_speed=100 ----
        print(f"\n[步骤 1/5] 基线: plan_speed=100, v={args.v}")
        t, ret, idle = _timed_movej(arm, q_target_list, args.v, 100)
        plan_results.append({"plan_speed": 100, "v": args.v, "time": t,
                             "ret": ret, "idle": idle, "note": "基线"})
        print(f"  → 耗时 {t:.3f}s (ret={ret}, idle={idle})")
        _return_home(arm, q_start_list)

        # ---- 步骤 2: plan_speed=50（期望≈2×基线）----
        print(f"\n[步骤 2/5] 降速: plan_speed=50, v={args.v}")
        t, ret, idle = _timed_movej(arm, q_target_list, args.v, 50)
        plan_results.append({"plan_speed": 50, "v": args.v, "time": t,
                             "ret": ret, "idle": idle, "note": "期望≈2×基线"})
        print(f"  → 耗时 {t:.3f}s (ret={ret}, idle={idle})")
        _return_home(arm, q_start_list)

        # ---- 步骤 3: plan_speed=20（期望≈5×基线）----
        print(f"\n[步骤 3/5] 低速: plan_speed=20, v={args.v}")
        t, ret, idle = _timed_movej(arm, q_target_list, args.v, 20)
        plan_results.append({"plan_speed": 20, "v": args.v, "time": t,
                             "ret": ret, "idle": idle, "note": "期望≈5×基线"})
        print(f"  → 耗时 {t:.3f}s (ret={ret}, idle={idle})")
        _return_home(arm, q_start_list)

        # ---- 步骤 4: plan_speed=100, v=50（对比 v 参数）----
        print("\n[步骤 4/5] v 参数对比: plan_speed=100, v=50")
        t, ret, idle = _timed_movej(arm, q_target_list, 50, 100)
        plan_results.append({"plan_speed": 100, "v": 50, "time": t,
                             "ret": ret, "idle": idle, "note": "对比 v 参数"})
        print(f"  → 耗时 {t:.3f}s (ret={ret}, idle={idle})")
        _return_home(arm, q_start_list)

        # ---- 步骤 5: 跟随运动受 plan_speed 影响？ ----
        print("\n[步骤 5/5] 跟随运动: plan_speed=20 vs plan_speed=100")
        t20, ret20 = _timed_follow(ri, arm, q_target_deg, 20)
        follow_results.append({"plan_speed": 20, "time": t20, "ret": ret20,
                               "note": "跟随运动"})
        print(f"  plan_speed=20  → 到位 {t20:.3f}s (ret={ret20})")
        _return_home(arm, q_start_list)

        t100, ret100 = _timed_follow(ri, arm, q_target_deg, 100)
        follow_results.append({"plan_speed": 100, "time": t100, "ret": ret100,
                               "note": "跟随运动"})
        print(f"  plan_speed=100 → 到位 {t100:.3f}s (ret={ret100})")
        _return_home(arm, q_start_list)

    except KeyboardInterrupt:
        print("\n\n⚠️ Ctrl+C — 正在缓停...")
        try:
            ri.slow_stop()
            time.sleep(0.5)
            print("已缓停")
        except Exception:
            print("缓停调用异常")
    finally:
        # 关键: 无论正常结束、异常还是 Ctrl+C，都恢复 plan_speed=100，
        # 避免持久设置拖慢后续所有规划运动
        try:
            arm.rm_set_plan_speed(100)
            print("\n已恢复 plan_speed=100")
        except Exception:
            print("\n⚠️ 恢复 plan_speed 失败，请手动 rm_set_plan_speed(100)")
        safe_disconnect(ri)

    # 输出对比表（断开后纯数据展示）
    _print_table(plan_results, follow_results)


if __name__ == "__main__":
    main()
