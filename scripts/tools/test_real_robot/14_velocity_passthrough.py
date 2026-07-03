#!/usr/bin/env python3
"""14_velocity_passthrough.py — 速度透传测试。

测试 rm_movev_canfd 笛卡尔空间速度透传功能。
中风险: 连续运动，需要实时监控 + 急停准备。

用法:
    python 14_velocity_passthrough.py
    python 14_velocity_passthrough.py --speed 0.01   # 超低速验证方向
    python 14_velocity_passthrough.py --speed 0.05   # 标准测试速度

测试维度:
    1. 低速方向验证（固定 0.01 m/s 沿 +X，需人工确认方向安全）
    2. 位移精度（v=--speed 基线，2 秒匀速）
    3. 速度线性度（0.5S / S / 2S 三档，1 秒各）
    4. follow=True vs False 对比
    5. trajectory_mode 0/1/2 对比
    6. 多轴叠加（线速度 + 角速度）
    7. 停止响应延迟（follow True/False）
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import (
    add_algo_check_arg,
    add_config_arg,
    load_and_connect,
    safe_disconnect,
)
from src.real.safety_monitor import SafetyMonitor

# 透传周期（与 rm_set_movev_canfd_init 的 dt 一致，控制器据此估算加减速）
_DT_MS = 10                  # 透传周期 ms
_DT_S = _DT_MS / 1000.0     # 秒
# 安全余量
_DEFAULT_SAFE_RADIUS = 0.15  # TCP 偏离起点最大允许距离 m
_RETURN_V = 50               # 回起点 rm_movej 速度比例
_RETURN_PLAN_SPEED = 100     # 回起点规划速度（全速）
_RETURN_TOL_DEG = 1.0        # 回起点到位容差（度）
_STOP_QDOT_THRESH = 0.01     # 判定已停止的关节速度阈值 rad/s
_STOP_POLL = 0.005           # 停止延迟轮询间隔 s
_STOP_TIMEOUT = 1.0          # 停止延迟超时 s
# 低速方向验证（固定，独立于 --speed，确保方向安全后再提速）
_CREEP_SPEED = 0.01          # m/s
_CREEP_DURATION = 1.0        # s
# 最大允许速度（硬上限，防止 --speed 过大）
_MAX_SPEED = 0.10            # m/s
# 多轴测试速度（基坐标系，X 线速度 + Z 线速度 + Z 角速度）
_MULTI_AXIS_VEL = [0.02, 0.0, 0.01, 0.0, 0.0, 0.1]


# --------------------------------------------------------------------------- #
# 底层工具函数
# --------------------------------------------------------------------------- #
def _send_zero(arm) -> None:
    """连续发送零速度确保机械臂停止（安全关键）。

    速度透传停止运动必须显式下发 ``[0]*6``，否则控制器可能保持最后速度。
    连发 3 次冗余，保证至少一帧到达控制器。
    """
    try:
        for _ in range(3):
            arm.rm_movev_canfd([0.0] * 6, True, 0, 0)
            time.sleep(0.005)
    except Exception as e:
        print(f"  ⚠️ 发送零速度异常: {e}")


def _get_tcp_pose(arm) -> np.ndarray | None:
    """读取当前末端位姿 [x,y,z,rx,ry,rz]。

    Args:
        arm: SDK RoboticArm 实例。

    Returns:
        (6,) 位姿数组；读取失败返回 None。
    """
    try:
        ret, state = arm.rm_get_current_arm_state()
        if ret == 0 and isinstance(state, dict):
            return np.array(state.get("pose", [0.0] * 6), dtype=float)
    except Exception:
        pass
    return None


def _displacement_from(p_start: np.ndarray, arm) -> float:
    """计算当前 TCP 与起点的欧氏位移（m）。

    Args:
        p_start: 起点位姿 [x,y,z,rx,ry,rz]。
        arm: SDK 实例。

    Returns:
        位移标量 m；读取失败返回 0.0（不触发越界误报）。
    """
    pose = _get_tcp_pose(arm)
    if pose is None:
        return 0.0
    return float(np.linalg.norm(pose[:3] - p_start[:3]))


# --------------------------------------------------------------------------- #
# 核心测试函数
# --------------------------------------------------------------------------- #
def _move_and_measure(
    arm,
    ri,
    p_start: np.ndarray,
    velocity: list[float],
    duration: float,
    follow: bool,
    mode: int,
    radio: int,
    safe_radius: float,
    label: str,
) -> dict:
    """发送指定速度运动 duration 秒，测量实际位移。

    用 perf_counter 补偿 sleep 抖动，保持发送周期 ≈ _DT_S
    （控制器用 init 的 dt 估算加速度，周期偏差会导致运动过冲/欠冲）。
    运动中每 10 步监控 TCP 位移，超 safe_radius 立即停止并标记越界。

    Args:
        arm: SDK RoboticArm 实例。
        ri: RobotInterface 实例。
        p_start: 起点位姿 [x,y,z,rx,ry,rz]。
        velocity: [vx,vy,vz,wx,wy,wz]，线 m/s，角 rad/s。
        duration: 持续时间 s。
        follow: True 高跟随，False 低跟随。
        mode: trajectory_mode 0/1/2（仅高跟随有效）。
        radio: 模式参数（mode=1: 0-100, mode=2: 0-1000）。
        safe_radius: TCP 允许最大位移 m。
        label: 测试标签（用于日志与汇总）。

    Returns:
        结果字典 {label, velocity, duration, displacement,
                 deltas, ret, overbound}。
    """
    n_steps = max(1, int(round(duration / _DT_S)))
    t0 = time.perf_counter()
    last_tick = t0
    ret_last = 0
    overbound = False

    for i in range(n_steps):
        try:
            ret_last = arm.rm_movev_canfd(list(velocity), follow, mode, radio)
        except Exception as e:
            print(f"  ⚠️ [{label}] 发送异常: {e}")
            break

        # 周期补偿：以 last_tick 为基准对齐，消除累积抖动
        target = last_tick + _DT_S
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        last_tick = target

        # 每 10 步（≈100ms）监控位移，超限立即停止
        if i > 0 and i % 10 == 0:
            disp = _displacement_from(p_start, arm)
            if disp > safe_radius:
                print(
                    f"  ⚠️ [{label}] TCP 越界 "
                    f"({disp * 1000:.0f}mm > {safe_radius * 1000:.0f}mm) 停止"
                )
                overbound = True
                break

    elapsed = time.perf_counter() - t0
    _send_zero(arm)

    pose_end = _get_tcp_pose(arm)
    if pose_end is None:
        deltas = np.full(6, float("nan"))
        displacement = float("nan")
    else:
        deltas = pose_end - p_start
        displacement = float(np.linalg.norm(deltas[:3]))

    return {
        "label": label,
        "velocity": list(velocity),
        "follow": follow,
        "mode": mode,
        "radio": radio,
        "duration": elapsed,
        "displacement": displacement,
        "deltas": deltas,
        "ret": ret_last,
        "overbound": overbound,
    }


def _measure_stop_latency(
    arm,
    ri,
    velocity: list[float],
    follow: bool,
) -> dict:
    """测量停止响应延迟。

    流程：先以指定速度运动 0.5s 进入稳态 → 发零速度 →
    轮询关节速度直到 |qdot| < _STOP_QDOT_THRESH 或超时。

    Args:
        arm: SDK 实例。
        ri: RobotInterface 实例。
        velocity: 稳态运动速度。
        follow: 高/低跟随。

    Returns:
        结果字典 {follow, latency_ms, timed_out}。
    """
    # 稳态运动 0.5s
    n_ramp = max(1, int(round(0.5 / _DT_S)))
    last_tick = time.perf_counter()
    for _ in range(n_ramp):
        try:
            arm.rm_movev_canfd(list(velocity), follow, 0, 0)
        except Exception:
            break
        target = last_tick + _DT_S
        now = time.perf_counter()
        if now < target:
            time.sleep(target - now)
        last_tick = target

    # 发零速度并计时到停止
    t0 = time.perf_counter()
    try:
        arm.rm_movev_canfd([0.0] * 6, follow, 0, 0)
    except Exception:
        pass
    # 预读一次清空数值微分缓存
    try:
        ri.get_arm_state()
    except Exception:
        pass

    latency_ms = float("nan")
    timed_out = True
    while time.perf_counter() - t0 < _STOP_TIMEOUT:
        try:
            state = ri.get_arm_state()
            qdot_max = float(np.max(np.abs(state[6:])))
        except Exception:
            qdot_max = float("inf")
        if qdot_max < _STOP_QDOT_THRESH:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            timed_out = False
            break
        time.sleep(_STOP_POLL)

    _send_zero(arm)
    return {
        "follow": follow,
        "latency_ms": latency_ms,
        "timed_out": timed_out,
    }


def _return_home(arm, ri, q_start_deg: np.ndarray) -> bool:
    """退出速度透传并关节跟随回起点。

    速度透传与关节跟随互斥，需先发零速度 + 等待再切换。
    若 rm_movej_follow 不能自动切换模式，回程失败时打印告警。

    Args:
        arm: SDK 实例。
        ri: RobotInterface 实例。
        q_start_deg: 起点关节角（度，(6,)）。

    Returns:
        是否回到起点（最大关节误差 < _RETURN_TOL_DEG）。
    """
    _send_zero(arm)
    time.sleep(0.2)
    try:
        arm.rm_set_plan_speed(_RETURN_PLAN_SPEED)
        arm.rm_movej_follow(q_start_deg.tolist())
    except Exception as e:
        print(f"  ⚠️ 回起点发送失败: {e}")
        return False
    # 轮询到位
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 5.0:
        try:
            state = ri.get_arm_state()
            err = float(np.max(np.abs(np.degrees(state[:6]) - q_start_deg)))
            if err < _RETURN_TOL_DEG:
                return True
        except Exception:
            pass
        time.sleep(0.05)
    return False


# --------------------------------------------------------------------------- #
# 汇总打印
# --------------------------------------------------------------------------- #
def _print_summary(
    baseline: dict,
    linearity: list[dict],
    follow_cmp: list[dict],
    mode_cmp: list[dict],
    multi: dict,
    stop_lat: list[dict],
) -> None:
    """打印所有测试维度汇总表。

    Args:
        baseline: 基线测量结果。
        linearity: 速度线性度结果列表。
        follow_cmp: follow True/False 对比结果。
        mode_cmp: trajectory_mode 0/1/2 对比结果。
        multi: 多轴运动结果。
        stop_lat: 停止延迟结果列表。
    """
    bar = "=" * 64
    print(f"\n{bar}\n速度透传测试汇总\n{bar}")

    # 1. 基线
    exp = baseline["velocity"][0] * baseline["duration"]
    print("\n[1. 位移精度基线]")
    print(
        f"  v={baseline['velocity'][0]:.3f} m/s, follow={baseline['follow']}, "
        f"mode={baseline['mode']}"
    )
    print(f"  实际位移 Δ={baseline['displacement'] * 1000:6.1f} mm"
          f"  预期 ≈{exp * 1000:6.1f} mm"
          f"  耗时 {baseline['duration']:.2f}s")
    if exp > 0:
        err_pct = abs(baseline["displacement"] - exp) / exp * 100
        print(f"  位移误差 {err_pct:5.1f}%")
    _print_deltas(baseline)

    # 2. 线性度
    print("\n[2. 速度线性度]")
    hdr = f"{'指令速度':>10s} | {'实际位移(mm)':>12s} | {'耗时(s)':>8s} | 备注"
    print(hdr)
    print("-" * len(hdr))
    for r in linearity:
        note = "越界!" if r["overbound"] else ""
        if r["ret"] != 0:
            note = f"ret={r['ret']} " + note
        v_cmd = r["velocity"][0]
        print(
            f"{v_cmd:>10.3f} | {r['displacement'] * 1000:>12.1f} | "
            f"{r['duration']:>8.2f} | {note}"
        )

    # 3. follow 对比
    print("\n[3. follow=True vs False]")
    hdr = f"{'follow':>7s} | {'位移(mm)':>10s} | {'耗时(s)':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in follow_cmp:
        print(
            f"{str(r['follow']):>7s} | {r['displacement'] * 1000:>10.1f} | "
            f"{r['duration']:>8.2f}"
        )

    # 4. mode 对比
    print("\n[4. trajectory_mode 对比]")
    hdr = f"{'mode':>5s} | {'radio':>6s} | {'位移(mm)':>10s} | {'耗时(s)':>8s}"
    print(hdr)
    print("-" * len(hdr))
    for r in mode_cmp:
        print(
            f"{r['mode']:>5d} | {r['radio']:>6d} | "
            f"{r['displacement'] * 1000:>10.1f} | {r['duration']:>8.2f}"
        )

    # 5. 多轴
    print("\n[5. 多轴运动]")
    print(f"  指令 {multi['velocity']}")
    _print_deltas(multi)

    # 6. 停止延迟
    print("\n[6. 停止响应延迟]")
    hdr = f"{'follow':>7s} | {'延迟(ms)':>10s} | 备注"
    print(hdr)
    print("-" * len(hdr))
    for r in stop_lat:
        if r["timed_out"]:
            note = f"超时(>{_STOP_TIMEOUT * 1000:.0f}ms)"
            lat = "N/A"
        else:
            note = ""
            lat = f"{r['latency_ms']:.1f}"
        print(f"{str(r['follow']):>7s} | {lat:>10s} | {note}")

    print(bar)
    print(
        "结论参考: follow/mode 的位移精度差异 + 停止延迟，"
        "决定速度透传是否适合 MPC 末端速度输出。"
    )
    print(bar)


def _print_deltas(r: dict) -> None:
    """打印单次测量的各轴位移增量。

    Args:
        r: _move_and_measure 返回的结果字典。
    """
    d = r["deltas"]
    if np.any(np.isnan(d)):
        print("  (末端位姿读取失败，无轴位移)")
        return
    print(
        f"  Δxyz(mm)=[{d[0] * 1000:+.1f},{d[1] * 1000:+.1f},"
        f"{d[2] * 1000:+.1f}]  "
        f"Δrpy(°)=[{np.degrees(d[3]):+.2f},{np.degrees(d[4]):+.2f},"
        f"{np.degrees(d[5]):+.2f}]"
    )


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main() -> None:
    """速度透传测试主流程。"""
    global _DT_MS, _DT_S  # 必须在任何 _DT_MS 引用前声明（argparse default 用到）
    import argparse

    # Windows UTF-8 垫片（emoji 输出兼容）
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="RM-65B 速度透传测试（中风险）")
    add_config_arg(parser)
    add_algo_check_arg(parser)
    parser.add_argument(
        "--speed", type=float, default=0.02,
        help=f"标准测试速度 m/s（默认 0.02，硬上限 {_MAX_SPEED}）",
    )
    parser.add_argument(
        "--dt", type=int, default=_DT_MS,
        help=f"透传周期 ms（默认 {_DT_MS}，需与发送周期一致）",
    )
    parser.add_argument(
        "--safe-radius", type=float, default=_DEFAULT_SAFE_RADIUS,
        help=f"TCP 偏离起点最大允许距离 m（默认 {_DEFAULT_SAFE_RADIUS}）",
    )
    args = parser.parse_args()

    # 速度安全校验
    if not (0.0 < args.speed <= _MAX_SPEED):
        print(f"❌ --speed 必须在 (0, {_MAX_SPEED}] m/s 范围内")
        sys.exit(1)
    if args.dt < 2:
        print("❌ --dt 不能小于 2ms（第三代控制器极限）")
        sys.exit(1)

    # 用 args.dt 覆盖模块常量（控制器估算加减速依赖此值）
    _DT_MS = args.dt
    _DT_S = _DT_MS / 1000.0

    ri, config = load_and_connect(args.config)
    monitor = SafetyMonitor(config, ri)
    # 速度透传无固定目标关节角，不做路径自碰撞/奇异性预检
    # （保留 --no-algo-check 仅保持与其他测试脚本 CLI 一致）
    arm = ri._arm  # 底层 SDK：rm_movev_canfd / rm_set_movev_canfd_init

    # 1. 读起始状态（关节角用于回起点，末端位姿用于位移测量）
    state = ri.get_arm_state()
    q_start_rad = state[:6].copy()
    q_start_deg = np.degrees(q_start_rad)
    p_start = _get_tcp_pose(arm)
    if p_start is None:
        print("❌ 无法读取当前末端位姿，终止测试")
        safe_disconnect(ri)
        return

    # 2. 当前姿态限位检查（速度透传无固定目标关节角，
    #    仅确认起始姿态安全，不做路径预检）
    if not monitor.is_safe(state, q_start_rad):
        print("❌ 当前姿态超限，请先将机械臂移至安全中间姿态")
        safe_disconnect(ri)
        return

    # 3. 初始化速度透传
    try:
        ret = arm.rm_set_movev_canfd_init(
            1,             # avoid_singularity_flag: 开启避奇异
            0,             # frame_type: 基坐标系
            _DT_MS,        # dt: 透传周期 ms
        )
    except Exception as e:
        print(f"❌ rm_set_movev_canfd_init 异常: {e}")
        safe_disconnect(ri)
        return
    if ret != 0:
        print(f"❌ 速度透传初始化失败 (ret={ret})，检查 dt/控制器版本")
        safe_disconnect(ri)
        return

    print("\n测试参数:")
    print(f"  标准速度: {args.speed:.3f} m/s")
    print(f"  透传周期 dt: {_DT_MS} ms")
    print(f"  安全半径: {args.safe_radius * 1000:.0f} mm")
    print(f"  起点关节角(°): {q_start_deg.round(2)}")
    print(f"  起点位姿(xyz mm): {(p_start[:3] * 1000).round(1)}")
    print("  测试维度: 方向验证 → 基线 → 线性度 → "
          "follow → mode → 多轴 → 停止延迟")

    # 4. YES 确认
    confirm = input(
        "\n⚠️ 即将开始笛卡尔空间连续运动（中风险）。"
        "急停按钮必须在手边。输入 YES 确认: "
    )
    if confirm.strip().upper() != "YES":
        print("已取消")
        safe_disconnect(ri)
        return

    speed = args.speed
    safe_radius = args.safe_radius

    # 结果容器
    baseline_res: dict = {}
    linearity_res: list[dict] = []
    follow_res: list[dict] = []
    mode_res: list[dict] = []
    multi_res: dict = {}
    stop_res: list[dict] = []

    try:
        # ---- 阶段 0: 低速方向验证（安全门禁）----
        creep = min(_CREEP_SPEED, speed)
        print(f"\n[阶段 0/7] 低速方向验证: +X {creep:.3f} m/s × "
              f"{_CREEP_DURATION}s（预期 +{creep * _CREEP_DURATION * 1000:.0f}mm）")
        d = _move_and_measure(
            arm, ri, p_start,
            [creep, 0.0, 0.0, 0.0, 0.0, 0.0],
            _CREEP_DURATION, follow=True, mode=0, radio=0,
            safe_radius=safe_radius, label="方向验证",
        )
        dx = d["deltas"][0] if not np.isnan(d["deltas"][0]) else float("nan")
        print(f"  X 位移 Δ={dx * 1000:+.1f}mm")
        if d["overbound"]:
            print("  ⚠️ 已越界，终止后续高速测试")
        else:
            ans = input("  +X 方向是否安全？(输入 yes 继续，其他取消): ")
            if ans.strip().lower() != "yes":
                print("  已取消后续测试")
            else:
                # 回起点消除蠕偏移，避免基线位移包含方向验证的 ~10mm
                if not _return_home(arm, ri, q_start_deg):
                    print("  ⚠️ 回起点未到位，后续位移测量可能有偏差")
                # 重新捕获起点（消除残余偏差）
                p_start = _get_tcp_pose(arm)
                # 进入完整测试矩阵
                _run_full_suite(
                    arm, ri, p_start, q_start_deg, speed, safe_radius,
                    baseline_res, linearity_res, follow_res,
                    mode_res, multi_res, stop_res,
                )

        # 回起点
        print("\n回起点...")
        if not _return_home(arm, ri, q_start_deg):
            print("  ⚠️ 未到位，请手动复位")

    except KeyboardInterrupt:
        print("\n\n⚠️ Ctrl+C — 发送零速度并缓停...")
        _send_zero(arm)
        try:
            ri.slow_stop()
            time.sleep(0.5)
            print("已缓停")
        except Exception:
            print("缓停调用异常")
    finally:
        # 安全关键: 无论正常/异常/Ctrl+C 都发零速度 + 缓停 + 断开
        _send_zero(arm)
        try:
            ri.slow_stop()
        except Exception:
            pass
        # 恢复 plan_speed，避免拖慢后续规划运动
        try:
            arm.rm_set_plan_speed(_RETURN_PLAN_SPEED)
        except Exception:
            pass
        safe_disconnect(ri)

    # 汇总（断开后纯数据展示）
    if baseline_res:
        _print_summary(
            baseline_res, linearity_res, follow_res,
            mode_res, multi_res, stop_res,
        )


def _run_full_suite(
    arm, ri, p_start: np.ndarray, q_start_deg: np.ndarray,
    speed: float, safe_radius: float,
    baseline_res: dict, linearity_res: list, follow_res: list,
    mode_res: list, multi_res: dict, stop_res: list,
) -> None:
    """执行方向验证通过后的完整测试矩阵。

    阶段 1-7，每阶段结束后调用 _return_home 回起点。
    任一阶段越界则打印告警并跳过后续高速阶段。

    Args:
        arm: SDK 实例。
        ri: RobotInterface 实例。
        p_start: 起点位姿。
        q_start_deg: 起点关节角（度）。
        speed: 标准测试速度 m/s。
        safe_radius: TCP 最大允许位移 m。
        baseline_res: 基线结果输出槽。
        linearity_res: 线性度结果列表输出槽。
        follow_res: follow 对比结果列表输出槽。
        mode_res: mode 对比结果列表输出槽。
        multi_res: 多轴结果输出槽。
        stop_res: 停止延迟结果列表输出槽。
    """
    # ---- 阶段 1: 位移精度基线 ----
    print(f"\n[阶段 1/7] 基线: +X {speed:.3f} m/s × 2.0s")
    baseline_res.update(_move_and_measure(
        arm, ri, p_start,
        [speed, 0.0, 0.0, 0.0, 0.0, 0.0], 2.0,
        follow=True, mode=0, radio=0,
        safe_radius=safe_radius, label="基线",
    ))
    print(f"  位移 {baseline_res['displacement'] * 1000:.1f}mm")
    _return_home(arm, ri, q_start_deg)

    # ---- 阶段 2: 速度线性度（0.5S / S / 2S，2S 截断到 _MAX_SPEED）----
    linearity_speeds = sorted({
        round(0.5 * speed, 4),
        round(speed, 4),
        round(min(2.0 * speed, _MAX_SPEED), 4),
    })
    print(f"\n[阶段 2/7] 线性度: {linearity_speeds}")
    for v in linearity_speeds:
        r = _move_and_measure(
            arm, ri, p_start,
            [v, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0,
            follow=True, mode=0, radio=0,
            safe_radius=safe_radius, label=f"线性度 {v}",
        )
        linearity_res.append(r)
        print(f"  v={v:.3f} → {r['displacement'] * 1000:.1f}mm")
        _return_home(arm, ri, q_start_deg)

    # ---- 阶段 3: follow=True vs False ----
    print(f"\n[阶段 3/7] follow 对比: +X {speed:.3f} m/s × 1.0s")
    for flw in (True, False):
        r = _move_and_measure(
            arm, ri, p_start,
            [speed, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0,
            follow=flw, mode=0, radio=0,
            safe_radius=safe_radius, label=f"follow={flw}",
        )
        follow_res.append(r)
        print(f"  follow={flw} → {r['displacement'] * 1000:.1f}mm")
        _return_home(arm, ri, q_start_deg)

    # ---- 阶段 4: trajectory_mode 0/1/2（均 follow=True）----
    mode_specs = [
        (0, 0),     # 完全透传
        (1, 50),    # 曲线拟合，平滑系数 50
        (2, 500),   # 滤波，参数 500
    ]
    print(f"\n[阶段 4/7] trajectory_mode 对比: +X {speed:.3f} m/s × 1.0s")
    for m, rad in mode_specs:
        r = _move_and_measure(
            arm, ri, p_start,
            [speed, 0.0, 0.0, 0.0, 0.0, 0.0], 1.0,
            follow=True, mode=m, radio=rad,
            safe_radius=safe_radius, label=f"mode={m}",
        )
        mode_res.append(r)
        print(f"  mode={m} radio={rad} → {r['displacement'] * 1000:.1f}mm")
        _return_home(arm, ri, q_start_deg)

    # ---- 阶段 5: 多轴运动 ----
    print(f"\n[阶段 5/7] 多轴: {_MULTI_AXIS_VEL} × 1.0s")
    multi_res.update(_move_and_measure(
        arm, ri, p_start, list(_MULTI_AXIS_VEL), 1.0,
        follow=True, mode=0, radio=0,
        safe_radius=safe_radius, label="多轴",
    ))
    _print_deltas(multi_res)
    _return_home(arm, ri, q_start_deg)

    # ---- 阶段 6: 停止响应延迟 ----
    print(f"\n[阶段 6/7] 停止延迟: +X {speed:.3f} m/s")
    for flw in (True, False):
        r = _measure_stop_latency(
            arm, ri, [speed, 0.0, 0.0, 0.0, 0.0, 0.0], flw,
        )
        stop_res.append(r)
        if r["timed_out"]:
            print(f"  follow={flw} → 超时(>{_STOP_TIMEOUT * 1000:.0f}ms)")
        else:
            print(f"  follow={flw} → {r['latency_ms']:.1f}ms")
        _return_home(arm, ri, q_start_deg)

    print("\n[阶段 7/7] 完成")


if __name__ == "__main__":
    main()
