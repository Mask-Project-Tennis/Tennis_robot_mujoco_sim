#!/usr/bin/env python3
"""11_joint_config_query.py — 关节级限位与管理状态查询。

读取关节级/驱动级限位、关节使能/错误/里程计状态，生成完整快照。
零风险：只读查询，不发送任何写指令。

用法:
    python 11_joint_config_query.py
    python 11_joint_config_query.py --hz 1    # 持续模式
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Windows GBK 控制台兼容：强制 stdout/stderr 用 UTF-8，避免 emoji 输出报错
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
    except (AttributeError, OSError):
        pass

# 确保能导入项目模块（_connect、src.*）
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _connect import add_config_arg, load_and_connect, safe_disconnect  # noqa: E402

# RPM → rad/s 换算系数（关节速度限位 SDK 单位为 RPM，config 为 rad/s）
RPM_TO_RAD_S = 2.0 * np.pi / 60.0


# ──────────────────────────────────────────────────────────────
# 查询工具函数
# ──────────────────────────────────────────────────────────────


def _query(arm: Any, name: str) -> tuple[int, Any]:
    """统一调用 SDK 查询接口，返回 (ret, payload)。

    适配两种返回形态：
      - tuple[int, list/dict]：直接拆分为 (ret, payload)
      - dict（内含 return_code）：读 return_code 作为 ret，dict 本身作为 payload
    异常时返回 (-1, {"exception": msg})，避免单接口失败拖垮整脚本。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        name: SDK 方法名（如 "rm_get_joint_max_pos"）。

    Returns:
        (ret, payload) — ret==0 表示成功。
    """
    try:
        result = getattr(arm, name)()
    except Exception as e:  # noqa: BLE001
        return -1, {"exception": str(e)}
    if isinstance(result, tuple) and len(result) == 2:
        ret, payload = result
        return int(ret), payload
    if isinstance(result, dict):
        return int(result.get("return_code", 0)), result
    return 0, result


def _call_list(
    arm: Any, name: str, label: str, errors: list[str]
) -> list[float] | None:
    """调用返回 list 的限位接口，取前 6 关节。

    Args:
        arm: RoboticArm 实例。
        name: SDK 方法名。
        label: 显示用标签（错误信息前缀）。
        errors: 错误收集列表，失败时追加条目。

    Returns:
        6 关节值列表，失败时返回 None。
    """
    ret, payload = _query(arm, name)
    if ret != 0:
        errors.append(f"{label} ret={ret}")
        return None
    try:
        return [float(v) for v in list(payload)[:6]]
    except (TypeError, ValueError):
        errors.append(f"{label} 返回结构异常: {payload!r}")
        return None


def _extract_joint_list(data: Any) -> list[float] | None:
    """从 dict 结果中提取 6 关节列表。

    dict 返回结构（如 rm_get_joint_err_flag）键名不确定，扫描所有非
    return_code 的值，取第一个长度 >=6 的列表前 6 项。

    Args:
        data: dict 或其他结构。

    Returns:
        6 元素列表，或 None（未找到合适列表）。
    """
    if not isinstance(data, dict):
        return None
    for key, value in data.items():
        if key == "return_code":
            continue
        if isinstance(value, (list, tuple)) and len(value) >= 6:
            return [float(v) for v in list(value)[:6]]
    return None


def _s(val: float | None, width: int = 8, digits: int = 2) -> str:
    """数值格式化为定宽字符串，None → 'N/A'。

    Args:
        val: 数值或 None。
        width: 字段宽度。
        digits: 小数位数。

    Returns:
        定宽字符串。
    """
    if val is None:
        return f"{'N/A':>{width}s}"
    return f"{val:{width}.{digits}f}"


# ──────────────────────────────────────────────────────────────
# 快照打印
# ──────────────────────────────────────────────────────────────


def _print_limit_table(title: str, pos: list[float] | None,
                       spd: list[float] | None, acc: list[float] | None) -> None:
    """打印单层限位表（位置°/速度RPM/加速度RPM·s⁻¹）。

    Args:
        title: 表标题。
        pos: 位置限位列表（°）或 None。
        spd: 速度限位列表（RPM）或 None。
        acc: 加速度限位列表（RPM/s）或 None。
    """
    print(f"\n  {title}")
    header = (
        f"  {'关节':>4s} | {'max_pos(°)':>10s} | "
        f"{'max_speed(RPM)':>14s} | {'max_acc(RPM/s)':>14s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(6):
        p = pos[i] if pos else None
        s = spd[i] if spd else None
        a = acc[i] if acc else None
        print(f"  {'J' + str(i + 1):>4s} | {_s(p, 10)} | {_s(s, 14)} | {_s(a, 14)}")


def _print_comparison(
    drive_pos: list[float] | None,
    joint_pos: list[float] | None,
    drive_spd: list[float] | None,
    joint_spd: list[float] | None,
    cfg_upper_deg: np.ndarray,
    cfg_max_qdot: np.ndarray,
    anomalies: list[str],
) -> None:
    """打印三层限位层级对比表，记录异常项。

    层级期望：
      - 位置：驱动级 max_pos >= 关节级 max_pos >= config q_upper（度）
      - 速度：驱动级 RPM >= 关节级 RPM；关节级(rad/s) >= config max_qdot

    Args:
        drive_pos: 驱动级位置限位（°）或 None。
        joint_pos: 关节级位置限位（°）或 None。
        drive_spd: 驱动级速度限位（RPM）或 None。
        joint_spd: 关节级速度限位（RPM）或 None。
        cfg_upper_deg: config 上限位（度，6 元素）。
        cfg_max_qdot: config 速度上限（rad/s，6 元素）。
        anomalies: 异常收集列表，发现层级违反时追加。
    """
    # --- 位置层级对比 ---
    print("\n  位置限位层级对比（驱动级 >= 关节级 >= config）")
    header = (
        f"  {'关节':>4s} | {'驱动(°)':>10s} | {'关节(°)':>10s} "
        f"| {'config(°)':>10s} | {'层级':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(6):
        d = drive_pos[i] if drive_pos else None
        j = joint_pos[i] if joint_pos else None
        c = float(cfg_upper_deg[i])
        ok = True
        if d is not None and j is not None and d + 1e-6 < j:
            ok = False
            anomalies.append(
                f"J{i + 1} 位置层级异常: 驱动级({d:.2f}°) < 关节级({j:.2f}°)"
            )
        if j is not None and j + 1e-6 < c:
            ok = False
            anomalies.append(
                f"J{i + 1} 位置层级异常: 关节级({j:.2f}°) < config({c:.2f}°)"
            )
        mark = "✅" if ok else "⚠️"
        print(f"  {'J' + str(i + 1):>4s} | {_s(d, 10)} | {_s(j, 10)} | {_s(c, 10)} | {mark:>6s}")

    # --- 速度层级对比 ---
    print("\n  速度限位层级对比（驱动级 >= 关节级；关节级 >= config）")
    header = (
        f"  {'关节':>4s} | {'驱动(rad/s)':>12s} | {'关节(rad/s)':>12s} "
        f"| {'config(rad/s)':>13s} | {'层级':>6s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(6):
        d_rpm = drive_spd[i] if drive_spd else None
        j_rpm = joint_spd[i] if joint_spd else None
        d_rads = d_rpm * RPM_TO_RAD_S if d_rpm is not None else None
        j_rads = j_rpm * RPM_TO_RAD_S if j_rpm is not None else None
        c_rads = float(cfg_max_qdot[i])
        ok = True
        if d_rads is not None and j_rads is not None and d_rads + 1e-6 < j_rads:
            ok = False
            anomalies.append(
                f"J{i + 1} 速度层级异常: 驱动级({d_rads:.3f}) < 关节级({j_rads:.3f})"
            )
        if j_rads is not None and j_rads + 1e-6 < c_rads:
            ok = False
            anomalies.append(
                f"J{i + 1} 速度层级异常: 关节级({j_rads:.3f}) < config({c_rads:.3f})"
            )
        mark = "✅" if ok else "⚠️"
        print(
            f"  {'J' + str(i + 1):>4s} | {_s(d_rads, 12, 3)} | {_s(j_rads, 12, 3)} "
            f"| {_s(c_rads, 13, 3)} | {mark:>6s}"
        )


def print_snapshot(arm: Any, config: Any) -> None:
    """打印一次完整关节限位 + 管理状态快照。

    读取 C1-C6（限位）、D1-D3（使能/错误/里程计），生成三层限位对比
    表与管理状态表，标注异常项。单次快照逻辑，循环模式可重复调用。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        config: RealRobotConfig 实例（提供 q_upper/max_qdot 基准）。
    """
    errors: list[str] = []
    anomalies: list[str] = []

    # 读取关节级限位（C1-C3）
    joint_pos = _call_list(arm, "rm_get_joint_max_pos", "关节级 pos", errors)
    joint_spd = _call_list(arm, "rm_get_joint_max_speed", "关节级 spd", errors)
    joint_acc = _call_list(arm, "rm_get_joint_max_acc", "关节级 acc", errors)

    # 读取驱动级限位（C4-C6）
    drive_pos = _call_list(arm, "rm_get_joint_drive_max_pos", "驱动级 pos", errors)
    drive_spd = _call_list(arm, "rm_get_joint_drive_max_speed", "驱动级 spd", errors)
    drive_acc = _call_list(arm, "rm_get_joint_drive_max_acc", "驱动级 acc", errors)

    # 读取管理状态（D1-D3）
    en_state = _call_list(arm, "rm_get_joint_en_state", "使能状态", errors)
    odom = _call_list(arm, "rm_get_joint_odom", "里程计", errors)
    ret_err, err_payload = _query(arm, "rm_get_joint_err_flag")

    # config 基准（度 / rad/s）
    cfg_upper_deg = np.degrees(config.q_upper)

    # --- 1. 关节级限位表 ---
    _print_limit_table("关节级限位（C1-C3）", joint_pos, joint_spd, joint_acc)

    # --- 2. 驱动级限位表 ---
    _print_limit_table("驱动级限位（C4-C6）", drive_pos, drive_spd, drive_acc)

    # --- 3. 三层限位对比 ---
    _print_comparison(
        drive_pos, joint_pos, drive_spd, joint_spd,
        cfg_upper_deg, config.max_qdot, anomalies,
    )

    # --- 4. 关节管理状态 ---
    print("\n  关节管理状态（D1-D3）")
    header = (
        f"  {'关节':>4s} | {'使能':>4s} | {'错误标志':>8s} | {'里程计(°)':>12s}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    # D2 错误标志：dict 结构，先全量打印原始 dict，再尝试提取关节列表
    err_list: list[float] | None = None
    if ret_err != 0 and isinstance(err_payload, dict) and "exception" in err_payload:
        errors.append(f"关节错误标志 异常: {err_payload['exception']}")
    else:
        print(f"\n  D2 rm_get_joint_err_flag 原始返回: {err_payload!r}")
        err_list = _extract_joint_list(err_payload)
        if err_list is None:
            print("  （未能从返回 dict 中提取 6 关节错误列表，按 N/A 显示）")

    max_odom_idx = -1
    max_odom_val = -1.0
    for i in range(6):
        en = en_state[i] if en_state else None
        od = odom[i] if odom else None
        err = err_list[i] if err_list else None

        if en is not None and en != 1:
            anomalies.append(f"J{i + 1} 未使能（使能状态={en}）")
        if err is not None and err != 0:
            anomalies.append(f"J{i + 1} 错误标志非零（={err}）")
        if od is not None and od > max_odom_val:
            max_odom_val = od
            max_odom_idx = i

        en_str = f"{int(en):>4d}" if en is not None else f"{'N/A':>4s}"
        err_str = f"{int(err):>8d}" if err is not None else f"{'N/A':>8s}"
        print(f"  {'J' + str(i + 1):>4s} | {en_str} | {err_str} | {_s(od, 12)}")

    # 使能/错误总体判定
    if en_state is not None:
        en_ok = all(v == 1 for v in en_state)
        print(f"\n  使能状态: {en_state} {'✅' if en_ok else '⚠️ 有关节未使能'}")
    if err_list is not None:
        err_ok = all(v == 0 for v in err_list)
        print(f"  错误标志: {err_list} {'✅' if err_ok else '⚠️ 存在错误'}")
    if odom is not None and max_odom_idx >= 0:
        print(f"  里程最大: J{max_odom_idx + 1} = {max_odom_val:.1f}°（磨损参考）")

    # --- 5. 异常汇总 ---
    print("\n  === 异常项 ===")
    all_issues = errors + anomalies
    if all_issues:
        for item in all_issues:
            print(f"    ⚠️ {item}")
    else:
        print("    无异常 ✅")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """主函数：连接 → 前置状态确认 → 快照（单次/持续）→ 安全断开。"""
    parser = argparse.ArgumentParser(
        description="RM-65B 关节级限位与管理状态查询（零风险，只读）"
    )
    add_config_arg(parser)
    parser.add_argument(
        "--hz",
        type=float,
        default=None,
        help="持续刷新频率 Hz；不指定则打印单次快照后退出",
    )
    args = parser.parse_args()

    print("=" * 64)
    print("11_joint_config_query — 关节级限位与管理状态查询")
    print("=" * 64)

    ri, config = load_and_connect(args.config)
    arm = ri._arm

    # 前置状态确认（通信/电源/运行模式）
    print("\n[1/2] 前置状态确认")
    try:
        ret, deg = arm.rm_get_joint_degree()
        if ret == 0:
            print(f"  ✅ 通信正常，当前关节角(°) = {[round(v, 1) for v in list(deg)[:6]]}")
        else:
            print(f"  ⚠️ rm_get_joint_degree ret={ret}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 读取关节角异常: {e}")

    try:
        ret, power = arm.rm_get_arm_power_state()
        power_str = "上电" if ret == 0 and power == 1 else f"ret={ret},power={power}"
        print(f"  电源状态: {power_str} {'✅' if ret == 0 and power == 1 else '⚠️'}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 电源状态异常: {e}")

    try:
        ret, mode = arm.rm_get_arm_run_mode()
        mode_str = "空闲" if ret == 0 and mode == 0 else f"ret={ret},mode={mode}"
        print(f"  运行模式: {mode_str} {'✅' if ret == 0 and mode == 0 else '⚠️'}")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ 运行模式异常: {e}")

    # 快照
    try:
        if args.hz is None:
            print("\n[2/2] 关节限位 + 管理状态快照")
            print_snapshot(arm, config)
        else:
            dt = 1.0 / args.hz
            print(f"\n[2/2] 持续刷新（{args.hz:.1f}Hz），按 Ctrl+C 停止")
            while True:
                print("\033[2J\033[H", end="")
                print("[11_joint_config_query.py] 持续模式  Ctrl+C 停止\n")
                print_snapshot(arm, config)
                import time

                time.sleep(dt)
    except KeyboardInterrupt:
        print("\n\n已停止")
    finally:
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
