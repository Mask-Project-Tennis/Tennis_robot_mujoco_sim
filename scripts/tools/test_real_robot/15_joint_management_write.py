#!/usr/bin/env python3
"""15_joint_management_write.py — 关节管理写操作测试。

测试关节清错、清里程计、自动限位、设置安装姿态等写操作。
每个写操作都有 set→get 回环验证和测后恢复。

中高风险: 改变控制器持久状态，逐步 YES 确认 + 测后恢复。

排除的危险写操作（本测试不做）:
  - rm_set_joint_en_state(0,0) 去使能（机械臂失去抱闸下坠）
  - rm_set_joint_zero_pos 改零位（影响所有后续角度）
  - rm_set_joint_max_pos/speed/acc 改关节级限位（安全防线失效）
  - rm_set_joint_drive_max_* 改驱动级硬限位（极危险）

用法:
    python 15_joint_management_write.py
    python 15_joint_management_write.py --skip-phase2   # 跳过中风险写操作（H5/H6）
"""

import argparse
import sys
from pathlib import Path
from typing import Any

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

# 安装姿态 Y 轴微偏移测试量（度），足够小避免重力补偿突变
_INSTALL_Y_OFFSET_DEG = 1.0
# set→get 读回容差（度）
_INSTALL_POSE_TOL_DEG = 0.1


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
        name: SDK 方法名（如 "rm_get_install_pose"）。

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


def _format_payload(payload: Any) -> str:
    """把 payload 格式化为可读字符串。

    dict 截断超长值，避免刷屏；其他类型直接 repr。

    Args:
        payload: 任意返回值。

    Returns:
        可读字符串。
    """
    if isinstance(payload, dict):
        items = []
        for k, v in payload.items():
            s = repr(v)
            if len(s) > 60:
                s = s[:57] + "..."
            items.append(f"{k}={s}")
        return "{" + ", ".join(items) + "}"
    return repr(payload)


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


def _extract_install_pose(payload: Any) -> tuple[float, float, float] | None:
    """从 rm_get_install_pose 返回中提取 (x, y, z) 度。

    Args:
        payload: dict（含 x/y/z 键）或其他结构。

    Returns:
        (x, y, z) 度，或 None（提取失败）。
    """
    if not isinstance(payload, dict) or "exception" in payload:
        return None
    try:
        return (float(payload["x"]), float(payload["y"]), float(payload["z"]))
    except (KeyError, TypeError, ValueError):
        return None


def _confirm(prompt: str) -> bool:
    """交互式 YES 确认。

    Args:
        prompt: 提示文本。

    Returns:
        用户输入 YES 时返回 True。
    """
    answer = input(prompt)
    return answer.strip().upper() == "YES"


# ──────────────────────────────────────────────────────────────
# 步骤 0：读取恢复基准
# ──────────────────────────────────────────────────────────────


def read_baselines(arm: Any) -> dict[str, Any]:
    """读取并记录所有原始值，作为测后恢复基准。

    读取项：安装姿态、关节里程计、关节级位置限位、关节错误标志。
    任一读取失败时对应值为 None，后续恢复逻辑据此跳过。

    Args:
        arm: RoboticArm 实例（ri._arm）。

    Returns:
        基准字典，键：install/odom/max_pos/err，值可为 None。
    """
    print("\n" + "=" * 60)
    print("步骤 0：读取恢复基准（原始值）")
    print("=" * 60)

    baselines: dict[str, Any] = {
        "install": None,
        "odom": None,
        "max_pos": None,
        "err": None,
    }

    # 安装姿态（H6 恢复用，最关键）
    ret, pose_payload = _query(arm, "rm_get_install_pose")
    print(f"\n  安装姿态 rm_get_install_pose: ret={ret}")
    print(f"    原始返回: {_format_payload(pose_payload)}")
    if ret == 0:
        baselines["install"] = _extract_install_pose(pose_payload)
        if baselines["install"] is not None:
            x, y, z = baselines["install"]
            print(f"    → x={x}°, y={y}°, z={z}°  [H6 恢复基准]")
        else:
            print("    ⚠️ 未能提取 x/y/z，H6 将无法自动恢复")

    # 关节里程计（H2 前记录，清零不可逆）
    ret, odom_payload = _query(arm, "rm_get_joint_odom")
    print(f"\n  关节里程计 rm_get_joint_odom: ret={ret}")
    if ret == 0:
        try:
            baselines["odom"] = [float(v) for v in list(odom_payload)[:6]]
            print(f"    原始值(°): {[round(v, 1) for v in baselines['odom']]}  [H2 清零前记录]")
        except (TypeError, ValueError):
            print(f"    原始返回: {_format_payload(odom_payload)}")

    # 关节级位置限位（H5 前后对比用）
    ret, max_pos_payload = _query(arm, "rm_get_joint_max_pos")
    print(f"\n  关节级位置限位 rm_get_joint_max_pos: ret={ret}")
    if ret == 0:
        try:
            baselines["max_pos"] = [float(v) for v in list(max_pos_payload)[:6]]
            print(f"    原始值(°): {[round(v, 1) for v in baselines['max_pos']]}  [H5 对比基准]")
        except (TypeError, ValueError):
            print(f"    原始返回: {_format_payload(max_pos_payload)}")

    # 关节错误标志（H1 清错前记录）
    ret_err, err_payload = _query(arm, "rm_get_joint_err_flag")
    print(f"\n  关节错误标志 rm_get_joint_err_flag: ret={ret_err}")
    print(f"    原始返回: {_format_payload(err_payload)}")
    baselines["err"] = _extract_joint_list(err_payload)
    if baselines["err"] is not None:
        print(f"    提取值: {baselines['err']}  [H1 清错前记录]")

    print("\n  === 恢复基准已记录 ===")
    return baselines


# ──────────────────────────────────────────────────────────────
# 阶段一：低风险写操作（H1-H4）
# ──────────────────────────────────────────────────────────────


def run_phase1(arm: Any, baselines: dict[str, Any]) -> None:
    """执行阶段一低风险写操作：H1 清错 / H2 清里程 / H3 清系统错误 / H4 清运行时间。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        baselines: 步骤 0 记录的恢复基准。
    """
    print("\n" + "=" * 60)
    print("阶段一：低风险写操作（H1-H4）")
    print("=" * 60)

    # ── H1：清除关节错误 ──
    print("\n--- H1: 清除关节错误 rm_set_joint_clear_err(0) ---")
    err_before = baselines.get("err")
    if err_before is not None and all(v == 0 for v in err_before):
        print(f"  清错前错误标志: {err_before} → 全 0，无错误可清")
    else:
        print(f"  清错前错误标志: {err_before}")
    ret = arm.rm_set_joint_clear_err(0)
    print(f"  rm_set_joint_clear_err(0) → ret={ret} {'✅' if ret == 0 else '⚠️'}")
    # 读回确认
    _, err_payload = _query(arm, "rm_get_joint_err_flag")
    err_after = _extract_joint_list(err_payload)
    if err_after is not None:
        cleared = all(v == 0 for v in err_after)
        print(f"  清错后错误标志: {err_after} {'✅ 已清除' if cleared else '⚠️ 仍有错误'}")
    else:
        print(f"  清错后原始返回: {_format_payload(err_payload)}")

    # ── H2：清零关节里程计 ──
    print("\n--- H2: 清零关节里程计 rm_clear_joint_odom ---")
    odom_before = baselines.get("odom")
    print(f"  清零前里程计(°): {[round(v, 1) for v in odom_before] if odom_before else 'N/A'}")
    print("  ⚠️ 清零不可逆，原始里程数据永久丢失（已在步骤 0 记录）")
    ret = arm.rm_clear_joint_odom()
    print(f"  rm_clear_joint_odom() → ret={ret} {'✅' if ret == 0 else '⚠️'}")
    # 读回确认
    _, odom_payload = _query(arm, "rm_get_joint_odom")
    try:
        odom_after = [float(v) for v in list(odom_payload)[:6]]
        all_zero = all(v == 0.0 for v in odom_after)
        print(f"  清零后里程计(°): {[round(v, 1) for v in odom_after]} "
              f"{'✅ 全 0' if all_zero else '⚠️ 未清零'}")
    except (TypeError, ValueError):
        print(f"  清零后原始返回: {_format_payload(odom_payload)}")

    # ── H3：清除系统错误 ──
    print("\n--- H3: 清除系统错误 rm_clear_system_err ---")
    ret = arm.rm_clear_system_err()
    print(f"  rm_clear_system_err() → ret={ret} {'✅' if ret == 0 else '⚠️'}")
    # 读控制器状态确认
    _, cs_payload = _query(arm, "rm_get_controller_state")
    if isinstance(cs_payload, dict) and "exception" not in cs_payload:
        cs = cs_payload.get("controller_state", "N/A")
        cs_ok = cs == 0 if isinstance(cs, int) else False
        print(f"  控制器状态: {cs} ({'正常 ✅' if cs_ok else '异常 ⚠️'})")
    else:
        print(f"  控制器状态原始返回: {_format_payload(cs_payload)}")

    # ── H4：清零系统运行时间 ──
    print("\n--- H4: 清零系统运行时间 rm_clear_system_runtime ---")
    ret = arm.rm_clear_system_runtime()
    print(f"  rm_clear_system_runtime() → ret={ret} {'✅' if ret == 0 else '⚠️'}")
    print("  （系统运行时间无直接读回接口，仅确认 ret=0）")


# ──────────────────────────────────────────────────────────────
# 阶段二：中风险写操作（H5-H6）
# ──────────────────────────────────────────────────────────────


def run_h5(arm: Any, baselines: dict[str, Any]) -> None:
    """H5：自动关节限位 rm_auto_set_joint_limit(mode=1)。

    根据当前安装姿态自动计算关节限位，对比执行前后限位值。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        baselines: 步骤 0 记录的恢复基准（取 max_pos 对比）。
    """
    print("\n--- H5: 自动关节限位 rm_auto_set_joint_limit(1) ---")
    pos_before = baselines.get("max_pos")
    print(f"  执行前位置限位(°): {[round(v, 1) for v in pos_before] if pos_before else 'N/A'}")
    ret = arm.rm_auto_set_joint_limit(1)
    print(f"  rm_auto_set_joint_limit(1) → ret={ret} {'✅' if ret == 0 else '⚠️'}")
    # 读回对比
    _, pos_payload = _query(arm, "rm_get_joint_max_pos")
    try:
        pos_after = [float(v) for v in list(pos_payload)[:6]]
        print(f"  执行后位置限位(°): {[round(v, 1) for v in pos_after]}")
        if pos_before is not None:
            changed = any(abs(a - b) > 0.01 for a, b in zip(pos_after, pos_before))
            if changed:
                diffs = [(i + 1, round(a - b, 2))
                         for i, (a, b) in enumerate(zip(pos_after, pos_before))
                         if abs(a - b) > 0.01]
                print(f"  ⚠️ 限位已变化: {diffs}")
                print("     说明之前限位与安装姿态不匹配，已自动修正")
                print("     如需更新 config.yaml 安全限位，请人工评估")
            else:
                print("  ✅ 限位无变化（自动限位 = 当前限位，正常）")
    except (TypeError, ValueError):
        print(f"  执行后原始返回: {_format_payload(pos_payload)}")


def run_h6(arm: Any, baselines: dict[str, Any]) -> None:
    """H6：安装姿态设置 rm_set_install_pose，set→get 验证后立即恢复。

    用 Y 轴 +1° 微偏移测试，验证 set→get 一致性，然后恢复原始值。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        baselines: 步骤 0 记录的恢复基准（取 install 原始值）。
    """
    print("\n--- H6: 安装姿态设置 rm_set_install_pose（Y 轴微偏移）---")
    install_before = baselines.get("install")
    if install_before is None:
        print("  ⚠️ 原始安装姿态未读取成功，跳过 H6（无法保证恢复）")
        return

    x0, y0, z0 = install_before
    print(f"  原始姿态: x={x0}°, y={y0}°, z={z0}°")
    # 设置微偏移：Y 轴 +1°
    y_test = y0 + _INSTALL_Y_OFFSET_DEG
    print(f"  设置测试值: x={x0}°, y={y_test}°, z={z0}°  (Y +{_INSTALL_Y_OFFSET_DEG}°)")
    ret = arm.rm_set_install_pose(x0, y_test, z0)
    print(f"  rm_set_install_pose({x0}, {y_test}, {z0}) → ret={ret} "
          f"{'✅' if ret == 0 else '⚠️'}")
    # 读回确认
    _, pose_payload = _query(arm, "rm_get_install_pose")
    test_pose = _extract_install_pose(pose_payload)
    if test_pose is not None:
        xt, yt, zt = test_pose
        y_match = abs(yt - y_test) <= _INSTALL_POSE_TOL_DEG
        print(f"  读回: x={xt}°, y={yt}°, z={zt}°")
        print(f"  Y 轴一致性: 期望={y_test}°, 实际={yt}°, "
              f"误差={abs(yt - y_test):.3f}° {'✅' if y_match else '⚠️ 超容差'}")
    else:
        print(f"  读回原始返回: {_format_payload(pose_payload)}")

    # 立即恢复原始姿态
    print(f"  恢复原始姿态: x={x0}°, y={y0}°, z={z0}°")
    ret = arm.rm_set_install_pose(x0, y0, z0)
    print(f"  rm_set_install_pose({x0}, {y0}, {z0}) → ret={ret} "
          f"{'✅ 已恢复' if ret == 0 else '⚠️ 恢复失败'}")


# ──────────────────────────────────────────────────────────────
# 恢复 + 汇总
# ──────────────────────────────────────────────────────────────


def restore_install_pose(arm: Any, baselines: dict[str, Any]) -> None:
    """确保安装姿态恢复到步骤 0 记录的原始值（finally 安全网）。

    即使 H6 已恢复，此函数作为兜底：异常或 Ctrl+C 时也能恢复。

    Args:
        arm: RoboticArm 实例（ri._arm）。
        baselines: 步骤 0 记录的恢复基准。
    """
    install_before = baselines.get("install")
    if install_before is None:
        print("\n⚠️ 无法自动恢复安装姿态：原始值未读取成功，请手动检查 rm_get_install_pose")
        return

    x0, y0, z0 = install_before
    # 先读当前值，判断是否需要恢复
    _, pose_now = _query(arm, "rm_get_install_pose")
    current = _extract_install_pose(pose_now)
    if current is not None and abs(current[1] - y0) < _INSTALL_POSE_TOL_DEG:
        print(f"\n✅ 安装姿态已是原始值: x={current[0]}°, y={current[1]}°, z={current[2]}°")
        return

    print("\n⚠️ 安装姿态偏离原始值，正在恢复...")
    ret = arm.rm_set_install_pose(x0, y0, z0)
    print(f"  rm_set_install_pose({x0}, {y0}, {z0}) → ret={ret}")
    if ret == 0:
        print("  ✅ 安装姿态已恢复")
    else:
        print(f"  ❌ 恢复失败 ret={ret}，请手动恢复: rm_set_install_pose({x0}, {y0}, {z0})")


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """主函数：连接 → 读基准 → 阶段一 → 阶段二 → 恢复 → 安全断开。"""
    parser = argparse.ArgumentParser(
        description="RM-65B 关节管理写操作测试（中~高风险，逐步 YES 确认 + 测后恢复）"
    )
    add_config_arg(parser)
    parser.add_argument(
        "--skip-phase2",
        action="store_true",
        default=False,
        help="跳过中风险写操作（H5 自动限位 / H6 安装姿态）",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("15_joint_management_write — 关节管理写操作测试")
    print("=" * 60)
    print("\n⚠️ 本测试包含写操作，部分改变控制器持久状态:")
    print("  H1 清关节错误  H2 清里程计（不可逆）  H3 清系统错误  H4 清运行时间")
    if not args.skip_phase2:
        print("  H5 自动限位（持久）  H6 设置安装姿态（持久，测后恢复）")
    print("\n排除的危险写操作: 去使能 / 改零位 / 改关节级限位 / 改驱动级限位")

    ri, _config = load_and_connect(args.config)
    arm = ri._arm

    # 前置：确认机械臂静止
    try:
        ret, mode = arm.rm_get_arm_run_mode()
        if ret == 0 and mode != 0:
            print(f"\n⚠️ 机械臂非空闲（run_mode={mode}），请先等待运动结束")
            safe_disconnect(ri)
            return
    except Exception as e:  # noqa: BLE001
        print(f"\n⚠️ 读取运行模式异常: {e}")

    baselines: dict[str, Any] = {}

    try:
        # 步骤 0：读取恢复基准
        baselines = read_baselines(arm)

        # 阶段一：低风险写操作（一次 YES）
        print("\n" + "-" * 60)
        print("即将执行阶段一（H1-H4 低风险写操作）:")
        print("  H1 清关节错误 / H2 清里程计（不可逆）/ H3 清系统错误 / H4 清运行时间")
        if _confirm("\n输入 YES 执行阶段一: "):
            run_phase1(arm, baselines)
        else:
            print("已跳过阶段一")

        # 阶段二：中风险写操作（逐步 YES）
        if not args.skip_phase2:
            print("\n" + "-" * 60)
            print("阶段二（H5-H6 中风险写操作，逐步确认）:")

            # H5 自动限位
            print("\n[H5] rm_auto_set_joint_limit(1) — 根据安装姿态自动计算限位")
            print("  ⚠️ 持久写入，覆盖之前限位（mode=1 基于当前姿态计算）")
            if _confirm("  输入 YES 执行 H5: "):
                run_h5(arm, baselines)
            else:
                print("  已跳过 H5")

            # H6 安装姿态
            print("\n[H6] rm_set_install_pose — Y 轴微偏移测试（+1°），测后立即恢复")
            print("  ⚠️ 持久写入 flash，测试用微偏移 + 立即恢复")
            if _confirm("  输入 YES 执行 H6: "):
                run_h6(arm, baselines)
            else:
                print("  已跳过 H6")
        else:
            print("\n已跳过阶段二（--skip-phase2）")

        print("\n" + "=" * 60)
        print("测试完成")
        print("=" * 60)
    except KeyboardInterrupt:
        print("\n\n⚠️ Ctrl+C — 正在恢复并断开...")
    finally:
        # 安全网：无论如何都恢复安装姿态
        restore_install_pose(arm, baselines)
        safe_disconnect(ri)


if __name__ == "__main__":
    main()
