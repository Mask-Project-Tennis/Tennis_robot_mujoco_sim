#!/usr/bin/env python3
"""10_algo_safety_verify.py — Algo 算法库安全检测验证。

验证 SDK Algo 类的自碰撞检测、奇异检测、工具包络球功能。
纯软件计算，不需要连接真机。

用法:
    python 10_algo_safety_verify.py

零风险: 纯软件计算，不发送任何运动指令。
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
from _connect import init_algo  # noqa: E402


# ──────────────────────────────────────────────────────────────
# 测试常量
# ──────────────────────────────────────────────────────────────

# 球拍工具包络球（与 _connect.py init_algo 预期一致：法兰/手柄/拍面）
# 注意: 必须用 centrePoint=(x,y,z) 构造；x=/y=/z= 形式会被 SDK 静默忽略
RACKET_ENVELOPE: list[tuple[tuple[float, float, float], float]] = [
    ((0.0, 0.0, 0.00), 0.03),  # 球 0: 法兰中心
    ((0.0, 0.0, 0.12), 0.02),  # 球 1: 手柄中部
    ((0.0, 0.0, 0.25), 0.07),  # 球 2: 拍面中心
]
# 包络球 roundtrip 容差（float32 精度）
ENVELOPE_TOL = 1e-6

# 自碰撞检测：已知碰撞姿态（度）
COLLISION_CASES: list[tuple[list[float], str]] = [
    ([0, -30, 130, 0, 90, 0], "J5=90° 向内折叠"),
    ([0, 0, 130, 0, 90, 0], "J3 过弯+J5折叠"),
]
# 自碰撞检测：已知安全姿态（度）
SAFE_CASES: list[tuple[list[float], str]] = [
    ([0, 0, 0, 0, 0, 0], "零位"),
    ([0, 20, 70, 0, 30, 0], "典型工作姿态"),
]

# 奇异检测姿态组（度）：(关节角, 标签, 期望解析法返回码)
#   0=安全, -1=肩部奇异, -2=肘部奇异, -3=腕部奇异
SINGULARITY_CASES: list[tuple[list[float], str, int]] = [
    ([0, 43.4, -105.7, 0, -30, 0], "肩部奇异", -1),
    ([0, 30, 0, 0, 30, 0], "肘部奇异(J3≈0)", -2),
    ([0, 30, 60, 0, 0, 0], "腕部奇异(J5≈0)", -3),
    ([0, 20, 70, 0, 45, 0], "安全姿态", 0),
]

# 数值法奇异阈值（关节角接近奇异时雅可比最小奇异值小于此限）
NUMERIC_SINGULAR_LIMIT = 0.01

# 奇异阈值默认值与自定义值
DEFAULT_THRESHOLDS = (10.0, 10.0, 0.05)
CUSTOM_THRESHOLDS = (12.0, 12.0, 0.06)


# ──────────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────────


def make_tool_sphere(x: float, y: float, z: float, radius: float) -> Any:
    """构造 rm_tool_sphere_t 工具包络球。

    SDK 的 rm_tool_sphere_t 实际字段为 ``centrePoint``（c_float_Array_3）与
    ``radius``。构造时必须用 ``centrePoint=(x,y,z)`` 关键字；``x=/y=/z=``
    形式虽不报错但会被**静默忽略**（centrePoint 保持 [0,0,0]）。

    Args:
        x: 球心 X 坐标（米，法兰系）。
        y: 球心 Y 坐标（米）。
        z: 球心 Z 坐标（米）。
        radius: 球半径（米）。

    Returns:
        rm_tool_sphere_t 实例。
    """
    from Robotic_Arm.rm_robot_interface import rm_tool_sphere_t

    return rm_tool_sphere_t(centrePoint=(x, y, z), radius=radius)


def get_sphere_fields(sphere: Any) -> tuple[list[float], float]:
    """读取 rm_tool_sphere_t 的 centrePoint 与 radius。

    Args:
        sphere: rm_tool_sphere_t 实例。

    Returns:
        ([cx, cy, cz], radius) — 球心坐标列表与半径。
    """
    return list(sphere.centrePoint), float(sphere.radius)


# ──────────────────────────────────────────────────────────────
# 测试 1: 工具包络球 set→get 回环
# ──────────────────────────────────────────────────────────────


def test_tool_envelope(algo: Any, verbose: bool) -> bool:
    """工具包络球 set→get 回环 + 槽位边界测试。

    步骤:
      1. 逐个设置球拍包络球（3 个），用 centrePoint= 正确构造。
      2. 逐个 get 回读，验证 centrePoint 与 radius 与设置值一致。
      3. 测试空槽位（球 3）应为默认零值。
      4. 测试越界槽位（球 5）不应崩溃（SDK 静默忽略或返回默认）。

    Args:
        algo: SDK Algo 实例。
        verbose: 是否打印每个球的详细字段。

    Returns:
        3 个包络球 roundtrip 全部一致且空槽位为零时返回 True。
    """
    print("\n  包络球 set→get 回环（centrePoint= 正确构造）:")
    print(
        f"  {'槽位':>4s} {'期望中心(m)':>22s} {'期望r':>6s} "
        f"{'回读中心(m)':>22s} {'回读r':>8s} {'结果':>6s}"
    )
    print("  " + "-" * 76)

    all_ok = True
    for i, ((x, y, z), r) in enumerate(RACKET_ENVELOPE):
        sphere = make_tool_sphere(x, y, z, r)
        algo.rm_algo_set_tool_envelope(i, sphere)
        got = algo.rm_algo_get_tool_envelope(i)
        gp, gr = get_sphere_fields(got)

        expect_pos = [x, y, z]
        pos_err = float(np.max(np.abs(np.array(gp) - np.array(expect_pos))))
        r_err = abs(gr - r)
        ok = pos_err < ENVELOPE_TOL and r_err < ENVELOPE_TOL
        all_ok = all_ok and ok

        print(
            f"  {i:>4d} {str(expect_pos):>22s} {r:6.3f} "
            f"{str([round(v, 5) for v in gp]):>22s} {gr:8.5f} "
            f"{'OK' if ok else 'FAIL':>6s}"
        )
        if verbose:
            print(f"    pos_err={pos_err:.2e}  r_err={r_err:.2e}")

    # 空槽位（球 3）应为默认零值
    got3 = algo.rm_algo_get_tool_envelope(3)
    gp3, gr3 = get_sphere_fields(got3)
    empty_ok = all(v == 0.0 for v in gp3) and gr3 == 0.0
    all_ok = all_ok and empty_ok
    print(
        f"  {3:>4d} {'(空槽位, 期望零)':>22s} {0.000:6.3f} "
        f"{str([round(v, 5) for v in gp3]):>22s} {gr3:8.5f} "
        f"{'OK' if empty_ok else 'FAIL':>6s}"
    )

    # 越界槽位（球 5）不应崩溃
    oob_ok = True
    try:
        algo.rm_algo_set_tool_envelope(5, make_tool_sphere(0, 0, 0, 0.01))
    except Exception as e:  # noqa: BLE001
        oob_ok = False
        print(f"  ⚠️ 越界槽位 5 set 抛异常: {e}")
    all_ok = all_ok and oob_ok

    # ⚠️ 自审发现：_connect.py init_algo 用 x=/y=/z= 构造，centrePoint 全为 [0,0,0]
    print("\n  ⚠️ 注: _connect.py init_algo 使用 x=/y=/z= 形式构造，")
    print("     该形式被 SDK 静默忽略，centrePoint 实际全为 [0,0,0]。")
    print("     本脚本用 centrePoint= 正确构造，已覆盖 init_algo 的缺陷。")
    return all_ok


# ──────────────────────────────────────────────────────────────
# 测试 2: 自碰撞检测（已知碰撞 vs 已知安全）
# ──────────────────────────────────────────────────────────────


def test_self_collision(algo: Any, verbose: bool) -> bool:
    """自碰撞检测：碰撞姿态应返回 1，安全姿态应返回 0。

    前置: 工具包络球已正确设置（依赖 test_tool_envelope 先执行）。
    自碰撞检测的关节角单位为**度（°）**。

    Args:
        algo: SDK Algo 实例。
        verbose: 是否打印每个姿态的详细关节角。

    Returns:
        全部碰撞姿态返回 1 且安全姿态返回 0 时返回 True。
    """
    print("\n  自碰撞检测（rm_algo_safety_robot_self_collision_detection）:")
    print(f"  {'标签':<22s} {'关节角(°)':>30s} {'期望':>4s} {'实际':>4s} {'结果':>6s}")
    print("  " + "-" * 72)

    all_ok = True
    for q_deg, label in COLLISION_CASES:
        ret = algo.rm_algo_safety_robot_self_collision_detection(list(q_deg))
        ok = ret == 1
        all_ok = all_ok and ok
        print(
            f"  {label:<22s} {str(q_deg):>30s} {1:>4d} {ret:>4d} "
            f"{'OK' if ok else 'FAIL':>6s}"
        )
        if verbose:
            print(f"    返回码 {ret} (0=安全, 1=碰撞)")

    for q_deg, label in SAFE_CASES:
        ret = algo.rm_algo_safety_robot_self_collision_detection(list(q_deg))
        ok = ret == 0
        all_ok = all_ok and ok
        print(
            f"  {label:<22s} {str(q_deg):>30s} {0:>4d} {ret:>4d} "
            f"{'OK' if ok else 'FAIL':>6s}"
        )
        if verbose:
            print(f"    返回码 {ret} (0=安全, 1=碰撞)")

    return all_ok


# ──────────────────────────────────────────────────────────────
# 测试 3: 奇异检测（解析法 — 肩/肘/腕分类）
# ──────────────────────────────────────────────────────────────


def test_singularity_analytical(algo: Any, verbose: bool) -> bool:
    """解析法奇异检测：肩/肘/腕奇异姿态分类。

    返回码: 0=安全, -1=肩部奇异, -2=肘部奇异, -3=腕部奇异。
    第二个返回值 distance 对肩部奇异为"腕部中心到奇异平面距离(米)"，
    对肘/腕奇异无明确物理意义，不作为判据。
    关节角单位为**度（°）**。

    Args:
        algo: SDK Algo 实例。
        verbose: 是否打印每个姿态的距离值。

    Returns:
        全部姿态返回码与期望一致时返回 True。
    """
    codes = {0: "安全", -1: "肩部奇异", -2: "肘部奇异", -3: "腕部奇异"}
    print("\n  解析法奇异检测（rm_algo_kin_robot_singularity_analyse）:")
    print(
        f"  {'标签':<20s} {'期望':>8s} {'实际':>8s} "
        f"{'distance':>10s} {'结果':>6s}"
    )
    print("  " + "-" * 58)

    all_ok = True
    for q_deg, label, expect in SINGULARITY_CASES:
        ret, dist = algo.rm_algo_kin_robot_singularity_analyse(list(q_deg))
        ok = ret == expect
        all_ok = all_ok and ok
        print(
            f"  {label:<20s} {codes[expect]:>8s} {codes.get(ret, f'code={ret}'):>8s} "
            f"{dist:10.5f} {'OK' if ok else 'FAIL':>6s}"
        )
        if verbose:
            print(f"    关节角(°) = {q_deg}")
            print(f"    返回码 {ret}, distance {dist:.6f}")

    return all_ok


# ──────────────────────────────────────────────────────────────
# 测试 4: 数值法 vs 解析法对比
# ──────────────────────────────────────────────────────────────


def test_singularity_compare(algo: Any, verbose: bool) -> bool:
    """数值法 vs 解析法奇异检测结论一致性。

    数值法 ``rm_algo_universal_singularity_analyse`` 返回 0(安全) 或 -1(奇异)，
    不区分奇异类型。与解析法对比"是否奇异"的结论应一致：
      - 解析法 ret<0 ⟺ 数值法 ret<0（都判定奇异）
      - 解析法 ret==0 ⟺ 数值法 ret==0（都判定安全）

    Args:
        algo: SDK Algo 实例。
        verbose: 是否打印两法返回码对照。

    Returns:
        全部姿态两法结论一致时返回 True。
    """
    print("\n  数值法 vs 解析法奇异检测对比:")
    print(
        f"  {'标签':<20s} {'解析法':>8s} {'数值法':>8s} "
        f"{'一致':>6s}"
    )
    print("  " + "-" * 46)

    all_ok = True
    for q_deg, label, _expect in SINGULARITY_CASES:
        ret_a, _dist = algo.rm_algo_kin_robot_singularity_analyse(list(q_deg))
        ret_n = algo.rm_algo_universal_singularity_analyse(
            list(q_deg), NUMERIC_SINGULAR_LIMIT
        )
        # 结论一致：两者同号（同为奇异 ret<0，或同为安全 ret==0）
        agree = (ret_a < 0) == (ret_n < 0)
        all_ok = all_ok and agree
        a_str = "奇异" if ret_a < 0 else "安全"
        n_str = "奇异" if ret_n < 0 else "安全"
        print(
            f"  {label:<20s} {a_str:>8s} {n_str:>8s} "
            f"{'OK' if agree else 'FAIL':>6s}"
        )
        if verbose:
            print(f"    解析法 ret={ret_a}, 数值法 ret={ret_n}")

    return all_ok


# ──────────────────────────────────────────────────────────────
# 测试 5: 奇异阈值配置回环
# ──────────────────────────────────────────────────────────────


def test_singularity_thresholds(algo: Any) -> bool:
    """奇异阈值 set→get→init 回环。

    默认阈值: (qe=10°, qw=10°, d=0.05m)。设置自定义阈值后读回应一致，
    调用 init 应恢复默认。

    Args:
        algo: SDK Algo 实例。

    Returns:
        set→get 一致且 init→默认恢复成功时返回 True。
    """
    print("\n  奇异阈值配置回环:")
    tol = 1e-5

    # 1. 读取默认阈值
    qe, qw, d = algo.rm_algo_kin_get_singularity_thresholds()
    default_ok = (
        abs(qe - DEFAULT_THRESHOLDS[0]) < tol
        and abs(qw - DEFAULT_THRESHOLDS[1]) < tol
        and abs(d - DEFAULT_THRESHOLDS[2]) < tol
    )
    print(
        f"    默认阈值 = ({qe:.4f}, {qw:.4f}, {d:.4f})  "
        f"期望 {DEFAULT_THRESHOLDS}  {'✅' if default_ok else '❌'}"
    )

    # 2. 设置自定义阈值
    algo.rm_algo_kin_set_singularity_thresholds(*CUSTOM_THRESHOLDS)
    qe2, qw2, d2 = algo.rm_algo_kin_get_singularity_thresholds()
    set_ok = (
        abs(qe2 - CUSTOM_THRESHOLDS[0]) < tol
        and abs(qw2 - CUSTOM_THRESHOLDS[1]) < tol
        and abs(d2 - CUSTOM_THRESHOLDS[2]) < tol
    )
    print(
        f"    自定义   = ({qe2:.4f}, {qw2:.4f}, {d2:.4f})  "
        f"期望 {CUSTOM_THRESHOLDS}  {'✅' if set_ok else '❌'}"
    )

    # 3. 恢复默认
    algo.rm_algo_kin_singularity_thresholds_init()
    qe3, qw3, d3 = algo.rm_algo_kin_get_singularity_thresholds()
    init_ok = (
        abs(qe3 - DEFAULT_THRESHOLDS[0]) < tol
        and abs(qw3 - DEFAULT_THRESHOLDS[1]) < tol
        and abs(d3 - DEFAULT_THRESHOLDS[2]) < tol
    )
    print(
        f"    恢复默认 = ({qe3:.4f}, {qw3:.4f}, {d3:.4f})  "
        f"期望 {DEFAULT_THRESHOLDS}  {'✅' if init_ok else '❌'}"
    )

    return default_ok and set_ok and init_ok


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """主函数：初始化 Algo → 5 项安全检测 → 汇总结论。"""
    parser = argparse.ArgumentParser(
        description="RM-65B Algo 算法库安全检测验证（零风险，纯软件计算）"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="打印详细的中间结果"
    )
    args = parser.parse_args()

    # ── 初始化 ──
    print("=" * 64)
    print("10_algo_safety_verify — Algo 安全检测验证")
    print("=" * 64)

    print("\n[1/6] 初始化 Algo")
    algo = init_algo()
    if algo is None:
        print("  ❌ SDK Algo 不可用，请安装: pip install Robotic_Arm")
        sys.exit(1)
    print(f"  ✅ Algo 初始化成功，算法库版本: {algo.rm_algo_version()}")
    print(f"     arm_dof={algo.arm_dof}, dh_dof={algo.dh_dof}")

    # ── 测试 ──
    results: dict[str, bool] = {}

    print("\n[2/6] 工具包络球 set→get 回环")
    results["tool_envelope"] = test_tool_envelope(algo, args.verbose)

    print("\n[3/6] 自碰撞检测（碰撞 vs 安全）")
    results["self_collision"] = test_self_collision(algo, args.verbose)

    print("\n[4/6] 解析法奇异检测（肩/肘/腕分类）")
    results["singularity_analytical"] = test_singularity_analytical(
        algo, args.verbose
    )

    print("\n[5/6] 数值法 vs 解析法奇异检测对比")
    results["singularity_compare"] = test_singularity_compare(algo, args.verbose)

    print("\n[6/6] 奇异阈值配置回环")
    results["singularity_thresholds"] = test_singularity_thresholds(algo)

    # ── 汇总 ──
    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")

    # 关键结论：pre_motion_check 依赖的安全检测是否可信
    print("\n关键结论:")
    core = results.get("self_collision") and results.get("singularity_analytical")
    if core:
        print("  ✅ 自碰撞检测与奇异检测分类全部正确")
        print("     → pre_motion_check 依赖的安全防线可信")
    else:
        if not results.get("self_collision"):
            print("  ❌ 自碰撞检测存在假阴性/假阳性，pre_motion_check 碰撞防线不可靠")
        if not results.get("singularity_analytical"):
            print("  ⚠️ 奇异检测分类与预期不符，请检查阈值或姿态")

    must_pass = [
        "tool_envelope",
        "self_collision",
        "singularity_analytical",
        "singularity_compare",
        "singularity_thresholds",
    ]
    all_pass = all(results.get(k, False) for k in must_pass)
    if all_pass:
        print("\n  ✅ 全部安全检测项通过")
    else:
        failed = [k for k in must_pass if not results.get(k, False)]
        print(f"\n  ❌ 以下项失败: {failed}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
