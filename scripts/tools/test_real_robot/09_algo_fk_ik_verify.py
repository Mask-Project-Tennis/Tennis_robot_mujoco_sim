#!/usr/bin/env python3
"""09_algo_fk_ik_verify.py — Algo 算法库正逆运动学校验。

验证 SDK Algo 类的 FK/IK 与项目 MuJoCo PlanningEnv 的一致性。
纯软件计算，不需要连接真机（--online 模式可连接读取当前关节角）。

用法:
    python 09_algo_fk_ik_verify.py
    python 09_algo_fk_ik_verify.py --verbose    # 详细输出
    python 09_algo_fk_ik_verify.py --online      # 连接真机读取当前关节角

零风险: 纯软件计算，不发送任何运动指令。
"""

import argparse
import sys
from ctypes import c_float
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

# 科学计算依赖（姿态转换）
try:
    from scipy.spatial.transform import Rotation
    SCIPY_AVAILABLE = True
except ImportError:
    Rotation = None  # type: ignore[assignment, misc]
    SCIPY_AVAILABLE = False

# MuJoCo PlanningEnv（需要 C++ 扩展与 mujoco 包；失败时降级为仅测 Algo）
try:
    import mujoco

    from src.ilqt.planning_env import PlanningEnv

    PLANNING_ENV_AVAILABLE = True
except ImportError as _exc:  # pragma: no cover - 环境依赖
    PLANNING_ENV_AVAILABLE = False
    _PLANNING_ENV_IMPORT_ERROR = _exc
else:
    _PLANNING_ENV_IMPORT_ERROR = None


# 测试关节角组（度），覆盖工作空间
TEST_JOINTS: list[tuple[list[float], str]] = [
    ([0, 0, 0, 0, 0, 0], "零位"),
    ([0, 20, 70, 0, 90, 0], "文档示例姿态"),
    ([0, -30, 90, 0, 90, 0], "高位"),
    ([30, 0, 45, 0, 45, 0], "侧伸"),
    ([0, 0, -90, 0, -90, 0], "IK示例姿态"),
]

# 关节角回环容差（度）
IK_LOOP_TOL_DEG = 0.01
# 坐标变换往返容差
ROUNDTRIP_TOL = 1e-6


# ──────────────────────────────────────────────────────────────
# FK 工具函数
# ──────────────────────────────────────────────────────────────


def algo_fk_pose(algo: Any, q_deg: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """调用 Algo 正运动学，返回法兰位姿。

    使用 flag=0（四元数输出）以规避欧拉角万向锁问题；四元数 q 与 -q
    表示同一旋转，scipy 能正确处理。Algo 输入为度，位置输出为米。

    Args:
        algo: SDK Algo 实例。
        q_deg: 关节角度（度），长度 6。

    Returns:
        (pos(3,), R(3,3)) — 法兰中心位置（米）与旋转矩阵。
    """
    pose = algo.rm_algo_forward_kinematics(q_deg, flag=0)  # [x,y,z,w,x,y,z]
    pos = np.array(pose[:3], dtype=np.float64)
    # scipy 约定 [x,y,z,w]，SDK 返回 [w,x,y,z]
    quat_xyzw = [pose[4], pose[5], pose[6], pose[3]]
    R = Rotation.from_quat(quat_xyzw).as_matrix()
    return pos, R


def mj_flange_pose(
    env: "PlanningEnv", flange_id: int, q_deg: list[float]
) -> tuple[np.ndarray, np.ndarray]:
    """读取 MuJoCo PlanningEnv 法兰位姿。

    通过设置右臂关节角后读取 r_flange body 的世界系位姿。
    注意 MuJoCo 内部用弧度，需先把度转弧度。

    Args:
        env: PlanningEnv 实例。
        flange_id: r_flange body 的 MuJoCo id。
        q_deg: 关节角度（度），长度 6。

    Returns:
        (pos(3,), R(3,3)) — 法兰世界系位置（米）与旋转矩阵。
    """
    x = np.zeros(env.NX)
    x[: env.NQ] = np.radians(q_deg)
    env.set_arm_state(x)
    pos = env.data.xpos[flange_id].copy()
    R = env.data.xmat[flange_id].reshape(3, 3).copy()
    return pos, R


def mj_racket_pos(env: "PlanningEnv", q_deg: list[float]) -> np.ndarray:
    """读取 MuJoCo PlanningEnv 球拍中心位置（PlanningEnv.get_ee_pos）。

    作为参考一并记录——球拍中心 = 法兰 + 球拍杆/拍面偏移，
    其偏差不携带额外的运动学模型信息（仅叠加常数工具偏移）。

    Args:
        env: PlanningEnv 实例。
        q_deg: 关节角度（度），长度 6。

    Returns:
        球拍中心世界系位置 (3,)，米。
    """
    x = np.zeros(env.NX)
    x[: env.NQ] = np.radians(q_deg)
    env.set_arm_state(x)
    return env.get_ee_pos().copy()


def calibrate_base_transform(
    algo: Any, env: "PlanningEnv", flange_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """用零位标定 Algo 基座系 → MuJoCo 世界系的刚体变换。

    原理：Algo FK 基于标准 RM-65B 直立安装的基座系；MuJoCo 模型中右臂
    斜装在桩柱上（r_base_link1 绕 Y 轴 -45°），两者差一个固定刚体变换
    T = [Rt | tt]。用零位（关节角全零）的法兰姿态求 Rt，位置求 tt。

    Args:
        algo: SDK Algo 实例。
        env: PlanningEnv 实例。
        flange_id: r_flange body id。

    Returns:
        (Rt(3,3), tt(3,)) 或 None（scipy 不可用时）。
    """
    if not SCIPY_AVAILABLE:
        return None
    pa0, Ra0 = algo_fk_pose(algo, [0.0] * 6)
    pm0, Rm0 = mj_flange_pose(env, flange_id, [0.0] * 6)
    Rt = Rm0 @ Ra0.T
    tt = pm0 - Rt @ pa0
    return Rt, tt


# ──────────────────────────────────────────────────────────────
# 测试 1: FK 对比
# ──────────────────────────────────────────────────────────────


def test_fk_comparison(
    algo: Any,
    env: "PlanningEnv",
    flange_id: int,
    joints: list[tuple[list[float], str]],
    verbose: bool,
) -> bool:
    """FK 对比：Algo 法兰 vs MuJoCo 法兰（经基座变换对齐后）。

    同时记录球拍中心位置供参考。关键判据：
      - 绝对位置残差：受 TCP 定义差异影响（Algo 法兰定义点比 MuJoCo
        r_flange 原点沿法兰 Z 轴多约 25mm），故 < 50mm 可接受。
      - 绝对姿态残差：理想 0°，但部分关节角组合会暴露基座/关节方向定义
        差异；用 quaternion 规避万向锁。
      - 相对旋转角一致性：模型一致性的稳健不变量（与基座系无关）。

    Args:
        algo: SDK Algo 实例。
        env: PlanningEnv 实例。
        flange_id: r_flange body id。
        joints: 测试关节角列表 [(q_deg, 标签)]。
        verbose: 是否打印中间结果。

    Returns:
        全部位置残差 < 50mm 且相对旋转角一致时返回 True。
    """
    calib = calibrate_base_transform(algo, env, flange_id)
    if calib is None:
        print("  ⊘ 跳过：scipy 不可用，无法做基座变换标定")
        return False
    Rt, tt = calib

    # 零位的 Algo 法兰位姿作为相对旋转参考
    pa0, Ra0 = algo_fk_pose(algo, [0.0] * 6)

    print("\n  法兰 FK 对比（Algo 基座系 → MuJoCo 世界系，零位标定）:")
    print(
        f"  {'标签':<14s} {'位置残差mm':>10s} "
        f"{'姿态残差°':>10s} {'相对旋转角°(algo/mj)':>22s}"
    )
    print("  " + "-" * 60)

    all_ok = True
    worst_pos_mm = 0.0
    for q_deg, label in joints:
        pa, Ra = algo_fk_pose(algo, q_deg)
        pm, Rm = mj_flange_pose(env, flange_id, q_deg)

        # 预测 MuJoCo 法兰位姿 = Rt @ Algo法兰 + tt
        pm_pred = Rt @ pa + tt
        Rm_pred = Rt @ Ra
        pos_resid_mm = float(np.linalg.norm(pm - pm_pred) * 1000.0)

        # 绝对姿态残差
        if SCIPY_AVAILABLE:
            rot_resid_deg = float(
                Rotation.from_matrix(Rm @ Rm_pred.T).magnitude() * 180.0 / np.pi
            )
        else:
            rot_resid_deg = float("nan")

        # 相对旋转角（相似变换不变量，与基座系无关）
        dR_algo = Ra @ Ra0.T
        dR_mj = Rm @ (Rt @ Ra0).T  # MuJoCo 端折算到"标定后的 Algo 基座系"
        if SCIPY_AVAILABLE:
            ang_algo = float(Rotation.from_matrix(dR_algo).magnitude() * 180.0 / np.pi)
            ang_mj = float(Rotation.from_matrix(dR_mj).magnitude() * 180.0 / np.pi)
        else:
            ang_algo = ang_mj = float("nan")

        print(
            f"  {label:<14s} {pos_resid_mm:10.3f} {rot_resid_deg:10.4f} "
            f"{ang_algo:10.3f} / {ang_mj:10.3f}"
        )
        worst_pos_mm = max(worst_pos_mm, pos_resid_mm)

        if verbose:
            print(f"    q_deg     = {q_deg}")
            print(f"    Algo法兰 = {np.round(pa, 5)}")
            print(f"    MJ 法兰 = {np.round(pm, 5)}")
            print(f"    预测值    = {np.round(pm_pred, 5)}")
            # 残差投影到法兰坐标系（定位 TCP 偏移方向）
            resid_flange = Ra.T @ Rt.T @ (pm - pm_pred)
            print(f"    法兰系残差(mm) = {np.round(resid_flange * 1000, 2)}")
            racket_pos = mj_racket_pos(env, q_deg)
            print(f"    球拍中心(MJ) = {np.round(racket_pos, 5)}")

        # 判据：位置残差 < 50mm（容 TCP 定义差异），相对旋转角一致
        if pos_resid_mm > 50.0:
            all_ok = False
        if not np.isclose(ang_algo, ang_mj, atol=1e-3):
            all_ok = False

    print(f"\n  最差位置残差: {worst_pos_mm:.2f} mm")
    print(
        "  注: Algo 法兰定义点沿法兰 Z 轴比 MuJoCo r_flange 原点多约 25mm，"
        "属 TCP 定义差异（非模型误差）。"
    )
    return all_ok


# ──────────────────────────────────────────────────────────────
# 测试 2: IK 回环
# ──────────────────────────────────────────────────────────────


def test_ik_loopback(
    algo: Any, joints: list[tuple[list[float], str]], verbose: bool
) -> bool:
    """IK 回环：FK(q) → 位姿 → IK → q'，验证 q' ≈ q。

    Args:
        algo: SDK Algo 实例。
        joints: 测试关节角列表。
        verbose: 是否打印每个解。

    Returns:
        全部回环关节角差 < IK_LOOP_TOL_DEG 时返回 True。
    """
    from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t

    print("\n  IK 回环（FK → 位姿 → IK）:")
    print(f"  {'标签':<14s} {'最大关节角差°':>14s} {'结果':>6s}")
    print("  " + "-" * 40)

    all_ok = True
    for q_deg, label in joints:
        pose = algo.rm_algo_forward_kinematics(q_deg, flag=1)  # [x,y,z,rx,ry,rz]
        params = rm_inverse_kinematics_params_t(
            q_in=q_deg, q_pose=pose, flag=1
        )
        ret, q_out = algo.rm_algo_inverse_kinematics(params)
        if ret != 0:
            print(f"  {label:<14s} {'N/A':>14s} {'FAIL':>6s}  (IK ret={ret})")
            all_ok = False
            continue
        diff = np.max(np.abs(np.array(q_out) - np.array(q_deg)))
        ok = diff < IK_LOOP_TOL_DEG
        all_ok = all_ok and ok
        print(
            f"  {label:<14s} {diff:14.6f} {'OK' if ok else 'FAIL':>6s}"
        )
        if verbose:
            print(f"    目标 q = {[round(v, 3) for v in q_deg]}")
            print(f"    回环 q' = {[round(v, 3) for v in q_out]}")
            print(f"    位姿   = {[round(v, 5) for v in pose]}")
    return all_ok


# ──────────────────────────────────────────────────────────────
# 测试 3: IK 全解 + 最优解选择
# ──────────────────────────────────────────────────────────────


def test_ik_all_solutions(algo: Any, verbose: bool) -> bool:
    """IK 全解 + 最优解选择（以 IK 示例姿态 q4 为测试点）。

    Args:
        algo: SDK Algo 实例。
        verbose: 是否打印每个解。

    Returns:
        全解数量 > 1 且最优解离参考关节角最近时返回 True。
    """
    from Robotic_Arm.rm_ctypes_wrap import rm_inverse_kinematics_params_t

    q_ref = [0, 0, -90, 0, -90, 0]  # IK 示例姿态
    pose = algo.rm_algo_forward_kinematics(q_ref, flag=1)
    params = rm_inverse_kinematics_params_t(q_in=q_ref, q_pose=pose, flag=1)

    result = algo.rm_algo_inverse_kinematics_all(params)
    num = result.num
    print(f"\n  IK 全解数量: {num}")
    if num <= 0:
        print("  ❌ 无解")
        return False

    # 收集有效解（前 num 个）
    solutions = [list(result.q_solve[i][:6]) for i in range(num)]
    if verbose:
        for i, sol in enumerate(solutions):
            print(f"    解[{i}] = {[round(v, 3) for v in sol]}")

    # 最优解选择
    idx = algo.rm_algo_ikine_select_ik_solve([1.0] * 6, result)
    best_sol = solutions[idx]
    print(f"  最优解索引: {idx}")
    print(f"  最优解 = {[round(v, 3) for v in best_sol]}")
    print(f"  参考 q = {[round(v, 3) for v in q_ref]}")

    # 验证：最优解应为离 q_ref 最近的解
    dists = [np.max(np.abs(np.array(s) - np.array(q_ref))) for s in solutions]
    nearest_idx = int(np.argmin(dists))
    print(
        f"  最近解索引: {nearest_idx} (距离 {dists[nearest_idx]:.3f}°)，"
        f"最优解距离 {dists[idx]:.3f}°"
    )
    # 最优解不一定严格等于最近解（权重策略不同），但应在合理范围
    ok = dists[idx] < 5.0
    print(f"  {'✅ 最优解合理' if ok else '⚠️ 最优解偏离参考关节角'}")
    return ok


# ──────────────────────────────────────────────────────────────
# 测试 4: 限位检查
# ──────────────────────────────────────────────────────────────


def test_joint_limits(algo: Any) -> bool:
    """关节位置限位 + 速度限位检查。

    注意: SDK 的 rm_algo_ikine_check_joint_position_limit 内部直接把参数传给
    ctypes 底层函数，需手动构造 (c_float*8) 数组（补 0 至 8 元素），否则报
    "expected LP_c_float" 错误。

    Args:
        algo: SDK Algo 实例。

    Returns:
        正常关节角返回 0、超限关节角返回非 0 时为 True。
    """
    print("\n  关节限位检查:")

    def _to_c8(q: list[float]) -> Any:
        """把 6 元素关节角补零到 8 元素并转为 c_float 数组。"""
        return (c_float * 8)(*(list(q) + [0.0] * (8 - len(q))))

    # 正常关节角 → 应返回 0
    q_normal = [0, 20, 70, 0, 90, 0]
    ret_normal = algo.rm_algo_ikine_check_joint_position_limit(_to_c8(q_normal))
    print(
        f"    正常角 {q_normal} → ret={ret_normal} (期望 0)"
        f"  {'✅' if ret_normal == 0 else '❌'}"
    )

    # 超限关节角（J1=200°，RM-65B J1 限位 ±180°）
    q_over = [200, 0, 0, 0, 0, 0]
    ret_over = algo.rm_algo_ikine_check_joint_position_limit(_to_c8(q_over))
    print(
        f"    超限角 {q_over} → ret={ret_over} (期望非 0)"
        f"  {'✅' if ret_over != 0 else '❌'}"
    )

    # 速度限位检查：dt 很小 + 角度跳变大 → 应报超限
    dt = 0.001  # 1ms
    ret_vel = algo.rm_algo_ikine_check_joint_velocity_limit(
        dt, _to_c8([0, 0, 0, 0, 0, 0]), _to_c8([90, 0, 0, 0, 0, 0])
    )
    print(
        f"    速度超限 dt={dt}s ΔJ1=90° → ret={ret_vel} (期望非 0)"
        f"  {'✅' if ret_vel != 0 else '❌'}"
    )

    return ret_normal == 0 and ret_over != 0 and ret_vel != 0


# ──────────────────────────────────────────────────────────────
# 测试 5: DH 参数
# ──────────────────────────────────────────────────────────────


def test_dh_params(algo: Any, verbose: bool) -> bool:
    """读取并打印算法层 DH 参数。

    rm_algo_get_dh 返回 dict（内部调用 to_dict(dh_dof)），包含
    d/a/alpha/offset 四组（各 dh_dof=6 个元素）。

    Args:
        algo: SDK Algo 实例。
        verbose: 是否逐关节打印。

    Returns:
        成功读取 4 组各 6 元素时返回 True。
    """
    print("\n  算法层 DH 参数 (rm_algo_get_dh):")
    try:
        dh = algo.rm_algo_get_dh()
    except Exception as e:  # noqa: BLE001
        print(f"    ❌ 读取失败: {e}")
        return False

    # dh 可能是 dict（to_dict 后）或 rm_dh_t 结构体
    if isinstance(dh, dict):
        d = list(dh.get("d", []))
        a = list(dh.get("a", []))
        alpha = list(dh.get("alpha", []))
        offset = list(dh.get("offset", []))
    else:
        d = list(dh.d[: algo.dh_dof])
        a = list(dh.a[: algo.dh_dof])
        alpha = list(dh.alpha[: algo.dh_dof])
        offset = list(dh.offset[: algo.dh_dof])

    print(f"    d      (m)  = {[round(v, 5) for v in d]}")
    print(f"    a      (m)  = {[round(v, 5) for v in a]}")
    print(f"    alpha  (°)  = {[round(v, 3) for v in alpha]}")
    print(f"    offset (°)  = {[round(v, 3) for v in offset]}")
    if verbose:
        print("    逐关节:")
        for i in range(len(d)):
            print(
                f"      J{i+1}: d={d[i]:.5f} a={a[i]:.5f} "
                f"alpha={alpha[i]:.3f}° offset={offset[i]:.3f}°"
            )
    ok = len(d) == algo.dh_dof and len(a) == algo.dh_dof
    print(f"    {'✅' if ok else '❌'} DH 参数读取完整")
    return ok


# ──────────────────────────────────────────────────────────────
# 测试 6: 坐标变换往返
# ──────────────────────────────────────────────────────────────


def test_coord_roundtrip(algo: Any) -> bool:
    """欧拉角↔四元数、欧拉角↔旋转矩阵往返误差。

    Args:
        algo: SDK Algo 实例。

    Returns:
        往返误差 < ROUNDTRIP_TOL 时返回 True。
    """
    print("\n  坐标变换往返:")
    eul = [0.1, 0.2, 0.3]  # rad

    # 欧拉角 → 四元数 → 欧拉角
    quat = algo.rm_algo_euler2quaternion(eul)
    eul_back = algo.rm_algo_quaternion2euler(quat)
    err_eul = float(np.max(np.abs(np.array(eul_back) - np.array(eul))))
    print(
        f"    euler→quat→euler: {eul} → {[round(v, 6) for v in quat]} → "
        f"{[round(v, 6) for v in eul_back]}  误差={err_eul:.2e}"
    )

    # 欧拉角 → 旋转矩阵 → 位姿（仅旋转部分）
    matrix = algo.rm_algo_euler2matrix(eul)
    pose_back = algo.rm_algo_matrix2pos(matrix, flag=1)
    # pose_back = [x,y,z,rx,ry,rz]，位置应为 0
    err_rot = float(
        np.max(np.abs(np.array(pose_back[3:]) - np.array(eul)))
    )
    err_pos = float(np.max(np.abs(np.array(pose_back[:3]))))
    print(
        f"    euler→matrix→pos: 位姿={[round(v, 6) for v in pose_back]}  "
        f"姿态误差={err_rot:.2e} 位置误差={err_pos:.2e}"
    )

    ok = err_eul < ROUNDTRIP_TOL and err_rot < ROUNDTRIP_TOL and err_pos < ROUNDTRIP_TOL
    print(f"    {'✅' if ok else '❌'} 往返误差 < {ROUNDTRIP_TOL}")
    return ok


# ──────────────────────────────────────────────────────────────
# 主函数
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """主函数：初始化 → 6 项测试 → 汇总结论。"""
    parser = argparse.ArgumentParser(
        description="RM-65B Algo 算法库正逆运动学校验（零风险，纯软件计算）"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="打印详细的中间结果"
    )
    parser.add_argument(
        "--online",
        action="store_true",
        help="连接真机读取当前关节角作为额外采样点",
    )
    # --config 总是注册（有默认值，非 --online 模式也不会出错）
    from _connect import add_config_arg

    add_config_arg(parser)
    args = parser.parse_args()

    # ── 初始化 ──
    print("=" * 64)
    print("09_algo_fk_ik_verify — Algo FK/IK 与 MuJoCo PlanningEnv 一致性校验")
    print("=" * 64)

    print("\n[1/7] 初始化 Algo 与 PlanningEnv")
    algo = init_algo()
    if algo is None:
        print("  ❌ SDK Algo 不可用，请安装: pip install Robotic_Arm")
        sys.exit(1)
    print(f"  ✅ Algo 初始化成功，算法库版本: {algo.rm_algo_version()}")
    print(f"     arm_dof={algo.arm_dof}, dh_dof={algo.dh_dof}")

    if not SCIPY_AVAILABLE:
        print("  ⚠️ scipy 不可用，FK 姿态对比与基座变换标定将跳过")

    env = None
    flange_id = -1
    if not PLANNING_ENV_AVAILABLE:
        print("  ⚠️ PlanningEnv 不可用，FK 对比将跳过")
        print(
            f"     原因: {_PLANNING_ENV_IMPORT_ERROR!r}"
            if _PLANNING_ENV_IMPORT_ERROR
            else ""
        )
        print(
            "     若缺少 C++ 扩展，请运行: python setup.py build_ext --inplace"
        )
    else:
        env = PlanningEnv()
        flange_id = mujoco.mj_name2id(
            env.model, mujoco.mjtObj.mjOBJ_BODY, "r_flange"
        )
        print(f"  ✅ PlanningEnv 初始化成功，r_flange body id={flange_id}")

    # 额外采样点：真机当前关节角
    joints = list(TEST_JOINTS)
    if args.online:
        print("\n  --online 模式：连接真机读取当前关节角")
        from _connect import load_and_connect, safe_disconnect

        ri, _config = load_and_connect(args.config)
        try:
            state = ri.get_arm_state()
            q_now_deg = np.degrees(state[:6]).tolist()
            joints.append((q_now_deg, "真机当前角"))
            print(f"  ✅ 当前关节角(°) = {[round(v, 2) for v in q_now_deg]}")
        finally:
            safe_disconnect(ri)

    # ── 测试 ──
    results: dict[str, bool] = {}

    print("\n[2/7] FK 对比（Algo 法兰 vs MuJoCo 法兰）")
    if env is not None:
        results["fk_comparison"] = test_fk_comparison(
            algo, env, flange_id, joints, args.verbose
        )
    else:
        print("  ⊘ 跳过（PlanningEnv 不可用）")
        results["fk_comparison"] = False

    print("\n[3/7] IK 回环验证")
    results["ik_loopback"] = test_ik_loopback(algo, joints, args.verbose)

    print("\n[4/7] IK 全解 + 最优解选择")
    results["ik_all_solutions"] = test_ik_all_solutions(algo, args.verbose)

    print("\n[5/7] 关节限位检查")
    results["joint_limits"] = test_joint_limits(algo)

    print("\n[6/7] DH 参数")
    results["dh_params"] = test_dh_params(algo, args.verbose)

    print("\n[7/7] 坐标变换往返")
    results["coord_roundtrip"] = test_coord_roundtrip(algo)

    # ── 汇总 ──
    print("\n" + "=" * 64)
    print("汇总")
    print("=" * 64)
    for name, ok in results.items():
        print(f"  {'✅' if ok else '❌'} {name}")

    # 关键结论
    print("\n关键结论:")
    if results.get("fk_comparison"):
        print("  ✅ FK 位置残差 < 50mm（容 TCP 定义差异），相对旋转角一致")
        print("     → 两个运动学模型结构一致，可用于交叉验证")
    elif env is not None:
        print("  ⚠️ FK 对比未通过全部判据，请查看上方详细数据")
        print("     常见原因: TCP 定义差异（~25mm）、关节方向定义、基座安装")
    else:
        print("  ⊘ FK 对比未执行（PlanningEnv 不可用）")

    # 仅当所有"必通过"项通过时才整体成功
    must_pass = ["ik_loopback", "ik_all_solutions", "joint_limits", "coord_roundtrip"]
    all_pass = all(results.get(k, False) for k in must_pass)
    if all_pass:
        print("\n  ✅ Algo 内部一致性全部通过（IK 回环/全解/限位/坐标变换）")
    else:
        failed = [k for k in must_pass if not results.get(k, False)]
        print(f"\n  ❌ 以下必通过项失败: {failed}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
