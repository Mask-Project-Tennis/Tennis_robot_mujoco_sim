"""RealRunner 组装工厂 — 共享常量与构建函数。

将 scripts/run_real_robot.py / tests/test_real_runner.py /
tests/test_replan_core.py 中逐字复制的工厂函数
（_build_robot_limits / _build_solver / _build_replan_cfg）和共享常量
集中到此模块，消除 ~200 行重复代码。

常量对齐 V11 真机配置；函数去掉下划线前缀作为公开 API，
供真机入口脚本与测试用例共同复用，确保规划行为完全一致。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import TubeConfig

# ── 共享常量（对齐 V11 真机配置）──
DT: float = 0.005
INIT_Q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q_LEFT = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)
SHOULDER_POS = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
WORKSPACE_RADIUS = 0.90
KP = np.array([200.0, 200.0, 100.0, 50.0, 50.0, 20.0], dtype=np.float64)
KD = np.array([20.0, 20.0, 10.0, 5.0, 5.0, 2.0], dtype=np.float64)


def build_robot_limits(env: PlanningEnv) -> RobotLimits:
    """从 default.yaml 风格配置构建 RobotLimits。

    Args:
        env: 规划环境，用于读取 actuator_ctrlrange。

    Returns:
        RobotLimits 实例（含位置/速度/加速度/TCP 限速约束）。
    """
    rl_cfg = {
        "q_min_deg": [-178, -130, -135, -178, -128, -360],
        "q_max_deg": [178, 130, 135, 178, 128, 360],
        "q_margin_deg": [2, 1, 3, 3, 3, 3],
        "qdot_max_deg_s": [180, 180, 225, 225, 225, 225],
        "qdot_scale": 1.0,
        "qddot_max_deg_s2": [400, 400, 500, 500, 500, 500],
        "qddot_scale": 0.85,
        "max_tcp_speed": 1.8,
        "terminal_exempt_steps": 20,
        "dq_max_fraction": 0.5,
    }
    return RobotLimits.from_config(
        rl_cfg, dt=DT, ctrlrange=env.model.actuator_ctrlrange[: env.NU]
    )


def build_solver() -> Any:
    """构建 ILQTSolver（优先 C++ 加速版，失败回退纯 Python 版）。

    Returns:
        ILQTSolver 实例（C++ 或 Python 实现，二者接口一致）。
    """
    try:
        from src.cpp.solver_cpp import ILQTSolver
    except ImportError:
        from src.ilqt.solver import ILQTSolver
    return ILQTSolver(
        {
            "max_iter": 10,
            "tol": 1e-4,
            "horizon": 60,
            "mu_min": 1e-6,
            "mu_max": 1e10,
            "mu_init": 0.01,
            "delta_0": 1.6,
            "alpha_list": [1.0, 0.5, 0.25, 0.1, 0.05, 0.01],
            "lin_eps": 1e-6,
        }
    )


def build_replan_cfg(
    env: PlanningEnv,
    robot_limits: RobotLimits,
    solver: Any,
    d_hat: np.ndarray,
    v_hit_desired: np.ndarray,
) -> dict:
    """构建 do_replan 所需的完整配置字典（43 键）。

    参数集合在真机入口与测试用例间完全对齐，确保规划行为一致。
    ``env`` 参数预留以备未来扩展（如从 env 读取额外配置），当前未使用。

    Args:
        env: 规划环境。
        robot_limits: 关节约束 + 安全滤波器。
        solver: ILQTSolver 实例。
        d_hat: 期望击球方向单位向量（来球反方向）。
        v_hit_desired: 期望击球时刻末端速度。

    Returns:
        replan_cfg dict，可直接传给 AsyncReplanner / do_replan。
    """
    total_horizon = 200
    return {
        # ── 必需键 ──
        "dt": DT,
        "shoulder_pos": SHOULDER_POS,
        "workspace_radius": WORKSPACE_RADIUS,
        "total_horizon": total_horizon,
        "fixed_horizon": 60,
        "replan_interval": 20,
        "max_iter_per_plan": 3,
        "first_plan_iters": 5,
        "near_plan_iters": 2,
        "near_threshold": 80,
        "R": 0.0001,
        "Q_p_scale_far": 5.0,
        "Q_v_scale_far": 3.0,
        "Q_p_scale_near": 8.0,
        "Q_v_scale_near": 120.0,
        "robot_limits": robot_limits,
        "solver": solver,
        "d_hat": d_hat,
        "v_hit_desired": v_hit_desired,
        "v_hit_at_contact": v_hit_desired,
        "hit_shift": 0.0,
        "follow_through_length": 0.0,
        "time_perturb_s": 0.0,
        "space_perturb_m": 0.0,
        # ── 可选键（真机位置模式）──
        "ablation_mode": "full",
        "is_position_mode": True,
        "use_backswing": False,
        "use_r_decay": False,
        "fix_joint5_angle": None,
        "backswing_offset": 0.0,
        "backswing_ratio": 0.3,
        "r_decay_ratio": 0.3,
        "racket_speed": 5.0,
        "normal_weight": 500000.0,
        "normal_flip": False,
        "max_tcp_speed": 1.8,
        "perturb_alpha_min": 0.0,
        "k_hit_total": total_horizon,
        "tube_cfg": TubeConfig(),
        "smooth_far": {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0},
        "smooth_mid": {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0},
        "smooth_near": {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 3.0},
    }
