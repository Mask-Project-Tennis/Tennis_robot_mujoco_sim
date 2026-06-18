"""RealRunner 组装工厂 — 共享常量与构建函数。

将 scripts/run_real_robot.py / tests/test_real_runner.py /
tests/test_replan_core.py 中逐字复制的工厂函数
（_build_robot_limits / _build_solver / _build_replan_cfg）和共享常量
集中到此模块，消除 ~200 行重复代码。

常量对齐 V11 真机配置；函数去掉下划线前缀作为公开 API，
供真机入口脚本与测试用例共同复用，确保规划行为完全一致。

B3 统一：build_replan_cfg 经 ReplanConfig.from_mpc_config().to_dict() 构建，
与 MPCController._build_replan_cfg() 共享同一字段映射，消除重复翻译。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_config import ReplanConfig
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import TubeConfig

if TYPE_CHECKING:
    # 仅类型检查期导入 MPCConfig，运行期用延迟导入打破与 mpc_controller 的循环依赖
    from src.ilqt.mpc_controller import MPCConfig

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


def _build_real_robot_mpc_config() -> "MPCConfig":
    """构建真机专用 MPCConfig（对齐旧 build_replan_cfg 硬编码值）。

    旧 build_replan_cfg 内联 42 个键值；本函数将这些值映射到 MPCConfig 字段，
    再由 ReplanConfig.from_mpc_config().to_dict() 统一产出 51-key dict。

    关键差异（vs MPCConfig 默认）：
    - total_horizon=200, fixed_horizon=60（短 horizon 加速真机响应）
    - max_iter_per_plan=3, first_plan_iters=5, near_plan_iters=2（少迭代减延迟）
    - is_position_mode=True（真机位置控制）
    - racket_speed=5.0, backswing_offset=0.0, backswing_ratio=0.3, r_decay_ratio=0.3
    - smooth_far/mid/near 真机专用权重

    行为保留：Q_p_base/Q_v_base/Q_qdot_base/Q_qddot_base/Q_du_base 显式置空/零，
    使 do_replan 的 .get() 默认路径（eye(3) / 0.0）与旧路径（键缺失）等价。

    Returns:
        MPCConfig 实例（真机专用参数集）。
    """
    # 延迟导入：mpc_controller 反向依赖本模块的 build_robot_limits/build_solver，
    # 运行期延迟到此函数调用时再加载，打破模块级循环依赖
    from src.ilqt.mpc_controller import MPCConfig

    return MPCConfig(
        dt=DT,
        total_horizon=200,
        fixed_horizon=60,
        replan_interval=20,
        max_iter_per_plan=3,
        first_plan_iters=5,
        near_plan_iters=2,
        near_threshold=80,
        R=0.0001,
        Q_p_scale_far=5.0,
        Q_v_scale_far=3.0,
        Q_p_scale_near=8.0,
        Q_v_scale_near=120.0,
        normal_weight=500000.0,
        racket_speed=5.0,
        max_tcp_speed=1.8,
        is_position_mode=True,
        ablation_mode="full",
        use_backswing=False,
        use_r_decay=False,
        r_decay_ratio=0.3,
        fix_joint5_angle=None,
        backswing_offset=0.0,
        backswing_ratio=0.3,
        normal_flip=False,
        shoulder_pos=SHOULDER_POS,
        workspace_radius=WORKSPACE_RADIUS,
        follow_through_length=0.0,
        follow_through_steps=0,
        follow_through_v_terminal=0.3,
        time_perturb_s=0.0,
        space_perturb_m=0.0,
        perturb_alpha_min=0.0,
        smooth_far={"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0},
        smooth_mid={"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0},
        smooth_near={"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 3.0},
        tube_cfg=TubeConfig(),
        # 行为保留：do_replan 对这些键用 .get(..., None/0.0)，
        # 显式置空使值与旧路径（键缺失）完全等价
        Q_p_base=None,
        Q_v_base=None,
        Q_qdot_base=0.0,
        Q_qddot_base=0.0,
        Q_du_base=0.0,
        far_threshold=50,
    )


def build_replan_cfg(
    _env: PlanningEnv,
    robot_limits: RobotLimits,
    solver: Any,
    d_hat: np.ndarray,
    v_hit_desired: np.ndarray,
) -> dict:
    """构建 do_replan 所需的完整配置字典（51 键）。

    B3 统一：经 ``ReplanConfig.from_mpc_config().to_dict()`` 构建，
    与 ``MPCController._build_replan_cfg()`` 共享同一字段映射工厂，
    消除两条路径各自维护字段翻译的重复。

    ``env`` 参数预留以备未来扩展（如从 env 读取额外配置），当前未使用。

    Args:
        env: 规划环境。
        robot_limits: 关节约束 + 安全滤波器。
        solver: ILQTSolver 实例。
        d_hat: 期望击球方向单位向量（来球反方向）。
        v_hit_desired: 期望击球时刻末端速度。

    Returns:
        replan_cfg dict，可直接传给 AsyncReplanner / do_replan / RealRunner。
    """
    config = _build_real_robot_mpc_config()
    replan_cfg = ReplanConfig.from_mpc_config(
        config,
        robot_limits=robot_limits,
        solver=solver,
        d_hat=d_hat,
        v_hit_desired=v_hit_desired,
    )
    return replan_cfg.to_dict()
