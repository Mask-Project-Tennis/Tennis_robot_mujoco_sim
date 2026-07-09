"""RealRunner 组装工厂 — 共享常量与构建函数。

将 scripts/run_real_robot.py / tests/test_real_runner.py /
tests/test_replan_core.py 中逐字复制的工厂函数
（_build_robot_limits）和共享常量
集中到此模块，消除 ~200 行重复代码。

常量对齐 V11 真机配置；函数去掉下划线前缀作为公开 API，
供真机入口脚本与测试用例共同复用，确保规划行为完全一致。

依赖方向：build_solver 已下沉到 src.ilqt.solver（纯 ILQTSolver 工厂属于
ilqt 层）；本模块重导出 build_solver 仅为保持真机入口与测试导入路径零改动。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.ilqt.planning_env import PlanningEnv
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.solver import build_solver
from src.ilqt.tube_types import TubeConfig
from src.real.config import RealRobotConfig
from src.robot.constants import (
    DT,
    INIT_Q_LEFT,
    INIT_Q_REAL as INIT_Q,
    KD,
    KP,
    SHOULDER_POS,
    WORKSPACE_RADIUS,
)

if TYPE_CHECKING:
    # 仅类型检查期导入 MPCConfig，运行期用延迟导入打破与 mpc_controller 的循环依赖
    from src.ilqt.mpc_controller import MPCConfig

# 公开 API（显式声明 re-export，供 `from src.real.runner_factory import build_solver` 零改动）
__all__ = [
    "DT",
    "INIT_Q",
    "INIT_Q_LEFT",
    "SHOULDER_POS",
    "WORKSPACE_RADIUS",
    "KP",
    "KD",
    "build_robot_limits",
    "build_solver",
    "build_real_robot_mpc_config",
]


def build_robot_limits(env: PlanningEnv, config: RealRobotConfig) -> RobotLimits:
    """从 RealRobotConfig 构建 RobotLimits。

    关节限位、TCP 速度、关节速度上限、dt 均从 config 读取。
    q_margin_deg / qddot 等规划层参数保持为模块常量。

    Args:
        env: 规划环境，用于读取 actuator_ctrlrange。
        config: 真机配置（YAML 为唯一真相源）。

    Returns:
        RobotLimits 实例（含位置/速度/加速度/TCP 限速约束）。
    """
    rl_cfg = {
        "q_min_deg": np.degrees(config.q_lower).tolist(),
        "q_max_deg": np.degrees(config.q_upper).tolist(),
        "q_margin_deg": [2, 1, 3, 3, 3, 3],
        "qdot_max_deg_s": np.degrees(config.max_qdot).tolist(),
        "qdot_scale": 1.0,
        "qddot_max_deg_s2": [400, 400, 500, 500, 500, 500],
        "qddot_scale": 0.85,
        "max_tcp_speed": config.max_tcp_speed,
        "terminal_exempt_steps": 20,
        "dq_max_fraction": 0.5,
    }
    return RobotLimits.from_config(
        rl_cfg, dt=config.dt, ctrlrange=env.model.actuator_ctrlrange[: env.NU]
    )


def build_real_robot_mpc_config(config: RealRobotConfig) -> "MPCConfig":
    """构建真机专用 MPCConfig（从 RealRobotConfig 读取可调参数）。

    dt / racket_speed 从 config 读取，
    其余 MPC 参数保持为真机对齐常量。
    TCP 速度由 robot_limits（build_robot_limits）单独管理，
    MPCConfig 不再持有 max_tcp_speed 死字段。

    Args:
        config: 真机配置（YAML 为唯一真相源）。

    Returns:
        MPCConfig 实例（真机专用参数集）。
    """
    # 延迟导入 MPCConfig：仅在真机配置构建路径需要，避免模块加载期耦合
    from src.ilqt.mpc_controller import MPCConfig

    return MPCConfig(
        dt=config.dt,
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
        racket_speed=config.target_hit_speed,
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
