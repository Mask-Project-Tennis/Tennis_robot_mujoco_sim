"""RealRunner 组装工厂 — 共享常量重导出与构建函数。

共享常量已移至 src.robot.constants（单一事实来源），本模块重导出
以保持真机入口脚本与测试导入路径零改动（INIT_Q 别名为 INIT_Q_REAL）。

工厂函数（build_robot_limits / build_real_robot_mpc_config）从 RealRobotConfig
读取参数；build_solver 从 src.ilqt.solver 重导出。

依赖方向：build_solver 属于 ilqt 层；本模块重导出仅为保持导入路径兼容。
"""

from __future__ import annotations

from pathlib import Path
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
from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.robot_profile import RobotProfile
from src.utils.mujoco_loader import load_mujoco_model

if TYPE_CHECKING:
    # 仅类型检查期导入 MPCConfig，运行期用延迟导入打破与 mpc_controller 的循环依赖
    from src.ilqt.mpc_controller import MPCConfig
    from src.real.robot_arm_protocol import RobotArmInterface

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
    "validate_rm65_profile",
    "validate_rm65_runtime",
    "load_validated_rm65_binding",
    "build_robot_arm",
]

# ── RM65 运行能力常量（与 src/robot/rm65_model.xml、configs/robots/rm65.yaml 对齐）──
# 描述能力与运行能力分开：robot_profile 接受任何合法描述，
# 本层校验当前入口实际可运行的机器人范围（阶段 1 仅 RM65）。
_RM65_ROBOT_ID = "rm65"
_RM65_ACTIVE_JOINTS = tuple(f"r_joint{i}" for i in range(1, 7))
_RM65_ACTUATORS = tuple(f"torque_r_joint{i}" for i in range(1, 7))
_RM65_HELD_JOINTS = tuple(f"l_joint{i}" for i in range(1, 7))
_RM65_TCP_SITE = "racket_center"
_RM65_MODEL_PATH = (
    Path(__file__).resolve().parent.parent / "robot" / "rm65_model.xml"
).resolve()


def validate_rm65_profile(profile: RobotProfile) -> None:
    """校验描述自身满足当前入口的 RM65 运行能力（纯比较，不加载模型）。

    供入口预检查与 mock 路径复用：身份/名称/TCP/模型路径与 RM65 期望
    完全一致。绑定布局校验见 validate_rm65_runtime。

    Args:
        profile: 机器人描述。

    Raises:
        ValueError: 身份不是 rm65、名称与 RM65 期望不符、TCP 名称不符、
            或模型路径被替换。
    """
    if profile.robot_id != _RM65_ROBOT_ID:
        raise ValueError(
            f"运行入口当前仅支持 RM65，不支持 robot_id={profile.robot_id!r}"
        )
    expected_names = (
        (profile.active_joint_names, _RM65_ACTIVE_JOINTS, "active_joint_names"),
        (profile.actuator_names, _RM65_ACTUATORS, "actuator_names"),
        (profile.held_joint_names, _RM65_HELD_JOINTS, "held_joint_names"),
    )
    for actual, expected, field in expected_names:
        if actual != expected:
            raise ValueError(
                f"字段 {field} 与 RM65 期望不符: {actual}（期望 {expected}）"
            )
    if profile.tcp_site_name != _RM65_TCP_SITE:
        raise ValueError(
            f"tcp_site_name 与 RM65 期望不符: {profile.tcp_site_name!r}"
        )
    if profile.model_path != _RM65_MODEL_PATH:
        raise ValueError(
            f"模型路径与 RM65 现有模型不符: {profile.model_path}"
            f"（期望 {_RM65_MODEL_PATH}）"
        )


def validate_rm65_runtime(profile: RobotProfile, binding: ModelBinding) -> None:
    """校验描述与绑定满足当前入口的 RM65 运行能力。

    Args:
        profile: 机器人描述。
        binding: 模型名称绑定结果。

    Raises:
        ValueError: 描述不符合 RM65 期望（见 validate_rm65_profile），
            或绑定布局不是既有 RM65 原布局（主动 0..5、保持 6..11、执行器 0..5）。
    """
    validate_rm65_profile(profile)
    # 布局校验：主动/执行器 0..5，保持 6..11。
    # tcp 有效性由 bind_robot_model 保证（查不到名称即抛错），无需重复检查。
    if (
        binding.joint_ids != tuple(range(6))
        or binding.qpos_indices != tuple(range(6))
        or binding.qvel_indices != tuple(range(6))
        or binding.actuator_ids != tuple(range(6))
        or binding.held_qpos_indices != tuple(range(6, 12))
        or binding.held_qvel_indices != tuple(range(6, 12))
    ):
        raise ValueError(
            f"绑定布局非 RM65 原布局（主动/执行器应 0..5、保持 6..11，"
            f"实际 qpos={binding.qpos_indices}, 执行器={binding.actuator_ids}, "
            f"保持={binding.held_qpos_indices}）"
        )


def load_validated_rm65_binding(profile: RobotProfile) -> ModelBinding:
    """加载并绑定 profile 的模型，完成 RM65 运行能力校验。

    入口预检查与驱动工厂真机路径的共同实现（消除两处重复的
    load→bind→validate 序列）。纯模型加载，不启动线程、不接触硬件。

    Args:
        profile: 已通过 validate_rm65_profile 预检查的描述。

    Returns:
        通过运行能力校验的 ModelBinding（可复用给 build_robot_arm）。

    Raises:
        ValueError: 绑定失败或运行能力校验不通过。
    """
    model = load_mujoco_model(profile.model_path)
    binding = bind_robot_model(profile, model)
    validate_rm65_runtime(profile, binding)
    return binding


def build_robot_arm(
    config: RealRobotConfig,
    *,
    profile: RobotProfile,
    mock: bool,
    init_q: np.ndarray,
    binding: ModelBinding | None = None,
) -> "RobotArmInterface":
    """按描述构造机械臂驱动（RM65 真机或 FakeRobot Mock）。

    只构造对象，不调用 connect（连接仍归原生命周期管理）。
    两条路径都先做纯描述校验（validate_rm65_profile，含名称/TCP/路径，
    不加载模型）；真机路径另做绑定布局校验。

    Args:
        config: 真机配置。
        profile: 机器人描述。
        mock: True 返回 FakeRobot；False 返回 RobotInterface。
        init_q: 初始关节角度 (ndof,)，弧度。
        binding: 调用方（如 create_runner）已取得的校验绑定；
            传入则复用、免重复加载模型，None 时自行加载并校验。

    Returns:
        实现 RobotArmInterface 的驱动实例（未连接）。

    Raises:
        ValueError: init_q 长度与 profile.ndof 不符、含非有限值，
            或描述不符合当前支持的 rm65（名称/布局任一不符）。
    """
    init_q = np.asarray(init_q, dtype=float)
    if init_q.ndim != 1 or init_q.shape[0] != profile.ndof:
        raise ValueError(
            f"init_q 形状 {init_q.shape} 与自由度 {profile.ndof} 不符"
        )
    if not bool(np.all(np.isfinite(init_q))):
        raise ValueError("init_q 含非有限值（NaN/Inf）")
    # 描述前置校验：在加载模型/构造驱动之前拒绝不支持的机器人
    # （mock 路径同样过名称校验——FakeRobot 是 RM65 专属 6 维 mock）
    validate_rm65_profile(profile)

    if mock:
        from src.real.fake_robot import FakeRobot

        return FakeRobot(init_q=init_q.copy(), dt=config.dt)

    if binding is None:
        binding = load_validated_rm65_binding(profile)
    else:
        validate_rm65_runtime(profile, binding)

    from src.real.robot_interface import RobotInterface

    return RobotInterface(config)


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
