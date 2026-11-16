"""RM65 旧参数兼容适配器 — 仿真侧（阶段 2 Task 2.3）。

把 RM65 现有仿真参数来源（src.robot.constants 初态、configs/default.yaml
结构的 config 映射、RobotLimits.from_config 限位、模型 actuator ctrlrange
力矩界）映射进 Task 2.2 的 RobotRuntimeParameters 结构。

本阶段仅新增显式调用接口，不接入生产入口（阶段 3 才迁移实际入口），
不改变旧入口的默认行为。

分工（shared-constraints §2.2 接口契约）：
  - 本模块（ilqt 层）：仿真适配器 resolve_rm65_sim_parameters + 对称力矩
    折算 symmetric_torque_max（真机适配器复用同一折算规则，不另写一份）。
  - src/real/runner_factory.py（real 层）：真机适配器
    resolve_rm65_real_parameters，复用原 build_robot_limits 安全规则。

ilqt 层不反向依赖 real 层，故 RM65 身份期望在本模块独立声明
（与 configs/robots/rm65.yaml、runner_factory 阶段 1 常量对齐）。
"""

from __future__ import annotations

import logging
import math
from typing import Any, Mapping

import mujoco
import numpy as np

from src.ilqt.robot_limits import RobotLimits
from src.robot.constants import INIT_Q, INIT_Q_LEFT
from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    validate_runtime_parameters,
)

logger = logging.getLogger(__name__)

# ── RM65 身份期望（与 configs/robots/rm65.yaml 对齐）──
_RM65_ROBOT_ID = "rm65"
_RM65_ACTIVE_JOINTS = tuple(f"r_joint{i}" for i in range(1, 7))
_RM65_HELD_JOINTS = tuple(f"l_joint{i}" for i in range(1, 7))

# 位置模式 PD 缺省回退（与 scripts/rm65_mpc_v12.py 当前入口一致，勿改数值）
_SIM_KP_FALLBACK = (200.0, 200.0, 200.0, 50.0, 50.0, 20.0)
_SIM_KD_FALLBACK = (20.0, 20.0, 5.0, 5.0, 5.0, 2.0)

# 对称性判定的绝对容差（IEEE 取反精确，仅容许序列化级别噪声）
_SYMMETRY_ATOL = 1e-9

# 参数来源说明（source 仅作溯源，不作为安全授权）
_SIM_SOURCE = (
    "RM65 仿真旧参数（src.robot.constants 初态 + config 映射"
    "（sim/actuator/robot_limits）+ RobotLimits.from_config + 模型 ctrlrange）"
)


def symmetric_torque_max(u_min: np.ndarray, u_max: np.ndarray) -> np.ndarray:
    """把对称的力矩上下限折算为对称幅值数组 torque_max。

    仅当 u_min == -u_max（逐元素，容差 1e-9）时折算为
    torque_max = u_max；不对称时明确拒绝，不取较大值放宽限值。

    Args:
        u_min: (n,) 力矩下界（N·m）。
        u_max: (n,) 力矩上界（N·m）。

    Returns:
        (n,) 对称力矩幅值（N·m），等于 u_max 的独立副本。

    Raises:
        ValueError: 形状不一致、含非有限值、上界非正或上下限不对称。
    """
    u_min = np.asarray(u_min, dtype=np.float64)
    u_max = np.asarray(u_max, dtype=np.float64)
    if u_min.ndim != 1 or u_min.shape != u_max.shape:
        raise ValueError(
            f"字段 torque_max 力矩上下限形状不一致: "
            f"{u_min.shape} vs {u_max.shape}"
        )
    if not bool(np.all(np.isfinite(u_min)) and np.all(np.isfinite(u_max))):
        raise ValueError("字段 torque_max 力矩上下限含 NaN/Inf")
    if bool(np.any(u_max <= 0.0)):
        raise ValueError("字段 torque_max 力矩上界 u_max 必须逐元素为正")
    if not bool(np.allclose(u_min, -u_max, rtol=0.0, atol=_SYMMETRY_ATOL)):
        raise ValueError(
            "字段 torque_max 折算失败：力矩上下限不对称"
            f"（u_min={u_min.tolist()}，u_max={u_max.tolist()}；"
            "明确拒绝，不取较大值放宽限值）"
        )
    return u_max.copy()


def resolve_rm65_sim_parameters(
    profile: RobotProfile,
    model: mujoco.MjModel,
    config: Mapping[str, Any],
) -> RobotRuntimeParameters:
    """把 RM65 仿真旧参数来源映射为 RobotRuntimeParameters。

    逐项来源（与 scripts/rm65_mpc_v12.py 当前入口一致）：
      - init_q/held_q：constants.INIT_Q / constants.INIT_Q_LEFT（仿真初态，
        与真机 INIT_Q_REAL 保持区别，不互相顶替）；
      - dt：config["sim"]["dt"]（缺项直接 KeyError，与入口同行为，不设回退）；
      - kp/kd：config["actuator"]，仅位置模式（mode == "position"）读取，
        缺项回退入口同款缺省值；力矩模式未配置位置控制 → 整组 None；
      - q_lower/q_upper/qdot_max/qddot_max/max_tcp_speed：
        复用 RobotLimits.from_config（config["robot_limits"]，缺项走其内置
        缺省），不含 beta/窗口/豁免等策略字段；
      - torque_max：按 binding.actuator_ids 选出的 ctrlrange 行对称折算。
    不读取 default.yaml 中旧 UR5e 的 init_q/hitting 几何作为 RM65 事实。

    Args:
        profile: RM65 机器人描述（身份与关节顺序基准）。
        model: RM65 MuJoCo 模型（读取 actuator ctrlrange）。
        config: configs/default.yaml 同构的配置映射
            （消费 sim.dt、actuator.mode/kp/kd、robot_limits 节）。

    Returns:
        通过 kinematics 级校验的 RobotRuntimeParameters
        （init_q=INIT_Q、held_q=INIT_Q_LEFT、scene=None、无临时字段）。

    Raises:
        ValueError: 描述不是 RM65、名称绑定失败、力矩界限不对称、
            数组限值含非有限值或任一 kinematics 级校验失败。
    """
    _require_rm65_profile(profile)
    # 名称绑定：校验 profile 与模型一致，并取得 actuator_ids 用于选 ctrlrange 行
    binding = bind_robot_model(profile, model)

    # dt：与当前入口一致，config["sim"]["dt"] 直接读取（缺项 KeyError 不做回退）
    dt = float(config["sim"]["dt"])

    # PD：仅位置模式读取（缺项回退与当前入口 rm65_mpc_v12.py 一致）；
    # 力矩模式未配置位置控制 → kp/kd 整组 None
    actuator_cfg = config.get("actuator", {})
    if actuator_cfg.get("mode", "torque") == "position":
        kp: np.ndarray | None = np.asarray(
            actuator_cfg.get("kp", _SIM_KP_FALLBACK), dtype=np.float64
        )
        kd: np.ndarray | None = np.asarray(
            actuator_cfg.get("kd", _SIM_KD_FALLBACK), dtype=np.float64
        )
    else:
        kp = None
        kd = None

    # 限位复用 RobotLimits.from_config（唯一来源，不另写一份安全规则）
    limits = _sim_limits_from_config(
        binding, model, dt, config.get("robot_limits", {})
    )
    torque_max = symmetric_torque_max(limits.u_min, limits.u_max)

    # TCP 限速：RobotLimits 的 inf 表示"未配置限速"（from_config 缺省），
    # 映射为 None（未提供有限限值），不发明数值；数组限值不适用此映射，
    # 含非有限值时由 validate_runtime_parameters 按字段名拒绝
    max_tcp_speed: float | None = float(limits.max_tcp_speed)
    if not math.isfinite(max_tcp_speed):
        logger.debug(
            "RM65 仿真适配器：robot_limits 未配置有限 TCP 限速"
            "（RobotLimits.max_tcp_speed=%s），映射为 None",
            limits.max_tcp_speed,
        )
        max_tcp_speed = None

    params = RobotRuntimeParameters(
        robot_id=_RM65_ROBOT_ID,
        joint_names=tuple(profile.active_joint_names),
        dt=dt,
        init_q=INIT_Q,
        held_q=INIT_Q_LEFT,
        kp=kp,
        kd=kd,
        q_lower=limits.q_lower,
        q_upper=limits.q_upper,
        qdot_max=limits.qdot_max,
        qddot_max=limits.qddot_max,
        torque_max=torque_max,
        max_tcp_speed=max_tcp_speed,
        scene=None,
        provisional_fields=(),
        source=_SIM_SOURCE,
    )
    validate_runtime_parameters(params, profile, purpose="kinematics")
    return params


def _require_rm65_profile(profile: RobotProfile) -> None:
    """校验描述身份为 RM65（本适配器只服务 RM65 旧参数来源）。

    INIT_Q/INIT_Q_LEFT 是 RM65 专属六维初态，身份或关节名不符时映射
    无意义，须在绑定模型之前拒绝。

    Raises:
        ValueError: robot_id 不是 rm65，或主动/保持关节名与 RM65 期望不符。
    """
    if profile.robot_id != _RM65_ROBOT_ID:
        raise ValueError(
            f"字段 robot_id 适配器仅支持 RM65（{_RM65_ROBOT_ID!r}），"
            f"不支持 {profile.robot_id!r}"
        )
    if tuple(profile.active_joint_names) != _RM65_ACTIVE_JOINTS:
        raise ValueError(
            f"字段 active_joint_names 与 RM65 期望不符: "
            f"{list(profile.active_joint_names)}（期望 {list(_RM65_ACTIVE_JOINTS)}，"
            f"robot_id: {profile.robot_id}）"
        )
    if tuple(profile.held_joint_names) != _RM65_HELD_JOINTS:
        raise ValueError(
            f"字段 held_joint_names 与 RM65 期望不符: "
            f"{list(profile.held_joint_names)}（期望 {list(_RM65_HELD_JOINTS)}，"
            f"robot_id: {profile.robot_id}）"
        )


def _sim_limits_from_config(
    binding: ModelBinding,
    model: mujoco.MjModel,
    dt: float,
    rl_cfg: Mapping[str, Any],
) -> RobotLimits:
    """按 binding 的 actuator_ids 选 ctrlrange 行并复用 RobotLimits.from_config。

    与当前入口的差异仅在于 ctrlrange 行的选取方式：入口用
    ``actuator_ctrlrange[:NU]`` 前缀切片，本适配器按绑定解析出的
    actuator_ids 逐行选取（RM65 恒为 0..5，两者等价，但绑定选取不依赖
    执行器排列在前这一隐含假设）。力矩对称折算由调用方负责。

    Args:
        binding: profile 与模型绑定结果（提供 actuator_ids）。
        model: MuJoCo 模型（读取 actuator_ctrlrange）。
        dt: 时间步长（s）。
        rl_cfg: robot_limits 配置节（与入口同结构）。

    Returns:
        RobotLimits 实例（含位置/速度/加速度/TCP 限速与 u_min/u_max）。
    """
    actuator_index = np.asarray(binding.actuator_ids, dtype=int)
    ctrlrange = model.actuator_ctrlrange[actuator_index, :]
    return RobotLimits.from_config(rl_cfg, dt=dt, ctrlrange=ctrlrange)
