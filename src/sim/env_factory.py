"""R1D 环境工厂 — 带球执行环境 + 规划副本（阶段 3 · Task 3.4）。

职责（施工计划 §共享接口契约 RobotEnvironments 条目）：
- ``RobotEnvironments``：冻结 dataclass，``execution: ProfileSimEnv``（带
  物理球执行环境）+ ``planning: BoundRobotEnv``（规划副本，实际交付
  ``ProfilePlanningEnv`` —— 绑定 BallPredictor 球预测适配层）；
- ``build_r1d_environments(profile, parameters, task)``：分别构造执行与
  规划模型 —— 执行 = 3.2 带球场景（球物理 + 保护碰撞），规划 = 阶段 2
  裸模型（无球、无显式 pair、几何掩码全 0：禁用球接触、不承担真实碰撞
  判定）。两侧 model 与 data 均为独立编译实例，互不共享可变状态。

预检复用说明：3.1 的 ``validate_simulation_inputs`` 每次调用会加载一次
裸模型；本工厂组装带球场景时已获得回填限位后的同源参数，故直接调用
``validate_runtime_parameters(purpose="simulation")`` 复刻同一仿真预检
（身份/关节序校验由 ``build_tennis_scene`` 承担），避免双重加载。

执行器模式同步：``task.actuator_mode == "position"`` 时两个环境同时
``configure_actuator_mode("position", kp, kd)``（增益取 runtime kp/kd，
3.1 预检保证完整），保证规划与执行动力学一致；torque 模式保持构造默认。

球初态：``execution.set_initial_ball_state(task.initial_ball_position,
task.initial_ball_velocity)`` 把 task 球初态编入 reset 基准（reset 可
重现发球初态）。球预测复用 ``BallPredictor``，绑定在 planning 侧
（与 execution 物理球存储分离）。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from src.ilqt.profile_planning_env import ProfilePlanningEnv
from src.robot.model_factory import load_robot_model
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    validate_runtime_parameters,
)
from src.sim.bound_robot_env import BoundRobotEnv
from src.sim.profile_sim_env import ProfileSimEnv
from src.sim.sim_config import SimTaskConfig
from src.sim.tennis_scene import build_tennis_scene

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RobotEnvironments:
    """R1D 仿真环境对 — 带球执行环境 + 规划副本。

    Attributes:
        execution: 带物理球的执行环境（球由 MuJoCo 推进；球—拍接触经
            pair_margin 单独控制通道开关）。
        planning: 规划副本环境（裸模型，无球物理/无碰撞判定；实际类型
            为 ``ProfilePlanningEnv``，绑定解析球预测适配层）。
    """

    execution: ProfileSimEnv
    planning: BoundRobotEnv


def build_r1d_environments(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    task: SimTaskConfig,
) -> RobotEnvironments:
    """构造 R1D 执行 + 规划环境对（模型/data 独立，预检严格）。

    流程：
    1. ``build_tennis_scene`` 组装带球场景（内含 task/profile 身份与
       关节序校验、带球形状校验、q 限位回填）；
    2. 仿真完整性预检：``validate_runtime_parameters(purpose="simulation")``
       （复用场景回填后的同源参数，不二次加载模型）+ 位置模式 kp/kd
       完整性（同 3.1 规则）；
    3. ``load_robot_model`` 组装裸模型（规划副本，独立编译实例）；
    4. 构造 ``ProfileSimEnv``（执行）+ ``ProfilePlanningEnv``（规划）；
    5. 位置模式同步配置两环境执行器；task 球初态编入 execution 的
       reset 基准。

    Args:
        profile: 机器人描述（robot_id 须为 unitree_r1d，侧别经 model_path
            显式表达）。
        parameters: 仿真运行参数（q_lower/q_upper 通常为 None，由模型回填）。
        task: 仿真任务配置（球初态/执行器模式/几何名称来源）。

    Returns:
        环境对（execution/planning 的 model 与 data 均不共享可变实例）。

    Raises:
        ValueError: 身份/关节序不一致、场景组装失败、仿真完整性预检
            失败，或位置模式缺 kp/kd（不隐式补默认值）。
    """
    # 1. 带球场景（身份校验 + 限位回填内含；每次调用独立编译模型）
    scene = build_tennis_scene(profile, parameters, task)

    # 2. 仿真完整性预检（scene.robot.parameters 已从 jnt_range 回填；
    #    与 validate_simulation_inputs 同规则，但不二次加载模型）
    filled = scene.robot.parameters
    validate_runtime_parameters(filled, profile, purpose="simulation")
    if task.actuator_mode == "position":
        missing = [
            field
            for field in ("kp", "kd")
            if getattr(filled, field) is None
        ]
        if missing:
            context = f"robot_id: {profile.robot_id}，来源: {filled.source}"
            raise ValueError(
                f"字段 {'/'.join(missing)} position 模式需要完整 kp/kd "
                f"仿真 PD 增益，当前为 None（不得隐式补默认值）（{context}）"
            )

    # 3. 规划副本（裸模型独立编译：无球、无显式 pair、掩码全 0）
    bare = load_robot_model(profile, parameters)

    # 4. 环境构造（两个独立 model/data 实例）
    execution = ProfileSimEnv(scene)
    planning = ProfilePlanningEnv(bare)

    # 5. 执行器模式同步（规划与执行动力学一致）
    if task.actuator_mode == "position":
        kp = np.asarray(filled.kp, dtype=np.float64)
        kd = np.asarray(filled.kd, dtype=np.float64)
        execution.configure_actuator_mode("position", kp, kd)
        planning.configure_actuator_mode("position", kp, kd)

    # task 球初态编入 reset 基准（reset 重现发球初态）
    execution.set_initial_ball_state(
        task.initial_ball_position, task.initial_ball_velocity
    )

    logger.info(
        "R1D 环境对构造完成: execution(带球 nq=%d) + planning(裸 nq=%d) "
        "模式=%s 球初态=%s（robot_id: %s）",
        execution.model.nq,
        planning.model.nq,
        task.actuator_mode,
        np.asarray(task.initial_ball_position).tolist(),
        profile.robot_id,
    )
    return RobotEnvironments(execution=execution, planning=planning)
