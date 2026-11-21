"""机器人模型构建层共享类型 — 模型组装结果与球拍装配事实。

本模块是 :class:`RacketAssembly` 与 :class:`RobotModelBundle` 的**唯一类定义**
所在；旧路径 ``src.robot.model_factory`` 只做同一对象重导出（``is`` 相同）。

当前阶段（P1）只迁移类型定义：模型加载、XML 模板生成与依赖清单收集仍在
``src.robot.model_factory``，后续由 A1/A2 任务迁入本包。

设计约束：

* 不在导入期加载 MuJoCo 模型、不读配置文件、不起线程；
* ``mujoco`` 仅在 ``TYPE_CHECKING`` 下导入，运行期注解保持字符串；
* 不反向 import ``src.robot.model_factory``，避免与旧 facade 循环依赖。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.robot.model_binding import ModelBinding
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import RobotRuntimeParameters

if TYPE_CHECKING:
    import mujoco

__all__ = ["RacketAssembly", "RobotModelBundle"]


@dataclass(frozen=True)
class RacketAssembly:
    """球拍装配事实（RacketProfile → 模型 body 后的独立核对记录）。

    由工厂在装配时按档案推导，供 inspect 报告与上层（T4/T5）消费；字段
    全部为只读 float64 数组，不随模型对象生命周期变化。

    Attributes:
        racket_id: 档案标识（如 ``rm65_legacy`` / ``blade100_v9_reference``）。
        source_path: 档案文件路径。
        total_mass: 拍体总质量（kg，= 球拍质量 + 连接件质量，连接件只计一次）。
        com: 拍体质心在拍体局部系中的位置（m）。
        inertia_diagonal: 绕质心的主惯量（kg·m²，升序）。
        face_center: 拍面碰撞中心（拍体局部系，m）。
        face_semi_axes: 拍面碰撞椭球半轴（m，第一项为法向半厚）。
        mount_position: 法兰系中柄底位置（m）。
        mount_wxyz: 拍体局部系在法兰系中的姿态（wxyz 单位四元数）。
        connector_mass: 连接件质量（kg；无连接件为 0）。
    """

    racket_id: str
    source_path: Path
    total_mass: float
    com: np.ndarray
    inertia_diagonal: np.ndarray
    face_center: np.ndarray
    face_semi_axes: np.ndarray
    mount_position: np.ndarray
    mount_wxyz: np.ndarray
    connector_mass: float


@dataclass(frozen=True)
class RobotModelBundle:
    """同源组装结果 — 描述、参数、模型与绑定出自同一次配置。

    Attributes:
        profile: 机器人描述（身份与语义名称）。
        parameters: 运行参数；q_lower/q_upper 已从模型 jnt_range 填充。
        model: 按 parameters 组装并加载的 MuJoCo 模型（每次调用独立实例）。
        binding: model 的名称 → 地址绑定结果。
        racket: 球拍档案装配事实；旧 API（未提供档案）为 None。
        dependency_manifest: 本次装配**实际读取**的文件闭包（模型 XML、
            ``<include>`` 片段、对侧臂片段、mesh/texture）。每条含
            ``role/relative_path/source/sha256/size``，XML 片段另含
            ``content_base64``；由 :func:`collect_model_dependencies` 生成，
            供产物快照完整记录装配依赖（F08）。空元组表示未记录。
    """

    profile: RobotProfile
    parameters: RobotRuntimeParameters
    model: mujoco.MjModel
    binding: ModelBinding
    racket: RacketAssembly | None = None
    dependency_manifest: tuple[dict[str, Any], ...] = ()
