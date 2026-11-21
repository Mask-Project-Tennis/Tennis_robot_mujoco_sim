"""V12 episode 层共享类型 — 闭环运行结果。

本模块是 :class:`V12EpisodeResult` 的**唯一类定义**所在；旧路径
``src.sim.v12_episode`` 只做同一对象重导出（``is`` 相同）。``config`` 字段
从 :mod:`src.sim.configuration.types` 导入 :class:`ResolvedV12Config`，禁止在
本包重复定义配置类型。

当前阶段（P1）只迁移类型：episode 执行与产物写盘仍在
``src.sim.v12_episode``，后续由 B2 任务迁入本包。

设计约束：

* 不 import ``src.sim.v12_episode`` / ``src.sim.v12_config``（避免循环依赖）；
* 不导入 MuJoCo、不启动环境、不做文件 IO；
* 字段顺序、默认工厂与数组语义与迁移前逐字一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.sim.configuration.types import ResolvedV12Config

__all__ = ["V12EpisodeResult"]


@dataclass(frozen=True)
class V12EpisodeResult:
    """V12 共享闭环管线的完整结果（只含可 JSON 序列化数据）。

    Attributes:
        config: 本次运行的解析配置（唯一配置来源）。
        robot_id: 机器人标识 ``"rm65"`` / ``"unitree_r1d"``。
        mode: 执行器模式 ``"torque"`` / ``"position"``。
        joint_names: 主动关节名（长度 n）。
        dt: 仿真步长 [s]。
        racket_id: 球拍标识（RM65 为 None）。
        X: 主动臂状态序列 ``(N+1, 2n)``（``[q, qdot]``）。
        U: 控制序列 ``(N, n)``。
        ball_pos: 球位置序列 ``(N+1, 3)``。
        ball_vel: 球速度序列 ``(N+1, 3)``。
        time: 时间序列 ``(N+1,)``。
        racket_pos: 拍心位置序列 ``(N+1, 3)``（可视化用）。
        hit_type: ``"active"`` / ``"passive"`` / ``"miss"``。
        hit_step: 发生接触的执行步（未命中为 -1）。
        active_contact: 是否主动击球（拍面接触速度 > 阈值）。
        passive_contact: 是否被动接触。
        v_ee_at_hit: 接触时末端速度 [m/s]（未命中 None）。
        p_ee_at_hit: 接触时末端位置 ``(3,)``（未命中 None）。
        ball_pos_at_hit: 接触时球位置 ``(3,)``（未命中 None）。
        ball_out_speed: 击球出球速度 [m/s]（未命中 None）。
        ball_out_vy: 击球出球 y 速度 [m/s]（未命中 None）。
        metrics: 指标字典（耗时/后端/失败原因/几何/直方图等，JSON 可序列化）。
        model_dependency_manifest: 模型/资产依赖清单（I1 接线；默认空列表，
            保持旧构造点与旧 artifact 兼容）。
    """

    config: ResolvedV12Config
    robot_id: str
    mode: str
    joint_names: tuple[str, ...]
    dt: float
    racket_id: str | None
    X: np.ndarray
    U: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    time: np.ndarray
    racket_pos: np.ndarray
    hit_type: str
    hit_step: int
    active_contact: bool
    passive_contact: bool
    v_ee_at_hit: float | None
    p_ee_at_hit: np.ndarray | None
    ball_pos_at_hit: np.ndarray | None
    ball_out_speed: float | None
    ball_out_vy: float | None
    metrics: dict[str, Any] = field(default_factory=dict)
    model_dependency_manifest: list[dict[str, Any]] = field(default_factory=list)
