"""Tube-based robust hitting 框架的纯数据结构定义。

本模块包含 Tube/MPC 重规划所需的 dataclass，无任何业务逻辑依赖，
可被 tube_builder、tube_cost、replan_core 以及仿真主脚本（V11）/真机 runner 共享复用。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TubeConfig:
    """Tube-based robust hitting 配置参数。"""

    window_half_ms: float = 50.0
    """候选窗口半宽（毫秒），以 best_k 为中心前后扩展。"""

    contact_offset: float = 0.0
    """球拍接触点偏移（米），球拍中心到击球面的偏移。"""

    Q_p_tube: float = 50000.0
    """Tube 位置代价权重。"""

    Q_v_tube: float = 200.0
    """Tube 速度代价权重。"""

    Q_n_tube: float = 100000.0
    """Tube 法向量代价权重。"""

    tube_cost_mode: str = "weighted_sum"
    """Tube 代价聚合模式: 'weighted_sum' 或 'softmin'（暂仅支持 weighted_sum）。"""

    tube_cost_ratio: float = 1.0
    """Tube 代价占总代价的比例（0~1），剩余来自原 HittingCost 终端代价。"""

    softmin_beta: float = 5.0
    """终端 softmin 锐度参数 β。β 越大越接近 hard-min（只选最优候选），
    β 越小越接近均匀平均（所有候选等权）。建议范围 1.0~20.0。"""

    use_softmin_terminal: bool = True
    """是否启用多终端 softmin 代价（P0-2 改进）。
    True: 终端代价在多个候选位置上 softmin，容忍时间不确定性。
    False: 仅在 best_k 单点终端代价（原始行为）。"""


@dataclass
class BallTrajectoryTube:
    """带不确定性半径的球轨迹管道。"""

    positions: np.ndarray
    """球位置轨迹，形状 (N, 3)。"""

    velocities: np.ndarray
    """球速度轨迹，形状 (N, 3)。"""

    times: np.ndarray
    """时间序列，形状 (N,)。"""


@dataclass
class HitWindow:
    """候选击球时间窗口。"""

    best_k: int
    """最佳击球步数（find_hitting_point_physics 的结果）。"""

    k_candidates: np.ndarray
    """候选击球步数列表。"""

    p_ball_candidates: np.ndarray
    """候选时刻球位置，形状 (M, 3)。"""

    v_ball_candidates: np.ndarray
    """候选时刻球速度，形状 (M, 3)。"""

    weights: np.ndarray
    """候选权重（高斯衰减），形状 (M,)。"""


@dataclass
class HittingTube:
    """击球管道：包含多个候选时刻的期望球拍状态。"""

    k_candidates: np.ndarray
    """候选击球步数列表。"""

    p_racket_des: np.ndarray
    """期望球拍中心位置，形状 (M, 3)。"""

    v_racket_des: np.ndarray
    """期望球拍速度，形状 (M, 3)。"""

    n_racket_des: np.ndarray
    """期望球拍法向量，形状 (M, 3)。"""

    p_ball: np.ndarray
    """候选时刻球位置，形状 (M, 3)。"""

    v_ball: np.ndarray
    """候选时刻球速度，形状 (M, 3)。"""

    weights: np.ndarray
    """候选权重，形状 (M,)。"""

    best_k: int


@dataclass
class ReplanState:
    """重规划所需的可变状态（主线程和后台线程共享的快照）。"""
    k_hit_new: int = 0
    p_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ball_hit_new: np.ndarray = field(default_factory=lambda: np.zeros(3))
    current_n_des: np.ndarray = field(default_factory=lambda: np.zeros(3))
    U_prev: np.ndarray = field(default_factory=lambda: np.zeros((0, 6)))
    is_first_plan: bool = True
    hitting_tube: object = None
    cost_fn_type: str = ""
