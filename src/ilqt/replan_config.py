"""ReplanConfig — do_replan 的类型安全配置。

消除 MPCConfig → ``_build_replan_cfg()`` → 51-key dict 的机械翻译，
用类型安全 dataclass 替代 untyped dict。

设计要点：
- ``from_mpc_config()`` 工厂统一 MPCController / runner_factory 两条构建路径
- ``to_dict()`` 向后兼容：旧 do_replan 接口仍接受 dict，无需改动核心循环
- 字段集合严格对齐 ``MPCController._build_replan_cfg()`` 产出的 51 个键
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import TYPE_CHECKING, Any

import numpy as np

from src.ilqt.tube_types import TubeConfig

if TYPE_CHECKING:
    from src.ilqt.mpc_controller import MPCConfig


@dataclass
class ReplanConfig:
    """do_replan 的类型安全配置，替代 51-key dict。

    所有字段严格对应 ``MPCController._build_replan_cfg()`` 产出的 dict 键，
    确保 ``to_dict()`` 往返等价（消除翻译层的核心保证）。
    """

    # ── 时间 ──
    dt: float
    total_horizon: int
    fixed_horizon: int
    replan_interval: int

    # ── 迭代 ──
    max_iter_per_plan: int
    first_plan_iters: int
    near_plan_iters: int
    near_threshold: int

    # ── 代价标量 ──
    R: float
    Q_p_scale_far: float
    Q_v_scale_far: float
    Q_p_scale_near: float
    Q_v_scale_near: float
    normal_weight: float
    racket_speed: float
    max_tcp_speed: float

    # ── 代价基底（v12 对齐 V11 base_cost_fn，可空以兼容真机旧路径）──
    Q_p_base: np.ndarray | None
    Q_v_base: np.ndarray | None
    Q_qdot_base: float
    Q_qddot_base: float
    Q_du_base: float

    # ── 几何 ──
    shoulder_pos: np.ndarray
    workspace_radius: float

    # ── 方向与随挥 ──
    d_hat: np.ndarray
    d_follow: np.ndarray
    v_hit_desired: np.ndarray
    v_hit_at_contact: np.ndarray
    hit_shift: float
    follow_through_length: float
    follow_through_steps: int
    follow_through_v_terminal: float

    # ── Tube ──
    tube_cfg: TubeConfig
    ablation_mode: str

    # ── 模式开关 ──
    is_position_mode: bool
    use_backswing: bool
    use_r_decay: bool
    r_decay_ratio: float
    fix_joint5_angle: float | None
    backswing_offset: float
    backswing_ratio: float
    normal_flip: bool

    # ── 扰动（实验用）──
    time_perturb_s: float
    space_perturb_m: float
    perturb_alpha_min: float

    # ── 分阶段调度 ──
    smooth_far: dict
    smooth_mid: dict
    smooth_near: dict
    k_hit_total: int

    # ── 远段阈值（V11 far_threshold：k_hit > 此值时用 JT 控制）──
    far_threshold: int

    # ── 外部对象（运行期注入，不参与值比较）──
    robot_limits: Any
    solver: Any

    @classmethod
    def from_mpc_config(
        cls,
        config: "MPCConfig",
        robot_limits: Any,
        solver: Any,
        d_hat: np.ndarray,
        v_hit_desired: np.ndarray,
        d_follow: np.ndarray | None = None,
    ) -> "ReplanConfig":
        """从 MPCConfig 构建 ReplanConfig。

        消除 ``MPCController._build_replan_cfg()`` 与
        ``runner_factory.build_replan_cfg()`` 中重复的 80 行字段翻译。

        Args:
            config: MPC 控制器配置（控制规划行为）。
            robot_limits: 关节约束 + 安全滤波器实例。
            solver: ILQTSolver 实例（C++ 或 Python 实现）。
            d_hat: 期望击球方向单位向量（来球反方向）。
            v_hit_desired: 期望击球时刻末端速度。
            d_follow: 随挥方向单位向量，None 时回退为 d_hat（与旧逻辑一致）。

        Returns:
            ReplanConfig 实例，可直接传给 do_replan 或经 to_dict() 转旧 dict。
        """
        return cls(
            # 时间与迭代
            dt=config.dt,
            total_horizon=config.total_horizon,
            fixed_horizon=config.fixed_horizon,
            replan_interval=config.replan_interval,
            max_iter_per_plan=config.max_iter_per_plan,
            first_plan_iters=config.first_plan_iters,
            near_plan_iters=config.near_plan_iters,
            near_threshold=config.near_threshold,
            # 代价标量
            R=config.R,
            Q_p_scale_far=config.Q_p_scale_far,
            Q_v_scale_far=config.Q_v_scale_far,
            Q_p_scale_near=config.Q_p_scale_near,
            Q_v_scale_near=config.Q_v_scale_near,
            normal_weight=config.normal_weight,
            racket_speed=config.racket_speed,
            max_tcp_speed=config.max_tcp_speed,
            # 代价基底
            Q_p_base=config.Q_p_base,
            Q_v_base=config.Q_v_base,
            Q_qdot_base=config.Q_qdot_base,
            Q_qddot_base=config.Q_qddot_base,
            Q_du_base=config.Q_du_base,
            # 几何
            shoulder_pos=config.shoulder_pos,
            workspace_radius=config.workspace_radius,
            # 方向与随挥（d_follow 默认 = d_hat，对齐旧 _build_replan_cfg）
            d_hat=d_hat,
            d_follow=d_follow if d_follow is not None else d_hat,
            v_hit_desired=v_hit_desired,
            v_hit_at_contact=v_hit_desired,
            hit_shift=config.follow_through_length,
            follow_through_length=config.follow_through_length,
            follow_through_steps=config.follow_through_steps,
            follow_through_v_terminal=config.follow_through_v_terminal,
            # Tube
            tube_cfg=config.tube_cfg,
            ablation_mode=config.ablation_mode,
            # 模式
            is_position_mode=config.is_position_mode,
            use_backswing=config.use_backswing,
            use_r_decay=config.use_r_decay,
            r_decay_ratio=config.r_decay_ratio,
            fix_joint5_angle=config.fix_joint5_angle,
            backswing_offset=config.backswing_offset,
            backswing_ratio=config.backswing_ratio,
            normal_flip=config.normal_flip,
            # 扰动
            time_perturb_s=config.time_perturb_s,
            space_perturb_m=config.space_perturb_m,
            perturb_alpha_min=config.perturb_alpha_min,
            # 调度
            smooth_far=config.smooth_far,
            smooth_mid=config.smooth_mid,
            smooth_near=config.smooth_near,
            k_hit_total=config.total_horizon,
            # 远段阈值
            far_threshold=config.far_threshold,
            # 外部对象
            robot_limits=robot_limits,
            solver=solver,
        )

    def to_dict(self) -> dict:
        """转为 51-key dict（向后兼容 do_replan 旧 dict 接口）。

        do_replan 入口的兼容层会调用本方法，使 ReplanConfig 对象透明地
        适配旧 cfg["..."] 访问模式，无需改动核心规划循环。

        Returns:
            dict，键集与旧 ``_build_replan_cfg()`` 产出完全一致。
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}
