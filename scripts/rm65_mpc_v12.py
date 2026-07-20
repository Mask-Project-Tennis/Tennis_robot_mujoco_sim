#!/usr/bin/env python3
"""RM-65 V12 — EpisodeRunner 管线架构。

V11 的规划逻辑提取到 MPCController（src/ilqt/mpc_controller.py），
仿真执行/诊断封装到 SimComponent，
管线编排由 EpisodeRunner（src/ilqt/episode_runner.py）驱动。
V12 是组装器 + 球生成 + 评估。

=== V12 vs V11 ===
  - 规划逻辑（~800 行 replan + buffer + follow-through）→ MPCController
  - 安全滤波（~130 行 X-wall + safety filter）→ PredictiveSafetyFilter
  - 主循环编排 → EpisodeRunner.run()
  - 仿真专属（碰撞/接触/PD推回/诊断指标）→ SimComponent
  - argparse / 配置 / 球生成 / 评估 → 保留在 V12

用法（与 V11 一致）:
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --no-plot
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 2 --viewer --position-mode --no-r-decay
"""

from __future__ import annotations

import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.sim.rm65_env import RM65Env
from src.tennis.ball import (
    generate_ball_to_target_box,
    generate_ball_from_serve_box,
)
from src.tennis.hitting import find_hitting_point_physics
from src.ilqt.robot_limits import RobotLimits
from src.ilqt.tube_types import TubeConfig, HitWindow, HittingTube
from src.ilqt.tube_builder import search_hit_window, build_hitting_tube
from src.ilqt.mpc_controller import MPCController, MPCConfig
from src.ilqt.episode_runner import EpisodeRunner
from src.ilqt.components.sim_perception import SimPerception
from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
from src.ilqt.components.sim_component import SimComponent
from src.real.trajectory_recorder import TrajectoryRecorder
from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS, INIT_Q, INIT_Q_LEFT
from src.utils.yaml_utils import load_config, merge_configs

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("src.ilqt.robot_limits").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 可视化（从 V11 复用）
# ==============================================================================

def _import_v11_visuals():
    """惰性导入可视化函数（避免无 --viewer/--plot 时的导入开销）。"""
    from src.sim.v11_visuals import visualize_rm65_result, plot_tube_results
    return visualize_rm65_result, plot_tube_results


# ==============================================================================
# 主函数
# ==============================================================================

def main() -> None:
    """RM-65 V12 EpisodeRunner 管线架构主函数。"""
    # ==========================================================================
    # 1. 参数解析（与 V11 一致）
    # ==========================================================================
    parser = argparse.ArgumentParser(description="RM-65 V12 EpisodeRunner 管线架构")
    parser.add_argument("--viewer", action="store_true", help="计算完成后以真实速度回放")
    parser.add_argument("--seed", type=int, default=None, help="随机种子")
    parser.add_argument("--fd", action="store_true", help="使用有限差分线性化")
    parser.add_argument("--horizon", type=int, default=None, help="短地平线步数")
    parser.add_argument("--iter", type=int, default=None, help="每次重规划迭代数")
    parser.add_argument("--fix-joint5", action="store_true", help="固定第 6 关节")
    parser.add_argument("--backswing", type=float, default=0.6, help="后摆幅度 (rad)")
    parser.add_argument("--bs-ratio", type=float, default=0.35, help="后摆占比")
    parser.add_argument("--use-backswing", action="store_true", help="启用后摆（v11 默认无后摆）")
    parser.add_argument("--r-decay", type=float, default=0.40, help="R 衰减占比")
    parser.add_argument("--no-r-decay", action="store_true", help="禁用 R 退火")
    parser.add_argument("--hit-shift", type=float, default=0.0, help="随挥偏移距离 (m)")
    parser.add_argument("--ball-speed", type=float, default=None, help="球到达击打点时水平速度 (m/s)")
    parser.add_argument("--ball-distance", type=float, default=None, help="球起始位置到击打点的直线距离 (m)")
    parser.add_argument("--approach-angle", type=float, default=0.0, help="球飞来方向角 (度)")
    parser.add_argument("--serve-box", action="store_true", help="使用长方体发球区模式")
    parser.add_argument("--no-bounce", action="store_true", help="禁用地面弹跳")
    parser.add_argument("--serve-distance", type=float, default=8.0, help="发球区 Y 方向距离 (m)")
    parser.add_argument("--serve-height", type=float, default=1.2, help="发球区中心高度 (m)")
    parser.add_argument("--serve-x-size", type=float, default=8.0, help="发球区 X 轴全长 (m)")
    parser.add_argument("--serve-y-size", type=float, default=0.2, help="发球区 Y 轴全长 (m)")
    parser.add_argument("--serve-z-size", type=float, default=0.3, help="发球区 Z 轴全长 (m)")
    parser.add_argument("--Q-tcp-soft", type=float, default=0.0, help="TCP 速度软惩罚权重")
    parser.add_argument("--Q-qdot-limit", type=float, default=0.0, help="关节速度软惩罚权重")
    parser.add_argument("--normal-weight", type=float, default=500000.0, help="拍面法向量代价权重")
    parser.add_argument("--normal-flip", action="store_true", help="翻转法向量方向")
    parser.add_argument("--replan-interval", type=int, default=None, help="重规划间隔步数")
    parser.add_argument("--near-iters", type=int, default=None, help="near阶段 iLQR 迭代次数")
    parser.add_argument("--window-ms", type=float, default=50.0, help="Tube 候选窗口半宽 (ms)")
    parser.add_argument("--softmin-beta", type=float, default=5.0, help="终端 softmin 锐度 β")
    parser.add_argument("--ablation", choices=["full", "tube_only", "softmin_only", "none"],
                        default=None, help="消融模式")
    parser.add_argument("--no-softmin", action="store_true", help="[已废弃] 禁用 softmin")
    parser.add_argument("--no-tube", action="store_true", help="[已废弃] 禁用 Tube 走廊")
    parser.add_argument("--no-follow-through", action="store_true", help="禁用球轨迹回溯随挥")
    parser.add_argument("--follow-trigger", choices=["planned", "contact"], default="planned",
                        help="随挥触发方式")
    parser.add_argument("--no-v-maximize", action="store_true", help="禁用中点速度最大化")
    parser.add_argument("--fixed-direction", action="store_true", help="使用固定 YAML hit_direction")
    parser.add_argument("--target-speed", type=float, default=1.8, help="终端目标速度 (m/s)")
    parser.add_argument("--no-plot", action="store_true", help="禁用 matplotlib 可视化")
    parser.add_argument("--dump-trajectory", type=str, default=None, help="保存轨迹到 npz 文件（可被 replay_trajectory.py 加载）")
    parser.add_argument("--realtime", action="store_true", help="模拟实时节奏")
    parser.add_argument("--async-replan", action="store_true", help="启用异步重规划")
    parser.add_argument("--time-perturb-ms", type=float, default=0.0, help="球到达时间预测扰动 (ms)")
    parser.add_argument("--space-perturb-m", type=float, default=0.0, help="击打点空间偏移 (m)")
    parser.add_argument("--perturb-alpha-min", type=float, default=0.0, help="衰减扰动保底值")
    parser.add_argument("--random-perturb", action="store_true", help="随机扰动")
    parser.add_argument("--perturb-sign", choices=["random", "positive", "negative"],
                        default="random", help="扰动符号方向")
    parser.add_argument("--time-perturb-min-ms", type=float, default=0.0, help="随机扰动时间下限 (ms)")
    parser.add_argument("--space-perturb-min-m", type=float, default=0.0, help="随机扰动空间下限 (m)")
    parser.add_argument("--ball-speed-perturb-pct", type=float, default=0.0, help="球速耦合扰动百分比")
    parser.add_argument("--max-tcp", type=float, default=None, help="TCP 线速度硬限制 (m/s)")
    parser.add_argument("--terminal-exempt-steps", type=int, default=None, help="终段豁免步数（默认沿用 RobotLimits 配置值 20）")
    parser.add_argument("--sim-limits", action="store_true",
        help="(已废弃) 仿真限位现为默认行为，此标志无效")
    parser.add_argument("--limits-config", type=Path, default=None,
        help="限位 YAML(RealRobotConfig 格式). 优先级高于 --sim-limits")
    parser.add_argument("--obs-freq", type=float, default=0, help="观测频率 Hz")
    parser.add_argument("--obs-noise-pos", type=float, default=0, help="观测位置噪声 std (m)")
    parser.add_argument("--obs-noise-vel", type=float, default=0, help="观测速度噪声 std (m/s)")
    parser.add_argument("--obs-use-kf", action="store_true", help="观测启用 KF 滤波")
    parser.add_argument("--position-mode", action="store_true", help="启用位置控制模式")
    parser.add_argument("--no-feedforward", action="store_true", help="禁用前馈补偿")
    parser.add_argument("--kp", type=float, nargs='+', default=None, help="位置模式 PD Kp")
    parser.add_argument("--kd", type=float, nargs='+', default=None, help="位置模式 PD Kd")
    parser.add_argument("--dq-max-fraction", type=float, default=None, help="单步角度变化系数")
    # ── 阶段调度参数（原硬编码）──
    parser.add_argument("--far-threshold", type=int, default=None, help="far/mid 阶段边界步数")
    parser.add_argument("--near-threshold", type=int, default=None, help="mid/near 阶段边界步数（0=自动）")
    parser.add_argument("--first-plan-iters", type=int, default=None, help="首次规划 iLQR 迭代数")
    # ── 代价阶段倍率（原硬编码）──
    parser.add_argument("--Qp-scale-far", type=float, default=None, help="far 阶段位置代价倍率")
    parser.add_argument("--Qv-scale-far", type=float, default=None, help="far 阶段速度代价倍率")
    parser.add_argument("--Qp-scale-near", type=float, default=None, help="near 阶段位置代价倍率")
    parser.add_argument("--Qv-scale-near", type=float, default=None, help="near 阶段速度代价倍率")
    # ── 随挥 PD 增益（原硬编码在 follow_through.py）──
    parser.add_argument("--follow-kp", type=float, default=None, help="随挥 PD 比例增益")
    parser.add_argument("--follow-kd", type=float, default=None, help="随挥 PD 微分增益")
    # ── HitPointRefiner 阈值（原硬编码在 hit_point_refiner.py）──
    parser.add_argument("--hit-lock", type=int, default=None, help="击球点防抖锁定步数")
    parser.add_argument("--hard-margin", type=float, default=None, help="IK 硬裕度 (度)")
    parser.add_argument("--warn-margin", type=float, default=None, help="IK 警告裕度 (度)")
    parser.add_argument("--j1-warn", type=float, default=None, help="关节1 警告裕度 (度)")
    parser.add_argument("--refiner-window", type=int, default=None, help="Refiner 搜索窗口半宽 (步)")
    args = parser.parse_args()

    # ==========================================================================
    # 2. 消融模式推导 + 随机扰动（与 V11 一致）
    # ==========================================================================
    if args.ablation is not None:
        ablation_mode = args.ablation
    else:
        if args.no_tube and args.no_softmin:
            ablation_mode = "none"
        elif args.no_tube:
            ablation_mode = "softmin_only"
        elif args.no_softmin:
            ablation_mode = "tube_only"
        else:
            ablation_mode = "full"
    logger.info(f"[ablation] mode={ablation_mode}")

    _use_tube = ablation_mode in ("full", "tube_only")
    need_candidates = ablation_mode in ("full", "tube_only", "softmin_only")
    time_perturb_s = args.time_perturb_ms / 1000.0
    space_perturb_m = args.space_perturb_m
    perturb_alpha_min = args.perturb_alpha_min

    if args.random_perturb:
        rng_perturb = np.random.default_rng(args.seed + 99999 if args.seed is not None else None)
        perturb_sign_cfg = args.perturb_sign
        tp_max = abs(args.time_perturb_ms)
        tp_min = abs(args.time_perturb_min_ms)
        if tp_max > 1e-9:
            tp_abs = rng_perturb.uniform(tp_min, tp_max)
            tp_sign = {"positive": 1.0, "negative": -1.0}.get(perturb_sign_cfg, rng_perturb.choice([-1.0, 1.0]))
            time_perturb_s = tp_sign * tp_abs / 1000.0
        else:
            time_perturb_s = 0.0
        sp_max = abs(args.space_perturb_m)
        sp_min = abs(args.space_perturb_min_m)
        if sp_max > 1e-9:
            sp_abs = rng_perturb.uniform(sp_min, sp_max)
            sp_sign = {"positive": 1.0, "negative": -1.0}.get(perturb_sign_cfg, rng_perturb.choice([-1.0, 1.0]))
            space_perturb_m = sp_sign * sp_abs
        else:
            space_perturb_m = 0.0

    # ==========================================================================
    # 3. 配置加载（与 V11 一致）
    # ==========================================================================
    base_path = Path(__file__).resolve().parent.parent / "configs"
    config_dict = load_config(base_path / "default.yaml")
    v5_config_path = base_path / "v5_active_hit.yaml"
    if v5_config_path.exists():
        config_dict = merge_configs(config_dict, load_config(v5_config_path))
    mpc_config_path = base_path / "mpc.yaml"
    if mpc_config_path.exists():
        config_dict = merge_configs(config_dict, load_config(mpc_config_path))

    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)

    if args.no_follow_through:
        config_dict["hitting"]["follow_through_steps"] = 0
        config_dict["hitting"]["follow_through_length"] = 0.0
        config_dict["hitting"]["follow_through_v_terminal"] = 0.0

    shoulder_pos = SHOULDER_POS
    workspace_radius = WORKSPACE_RADIUS

    # ==========================================================================
    # 4. 时间参数推导（与 V11 一致）
    # ==========================================================================
    total_horizon = 200
    fixed_horizon = 40
    replan_interval = args.replan_interval if args.replan_interval is not None else 30
    max_iter_per_plan = 5
    first_plan_iters = args.first_plan_iters if args.first_plan_iters is not None else 15
    near_plan_iters = 20
    Q_p_scale_far = args.Qp_scale_far if args.Qp_scale_far is not None else 5.0
    Q_v_scale_far = args.Qv_scale_far if args.Qv_scale_far is not None else 3.0
    Q_p_scale_near = args.Qp_scale_near if args.Qp_scale_near is not None else 8.0
    Q_v_scale_near = args.Qv_scale_near if args.Qv_scale_near is not None else 120.0

    if args.near_iters is not None:
        near_plan_iters = args.near_iters
    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter

    if args.serve_box:
        if args.horizon is None:
            fixed_horizon = 120
        if args.iter is None:
            max_iter_per_plan = 10
        if args.replan_interval is None:
            replan_interval = 20
        if args.near_iters is None:
            near_plan_iters = 5
        first_plan_iters = max(first_plan_iters, 30)
        total_horizon = max(total_horizon, 250)
        logger.info(f"serve-box auto params: horizon={fixed_horizon}, iter={max_iter_per_plan}, "
                     f"first_plan_iters={first_plan_iters}, total_horizon={total_horizon}, "
                     f"replan_interval={replan_interval}, near_plan_iters={near_plan_iters}")

    # ==========================================================================
    # 5. 关节 + 执行器配置（与 V11 一致）
    # ==========================================================================
    init_q = INIT_Q
    init_q_left = INIT_Q_LEFT
    fix_joint5_angle: float | None = init_q[5] if args.fix_joint5 else None
    use_backswing = args.use_backswing
    backswing_offset = -abs(args.backswing)
    backswing_ratio = args.bs_ratio
    use_r_decay = not args.no_r_decay
    r_decay_ratio = args.r_decay

    actuator_cfg = config_dict.get("actuator", {})
    is_position_mode = args.position_mode or actuator_cfg.get("mode", "torque") == "position"
    if is_position_mode:
        kp_cfg = np.array(actuator_cfg.get("kp", [200.0, 200.0, 200.0, 50.0, 50.0, 20.0]), dtype=np.float64)
        kd_cfg = np.array(actuator_cfg.get("kd", [20.0, 20.0, 5.0, 5.0, 5.0, 2.0]), dtype=np.float64)
        if args.kp is not None:
            kp_cfg = np.full(6, args.kp[0], dtype=np.float64) if len(args.kp) == 1 else np.array(args.kp, dtype=np.float64)
        if args.kd is not None:
            kd_cfg = np.full(6, args.kd[0], dtype=np.float64) if len(args.kd) == 1 else np.array(args.kd, dtype=np.float64)
        use_r_decay = False

    # Tube 配置
    tube_cfg = TubeConfig(
        window_half_ms=args.window_ms,
        Q_p_tube=500.0,
        Q_v_tube=0.0,
        Q_n_tube=0.0,
        tube_cost_ratio=1.0,
        softmin_beta=args.softmin_beta,
        use_softmin_terminal=ablation_mode in ("full", "softmin_only"),
    )

    # ==========================================================================
    # 6. 创建环境 + RobotLimits（与 V11 一致）
    # ==========================================================================
    model_path = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = init_q_left
    if is_position_mode:
        env.configure_actuator_mode("position", kp=kp_cfg, kd=kd_cfg)
        use_ff_cfg = actuator_cfg.get("feedforward", True)
        if args.no_feedforward or not use_ff_cfg:
            env.configure_feedforward(False)

    # ── 构建关节限位 ──
    # 默认: 仿真限位 (default.yaml, TCP 1.8 m/s)
    #   网球挥拍末端速度 ~1.2 m/s，TCP 1.8 是物理下限。
    #   TCP 1.0 会锁死安全滤波加速 → 命中率 30%（exp16 验证）。
    # 真机部署: --limits-config configs/real_robot.yaml
    from src.real.config import RealRobotConfig as _RRC
    from src.real.runner_factory import build_robot_limits as _brl
    if args.limits_config is not None:
        try:
            _real_cfg = _RRC.from_yaml(args.limits_config)
        except (FileNotFoundError, ValueError) as e:
            parser.error(f"--limits-config 加载失败: {e}")
        robot_limits = _brl(env, _real_cfg)
        # 真机限位模式：切换到真机初始姿势（J2=80° 有 10° 裕度，vs INIT_Q J2=90° 在限位边界）
        from src.robot.constants import INIT_Q_REAL
        init_q = INIT_Q_REAL.copy()
        logger.info("真机限位模式: init_q 切换到 INIT_Q_REAL (J2=%.1f°)", np.degrees(init_q[1]))
        # I5: 未显式 --max-tcp 警告。real_robot.yaml 的 TCP=1.0 会锁死仿真安全滤波加速，
        # 须用 --max-tcp 1.8 覆盖（exp16: TCP 1.0 命中率 30% vs 1.8 = 78%）。
        if args.max_tcp is None:
            logger.warning(
                "--limits-config 加载真机限位（TCP=%.1f m/s），仿真生成轨迹需显式 --max-tcp 1.8，"
                "当前未指定，安全滤波可能锁死加速 → 轨迹无效。建议追加 --max-tcp 1.8",
                _real_cfg.max_tcp_speed,
            )
    else:
        rl_cfg = config_dict.get("robot_limits", {})
        robot_limits = RobotLimits.from_config(
            rl_cfg, dt=dt, ctrlrange=env.model.actuator_ctrlrange[:env.NU])

    # CLI 覆盖（所有模式通用）
    if args.max_tcp is not None:
        robot_limits.max_tcp_speed = float("inf") if args.max_tcp == 0 else args.max_tcp
    if args.terminal_exempt_steps is not None:
        robot_limits.terminal_exempt_steps = args.terminal_exempt_steps
    if args.dq_max_fraction is not None:
        robot_limits.dq_max = robot_limits.qdot_max * dt * args.dq_max_fraction

    logger.info(
        "关节限位: %s, TCP=%.1f m/s, terminal_exempt=%d",
        f"自定义({args.limits_config})" if args.limits_config else "仿真默认(default.yaml)",
        robot_limits.max_tcp_speed,
        robot_limits.terminal_exempt_steps,
    )
    if args.sim_limits:
        logger.warning("--sim-limits 已废弃（仿真限位现为默认行为）")

    # X 平面墙 body IDs 缓存（引用共享常量，消除重复）
    from src.ilqt.components.predictive_safety import X_WALL_BODY_NAMES
    import mujoco as _mj
    _hard_x_body_ids = [
        _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_BODY, n)
        for n in X_WALL_BODY_NAMES
    ]

    # ==========================================================================
    # 7. 生成发球轨迹 + 寻找击打点（与 V11 一致，仿真专属）
    # ==========================================================================
    rng = np.random.default_rng(args.seed)
    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    p_racket_init = env.get_ee_pos().copy()
    logger.info(f"球拍初始位置: {p_racket_init}")

    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.10
    use_bounce = not args.no_bounce

    if args.serve_box:
        serve_half_x = args.serve_x_size / 2.0
        serve_half_y = args.serve_y_size / 2.0
        serve_half_z = args.serve_z_size / 2.0
        p0, v0, p_hit_expected = generate_ball_from_serve_box(
            serve_box_center=(0.0, -args.serve_distance, args.serve_height),
            serve_box_halfsize=(serve_half_x, serve_half_y, serve_half_z),
            target_center=target_center,
            target_offset=target_offset,
            shoulder_pos=shoulder_pos,
            workspace_radius=workspace_radius,
            g=g,
            ball_speed=args.ball_speed,
            speed_range=(8.0, 18.0),
            use_bounce=use_bounce,
            bounce_restitution=0.75,
            rng=rng,
        )
    else:
        hit_time = total_horizon * dt * rng.uniform(0.3, 0.4)
        p0, v0, p_hit_expected = generate_ball_to_target_box(
            target_center, target_offset, hit_time, g,
            shoulder_pos=shoulder_pos, workspace_radius=workspace_radius,
            ball_speed=args.ball_speed,
            ball_distance=args.ball_distance,
            approach_angle_deg=args.approach_angle,
            rng=rng,
            ball_direction="y",
            ball_start_y_range=(-5.5, -4.5),
            ball_start_z_range=(1.4, 1.8),
        )

    hit_info = find_hitting_point_physics(
        env, p0, v0, shoulder_pos, workspace_radius, total_horizon
    )
    if hit_info is None:
        print("\n========================================")
        print("  网球不在工作空间内，机械臂不击打！")
        print("========================================\n")
        return

    k_hit_total = hit_info["k_hit"]
    p_hit = hit_info["p_hit"]
    v_ball_hit = hit_info["v_ball_hit"]

    # IK 可达性后过滤
    ball_positions_all, ball_velocities_all = env.predict_ball_trajectory(p0, v0, total_horizon)
    q_ik_init = env.solve_ik(p_hit, q_init=init_q, max_iter=50, eps=1e-2)
    m_low_deg = (q_ik_init - robot_limits.q_lower) * 180.0 / np.pi
    m_up_deg = (robot_limits.q_upper - q_ik_init) * 180.0 / np.pi
    min_margin_deg = float(np.min(np.minimum(m_low_deg, m_up_deg)))
    if min_margin_deg < 3.0:
        search_range = 30
        best_alt_k = k_hit_total
        best_alt_margin = min_margin_deg
        for dk in range(-search_range, search_range + 1):
            kk = k_hit_total + dk
            if kk < 1 or kk > len(ball_positions_all):
                continue
            p_alt = ball_positions_all[kk - 1]
            dist_alt = np.linalg.norm(p_alt - shoulder_pos)
            dz = p_alt[2] - shoulder_pos[2]
            if not (dist_alt < workspace_radius and p_alt[2] > 0.3 and -0.60 < dz < 0.55):
                continue
            q_alt = env.solve_ik(p_alt, q_init=init_q, max_iter=30, eps=2e-2)
            m_low_a = (q_alt - robot_limits.q_lower) * 180.0 / np.pi
            m_up_a = (robot_limits.q_upper - q_alt) * 180.0 / np.pi
            m_a = float(np.min(np.minimum(m_low_a, m_up_a)))
            if m_a > best_alt_margin:
                best_alt_margin = m_a
                best_alt_k = kk
        if best_alt_k != k_hit_total:
            p_hit = ball_positions_all[best_alt_k - 1].copy()
            v_ball_hit = ball_velocities_all[best_alt_k - 1].copy()
            k_hit_total = best_alt_k

    n_des_single = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    if args.normal_flip:
        n_des_single = -n_des_single

    near_threshold = args.near_threshold if args.near_threshold else max(50, k_hit_total // 3)
    far_threshold = args.far_threshold if args.far_threshold is not None else 50

    hit_direction = np.array(config_dict["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config_dict["hitting"]["racket_speed"])

    if args.fixed_direction:
        d_follow = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)
    else:
        d_follow = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    v_hit_at_contact = args.target_speed * d_follow

    if args.no_follow_through:
        follow_through_length = 0.0
        follow_through_steps = 0
        follow_through_v_terminal = 0.0
    else:
        follow_through_length = args.hit_shift
        follow_through_steps = int(config_dict["hitting"].get("follow_through_steps", 160))
        follow_through_v_terminal = float(config_dict["hitting"].get("follow_through_v_terminal", 0.3))

    logger.info(f"击打步数: {k_hit_total}, 击打位置: {p_hit}")
    logger.info(f"随挥方向: {np.round(d_follow, 3)}, 随挥步数: {follow_through_steps}")

    # ==========================================================================
    # 8. 构建初始 Tube（仿真专属）
    # ==========================================================================
    initial_hitting_tube: HittingTube | None = None
    hit_window: HitWindow | None = None
    if need_candidates:
        hit_window = search_hit_window(
            env, p0, v0, shoulder_pos, workspace_radius,
            k_hit_total + 30, tube_cfg,
            ball_direction="y", current_step=0,
            robot_limits=robot_limits, init_q=init_q,
        )
        if hit_window is not None:
            initial_hitting_tube = build_hitting_tube(
                hit_window, racket_speed, d_follow, tube_cfg,
            )
            logger.info(
                f"候选窗口已构建: best_k={initial_hitting_tube.best_k}, "
                f"candidates={len(initial_hitting_tube.k_candidates)}"
            )

    # ==========================================================================
    # 9. 球速耦合扰动（与 V11 一致）
    # ==========================================================================
    ball_speed_perturb_pct = args.ball_speed_perturb_pct
    if abs(ball_speed_perturb_pct) > 0.01:
        v0_norm = np.linalg.norm(v0)
        if v0_norm > 0.1:
            v0_dir = v0 / v0_norm
            offset_m = ball_speed_perturb_pct / 100.0 * 2.0
            p0_real = p0 + v0_dir * offset_m
            v0_real = v0.copy()
        else:
            p0_real = p0
            v0_real = v0
    else:
        p0_real = p0
        v0_real = v0

    # ==========================================================================
    # 10. 重置环境 + 设置球的初始状态（与 V11 一致）
    # ==========================================================================
    env.reset(init_q)
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
    env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0_real, v0_real)

    x0 = np.zeros(env.NX)
    x0[:env.NQ] = init_q

    # ==========================================================================
    # 11. 构建 MPCConfig + MPCController
    # ==========================================================================
    # 代价基底（对齐 V11 base_cost_fn: config Q_p * 2.0）
    Q_p_base = np.array(config_dict["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v_base = np.array(config_dict["cost"]["Q_v"], dtype=np.float64) * 2.0

    # 分阶段权重
    stage_cfg = config_dict.get("stage_weights", {})
    far_stage = stage_cfg.get("far", {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0})
    mid_stage = stage_cfg.get("mid", {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0})
    near_stage = stage_cfg.get("near", {"Q_qdot_mult": 0.0, "Q_qddot_mult": 0.0, "Q_du_mult": 0.0})

    mpc_config = MPCConfig(
        version="v12",
        is_position_mode=is_position_mode,
        ablation_mode=ablation_mode,
        dt=dt,
        total_horizon=total_horizon,
        fixed_horizon=fixed_horizon,
        replan_interval=replan_interval,
        max_iter_per_plan=max_iter_per_plan,
        first_plan_iters=first_plan_iters,
        near_plan_iters=near_plan_iters,
        near_threshold=near_threshold,
        Q_p_scale_far=Q_p_scale_far,
        Q_v_scale_far=Q_v_scale_far,
        Q_p_scale_near=Q_p_scale_near,
        Q_v_scale_near=Q_v_scale_near,
        R=float(config_dict["cost"]["R"]),
        normal_weight=args.normal_weight,
        racket_speed=racket_speed,
        target_speed=args.target_speed,
        Q_tcp_soft=args.Q_tcp_soft,
        Q_qdot_limit=args.Q_qdot_limit,
        tube_cfg=tube_cfg,
        softmin_beta=args.softmin_beta,
        follow_through_length=follow_through_length,
        follow_through_steps=follow_through_steps,
        follow_through_v_terminal=follow_through_v_terminal,
        follow_trigger=args.follow_trigger,
        no_follow_through=args.no_follow_through,
        use_backswing=use_backswing,
        backswing_offset=backswing_offset,
        backswing_ratio=backswing_ratio,
        shoulder_pos=shoulder_pos,
        workspace_radius=workspace_radius,
        time_perturb_s=time_perturb_s,
        space_perturb_m=space_perturb_m,
        perturb_alpha_min=perturb_alpha_min,
        use_r_decay=use_r_decay,
        r_decay_ratio=r_decay_ratio,
        fix_joint5_angle=fix_joint5_angle,
        normal_flip=args.normal_flip,
        smooth_far=far_stage,
        smooth_mid=mid_stage,
        smooth_near=near_stage,
        async_mode=args.async_replan,
        # v12 新增：代价基底 + 远段阈值
        Q_p_base=Q_p_base,
        Q_v_base=Q_v_base,
        Q_qdot_base=float(config_dict["cost"].get("Q_qdot", 0.0)),
        Q_qddot_base=float(config_dict["cost"].get("Q_qddot", 0.0)),
        Q_du_base=float(config_dict["cost"].get("Q_du", 0.0)),
        far_threshold=far_threshold,
        # 策略内部参数（原硬编码，现可通过 CLI 调整）
        follow_kp=args.follow_kp if args.follow_kp is not None else 200.0,
        follow_kd=args.follow_kd if args.follow_kd is not None else 20.0,
        hit_lock_threshold=args.hit_lock if args.hit_lock is not None else 60,
        hard_margin_deg=args.hard_margin if args.hard_margin is not None else 2.0,
        warn_margin_deg=args.warn_margin if args.warn_margin is not None else 5.0,
        j1_warn_margin_deg=args.j1_warn if args.j1_warn is not None else 8.0,
        refiner_window_half=args.refiner_window if args.refiner_window is not None else 15,
    )

    mpc = MPCController(env, mpc_config, robot_limits=robot_limits)

    # ==========================================================================
    # 12. 组装管线组件
    # ==========================================================================
    # 观测门控
    obs_gate = None
    _obs_freq_eff = args.obs_freq if args.obs_freq > 0 else (
        1.0 / dt if (args.obs_noise_pos > 0 or args.obs_noise_vel > 0) else 0
    )
    if _obs_freq_eff > 0:
        from src.perception.ball_obs_gate import BallObservationGate
        from src.perception.ball_estimator import BallEstimator as _BE
        _obs_kf = None
        if args.obs_use_kf:
            _obs_kf = _BE(dt=dt, pos_noise_std=args.obs_noise_pos, vel_noise_std=args.obs_noise_vel)
        obs_gate = BallObservationGate(
            _obs_freq_eff, dt,
            noise_pos=args.obs_noise_pos, noise_vel=args.obs_noise_vel,
            kf=_obs_kf, rng=rng,
        )

    perception = SimPerception(env, obs_gate=obs_gate)
    safety = PredictiveSafetyFilter(
        env, robot_limits,
        is_position_mode=is_position_mode,
    )
    sim_component = SimComponent(
        env, mpc, robot_limits, init_q, is_position_mode,
        _hard_x_body_ids, initial_hitting_tube, tube_cfg, dt,
    )
    # 初始化 history（与 V11 一致：X_history[0] = x0）
    sim_component.X_history = [x0.copy()]
    sim_component.ball_pos_history = [p0.copy()]

    # 创建轨迹记录器（如果 --dump-trajectory 指定）
    recorder = None
    if args.dump_trajectory:
        recorder = TrajectoryRecorder(
            env=env,
            init_q=init_q,
            init_q_left=init_q_left,
            dt=dt,
            metadata={
                "p0": p0_real.tolist(),
                "v0": v0_real.tolist(),
                "ball_speed": args.ball_speed,
                "is_position_mode": is_position_mode,
            },
        )

    runner = EpisodeRunner(
        mpc=mpc,
        perception=perception,
        safety=safety,
        executor=sim_component,
        post_exec_hooks=[recorder.make_hook()] if recorder else [],
    )

    # ==========================================================================
    # 13. 运行 episode
    # ==========================================================================
    total_steps = total_horizon + follow_through_steps
    logger.info(f"开始 V12 EpisodeRunner，总步数={total_steps} "
                f"(MPC={total_horizon}, 随挥={follow_through_steps})，击打步数={k_hit_total}")

    t_total_start = time.perf_counter()
    _metrics = runner.run(max_steps=total_steps)
    t_mpc_end = time.perf_counter()

    # ==========================================================================
    # 14. 击打后继续仿真（V11 2225-2243，PD 保持 20 步）
    # ==========================================================================
    post_hit_steps = 20
    logger.info(f"击打后继续仿真 {post_hit_steps} 步...")
    x_current = env.get_arm_state()
    for i in range(post_hit_steps):
        q_hold = x_current[:env.NQ].copy()
        if is_position_mode:
            u_hold = q_hold.copy()
        else:
            u_hold = 100.0 * (q_hold - x_current[:env.NQ]) - 10.0 * x_current[env.NQ:]
            u_hold = np.clip(u_hold,
                             env.model.actuator_ctrlrange[:env.NU, 0],
                             env.model.actuator_ctrlrange[:env.NU, 1])
        x_current, ball_pos_post, _ = env.step_full(u_hold)
        sim_component.X_history.append(x_current.copy())
        sim_component.U_history.append(u_hold.copy())
        sim_component.ball_pos_history.append(ball_pos_post.copy())

        # 记录 post-hit 步到 TrajectoryRecorder（hook 不覆盖此段）
        if recorder is not None:
            step_idx = total_horizon + follow_through_steps + i
            recorder.record(
                q_desired=q_hold.copy(),
                q_actual=x_current[:env.NQ].copy(),
                timestamp=step_idx * dt,
                tcp_pos=env.get_ee_pos().copy(),
                ball_pos=ball_pos_post.copy(),
            )

    # ==========================================================================
    # 15. 评估（对齐 V11 2245-2505）
    # ==========================================================================
    t_total = time.perf_counter() - t_total_start
    t_mpc = t_mpc_end - t_total_start

    X_history = sim_component.X_history
    U_history = sim_component.U_history
    ball_pos_history = sim_component.ball_pos_history
    distances_history = sim_component.distances_history
    ball_near_history = sim_component.ball_near_history
    tube_ready_history = sim_component.tube_ready_history

    # 末端最终位置
    p_ee_at_hit = sim_component.p_ee_at_hit
    ball_pos_at_hit = sim_component.ball_pos_at_hit
    if p_ee_at_hit is not None:
        p_ee_final = p_ee_at_hit
    else:
        env.set_arm_state(x_current)
        env.update_kinematics()
        p_ee_final = env.get_ee_pos()
    v_ee_final = env.get_ee_vel()

    pos_error_plan = float(np.linalg.norm(p_ee_final - p_hit))
    if ball_pos_at_hit is not None:
        pos_error = float(np.linalg.norm(p_ee_final - ball_pos_at_hit))
    else:
        pos_error = pos_error_plan
    vel_error = float(np.linalg.norm(v_ee_final - v_hit_at_contact))

    # Tube 指标
    d_arr = np.array(distances_history)
    min_dist = float(np.min(d_arr))
    ball_near_duration = int(np.sum(np.array(ball_near_history, dtype=bool)))
    ball_near_ms = ball_near_duration * dt * 1000
    tube_ready_duration = int(np.sum(np.array(tube_ready_history, dtype=bool)))
    tube_ready_ms = tube_ready_duration * dt * 1000

    # 命中时刻误差
    hit_step = sim_component.hit_step
    if p_ee_at_hit is not None and hit_step >= 0:
        hit_time_actual = hit_step * dt
        hit_time_expected = k_hit_total * dt
        hit_time_error = abs(hit_time_actual - hit_time_expected) * 1000
        hit_position_error = (
            float(np.linalg.norm(p_ee_at_hit - ball_pos_at_hit))
            if ball_pos_at_hit is not None
            else float(np.linalg.norm(p_ee_at_hit - p_hit))
        )
    else:
        hit_time_error = 0.0
        hit_position_error = pos_error

    # 执行层指标
    max_tcp = sim_component.max_tcp_speed
    max_qdot = sim_component.max_qdot_ratio
    max_face = sim_component.max_racket_face_speed
    active_contact = sim_component.active_contact
    passive_contact = sim_component.passive_contact
    v_racket_at_hit_val = sim_component.v_ee_at_hit if sim_component.v_ee_at_hit is not None else 0.0

    # ---- 打印评估结果 ----
    from src.sim.hit_detection import classify_hit_result
    print("\n========================================")
    print(f"  {classify_hit_result(active_contact, passive_contact, pos_error)}")

    print("  [V12 EpisodeRunner 管线架构]")
    print(f"  ablation: {ablation_mode}")
    print(f"  击打目标位置: {np.round(p_hit, 3)}")
    print(f"  末端实际位置: {np.round(p_ee_final, 3)}")
    print(f"  规划跟踪误差: {pos_error_plan:.4f} m")
    print(f"  速度误差: {vel_error:.4f} m/s")
    print(f"  MPC 总步数: {len(U_history)}")
    print("  --- Tube 专用指标 ---")
    print(f"  最小球拍-球距离: {min_dist:.4f} m")
    print(f"  ball_near 步数: {ball_near_duration} = {ball_near_ms:.1f} ms")
    print(f"  tube_ready 步数: {tube_ready_duration} = {tube_ready_ms:.1f} ms")
    print(f"  击打时间误差: {hit_time_error:.1f} ms")
    print(f"  击打位置误差: {hit_position_error:.4f} m")
    print("  --- 计算性能 ---")
    print(f"  总墙钟时间: {t_total:.2f}s (MPC={t_mpc:.2f}s)")
    print("  --- 执行层约束 ---")
    hit_type = "主动击球" if active_contact else ("被动接触" if passive_contact else "未触球")
    print(f"  max_qdot={max_qdot:.2f}x, max_tcp={max_tcp:.1f}m/s, max_face={max_face:.1f}m/s")
    print(f"  击球类型: {hit_type}")
    print("========================================\n")

    # 结构化结果行
    hit_type_en = "active" if active_contact else ("passive" if passive_contact else "miss")
    print(f"__RESULT__: pos_error={pos_error:.6f} vel_error={vel_error:.6f} "
          f"min_dist={min_dist:.6f} ball_near_ms={ball_near_ms:.1f} "
          f"tube_ready_ms={tube_ready_ms:.1f} max_tcp={max_tcp:.2f} "
          f"max_qdot={max_qdot:.2f} "
          f"max_face={max_face:.1f} "
          f"hit_type={hit_type_en} "
          f"hit_time_error_ms={hit_time_error:.1f} hit_pos_error={hit_position_error:.6f} "
          f"v_racket_at_hit={v_racket_at_hit_val:.3f}")

    # ==========================================================================
    # 16. 保存轨迹
    # ==========================================================================
    if args.dump_trajectory and recorder is not None:
        # 评估完成后更新 metadata
        recorder._metadata.update({
            "hit_type": hit_type_en,
            "pos_error": float(pos_error),
            "hit_step": hit_step,
            "post_hit_steps": post_hit_steps,
        })
        dump_path = Path(args.dump_trajectory)
        recorder.save(dump_path, hit_step=hit_step)
        logger.info(f"轨迹已保存至 {dump_path}（.npz 格式，{len(recorder._q_desired_list)} 步）")

    # ==========================================================================
    # 17. 可视化
    # ==========================================================================
    if not args.no_plot:
        try:
            _, plot_fn = _import_v11_visuals()
            results_dir = Path(__file__).resolve().parent.parent / "results"
            tag = f"v12_{args.seed or 'default'}"
            ball_arr = np.array(ball_pos_history)
            # 从环境读取球拍位置轨迹
            racket_pos_list: list[np.ndarray] = []
            env.reset(init_q)
            env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
            env.update_kinematics()
            for i in range(min(len(X_history), len(U_history) + 1)):
                env.set_arm_state(X_history[i])
                racket_pos_list.append(env.get_ee_pos().copy())
            racket_pos_arr = np.array(racket_pos_list)

            normal_align_history = sim_component.normal_align_history
            pos_error_history = sim_component.pos_error_history
            plot_fn(
                results_dir, tag,
                ball_arr, racket_pos_arr, hit_window,
                distances_history, normal_align_history,
                ball_near_history, tube_ready_history,
                [],  # k_hit_steps_history（V12 未单独追踪）
                pos_error_history,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"可视化失败: {e}")

    if args.viewer:
        try:
            visualize_fn, _ = _import_v11_visuals()
            U_arr = np.array(U_history)
            from src.sim.replay import replay_trajectory
            replay_result = replay_trajectory(
                env, U_arr, init_q, init_q_left, p0, v0, hit_step,
            )
            visualize_fn(
                env, replay_result.X_replay, U_arr, replay_result.ball_replay,
                config_dict,
                init_q_left=init_q_left,
                post_hit_steps=post_hit_steps,
            )
        except (ImportError, OSError, RuntimeError, ValueError) as e:
            logger.warning(f"回放可视化失败: {e}")


if __name__ == "__main__":
    main()
