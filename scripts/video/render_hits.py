#!/usr/bin/env python
"""随稿视频击球录像渲染器（S1/S5/S9 仿真镜头）。

复用 run_20hits_video.py 的 episode 管线（同款 MPC / 安全滤波 / 接触判定 /
随挥），叠加球尾迹残影（ghost geom 池，手法同 render_ball_trail.py），逐物理
步离屏渲染并缓存帧，再按 30 fps 输出时间轴合成两版 MP4：
  - {label}_clean.mp4    无任何叠加（供 assemble.py 组装成片）
  - {label}_preview.mp4  带 Simulation 角标 / 短标题 / 慢放标注（供人工检查）

与 v12/exp17 口径对齐：serve-box 发球距离恒为 8.0 m（exp17 config_json 未覆盖
v12 默认值），不使用 run_20hits_video 的 SERVE_DIST_MAP。渲染后输出 hit_type
核验，与 pick_seeds.py 核验结果不符时打 WARN（此时改用 npz 离线重渲染兜底）。

用法（mujoco_tennis 环境，仓库根目录；egl 环境变量由本脚本自设）：
    # S1 开场段：常速完整挥拍 + 0.25× 接触慢放重播
    python scripts/video/render_hits.py --seed 47 --ball-speed 7 --ablation full --label s1
    # S5 配对素材（12 m/s 同 seed 两变体，常速原始录像，无慢放）
    python scripts/video/render_hits.py --seed 25 --ball-speed 12 --ablation none  --label s5_pt --realtime-only
    python scripts/video/render_hits.py --seed 25 --ball-speed 12 --ablation full --label s5_full --realtime-only
"""
from __future__ import annotations

import os

# egl 无头渲染：必须固定 NVIDIA vendor 库（否则 EGL 初始化失败，见 checklist Phase 0）
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mujoco
from PIL import Image, ImageDraw, ImageFont

from src.ilqt.async_replanner import AsyncReplanner, PlanRequest
from src.ilqt.cost import CompositeCost
from src.ilqt.cost_terms import ControlEffortTerm, SmoothnessTerm, TerminalHitTerm
from src.ilqt.mpc_controller import MPCConfig
from src.ilqt.replan_core import do_replan
from src.ilqt.robot_limits import RobotLimits, check_one_step_feasibility
from src.robot.constants import INIT_Q, INIT_Q_LEFT, SHOULDER_POS, WORKSPACE_RADIUS
from src.sim.rm65_env import RM65Env
from src.tennis.ball import generate_ball_from_serve_box
from src.tennis.hitting import find_hitting_point_physics

try:
    from src.cpp.solver_cpp import ILQTSolver
except ImportError:
    from src.ilqt.solver import ILQTSolver
from src.ilqt.mpc_helpers import (
    compute_jacobian_init_control,
    compute_joint1_backswing_trajectory,
    compute_r_schedule,
    generate_backswing_warm_start,
    resample_control_sequence,
)
from src.ilqt.tube_builder import build_hitting_tube, search_hit_window
from src.ilqt.tube_cost import SoftminOnlyCost, TubeHittingCostWrapper, TubeOnlyCost
from src.ilqt.tube_types import ReplanState, TubeConfig
from src.utils.yaml_utils import load_config, merge_configs

REPO = Path(__file__).resolve().parents[2]
MODEL_XML = REPO / "src" / "robot" / "rm65_model.xml"
OUT_DIR = REPO / "paper" / "video" / "assets"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# 尾迹 ghost 池参数（手法见 render_ball_trail.py 头注：池化防 ngeom 静默丢 geom）
BALL_R = 0.033
BALL_RGB = (0.98, 0.95, 0.25)
GHOST_N = 64                          # 池槽位（0.36 s 窗口 / 2 步间隔 = 36 个，留余量）
GHOST_HIDDEN = np.array([0.0, 0.0, -50.0])
TRAIL_WINDOW_STEPS = 72               # 尾迹窗口（×5 ms = 0.36 s）
TRAIL_STRIDE = 2                      # 密带间隔（步）
TRAIL_RIBBON_ALPHA = 0.10             # 密带透明度
TRAIL_STROBE_STEPS = 8                # 频闪间隔（×5 ms = 40 ms）
TRAIL_STROBE_ALPHA = 0.55             # 频闪透明度

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logging.getLogger("src.ilqt.robot_limits").setLevel(logging.WARNING)
logger = logging.getLogger("render_hits")


def build_trail_model(width: int, height: int) -> Path:
    """生成含球尾迹 ghost 池的包装模型 XML 路径（include 原模型 + N 个隐藏球 geom）。

    ghost 挂 worldbody（geom 追加在原模型之后，不改动既有 geom 索引），且
    contype/conaffinity=0 不参与物理；offwidth/offheight 须与 Renderer 一致。

    Args:
        width: 离屏帧缓冲宽（px）。
        height: 离屏帧缓冲高（px）。

    Returns:
        包装模型 XML 的临时文件路径（供 RM65Env 加载）。
    """
    ghosts = "".join(
        f'    <geom name="ghost_{k}" type="sphere" size="{BALL_R:.4f}" '
        f'pos="{GHOST_HIDDEN[0]} {GHOST_HIDDEN[1]} {GHOST_HIDDEN[2]}" '
        f'rgba="1 1 1 0" contype="0" conaffinity="0"/>\n' for k in range(GHOST_N))
    wrapper = (
        "<mujoco>\n"
        f'  <include file="{MODEL_XML.as_posix()}"/>\n'
        "  <visual>\n"
        f'    <global offwidth="{width}" offheight="{height}"/>\n'
        # headlight 跟随相机，补亮相机-facing 面（球背光侧发黑的修复）
        '    <headlight ambient="0.35 0.35 0.35" diffuse="0.55 0.55 0.55" '
        'specular="0.15 0.15 0.15"/>\n'
        "  </visual>\n"
        "  <worldbody>\n"
        f"{ghosts}"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    tmp = Path(tempfile.mkdtemp(prefix="trail_")) / "wrap_trail.xml"
    tmp.write_text(wrapper, encoding="utf-8")
    return tmp


def run_episode(args: argparse.Namespace) -> dict:
    """跑单球 episode 并逐物理步渲染缓存（episode 管线复制自 run_20hits_video）。

    Returns:
        含 frames（uint8 帧列表）、ball_pos/ee_pos（逐步）、hit_step、hit_type、
        pos_error、v_racket_at_hit 的字典。
    """
    base_path = REPO / "configs"
    config_dict = load_config(base_path / "default.yaml")
    for extra in ("v5_active_hit.yaml", "mpc.yaml"):
        p = base_path / extra
        if p.exists():
            config_dict = merge_configs(config_dict, load_config(p))

    dt = float(config_dict["sim"]["dt"])
    g = np.array(config_dict["ball"]["gravity"], dtype=np.float64)
    ablation_mode = args.ablation
    need_candidates = ablation_mode in ("full", "tube_only", "softmin_only")

    total_horizon, fixed_horizon = 250, 120
    replan_interval, max_iter_per_plan = 20, 10
    first_plan_iters, near_plan_iters = 30, 5
    serve_dist = args.serve_distance          # 恒 8.0，与 exp17/v12 默认对齐

    follow_cfg = config_dict["hitting"]
    follow_through_steps = 0 if args.no_follow_through else int(
        follow_cfg.get("follow_through_steps", 160))
    follow_through_v_terminal = float(follow_cfg.get("follow_through_v_terminal", 0.3))
    follow_through_length = args.hit_shift

    tube_cfg = TubeConfig(
        window_half_ms=args.window_ms, Q_p_tube=500.0, Q_v_tube=0.0, Q_n_tube=0.0,
        tube_cost_ratio=1.0, softmin_beta=args.softmin_beta,
        use_softmin_terminal=ablation_mode in ("full", "softmin_only"),
    )

    model_path = build_trail_model(args.width, args.height)
    env = RM65Env(model_path, dt=dt)
    env.init_q_left = INIT_Q_LEFT
    NQ, NU, NX, LEFT_NQ = env.NQ, env.NU, env.NX, env.LEFT_ARM_NQ
    ghost_ids = np.array(
        [mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_{k}")
         for k in range(GHOST_N)], dtype=int)

    rl_cfg = dict(config_dict.get("robot_limits", {}))
    if args.max_tcp > 0:
        rl_cfg["max_tcp_speed"] = args.max_tcp
    if args.terminal_exempt_steps > 0:
        rl_cfg["terminal_exempt_steps"] = args.terminal_exempt_steps
    robot_limits = RobotLimits.from_config(
        rl_cfg, dt=dt, ctrlrange=env.model.actuator_ctrlrange[:NU])

    _hard_x_body_ids = [
        mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, n)
        for n in ("r_link1", "r_link2", "r_link3", "r_link4",
                  "r_link5", "r_link6", "r_flange", "r_racket_body")
    ]
    ctrl_lo = env.model.actuator_ctrlrange[:NU, 0]
    ctrl_hi = env.model.actuator_ctrlrange[:NU, 1]

    Q_p = np.array(config_dict["cost"]["Q_p"], dtype=np.float64) * 2.0
    Q_v = np.array(config_dict["cost"]["Q_v"], dtype=np.float64) * 2.0
    R = float(config_dict["cost"]["R"])
    solver = ILQTSolver(dict(config_dict["ilqt"]), use_analytical=True)

    racket_speed = float(config_dict["hitting"]["racket_speed"])
    target_center = np.array([-0.82765693, -0.47411682, 0.86947444])
    target_offset = 0.10

    stage_cfg = config_dict.get("stage_weights", {})
    far_stage = stage_cfg.get("far", {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0})
    mid_stage = stage_cfg.get("mid", {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0})
    near_stage = stage_cfg.get("near", {"Q_qdot_mult": 0.0, "Q_qddot_mult": 0.0, "Q_du_mult": 0.0})

    rng = np.random.default_rng(args.seed)

    # ── 渲染器：自由相机 + 四灯（与 run_20hits_video 同款机位） ──
    renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = (
        args.cam_distance, args.cam_elevation, args.cam_azimuth)
    cam.lookat[:] = [0.0, 0.0, 1.0]
    env.model.light_pos[0] = [0.0, 0.0, 8.0]
    env.model.light_dir[0] = [0.0, 0.0, -1.0]
    env.model.light_diffuse[0] = [1.4, 1.45, 1.55]
    env.model.light_ambient[0] = [0.3, 0.3, 0.35]
    env.model.light_specular[0] = [0.5, 0.5, 0.5]
    if env.model.nlight > 1:
        env.model.light_pos[1] = [2.0, -2.0, 3.0]
        env.model.light_dir[1] = [-0.4, 0.3, -0.8]
        env.model.light_diffuse[1] = [1.2, 1.15, 1.05]
        env.model.light_specular[1] = [0.6, 0.6, 0.6]
        env.model.light_active[1] = True
    if env.model.nlight > 2:
        env.model.light_pos[2] = [-1.5, -1.0, 2.5]
        env.model.light_dir[2] = [0.3, 0.2, -0.7]
        env.model.light_diffuse[2] = [0.8, 0.85, 0.95]
        env.model.light_specular[2] = [0.4, 0.4, 0.4]
        env.model.light_active[2] = True

    # ── 视觉统一（与论文 Fig.1/3 一致）：industrial 灰配色 + 球改亮黄色 ──
    sys.path.insert(0, str(REPO / "scripts" / "tools"))
    from render_ball_trail import apply_palette
    palette_report = apply_palette(env.model, "industrial")
    logger.info("调色板 industrial: %d 条规则生效", len(palette_report))
    ball_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    for gid in range(env.model.ngeom):
        if env.model.geom_bodyid[gid] == ball_body_id:
            env.model.geom_rgba[gid] = [1.0, 0.82, 0.10, 1.0]   # 亮黄，低码率可辨
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0        # 隐藏碰撞体可视化
    opt.sitegroup[0] = 0        # 隐藏 hit_target 等标记点

    frames: list[np.ndarray] = []
    ball_hist: list[np.ndarray] = []
    closeup_frames: list[np.ndarray] = []   # 慢放推近段帧（与 closeup_steps 对齐）
    closeup_start = -1
    closeup_end = -1

    # 推近相机（慢放段专用；参数在 p_hit 求出后于首帧渲染前设置）
    cam2 = mujoco.MjvCamera()
    cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam2.azimuth = 135
    cam2.elevation = -12

    def capture() -> None:
        """更新尾迹 ghost 池 → 渲染当前步 → 缓存帧与球位。

        索引以 ball_hist 自身长度为准（far-JT 重规划分支不推进物理步也不出帧，
        帧序号与循环计数可能错位，不能用外层 step 计数）。
        """
        idx = len(ball_hist)
        lo = max(0, idx - TRAIL_WINDOW_STEPS)
        slot = 0
        for i in range(lo, idx, TRAIL_STRIDE):   # 不含当前步（真球在场）
            alpha = (TRAIL_STROBE_ALPHA if i % TRAIL_STROBE_STEPS == 0
                     else TRAIL_RIBBON_ALPHA)
            env.model.geom_pos[ghost_ids[slot]] = ball_hist[i]
            env.model.geom_rgba[ghost_ids[slot]] = [*BALL_RGB, alpha]
            slot += 1
            if slot >= GHOST_N:
                break
        for k in range(slot, GHOST_N):            # 未用槽位隐藏
            env.model.geom_pos[ghost_ids[k]] = GHOST_HIDDEN
            env.model.geom_rgba[ghost_ids[k]] = [1, 1, 1, 0]
        renderer.update_scene(env.data, camera=cam, scene_option=opt)
        frames.append(renderer.render())
        ball_hist.append(env.get_ball_state()[0].copy())
        # 慢放窗口内同步渲染推近镜头（窗口按计划击球步划定，接触前后各扩若干步）
        if closeup_start >= 0 and closeup_start <= idx <= closeup_end:
            renderer.update_scene(env.data, camera=cam2, scene_option=opt)
            closeup_frames.append(renderer.render())

    # ── 发球（与 v12 serve-box 默认一致：中心 (0,-8,1.2)，半箱 (4,0.1,0.15)） ──
    init_q = INIT_Q
    x0 = np.zeros(NX)
    x0[:NQ] = init_q
    env.reset(init_q)
    env.data.qpos[NQ:NQ + LEFT_NQ] = INIT_Q_LEFT
    env.data.qvel[NQ:NQ + LEFT_NQ] = 0.0
    env.update_kinematics()

    p0, v0, p_hit = generate_ball_from_serve_box(
        serve_box_center=(0.0, -serve_dist, 1.2),
        serve_box_halfsize=(4.0, 0.1, 0.15),
        target_center=target_center, target_offset=target_offset,
        shoulder_pos=SHOULDER_POS, workspace_radius=WORKSPACE_RADIUS, g=g,
        ball_speed=args.ball_speed, speed_range=(8.0, 18.0),
        use_bounce=True, bounce_restitution=0.75, rng=rng,
    )
    hit_info = find_hitting_point_physics(
        env, p0, v0, SHOULDER_POS, WORKSPACE_RADIUS, total_horizon)
    if hit_info is None:
        raise RuntimeError(f"seed={args.seed} 球不可达，换种子")
    k_hit_total, p_hit, v_ball_hit = (
        hit_info["k_hit"], hit_info["p_hit"], hit_info["v_ball_hit"])

    ball_positions_all, ball_velocities_all = env.predict_ball_trajectory(
        p0, v0, total_horizon)

    # IK 关节裕度不足时在候选窗内改选击球点（与 run_20hits_video 相同）
    q_ik_init = env.solve_ik(p_hit, q_init=init_q, max_iter=50, eps=1e-2)
    min_margin = float(np.min(np.minimum(
        (q_ik_init - robot_limits.q_lower) * 180.0 / np.pi,
        (robot_limits.q_upper - q_ik_init) * 180.0 / np.pi)))
    if min_margin < 3.0:
        best_k, best_m = k_hit_total, min_margin
        for dk in range(-30, 31):
            kk = k_hit_total + dk
            if kk < 1 or kk > len(ball_positions_all):
                continue
            p_alt = ball_positions_all[kk - 1]
            if not (np.linalg.norm(p_alt - SHOULDER_POS) < WORKSPACE_RADIUS
                    and p_alt[2] > 0.3
                    and -0.60 < p_alt[2] - SHOULDER_POS[2] < 0.55):
                continue
            q_alt = env.solve_ik(p_alt, q_init=init_q, max_iter=30, eps=2e-2)
            m_a = float(np.min(np.minimum(
                (q_alt - robot_limits.q_lower) * 180.0 / np.pi,
                (robot_limits.q_upper - q_alt) * 180.0 / np.pi)))
            if m_a > best_m:
                best_m, best_k = m_a, kk
        if best_k != k_hit_total:
            p_hit = ball_positions_all[best_k - 1].copy()
            v_ball_hit = ball_velocities_all[best_k - 1].copy()
            k_hit_total = best_k

    n_des_single = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    d_follow = n_des_single.copy()
    v_hit_at_contact = args.target_speed * d_follow
    hit_shift = follow_through_length
    near_threshold = max(50, k_hit_total // 3)
    far_threshold = 50
    p_follow = p_hit + hit_shift * d_follow

    # 慢放推近窗口与相机（以计划击球步为中心，接触前后各扩 slow_pre/slow_post；
    # 方位角偏向侧视，把来球走廊与球拍同时收进画面、避免立柱占满前景）
    cam2.lookat[:] = p_hit + np.array([0.0, -0.20, 0.02])
    cam2.distance = args.slow_cam_distance
    cam2.azimuth = args.slow_cam_azimuth
    cam2.elevation = -10
    closeup_start = max(0, k_hit_total - int(args.slow_pre_ms / 1000.0 / dt) - 6)
    closeup_end = k_hit_total + int(args.slow_post_ms / 1000.0 / dt) + 6

    dist_to_ball = np.linalg.norm(p_hit - env.get_ee_pos())
    adaptive_bs = 0.4 + np.clip((dist_to_ball - 0.8) / 0.7, 0.0, 1.0) * 0.6
    cur_backswing_offset = -adaptive_bs

    hit_window = None
    hitting_tube = None
    if need_candidates:
        hit_window = search_hit_window(
            env, p0, v0, SHOULDER_POS, WORKSPACE_RADIUS,
            k_hit_total + 30, tube_cfg, ball_direction="y",
            current_step=0, robot_limits=robot_limits, init_q=init_q)
        if hit_window is not None:
            hitting_tube = build_hitting_tube(hit_window, racket_speed, d_follow, tube_cfg)

    r_joint_scale: dict[int, float] = {}
    if not args.no_backswing:
        r_joint_scale[0] = 0.3
    R_schedule_init = compute_r_schedule(
        k_hit_total, R, decay_ratio=args.r_decay) if args.r_decay else None

    base_cost_fn = CompositeCost(
        env,
        running_terms=[
            ControlEffortTerm(R=R, R_joint_scale=r_joint_scale or None,
                              R_schedule=R_schedule_init, NU=NU),
            SmoothnessTerm(
                Q_qdot=float(config_dict["cost"].get("Q_qdot", 0.0)),
                Q_qddot=float(config_dict["cost"].get("Q_qddot", 0.0)),
                Q_du=float(config_dict["cost"].get("Q_du", 0.0)),
                NQ=NQ, NX=NX, NU=NU, dt=dt),
        ],
        terminal_terms=[
            TerminalHitTerm(p_follow, v_hit_at_contact, Q_p, Q_v,
                            Q_n=args.normal_weight, n_des=n_des_single,
                            NX=NX, NQ=NQ),
        ],
    )

    def wrap_cost(tube) -> object:
        """按消融模式包装基准代价（none 返回底座）。"""
        if tube is None:
            return base_cost_fn
        if ablation_mode == "full":
            return TubeHittingCostWrapper(env, base_cost_fn, tube, k_hit_total, tube_cfg)
        if ablation_mode == "tube_only":
            return TubeOnlyCost(env, base_cost_fn, tube, k_hit_total, tube_cfg)
        if ablation_mode == "softmin_only":
            return SoftminOnlyCost(env, base_cost_fn, tube, k_hit_total, tube_cfg)
        return base_cost_fn

    cost_fn = wrap_cost(hitting_tube)

    if not args.no_backswing:
        U_prev, _ = generate_backswing_warm_start(
            env, x0, p_follow, v_hit_at_contact, k_hit_total,
            backswing_offset=cur_backswing_offset,
            backswing_ratio=args.bs_ratio, fix_joint5_angle=None, n_des=n_des_single)
    else:
        U_prev = compute_jacobian_init_control(
            env, x0, p_follow, k_hit_total, gain=60.0, fix_joint5_angle=None)

    env.reset(init_q)
    env.data.qpos[NQ:NQ + LEFT_NQ] = INIT_Q_LEFT
    env.data.qvel[NQ:NQ + LEFT_NQ] = 0.0
    env.update_kinematics()
    env.set_ball_state(p0, v0)

    ball_pos_init, ball_vel_init = env.get_ball_state()
    first_request = PlanRequest(
        x_current=x0.copy(), ball_pos=ball_pos_init, ball_vel=ball_vel_init,
        step=0, k_hit_current=k_hit_total,
        U_prev=np.zeros((0, NU)), p_hit_current=p_hit.copy(),
        v_hit_desired=v_hit_at_contact, n_des_current=n_des_single.copy(),
        d_hat=d_follow.copy(), is_first_plan=True)
    replan_state = ReplanState(
        k_hit_new=k_hit_total, p_hit_new=p_hit.copy(),
        v_ball_hit_new=v_ball_hit.copy(), current_n_des=n_des_single.copy(),
        U_prev=np.zeros((0, NU)), is_first_plan=True)
    mpc_config = MPCConfig(
        dt=dt, total_horizon=total_horizon, fixed_horizon=fixed_horizon,
        replan_interval=replan_interval, max_iter_per_plan=max_iter_per_plan,
        first_plan_iters=first_plan_iters, near_plan_iters=near_plan_iters,
        near_threshold=near_threshold, R=R,
        Q_p_scale_far=5.0, Q_v_scale_far=3.0, Q_p_scale_near=8.0, Q_v_scale_near=120.0,
        follow_through_length=follow_through_length,
        follow_through_steps=follow_through_steps,
        follow_through_v_terminal=follow_through_v_terminal,
        use_backswing=not args.no_backswing, use_r_decay=bool(args.r_decay),
        r_decay_ratio=args.r_decay, time_perturb_s=0.0, space_perturb_m=0.0,
        perturb_alpha_min=0.0, normal_flip=False, fix_joint5_angle=None,
        backswing_offset=cur_backswing_offset, backswing_ratio=args.bs_ratio,
        shoulder_pos=SHOULDER_POS, workspace_radius=WORKSPACE_RADIUS,
        smooth_far=far_stage, smooth_mid=mid_stage, smooth_near=near_stage,
        normal_weight=args.normal_weight, racket_speed=racket_speed,
        far_threshold=far_threshold, tube_cfg=tube_cfg)

    env_plan = AsyncReplanner(env, do_replan, mpc_config, robot_limits, solver,
                              state=replan_state, model_path=model_path)
    _ = env_plan._ensure_env_plan()

    first_result = do_replan(first_request, env_plan.env_plan,
                             replan_state, mpc_config, robot_limits, solver)
    replan_state.is_first_plan = False
    replan_state.k_hit_new = first_result.k_hit_new
    replan_state.p_hit_new = first_result.p_hit_new.copy()
    replan_state.v_ball_hit_new = first_result.v_ball_hit_new.copy()
    replan_state.current_n_des = first_result.n_des_new.copy()
    replan_state.U_prev = first_result.U_prev.copy()

    U_buffer = first_result.U_buffer.copy()
    buffer_idx = 0
    U_prev = first_result.U_prev.copy()
    k_hit_new = first_result.k_hit_new
    p_hit_new = first_result.p_hit_new.copy()
    v_ball_hit_new = first_result.v_ball_hit_new.copy()
    current_n_des = first_result.n_des_new.copy()
    is_first_plan = False

    x_current = x0.copy()
    hit_step, ball_was_hit = -1, False
    p_ee_at_hit = ball_pos_at_hit = None
    v_ee_at_hit = None
    follow_through_start = -1
    active_contact = passive_contact = False


    effective_total = total_horizon + follow_through_steps
    for step in range(effective_total):
        ball_pos, ball_vel = env.get_ball_state()
        dist_cur = np.linalg.norm(env.get_ee_pos() - ball_pos)
        remaining_horizon = max(5, min(effective_total - step, total_horizon - step))
        need_replan = ((step % replan_interval == 0) or (step == 0)
                       or (buffer_idx >= len(U_buffer)))

        if need_replan:
            hit_info_new = find_hitting_point_physics(
                env, ball_pos, ball_vel, SHOULDER_POS, WORKSPACE_RADIUS,
                remaining_horizon)
            if hit_info_new is None:
                if ball_was_hit and follow_through_start < 0:
                    follow_through_start = step
                elif not ball_was_hit:
                    break
            else:
                k_hit_candidate = hit_info_new["k_hit"]
                if (k_hit_candidate < max(10, k_hit_new // 4) and k_hit_new > 30):
                    k_hit_candidate = max(1, k_hit_new - replan_interval)
                p_hit_new = hit_info_new["p_hit"]
                v_ball_hit_new = hit_info_new["v_ball_hit"]
                k_hit_new = k_hit_candidate
                current_n_des = -v_ball_hit_new / (np.linalg.norm(v_ball_hit_new) + 1e-8)
                p_follow_new = p_hit_new + hit_shift * d_follow

                base_cost_fn.update_target(
                    p_hit_new + follow_through_length * d_follow,
                    v_hit_at_contact, n_des=current_n_des)
                base_cost_fn.set_midpoint_target(None, None)

                if need_candidates:
                    hw = search_hit_window(
                        env, ball_pos, ball_vel, SHOULDER_POS, WORKSPACE_RADIUS,
                        remaining_horizon, tube_cfg, ball_direction="y",
                        current_step=0, robot_limits=robot_limits,
                        init_q=x_current[:NQ].copy())
                    cost_fn = wrap_cost(
                        build_hitting_tube(hw, racket_speed, d_follow, tube_cfg)
                        if hw is not None else None)

                if k_hit_new > far_threshold and not is_first_plan:
                    ball_save_far, ball_vel_far = env.get_ball_state()
                    u_jt = compute_jacobian_init_control(
                        env, x_current, p_follow_new, replan_interval,
                        gain=60.0, fix_joint5_angle=None)
                    env.set_ball_state(ball_save_far, ball_vel_far)
                    env.set_arm_state(x_current)
                    U_buffer, buffer_idx = u_jt, 0
                    continue

                env.set_arm_state(x_current)
                env.update_kinematics()
                pos_err_now = np.linalg.norm(env.get_ee_pos() - p_hit_new)
                if pos_err_now > 0.10:
                    Q_p_scale, Q_v_scale = 5.0, 3.0
                else:
                    ratio_s = pos_err_now / 0.10
                    Q_p_scale = 8.0 + (5.0 - 8.0) * ratio_s
                    Q_v_scale = 120.0 + (3.0 - 120.0) * ratio_s
                cost_fn.update_weights(Q_p_scale, Q_v_scale)

                s = (far_stage if k_hit_new > 50
                     else mid_stage if k_hit_new > 20 else near_stage)
                if hasattr(cost_fn, "set_smoothness_scale"):
                    cost_fn.set_smoothness_scale(
                        float(s["Q_qdot_mult"]), float(s["Q_qddot_mult"]),
                        float(s["Q_du_mult"]))
                if args.r_decay:
                    horizon_full = k_hit_new
                    horizon_plan = min(horizon_full, fixed_horizon)
                    cost_fn.set_R_schedule(compute_r_schedule(
                        horizon_full, R, decay_ratio=args.r_decay)[:horizon_plan])
                else:
                    cost_fn.set_R_schedule(None)

                iters_plan, skip_ls, fp_limits, fast_lin = max_iter_per_plan, True, robot_limits, False
                if is_first_plan:
                    iters_plan, skip_ls, fp_limits = first_plan_iters, True, None
                    is_first_plan = False
                elif k_hit_new <= near_threshold:
                    fast_lin, fp_limits = True, None
                    iters_plan = (min(near_plan_iters, max(max_iter_per_plan, 10))
                                  if k_hit_new > 30 else near_plan_iters)
                else:
                    fast_lin = True

                if (not args.no_backswing) and k_hit_new > 0:
                    q_hit_ik = env.solve_ik(p_hit_new, q_init=x_current[:NQ],
                                            max_iter=150, eps=1e-3)
                    env.set_arm_state(np.concatenate([q_hit_ik, np.zeros(NQ)]))
                    qdot_hit_new = np.linalg.lstsq(
                        env.get_ee_jacp(), v_hit_at_contact, rcond=None)[0]
                    qd_n = np.linalg.norm(qdot_hit_new)
                    if qd_n > 3.0:
                        qdot_hit_new *= 3.0 / qd_n
                    bs_scale_plan = k_hit_new / max(k_hit_total, 1)
                    q_des_full = np.zeros((k_hit_new, NQ))
                    q_des_full[:, 0] = compute_joint1_backswing_trajectory(
                        x_current[0], x_current[NQ], q_hit_ik[0], qdot_hit_new[0],
                        k_hit_new, backswing_offset=cur_backswing_offset * bs_scale_plan,
                        backswing_ratio=args.bs_ratio)
                    for j in range(1, NQ):
                        q_des_full[:, j] = np.linspace(
                            x_current[j], q_hit_ik[j], k_hit_new)
                    cost_fn.set_q_des_traj(q_des_full[:min(k_hit_new, fixed_horizon)],
                                           Q_joint=None)

                if len(U_prev) >= k_hit_new // 3:
                    U_warm = resample_control_sequence(U_prev, k_hit_new)[:fixed_horizon]
                elif (not args.no_backswing) and k_hit_new > 0:
                    U_warm_full, _ = generate_backswing_warm_start(
                        env, x_current, p_follow_new, v_hit_at_contact, k_hit_new,
                        cur_backswing_offset, args.bs_ratio, None, current_n_des)
                    U_warm = U_warm_full[:min(k_hit_new, fixed_horizon)]
                else:
                    U_warm = compute_jacobian_init_control(
                        env, x_current, p_follow_new, k_hit_new, gain=30.0,
                        fix_joint5_angle=None)[:fixed_horizon]

                ball_save, ball_vel_save = env.get_ball_state()
                _, U_mpc, _, _ = solver.solve_few_iters(
                    env, cost_fn, x_current, U_warm, max_iter=iters_plan,
                    skip_linesearch=skip_ls, limits=fp_limits, use_fast_lin=fast_lin)
                env.set_ball_state(ball_save, ball_vel_save)
                env.set_arm_state(x_current)

                U_prev = (U_mpc[replan_interval:] if len(U_mpc) > replan_interval
                          else U_mpc[1:] if len(U_mpc) > 0 else np.zeros((0, NU)))
                U_buffer, buffer_idx = U_mpc[:replan_interval], 0
                if k_hit_new <= 30 and len(U_mpc) >= replan_interval * 2:
                    U_buffer = U_mpc[:replan_interval * 2]

        if buffer_idx < len(U_buffer):
            u_cmd = U_buffer[buffer_idx]
            buffer_idx += 1
        else:
            env.set_arm_state(x_current)
            tau_backup = (env.get_ee_jacp().T
                          @ (p_hit_new - env.get_ee_pos()) * 30.0
                          - 2.0 * x_current[NQ:])
            u_cmd = np.clip(tau_backup, ctrl_lo, ctrl_hi)

        enable_collision = False
        if not ball_was_hit and (k_hit_new <= 30 and dist_cur < 0.35
                                 or k_hit_new <= 10):
            enable_collision = True
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)

        # X 平面墙预判（臂不越过身体中线）
        ball_save_x = env.get_ball_state()
        x_save_x = x_current.copy()
        for beta_x in (1.0, 0.6, 0.3, 0.0):
            u_try = np.clip(beta_x * u_cmd, ctrl_lo, ctrl_hi)
            env.step_from_state(x_current, u_try)
            env.update_kinematics()
            ok_x = all(env.data.xpos[bid, 0] <= -0.1 for bid in _hard_x_body_ids)
            env.set_ball_state(*ball_save_x)
            env.set_arm_state(x_save_x)
            if ok_x:
                u_cmd = u_try
                break

        ball_save_sf = env.get_ball_state()
        arm_save_sf = x_current.copy()
        ball_ref = ball_save_sf

        def _safety_step(x_s, u_s, _env=env, _ref=ball_ref):
            x_n = _env.step_from_state(x_s, u_s)
            _env.set_ball_state(*_ref)
            return x_n

        ok_f, _ = check_one_step_feasibility(
            x_current, u_cmd, robot_limits, dt,
            step_predictor=_safety_step, k_hit_remaining=k_hit_new, env=env)
        if not ok_f:
            for beta_s in (0.8, 0.6, 0.4, 0.2, 0.0):
                u_s = np.clip(beta_s * u_cmd, ctrl_lo, ctrl_hi)
                env.set_arm_state(arm_save_sf)
                ok_ss, _ = check_one_step_feasibility(
                    x_current, u_s, robot_limits, dt,
                    step_predictor=_safety_step, k_hit_remaining=k_hit_new, env=env)
                if ok_ss:
                    u_cmd = u_s
                    break
            else:
                u_cmd = -20.0 * x_current[NQ:]
        env.set_ball_state(*ball_save_sf)
        env.set_arm_state(arm_save_sf)

        ball_vel_before = ball_vel.copy()
        x_current, ball_pos, ball_vel = env.step_full(u_cmd)

        ball_racket_hit = False
        if enable_collision and not ball_was_hit:
            for ci in range(env.data.ncon):
                c = env.data.contact[ci]
                g1 = env.model.geom(c.geom1).name
                g2 = env.model.geom(c.geom2).name
                if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
                    ball_racket_hit = True
                    v_ee_at_hit = float(np.linalg.norm(env.get_ee_vel()))
                    if v_ee_at_hit > 0.3:
                        active_contact = True
                    else:
                        passive_contact = True
                    n_hat = env.get_ee_normal()
                    n_hat = n_hat / (np.linalg.norm(n_hat) + 1e-8)
                    v_rel_n = np.dot(ball_vel_before - env.get_ee_vel(), n_hat)
                    env.set_ball_vel(ball_vel_before - 1.8 * v_rel_n * n_hat)
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        capture()

        if ball_racket_hit and not ball_was_hit:
            ball_was_hit = True
            hit_step = step
            p_ee_at_hit = env.get_ee_pos().copy()
            ball_pos_at_hit = ball_pos.copy()
        if (ball_was_hit and follow_through_start < 0 and hit_step >= 0
                and (step - hit_step) >= 3):
            follow_through_start = step
        if (args.follow_trigger == "planned" and k_hit_new <= 1
                and follow_through_start < 0):
            hit_step = step if hit_step < 0 else hit_step
            p_ee_at_hit = p_ee_at_hit or env.get_ee_pos().copy()
            ball_pos_at_hit = ball_pos_at_hit or ball_pos.copy()
            follow_through_start = step

        if follow_through_start >= 0 and step > follow_through_start:
            dt_follow = step - follow_through_start
            if dt_follow <= follow_through_steps:
                v_max = np.linalg.norm(v_hit_at_contact)
                T_follow = follow_through_steps * dt
                a_follow = v_max / T_follow if T_follow > 0 else 0.0
                t_el = dt_follow * dt
                p_des = p_ee_at_hit + d_follow * (
                    v_max * t_el - 0.5 * a_follow * t_el ** 2)
                env.update_kinematics()
                J_p_f = env.get_ee_jacp()[:, :NQ]
                F_f = 200.0 * (p_des - env.get_ee_pos()) - 20.0 * (J_p_f @ x_current[NQ:])
                u_follow = np.clip(J_p_f.T @ F_f, ctrl_lo, ctrl_hi)
                x_current, ball_pos, ball_vel = env.step_full(u_follow)
                capture()
                continue
            else:
                break

    # ── 击后保持 + 离场窗口（球飞离 + 手臂缓回，供 S1 尾与 miss 镜头） ──
    for _ in range(20):
        u_hold = np.clip(100.0 * (init_q - x_current[:NQ]) - 10.0 * x_current[NQ:],
                         ctrl_lo, ctrl_hi)
        x_current, _, _ = env.step_full(u_hold)
        capture()
    for _ in range(args.departure_steps):
        u_hold = np.clip(100.0 * (init_q - x_current[:NQ]) - 10.0 * x_current[NQ:],
                         ctrl_lo, ctrl_hi)
        x_current, _, _ = env.step_full(u_hold)
        capture()

    pos_error = (float(np.linalg.norm(p_ee_at_hit - ball_pos_at_hit))
                 if p_ee_at_hit is not None and ball_pos_at_hit is not None else 999.0)
    hit_type = ("active" if active_contact
                else "passive" if passive_contact else "miss")
    renderer.close()

    return {
        "frames": frames, "closeup_frames": closeup_frames,
        "closeup_start": closeup_start, "ball_pos": np.array(ball_hist),
        "hit_step": hit_step, "hit_type": hit_type, "pos_error": pos_error,
        "v_racket_at_hit": v_ee_at_hit or 0.0, "dt": dt,
    }


# ───────────────────────────── 合成与叠加 ─────────────────────────────

def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """加载 DejaVu 字体（大小按 720p 画面标定）。"""
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


BADGE_TEXT = "MuJoCo simulation"


def draw_badge(img: Image.Image) -> None:
    """左上角仿真角标（半透明黑底白字，全片常驻）。

    文字标注仿真器名（用户 2026-09-20 增补）：MuJoCo 为第三方开源仿真器，
    论文正文已写明，标注它提升可信度且不破坏双盲。
    """
    d = ImageDraw.Draw(img, "RGBA")
    font = _font(22)
    tw = d.textlength(BADGE_TEXT, font=font)
    x0, y0 = 18, 16
    d.rounded_rectangle((x0, y0, x0 + tw + 22, y0 + 34), radius=8,
                        fill=(0, 0, 0, 130))
    d.text((x0 + 11, y0 + 5), BADGE_TEXT, font=font, fill=(255, 255, 255, 235))


def draw_title(img: Image.Image, text: str, alpha: float) -> None:
    """顶部居中短标题（alpha 淡入淡出；避开底部字幕区）。"""
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(img, "RGBA")
    font = _font(30, bold=True)
    tw = d.textlength(text, font=font)
    cx = img.width / 2
    band_h = 50
    d.rectangle((cx - tw / 2 - 18, 14, cx + tw / 2 + 18, 14 + band_h),
                fill=(0, 0, 0, int(110 * alpha)))
    d.text((cx - tw / 2, 22), text, font=font,
           fill=(255, 255, 255, int(240 * alpha)))


def draw_tag(img: Image.Image, text: str) -> None:
    """右下角小标签（慢放倍速等）。"""
    d = ImageDraw.Draw(img, "RGBA")
    font = _font(20)
    tw = d.textlength(text, font=font)
    d.rounded_rectangle((img.width - tw - 30, img.height - 46,
                         img.width - 12, img.height - 16), radius=7,
                        fill=(0, 0, 0, 130))
    d.text((img.width - tw - 21, img.height - 42), text, font=font,
           fill=(255, 255, 255, 235))


def compose(args: argparse.Namespace, ep: dict) -> dict:
    """把逐物理步帧缓存按 30 fps 时间轴合成 clean/preview 两版 MP4。

    时间轴（S1 默认）：A 常速完整挥拍（含接触+离场）→ B 0.25× 接触慢放重播；
    --realtime-only 时输出整段常速录像（S5 素材）。

    Returns:
        {clean 路径, preview 路径, 时长秒, 输出帧数}。
    """
    import imageio

    frames, dt, hit_step = ep["frames"], ep["dt"], ep["hit_step"]
    n = len(frames)
    fps = args.fps
    t_hit = hit_step * dt if hit_step >= 0 else n * dt * 0.6
    slow_tag = f"{1.0 / args.slow_factor:g}× playback"

    if args.realtime_only:
        plan = [("A", 0.0, n * dt, 1.0)]
    else:
        plan = [
            ("A", 0.0, min(t_hit + args.normal_post_ms / 1000.0, n * dt), 1.0),
            ("B", max(t_hit - args.slow_pre_ms / 1000.0, 0.0),
             min(t_hit + args.slow_post_ms / 1000.0, n * dt), args.slow_factor),
        ]

    clean_path = OUT_DIR / f"{args.label}_clean.mp4"
    prev_path = OUT_DIR / f"{args.label}_preview.mp4"
    wc = imageio.get_writer(str(clean_path), fps=fps,
                            output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
    wp = imageio.get_writer(str(prev_path), fps=fps,
                            output_params=["-crf", "18", "-pix_fmt", "yuv420p"])

    fade_out = int(0.4 * fps)          # 结尾淡出帧数
    # 输出帧数 = 仿真时长 × fps × speed（慢放 ×speed 倍时长）
    total_out = sum(int((t1 - t0) * fps * speed) for _, t0, t1, speed in plan)
    out_i = 0
    title_text = "Relaxing the Point Target"
    cu_frames = ep.get("closeup_frames") or []
    cu_start = ep.get("closeup_start", -1)
    for seg, t0, t1, speed in plan:
        n_seg = int((t1 - t0) * fps * speed)
        seg_a_len = next(int((b - a) * fps * s) for s2, a, b, s in plan if s2 == "A")
        for j in range(n_seg):
            # 慢放 ×speed：每输出帧前进 1/(fps*speed) 秒源时间
            t_src = t0 + j / (fps * speed)
            idx = round(t_src / dt)
            idx = min(idx, n - 1)
            # B 段优先用推近镜头帧（窗口外的边缘帧回退广角）
            frame = frames[idx]
            if seg != "A" and cu_frames:
                ci = idx - cu_start
                if 0 <= ci < len(cu_frames):
                    frame = cu_frames[ci]
            wc.append_data(frame)
            img = Image.fromarray(frame).convert("RGBA")
            draw_badge(img)
            if seg == "A" and not args.realtime_only:
                fade_in = min(1.0, j / (0.5 * fps))
                fade_out_a = min(1.0, max(0.0, (seg_a_len - j) / (0.4 * fps)))
                draw_title(img, title_text, min(fade_in, fade_out_a))
            else:
                draw_tag(img, slow_tag)
            # 结尾统一淡出到黑
            rem = total_out - out_i - 1
            if rem < fade_out:
                black = Image.new("RGBA", img.size, (0, 0, 0, 255))
                img = Image.blend(img, black, 1.0 - (rem + 1) / fade_out)
            wp.append_data(np.array(img.convert("RGB")))
            out_i += 1
    wc.close()
    wp.close()
    return {"clean": str(clean_path), "preview": str(prev_path),
            "duration_s": total_out / fps, "frames": total_out}


def save_sheet(args: argparse.Namespace, ep: dict) -> str:
    """保存关键帧拼贴 PNG（A 段 4 帧广角 + B 段 4 帧推近），供快速人工检查。"""
    frames, dt, hit_step = ep["frames"], ep["dt"], max(ep["hit_step"], 0)
    n = len(frames)
    cu_frames = ep.get("closeup_frames") or []
    cu_start = ep.get("closeup_start", -1)
    a_ts = [0.05, 0.35, 0.7, 0.97]     # A 段相对时刻（比例）
    n_a = max(hit_step, 1)
    idxs = [min(int(n_a * r), n - 1) for r in a_ts]
    idxs += [min(hit_step + int(ms / 1000 / dt), n - 1)
             for ms in (-200, 0, 150, 400)]
    idxs = [max(i, 0) for i in idxs]
    tile_w = 640
    tile_h = 360
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 2), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    font = _font(18, bold=True)
    for k, idx in enumerate(idxs):
        frame = frames[idx]
        if k >= 4 and cu_frames:       # B 行 = 推近镜头
            ci = idx - cu_start
            if 0 <= ci < len(cu_frames):
                frame = cu_frames[ci]
        tile = Image.fromarray(frame).resize((tile_w, tile_h))
        px, py = (k % 4) * tile_w, (k // 4) * tile_h
        sheet.paste(tile, (px, py))
        tag = (f"t={idx * dt:.2f}s" + ("  CONTACT" if abs(idx - hit_step) <= 1 else ""))
        d.rectangle((px + 8, py + 8, px + 8 + d.textlength(tag, font=font) + 14, py + 34),
                    fill=(0, 0, 0, 160))
        d.text((px + 15, py + 12), tag, font=font, fill=(255, 255, 255))
    out = OUT_DIR / f"{args.label}_sheet.png"
    sheet.save(out)
    # 接触瞬间全分辨率单帧（广角 + 推近各一张）
    Image.fromarray(frames[min(hit_step, n - 1)]).save(OUT_DIR / f"{args.label}_contact.png")
    if cu_frames:
        ci = min(hit_step, n - 1) - cu_start
        if 0 <= ci < len(cu_frames):
            Image.fromarray(cu_frames[ci]).save(
                OUT_DIR / f"{args.label}_contact_closeup.png")
    return str(out)


def main() -> None:
    """入口：渲染单球 episode → 合成 clean/preview → 落盘元数据与拼贴。"""
    parser = argparse.ArgumentParser(description="随稿视频击球录像渲染器")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--ball-speed", type=float, default=7.0)
    parser.add_argument("--ablation", choices=["full", "tube_only", "softmin_only", "none"],
                        default="full")
    parser.add_argument("--label", type=str, required=True, help="输出文件名前缀")
    parser.add_argument("--expect", type=str, default=None,
                        help="期望 hit_type（active/miss），不符时打 WARN")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serve-distance", type=float, default=8.0,
                        help="恒 8.0 与 exp17/v12 对齐，勿改")
    parser.add_argument("--max-tcp", type=float, default=1.8)
    parser.add_argument("--terminal-exempt-steps", type=int, default=0)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--softmin-beta", type=float, default=5.0)
    parser.add_argument("--target-speed", type=float, default=1.8)
    parser.add_argument("--normal-weight", type=float, default=500000.0)
    parser.add_argument("--backswing", type=float, default=0.6)
    parser.add_argument("--bs-ratio", type=float, default=0.35)
    parser.add_argument("--r-decay", type=float, default=0.40)
    parser.add_argument("--no-backswing", action="store_true")
    parser.add_argument("--no-follow-through", action="store_true")
    parser.add_argument("--follow-trigger", choices=["planned", "contact"],
                        default="planned")
    parser.add_argument("--hit-shift", type=float, default=0.0)
    parser.add_argument("--departure-steps", type=int, default=80,
                        help="击后离场窗口步数（×5 ms）")
    parser.add_argument("--cam-distance", type=float, default=3.5,
                        help="广角段相机距离（m）")
    parser.add_argument("--cam-elevation", type=float, default=-15,
                        help="广角段相机俯仰（度）")
    parser.add_argument("--cam-azimuth", type=float, default=135,
                        help="广角段相机方位角（度）")
    # 合成参数
    parser.add_argument("--realtime-only", action="store_true",
                        help="整段常速输出（S5 素材），不做慢放")
    parser.add_argument("--slow-pre-ms", type=int, default=350)
    parser.add_argument("--slow-post-ms", type=int, default=550)
    parser.add_argument("--normal-post-ms", type=int, default=2000,
                        help="A 段常速播放到接触后多久（默认覆盖到片尾）")
    parser.add_argument("--slow-factor", type=int, default=4)
    parser.add_argument("--slow-cam-distance", type=float, default=2.2,
                        help="慢放推近相机距离（m）")
    parser.add_argument("--slow-cam-azimuth", type=float, default=112.0,
                        help="慢放推近相机方位角（度）")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    logger.info("[render] seed=%d speed=%.0f ablation=%s …",
                args.seed, args.ball_speed, args.ablation)
    ep = run_episode(args)
    logger.info("[render] 完成: hit_step=%d hit_type=%s pos_error=%.1fmm "
                "v_racket=%.2fm/s frames=%d (%.0fs)",
                ep["hit_step"], ep["hit_type"], ep["pos_error"] * 1000,
                ep["v_racket_at_hit"], len(ep["frames"]), time.perf_counter() - t0)
    if args.expect and ep["hit_type"] != args.expect:
        logger.warning("[核验] hit_type=%s 与期望 %s 不符！渲染管线与 v12 存在漂移，"
                       "该镜头应改用 npz 离线重渲染兜底", ep["hit_type"], args.expect)

    info = compose(args, ep)
    sheet = save_sheet(args, ep)
    meta = {
        "seed": args.seed, "ball_speed": args.ball_speed,
        "ablation": args.ablation, "hit_step": ep["hit_step"],
        "hit_type": ep["hit_type"], "pos_error": ep["pos_error"],
        "v_racket_at_hit": ep["v_racket_at_hit"], "dt": ep["dt"],
        "n_frames": len(ep["frames"]), **info,
    }
    (OUT_DIR / f"{args.label}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[compose] %s | sheet=%s", info, sheet)


if __name__ == "__main__":
    main()
