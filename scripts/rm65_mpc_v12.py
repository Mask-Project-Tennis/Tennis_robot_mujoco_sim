#!/usr/bin/env python3
"""RM-65 V12 — EpisodeRunner 管线架构。

V11 的规划逻辑提取到 MPCController（src/ilqt/mpc_controller.py），
仿真执行/诊断封装到 _V12SimComponent，
管线编排由 EpisodeRunner（src/ilqt/episode_runner.py）驱动。
V12 是组装器 + 球生成 + 评估。

=== V12 vs V11 ===
  - 规划逻辑（~800 行 replan + buffer + follow-through）→ MPCController
  - 安全滤波（~130 行 X-wall + safety filter）→ PredictiveSafetyFilter
  - 主循环编排 → EpisodeRunner.run()
  - 仿真专属（碰撞/接触/PD推回/诊断指标）→ _V12SimComponent
  - argparse / 配置 / 球生成 / 评估 → 保留在 V12

用法（与 V11 一致）:
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --seed 42
    python scripts/rm65_mpc_v12.py --serve-box --ball-speed 7 --no-plot
"""

from __future__ import annotations

import sys
import time
import argparse
import logging
from pathlib import Path

import numpy as np
import yaml

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("src.ilqt.robot_limits").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ==============================================================================
# 配置加载工具（从 V11 复制）
# ==============================================================================

def load_config(config_path: Path) -> dict:
    """加载 YAML 配置文件。"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def merge_configs(base: dict, override: dict) -> dict:
    """递归合并两个配置字典，override 覆盖 base。"""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


# ==============================================================================
# 可视化（从 V11 复用）
# ==============================================================================

def _import_v11_visuals():
    """惰性导入 V11 的可视化函数（避免无 --viewer/--plot 时的导入开销）。"""
    v11_path = Path(__file__).resolve().parent / "rm65_mpc_v11.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("rm65_mpc_v11", v11_path)
    assert spec is not None and spec.loader is not None, f"无法加载 V11 模块: {v11_path}"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.visualize_rm65_result, mod.plot_tube_results


# ==============================================================================
# V12 仿真执行+诊断一体化组件
# ==============================================================================

class _V12SimComponent:
    """V12 仿真执行+诊断一体化组件（实现 Executor + Diagnostics Protocol）。

    封装 V11 主循环中 step_full 前后的所有仿真专属逻辑：

      - PRE-step 诊断（distances / normal_align / ball_near / tube_ready）
      - 碰撞窗口管理（set_arm_collision）
      - 前向物理步进（step_full）
      - X 平面墙 PD 推回
      - 接触检测 + 弹性反弹
      - 执行层指标累积（max_qdot / max_tcp / max_face）
      - history 记录（X / U / ball_pos）

    对齐 V11 ``rm65_mpc_v11.py`` 1428-2123 行的逐步仿真逻辑。
    """

    def __init__(
        self,
        env: RM65Env,
        mpc: MPCController,
        robot_limits: RobotLimits,
        init_q: np.ndarray,
        is_position_mode: bool,
        hard_x_body_ids: list[int],
        initial_hitting_tube: HittingTube | None,
        tube_cfg: TubeConfig,
        dt: float,
    ) -> None:
        """初始化仿真组件。

        Args:
            env: RM65Env 仿真环境。
            mpc: MPCController（读取 _k_hit / _p_hit / _n_des）。
            robot_limits: 关节约束参数。
            init_q: 初始关节角度（PD 推回用）。
            is_position_mode: 位置模式标志。
            hard_x_body_ids: X 平面墙刚体 IDs。
            initial_hitting_tube: 初始候选窗口（tube_ready 诊断用）。
            tube_cfg: Tube 配置。
            dt: 仿真时间步长。
        """
        self._env = env
        self._mpc = mpc
        self._limits = robot_limits
        self._init_q = init_q
        self._is_position_mode = is_position_mode
        self._hard_x_body_ids = hard_x_body_ids
        self._initial_tube = initial_hitting_tube
        self._tube_cfg = tube_cfg
        self._dt = dt
        self._nu = env.NU
        self._nq = env.NQ

        # history 列表
        self.X_history: list[np.ndarray] = []
        self.U_history: list[np.ndarray] = []
        self.ball_pos_history: list[np.ndarray] = []
        self.distances_history: list[float] = []
        self.normal_align_history: list[float] = []
        self.ball_near_history: list[bool] = []
        self.tube_ready_history: list[bool] = []
        self.pos_error_history: list[float] = []

        # 执行层指标
        self.max_qdot_ratio: float = 0.0
        self.max_tcp_speed: float = 0.0
        self.max_racket_face_speed: float = 0.0
        self.total_mpc_steps: int = 0
        self.emergency_stop_count: int = 0
        self.buffer_exhaustion_count: int = 0

        # 击球追踪
        self.ball_was_hit: bool = False
        self.hit_step: int = -1
        self.p_ee_at_hit: np.ndarray | None = None
        self.ball_pos_at_hit: np.ndarray | None = None
        self.v_ee_at_hit: float | None = None
        self.active_contact: bool = False
        self.passive_contact: bool = False

        self._step_count: int = 0

    # ── ExecutorComponent Protocol ──────────────────────────────────────

    def get_arm_state(self) -> np.ndarray:
        """返回当前右臂状态 [q(6), qdot(6)]，形状 (12,)。"""
        return self._env.get_arm_state()

    def execute(self, u_cmd: np.ndarray) -> None:
        """执行控制指令 — 封装 V11 逐步仿真逻辑。

        Args:
            u_cmd: 安全滤波后的控制指令 (6,)。
        """
        env = self._env
        step = self._step_count
        k_hit = self._mpc._k_hit
        p_hit = self._mpc._p_hit.copy()

        # ---- PRE-step 诊断（V11 1428-1471）----
        env.update_kinematics()
        ball_pos = env.get_ball_pos()
        p_ee_cur = env.get_ee_pos()
        n_rack_cur = env.get_ee_normal()
        dist_cur = float(np.linalg.norm(p_ee_cur - ball_pos))
        self.distances_history.append(dist_cur)

        # normal_align
        if self._initial_tube is not None and len(self._initial_tube.n_racket_des) > 0:
            n_des_cur = self._initial_tube.n_racket_des[0]
        else:
            n_des_cur = self._mpc._n_des
        n_align = float(n_rack_cur @ n_des_cur)
        self.normal_align_history.append(n_align)

        # ball_near
        is_ball_near = (dist_cur < 0.033 + 0.12 + 0.03) and (abs(n_align) > 0.7)
        self.ball_near_history.append(is_ball_near)

        # tube_ready
        is_tube_ready = False
        if self._initial_tube is not None and len(self._initial_tube.p_ball) > 0:
            window_half_steps = int(round(self._tube_cfg.window_half_ms / 1000.0 / self._dt))
            tube_center = self._initial_tube.best_k
            if abs(step - tube_center) <= window_half_steps:
                v_ball_mean = np.mean(self._initial_tube.v_ball, axis=0)
                v_norm_ball = np.linalg.norm(v_ball_mean)
                d_ball = v_ball_mean / v_norm_ball if v_norm_ball > 1e-6 else np.array([0.0, -1.0, 0.0])
                P_perp = np.eye(3) - np.outer(d_ball, d_ball)
                best_idx = int(np.argmin(np.abs(self._initial_tube.k_candidates - tube_center)))
                p_ref = self._initial_tube.p_ball[best_idx]
                dp = p_ee_cur - p_ref
                perp_dist = float(np.linalg.norm(P_perp @ dp))
                if perp_dist < 0.15 and abs(n_align) > 0.7:
                    is_tube_ready = True
        self.tube_ready_history.append(is_tube_ready)

        # ---- 碰撞窗口管理（V11 1914-1923）----
        enable_collision = False
        if not self.ball_was_hit:
            if k_hit <= 30 and dist_cur < 0.35:
                enable_collision = True
            elif k_hit <= 10:
                enable_collision = True
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(enable_collision)

        # 记录碰撞前的球速度（用于弹性反弹计算）
        ball_vel = env.get_ball_vel()
        ball_vel_before_step = ball_vel.copy() if enable_collision else ball_vel

        # ---- 前向物理步进 ----
        x_new, ball_pos_new, ball_vel_new = env.step_full(u_cmd)

        # ---- POST-step X 平面墙 PD 推回（V11 2011-2034）----
        env.update_kinematics()
        violated = [
            bid for bid in self._hard_x_body_ids
            if env.data.xpos[bid, 0] > -0.1
        ]
        if violated:
            q_now = x_new[: self._nq]
            qdot_now = x_new[self._nq:]
            if self._is_position_mode:
                u_push = self._init_q.copy()
            else:
                u_push = 300.0 * (self._init_q - q_now) - 20.0 * qdot_now
                ctrl_lo = env.model.actuator_ctrlrange[: self._nu, 0]
                ctrl_hi = env.model.actuator_ctrlrange[: self._nu, 1]
                u_push = np.clip(u_push, ctrl_lo, ctrl_hi)
            x_new, ball_pos_new, ball_vel_new = env.step_full(u_push)

        # ---- 接触检测（V11 2036-2062）----
        ball_racket_hit = False
        if enable_collision and not self.ball_was_hit:
            n_contacts = env.data.ncon
            if n_contacts > 0:
                for ci in range(n_contacts):
                    c = env.data.contact[ci]
                    g1 = env.model.geom(c.geom1).name
                    g2 = env.model.geom(c.geom2).name
                    if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
                        ball_racket_hit = True
                        ee_vel = env.get_ee_vel()
                        ee_speed = float(np.linalg.norm(ee_vel))
                        self.v_ee_at_hit = ee_speed
                        if ee_speed > 0.3:
                            self.active_contact = True
                        else:
                            self.passive_contact = True
                        break

        # ---- 碰撞恢复 ----
        if hasattr(env, "set_arm_collision"):
            env.set_arm_collision(True)

        # ---- history 记录（V11 2067-2069）----
        self.X_history.append(x_new.copy())
        self.U_history.append(u_cmd.copy())
        self.ball_pos_history.append(ball_pos_new.copy())

        # ---- pos_error 记录（V11 2071-2073）----
        env.update_kinematics()
        pos_err = float(np.linalg.norm(env.get_ee_pos() - p_hit))
        self.pos_error_history.append(pos_err)

        # ---- 执行层指标（V11 2090-2098）----
        qdot_cur = x_new[self._nq:]
        qdot_ratio = float(np.max(np.abs(qdot_cur) / np.maximum(self._limits.qdot_max, 1e-8)))
        racket_speed = float(np.linalg.norm(env.get_ee_vel()))
        face_speed = env.get_racket_face_speed()
        self.max_qdot_ratio = max(self.max_qdot_ratio, qdot_ratio)
        self.max_tcp_speed = max(self.max_tcp_speed, racket_speed)
        self.max_racket_face_speed = max(self.max_racket_face_speed, face_speed)
        self.total_mpc_steps += 1

        # ---- 击球追踪 + 弹性反弹（V11 2105-2123）----
        if ball_racket_hit and not self.ball_was_hit:
            self.ball_was_hit = True
            self.hit_step = step
            env.update_kinematics()
            self.p_ee_at_hit = env.get_ee_pos().copy()
            self.ball_pos_at_hit = ball_pos_new.copy()
            n_racket = env.get_ee_normal()
            n_hat = n_racket / (np.linalg.norm(n_racket) + 1e-8)
            v_ee = env.get_ee_vel()
            v_ball_pre = ball_vel_before_step
            v_rel_n = np.dot(v_ball_pre - v_ee, n_hat)
            e = 0.8
            v_ball_rebound = v_ball_pre - (1 + e) * v_rel_n * n_hat
            env.set_ball_vel(v_ball_rebound)

        self._step_count += 1

    # ── DiagnosticsComponent Protocol ───────────────────────────────────

    def record(self, result: object, arm_state: np.ndarray) -> None:
        """Diagnostics Protocol — 记录已在 execute 中完成，此处为 no-op。"""
        pass

    def get_metrics(self) -> dict:
        """返回汇总指标字典。"""
        return {
            "X_history": self.X_history,
            "U_history": self.U_history,
            "ball_pos_history": self.ball_pos_history,
            "distances_history": self.distances_history,
            "normal_align_history": self.normal_align_history,
            "ball_near_history": self.ball_near_history,
            "tube_ready_history": self.tube_ready_history,
            "pos_error_history": self.pos_error_history,
            "max_qdot_ratio": self.max_qdot_ratio,
            "max_tcp_speed": self.max_tcp_speed,
            "max_racket_face_speed": self.max_racket_face_speed,
            "total_mpc_steps": self.total_mpc_steps,
            "emergency_stop_count": self.emergency_stop_count,
            "buffer_exhaustion_count": self.buffer_exhaustion_count,
            "ball_was_hit": self.ball_was_hit,
            "hit_step": self.hit_step,
            "p_ee_at_hit": self.p_ee_at_hit,
            "ball_pos_at_hit": self.ball_pos_at_hit,
            "v_ee_at_hit": self.v_ee_at_hit,
            "active_contact": self.active_contact,
            "passive_contact": self.passive_contact,
        }


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
    parser.add_argument("--Q-tcp-soft", type=float, default=5000.0, help="TCP 速度软惩罚权重")
    parser.add_argument("--Q-qdot-limit", type=float, default=1000.0, help="关节速度软惩罚权重")
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
    parser.add_argument("--dump-trajectory", type=str, default=None, help="保存轨迹到 pickle")
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
    parser.add_argument("--terminal-exempt-steps", type=int, default=0, help="终段豁免步数")
    parser.add_argument("--obs-freq", type=float, default=0, help="观测频率 Hz")
    parser.add_argument("--obs-noise-pos", type=float, default=0, help="观测位置噪声 std (m)")
    parser.add_argument("--obs-noise-vel", type=float, default=0, help="观测速度噪声 std (m/s)")
    parser.add_argument("--obs-use-kf", action="store_true", help="观测启用 KF 滤波")
    parser.add_argument("--position-mode", action="store_true", help="启用位置控制模式")
    parser.add_argument("--no-feedforward", action="store_true", help="禁用前馈补偿")
    parser.add_argument("--kp", type=float, nargs='+', default=None, help="位置模式 PD Kp")
    parser.add_argument("--kd", type=float, nargs='+', default=None, help="位置模式 PD Kd")
    parser.add_argument("--dq-max-fraction", type=float, default=None, help="单步角度变化系数")
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

    shoulder_pos = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
    workspace_radius = 0.90

    # ==========================================================================
    # 4. 时间参数推导（与 V11 一致）
    # ==========================================================================
    total_horizon = 200
    fixed_horizon = 40
    replan_interval = args.replan_interval if args.replan_interval is not None else 30
    max_iter_per_plan = 5
    first_plan_iters = 15
    near_plan_iters = 20
    Q_p_scale_far = 5.0
    Q_v_scale_far = 3.0
    Q_p_scale_near = 8.0
    Q_v_scale_near = 120.0

    if args.near_iters is not None:
        near_plan_iters = args.near_iters
    if args.horizon is not None:
        fixed_horizon = args.horizon
    if args.iter is not None:
        max_iter_per_plan = args.iter
    if args.replan_interval is not None:
        replan_interval = args.replan_interval

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
    init_q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
    init_q_left = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)

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

    rl_cfg = config_dict.get("robot_limits", {})
    if args.max_tcp is not None:
        rl_cfg["max_tcp_speed"] = float("inf") if args.max_tcp == 0 else args.max_tcp
    if args.terminal_exempt_steps is not None:
        rl_cfg["terminal_exempt_steps"] = args.terminal_exempt_steps
    if args.dq_max_fraction is not None:
        rl_cfg["dq_max_fraction"] = args.dq_max_fraction
    robot_limits = RobotLimits.from_config(
        rl_cfg, dt=dt, ctrlrange=env.model.actuator_ctrlrange[:env.NU],
    )

    # X 平面墙 body IDs 缓存
    import mujoco as _mj
    _hard_x_body_ids = [
        _mj.mj_name2id(env.model, _mj.mjtObj.mjOBJ_BODY, n)
        for n in ("r_link1", "r_link2", "r_link3", "r_link4",
                   "r_link5", "r_link6", "r_flange", "r_racket_body")
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

    near_threshold = max(50, k_hit_total // 3)
    far_threshold = 50

    hit_direction = np.array(config_dict["hitting"]["hit_direction"], dtype=np.float64)
    racket_speed = float(config_dict["hitting"]["racket_speed"])

    if args.fixed_direction:
        d_follow = hit_direction / (np.linalg.norm(hit_direction) + 1e-8)
    else:
        d_follow = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    _d_hat = d_follow
    v_hit_at_contact = args.target_speed * d_follow

    if args.no_follow_through:
        follow_through_length = 0.0
        follow_through_steps = 0
        follow_through_v_terminal = 0.0
    else:
        follow_through_length = args.hit_shift
        follow_through_steps = int(config_dict["hitting"].get("follow_through_steps", 160))
        follow_through_v_terminal = float(config_dict["hitting"].get("follow_through_v_terminal", 0.3))

    _hit_shift = follow_through_length
    _v_hit_desired = v_hit_at_contact

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
        max_tcp_speed=float(args.max_tcp) if args.max_tcp and args.max_tcp > 0 else 1.8,
        terminal_exempt_steps=rl_cfg.get("terminal_exempt_steps", 0),
        dq_max_fraction=rl_cfg.get("dq_max_fraction", 0.5),
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
    )

    mpc = MPCController(env, mpc_config)
    # 覆盖 MPCController 内部的 robot_limits 为配置派生值（对齐 V11）
    mpc._robot_limits = robot_limits

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
    sim_component = _V12SimComponent(
        env, mpc, robot_limits, init_q, is_position_mode,
        _hard_x_body_ids, initial_hitting_tube, tube_cfg, dt,
    )
    # 初始化 history（与 V11 一致：X_history[0] = x0）
    sim_component.X_history = [x0.copy()]
    sim_component.ball_pos_history = [p0.copy()]

    runner = EpisodeRunner(
        mpc=mpc,
        perception=perception,
        safety=safety,
        executor=sim_component,
        diagnostics=sim_component,
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
    for _ in range(post_hit_steps):
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

    # ==========================================================================
    # 15. 评估（对齐 V11 2245-2505）
    # ==========================================================================
    t_total = time.perf_counter() - t_total_start
    t_mpc = t_mpc_end - t_total_start
    _n_mpc_steps = len(sim_component.U_history) - post_hit_steps

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
    d_arr = np.array(distances_history) if distances_history else np.array([float("inf")])
    min_dist = float(np.min(d_arr))
    ball_near_duration = int(np.sum(np.array(ball_near_history, dtype=bool))) if ball_near_history else 0
    ball_near_ms = ball_near_duration * dt * 1000
    tube_ready_duration = int(np.sum(np.array(tube_ready_history, dtype=bool))) if tube_ready_history else 0
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
    ball_racket_threshold = 0.033 + 0.12
    print("\n========================================")
    if pos_error < 0.05:
        print("  RM-65 击打成功！（球-拍距离 < 5cm，精准命中）")
    elif pos_error < ball_racket_threshold:
        print(f"  RM-65 击打命中！（球-拍距离 {pos_error:.4f}m < {ball_racket_threshold:.3f}m）")
    elif pos_error < 0.1:
        print("  RM-65 击打接近！（球-拍距离 < 10cm）")
    else:
        print("  RM-65 击打偏差较大，需要调整参数。")

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
    print(f"  emergency_stop={sim_component.emergency_stop_count}")
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
    if args.dump_trajectory:
        import pickle as _pickle
        traj_data = {
            "X_history": X_history,
            "U_history": U_history,
            "ball_pos_history": ball_pos_history,
            "init_q": init_q,
            "init_q_left": init_q_left,
            "pos_error": pos_error,
            "hit_type": hit_type_en,
            "p0": p0_real,
            "v0": v0_real,
            "hit_step": hit_step,
            "post_hit_steps": post_hit_steps,
        }
        dump_path = Path(args.dump_trajectory)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dump_path, "wb") as _f:
            _pickle.dump(traj_data, _f)

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
                if i < len(X_history):
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
        except Exception as e:
            logger.warning(f"可视化失败: {e}")

    if args.viewer:
        try:
            visualize_fn, _ = _import_v11_visuals()
            _X_arr = np.array(X_history)
            U_arr = np.array(U_history) if len(U_history) > 0 else np.zeros((0, env.NU))
            env.reset(init_q)
            env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = init_q_left
            env.data.qvel[env.NQ:env.NQ + env.LEFT_ARM_NQ] = 0.0
            env.update_kinematics()
            env.set_ball_state(p0, v0)
            ball_replay = [env.get_ball_pos().copy()]
            X_replay = [env.get_arm_state().copy()]
            for u_cmd in U_arr:
                env.step(u_cmd)
                X_replay.append(env.get_arm_state().copy())
                ball_replay.append(env.get_ball_pos().copy())
            if hasattr(env, "set_arm_collision"):
                env.set_arm_collision(True)
            visualize_fn(
                env, np.array(X_replay), U_arr, np.array(ball_replay), config_dict,
                init_q_left=init_q_left,
                post_hit_steps=post_hit_steps,
            )
        except Exception as e:
            logger.warning(f"回放可视化失败: {e}")


if __name__ == "__main__":
    main()
