"""V11/V12 共享可视化函数。

从 scripts/rm65_mpc_v11.py 提取，供 V11 薄壳和 V12 共同使用。
"""

from __future__ import annotations

import time
import logging
from pathlib import Path

import numpy as np

from src.sim.rm65_env import RM65Env
from src.ilqt.tube_types import HitWindow

logger = logging.getLogger(__name__)


def visualize_rm65_result(
    env: RM65Env,
    X: np.ndarray,
    U: np.ndarray,
    ball_positions_phys: np.ndarray,
    config: dict,
    init_q_left: np.ndarray,
    post_hit_steps: int = 80,
) -> None:
    """在 MuJoCo 查看器中可视化 RM-65 击打结果（含击打后球飞出效果）。

    Args:
        env: RM-65 环境实例。
        X: 右臂状态轨迹，形状 (N+1, 12)。
        U: 控制轨迹，形状 (N, 6)。
        ball_positions_phys: MuJoCo 物理球轨迹，形状 (M, 3)。
        config: 可视化配置。
        init_q_left: 左臂初始关节角度。
        post_hit_steps: 击打后额外仿真步数。
    """
    import mujoco
    import mujoco.viewer

    N = len(U)
    dt = env.dt
    viewer_cfg = config.get("viewer", {})
    playback_speed = viewer_cfg.get("playback_speed", 1.0)
    loop = viewer_cfg.get("loop", True)

    cam_distance = viewer_cfg.get("camera_distance", 3.5)
    cam_elevation = viewer_cfg.get("camera_elevation", -15)
    cam_azimuth = viewer_cfg.get("camera_azimuth", 135)

    total_frames = len(ball_positions_phys)

    bq = env.BALL_QPOS_START
    NQ = env.NQ
    data = env.data
    model = env.model

    data.qpos[:NQ] = X[0, :NQ]
    data.qvel[:NQ] = X[0, NQ:]
    data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left
    data.qpos[bq:bq + 3] = ball_positions_phys[0]
    data.qpos[bq + 3:bq + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    last_idx = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = cam_elevation
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

        model.light_pos[0] = [0.0, 0.0, 8.0]
        model.light_dir[0] = [0.0, 0.0, -1.0]
        model.light_diffuse[0] = [1.4, 1.45, 1.55]
        model.light_ambient[0] = [0.3, 0.3, 0.35]
        model.light_specular[0] = [0.5, 0.5, 0.5]
        if model.nlight > 1:
            model.light_pos[1] = [2.0, -2.0, 3.0]
            model.light_dir[1] = [-0.4, 0.3, -0.8]
            model.light_diffuse[1] = [1.2, 1.15, 1.05]
            model.light_ambient[1] = [0.0, 0.0, 0.0]
            model.light_specular[1] = [0.6, 0.6, 0.6]
            model.light_active[1] = True
        if model.nlight > 2:
            model.light_pos[2] = [-1.5, -1.0, 2.5]
            model.light_dir[2] = [0.3, 0.2, -0.7]
            model.light_diffuse[2] = [0.8, 0.85, 0.95]
            model.light_ambient[2] = [0.0, 0.0, 0.0]
            model.light_specular[2] = [0.4, 0.4, 0.4]
            model.light_active[2] = True
        if model.nlight > 3:
            model.light_pos[3] = [0.0, 2.0, 2.0]
            model.light_dir[3] = [0.0, -0.5, -0.6]
            model.light_diffuse[3] = [0.5, 0.5, 0.55]
            model.light_ambient[3] = [0.0, 0.0, 0.0]
            model.light_specular[3] = [0.3, 0.3, 0.3]
            model.light_active[3] = True

        start_time = time.perf_counter()

        while viewer.is_running():
            elapsed = time.perf_counter() - start_time
            sim_time = elapsed * playback_speed
            idx = int(sim_time / dt)

            if idx >= total_frames:
                if loop:
                    start_time = time.perf_counter()
                    idx = 0
                else:
                    idx = total_frames - 1

            if idx != last_idx:
                last_idx = idx

                if idx <= N:
                    arm_x = X[idx]
                else:
                    arm_x = X[-1]

                data.qpos[:NQ] = arm_x[:NQ]
                data.qvel[:NQ] = arm_x[NQ:]
                data.qpos[NQ:NQ + env.LEFT_ARM_NQ] = init_q_left

                if idx < len(ball_positions_phys):
                    bp = ball_positions_phys[idx]
                    data.qpos[bq: bq + 3] = bp
                    data.qpos[bq + 3: bq + 7] = [1, 0, 0, 0]

                mujoco.mj_forward(model, data)

            viewer.sync()
            time.sleep(1.0 / 120.0)


def plot_tube_results(
    results_dir: Path,
    tag: str,
    ball_positions: np.ndarray,
    racket_positions: np.ndarray,
    hit_window: HitWindow | None,
    distances: list[float],
    normal_alignments: list[float],
    ball_near_flags: list[bool],
    tube_ready_flags: list[bool],
    k_hit_history: list[int],
    pos_errors: list[float],
) -> None:
    """保存 Tube 实验的可视化图像（非阻塞）。

    Args:
        results_dir: 输出目录。
        tag: 文件名标签。
        ball_positions: 球轨迹，形状 (N, 3)。
        racket_positions: 球拍中心轨迹，形状 (N, 3)。
        hit_window: 击球窗口。
        distances: 球拍-球距离序列。
        normal_alignments: 法向量对齐序列。
        ball_near_flags: 球物理上在拍附近的布尔序列。
        tube_ready_flags: 球拍在 tube 窗口内保持击球姿态的布尔序列。
        k_hit_history: 每步的 k_hit 估计。
        pos_errors: 位置误差序列。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib 未安装，跳过可视化")
        return

    results_dir.mkdir(parents=True, exist_ok=True)

    N = min(len(distances), len(normal_alignments), len(ball_near_flags), len(tube_ready_flags))
    t_axis = np.arange(N)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"Tube-Based Robust Hitting Results [{tag}]", fontsize=14)

    # 子图 1: 球轨迹 + 球拍中心轨迹 3D
    ax3d = fig.add_subplot(3, 2, 1, projection="3d")
    ax3d.plot(ball_positions[:len(racket_positions), 0],
              ball_positions[:len(racket_positions), 1],
              ball_positions[:len(racket_positions), 2], "b-", alpha=0.5, label="Ball")
    ax3d.plot(racket_positions[:, 0], racket_positions[:, 1], racket_positions[:, 2],
              "r-", alpha=0.8, label="Racket")
    if hit_window is not None and len(hit_window.p_ball_candidates) > 0:
        ax3d.scatter(hit_window.p_ball_candidates[:, 0],
                     hit_window.p_ball_candidates[:, 1],
                     hit_window.p_ball_candidates[:, 2],
                     c="orange", s=20, marker="o", label="Hit Window")
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.legend()
    ax3d.set_title("Ball & Racket Trajectory")

    # 子图 2: 球拍-球距离
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(t_axis, distances[:N], "g-", linewidth=1)
    ax2.axhline(y=0.033 + 0.12, color="gray", linestyle="--", label="Racket+ball radius")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Distance (m)")
    ax2.set_title("Racket-Ball Distance")
    ax2.legend()

    # 子图 3: 法向量对齐
    ax3 = fig.add_subplot(3, 2, 3)
    ax3.plot(t_axis, normal_alignments[:N], "m-", linewidth=1)
    ax3.axhline(y=0.9, color="gray", linestyle="--", label="90% alignment")
    ax3.set_xlabel("Step")
    ax3.set_ylabel("dot(n_rack, n_des)")
    ax3.set_title("Normal Alignment")
    ax3.set_ylim(-1.05, 1.05)
    ax3.legend()

    # 子图 4: ball_near vs tube_ready 双指标
    ax4 = fig.add_subplot(3, 2, 4)
    ax4.fill_between(t_axis, 0, np.array(ball_near_flags[:N], dtype=float),
                     step="mid", alpha=0.4, color="orange", label="ball_near")
    ax4.fill_between(t_axis, 0, np.array(tube_ready_flags[:N], dtype=float),
                     step="mid", alpha=0.3, color="cyan", label="tube_ready")
    ax4.set_xlabel("Step")
    ax4.set_ylabel("Flag")
    ax4.set_title("ball_near vs tube_ready")
    ax4.set_ylim(-0.1, 1.3)
    ax4.legend()

    # 子图 5: k_hit 估计变化
    ax5 = fig.add_subplot(3, 2, 5)
    ax5.plot(range(len(k_hit_history)), k_hit_history, "b.-", markersize=3)
    if hit_window is not None:
        ax5.axhline(y=hit_window.best_k, color="orange", linestyle="--", label=f"best_k={hit_window.best_k}")
        k_low = hit_window.k_candidates[0] if len(hit_window.k_candidates) > 0 else hit_window.best_k
        k_high = hit_window.k_candidates[-1] if len(hit_window.k_candidates) > 0 else hit_window.best_k
        ax5.fill_between(range(len(k_hit_history)), k_low, k_high, alpha=0.15, color="green", label="Hit Window")
    ax5.set_xlabel("Replan")
    ax5.set_ylabel("k_hit")
    ax5.set_title("Hitting Step Estimation")
    ax5.legend()

    # 子图 6: 位置误差
    ax6 = fig.add_subplot(3, 2, 6)
    ax6.plot(range(len(pos_errors)), pos_errors, "r.-", markersize=3)
    ax6.set_xlabel("Step")
    ax6.set_ylabel("Position Error (m)")
    ax6.set_title("Position Error over Time")
    ax6.axhline(y=0.05, color="gray", linestyle="--", label="5cm")
    ax6.legend()

    plt.tight_layout()
    out_path = results_dir / f"tube_results_{tag}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    logger.info(f"可视化已保存到 {out_path}")
