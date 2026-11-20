"""可视化工具：回放轨迹（支持 MuJoCo 查看器和 matplotlib 绘图）。"""

import time
import logging
from dataclasses import dataclass

import numpy as np
import mujoco
import mujoco.viewer
from src.sim.env import MujocoEnv
from src.tennis.ball import ball_trajectory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlaybackLayout:
    """回放数据切分布局（支持 RM65 双臂 6 关节与 R1D 单臂 6/7 关节）。

    状态向量 ``X`` 恒为 ``[q(n), qdot(n)]``（n = 主动关节数），与 MuJoCo
    的 qpos 地址无关：主动关节在 qpos 中的地址由 ``arm_qpos_indices``
    给出（RM65 为 0..5；R1D 场景球体在 worldbody 首位时可能为 7..13），
    球的 free joint 地址单独记录。

    Attributes:
        joint_count: 主动关节数 n（``X.shape[1] // 2``）。
        arm_qpos_indices: 主动关节在 ``data.qpos`` 中的地址（长度 n）。
        arm_qvel_indices: 主动关节在 ``data.qvel`` 中的地址（长度 n）。
        ball_qpos_start: 球 free joint 的 qpos 起点（xyz + 四元数）。
        ball_qvel_start: 球 free joint 的 qvel 起点（线速度 + 角速度）。
    """

    joint_count: int
    arm_qpos_indices: tuple[int, ...]
    arm_qvel_indices: tuple[int, ...]
    ball_qpos_start: int
    ball_qvel_start: int


def resolve_playback_layout(env: object, frame_width: int) -> PlaybackLayout:
    """按环境暴露的地址信息解析回放布局（不假设臂从 qpos[0] 开始）。

    Args:
        env: MuJoCo 环境（RM65Env / ProfileSimEnv / 测试 fake 环境）。
        frame_width: 单帧状态向量宽度（``X.shape[1]``），等于 ``2 * n``。

    Returns:
        :class:`PlaybackLayout`。

    Raises:
        ValueError: ``frame_width`` 不是正偶数。
    """
    width = int(frame_width)
    if width <= 0 or width % 2 != 0:
        raise ValueError(f"单帧状态宽度必须是正偶数，收到 {frame_width}")
    joint_count = width // 2

    qpos_indices = getattr(env, "_qpos_indices", None)
    qvel_indices = getattr(env, "_qvel_indices", None)
    if qpos_indices is None or len(qpos_indices) != joint_count:
        arm_qpos = tuple(range(joint_count))
    else:
        arm_qpos = tuple(int(index) for index in np.asarray(qpos_indices).ravel())
    if qvel_indices is None or len(qvel_indices) != joint_count:
        arm_qvel = tuple(range(joint_count))
    else:
        arm_qvel = tuple(int(index) for index in np.asarray(qvel_indices).ravel())

    ball_start: int | None = None
    for attr in ("BALL_QPOS_START", "_ball_qpos_start"):
        value = getattr(env, attr, None)
        if value is not None:
            ball_start = int(value)
            break
    if ball_start is None:
        ball_start = int(getattr(env, "NQ", joint_count))

    ball_vel_start: int | None = None
    for attr in ("BALL_QVEL_START", "_ball_qvel_start"):
        value = getattr(env, attr, None)
        if value is not None:
            ball_vel_start = int(value)
            break
    if ball_vel_start is None:
        ball_vel_start = ball_start

    return PlaybackLayout(
        joint_count=joint_count,
        arm_qpos_indices=arm_qpos,
        arm_qvel_indices=arm_qvel,
        ball_qpos_start=ball_start,
        ball_qvel_start=ball_vel_start,
    )


def arm_state_from_frame(frame: np.ndarray, layout: PlaybackLayout) -> np.ndarray:
    """从单帧状态向量中切出主动臂状态 ``[q(n), qdot(n)]``。

    Args:
        frame: 形状 ``(2n,)`` 的状态向量。
        layout: 回放布局。

    Returns:
        形状 ``(2 * joint_count,)`` 的主动臂状态。
    """
    n = layout.joint_count
    state = np.asarray(frame, dtype=float)
    return np.concatenate([state[:n], state[n:2 * n]])


def write_playback_state(
    data: mujoco.MjData,
    layout: PlaybackLayout,
    joint_positions: np.ndarray,
    joint_velocities: np.ndarray,
    ball_position: np.ndarray,
) -> None:
    """把一帧状态写进 ``data``（按布局地址，兼容 nq=6/7 与球在 qpos 首位）。

    Args:
        data: MuJoCo data。
        layout: 回放布局。
        joint_positions: 主动关节角 ``(n,)``。
        joint_velocities: 主动关节速度 ``(n,)``。
        ball_position: 球位置 ``(3,)``。
    """
    qpos_indices = np.asarray(layout.arm_qpos_indices, dtype=int)
    qvel_indices = np.asarray(layout.arm_qvel_indices, dtype=int)
    data.qpos[qpos_indices] = np.asarray(joint_positions, dtype=float)
    data.qvel[qvel_indices] = np.asarray(joint_velocities, dtype=float)

    bq = int(layout.ball_qpos_start)
    bv = int(layout.ball_qvel_start)
    data.qpos[bq:bq + 3] = np.asarray(ball_position, dtype=float)
    data.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[bv:bv + 3] = 0.0
    data.qvel[bv + 3:bv + 6] = 0.0


def apply_playback_frame(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frame: np.ndarray,
    layout: PlaybackLayout,
    ball_position: np.ndarray,
) -> None:
    """写入一帧并 ``mj_forward``（供查看器/测试复用）。"""
    n = layout.joint_count
    state = np.asarray(frame, dtype=float)
    write_playback_state(data, layout, state[:n], state[n:2 * n], ball_position)
    mujoco.mj_forward(model, data)


def visualize_result(
    env: MujocoEnv,
    X: np.ndarray,
    U: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    g: np.ndarray,
    config: dict,
    ball_positions_phys: np.ndarray | None = None,
    *,
    init_q_left: np.ndarray | None = None,
    extra_frames: int = 30,
) -> None:
    """在 MuJoCo 查看器中可视化 iLQT 优化结果。

    同时显示机械臂运动和网球飞行轨迹。按 :func:`resolve_playback_layout`
    解析布局，支持 RM65（双臂 6 主动关节 + 左臂随动）与 R1D（nq=6/7 单臂）。

    Args:
        env: MuJoCo 环境实例。
        X: 状态轨迹，形状 (N+1, 2n)。
        U: 控制轨迹，形状 (N, n)。
        p0: 球初始位置，形状 (3,)。
        v0: 球初始速度，形状 (3,)。
        g: 重力加速度，形状 (3,)。
        config: 可视化配置字典。
        ball_positions_phys: MuJoCo 物理仿真得到的球轨迹，形状 (M, 3)。
            若提供则优先使用（比解析公式更真实），否则用解析抛物线公式。
        init_q_left: RM65 左臂随动关节角（长度 6）；未提供时保持模型初值。
        extra_frames: 轨迹播完后的额外帧数（默认 30，保持旧行为）。
    """
    N = len(U)
    dt = env.dt
    viewer_cfg = config.get("viewer", {})
    playback_speed = viewer_cfg.get("playback_speed", 1.0)
    loop = viewer_cfg.get("loop", True)

    cam_distance = viewer_cfg.get("camera_distance", 2.5)
    cam_elevation = viewer_cfg.get("camera_elevation", -20)
    cam_azimuth = viewer_cfg.get("camera_azimuth", 135)

    # 球轨迹：优先使用 MuJoCo 物理仿真结果，否则回退到解析公式
    total_frames = N + max(int(extra_frames), 0)
    ball_positions = np.zeros((total_frames, 3))
    if ball_positions_phys is not None and len(ball_positions_phys) > 0:
        n_phys = min(len(ball_positions_phys), total_frames)
        ball_positions[:n_phys] = ball_positions_phys[:n_phys]
        # 剩余帧用最后位置填充
        if n_phys < total_frames:
            ball_positions[n_phys:] = ball_positions_phys[-1]
    else:
        for k in range(total_frames):
            ball_positions[k] = ball_trajectory(p0, v0, g, k * dt)

    data = env.data
    model = env.model
    layout = resolve_playback_layout(env, int(np.asarray(X).shape[1]))

    # RM65 左臂随动（旧行为：由调用方传入初始左臂角）
    if init_q_left is not None:
        left = np.asarray(init_q_left, dtype=float).ravel()
        start = layout.joint_count
        data.qpos[start:start + left.size] = left
        data.qvel[start:start + left.size] = 0.0

    # 用墙钟时间驱动帧索引，确保真实速度播放
    start_time = time.perf_counter()
    last_idx = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = cam_distance
        viewer.cam.elevation = cam_elevation
        viewer.cam.azimuth = cam_azimuth
        viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

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
                arm_x = X[idx] if idx <= N else X[-1]
                apply_playback_frame(
                    model, data, arm_x, layout, ball_positions[idx]
                )

            viewer.sync()
            time.sleep(1.0 / 120.0)


class RealTimeViewer:
    """MPC 实时可视化查看器。

    在 MPC 循环中持续运行，每一步调用 sync() 更新画面。
    支持 overlay 文字显示剩余时间、误差等信息。
    """

    def __init__(
        self,
        env: MujocoEnv,
        config: dict,
    ) -> None:
        """初始化实时查看器。

        Args:
            env: MuJoCo 环境实例。
            config: 配置字典。
        """
        self.env = env
        viewer_cfg = config.get("viewer", {})

        self.cam_distance = viewer_cfg.get("camera_distance", 3.5)
        self.cam_elevation = viewer_cfg.get("camera_elevation", -15)
        self.cam_azimuth = viewer_cfg.get("camera_azimuth", 135)
        self.cam_lookat = viewer_cfg.get("camera_lookat", [0.0, 0.0, 1.0])

        self._viewer = None

    def start(self) -> None:
        """启动查看器窗口。"""
        self._viewer = mujoco.viewer.launch_passive(
            self.env.model, self.env.data
        )
        self._viewer.cam.distance = self.cam_distance
        self._viewer.cam.elevation = self.cam_elevation
        self._viewer.cam.azimuth = self.cam_azimuth
        self._viewer.cam.lookat[:] = self.cam_lookat

    def sync(self) -> None:
        """同步渲染一帧画面。"""
        if self._viewer is not None and self._viewer.is_running():
            mujoco.mj_forward(self.env.model, self.env.data)
            self._viewer.sync()

    def is_running(self) -> bool:
        """检查查看器是否仍在运行。"""
        if self._viewer is None:
            return False
        return self._viewer.is_running()

    def close(self) -> None:
        """关闭查看器。"""
        self._viewer = None


def control_plot_labels(
    joint_count: int,
    joint_names: list[str],
    control_label: str,
) -> tuple[str, list[str], str]:
    """返回控制量图的 (总标题, 子图标题列表, y 轴标签)（评审 N2 修复）。

    n==6（RM65 默认 6 关节）保持旧输出逐字不变：``Control Torques`` /
    ``tau_<joint>`` / ``Torque (N*m)``；其余关节数（如 R1D 7 关节或位置
    模式）使用动态 ``Control Inputs`` / ``u_<joint>`` / ``control_label``。

    Args:
        joint_count: 控制维度 n。
        joint_names: 关节名（长度应为 n）。
        control_label: 非默认分支使用的 y 轴标签（由调用方按模式给出）。

    Returns:
        ``(suptitle, subplot_titles, ylabel)``。
    """
    if int(joint_count) == 6:
        return (
            "Control Torques",
            [f"tau_{joint_names[index]}" for index in range(joint_count)],
            "Torque (N*m)",
        )
    return (
        "Control Inputs",
        [f"u_{joint_names[index]}" for index in range(joint_count)],
        control_label,
    )


def plot_results(
    X: np.ndarray,
    U: np.ndarray,
    p0: np.ndarray,
    v0: np.ndarray,
    g: np.ndarray,
    p_hit: np.ndarray,
    v_hit: np.ndarray,
    cost_history: list[float],
    dt: float,
    save_path: str | None = None,
    show: bool = False,
    *,
    joint_names: list[str] | None = None,
    control_label: str = "Torque (N*m)",
    env: MujocoEnv | None = None,
) -> None:
    """用 matplotlib 绘制优化结果图表（支持 nq=6/7 动态关节数）。

    Args:
        X: 状态轨迹，形状 (N+1, 2n)。
        U: 控制轨迹，形状 (N, n)。
        p0: 球初始位置。
        v0: 球初始速度。
        g: 重力加速度。
        p_hit: 期望击打位置。
        v_hit: 期望击打速度。
        cost_history: 代价历史。
        dt: 时间步长。
        save_path: 图片保存路径前缀。若为 None 则自动保存到 results/。
        show: 是否交互显示（会阻塞）。
        joint_names: 关节名（长度必须为 n）；默认 RM65 6 关节旧文案。
        control_label: 控制量单位标签（torque 模式 N*m / position 模式 rad）。
        env: 3D 轨迹 FK 用环境（缺省沿用旧 RM65 单臂 model.xml）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if save_path is None:
        from pathlib import Path
        results_dir = Path(__file__).resolve().parent.parent.parent / "results"
        results_dir.mkdir(exist_ok=True)
        save_path = str(results_dir / "ilqt_result")

    N = len(U)
    n = int(np.asarray(X).shape[1]) // 2
    if joint_names is None:
        if n == 6:
            joint_names = [
                "shoulder_pan", "shoulder_lift", "elbow",
                "wrist_1", "wrist_2", "wrist_3",
            ]
        else:
            joint_names = [f"joint_{index + 1}" for index in range(n)]
    if len(joint_names) != n:
        raise ValueError(
            f"joint_names 长度({len(joint_names)}) 与状态维度 {2 * n} 不匹配"
        )
    n_columns = 3
    n_rows = int(np.ceil(n / n_columns))
    t_arr = np.arange(N + 1) * dt
    t_u = np.arange(N) * dt

    # 关节角度 + 速度
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle("iLQT Optimization Results", fontsize=14)
    for j in range(n):
        ax = axes_arr[j]
        ax.plot(t_arr, X[:, j], label=f"{joint_names[j]} (q)")
        ax.plot(t_arr, X[:, j + n], "--", label=f"{joint_names[j]} (qdot)")
        ax.set_title(joint_names[j])
        ax.set_xlabel("Time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    for j in range(n, axes_arr.size):
        axes_arr[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_path}_joints.png", dpi=150)
    plt.close()

    # 控制量（n==6 保持 RM65 旧文案；其它关节数用动态标签）
    control_title, control_titles, control_ylabel = control_plot_labels(
        n, joint_names, control_label
    )
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(15, 4 * n_rows))
    axes_arr = np.atleast_1d(axes).ravel()
    fig.suptitle(control_title, fontsize=14)
    for j in range(n):
        ax = axes_arr[j]
        ax.plot(t_u, U[:, j])
        ax.set_title(control_titles[j])
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(control_ylabel)
        ax.grid(True, alpha=0.3)
    for j in range(n, axes_arr.size):
        axes_arr[j].axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_path}_controls.png", dpi=150)
    plt.close()

    # 3D 轨迹（默认沿用旧 RM65 单臂 model.xml；维度不符时跳过 FK）
    from pathlib import Path

    env_temp = env
    if env_temp is None:
        model_path = Path(__file__).resolve().parent.parent / "robot" / "model.xml"
        env_temp = MujocoEnv(model_path, dt=dt)

    ee_positions = []
    for k in range(N + 1):
        try:
            env_temp.set_arm_state(X[k])
        except (ValueError, IndexError) as exc:
            logger.warning("3D 轨迹 FK 跳过（状态维度与 env 不匹配）：%s", exc)
            ee_positions = []
            break
        ee_positions.append(env_temp.get_ee_pos())
    if ee_positions:
        ee_positions = np.array(ee_positions)
    else:
        ee_positions = np.zeros((0, 3))

    ball_positions = []
    for k in range(N + 1):
        t = k * dt
        ball_positions.append(ball_trajectory(p0, v0, g, t))
    ball_positions = np.array(ball_positions)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")
    if ee_positions.size:
        ax.plot(ee_positions[:, 0], ee_positions[:, 1], ee_positions[:, 2],
                "b-", linewidth=2, label="Racket")
        ax.scatter(*ee_positions[-1], color="b", s=100, marker="o",
                   label="EE Final Pos")
    ax.plot(ball_positions[:, 0], ball_positions[:, 1], ball_positions[:, 2],
            "y--", linewidth=2, label="Tennis Ball")
    ax.scatter(*p_hit, color="r", s=100, marker="*", label="Hit Target")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("3D Trajectories")
    ax.legend()
    plt.savefig(f"{save_path}_3d.png", dpi=150)
    plt.close()

    # 代价收敛
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(cost_history, "b-", linewidth=1.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total Cost")
    ax.set_title("iLQT Cost Convergence")
    ax.grid(True, alpha=0.3)
    plt.savefig(f"{save_path}_cost.png", dpi=150)
    plt.close()

    print(f"Plots saved to {save_path}_*.png")

    if show:
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt2
        figs = []
        for suffix in ["_joints.png", "_controls.png", "_3d.png", "_cost.png"]:
            img = plt2.imread(f"{save_path}{suffix}")
            fig, ax = plt2.subplots(1, 1, figsize=(12, 8))
            ax.imshow(img)
            ax.axis("off")
            figs.append(fig)
        plt2.show()
