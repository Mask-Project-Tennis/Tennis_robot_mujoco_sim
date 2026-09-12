"""轨迹记录器 — 仿真侧记录规划/执行轨迹，真机侧可重演。

TrajectoryRecorder 可作为 post_exec_hook 注入 EpisodeRunner（仿真侧），
也可在真机重演循环中手动调用（真机侧记录实际轨迹）。

保存格式:
    - 新格式 .npz: np.savez 保存所有字段，metadata 用 json.dumps 序列化。
    - 旧格式 .pkl: 兼容 rm65_mpc_v12.py --dump-trajectory 输出的 pickle。
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

import numpy as np

from src.real.trajectory_types import ReplayTrajectory

if TYPE_CHECKING:
    from src.ilqt.robot_env_protocol import RobotEnv
    from src.ilqt.step_context import StepContext

logger = logging.getLogger(__name__)


def _json_default(obj: object) -> object:
    """JSON 序列化辅助：处理 numpy 类型。

    Args:
        obj: 待序列化对象。

    Returns:
        numpy 数组 → list，numpy 标量 → Python 标量。

    Raises:
        TypeError: 无法序列化的类型。
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"无法序列化类型 {type(obj)}")


class TrajectoryRecorder:
    """轨迹记录器，可作为 post_exec_hook 注入 EpisodeRunner（仿真侧），
    也可在真机重演循环中手动调用（真机侧记录实际轨迹）。"""

    def __init__(
        self,
        env: "RobotEnv",
        init_q: np.ndarray,
        init_q_left: np.ndarray,
        dt: float,
        metadata: dict | None = None,
        is_position_mode: bool | None = None,
    ) -> None:
        """初始化记录器。

        Args:
            env: RobotEnv Protocol 实现（PlanningEnv 或 RM65Env），需提供
                 get_arm_state() -> (12,) 和 get_ee_pos() -> (3,) 方法。
            init_q: 右臂初始关节角度 (6,) 弧度。
            init_q_left: 左臂初始关节角度 (6,) 弧度。
            dt: 仿真步长（秒）。
            metadata: 额外元信息字典。
            is_position_mode: 控制模式。None 时按 env.actuator_mode 推断
                （1=位置）；无法推断时保守取 False（力矩）——未知模式下不把
                控制量当作位置指令记录，避免 q_desired/u 语义混淆。
        """
        self._env = env
        self._init_q = np.asarray(init_q, dtype=float).copy()
        self._init_q_left = np.asarray(init_q_left, dtype=float).copy()
        self._dt = float(dt)
        self._metadata: dict = dict(metadata) if metadata else {}
        self._is_position_mode = self._resolve_mode(is_position_mode)

        self._q_desired_list: list[np.ndarray] = []
        self._u_list: list[np.ndarray] = []
        self._q_actual_list: list[np.ndarray] = []
        self._timestamp_list: list[float] = []
        self._tcp_pos_list: list[np.ndarray] = []
        self._ball_pos_list: list[np.ndarray] = []

        self._t0: float = time.perf_counter()

    def _resolve_mode(self, explicit: bool | None) -> bool:
        """确定控制模式：显式参数 → env.actuator_mode → metadata → 保守 False。

        Args:
            explicit: 构造时显式传入的模式（None=自动推断）。

        Returns:
            True=位置模式，False=力矩模式。
        """
        if explicit is not None:
            return bool(explicit)
        actuator_mode = getattr(self._env, "actuator_mode", None)
        if actuator_mode is not None:
            return int(actuator_mode) == 1
        if "is_position_mode" in self._metadata:
            return bool(self._metadata["is_position_mode"])
        logger.warning(
            "无法推断控制模式（env 无 actuator_mode），保守按力矩模式记录："
            "q_desired 记为空，控制量写入 u"
        )
        return False

    def record(
        self,
        q_desired: np.ndarray | None,
        q_actual: np.ndarray,
        timestamp: float,
        tcp_pos: np.ndarray,
        ball_pos: np.ndarray | None = None,
        u: np.ndarray | None = None,
    ) -> None:
        """记录一步数据。在 post_exec_hook 或重演循环中调用。

        Args:
            q_desired: 目标关节角度 (6,)。力矩模式传 None（无位置指令）。
            q_actual: 实际关节角度 (6,)。
            timestamp: 时间戳（秒）。
            tcp_pos: 末端位置 (3,)。
            ball_pos: 球位置 (3,)，None 时记录为零向量。
            u: 安全滤波后的控制指令 (6,)（力矩模式=力矩；位置模式可省略）。
        """
        if q_desired is None:
            self._q_desired_list.append(np.zeros(6))
        else:
            self._q_desired_list.append(np.asarray(q_desired, dtype=float).copy())
        if u is None:
            self._u_list.append(np.zeros(6))
        else:
            self._u_list.append(np.asarray(u, dtype=float).copy())
        self._q_actual_list.append(np.asarray(q_actual, dtype=float).copy())
        self._timestamp_list.append(float(timestamp))
        self._tcp_pos_list.append(np.asarray(tcp_pos, dtype=float).copy())
        if ball_pos is not None:
            self._ball_pos_list.append(np.asarray(ball_pos, dtype=float).copy())
        else:
            self._ball_pos_list.append(np.zeros(3))

    def to_trajectory(self, hit_step: int = -1) -> ReplayTrajectory:
        """将内部累积列表转为 ReplayTrajectory。

        Args:
            hit_step: 击球步索引，默认 -1（未击中）。

        Returns:
            包含所有已记录数据的 ReplayTrajectory。
        """
        n = len(self._q_desired_list)
        q_actual = np.stack(self._q_actual_list) if n > 0 else np.zeros((0, 6))
        u = np.stack(self._u_list) if n > 0 else np.zeros((0, 6))
        timestamps = np.array(self._timestamp_list, dtype=float) if n > 0 else np.zeros(0)
        tcp_pos = np.stack(self._tcp_pos_list) if n > 0 else np.zeros((0, 3))
        ball_pos = np.stack(self._ball_pos_list) if n > 0 else np.zeros((0, 3))
        # q_desired 仅在位置模式下有意义：力矩模式的 u_cmd 是力矩，不得冒充位置指令
        if self._is_position_mode:
            q_desired = np.stack(self._q_desired_list) if n > 0 else np.zeros((0, 6))
        else:
            q_desired = np.zeros((0, 6))
        metadata = dict(self._metadata)
        metadata["is_position_mode"] = self._is_position_mode
        return ReplayTrajectory(
            q_desired=q_desired,
            q_actual=q_actual,
            timestamps=timestamps,
            tcp_pos=tcp_pos,
            ball_pos=ball_pos,
            init_q=self._init_q.copy(),
            init_q_left=self._init_q_left.copy(),
            dt=self._dt,
            hit_step=hit_step,
            metadata=metadata,
            u=u,
        )

    def make_hook(self) -> Callable[["StepContext"], None]:
        """生成 post_exec_hook 函数，注入 EpisodeRunner（仿真侧用）。

        hook 内部逻辑:
            1. 读 ctx.u_cmd（安全滤波后的控制指令；位置模式=弧度目标角，力矩模式=力矩）
            2. 位置模式: q_desired = u_cmd；力矩模式: q_desired 不记录（None）
            3. 读 env.get_arm_state()[:6]（执行后的 q_actual）
            4. 读 env.get_ee_pos()（TCP 位置）
            5. 读 ctx.ball_pos（球位置，可能为 None）
            6. 时间戳优先用仿真时间 step_count * dt，无 step_count 时回退墙钟
            7. 调用 self.record(...)

        Returns:
            post_exec_hook 回调函数。
        """

        def hook(ctx: "StepContext") -> None:
            """post_exec_hook: 记录当前步轨迹数据。"""
            u_cmd = ctx.u_cmd if ctx.u_cmd is not None else np.zeros(6)
            # 位置模式下 u_cmd 即弧度目标角；力矩模式下它是力矩，不得写入 q_desired
            q_desired = u_cmd if self._is_position_mode else None
            arm_state = self._env.get_arm_state()
            q_actual = arm_state[:6]
            tcp_pos = self._env.get_ee_pos()
            ball_pos = ctx.ball_pos
            # 优先使用仿真时间（step_count * dt），墙钟时间作为 fallback。
            # 非实时仿真（如 --no-plot 离线）墙钟 ≠ 仿真时间，会导致时间戳与
            # q_actual 不对应，破坏轨迹对比分析。
            if hasattr(ctx, "step_count") and ctx.step_count is not None:
                timestamp = ctx.step_count * self._dt
            else:
                timestamp = time.perf_counter() - self._t0
            self.record(q_desired, q_actual, timestamp, tcp_pos, ball_pos, u=u_cmd)

        return hook

    def save(self, path: Path, hit_step: int = -1) -> None:
        """保存为 .npz 格式。

        用 np.savez 保存所有字段。metadata 用 json.dumps 序列化为字符串。
        自动创建父目录。

        Args:
            path: 保存路径（.npz）。
            hit_step: 击球步索引，默认 -1。
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        traj = self.to_trajectory(hit_step=hit_step)
        metadata_json = json.dumps(traj.metadata, default=_json_default, ensure_ascii=False)
        np.savez(
            path,
            q_desired=traj.q_desired,
            u=traj.u if traj.u is not None else np.zeros((0, 6)),
            q_actual=traj.q_actual,
            timestamps=traj.timestamps,
            tcp_pos=traj.tcp_pos,
            ball_pos=traj.ball_pos,
            init_q=traj.init_q,
            init_q_left=traj.init_q_left,
            dt=traj.dt,
            hit_step=traj.hit_step,
            metadata=metadata_json,
        )
        logger.debug(f"轨迹已保存至 {path}（{len(traj.q_actual)} 步）")

    @staticmethod
    def load(path: Path) -> ReplayTrajectory:
        """从文件加载轨迹。

        支持两种格式:
            1. 新 .npz 格式: 直接加载所有字段。
            2. 旧 pickle 格式（兼容现有 --dump-trajectory）:
               - 含 X_history, U_history, ball_pos_history, init_q, init_q_left,
                 hit_step, p0, v0 等。
               - q_desired = np.array(U_history)
               - q_actual = X_history[1:][:, :6]（跳过初始状态）
               - timestamps = arange * dt（dt 默认 0.005）
               - tcp_pos = zeros（旧格式没有）
               - ball_pos = ball_pos_history if available else zeros
               - dt = 0.005（默认，旧格式未保存）
               - metadata 从 hit_type/pos_error/p0/v0 等构建

        通过尝试 np.load 判断格式，失败则回退到 pickle。

        Args:
            path: 文件路径（.npz 或 .pkl）。

        Returns:
            加载后的 ReplayTrajectory。

        Raises:
            FileNotFoundError: 文件不存在。
            ValueError: 格式无法识别、pickle 非法或缺少必需键。
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {path}")

        # 先尝试新 .npz 格式
        try:
            data = np.load(path, allow_pickle=False)
            if "q_desired" in data.files:
                return TrajectoryRecorder._from_npz(data)
        except (OSError, ValueError):
            pass

        # 回退到旧 pickle 格式
        import pickle

        try:
            with open(path, "rb") as f:
                old = pickle.load(f)
        except (pickle.UnpicklingError, EOFError, OSError) as e:
            raise ValueError(
                f"无法解析的轨迹文件（非 .npz 也非合法 pickle）: {path}: {e}"
            ) from e
        if not isinstance(old, dict):
            raise ValueError(f"无法识别的轨迹文件格式（pickle 内容非 dict）: {path}")
        return TrajectoryRecorder._from_old_pickle(old)

    @staticmethod
    def _from_npz(data: np.lib.npyio.NpzFile) -> ReplayTrajectory:
        """从 NpzFile 构建 ReplayTrajectory。

        Args:
            data: np.load 返回的 NpzFile。

        Returns:
            ReplayTrajectory 实例。
        """
        metadata_str = str(data["metadata"].item())
        metadata: dict = json.loads(metadata_str) if metadata_str else {}
        # u 为新字段：旧 npz 无此项时置 None（向后兼容）
        u = np.asarray(data["u"], dtype=float) if "u" in data.files else None
        return ReplayTrajectory(
            q_desired=np.asarray(data["q_desired"], dtype=float),
            q_actual=np.asarray(data["q_actual"], dtype=float),
            timestamps=np.asarray(data["timestamps"], dtype=float),
            tcp_pos=np.asarray(data["tcp_pos"], dtype=float),
            ball_pos=np.asarray(data["ball_pos"], dtype=float),
            init_q=np.asarray(data["init_q"], dtype=float),
            init_q_left=np.asarray(data["init_q_left"], dtype=float),
            dt=float(data["dt"]),
            hit_step=int(data["hit_step"]),
            metadata=metadata,
            u=u,
        )

    @staticmethod
    def _from_old_pickle(old: dict) -> ReplayTrajectory:
        """从旧 pickle 字典构建 ReplayTrajectory。

        Args:
            old: 旧格式字典，含 X_history/U_history 等键。

        Returns:
            转换后的 ReplayTrajectory。

        Raises:
            ValueError: 缺少必需键。
        """
        # 必需键校验（旧格式必须含这四个键才能正确转换）
        required_keys = ("X_history", "U_history", "init_q", "init_q_left")
        missing = [k for k in required_keys if k not in old]
        if missing:
            raise ValueError(f"旧 pickle 缺少必需键: {missing}")

        q_desired = np.asarray(old["U_history"], dtype=float)
        X_arr = np.asarray(old["X_history"], dtype=float)
        # 跳过初始状态，取右臂 q（前 6 维）
        q_actual = X_arr[1:, :6] if X_arr.shape[0] > 1 else np.zeros((0, 6))
        n = len(q_desired)
        dt = 0.005
        timestamps = np.arange(n, dtype=float) * dt
        tcp_pos = np.zeros((n, 3))
        # U_history 是控制指令（力矩模式下为力矩）：写入 u，q_desired 记为空
        u = q_desired.copy()
        q_desired = np.zeros((0, 6))

        ball_pos_history = old.get("ball_pos_history")
        if ball_pos_history is not None and len(ball_pos_history) > 0:
            ball_pos = np.asarray(ball_pos_history, dtype=float)
        else:
            ball_pos = np.zeros((n, 3))

        # 长度一致性校验：控制指令长度 = len(U_history)，
        # q_actual 长度 = len(X_history) - 1。若不一致则截断到较短长度。
        if len(q_actual) != n:
            min_len = min(len(q_actual), n)
            logger.warning(
                f"旧 pickle X_history/U_history 长度不匹配: "
                f"u={n}, q_actual={len(q_actual)}，截断至 {min_len}"
            )
            u = u[:min_len]
            q_actual = q_actual[:min_len]
            timestamps = timestamps[:min_len]
            tcp_pos = tcp_pos[:min_len]
            ball_pos = ball_pos[:min_len]

        # metadata 从旧字段构建
        metadata: dict = {}
        for key in ("hit_type", "pos_error", "p0", "v0", "post_hit_steps"):
            if key in old:
                metadata[key] = old[key]

        # 旧格式无法确认控制模式，保守标记为力矩模式（不安全）。
        # replay_trajectory.py 默认拒绝 is_position_mode=False 的轨迹。
        # 用户确认旧文件来源后，可用 --force-mode 跳过检查。
        metadata["is_position_mode"] = False
        logger.warning(
            "旧 pickle 格式无法确认控制模式，默认标记为力矩模式。"
            "如确认是位置模式生成，请用 --force-mode 跳过检查。"
        )

        return ReplayTrajectory(
            q_desired=q_desired,
            q_actual=q_actual,
            timestamps=timestamps,
            tcp_pos=tcp_pos,
            ball_pos=ball_pos,
            init_q=np.asarray(old["init_q"], dtype=float),
            init_q_left=np.asarray(old["init_q_left"], dtype=float),
            dt=dt,
            hit_step=int(old.get("hit_step", -1)),
            metadata=metadata,
            u=u,
        )
