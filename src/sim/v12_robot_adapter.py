"""V12 机器人适配层 — RM65 / R1D 统一构造入口（阶段 4 · Task 4）。

职责（T4 brief §4）：
- 定义 :class:`SimulationEnv` 结构协议：``RM65Env`` 与 R1D ``ProfileSimEnv``
  共同满足的执行环境接口（V12 管线只依赖本协议，不感知具体机器人）；
- 定义 :class:`V12RobotBundle`：把 T0 中散落在 ``scripts/rm65_mpc_v12.py``
  的机器人构造事实（执行/规划环境、限位、初始姿态、几何 ID、保护接触、
  X 墙语义、回放状态访问）收敛为单一事实来源；
- :func:`build_v12_robot`：按 :class:`ResolvedV12Config` 的 ``robot_id``
  分派 RM65 / R1D 两条构造路径，供 T5 的 V12 episode 入口唯一消费。

设计约束：
- 本模块顶层不导入任何真机 SDK，不调用脚本 ``main()``，不构造
  ``EpisodeRunner``；
- 两种机器人后续的 PD/前馈/限幅配置统一走
  :func:`_apply_actuator_configuration` 同一分支（brief §4.5 同源要求）；
- RM65 路径保持 T0 行为：``planning`` 与 ``execution`` 为同一对象（T0
  ``MPCController`` 以 ``RM65Env`` 同时承担执行与规划），
  ``planning_env_factory`` 复刻 T0 ``AsyncReplanner`` 的
  ``type(env)(model_path, dt)`` + ``init_q_left`` + clone 执行器配置；
- R1D 路径七关节全部按名绑定（绝不截前六）：执行环境 ``nq = n + 7``
  （自由球），规划环境 ``nq = n``；自由球 qpos/qvel 地址由
  ``ModelBinding`` / ``TennisSceneBundle`` 给出；
- R1D 强制 ``solver_backend == "cpp"`` 时立即 ``ValueError``（不静默降级）：
  七关节环境不是六维前缀布局，C++ 后端只支持 RM65 六维；``auto`` /
  ``python`` 由 ``build_solver`` 落到有限差分（FD），本模块在
  ``bundle.backend`` 记录实际后端事实。

回放状态按关节名读写（``get_replay_state`` / ``set_replay_state``），
不硬编码关节数、不假设“前六个关节就是主动臂”。
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.ilqt.components.predictive_safety import X_WALL_BODY_NAMES
from src.ilqt.profile_planning_env import ProfilePlanningEnv
from src.ilqt.robot_limits import RobotLimits
from src.robot.constants import INIT_Q, INIT_Q_LEFT
from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.model_factory import (
    RacketAssembly,
    RobotModelBundle,
    load_robot_model,
)
from src.robot.racket_profile import RacketProfile, load_racket_profile
from src.robot.robot_profile import (
    RobotProfile,
    default_rm65_profile_path,
    load_robot_profile,
)
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.env_factory import build_r1d_environments
from src.sim.rm65_env import RM65Env
from src.sim.sim_config import SimTaskConfig, load_sim_task_config
from src.sim.tennis_scene import TennisSceneBundle
from src.utils.yaml_utils import load_config

if TYPE_CHECKING:
    from src.ilqt.robot_env_protocol import RobotEnv
    from src.sim.v12_config import ResolvedV12Config

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]

#: 仓库根目录（``src/sim/v12_robot_adapter.py`` 向上三级）
_REPO_ROOT = Path(__file__).resolve().parents[2]
#: T0 仿真默认配置（RM65 仿真限位来源，与 ``scripts/rm65_mpc_v12.py`` 一致）
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "default.yaml"

# T0 的 X 平面墙语义：世界坐标 X 不得越过 -0.1（``src/ilqt/components/
# predictive_safety.py`` 中 ``_X_WALL_LIMIT`` 的同一数值，此处显式记录适配
# 事实，避免 adapter 依赖其私有常量）。
_RM65_X_WALL_LIMIT = -0.1

# RM65 场景几何名（与 ``src/robot/rm65_model.xml`` 一致）
_RM65_BALL_JOINT = "ball_freejoint"
_RM65_BALL_GEOM = "ball"
_RM65_RACKET_FACE_GEOM = "r_racket_face"
_RM65_RACKET_BODY = "r_racket_body"
_RM65_FLANGE_BODY = "r_flange"
_RM65_RACKET_CENTER_SITE = "racket_center"

# R1D 场景几何名（与 ``r1d_model_right.xml`` / T2 ``build_tennis_scene`` 一致）
_R1D_BALL_JOINT = "ball_freejoint"
#: R1D 力矩模式数值 parity 的估计关节阻尼（N·m·s/rad，7 主动关节同序）。
#:
#: 来源与性质（**estimated / provisional，非 URDF 声明值**）：
#: R1D 官方 URDF 导出模型未声明关节 damping / armature（T2 决定不虚构），
#: 而 RM65 模型 ``src/robot/rm65_model.xml`` 六关节声明了
#: damping=[2.0, 2.0, 1.0, 0.5, 0.3, 0.2]、armature=[0.2, 0.2, 0.1, 0.05,
#: 0.03, 0.02]。力矩模式 iLQR 的解析/有限差分校准与 warm-start JT 控制律
#: 依赖关节速度阻尼，缺失时 R1D 七关节模型在首 plan rollout 内发散
#: （见 T7 诊断 plan_debug_j7cap2 / probe_tcp）。为恢复与 RM65 同口径的
#: 数值 parity，这里按 RM65 同序延展末项，作为**仅 R1D V12 路径生效**的
#: 显式仿真参数注入；不改 URDF/XML 资产，不影响 RM65 与旧 R1D stage3 路径。
_R1D_ESTIMATED_DOF_DAMPING: tuple[float, ...] = (
    2.0,
    2.0,
    1.0,
    0.5,
    0.3,
    0.2,
    0.1,
)

#: R1D 力矩模式数值 parity 的估计关节 armature（kg·m²，7 主动关节同序）。
#: 性质与来源同 :data:`_R1D_ESTIMATED_DOF_DAMPING`（estimated / provisional）。
_R1D_ESTIMATED_DOF_ARMATURE: tuple[float, ...] = (
    0.2,
    0.2,
    0.1,
    0.05,
    0.03,
    0.02,
    0.01,
)

_R1D_BALL_GEOM = "ball"
_R1D_RACKET_FACE_GEOM = "racket_face"
_R1D_RACKET_BODY = "r1d_racket"
_R1D_RACKET_CENTER_SITE = "racket_center"
_R1D_TORSO_BODY = "r1d_torso"
_R1D_TORSO_GEOM = "torso_box"

# R1D overlay 任务未提供仿真任务文件时使用的临时来球初态：数值沿用阶段 3
# ``configs/r1d_sim_right.yaml``（provisional），T5 运行期以 serve 配置覆盖。
_R1D_PROVISIONAL_BALL_POSITION: tuple[float, float, float] = (0.3, -0.6, 2.2)
_R1D_PROVISIONAL_BALL_VELOCITY: tuple[float, float, float] = (-11.5, -0.3, 4.25)
_R1D_MAX_STEPS = 900
#: R1D 躯干接触保护对（右臂链路 7 个 visual geom + 拍面 vs 躯干盒）。
_R1D_RIGHT_PROTECTED_GEOM_PAIRS: tuple[tuple[str, str], ...] = (
    ("right_shoulder_pitch_Link_visual", "torso_box"),
    ("right_shoulder_roll_link_visual", "torso_box"),
    ("right_shoulder_yaw_link_visual", "torso_box"),
    ("right_elbow_link_visual", "torso_box"),
    ("right_wrist_roll_link_visual", "torso_box"),
    ("right_wrist_pitch_link_visual", "torso_box"),
    ("right_wrist_yaw_link_visual", "torso_box"),
    ("racket_face", "torso_box"),
)


@runtime_checkable
class SimulationEnv(Protocol):
    """V12 执行环境结构协议（``RM65Env`` / R1D ``ProfileSimEnv`` 共同满足）。

    管线只依赖本协议；实现方必须自洽地提供关节状态、球状态、末端几何与
    执行器配置入口。协议不规定内部数组布局，主动关节名与 qpos/qvel 地址
    由 ``V12RobotBundle`` 的按名绑定承载。
    """

    #: 主动关节自由度
    NQ: int
    #: 模型速度维数（RM65 含左臂与球 = 18；R1D 含球 = 13）
    NV: int
    #: 控制维数
    NU: int
    #: 执行环境状态维数（主动臂 q + qdot）
    NX: int
    #: MuJoCo 模型
    model: mujoco.MjModel
    #: MuJoCo 数据
    data: mujoco.MjData

    @property
    def dt(self) -> float:
        """仿真步长（秒）。"""

    @property
    def control_range(self) -> Array:
        """控制量允许范围，形状 ``(NU, 2)``。"""

    @property
    def torque_max(self) -> Array:
        """执行器力矩上限（绝对值），形状 ``(NU,)``。"""

    def get_arm_state(self) -> Array:
        """读取主动臂状态 ``[q, qdot]``。"""

    def set_arm_state(self, x: Array) -> None:
        """写入主动臂状态 ``[q, qdot]`` 并前向运动学。"""

    def reset(self, q0: Array | None = None) -> Array:
        """复位到初始姿态并返回主动臂状态。"""

    def get_ball_state(self) -> tuple[Array, Array]:
        """读取球位置与速度。"""

    def get_ball_pos(self) -> Array:
        """读取球位置，形状 ``(3,)``。"""

    def get_ball_vel(self) -> Array:
        """读取球速度，形状 ``(3,)``。"""

    def set_ball_state(self, pos: Array, vel: Array) -> None:
        """写入球位置与速度。"""

    def set_ball_vel(self, vel: Array) -> None:
        """只写入球速度（支持 3 维线速度或 6 维线速度+角速度）。"""

    def step_full(self, u: Array) -> tuple[Array, Array, Array]:
        """前进一步并返回 ``(主动臂状态, 球位置, 球速度)``。"""

    def get_ee_pos(self) -> Array:
        """读取球拍中心世界坐标，形状 ``(3,)``。"""

    def get_ee_vel(self) -> Array:
        """读取球拍中心世界线速度，形状 ``(3,)``。"""

    def get_ee_normal(self) -> Array:
        """读取拍面法向单位向量，形状 ``(3,)``。"""

    def get_racket_face_speed(self) -> float:
        """读取拍面中心速度标量（m/s）。"""

    def get_ee_angular_vel(self) -> Array:
        """读取末端角速度，形状 ``(3,)``。"""

    def step(self, u: Array) -> Array:
        """施加控制前进一步并返回主动臂状态。"""

    def set_arm_collision(self, enabled: bool) -> None:
        """开/关臂与球的碰撞通道（不改变非球几何掩码）。"""

    def update_kinematics(self) -> None:
        """刷新运动学量。"""

    def configure_actuator_mode(
        self,
        mode: str,
        kp: Array | None = None,
        kd: Array | None = None,
    ) -> None:
        """切换力矩/位置执行器模式。"""

    def configure_feedforward(self, enabled: bool) -> None:
        """开/关位置模式重力前馈。"""


class RM65SimulationEnv(RM65Env):
    """RM65 执行环境适配器 — 补齐 ``SimulationEnv`` 协议缺口。

    ``RM65Env`` 缺少 ``NV`` / ``torque_max`` 两个协议事实，且其
    ``step_from_state`` 直接改写 live ``MjData``（会污染主线程执行状态）。
    本适配器：
    - ``NV`` 取 ``model.nv``（RM65 为 18：双臂 12 + 球 6）；
    - ``torque_max`` 在构造时从力矩模式 ``ctrlrange`` 快照（切换位置模式后
      ``ctrlrange`` 会变成关节范围，不能晚读）；
    - ``step_from_state`` 保存/恢复 ``qpos`` / ``qvel`` / ``ctrl`` /
      ``qfrc_applied`` / ``time``，保证 trial step 不污染 live 状态。
    """

    def __init__(self, model_path: Path, *, dt: float) -> None:
        """构造 RM65 适配环境。

        Args:
            model_path: RM65 MuJoCo 模型路径。
            dt: 仿真步长（秒）。
        """
        super().__init__(model_path, dt=dt)
        ctrlrange = np.asarray(
            getattr(
                self,
                "_torque_ctrlrange",
                self.model.actuator_ctrlrange[: self.NU],
            ),
            dtype=np.float64,
        )
        self._torque_max_snapshot = np.maximum(
            np.abs(ctrlrange[:, 0]),
            np.abs(ctrlrange[:, 1]),
        )

    @property
    def NV(self) -> int:  # noqa: N802 - 协议属性名固定
        """模型速度维数（RM65 含左臂与球 = 18）。"""
        return int(self.model.nv)

    @property
    def torque_max(self) -> Array:
        """力矩上限快照，形状 ``(NU,)``。"""
        return self._torque_max_snapshot.copy()

    def step_from_state(
        self,
        x: Array,
        u: Array,
        preserve_cache: bool = False,
    ) -> Array:
        """从给定状态前进一步且不污染 live 数据（T4 §4.5 约束）。

        Args:
            x: 主动臂状态 ``[q, qdot]``，形状 ``(NX,)``。
            u: 控制量，形状 ``(NU,)``。
            preserve_cache: 是否沿用调用前的球状态缓存（trial step 语义）。

        Returns:
            前进一步后的主动臂状态，形状 ``(NX,)``。
        """
        data = self.data
        qpos = data.qpos.copy()
        qvel = data.qvel.copy()
        ctrl = data.ctrl.copy()
        qfrc_applied = data.qfrc_applied.copy()
        sim_time = float(data.time)
        try:
            return super().step_from_state(x, u, preserve_cache=preserve_cache)
        finally:
            data.qpos[:] = qpos
            data.qvel[:] = qvel
            data.ctrl[:] = ctrl
            data.qfrc_applied[:] = qfrc_applied
            data.time = sim_time
            mujoco.mj_forward(self.model, data)
            self._invalidate_jacobian_cache()


@dataclass(frozen=True)
class XWallSpec:
    """X 墙语义描述 — 把 T0 的“世界 X 平面墙”抽象为可映射规格。

    Attributes:
        mode: ``"world_x"`` 表示世界坐标平面墙；``"body_plane"`` 表示以
            ``body_names`` 指向的机体系对称面为参照的平面。
        limit: 允许越界的坐标上限（世界系或机体局部系，按 ``mode`` 解释）。
        body_names: 用于 X 墙检查的几何体名称（RM65 为右臂链路）；
            R1D 为躯干体。
        protected_geom_names: 需要显式保护（禁止与球/自身接触）的几何名。
        enabled: 该墙是否在运行期启用。
        coordinate_frame: 坐标参照说明（``"world"`` / ``"torso_local"``）。
    """

    mode: Literal["world_x", "body_plane"]
    limit: float
    body_names: tuple[str, ...]
    protected_geom_names: tuple[str, ...]
    enabled: bool
    coordinate_frame: str


@dataclass(frozen=True)
class ReplayState:
    """按关节名的回放状态 — 不硬编码关节数，不假设前六维。

    Attributes:
        joint_positions: 关节名 → 关节角（弧度）。
        joint_velocities: 关节名 → 关节角速度（rad/s）。
        ball_position: 球世界位置，形状 ``(3,)``。
        ball_velocity: 球世界线速度，形状 ``(3,)``。
    """

    joint_positions: Mapping[str, float]
    joint_velocities: Mapping[str, float]
    ball_position: Array
    ball_velocity: Array


@dataclass(frozen=True)
class RobotGeometry:
    """机器人几何事实 — 名称与 MuJoCo 编译后 ID 一并给出。

    Attributes:
        racket_center_site: 球拍中心 site 名。
        racket_center_site_id: 球拍中心 site id。
        racket_face_geom: 拍面 geom 名。
        racket_face_geom_id: 拍面 geom id。
        ball_geom: 球 geom 名。
        ball_geom_id: 球 geom id。
        racket_body: 拍体 body 名。
        racket_body_id: 拍体 body id。
        flange_body: 法兰 body 名（拍体的安装父体）。
        flange_body_id: 法兰 body id。
        ball_joint: 球自由关节名。
        ball_joint_id: 球自由关节 id。
        racket_geom_names: 拍体下的全部 geom 名（手柄 + 拍面等）。
    """

    racket_center_site: str
    racket_center_site_id: int
    racket_face_geom: str
    racket_face_geom_id: int
    ball_geom: str
    ball_geom_id: int
    racket_body: str
    racket_body_id: int
    flange_body: str
    flange_body_id: int
    ball_joint: str
    ball_joint_id: int
    racket_geom_names: tuple[str, ...]


@dataclass
class V12RobotBundle:
    """V12 机器人事实集合 — 执行/规划环境、限位、几何与安全语义。

    该对象是 T5 ``run_v12_episode`` 构造仿真闭环的唯一机器人来源；它只提供
    事实与轻量访问器，不构造 ``EpisodeRunner``、不调用脚本 ``main()``。

    Attributes:
        robot_id: 机器人标识（``"rm65"`` / ``"unitree_r1d"``）。
        joint_names: 主动关节名（RM65 6 个，R1D 7 个，顺序为绑定顺序）。
        execution: 执行环境（``SimulationEnv``）。
        planning: 规划环境（``RobotEnv``；RM65 与 execution 为同一对象）。
        robot_limits: 关节/速度/力矩限位。
        init_q: 主动关节初始角，形状 ``(NQ,)``。
        geometry: 几何事实。
        protected_contacts: 禁止接触的几何对（名称对，空元组表示无显式对）。
        x_wall: X 墙语义映射。
        planning_env_factory: 构造独立规划环境实例的工厂（异步重规划用）；
            每次调用必须返回全新的 model/data，不得共享可变状态。
        backend: 实际求解后端事实（``"cpp"`` / ``"python"``）。
        racket: R1D 球拍装配参数化事实（RM65 为 None，走 legacy 模型内嵌球拍）。
        robot_model_bundle: R1D 模型/绑定/球拍装配原始事实（RM65 为 None）。
    """

    robot_id: str
    joint_names: tuple[str, ...]
    execution: SimulationEnv
    planning: RobotEnv
    robot_limits: RobotLimits
    init_q: Array
    geometry: RobotGeometry
    protected_contacts: tuple[tuple[str, str], ...]
    x_wall: XWallSpec
    planning_env_factory: Callable[[], RobotEnv]
    backend: str = "auto"
    racket: RacketAssembly | None = None
    robot_model_bundle: RobotModelBundle | None = None
    _qpos_indices: tuple[int, ...] = field(default=(), repr=False)
    _qvel_indices: tuple[int, ...] = field(default=(), repr=False)
    _ball_qpos_start: int = field(default=-1, repr=False)
    _ball_qvel_start: int = field(default=-1, repr=False)

    def reset(self) -> Array:
        """复位执行环境与规划环境，返回执行环境主动臂状态。

        Returns:
            主动臂状态 ``[q, qdot]``，形状 ``(NX,)``。
        """
        arm_state = np.asarray(
            self.execution.reset(self.init_q),
            dtype=np.float64,
        ).copy()
        if self.planning is not self.execution:
            self.planning.reset(self.init_q)
            ball_pos = self.execution.get_ball_pos()
            ball_vel = self.execution.get_ball_vel()
            self.planning.set_ball_state(ball_pos.copy(), ball_vel.copy())
        return arm_state

    def joint_qpos_index(self, joint_name: str) -> int:
        """按关节名返回 qpos 地址。

        Args:
            joint_name: 主动关节名。

        Returns:
            qpos 中的地址。

        Raises:
            KeyError: 关节名不在本 bundle 的主动关节集合内。
        """
        try:
            offset = self.joint_names.index(joint_name)
        except ValueError as exc:
            raise KeyError(
                f"关节 {joint_name!r} 不在 {self.robot_id} 主动关节集合 "
                f"{self.joint_names!r} 中"
            ) from exc
        return self._qpos_indices[offset]

    def get_replay_state(self) -> ReplayState:
        """按关节名读取回放状态（不硬编码 6 维）。

        Returns:
            当前执行环境关节位置/速度（按名）与球位置/线速度。

        Raises:
            RuntimeError: bundle 未提供按名绑定信息。
        """
        if not self._qpos_indices or self._ball_qpos_start < 0:
            raise RuntimeError(
                f"{self.robot_id} bundle 缺少按名绑定/球地址，无法读取回放状态"
            )
        data = self.execution.data
        qpos = np.asarray(data.qpos)
        qvel = np.asarray(data.qvel)
        positions = {
            name: float(qpos[index])
            for name, index in zip(self.joint_names, self._qpos_indices)
        }
        velocities = {
            name: float(qvel[index])
            for name, index in zip(self.joint_names, self._qvel_indices)
        }
        ball_position = np.array(
            qpos[self._ball_qpos_start : self._ball_qpos_start + 3],
            dtype=np.float64,
            copy=True,
        )
        ball_velocity = np.array(
            qvel[self._ball_qvel_start : self._ball_qvel_start + 3],
            dtype=np.float64,
            copy=True,
        )
        return ReplayState(
            joint_positions=positions,
            joint_velocities=velocities,
            ball_position=ball_position,
            ball_velocity=ball_velocity,
        )

    def set_replay_state(self, state: ReplayState) -> None:
        """按关节名写回放状态（不硬编码 6 维）。

        Args:
            state: 关节名 → 位置/速度映射，以及球位置/线速度。

        Raises:
            ValueError: 缺失某个主动关节名。
        """
        missing = [
            name for name in self.joint_names if name not in state.joint_positions
        ]
        if missing:
            raise ValueError(
                f"回放状态缺少关节位置：{missing!r}（需要 {self.joint_names!r}）"
            )
        missing_vel = [
            name for name in self.joint_names if name not in state.joint_velocities
        ]
        if missing_vel:
            raise ValueError(
                f"回放状态缺少关节速度：{missing_vel!r}（需要 {self.joint_names!r}）"
            )
        q = np.array(
            [state.joint_positions[name] for name in self.joint_names],
            dtype=np.float64,
        )
        qdot = np.array(
            [state.joint_velocities[name] for name in self.joint_names],
            dtype=np.float64,
        )
        self.execution.set_arm_state(np.concatenate([q, qdot]))
        self.execution.set_ball_state(
            np.asarray(state.ball_position, dtype=np.float64).copy(),
            np.asarray(state.ball_velocity, dtype=np.float64).copy(),
        )


def _mj_id(model: mujoco.MjModel, objtype: int, name: str, context: str) -> int:
    """按名称查询 MuJoCo 对象 id，缺失时给出明确错误。

    Args:
        model: MuJoCo 模型。
        objtype: ``mujoco.mjtObj`` 类型常量。
        name: 对象名。
        context: 错误信息中的上下文（中文说明）。

    Returns:
        对象 id。

    Raises:
        ValueError: 名称在模型中不存在。
    """
    obj_id = mujoco.mj_name2id(model, objtype, name)
    if obj_id < 0:
        raise ValueError(f"{context} {name!r} 不存在于模型中")
    return int(obj_id)


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    """按 body id 反查名称。

    Args:
        model: MuJoCo 模型。
        body_id: body id。

    Returns:
        body 名称。

    Raises:
        ValueError: body id 无名称（匿名 body 不可用于按名保护配置）。
    """
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id))
    if name is None:
        raise ValueError(f"body id {body_id} 没有名称，无法参与按名保护配置")
    return str(name)


def _body_geom_names(model: mujoco.MjModel, body_id: int) -> tuple[str, ...]:
    """返回 body 下全部 geom 名。

    Args:
        model: MuJoCo 模型。
        body_id: body id。

    Returns:
        geom 名称元组。
    """
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    names: list[str] = []
    for geom_id in range(start, start + count):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name is None:
            raise ValueError(
                f"body {_body_name(model, body_id)!r} 下 geom id {geom_id} 无名称"
            )
        names.append(str(name))
    return tuple(names)


def _resolve_geometry(
    model: mujoco.MjModel,
    *,
    racket_center_site: str,
    racket_face_geom: str,
    racket_body: str,
    ball_geom: str,
    ball_joint: str,
    flange_body: str | None = None,
) -> RobotGeometry:
    """解析机器人几何事实（名称 + 编译后 id）。

    Args:
        model: MuJoCo 模型。
        racket_center_site: 球拍中心 site 名。
        racket_face_geom: 拍面 geom 名。
        racket_body: 拍体 body 名。
        ball_geom: 球 geom 名。
        ball_joint: 球自由关节名。
        flange_body: 法兰 body 名；None 时取拍体父 body。

    Returns:
        :class:`RobotGeometry`。
    """
    racket_body_id = _mj_id(model, mujoco.mjtObj.mjOBJ_BODY, racket_body, "拍体")
    if flange_body is None:
        flange_body = _body_name(model, int(model.body_parentid[racket_body_id]))
    return RobotGeometry(
        racket_center_site=racket_center_site,
        racket_center_site_id=_mj_id(
            model, mujoco.mjtObj.mjOBJ_SITE, racket_center_site, "球拍中心 site"
        ),
        racket_face_geom=racket_face_geom,
        racket_face_geom_id=_mj_id(
            model, mujoco.mjtObj.mjOBJ_GEOM, racket_face_geom, "拍面 geom"
        ),
        ball_geom=ball_geom,
        ball_geom_id=_mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, ball_geom, "球 geom"),
        racket_body=racket_body,
        racket_body_id=racket_body_id,
        flange_body=flange_body,
        flange_body_id=_mj_id(
            model, mujoco.mjtObj.mjOBJ_BODY, flange_body, "法兰 body"
        ),
        ball_joint=ball_joint,
        ball_joint_id=_mj_id(
            model, mujoco.mjtObj.mjOBJ_JOINT, ball_joint, "球自由关节"
        ),
        racket_geom_names=_body_geom_names(model, racket_body_id),
    )


def _broadcast_gain(value: object, count: int, name: str) -> Array:
    """把标量/长度 1/长度 n 的增益广播为长度 n 数组。

    Args:
        value: 增益值（标量、长度 1 或长度 ``count`` 的序列）。
        count: 主动关节数。
        name: 用于错误信息的字段名。

    Returns:
        形状 ``(count,)`` 的 float64 数组。

    Raises:
        ValueError: 长度既不是 1 也不是 ``count``，或含非有限值。
    """
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size == 1:
        array = np.full(count, float(array[0]), dtype=np.float64)
    elif array.size != count:
        raise ValueError(
            f"{name} 长度应为 1 或 {count}，实际为 {array.size}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含非有限值：{array.tolist()}")
    return array


def _apply_actuator_configuration(
    env: SimulationEnv,
    config: ResolvedV12Config,
    *,
    runtime_kp: Array | None = None,
    runtime_kd: Array | None = None,
) -> None:
    """按 resolved 配置统一配置执行器模式（两种机器人同源分支）。

    RM65 与 R1D 的力矩/位置模式、增益广播、前馈开关都走本函数，避免两条
    构造路径出现行为分叉（brief §4.5）。

    Args:
        env: 目标执行/规划环境。
        config: resolved V12 配置。
        runtime_kp: position 模式下 config.kp 为 None 时的运行参数回退。
        runtime_kd: position 模式下 config.kd 为 None 时的运行参数回退。

    Raises:
        ValueError: actuator_mode 非法，或 position 模式缺少 kp/kd。
    """
    mode = str(config.actuator_mode)
    if mode == "torque":
        env.configure_actuator_mode("torque")
        return
    if mode != "position":
        raise ValueError(
            f"未知 actuator_mode={mode!r}，只支持 'torque' / 'position'"
        )
    kp_source = config.kp if config.kp is not None else runtime_kp
    kd_source = config.kd if config.kd is not None else runtime_kd
    if kp_source is None or kd_source is None:
        raise ValueError(
            "position 模式必须提供 kp/kd：resolved 配置缺失时需由 runtime "
            "参数回退（RM65 走 CLI/默认配置，R1D 走 runtime kp/kd）"
        )
    kp = _broadcast_gain(kp_source, int(env.NU), "kp")
    kd = _broadcast_gain(kd_source, int(env.NU), "kd")
    env.configure_actuator_mode("position", kp, kd)
    feedforward = True if config.feedforward is None else bool(config.feedforward)
    env.configure_feedforward(feedforward)


def _apply_r1d_estimated_joint_dynamics(env: SimulationEnv) -> None:
    """给 R1D 七主动关节注入估计关节阻尼/armature（仅 R1D V12 路径）。

    只改内存中的 ``model.dof_damping`` / ``model.dof_armature``，不写回
    XML/URDF 资产。数值为 estimated / provisional（URDF 未声明，参考 RM65
    模型同口径延展，见 :data:`_R1D_ESTIMATED_DOF_DAMPING`）。

    Args:
        env: R1D 执行环境或规划环境（须带 ``binding`` 与 ``model``）。

    Raises:
        ValueError: 环境缺少 ``binding``，或绑定关节数与估计值长度不一致。
    """
    binding = getattr(env, "binding", None)
    if binding is None:
        raise ValueError(
            "R1D 环境缺少 binding，无法注入关节阻尼/armature（数值 parity）"
        )
    joint_ids = [int(joint_id) for joint_id in binding.joint_ids]
    if len(joint_ids) != len(_R1D_ESTIMATED_DOF_DAMPING):
        raise ValueError(
            "R1D 关节数不匹配：估计 damping/armature 针对 "
            f"{len(_R1D_ESTIMATED_DOF_DAMPING)} 个主动关节，"
            f"实际绑定 {len(joint_ids)} 个"
        )
    for index, joint_id in enumerate(joint_ids):
        dof = int(env.model.jnt_dofadr[joint_id])
        env.model.dof_damping[dof] = _R1D_ESTIMATED_DOF_DAMPING[index]
        env.model.dof_armature[dof] = _R1D_ESTIMATED_DOF_ARMATURE[index]


def _build_rm65_robot_limits(
    execution: RM65SimulationEnv,
    *,
    dt: float,
) -> RobotLimits:
    """按 T0 仿真路径构造 RM65 限位。

    T0 ``scripts/rm65_mpc_v12.py`` 使用 ``RobotLimits.from_config(
    config_dict["robot_limits"], dt=dt, ctrlrange=model.actuator_ctrlrange[:NU])
    ``；本函数逐字复刻该语义。

    Args:
        execution: RM65 执行环境。
        dt: 仿真步长（秒）。

    Returns:
        RM65 仿真限位。
    """
    config_dict = load_config(_DEFAULT_CONFIG_PATH)
    robot_limits_config = dict(config_dict.get("robot_limits", {}))
    ctrlrange = np.asarray(
        execution.model.actuator_ctrlrange[: execution.NU],
        dtype=np.float64,
    ).copy()
    return RobotLimits.from_config(
        robot_limits_config,
        dt=dt,
        ctrlrange=ctrlrange,
    )


def _build_rm65_v12_robot(config: ResolvedV12Config) -> V12RobotBundle:
    """构造 RM65 的 V12 机器人事实（保持 T0 执行/规划同一对象语义）。

    Args:
        config: resolved V12 配置（``robot_id == "rm65"``）。

    Returns:
        :class:`V12RobotBundle`。
    """
    profile_path = (
        Path(config.robot_profile_path)
        if config.robot_profile_path is not None
        else default_rm65_profile_path()
    )
    profile = load_robot_profile(profile_path)
    dt = float(config.dt)
    execution = RM65SimulationEnv(profile.model_path, dt=dt)
    # T0：env.init_q_left = INIT_Q_LEFT（左臂保持零位/初始位）
    execution.init_q_left = np.array(INIT_Q_LEFT, dtype=np.float64)
    _apply_actuator_configuration(execution, config)
    binding: ModelBinding = bind_robot_model(profile, execution.model)
    robot_limits = _build_rm65_robot_limits(execution, dt=dt)
    model_path = Path(profile.model_path)

    def _planning_env_factory() -> RobotEnv:
        """构造独立 RM65 规划环境（复刻 T0 AsyncReplanner 创建逻辑）。"""
        plan_env = RM65Env(model_path, dt=dt)
        plan_env.init_q_left = np.array(
            execution.init_q_left, dtype=np.float64, copy=True
        )
        execution.clone_actuator_config(plan_env)
        mujoco.mj_forward(plan_env.model, plan_env.data)
        logger.info(
            "RM65 planning_env_factory 创建独立规划环境：nq=%d nv=%d nu=%d",
            plan_env.NQ,
            plan_env.model.nv,
            plan_env.NU,
        )
        return plan_env

    geometry = _resolve_geometry(
        execution.model,
        racket_center_site=_RM65_RACKET_CENTER_SITE,
        racket_face_geom=_RM65_RACKET_FACE_GEOM,
        racket_body=_RM65_RACKET_BODY,
        ball_geom=_RM65_BALL_GEOM,
        ball_joint=_RM65_BALL_JOINT,
        flange_body=_RM65_FLANGE_BODY,
    )
    logger.info(
        "构造 RM65 V12 bundle：joints=%d nq=%d nv=%d nu=%d backend=%s",
        len(profile.active_joint_names),
        execution.NQ,
        execution.NV,
        execution.NU,
        config.solver_backend,
    )
    return V12RobotBundle(
        robot_id=profile.robot_id,
        joint_names=tuple(profile.active_joint_names),
        execution=execution,
        planning=execution,
        robot_limits=robot_limits,
        init_q=np.array(INIT_Q, dtype=np.float64),
        geometry=geometry,
        protected_contacts=(),
        x_wall=XWallSpec(
            mode="world_x",
            limit=_RM65_X_WALL_LIMIT,
            body_names=tuple(X_WALL_BODY_NAMES),
            protected_geom_names=(),
            enabled=True,
            coordinate_frame="world",
        ),
        planning_env_factory=_planning_env_factory,
        backend=str(config.solver_backend),
        racket=None,
        robot_model_bundle=None,
        _qpos_indices=tuple(binding.qpos_indices),
        _qvel_indices=tuple(binding.qvel_indices),
        _ball_qpos_start=int(RM65Env.BALL_QPOS_START),
        _ball_qvel_start=int(RM65Env.BALL_QVEL_START),
    )


def _resolve_r1d_sim_task(
    config: ResolvedV12Config,
    profile: RobotProfile,
) -> SimTaskConfig:
    """解析 R1D 仿真任务配置。

    ``r1d_fixed_ball_compat`` 模式直接读取 T3 指定的 legacy 任务文件（阶段 3
    ``SimTaskConfig`` 语义）；``r1d_overlay`` 模式由 resolved 配置合成任务，
    来球初态使用阶段 3 的 provisional 数值（T5 运行期以 serve 覆盖）。

    Args:
        config: resolved V12 配置。
        profile: R1D 机器人描述。

    Returns:
        :class:`SimTaskConfig`。

    Raises:
        ValueError: fixed_ball_compat 模式缺少 task 路径。
    """
    if config.task_mode == "r1d_fixed_ball_compat":
        if config.task_path is None:
            raise ValueError(
                "r1d_fixed_ball_compat 模式需要 --sim-task-config 指定任务文件"
            )
        return load_sim_task_config(Path(config.task_path), profile)
    return SimTaskConfig(
        robot_id=profile.robot_id,
        joint_names=tuple(profile.active_joint_names),
        ball_joint_name=_R1D_BALL_JOINT,
        ball_geom_name=_R1D_BALL_GEOM,
        racket_geom_names=(_R1D_RACKET_FACE_GEOM,),
        protected_geom_pairs=_R1D_RIGHT_PROTECTED_GEOM_PAIRS,
        workspace_center=np.asarray(config.target_center, dtype=np.float64),
        workspace_radius=float(config.workspace_radius),
        initial_ball_position=np.array(
            _R1D_PROVISIONAL_BALL_POSITION, dtype=np.float64
        ),
        initial_ball_velocity=np.array(
            _R1D_PROVISIONAL_BALL_VELOCITY, dtype=np.float64
        ),
        max_steps=_R1D_MAX_STEPS,
        actuator_mode=str(config.actuator_mode),
        solver_config=dict(config.solver_config),
        mpc_config=dict(config.mpc_config),
        cost_config=dict(config.cost_config),
    )


def _resolve_r1d_backend(config: ResolvedV12Config) -> str:
    """返回 R1D 实际求解后端事实。

    ``cpp`` 已在 :func:`_build_r1d_v12_robot` 入口拒绝；``auto`` / ``python``
    由 ``build_solver`` 落到有限差分（FD），本函数把该事实显式记录为
    ``"python"``，避免下游误以为仍可使用 C++ 解析线性化。

    Args:
        config: resolved V12 配置。

    Returns:
        ``"python"``。
    """
    backend = str(config.solver_backend)
    if backend not in {"auto", "python"}:
        raise ValueError(
            f"R1D 只支持 auto/python 后端，收到 {backend!r}"
        )
    return "python"


def _build_r1d_v12_robot(config: ResolvedV12Config) -> V12RobotBundle:
    """构造 R1D 的 V12 机器人事实（七关节按名绑定 + 独立规划环境）。

    Args:
        config: resolved V12 配置（``robot_id == "unitree_r1d"``）。

    Returns:
        :class:`V12RobotBundle`。

    Raises:
        ValueError: 强制 cpp 后端、缺少 profile/runtime 路径，或非右臂注册表。
    """
    if str(config.solver_backend) == "cpp":
        raise ValueError(
            "R1D 七关节绑定环境不支持 C++ 后端：C++ 线性化只实现 RM65 六维"
            "前缀布局。请使用 --solver-backend auto/python（有限差分 FD），"
            "不静默降级。"
        )
    if config.robot_profile_path is None:
        raise ValueError("R1D 构造需要 robot_profile_path（resolved 配置缺失）")
    if config.runtime_path is None:
        raise ValueError("R1D 构造需要 runtime_path（resolved 配置缺失）")
    profile = load_robot_profile(Path(config.robot_profile_path))
    parameters: RobotRuntimeParameters = load_runtime_parameters(
        Path(config.runtime_path), profile
    )
    racket_profile: RacketProfile | None = (
        load_racket_profile(Path(config.racket_path))
        if config.racket_path is not None
        else None
    )
    task = _resolve_r1d_sim_task(config, profile)
    envs = build_r1d_environments(
        profile,
        parameters,
        task,
        racket_profile=racket_profile,
    )
    execution = envs.execution
    planning = envs.planning
    scene: TennisSceneBundle = execution.scene
    filled = scene.robot.parameters
    _apply_actuator_configuration(
        execution, config, runtime_kp=filled.kp, runtime_kd=filled.kd
    )
    _apply_actuator_configuration(
        planning, config, runtime_kp=filled.kp, runtime_kd=filled.kd
    )
    # R1D 力矩模式数值 parity：URDF 未声明 damping/armature，按 RM65 同口径
    # 注入估计值（estimated / provisional，仅本 V12 R1D 路径生效）。
    _apply_r1d_estimated_joint_dynamics(execution)
    _apply_r1d_estimated_joint_dynamics(planning)
    logger.info(
        "R1D V12 数值 parity：注入估计关节阻尼/armature（URDF 未声明，"
        "estimated/provisional）damping=%s armature=%s",
        list(_R1D_ESTIMATED_DOF_DAMPING),
        list(_R1D_ESTIMATED_DOF_ARMATURE),
    )
    robot_limits = RobotLimits.from_runtime_parameters(
        filled,
        execution.control_range,
    )
    binding: ModelBinding = execution.binding

    def _planning_env_factory() -> RobotEnv:
        """构造独立 R1D 规划环境（全新模型与 MjData）。"""
        bare_bundle = load_robot_model(
            profile,
            parameters,
            racket_profile=racket_profile,
        )
        plan_env = ProfilePlanningEnv(bare_bundle)
        _apply_actuator_configuration(
            plan_env, config, runtime_kp=filled.kp, runtime_kd=filled.kd
        )
        _apply_r1d_estimated_joint_dynamics(plan_env)
        logger.info(
            "R1D planning_env_factory 创建独立规划环境：nq=%d nv=%d nu=%d",
            plan_env.NQ,
            plan_env.NV,
            plan_env.NU,
        )
        return plan_env

    geometry = _resolve_geometry(
        execution.model,
        racket_center_site=_R1D_RACKET_CENTER_SITE,
        racket_face_geom=_R1D_RACKET_FACE_GEOM,
        racket_body=_R1D_RACKET_BODY,
        ball_geom=_R1D_BALL_GEOM,
        ball_joint=_R1D_BALL_JOINT,
        flange_body=None,
    )
    logger.info(
        "构造 R1D V12 bundle：joints=%d nq_exec=%d nq_plan=%d nu=%d backend=%s",
        len(profile.active_joint_names),
        execution.NQ,
        planning.NQ,
        execution.NU,
        _resolve_r1d_backend(config),
    )
    return V12RobotBundle(
        robot_id=profile.robot_id,
        joint_names=tuple(profile.active_joint_names),
        execution=execution,
        planning=planning,
        robot_limits=robot_limits,
        init_q=np.asarray(filled.init_q, dtype=np.float64).copy(),
        geometry=geometry,
        protected_contacts=tuple(task.protected_geom_pairs),
        x_wall=XWallSpec(
            mode="body_plane",
            limit=0.0,
            body_names=(_R1D_TORSO_BODY,),
            protected_geom_names=(_R1D_TORSO_GEOM,),
            enabled=False,
            coordinate_frame="torso_local",
        ),
        planning_env_factory=_planning_env_factory,
        backend=_resolve_r1d_backend(config),
        racket=scene.robot.racket,
        robot_model_bundle=scene.robot,
        _qpos_indices=tuple(binding.qpos_indices),
        _qvel_indices=tuple(binding.qvel_indices),
        _ball_qpos_start=int(scene.ball_qpos_start),
        _ball_qvel_start=int(scene.ball_qvel_start),
    )


def build_v12_robot(config: ResolvedV12Config) -> V12RobotBundle:
    """按 resolved 配置构造 V12 机器人事实集合。

    这是 T5 ``run_v12_episode`` 唯一的机器人构造入口：两条路径都返回同一
    :class:`V12RobotBundle` 契约，管线不感知具体机器人。

    Args:
        config: :class:`ResolvedV12Config`（T3 交付）。

    Returns:
        :class:`V12RobotBundle`。

    Raises:
        ValueError: ``robot_id`` 未知。
    """
    robot_id = str(config.robot_id)
    if robot_id == "rm65":
        return _build_rm65_v12_robot(config)
    if robot_id == "unitree_r1d":
        return _build_r1d_v12_robot(config)
    raise ValueError(
        f"未知机器人 robot_id={robot_id!r}：build_v12_robot 只支持 "
        "'rm65' / 'unitree_r1d'"
    )


__all__ = [
    "RM65SimulationEnv",
    "ReplayState",
    "RobotGeometry",
    "SimulationEnv",
    "V12RobotBundle",
    "XWallSpec",
    "build_v12_robot",
]
