"""V12 回合的组装阶段（MPC → 感知/安全/执行组件 → runner/recorder）。

本模块承担原 ``run_v12_episode`` 中段（约 7–8 步）的职责：把准备阶段产出
的事实组装成可运行的 ``EpisodeRunner``。所有构造参数都来自
:class:`~src.sim.episode.preparation.EpisodePreparation`，本模块不读取配置
以外的全局状态、不启动仿真、不写文件（recorder 仅在 ``dump_trajectory``
非空时构造，构造本身不落盘）。

状态所有权：

* ``mpc`` / ``perception`` / ``safety`` / ``sim_component`` / ``runner``
  均由本阶段创建，并交给 :class:`EpisodeAssembly` 唯一持有；
* ``recorder`` 由 ``hooks.build_post_exec_hooks`` 创建后放入 assembly，
  由执行编排层在击打后阶段与落盘阶段使用；
* 本模块不调用 ``runner.run``，也不释放任何对象（环境生命周期由调用方
  的 bundle 持有）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
from src.ilqt.components.sim_component import (
    DEFAULT_BALL_RADIUS,
    DEFAULT_PROXIMITY_MARGIN,
    DEFAULT_RACKET_RADIUS,
    SimComponent,
)
from src.ilqt.components.sim_perception import SimPerception
from src.ilqt.episode_runner import EpisodeRunner
from src.ilqt.mpc_controller import MPCConfig, MPCController
from src.ilqt.tube_types import TubeConfig
from src.sim.collision_checks import ForbiddenContactChecker
from src.sim.configuration.types import ResolvedV12Config
from src.sim.episode.hooks import _seed_ball_history, build_post_exec_hooks
from src.sim.episode.preparation import (
    EpisodePreparation,
    _effective_backend,
    _mpc_workspace_center_pos,
)

# ---------------------------------------------------------------------------
# 配置/感知辅助
# ---------------------------------------------------------------------------


def _build_mpc_config(
    config: ResolvedV12Config,
    *,
    near_threshold: int,
    shoulder_pos: np.ndarray,
    workspace_radius: float,
    tube_cfg: TubeConfig,
    async_mode: bool,
) -> MPCConfig:
    """从 ``ResolvedV12Config`` 构建 ``MPCConfig``。

    规则（brief §4.2 ⑨）：

    * ``mpc`` 段按键覆盖 ``MPCConfig`` 字段（未知键在此前解析阶段已失败）；
    * ``near_threshold`` 由候选击打步派生（旧脚本 L764）；
    * ``shoulder_pos`` / ``workspace_radius`` / ``tube_cfg`` 显式注入；
    * ``solver_config``：RM65 保持 ``MPCConfig`` 缺省（旧脚本未透传
      ``configs/default.yaml`` 的 ilqt 节，透传会改变 T0 行为）；R1D 必须
      透传解析后的 iLQT 配置（legacy R1D task solver_config 语义）。
    """
    allowed = {f.name for f in fields(MPCConfig)}
    unknown = sorted(
        str(key) for key in dict(config.mpc_config) if str(key) not in allowed
    )
    if unknown:
        raise ValueError(
            f"mpc_config 含未知 MPCConfig 字段：{unknown}（未知键必须早失败，"
            "不得静默忽略）"
        )
    values: dict[str, Any] = {
        str(key): value for key, value in dict(config.mpc_config).items()
    }
    values.update(
        near_threshold=int(near_threshold),
        shoulder_pos=np.asarray(shoulder_pos, dtype=np.float64).copy(),
        workspace_radius=float(workspace_radius),
        tube_cfg=tube_cfg,
        async_mode=bool(async_mode),
    )
    if config.robot_id != "rm65":
        solver_cfg = dict(config.solver_config)
        solver_cfg.pop("backend", None)
        values["solver_config"] = solver_cfg
    return MPCConfig(**values)


def _build_obs_gate(
    config: ResolvedV12Config, dt: float, rng: np.random.Generator
) -> object | None:
    """按感知配置重建观测门控（旧脚本 L923-937 语义）。"""
    perception = config.perception_config
    freq = float(perception.get("obs_freq_effective", 0.0) or 0.0)
    if freq <= 0.0:
        return None
    from src.perception.ball_estimator import BallEstimator
    from src.perception.ball_obs_gate import BallObservationGate

    noise_pos = float(perception.get("obs_noise_pos", 0.0) or 0.0)
    noise_vel = float(perception.get("obs_noise_vel", 0.0) or 0.0)
    kf = None
    if bool(perception.get("obs_use_kf", False)):
        kf = BallEstimator(dt=dt, pos_noise_std=noise_pos, vel_noise_std=noise_vel)
    return BallObservationGate(
        freq, dt, noise_pos=noise_pos, noise_vel=noise_vel, kf=kf, rng=rng,
    )



def _resolve_body_ids(model: object, names: tuple[str, ...]) -> list[int]:
    """把 body 名解析为 MuJoCo body ID（旧 ``_resolve_body_ids`` 逐字搬移）。"""
    import mujoco

    return [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in names
    ]


def _racket_radius_from_bundle(bundle: object) -> float:
    """从机器人 bundle 的球拍几何推导接触判定半径。"""
    racket = getattr(bundle, "racket", None)
    if racket is None:
        return float(DEFAULT_RACKET_RADIUS)
    return float(tuple(racket.face_semi_axes)[1])


def _resolve_geom_ids(model: object, names: tuple[str, ...]) -> tuple[int, ...]:
    """把 geom 名解析为 MuJoCo geom ID 元组（旧实现逐字搬移）。"""
    import mujoco

    return tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in names
    )


def _resolve_geom_pairs(
    model: object, pairs: tuple[tuple[object, ...], ...]
) -> tuple[tuple[int, int], ...]:
    """把禁用接触对（名或 ID）解析为 geom ID 对。

    Raises:
        ValueError: 接触对含未知 geom 名（保护对必须可解析）。
    """
    import mujoco

    resolved: list[tuple[int, int]] = []
    for pair in pairs:
        ids: list[int] = []
        for item in pair:
            if isinstance(item, str):
                geom_id = int(
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, item)
                )
                if geom_id < 0:
                    raise ValueError(
                        f"禁止接触对含未知 geom 名 {item!r}（保护对必须可解析）"
                    )
                ids.append(geom_id)
            else:
                ids.append(int(item))
        resolved.append((ids[0], ids[1]))
    return tuple(resolved)


# ---------------------------------------------------------------------------
# 组装阶段结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeAssembly:
    """一次 V12 回合组装阶段产出的运行期对象集合。

    Attributes:
        mpc: MPC 控制器（``ball_unreachable`` 等事实的唯一来源）。
        mpc_config: 构造 MPC 时使用的配置。
        perception: 感知组件（``SimPerception``）。
        safety: 安全滤波组件（``PredictiveSafetyFilter``）。
        sim_component: 执行 + 诊断组件（history 宿主）。
        runner: ``EpisodeRunner`` 实例（尚未运行）。
        recorder: 轨迹记录器或 ``None``（仅 ``dump_trajectory`` 非空时）。
        hooks: 合并后的 post_exec hook 列表（含 recorder hook）。
        backend: 以 MPC 实例为准的 ``{"requested", "effective"}``。
        hard_x_body_ids: X 墙硬约束 body ID 列表（RM65 解析，R1D 为空）。
        ball_radius: 接触门控球半径。
        racket_radius: 接触门控球拍半径。
        proximity_margin: 接触门控余量。
        contact_ball_geom_id: R1D 显式球 geom ID（RM65 为 ``None``）。
        contact_racket_geom_ids: R1D 显式球拍 geom ID 元组（RM65 为 ``None``）。
        obs_gate: 观测门控或 ``None``（保留引用便于诊断/测试）。
    """

    mpc: Any
    mpc_config: MPCConfig
    perception: Any
    safety: Any
    sim_component: SimComponent
    runner: EpisodeRunner
    recorder: Any
    hooks: list[Callable[[object], None]]
    backend: dict[str, str]
    hard_x_body_ids: list[int]
    ball_radius: float
    racket_radius: float
    proximity_margin: float
    contact_ball_geom_id: int | None
    contact_racket_geom_ids: tuple[int, ...] | None
    obs_gate: Any


# ---------------------------------------------------------------------------
# 阶段入口
# ---------------------------------------------------------------------------


def assemble_episode(
    prep: EpisodePreparation,
    *,
    mpc_controller_cls: Callable[..., Any] = MPCController,
    sim_component_cls: Callable[..., Any] = SimComponent,
    dump_trajectory: Path | None = None,
    post_exec_hooks: tuple[Callable[[object], None], ...] = (),
) -> EpisodeAssembly:
    """把准备阶段事实组装成可运行的 ``EpisodeRunner``。

    行数说明（contracts §1.7）：约 156 行，主要是 RM65 / R1D 两条互斥的
    组件构造分支（安全过滤器、球/拍半径与 contact geom）。再抽子函数会把
    ``prep`` 的十余个字段作为参数透传，收益低；同一处对照两套分支更利于
    审查参数差异，故保留。

    Args:
        prep: 准备阶段结果（``hit_info`` 必须非 ``None``）。
        mpc_controller_cls: MPC 控制器类；facade 在调用时注入当前模块属性，
            使 monkeypatch 生效。
        sim_component_cls: 执行组件类；同样由 facade 在调用时注入。
        dump_trajectory: 轨迹 npz 落盘路径；决定是否构造 recorder。
        post_exec_hooks: 调用方额外 hook，追加在记录 hook 之后。

    Returns:
        :class:`EpisodeAssembly`。
    """
    config = prep.config
    bundle = prep.bundle
    execution = prep.execution
    robot_id = prep.robot_id
    scene_cfg = prep.scene_cfg
    limits = prep.limits
    init_q = prep.init_q

    # ── 7. MPC 配置/控制器（旧 L839-917）──────────────────────────────
    mpc_config = _build_mpc_config(
        config,
        near_threshold=int(prep.near_threshold),
        shoulder_pos=_mpc_workspace_center_pos(config, prep.shoulder_pos),
        workspace_radius=prep.workspace_radius,
        tube_cfg=prep.tube_cfg,
        async_mode=bool(prep.mpc_cfg.get("async_mode", False)),
    )
    # T4 适配器工厂对 RM65 已与 T0 逐位等价（见文件头说明）；R1D 亦复用
    # 同一 bundle 工厂，两个机器人不再有分支。
    planning_env_factory: Callable[[], object] | None = bundle.planning_env_factory
    mpc = mpc_controller_cls(
        prep.planning,
        mpc_config,
        robot_limits=limits,
        planning_env_factory=planning_env_factory,
    )
    # 求解器实际后端以构造出的实例为准（P2：旧实现只看环境兼容性，把已启用
    # 的 C++ 内核误记为 python-analytic）。早退路径仍用准备阶段的推断值。
    backend = _effective_backend(prep.planning, bundle, mpc)

    # ── 8. 组件组装（旧 L919-982）─────────────────────────────────────
    # 球速度 history 由 SimComponent 与 ball_pos_history 同步维护（R07），
    # 不再用 post_exec hook 另记（旧实现会整体错位一帧）。
    obs_gate = _build_obs_gate(config, prep.dt, prep.rng)
    perception = SimPerception(execution, obs_gate=obs_gate)

    hard_x_body_ids: list[int] = []
    if robot_id == "rm65":
        # 旧 RM65 路径：默认滤波参数构造（与 T0 逐位一致）
        safety = PredictiveSafetyFilter(
            execution, limits, is_position_mode=prep.is_position_mode,
        )
        hard_x_body_ids = _resolve_body_ids(
            execution.model, tuple(bundle.x_wall.body_names),
        )
    else:
        checker = ForbiddenContactChecker.from_env(
            execution,
            _resolve_geom_pairs(
                execution.model, tuple(bundle.protected_contacts),
            ),
        )
        safety = PredictiveSafetyFilter(
            execution,
            limits,
            is_position_mode=prep.is_position_mode,
            arm_nq=int(execution.NQ),
            enable_x_wall=bool(bundle.x_wall.enabled),
            contact_checker=checker,
            x_wall_body_names=tuple(bundle.x_wall.body_names),
            x_wall_limit=float(bundle.x_wall.limit),
        )

    ball_radius = float(scene_cfg.get("ball_radius", DEFAULT_BALL_RADIUS))
    racket_radius = _racket_radius_from_bundle(bundle)
    proximity_margin = float(DEFAULT_PROXIMITY_MARGIN)
    if robot_id == "rm65":
        # RM65 显式传 V12 旧值，确保与旧 SimComponent 缺省逐位一致；
        # 接触判定保持旧 geom 名子串路径（不传解析 ID）
        ball_radius = DEFAULT_BALL_RADIUS
        racket_radius = DEFAULT_RACKET_RADIUS
        contact_ball_geom_id: int | None = None
        contact_racket_geom_ids: tuple[int, ...] | None = None
    else:
        contact_ball_geom_id = int(bundle.geometry.ball_geom_id)
        contact_racket_geom_ids = _resolve_geom_ids(
            execution.model, bundle.geometry.racket_geom_names,
        )
    sim_component = sim_component_cls(
        execution,
        mpc,
        limits,
        init_q,
        prep.is_position_mode,
        hard_x_body_ids,
        prep.initial_hitting_tube,
        prep.tube_cfg,
        prep.dt,
        always_collision=prep.always_collision_value,
        ball_geom_id=contact_ball_geom_id,
        racket_geom_ids=contact_racket_geom_ids,
        ball_radius=ball_radius,
        racket_radius=racket_radius,
        proximity_margin=proximity_margin,
    )
    sim_component.X_history = [prep.x0.copy()]
    # R07：球位置/速度 history 首帧必须与环境实际初态（p0_real/v0_real）
    # 同源。旧实现用 p0（扰动前）会与 set_ball_state(p0_real, ...) 及产物
    # ball_initial 自相矛盾（ball_speed_perturb_pct≠0 时 p0_real ≠ p0）。
    _seed_ball_history(
        sim_component, ball_pos=prep.p0_real, ball_vel=prep.v0_real,
    )

    recorder, hooks = build_post_exec_hooks(
        prep,
        dump_trajectory=dump_trajectory,
        post_exec_hooks=post_exec_hooks,
    )

    runner = EpisodeRunner(
        mpc=mpc,
        perception=perception,
        safety=safety,
        executor=sim_component,
        post_exec_hooks=list(hooks),
    )

    return EpisodeAssembly(
        mpc=mpc,
        mpc_config=mpc_config,
        perception=perception,
        safety=safety,
        sim_component=sim_component,
        runner=runner,
        recorder=recorder,
        hooks=hooks,
        backend=backend,
        hard_x_body_ids=hard_x_body_ids,
        ball_radius=float(ball_radius),
        racket_radius=float(racket_radius),
        proximity_margin=proximity_margin,
        contact_ball_geom_id=contact_ball_geom_id,
        contact_racket_geom_ids=contact_racket_geom_ids,
        obs_gate=obs_gate,
    )
