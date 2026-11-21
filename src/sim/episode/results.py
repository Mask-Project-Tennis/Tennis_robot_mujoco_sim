"""V12 回合的结果构造阶段（指标汇总 → R1D artifact → 轨迹落盘 → 结果）。

本模块承担原 ``run_v12_episode`` 后半段（约 11–15 步）以及全部「纯函数」
形态的配置/产物辅助：

* ``_unreachable_result``：球不可达早退结果（与正常路径字段口径一致）；
* ``build_result``：把 ``SimComponent`` 的 history/接触事实、MPC 运行指标
  汇总成 :class:`V12EpisodeResult`，并按需写 ``dump_trajectory``；
* ``build_unreachable_result``：早退编排入口（显式接收准备阶段快照）；
* ``_explicit_runtime_options`` / ``_runtime_option_overrides`` /
  ``_apply_limits_overrides`` / ``_real_config_snapshot``：旧调用形态兼容
  适配器（被测直接调用，必须保持签名与语义）；
* ``_resolve_failure_reason`` / ``_serialize_hit_window`` /
  ``_r1d_artifact_metadata`` / ``_model_dependency_manifest``：结果字段
  的纯函数派生。

约定：本模块不推进仿真、不修改 ``SimComponent`` 的 history；唯一副作用是
``build_result`` 在 ``dump_trajectory`` 非空时调用 recorder.save（与原实现
一致）。所有返回数组均为副本。
"""

from __future__ import annotations

import copy
import hashlib
import time
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.ilqt.components.sim_component import DEFAULT_PROXIMITY_MARGIN
from src.ilqt.tube_types import HitWindow
from src.robot.constants import INIT_Q_LEFT
from src.robot.court_geometry import court_profile_metadata
from src.sim.configuration.types import ResolvedV12Config
from src.sim.episode.post_hit import _POST_HIT_ALLOWED_STOP_REASONS
from src.sim.episode.preparation import (
    EpisodePreparation,
    _set_rm65_left_arm,
)
from src.sim.episode.types import V12EpisodeResult
from src.sim.episode_runtime import (
    UnsetType,
    apply_runtime_overrides,
    real_config_snapshot,
    resolve_runtime_overrides,
)
from src.sim.hit_detection import classify_hit_result
from src.sim.v12_robot_adapter import V12RobotBundle

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查使用，避免运行期循环导入
    from src.sim.episode.assembly import EpisodeAssembly


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """返回文件 SHA256 十六进制摘要。"""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene_to_jsonable(scene: object) -> dict[str, Any]:
    """把 ``SceneParameters`` 逐字段转成普通 JSON 可序列化容器。

    不能直接用 ``dataclasses.asdict``：``fixed_joint_poses`` 是
    ``MappingProxyType``（只读视图），asdict 深拷贝时会抛
    ``TypeError: cannot pickle 'mappingproxy' object``（T4 报告 R1）。
    本函数在 asdict 之前先把映射转普通 dict（其余字段数值/元组/列表）。
    """
    result: dict[str, Any] = {}
    for item in fields(scene):  # type: ignore[arg-type]
        value = getattr(scene, item.name)
        if isinstance(value, dict) or hasattr(value, "items"):
            result[item.name] = {str(k): float(v) for k, v in dict(value).items()}
        elif isinstance(value, np.ndarray):
            result[item.name] = [float(x) for x in value.reshape(-1)]
        elif isinstance(value, (list, tuple)):
            result[item.name] = [
                float(x) if isinstance(x, (int, float, np.floating)) else x
                for x in value
            ]
        elif isinstance(value, (int, float, np.floating)):
            result[item.name] = float(value)
        else:
            result[item.name] = value
    return result


def _serialize_hit_window(window: HitWindow | None) -> dict[str, Any] | None:
    """把 ``HitWindow`` 序列化为普通容器（绘图/产物用）。"""
    if window is None:
        return None
    return {
        "best_k": int(window.best_k),
        "k_candidates": [int(k) for k in window.k_candidates],
        "p_ball_candidates": np.asarray(window.p_ball_candidates).tolist(),
        "v_ball_candidates": np.asarray(window.v_ball_candidates).tolist(),
        "weights": np.asarray(window.weights).tolist(),
    }


def _r1d_artifact_metadata(
    config: ResolvedV12Config,
    bundle: V12RobotBundle,
    *,
    side: str,
    joint_names: tuple[str, ...],
    ball_initial: tuple[np.ndarray, np.ndarray],
    backend: dict[str, str],
) -> dict[str, Any]:
    """生成 R1D 旧 ``r1d_sim_episode_v1`` schema 所需静态元数据。

    只依赖 bundle 事实与配置路径（不引用 env/mpc），供
    ``src/sim/r1d_runner.py::write_v12_r1d_artifacts`` 复用。
    """
    model_bundle = bundle.robot_model_bundle
    scene_snapshot: dict[str, Any] = {}
    provisional_fields: list[str] = []
    if model_bundle is not None:
        scene_snapshot = _scene_to_jsonable(model_bundle.parameters.scene)
        provisional_fields = list(model_bundle.parameters.provisional_fields or [])
        template_path = Path(model_bundle.profile.model_path)
        # 快照记录白线布局口径与几何版本，便于回放判断 V01/I2 的视觉来源。
        # legacy_v1 是既有默认：为保持 T7 冻结场景的 scene_snapshot 哈希不变，
        # 默认口径不写入 profile 键；只有显式 ITF 才记录额外视觉元数据。
        court_profile = str(scene_snapshot.get("court_profile", "legacy_v1"))
        if court_profile == "legacy_v1":
            scene_snapshot.pop("court_profile", None)
        else:
            court_meta = court_profile_metadata(court_profile)
            scene_snapshot = {
                **scene_snapshot,
                "court_geometry_version": court_meta["geometry_version"],
                "court_line_count": court_meta["line_count"],
            }
    else:  # pragma: no cover - R1D 必有模型 bundle
        template_path = Path("")

    hashes = {
        "profile_yaml": _sha256_file(Path(config.robot_profile_path)),
        "runtime_yaml": _sha256_file(Path(config.runtime_path)),
        "task_yaml": _sha256_file(Path(config.task_path))
        if config.task_path is not None
        else "",
        "model_template": _sha256_file(template_path) if template_path else "",
    }
    return {
        "schema": "r1d_sim_episode_v1",
        "simulation_only": True,
        "robot_id": config.robot_id,
        "side": side,
        "joint_names": [str(name) for name in joint_names],
        "dt": float(config.dt),
        "mode": str(config.actuator_mode),
        "backend": dict(backend),
        "input_sha256": hashes,
        "provisional_fields": provisional_fields,
        "fixed_scene_params": scene_snapshot,
        "workspace": {
            "center": np.asarray(config.target_center).tolist(),
            "radius": float(config.workspace_radius),
        },
        "ball_initial": {
            "position": np.asarray(ball_initial[0]).tolist(),
            "velocity": np.asarray(ball_initial[1]).tolist(),
        },
        "seed": None if config.seed is None else int(config.seed),
        "environment": classify_hit_result(False, False, 0.0, robot_name="R1-A7-D"),
    }


def _model_dependency_manifest(bundle: V12RobotBundle) -> list[dict[str, Any]]:
    """返回模型/资产依赖清单的独立副本（F08/I1 接线）。

    清单由 ``model_factory.collect_model_dependencies`` 在装配时生成，
    记录模板、include、兄弟片段与网格 SHA256；这里只做深拷贝，避免
    结果对象与 bundle 共享可变容器。
    """
    model_bundle = getattr(bundle, "robot_model_bundle", None)
    entries = getattr(model_bundle, "dependency_manifest", ()) or ()
    return copy.deepcopy([dict(item) for item in entries])


# ---------------------------------------------------------------------------
# 运行时覆盖适配器（旧调用形态，测试直接导入）
# ---------------------------------------------------------------------------


def _explicit_runtime_options(
    *,
    first_plan_flight: bool | UnsetType,
    always_collision: bool | UnsetType,
    ball_speed_perturb_pct: float | UnsetType,
    fixed_direction: bool | UnsetType,
    hit_direction: tuple[float, float, float] | None | UnsetType,
    limits_config: Path | None | UnsetType,
    max_tcp: float | None | UnsetType,
    terminal_exempt_steps: int | None | UnsetType,
    dq_max_fraction: float | None | UnsetType,
) -> dict[str, Any]:
    """收集调用方「真正传入」的运行时覆盖键（F01）。

    Args:
        first_plan_flight: 首规划飞行开关；``UNSET`` 表示未传。
        always_collision: 球-拍碰撞常开开关；``UNSET`` 表示未传。
        ball_speed_perturb_pct: 球速耦合扰动百分比；``UNSET`` 表示未传。
        fixed_direction: 固定随挥方向开关；``UNSET`` 表示未传。
        hit_direction: 固定随挥方向；``None`` 表示清除已保存覆盖。
        limits_config: 显式限位文件；``None`` 表示清除已保存文件层。
        max_tcp: TCP 上限覆盖；``None`` 表示清除已保存覆盖。
        terminal_exempt_steps: 终段豁免步数覆盖；``None`` 表示清除。
        dq_max_fraction: 关节增量分数覆盖；``None`` 表示清除。

    Returns:
        仅含显式键的字典（可空字段保留显式 ``None``）。
    """
    candidates: dict[str, Any] = {
        "first_plan_flight": first_plan_flight,
        "always_collision": always_collision,
        "ball_speed_perturb_pct": ball_speed_perturb_pct,
        "fixed_direction": fixed_direction,
        "hit_direction": hit_direction,
        "limits_config": limits_config,
        "max_tcp": max_tcp,
        "terminal_exempt_steps": terminal_exempt_steps,
        "dq_max_fraction": dq_max_fraction,
    }
    explicit = {
        key: value
        for key, value in candidates.items()
        if not isinstance(value, UnsetType)
    }
    # 非可空字段的显式 None 等价于「未提供」：继续继承已保存值/默认
    for key in (
        "first_plan_flight",
        "always_collision",
        "ball_speed_perturb_pct",
        "fixed_direction",
    ):
        if key in explicit and explicit[key] is None:
            del explicit[key]
    return explicit


def _real_config_snapshot(config: object) -> dict[str, Any]:
    """把 ``RealRobotConfig`` 拍平成 JSON 原生类型快照（R16）。

    委托 :func:`src.sim.episode_runtime.real_config_snapshot`，保证恢复侧与
    记录侧同一口径（数组/列表→float 列表，标量原样）。
    """
    return real_config_snapshot(config)


def _runtime_option_overrides(
    *,
    ball_speed_perturb_pct: float,
    always_collision: bool,
    first_plan_flight: bool,
    fixed_direction: bool,
    hit_direction: tuple[float, float, float] | None,
) -> dict[str, Any]:
    """把运行选项规范化为 JSON 原生覆盖项（R16）。

    键名与 ``run_v12_episode`` 参数同名；``hit_direction=None`` 保留 null，
    非空时转 ``list[float]``（配置字段本身是 tuple，JSON 需要 list）。

    Args:
        ball_speed_perturb_pct: 球速耦合扰动百分比。
        always_collision: 击球前球-拍碰撞是否常开。
        first_plan_flight: 首次规划期间是否让球飞行。
        fixed_direction: 随挥方向是否用配置固定方向。
        hit_direction: 固定随挥方向覆盖（None 表示用配置值）。

    Returns:
        仅含 JSON 原生类型的覆盖 dict。
    """
    hit_direction_value: list[float] | None = None
    if hit_direction is not None:
        hit_direction_value = [
            float(x)
            for x in np.asarray(hit_direction, dtype=np.float64).ravel()
        ]
    return {
        "ball_speed_perturb_pct": float(ball_speed_perturb_pct),
        "always_collision": bool(always_collision),
        "first_plan_flight": bool(first_plan_flight),
        "fixed_direction": bool(fixed_direction),
        "hit_direction": hit_direction_value,
    }


def _apply_limits_overrides(
    bundle: V12RobotBundle,
    *,
    limits_config: Path | None = None,
    max_tcp: float | None = None,
    terminal_exempt_steps: int | None = None,
    dq_max_fraction: float | None = None,
    dt: float,
    saved: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """应用限位 CLI 覆盖并返回实际生效值（旧 kwargs 兼容适配器）。

    该函数保留旧调用形态（位置参数 ``bundle`` + 关键字覆盖），内部委托
    :func:`src.sim.episode_runtime.resolve_runtime_overrides` 与
    :func:`~src.sim.episode_runtime.apply_runtime_overrides`，因此也能消费
    ``saved``（artifact 里已保存的 runtime_overrides）。

    与 ``run_v12_episode`` 的 UNSET 语义不同：这里的 ``None`` 表示「未提供」
    （保持旧适配器行为），清除覆盖请直接使用 episode_runtime 的公开接口。

    Returns:
        本次实际生效的运行时覆盖（JSON 原生类型，可直接写入产物）。
        字段含 ``limits_config``（绝对路径+sha256+解析内容）、``max_tcp``、
        ``terminal_exempt_steps``、``dq_max_fraction``、``qdot_max``、
        ``dq_max``、``init_q``；每项都同时给出 requested 与实际生效值（R16）。
    """
    explicit: dict[str, Any] = {}
    if limits_config is not None:
        explicit["limits_config"] = Path(limits_config)
    if max_tcp is not None:
        explicit["max_tcp"] = float(max_tcp)
    if terminal_exempt_steps is not None:
        explicit["terminal_exempt_steps"] = int(terminal_exempt_steps)
    if dq_max_fraction is not None:
        explicit["dq_max_fraction"] = float(dq_max_fraction)
    resolved = resolve_runtime_overrides(explicit, saved)
    return apply_runtime_overrides(bundle, resolved, dt=float(dt))


def _resolve_failure_reason(
    *,
    run_metrics: Mapping[str, Any],
    ball_unreachable: bool,
    stop_reason: str | None,
) -> str | None:
    """按旧 ``r1d_runner`` 顺序判定失败原因（R08 纯函数）。

    顺序固定为：``metrics["error"]`` → ``ball_unreachable`` →
    ``stop_reason == "unsafe"``（``safety_filter_stop``）→ ``no_ball`` 或
    提前退出且 ``safe_steps < total_steps``（``ball_state_unavailable``）→
    正常完成（``mpc_done``/``max_steps``）返回 ``None``。

    与旧实现的差别：不再以 ``not mpc_done`` 作为前置条件，因此
    ``mpc_done=True`` 但最后一步被安全滤波拒绝时仍会报
    ``safety_filter_stop``，不可达回合即使 done 也报 ``ball_unreachable``。

    Args:
        run_metrics: ``EpisodeRunner.run`` 返回的指标。
        ball_unreachable: MPC 层是否标记来球不可达。
        stop_reason: 终止原因（``"mpc_done"``/``"max_steps"``/``"no_ball"``/
            ``"unsafe"``/``None``）。

    Returns:
        失败原因字符串；正常完成返回 ``None``。
    """
    error = run_metrics.get("error")
    if error:
        return str(error)
    if ball_unreachable:
        return "ball_unreachable"
    if stop_reason == "unsafe":
        return "safety_filter_stop"
    safe_steps = int(run_metrics.get("safe_steps", 0))
    total_steps = int(run_metrics.get("total_steps", 0))
    if stop_reason == "no_ball" or (
        stop_reason not in _POST_HIT_ALLOWED_STOP_REASONS
        and safe_steps < total_steps
    ):
        return "ball_state_unavailable"
    return None


# ---------------------------------------------------------------------------
# 不可达早退
# ---------------------------------------------------------------------------


def _unreachable_result(
    *,
    config: ResolvedV12Config,
    bundle: V12RobotBundle,
    execution: object,
    backend: dict[str, str],
    p0: np.ndarray,
    v0: np.ndarray,
    rng: np.random.Generator,
    target_center: np.ndarray,
    workspace_radius: float,
) -> V12EpisodeResult:
    """构造「球不在工作空间」早退结果（旧脚本打印后 return 的等价物）。

    行数说明（contracts §1.7）：约 170 行，主体是与 :func:`build_result`
    对照的 metrics 字面量镜像。两份字面量放在同一文件相邻位置，便于 schema
    漂移审查；抽成共享构造函数会掩盖“早退口径与正常路径独立”的事实，故不抽。
    """
    nq = int(execution.NQ)  # type: ignore[attr-defined]
    nx = int(execution.NX)  # type: ignore[attr-defined]
    nu = int(execution.NU)  # type: ignore[attr-defined]
    x0 = np.zeros(nx)
    x0[:nq] = np.asarray(bundle.init_q, dtype=np.float64)
    dt = float(config.dt)
    runtime = dict(config.runtime_overrides)
    metrics: dict[str, Any] = {
        "total_steps": 0,
        "safe_steps": 0,
        "mpc_done": False,
        "stop_reason": None,
        "unsafe": False,
        "first_plan_ms": 0.0,
        "wall_time_s": 0.0,
        "mpc_time_s": 0.0,
        "phase_timings": {"total_s": 0.0, "mpc_s": 0.0},
        "backend": dict(backend),
        "failure_reason": "ball_unreachable",
        "runtime_overrides": dict(config.runtime_overrides),
        "ablation_mode": str(config.mpc_config.get("ablation_mode", "full")),
        "is_position_mode": str(config.actuator_mode) == "position",
        "n_mpc_steps": 0,
        "min_dist": float("inf"),
        "max_qdot_ratio": 0.0,
        "max_tcp_speed": 0.0,
        "max_racket_face_speed": 0.0,
        "ball_near_ms": 0.0,
        "tube_ready_ms": 0.0,
        "hit_time_error_ms": 0.0,
        "hit_position_error": 0.0,
        "pos_error": 0.0,
        "pos_error_plan": 0.0,
        "vel_error": 0.0,
        "v_racket_at_hit": 0.0,
        "k_hit": None,
        "p_hit": None,
        "v_hit_at_contact": None,
        "v_ball_hit": None,
        "n_des": None,
        "p_ee_final": None,
        "p_racket_init": None,
        "post_hit_steps": 0,
        "post_hit_skipped_reason": "ball_unreachable",
        "ball_speed_perturb_pct": float(
            runtime.get("ball_speed_perturb_pct", 0.0)
        ),
        "always_collision": bool(runtime.get("always_collision", False)),
        "unreachable": True,
        "proximity": {
            "ball_radius": float(config.scene_config.get("ball_radius", 0.033)),
            "racket_radius": (
                float(bundle.racket.face_semi_axes[1])
                if bundle.racket is not None else 0.12
            ),
            "proximity_margin": float(DEFAULT_PROXIMITY_MARGIN),
            "ball_near_threshold": float(
                float(config.scene_config.get("ball_radius", 0.033))
                + (float(bundle.racket.face_semi_axes[1])
                   if bundle.racket is not None else 0.12)
                + float(DEFAULT_PROXIMITY_MARGIN)
            ),
        },
        "histories": {
            "distances": [], "normal_align": [], "ball_near": [],
            "tube_ready": [], "pos_error": [],
        },
        "hit_window": None,
        "solver_config": dict(config.solver_config),
        "serve": {
            "mode": str(config.serve_config.get("mode")),
            "ball_speed": config.serve_config.get("ball_speed"),
            "p0": np.asarray(p0).tolist(),
            "v0": np.asarray(v0).tolist(),
        },
        "convergence": {},
        "workspace": {
            "center": np.asarray(target_center).tolist(),
            "radius": float(workspace_radius),
        },
    }

    if str(bundle.robot_id) != "rm65":
        side = str(bundle.joint_names[0]).split("_", 1)[0]
        max_steps = int(config.mpc_config.get("total_horizon", 200)) + int(
            config.mpc_config.get("follow_through_steps", 0) or 0
        )
        artifact = _r1d_artifact_metadata(
            config,
            bundle,
            side=side,
            joint_names=tuple(str(name) for name in bundle.joint_names),
            ball_initial=(
                np.asarray(p0, dtype=np.float64),
                np.asarray(v0, dtype=np.float64),
            ),
            backend=backend,
        )
        artifact.update(
            episode={
                "max_steps": int(max_steps),
                "total_steps": 0,
                "safe_steps": 0,
                "mpc_done": False,
                "first_plan_ms": 0.0,
                "wall_time_s": 0.0,
                "failure_reason": "ball_unreachable",
            },
            hit={
                "type": "miss",
                "step": -1,
                "time_s": None,
                "racket_face_speed": None,
                "p_ee_at_hit": None,
                "ball_pos_at_hit": None,
                "ball_out_speed": None,
                "ball_out_vy": None,
                "message": classify_hit_result(
                    False, False, 0.0, robot_name="R1-A7-D",
                ),
            },
            metrics={
                "n_mpc_steps": 0,
                "min_dist": None,
                "max_qdot_ratio": 0.0,
                "max_tcp_speed": 0.0,
                "max_racket_face_speed": 0.0,
                "active_contact": False,
                "passive_contact": False,
            },
        )
        metrics["r1d_artifact"] = artifact

    return V12EpisodeResult(
        config=config,
        robot_id=str(bundle.robot_id),
        mode=str(config.actuator_mode),
        joint_names=tuple(str(name) for name in bundle.joint_names),
        dt=dt,
        racket_id=(None if bundle.racket is None else str(bundle.racket.racket_id)),
        X=np.asarray([x0], dtype=np.float64),
        U=np.zeros((0, nu), dtype=np.float64),
        ball_pos=np.asarray([p0], dtype=np.float64),
        ball_vel=np.asarray([v0], dtype=np.float64),
        time=np.asarray([0.0], dtype=np.float64),
        racket_pos=np.zeros((1, 3), dtype=np.float64),
        hit_type="miss",
        hit_step=-1,
        active_contact=False,
        passive_contact=False,
        v_ee_at_hit=None,
        p_ee_at_hit=None,
        ball_pos_at_hit=None,
        ball_out_speed=None,
        ball_out_vy=None,
        metrics=metrics,
        model_dependency_manifest=_model_dependency_manifest(bundle),
    )


def build_unreachable_result(prep: EpisodePreparation) -> V12EpisodeResult:
    """早退编排入口：由准备阶段快照构造 ``ball_unreachable`` 结果。

    Args:
        prep: ``hit_info is None`` 的准备阶段结果。

    Returns:
        :class:`V12EpisodeResult`。
    """
    return _unreachable_result(
        config=prep.config,
        bundle=prep.bundle,
        execution=prep.execution,
        backend=prep.backend,
        p0=prep.p0,
        v0=prep.v0,
        rng=prep.rng,
        target_center=prep.target_center,
        workspace_radius=prep.workspace_radius,
    )


# ---------------------------------------------------------------------------
# 正常结果构造
# ---------------------------------------------------------------------------


def build_result(
    *,
    prep: EpisodePreparation,
    assembly: EpisodeAssembly,
    run_metrics: Mapping[str, Any],
    x_current: np.ndarray,
    t_total_start: float,
    t_mpc_end: float,
    post_hit_steps: int,
    post_hit_skipped_reason: str | None,
    dump_trajectory: Path | None,
) -> V12EpisodeResult:
    """把运行期事实汇总为 ``V12EpisodeResult``（旧 L1033-1150 语义）。

    指标键集合、数值口径、R1D artifact 子结构与轨迹落盘条件全部逐字保持；
    本函数不推进仿真，只读 ``SimComponent`` 的 history 与接触事实。

    行数说明（contracts §1.7）：约 370 行，主体是一段必须逐字保持的 metrics
    字面量（40+ 键）与拍心轨迹重建。把键集合抽成 helper 会退化为 40 参数的
    转发层且更容易与旧 schema 漂移；可独立判定的纯函数已抽到模块级
    （``_resolve_failure_reason``、``_serialize_hit_window``、
    ``_r1d_artifact_metadata``、``_unreachable_result`` 等）。

    Args:
        prep: 准备阶段结果（配置/初态/击打事实）。
        assembly: 组装阶段结果（MPC/执行组件/recorder/后端）。
        run_metrics: ``EpisodeRunner.run`` 的返回指标。
        x_current: 击打后阶段结束时的臂状态副本。
        t_total_start: 主运行段起点 ``perf_counter``。
        t_mpc_end: 主运行段终点 ``perf_counter``。
        post_hit_steps: 击打后实际执行步数。
        post_hit_skipped_reason: 跳过原因或 ``None``。
        dump_trajectory: 轨迹落盘路径或 ``None``。

    Returns:
        :class:`V12EpisodeResult`。
    """
    config = prep.config
    bundle = prep.bundle
    execution = prep.execution
    sim_component = assembly.sim_component
    recorder = assembly.recorder
    robot_id = prep.robot_id
    dt = prep.dt
    total_steps = prep.total_steps
    k_hit_total = int(prep.k_hit_total)
    p_hit = prep.p_hit
    v_ball_hit = prep.v_ball_hit
    n_des_single = prep.n_des_single
    d_follow = prep.d_follow
    hit_direction_source = prep.hit_direction_source
    v_hit_at_contact = prep.v_hit_at_contact
    ablation_mode = prep.ablation_mode
    shoulder_pos = prep.shoulder_pos
    shoulder_pos_source = prep.shoulder_pos_source
    p_racket_init = prep.p_racket_init
    p0 = prep.p0
    v0 = prep.v0
    p0_real = prep.p0_real
    v0_real = prep.v0_real
    ball_speed_perturb_value = prep.ball_speed_perturb_value
    always_collision_value = prep.always_collision_value
    hit_window = prep.hit_window
    backend = assembly.backend
    ball_radius = assembly.ball_radius
    racket_radius = assembly.racket_radius
    proximity_margin = assembly.proximity_margin
    ball_unreachable = bool(assembly.mpc.ball_unreachable)

    t_total = time.perf_counter() - t_total_start
    t_mpc = t_mpc_end - t_total_start

    X_history = [np.asarray(x, dtype=np.float64) for x in sim_component.X_history]
    U_history = [np.asarray(u, dtype=np.float64) for u in sim_component.U_history]
    ball_pos_history = [
        np.asarray(x, dtype=np.float64) for x in sim_component.ball_pos_history
    ]

    p_ee_at_hit = sim_component.p_ee_at_hit
    ball_pos_at_hit = sim_component.ball_pos_at_hit
    if p_ee_at_hit is not None:
        p_ee_final = np.asarray(p_ee_at_hit, dtype=np.float64).copy()
    else:
        execution.set_arm_state(x_current)
        execution.update_kinematics()
        p_ee_final = np.asarray(execution.get_ee_pos(), dtype=np.float64).copy()
    v_ee_final = np.asarray(execution.get_ee_vel(), dtype=np.float64).copy()

    pos_error_plan = float(np.linalg.norm(p_ee_final - p_hit))
    if ball_pos_at_hit is not None:
        pos_error = float(np.linalg.norm(p_ee_final - ball_pos_at_hit))
    else:
        pos_error = pos_error_plan
    vel_error = float(np.linalg.norm(v_ee_final - v_hit_at_contact))

    d_arr = np.asarray(sim_component.distances_history, dtype=np.float64)
    min_dist = float(np.min(d_arr)) if d_arr.size else float("inf")
    ball_near_duration = int(np.sum(np.asarray(
        sim_component.ball_near_history, dtype=bool,
    )))
    ball_near_ms = ball_near_duration * dt * 1000
    tube_ready_duration = int(np.sum(np.asarray(
        sim_component.tube_ready_history, dtype=bool,
    )))
    tube_ready_ms = tube_ready_duration * dt * 1000

    hit_step = int(sim_component.hit_step)
    if p_ee_at_hit is not None and hit_step >= 0:
        hit_time_error = abs(hit_step * dt - k_hit_total * dt) * 1000
        hit_position_error = (
            float(np.linalg.norm(
                np.asarray(p_ee_at_hit, dtype=np.float64)
                - np.asarray(ball_pos_at_hit, dtype=np.float64)
            ))
            if ball_pos_at_hit is not None
            else float(np.linalg.norm(
                np.asarray(p_ee_at_hit, dtype=np.float64) - p_hit
            ))
        )
    else:
        hit_time_error = 0.0
        hit_position_error = pos_error

    active_contact = bool(sim_component.active_contact)
    passive_contact = bool(sim_component.passive_contact)
    hit_type_en = "active" if active_contact else (
        "passive" if passive_contact else "miss"
    )
    v_racket_at_hit_val = (
        float(sim_component.v_ee_at_hit)
        if sim_component.v_ee_at_hit is not None else 0.0
    )

    # 拍心位置轨迹（旧 L1160-1169：reset 后逐帧 set_arm_state 取 FK）
    bundle.reset()
    if robot_id == "rm65":
        _set_rm65_left_arm(execution, np.asarray(INIT_Q_LEFT, dtype=np.float64))
    execution.update_kinematics()
    racket_pos_list: list[np.ndarray] = []
    for state in X_history:
        execution.set_arm_state(state)
        racket_pos_list.append(
            np.asarray(execution.get_ee_pos(), dtype=np.float64).copy()
        )
    racket_pos_arr = (
        np.asarray(racket_pos_list, dtype=np.float64)
        if racket_pos_list else np.zeros((0, 3))
    )

    X_arr = np.asarray(X_history, dtype=np.float64)
    U_arr = (
        np.asarray(U_history, dtype=np.float64)
        if U_history else np.zeros((0, int(execution.NU)))
    )
    ball_pos_arr = np.asarray(ball_pos_history, dtype=np.float64)
    # R07：球速度与位置必须逐帧对齐（同一长度），不静默补齐。
    ball_vel_list = [
        np.asarray(v, dtype=np.float64) for v in sim_component.ball_vel_history
    ]
    if len(ball_vel_list) != len(ball_pos_history):
        raise ValueError(
            "ball_vel_history 与 ball_pos_history 长度不一致："
            f"{len(ball_vel_list)} != {len(ball_pos_history)}"
        )
    ball_vel_arr = np.asarray(ball_vel_list, dtype=np.float64)
    time_arr = np.arange(X_arr.shape[0], dtype=np.float64) * dt

    # R08：终止原因优先于 mpc_done —— 安全拒绝/不可达不能被 done 掩盖。
    failure_reason = _resolve_failure_reason(
        run_metrics=run_metrics,
        ball_unreachable=ball_unreachable,
        stop_reason=str(run_metrics.get("stop_reason") or "") or None,
    )

    metrics: dict[str, Any] = {
        "total_steps": int(run_metrics.get("total_steps", 0)),
        "safe_steps": int(run_metrics.get("safe_steps", 0)),
        "mpc_done": bool(run_metrics.get("mpc_done", False)),
        "stop_reason": str(run_metrics.get("stop_reason") or "") or None,
        "unsafe": bool(run_metrics.get("unsafe", False)),
        "first_plan_ms": float(run_metrics.get("first_plan_ms", 0.0)),
        "wall_time_s": float(t_total),
        "mpc_time_s": float(t_mpc),
        "phase_timings": {
            "total_s": float(t_total),
            "mpc_s": float(t_mpc),
        },
        "backend": dict(backend),
        "failure_reason": failure_reason,
        "ablation_mode": ablation_mode,
        "is_position_mode": prep.is_position_mode,
        "n_mpc_steps": len(U_arr),
        "min_dist": min_dist,
        "max_qdot_ratio": float(sim_component.max_qdot_ratio),
        "max_tcp_speed": float(sim_component.max_tcp_speed),
        "max_racket_face_speed": float(sim_component.max_racket_face_speed),
        "ball_near_ms": float(ball_near_ms),
        "tube_ready_ms": float(tube_ready_ms),
        "hit_time_error_ms": float(hit_time_error),
        "hit_position_error": float(hit_position_error),
        "pos_error": float(pos_error),
        "pos_error_plan": float(pos_error_plan),
        "vel_error": float(vel_error),
        "v_racket_at_hit": float(v_racket_at_hit_val),
        "k_hit": int(k_hit_total),
        "p_hit": p_hit.tolist(),
        "v_hit_at_contact": v_hit_at_contact.tolist(),
        # 固定击打方向的实际生效值（单位向量）与来源；非 fixed 时 direction
        # 为来球反方向、来源为 "reflect"。只新增键，不改必需字段。
        "hit_direction": np.asarray(d_follow, dtype=np.float64).tolist(),
        "hit_direction_source": hit_direction_source,
        "v_ball_hit": v_ball_hit.tolist(),
        "n_des": n_des_single.tolist(),
        "p_ee_final": p_ee_final.tolist(),
        "p_racket_init": p_racket_init.tolist(),
        "post_hit_steps": int(post_hit_steps),
        "post_hit_skipped_reason": post_hit_skipped_reason,
        "runtime_overrides": dict(config.runtime_overrides),
        "ball_speed_perturb_pct": ball_speed_perturb_value,
        "always_collision": always_collision_value,
        "unreachable": False,
        # 球自由关节地址来自 bundle（R1D：qpos0-6/qvel0-5；RM65：12），
        # 供产物/报告校验收录是否按名映射而非截断为 RM65 布局。
        "ball_qpos_start": int(bundle._ball_qpos_start),
        "ball_qvel_start": int(bundle._ball_qvel_start),
        "shoulder_pos": np.asarray(shoulder_pos, dtype=np.float64).tolist(),
        "shoulder_pos_source": shoulder_pos_source,
        "proximity": {
            "ball_radius": float(ball_radius),
            "racket_radius": float(racket_radius),
            "proximity_margin": float(proximity_margin),
            "ball_near_threshold": float(
                ball_radius + racket_radius + proximity_margin
            ),
        },
        "histories": {
            "distances": [float(x) for x in sim_component.distances_history],
            "normal_align": [float(x) for x in sim_component.normal_align_history],
            "ball_near": [bool(x) for x in sim_component.ball_near_history],
            "tube_ready": [bool(x) for x in sim_component.tube_ready_history],
            "pos_error": [float(x) for x in sim_component.pos_error_history],
        },
        "hit_window": _serialize_hit_window(hit_window),
        "solver_config": dict(config.solver_config),
        "serve": {
            "mode": str(prep.serve_cfg.get("mode")),
            "ball_speed": prep.serve_cfg.get("ball_speed"),
            "p0": p0.tolist(),
            "v0": v0.tolist(),
        },
        "convergence": dict(run_metrics.get("convergence", {}) or {}),
    }

    # T6（R8 批准的最小集成）：把 SimComponent 在真实物理步采集的球—拍
    # 接触事件与新旧命中口径放入 result.metrics，供产物 writer 使用。
    # 只新增键，不改管线顺序/数值/筛选；旧 metric 键逐字不变。
    metrics.update(sim_component.contact_summary())

    if robot_id != "rm65":
        side = str(bundle.joint_names[0]).split("_", 1)[0]
        artifact = _r1d_artifact_metadata(
            config,
            bundle,
            side=side,
            joint_names=tuple(bundle.joint_names),
            ball_initial=(p0_real, v0_real),
            backend=backend,
        )
        artifact.update(
            episode={
                "max_steps": int(total_steps),
                "total_steps": int(run_metrics.get("total_steps", 0)),
                "safe_steps": int(run_metrics.get("safe_steps", 0)),
                "mpc_done": bool(run_metrics.get("mpc_done", False)),
                "first_plan_ms": float(run_metrics.get("first_plan_ms", 0.0)),
                "wall_time_s": float(t_total),
                "failure_reason": failure_reason,
                "stop_reason": str(run_metrics.get("stop_reason") or "") or None,
                "unsafe": bool(run_metrics.get("unsafe", False)),
            },
            hit={
                "type": hit_type_en,
                "step": int(hit_step),
                "time_s": float(hit_step * dt) if hit_step >= 0 else None,
                "racket_face_speed": (
                    float(sim_component.max_racket_face_speed)
                    if hit_step >= 0 else None
                ),
                "p_ee_at_hit": (
                    None if p_ee_at_hit is None
                    else np.asarray(p_ee_at_hit, dtype=np.float64).tolist()
                ),
                "ball_pos_at_hit": (
                    None if ball_pos_at_hit is None
                    else np.asarray(ball_pos_at_hit, dtype=np.float64).tolist()
                ),
                "ball_out_speed": (
                    None if sim_component.ball_out_speed is None
                    else float(sim_component.ball_out_speed)
                ),
                "ball_out_vy": (
                    None if sim_component.ball_out_vy is None
                    else float(sim_component.ball_out_vy)
                ),
                "message": classify_hit_result(
                    active_contact,
                    passive_contact,
                    pos_error,
                    robot_name="R1-A7-D",
                ),
            },
            metrics={
                "n_mpc_steps": len(U_arr),
                "min_dist": None if not np.isfinite(min_dist) else float(min_dist),
                "max_qdot_ratio": float(sim_component.max_qdot_ratio),
                "max_tcp_speed": float(sim_component.max_tcp_speed),
                "max_racket_face_speed": float(
                    sim_component.max_racket_face_speed
                ),
                "active_contact": active_contact,
                "passive_contact": passive_contact,
            },
        )
        metrics["r1d_artifact"] = artifact

    # ── 12. 轨迹落盘（旧 L1137-1150；仅 dump 路径，eph 产物仍由脚本/T6 写）
    if recorder is not None and dump_trajectory is not None:
        # 注意：这里只追加 ``post_hit_steps`` 的实际值，不追加
        # ``post_hit_skipped_reason`` —— 后者已进 result.metrics，而
        # recorder 的 metadata 是 T0 逐位基线的组成部分，正常路径必须
        # 与其逐字节一致（R01 只截断异常路径，正常路径行为不变）。
        recorder._metadata.update({
            "hit_type": hit_type_en,
            "pos_error": float(pos_error),
            "hit_step": int(hit_step),
            "post_hit_steps": int(post_hit_steps),
        })
        recorder.save(Path(dump_trajectory), hit_step=int(hit_step))

    return V12EpisodeResult(
        config=config,
        robot_id=robot_id,
        mode=str(config.actuator_mode),
        joint_names=tuple(str(name) for name in bundle.joint_names),
        dt=dt,
        racket_id=(None if bundle.racket is None else str(bundle.racket.racket_id)),
        X=X_arr,
        U=U_arr,
        ball_pos=ball_pos_arr,
        ball_vel=ball_vel_arr,
        time=time_arr,
        racket_pos=racket_pos_arr,
        hit_type=hit_type_en,
        hit_step=hit_step,
        active_contact=active_contact,
        passive_contact=passive_contact,
        v_ee_at_hit=(
            None if sim_component.v_ee_at_hit is None
            else float(sim_component.v_ee_at_hit)
        ),
        p_ee_at_hit=(
            None if p_ee_at_hit is None
            else np.asarray(p_ee_at_hit, dtype=np.float64).copy()
        ),
        ball_pos_at_hit=(
            None if ball_pos_at_hit is None
            else np.asarray(ball_pos_at_hit, dtype=np.float64).copy()
        ),
        ball_out_speed=(
            None if sim_component.ball_out_speed is None
            else float(sim_component.ball_out_speed)
        ),
        ball_out_vy=(
            None if sim_component.ball_out_vy is None
            else float(sim_component.ball_out_vy)
        ),
        metrics=metrics,
        model_dependency_manifest=_model_dependency_manifest(bundle),
    )
