"""V12 机器人无关共享闭环管线（Task 5）。

从旧 ``scripts/rm65_mpc_v12.py`` 主流程（L498-1135）提取为机器人无关的
共享核心：初始化 → 原发球 → 候选点/Tube → MPC/组件/hook → 执行 →
击打后 20 步 → 指标汇总。RM65 与 R1D 的唯一差异来源是
:func:`src.sim.v12_robot_adapter.build_v12_robot` 返回的
:class:`~src.sim.v12_robot_adapter.V12RobotBundle` 以及
:func:`src.sim.v12_config.resolve_v12_config` 解析出的配置。

设计约束（brief §4.2 / §7.6）：

* 不直接 import :class:`~src.sim.rm65_env.RM65Env` 或
  ``build_r1d_environments`` —— 机器人事实只经 ``bundle`` 获取；
* 不 import ``scripts/``、不读 ``sys.argv``、不打开 viewer/matplotlib；
* 不写 episode 的 JSON/NPZ 产物（唯一例外：``dump_trajectory`` 指定的
  轨迹 npz，与旧 ``--dump-trajectory`` 语义一致）；
* 返回 :class:`V12EpisodeResult` 只含可 JSON 序列化数据（数组为副本），
  不引用 env / mpc / component。

R1D 旧固定任务（``r1d_fixed_ball_compat``）的兼容入口仍保留在
``src/sim/r1d_runner.py::run_r1d_episode``（T6 迁移点）；本模块服务
RM65 与 R1D overlay 两种模式。
"""

from __future__ import annotations

import copy
import hashlib
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Callable

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
from src.ilqt.tube_builder import build_hitting_tube, search_hit_window
from src.ilqt.tube_types import HitWindow, HittingTube, TubeConfig
from src.robot.constants import INIT_Q_LEFT
from src.robot.court_geometry import court_profile_metadata
from src.sim.collision_checks import ForbiddenContactChecker
from src.sim.episode_runtime import (
    UNSET,
    UnsetType,
    apply_runtime_limits,
    apply_runtime_overrides,
    real_config_snapshot,
    resolve_runtime_options,
    resolve_runtime_overrides,
)
from src.sim.hit_detection import classify_hit_result
from src.sim.v12_robot_adapter import V12RobotBundle, build_v12_robot
from src.sim.v12_config import ResolvedV12Config
from src.tennis.ball import (
    generate_ball_from_serve_box,
    generate_ball_to_target_box,
)
from src.tennis.hitting import find_hitting_point_physics

__all__ = ["V12EpisodeResult", "run_v12_episode"]

logger = logging.getLogger(__name__)

# 击打后继续仿真步数（旧脚本 L1003，V11 语义）
_POST_HIT_STEPS = 20

# 允许执行击打后保持段的终止原因（R01）：只有 MPC 正常完成或达到步数上限
# 才继续推进仿真；安全拒绝/球状态不可用时立即截断，不再执行未滤波控制。
_POST_HIT_ALLOWED_STOP_REASONS = frozenset({"mpc_done", "max_steps"})

# 固定击打方向的兜底值（对应 ``configs/default.yaml`` 的
# ``hitting.hit_direction`` 缺省 [0, -1, 0.3]）。仅在 ``--fixed-direction``
# 且 runtime 与 ``hitting`` 配置都未给出方向时使用；完整优先级见
# :func:`_resolve_hit_direction`。
_DEFAULT_HIT_DIRECTION = (0.0, -1.0, 0.3)


# ── 机器人事实补充说明 ────────────────────────────────────────────────────
# T4 适配器 ``RM65SimulationEnv``（含 ``planning_env_factory``）已与 T0 基线
# 数值逐位等价（T5 初审 shim 消融 + 复审独立 5 模式消融均证实：RM65
# torque/position 对 T0 NPZ 全部键 max_abs_diff=0.0）。T5 不再保留任何
# “RM65 T0 数值兼容层”，RM65 与 R1D 统一使用 ``bundle.planning_env_factory``。


@dataclass(frozen=True)
class V12EpisodeResult:
    """V12 共享闭环管线的完整结果（只含可 JSON 序列化数据）。

    Attributes:
        config: 本次运行的解析配置（唯一配置来源）。
        robot_id: 机器人标识 ``"rm65"`` / ``"unitree_r1d"``。
        mode: 执行器模式 ``"torque"`` / ``"position"``。
        joint_names: 主动关节名（长度 n）。
        dt: 仿真步长 [s]。
        racket_id: 球拍标识（RM65 为 None）。
        X: 主动臂状态序列 ``(N+1, 2n)``（``[q, qdot]``）。
        U: 控制序列 ``(N, n)``。
        ball_pos: 球位置序列 ``(N+1, 3)``。
        ball_vel: 球速度序列 ``(N+1, 3)``。
        time: 时间序列 ``(N+1,)``。
        racket_pos: 拍心位置序列 ``(N+1, 3)``（可视化用）。
        hit_type: ``"active"`` / ``"passive"`` / ``"miss"``。
        hit_step: 发生接触的执行步（未命中为 -1）。
        active_contact: 是否主动击球（拍面接触速度 > 阈值）。
        passive_contact: 是否被动接触。
        v_ee_at_hit: 接触时末端速度 [m/s]（未命中 None）。
        p_ee_at_hit: 接触时末端位置 ``(3,)``（未命中 None）。
        ball_pos_at_hit: 接触时球位置 ``(3,)``（未命中 None）。
        ball_out_speed: 击球出球速度 [m/s]（未命中 None）。
        ball_out_vy: 击球出球 y 速度 [m/s]（未命中 None）。
        metrics: 指标字典（耗时/后端/失败原因/几何/直方图等，JSON 可序列化）。
        model_dependency_manifest: 模型/资产依赖清单（I1 接线；默认空列表，
            保持旧构造点与旧 artifact 兼容）。
    """

    config: ResolvedV12Config
    robot_id: str
    mode: str
    joint_names: tuple[str, ...]
    dt: float
    racket_id: str | None
    X: np.ndarray
    U: np.ndarray
    ball_pos: np.ndarray
    ball_vel: np.ndarray
    time: np.ndarray
    racket_pos: np.ndarray
    hit_type: str
    hit_step: int
    active_contact: bool
    passive_contact: bool
    v_ee_at_hit: float | None
    p_ee_at_hit: np.ndarray | None
    ball_pos_at_hit: np.ndarray | None
    ball_out_speed: float | None
    ball_out_vy: float | None
    metrics: dict[str, Any] = field(default_factory=dict)
    model_dependency_manifest: list[dict[str, Any]] = field(default_factory=list)


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


def _set_rm65_left_arm(execution: Any, left_q: np.ndarray) -> None:
    """把 RM65 左臂（不被驱动的 6 关节）设到待机角。

    旧脚本在每次 ``env.reset(init_q)`` 后手动写 ``qpos[NQ:NQ+LEFT_ARM_NQ]``
    与同段 qvel=0（L674-675、L831-832）。RM65Env.reset 不设置左臂，故共享
    管线必须在重置后补这一步以保持 T0 逐位一致；R1D 场景对侧臂已烘成静态
    几何、不存在左臂关节，因此不调用本函数。
    """
    nq = int(execution.NQ)
    count = int(left_q.shape[0])
    execution.data.qpos[nq:nq + count] = left_q
    execution.data.qvel[nq:nq + count] = 0.0


def _resolve_serve_box_center(
    robot_id: str,
    serve_config: dict[str, Any],
    target_center: np.ndarray,
) -> tuple[float, float, float]:
    """解析发球区中心的世界坐标（机器人朝向感知）。

    RM65 直接使用 V12 常量（``serve.serve_box_center``，T0 逐位一致）：
    球从 -Y 侧沿 +Y 飞来。

    R1D 的模型/任务世界坐标与朝向不同：T4 事实为 ``base_quaternion=[1,0,0,0]``
    （面向 +X），旧 ``configs/r1d_sim_right.yaml`` 的来球也来自 +X 侧
    （p0.x≈0.3、v0.x≈-11.5，沿 -X 飞向 target.x≈-11.75）。因此 R1D 发球区
    必须以目标中心为参照放在 +X 一侧：
    ``(target_x + serve_distance, target_y, serve_height)``，使球沿 -X 飞向
    目标（与拍面法向 +X 相对），而不是沿 -Y 侧向掠过。该平移只改发球采样框
    的位置，不改 RNG 调用顺序/球速/弹跳参数。
    """
    if robot_id == "rm65":
        return tuple(
            float(x) for x in serve_config["serve_box_center"]
        )
    serve_distance = float(serve_config.get("serve_distance", 8.0))
    serve_height = float(serve_config.get("serve_height", 1.2))
    target = np.asarray(target_center, dtype=np.float64)
    return (
        float(target[0] + serve_distance),
        float(target[1]),
        float(serve_height),
    )


def _resolve_serve_box_halfsize(
    robot_id: str,
    serve_config: dict[str, Any],
) -> tuple[float, float, float]:
    """解析发球区半尺寸（随来球方向旋转）。

    RM65 的来球沿 +Y，全尺寸 8m 长轴在 X 方向，故半尺寸为
    ``(4.0, 0.1, 0.15)``（T0 常量，逐位不变）。
    R1D 的来球沿 -X，8m 长轴应旋转到 Y 方向，故半尺寸为
    ``(0.1, 4.0, 0.15)``。这里只做 X/Y 互换，不改共享配置值本身。
    """
    halfsize = tuple(
        float(x) for x in serve_config["serve_box_halfsize"]
    )
    if robot_id == "rm65":
        return (halfsize[0], halfsize[1], halfsize[2])
    return (halfsize[1], halfsize[0], halfsize[2])


def _resolve_shoulder_pos(
    config: ResolvedV12Config,
    bundle: V12RobotBundle,
) -> np.ndarray:
    """解析规划/代价/发球用的 shoulder_pos（必须是模型/配置事实）。

    RM65 使用 V12 常量 ``SHOULDER_POS``（配置 ``scene.shoulder_pos``，T0 逐位
    一致）；R1D 场景未给该值，按 T5 计划从执行模型按名取肩部 body
    ``{side}_shoulder_pitch_Link`` 的世界坐标（模型事实，不是 target_center
    的代称，也不复用 RM65 常量）。
    """
    shoulder = config.scene_config.get("shoulder_pos")
    if shoulder is not None:
        return np.asarray(shoulder, dtype=np.float64).copy()
    return _model_shoulder_pos(bundle)


def _shoulder_body_name(bundle: V12RobotBundle) -> str:
    """R1D 肩部 body 名（由主动关节名侧别派生，如 right_shoulder_pitch_Link）。"""
    joint_names = tuple(str(name) for name in bundle.joint_names)
    side = joint_names[0].split("_", 1)[0] if joint_names else "right"
    return f"{side}_shoulder_pitch_Link"


def _model_shoulder_pos(bundle: V12RobotBundle) -> np.ndarray:
    """从执行模型读取肩部 body 的 world xpos（R1D 分支）。"""
    import mujoco  # noqa: PLC0415 - 仅 R1D 肩点派生使用

    execution = bundle.execution
    model = execution.model
    name = _shoulder_body_name(bundle)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:  # pragma: no cover - 模型契约破坏才会发生
        raise ValueError(
            f"R1D 模型缺少肩部 body {name!r}，无法派生 shoulder_pos"
        )
    mujoco.mj_kinematics(model, execution.data)
    return np.asarray(execution.data.xpos[body_id], dtype=np.float64).copy()


def _mpc_workspace_center_pos(
    config: ResolvedV12Config,
    shoulder_pos: np.ndarray,
) -> np.ndarray:
    """``MPCConfig.shoulder_pos`` 槽位：RM65 用 T0 常量，R1D 用工作空间中心。

    T4 ``MPCController.__init__`` 对绑定环境调用
    ``validate_workspace_against_model(env, config.shoulder_pos, radius)``，
    要求该点 IK 可达且在关节限位内。R1D 的模型肩点
    ``{side}_shoulder_pitch_Link`` 是 TCP 不可达点（实测 IK 残差 0.143 m
    超过 0.05 m 容差），直接注入会让 R1D 在构造控制器前 ValueError；因此
    该槽位保留 T4/legacy 语义（= workspace center = ``config.target_center``），
    模型肩点只用于发球生成/击球点候选搜索，并写入
    ``metrics["shoulder_pos"]`` / ``metrics["shoulder_pos_source"]``。
    若 T4 后续把“校验中心”拆成独立字段，只需改本函数。
    """
    if config.scene_config.get("shoulder_pos") is not None:
        return np.asarray(shoulder_pos, dtype=np.float64).copy()
    return np.asarray(config.target_center, dtype=np.float64).copy()


def _build_tube_config(cost_config: dict[str, Any]) -> TubeConfig:
    """从解析后的 cost 配置构建 ``TubeConfig``（缺省字段用 TubeConfig 默认）。"""
    tube_raw = cost_config.get("tube") or {}
    allowed = {f.name for f in fields(TubeConfig)}
    kwargs = {
        str(key): value
        for key, value in dict(tube_raw).items()
        if str(key) in allowed
    }
    return TubeConfig(**kwargs)


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


def _predict_effective_backend(planning: object, requested: str) -> str:
    """按 ``build_solver`` 的选择规则推断 effective 后端。

    仅用于 MPC 尚未构造的早退路径（球不可达，solver 实例不存在）；正常
    回合以 :attr:`MPCController.effective_backend` 为唯一事实来源。

    Args:
        planning: 规划环境（用于 C++ 内核兼容性判定）。
        requested: 请求的 solver 后端（``auto`` / ``python`` / ``cpp``）。

    Returns:
        ``"cpp"`` / ``"python-analytic"`` / ``"python-fd"``。
    """
    from src.ilqt.solver import is_cpp_kernel_compatible

    compatible = bool(is_cpp_kernel_compatible(planning))
    if requested == "python":
        # build_solver：兼容环境沿用解析线性化，否则强制有限差分。
        return "python-analytic" if compatible else "python-fd"
    if requested == "cpp":
        # build_solver 对显式 cpp 只接受兼容环境且扩展缺失会直接报错；
        # 早退路径没有实例可读，按请求值记录（不存在回退成 Python 的路径）。
        return "cpp"
    if not compatible:
        # auto + 非兼容环境：build_solver 直接给 Python FD。
        return "python-fd"
    try:
        from src.cpp import solver_cpp
    except ImportError:  # pragma: no cover - 入口缺失等价于扩展不可用
        return "python-analytic"
    if bool(getattr(solver_cpp, "_CPP_AVAILABLE", False)):
        return "cpp"
    # auto + 兼容环境 + 扩展缺失：build_solver 回退 Python 解析线性化。
    return "python-analytic"


def _effective_backend(
    planning: object,
    bundle: V12RobotBundle,
    mpc: MPCController | None = None,
) -> dict[str, str]:
    """描述求解后端事实（requested / effective）。

    ``requested`` 始终是请求值（``bundle.backend``，可能为 ``"auto"``）；
    ``effective`` 优先读取构造出的 :class:`MPCController` 实例
    （:attr:`MPCController.effective_backend`），MPC 尚未构造的早退路径
    才按 ``build_solver`` 规则推断——不能只看 requested 或环境兼容性，
    否则已启用的 C++ 内核会被误记为 Python。

    Args:
        planning: 规划环境。
        bundle: 机器人 bundle（提供 requested 后端）。
        mpc: 已构造的 MPC 控制器；``None`` 表示早退路径。

    Returns:
        ``{"requested": ..., "effective": ...}``。
    """
    requested = str(bundle.backend or "auto")
    if mpc is not None:
        return {"requested": requested, "effective": mpc.effective_backend}
    return {
        "requested": requested,
        "effective": _predict_effective_backend(planning, requested),
    }


def _normalize_hit_direction(raw: object, *, source: str) -> np.ndarray:
    """校验并归一化方向向量（长度 3、有限、非零）。

    Args:
        raw: 原始方向配置（序列或长度为 3 的一维数组）。
        source: 来源标签，仅用于错误信息定位配置层。

    Returns:
        单位方向向量 ``(3,)``。

    Raises:
        ValueError: 不是长度为 3 的有限数值序列，或为零向量。
    """
    if isinstance(raw, (str, bytes)) or not isinstance(
        raw, (list, tuple, np.ndarray)
    ):
        raise ValueError(
            f"击打方向配置（来源 {source}）必须是长度 3 的数值序列，"
            f"实际为 {type(raw).__name__}：{raw!r}"
        )
    try:
        vector = np.asarray(raw, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"击打方向配置（来源 {source}）无法转换为数值向量：{raw!r}"
        ) from exc
    if vector.size != 3:
        raise ValueError(
            f"击打方向配置（来源 {source}）必须恰好 3 维，"
            f"实际 {vector.size} 维：{raw!r}"
        )
    if not bool(np.all(np.isfinite(vector))):
        raise ValueError(
            f"击打方向配置（来源 {source}）含非有限值（NaN/Inf）：{raw!r}"
        )
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError(
            f"击打方向配置（来源 {source}）为零向量，无法确定击球方向：{raw!r}"
        )
    return vector / norm


def _resolve_hit_direction(
    runtime_hit_direction: object,
    hitting_config: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    """解析固定击打方向，返回单位向量与来源标签。

    优先级（高 → 低）：

    1. 显式 runtime 覆盖（``run_v12_episode(hit_direction=...)``）或已保存
       runtime 的 ``hit_direction``；
    2. ``config.hitting_config["hit_direction"]``（YAML 或
       ``--task-overrides hitting.hit_direction``）；
    3. 内置 :data:`_DEFAULT_HIT_DIRECTION`。

    只有 ``--fixed-direction`` 分支使用本结果；非 fixed 仍用来球反方向，
    不会误用配置里的固定方向。

    Args:
        runtime_hit_direction: ``runtime_overrides["hit_direction"]``
            （``None`` 表示未显式指定）。
        hitting_config: 合并后的 ``hitting`` 配置节。

    Returns:
        ``(单位方向向量, 来源标签)``；来源标签为 ``"runtime"`` /
        ``"config:hitting.hit_direction"`` / ``"default"``。

    Raises:
        ValueError: 选中的向量非法（长度非 3 / 非有限 / 零向量）。
    """
    if runtime_hit_direction is not None:
        raw: object = runtime_hit_direction
        source = "runtime"
    elif hitting_config.get("hit_direction") is not None:
        raw = hitting_config["hit_direction"]
        source = "config:hitting.hit_direction"
    else:
        raw = _DEFAULT_HIT_DIRECTION
        source = "default"
    return _normalize_hit_direction(raw, source=source), source


def _select_follow_direction(
    *,
    fixed_direction: bool,
    configured_direction: np.ndarray,
    configured_source: str,
    v_ball_hit: np.ndarray,
) -> tuple[np.ndarray, str]:
    """选择随挥/击球方向及实际来源标签。

    ``--fixed-direction`` 开启时使用配置方向（已归一化）；否则一律用击球点
    来球反方向，**不消费**配置的固定方向（避免非 fixed 模式误用配置）。

    Args:
        fixed_direction: 是否 ``--fixed-direction``。
        configured_direction: 已归一化的配置方向 ``(3,)``。
        configured_source: 配置方向来源标签。
        v_ball_hit: 击球点来球速度 ``(3,)``。

    Returns:
        ``(单位方向向量, 来源标签)``；非 fixed 的来源标签为 ``"reflect"``。
    """
    if fixed_direction:
        return configured_direction, configured_source
    return -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8), "reflect"


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
    """把刚体名元组解析成 MuJoCo ID 列表。"""
    import mujoco

    return [
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name))
        for name in names
    ]


def _racket_radius_from_bundle(bundle: V12RobotBundle) -> float:
    """读取球拍等效半径。

    R1D 由 ``bundle.racket.face_semi_axes`` 的宽度半轴提供（Blade100：
    ``(0.004, 0.123, 0.1669)`` → 0.123 m）；RM65 无球拍档案，保持 V12
    旧值 0.12 走 ``SimComponent`` 缺省。
    """
    racket = bundle.racket
    if racket is None:
        return float(DEFAULT_RACKET_RADIUS)
    axes = tuple(float(x) for x in racket.face_semi_axes)
    # face_semi_axes = (法向半厚, 半宽, 半长)：接近门控用横向半宽
    return float(axes[1])


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
# 主管线
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


def run_v12_episode(
    config: ResolvedV12Config,
    *,
    dump_trajectory: Path | None = None,
    first_plan_flight: bool | UnsetType = UNSET,
    always_collision: bool | UnsetType = UNSET,
    ball_speed_perturb_pct: float | UnsetType = UNSET,
    fixed_direction: bool | UnsetType = UNSET,
    hit_direction: tuple[float, float, float] | None | UnsetType = UNSET,
    post_exec_hooks: tuple[Callable[[object], None], ...] = (),
    limits_config: Path | None | UnsetType = UNSET,
    max_tcp: float | None | UnsetType = UNSET,
    terminal_exempt_steps: int | None | UnsetType = UNSET,
    dq_max_fraction: float | None | UnsetType = UNSET,
) -> V12EpisodeResult:
    """运行一次 V12 共享闭环回合。

    流程与旧 ``scripts/rm65_mpc_v12.py`` 主流程逐段对应：初始化（bundle）→
    原发球（同 seed 同 RNG 顺序）→ 击打点/IK 可达过滤 → 候选窗口/Tube →
    MPC 与组件组装 → EpisodeRunner 执行 → 击打后 20 步保持 → 指标汇总。

    Args:
        config: 解析后的唯一配置来源；``config.runtime_overrides`` 里的已保存
            覆盖会在未显式传参时被继承（F01）。
        dump_trajectory: 轨迹 npz 落盘路径（None 不记录/不落盘）。
        first_plan_flight: 首个规划期间是否让球飞行
            （旧 ``--no-freeze-first-plan``）；``UNSET`` 继承已保存值/默认。
        always_collision: 击球前球-拍碰撞常开（批量实验用）；``UNSET`` 继承。
        ball_speed_perturb_pct: 球速耦合扰动百分比（旧 CLI 同名参数）；
            ``UNSET`` 继承。
        fixed_direction: 随挥方向是否用配置固定方向（旧 ``--fixed-direction``）；
            ``UNSET`` 继承。
        hit_direction: 固定击打方向覆盖（runtime ``hit_direction``）。优先级为
            本参数 > ``config.hitting_config["hit_direction"]`` > 内置默认
            （见 :func:`_resolve_hit_direction`）；``None`` 表示本参数不覆盖
            （回落到配置/默认），非法向量报错而不是静默回退。
        post_exec_hooks: 额外 post_exec hook（脚本层计时/诊断用）。
        limits_config: RM65 真机限位配置（旧 ``--limits-config``）；``None``
            清除已保存文件层，``UNSET`` 用已保存内容恢复（不要求原文件存在）。
        max_tcp: TCP 速度上限覆盖（0 表示不限）；``None`` 清除已保存覆盖。
        terminal_exempt_steps: 终段豁免步数覆盖；``None`` 清除已保存覆盖。
        dq_max_fraction: 关节位置增量上限分数覆盖；``None`` 清除已保存覆盖。

    Note:
        覆盖优先级为「显式值 > ``config.runtime_overrides`` 已保存值 >
        当前默认」；显式 ``False`` / ``0`` 是有效覆盖，不会被当成缺省。

    Returns:
        :class:`V12EpisodeResult`（数组均为副本，不引用仿真对象）。

    Raises:
        TypeError: 传入的配置不是 :class:`ResolvedV12Config`。
        ValueError: 机器人/配置组合不受支持（由 ``build_v12_robot`` 抛出），
            或固定击打方向非法（长度非 3 / 非有限 / 零向量）。
    """
    if not isinstance(config, ResolvedV12Config):
        raise TypeError(
            "run_v12_episode 只接受 ResolvedV12Config（请用 "
            "resolve_v12_config 解析），收到 "
            f"{type(config).__name__}"
        )

    # ── 1. 运行时覆盖 + 机器人事实（唯一来源）──────────────────────────
    explicit = _explicit_runtime_options(
        first_plan_flight=first_plan_flight,
        always_collision=always_collision,
        ball_speed_perturb_pct=ball_speed_perturb_pct,
        fixed_direction=fixed_direction,
        hit_direction=hit_direction,
        limits_config=limits_config,
        max_tcp=max_tcp,
        terminal_exempt_steps=terminal_exempt_steps,
        dq_max_fraction=dq_max_fraction,
    )
    # F01：显式值 > 已保存覆盖 > 当前默认。必先解析再 build：已保存的
    # limits 内容快照随后由 apply_runtime_limits 重建，不依赖原文件路径。
    config = resolve_runtime_options(config, explicit)
    bundle = build_v12_robot(config)
    execution = bundle.execution
    planning = bundle.planning
    dt = float(config.dt)
    robot_id = str(bundle.robot_id)
    is_position_mode = str(config.actuator_mode) == "position"
    # F01：把覆盖真正落到 bundle（robot_limits/init_q），并记录实际生效值；
    # 之后 result/metrics/artifact 与快照全部使用该返回值。
    config = apply_runtime_limits(config, bundle)
    runtime = dict(config.runtime_overrides)
    init_q = np.asarray(bundle.init_q, dtype=np.float64).copy()
    limits = bundle.robot_limits
    backend = _effective_backend(planning, bundle)

    # 实际消费值统一取自 apply 后的 runtime_overrides（结果/快照同源）
    first_plan_flight_value = bool(runtime["first_plan_flight"])
    always_collision_value = bool(runtime["always_collision"])
    ball_speed_perturb_value = float(runtime["ball_speed_perturb_pct"])
    fixed_direction_value = bool(runtime["fixed_direction"])
    hit_direction_value: list[float] | None = runtime.get("hit_direction")
    # 固定击打方向：runtime > hitting 配置节 > 内置默认（见 _resolve_hit_direction）。
    # 无论 --fixed-direction 是否开启都先校验：非法配置必须早失败，不能等到
    # fixed 分支才暴露，更不能静默回退到硬编码默认值。
    hit_direction_vec, hit_direction_config_source = _resolve_hit_direction(
        hit_direction_value,
        config.hitting_config,
    )

    scene_cfg = dict(config.scene_config)
    serve_cfg = dict(config.serve_config)
    mpc_cfg = dict(config.mpc_config)
    cost_cfg = dict(config.cost_config)
    total_horizon = int(mpc_cfg.get("total_horizon", 200))
    workspace_radius = float(config.workspace_radius)
    target_center = np.asarray(config.target_center, dtype=np.float64).copy()
    target_offset = float(config.target_offset)
    g = scene_cfg.get("ball_gravity")
    tube_cfg = _build_tube_config(cost_cfg)

    # ── 2. 原发球（RNG 顺序与机器人分支无关）───────────────────────────
    rng = np.random.default_rng(None if config.seed is None else int(config.seed))
    bundle.reset()
    if robot_id == "rm65":
        _set_rm65_left_arm(execution, np.asarray(INIT_Q_LEFT, dtype=np.float64))
    execution.update_kinematics()
    p_racket_init = np.asarray(execution.get_ee_pos(), dtype=np.float64).copy()
    shoulder_pos = _resolve_shoulder_pos(config, bundle)
    shoulder_pos_source = (
        "cli/scene.shoulder_pos"
        if config.scene_config.get("shoulder_pos") is not None
        else f"model:{_shoulder_body_name(bundle)}"
    )

    if str(serve_cfg.get("mode")) == "serve_box":
        p0, v0, _p_hit_expected = generate_ball_from_serve_box(
            serve_box_center=_resolve_serve_box_center(
                robot_id, serve_cfg, target_center
            ),
            serve_box_halfsize=_resolve_serve_box_halfsize(
                robot_id, serve_cfg
            ),
            target_center=target_center,
            target_offset=target_offset,
            shoulder_pos=shoulder_pos,
            workspace_radius=workspace_radius,
            g=g,
            ball_speed=serve_cfg.get("ball_speed"),
            speed_range=tuple(float(x) for x in serve_cfg["speed_range"]),
            use_bounce=bool(scene_cfg.get("use_bounce", True)),
            bounce_restitution=float(scene_cfg.get("bounce_restitution", 0.75)),
            rng=rng,
        )
    else:
        hit_time = total_horizon * dt * rng.uniform(0.3, 0.4)
        p0, v0, _p_hit_expected = generate_ball_to_target_box(
            target_center,
            target_offset,
            hit_time,
            g,
            shoulder_pos=shoulder_pos,
            workspace_radius=workspace_radius,
            ball_speed=serve_cfg.get("ball_speed"),
            ball_distance=serve_cfg.get("ball_distance"),
            approach_angle_deg=float(serve_cfg.get("approach_angle", 0.0) or 0.0),
            rng=rng,
            ball_direction=str(serve_cfg.get("ball_direction", "y")),
            ball_start_y_range=tuple(
                float(x) for x in serve_cfg["ball_start_y_range"]
            ),
            ball_start_z_range=tuple(
                float(x) for x in serve_cfg["ball_start_z_range"]
            ),
        )
    p0 = np.asarray(p0, dtype=np.float64).copy()
    v0 = np.asarray(v0, dtype=np.float64).copy()

    # ── 3. 击打点 + IK 可达性后过滤（旧 L716-758）───────────────────────
    hit_info = find_hitting_point_physics(
        execution, p0, v0, shoulder_pos, workspace_radius, total_horizon,
    )
    if hit_info is None:
        return _unreachable_result(
            config=config,
            bundle=bundle,
            execution=execution,
            backend=backend,
            p0=p0,
            v0=v0,
            rng=rng,
            target_center=target_center,
            workspace_radius=workspace_radius,
        )

    k_hit_total = int(hit_info["k_hit"])
    p_hit = np.asarray(hit_info["p_hit"], dtype=np.float64).copy()
    v_ball_hit = np.asarray(hit_info["v_ball_hit"], dtype=np.float64).copy()

    ball_positions_all, ball_velocities_all = execution.predict_ball_trajectory(
        p0, v0, total_horizon,
    )
    q_ik_init = np.asarray(
        execution.solve_ik(p_hit, q_init=init_q, max_iter=50, eps=1e-2),
        dtype=np.float64,
    )
    m_low_deg = (q_ik_init - limits.q_lower) * 180.0 / np.pi
    m_up_deg = (limits.q_upper - q_ik_init) * 180.0 / np.pi
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
            if not (dist_alt < workspace_radius and p_alt[2] > 0.3
                    and -0.60 < dz < 0.55):
                continue
            q_alt = np.asarray(
                execution.solve_ik(p_alt, q_init=init_q, max_iter=30, eps=2e-2),
                dtype=np.float64,
            )
            m_low_a = (q_alt - limits.q_lower) * 180.0 / np.pi
            m_up_a = (limits.q_upper - q_alt) * 180.0 / np.pi
            m_a = float(np.min(np.minimum(m_low_a, m_up_a)))
            if m_a > best_alt_margin:
                best_alt_margin = m_a
                best_alt_k = kk
        if best_alt_k != k_hit_total:
            p_hit = np.asarray(
                ball_positions_all[best_alt_k - 1], dtype=np.float64,
            ).copy()
            v_ball_hit = np.asarray(
                ball_velocities_all[best_alt_k - 1], dtype=np.float64,
            ).copy()
            k_hit_total = best_alt_k

    n_des_single = -v_ball_hit / (np.linalg.norm(v_ball_hit) + 1e-8)
    if bool(mpc_cfg.get("normal_flip", False)):
        n_des_single = -n_des_single

    near_threshold = (
        int(mpc_cfg["near_threshold"])
        if mpc_cfg.get("near_threshold")
        else max(50, k_hit_total // 3)
    )
    far_threshold = (
        int(mpc_cfg["far_threshold"])
        if mpc_cfg.get("far_threshold") is not None
        else 50
    )
    mpc_cfg["near_threshold"] = near_threshold
    mpc_cfg["far_threshold"] = far_threshold

    d_follow, hit_direction_source = _select_follow_direction(
        fixed_direction=fixed_direction_value,
        configured_direction=hit_direction_vec,
        configured_source=hit_direction_config_source,
        v_ball_hit=v_ball_hit,
    )
    target_speed = float(mpc_cfg.get("target_speed", 1.8) or 1.8)
    v_hit_at_contact = target_speed * d_follow

    if bool(mpc_cfg.get("no_follow_through", False)):
        follow_through_steps = 0
    else:
        follow_through_steps = int(mpc_cfg.get("follow_through_steps", 0) or 0)
    racket_speed = float(mpc_cfg.get("racket_speed", 1.8) or 1.8)

    # ── 4. 候选窗口 / 初始 Tube（旧 L788-807）──────────────────────────
    ablation_mode = str(mpc_cfg.get("ablation_mode", "full"))
    need_candidates = ablation_mode in (
        "full", "tube_only", "softmin_only", "select_one",
    )
    initial_hitting_tube: HittingTube | None = None
    hit_window: HitWindow | None = None
    if need_candidates:
        hit_window = search_hit_window(
            execution,
            p0,
            v0,
            shoulder_pos,
            workspace_radius,
            k_hit_total + 30,
            tube_cfg,
            ball_direction="y",
            current_step=0,
            robot_limits=limits,
            init_q=init_q,
        )
        if hit_window is not None:
            initial_hitting_tube = build_hitting_tube(
                hit_window, racket_speed, d_follow, tube_cfg,
            )

    # ── 5. 球速耦合扰动（旧 L809-825）──────────────────────────────────
    if abs(ball_speed_perturb_value) > 0.01:
        v0_norm = float(np.linalg.norm(v0))
        if v0_norm > 0.1:
            v0_dir = v0 / v0_norm
            offset_m = ball_speed_perturb_value / 100.0 * 2.0
            p0_real = p0 + v0_dir * offset_m
            v0_real = v0.copy()
        else:
            p0_real, v0_real = p0, v0
    else:
        p0_real, v0_real = p0, v0

    # ── 6. 重置环境 + 球初态（旧 L827-834）────────────────────────────
    bundle.reset()
    if robot_id == "rm65":
        _set_rm65_left_arm(execution, np.asarray(INIT_Q_LEFT, dtype=np.float64))
    execution.update_kinematics()
    execution.set_ball_state(p0_real, v0_real)
    x0 = np.zeros(execution.NX)
    x0[:execution.NQ] = init_q

    # ── 7. MPC 配置/控制器（旧 L839-917）──────────────────────────────
    mpc_config = _build_mpc_config(
        config,
        near_threshold=near_threshold,
        shoulder_pos=_mpc_workspace_center_pos(config, shoulder_pos),
        workspace_radius=workspace_radius,
        tube_cfg=tube_cfg,
        async_mode=bool(mpc_cfg.get("async_mode", False)),
    )
    # T4 适配器工厂对 RM65 已与 T0 逐位等价（见文件头说明）；R1D 亦复用
    # 同一 bundle 工厂，两个机器人不再有分支。
    planning_env_factory: Callable[[], object] | None = bundle.planning_env_factory
    mpc = MPCController(
        planning,
        mpc_config,
        robot_limits=limits,
        planning_env_factory=planning_env_factory,
    )
    # 求解器实际后端以构造出的实例为准（P2：旧实现只看环境兼容性，把已启用
    # 的 C++ 内核误记为 python-analytic）。早退路径仍用第 679 行的推断值。
    backend = _effective_backend(planning, bundle, mpc)

    # ── 8. 组件组装（旧 L919-982）─────────────────────────────────────
    # 球速度 history 由 SimComponent 与 ball_pos_history 同步维护（R07），
    # 不再用 post_exec hook 另记（旧实现会整体错位一帧）。
    obs_gate = _build_obs_gate(config, dt, rng)
    perception = SimPerception(execution, obs_gate=obs_gate)

    hard_x_body_ids: list[int] = []
    if robot_id == "rm65":
        # 旧 RM65 路径：默认滤波参数构造（与 T0 逐位一致）
        safety = PredictiveSafetyFilter(
            execution, limits, is_position_mode=is_position_mode,
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
            is_position_mode=is_position_mode,
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
    sim_component = SimComponent(
        execution,
        mpc,
        limits,
        init_q,
        is_position_mode,
        hard_x_body_ids,
        initial_hitting_tube,
        tube_cfg,
        dt,
        always_collision=always_collision_value,
        ball_geom_id=contact_ball_geom_id,
        racket_geom_ids=contact_racket_geom_ids,
        ball_radius=ball_radius,
        racket_radius=racket_radius,
        proximity_margin=proximity_margin,
    )
    sim_component.X_history = [x0.copy()]
    # R07：球位置/速度 history 首帧必须与环境实际初态（p0_real/v0_real）
    # 同源。旧实现用 p0（扰动前）会与 set_ball_state(p0_real, ...) 及产物
    # ball_initial 自相矛盾（ball_speed_perturb_pct≠0 时 p0_real ≠ p0）。
    _seed_ball_history(sim_component, ball_pos=p0_real, ball_vel=v0_real)

    recorder = None
    hooks: list[Callable[[object], None]] = []
    if dump_trajectory is not None:
        from src.real.trajectory_recorder import TrajectoryRecorder

        recorder = TrajectoryRecorder(
            env=execution,
            init_q=init_q,
            init_q_left=(
                np.asarray(INIT_Q_LEFT, dtype=np.float64)
                if robot_id == "rm65" else np.zeros(0, dtype=np.float64)
            ),
            dt=dt,
            metadata={
                "p0": p0_real.tolist(),
                "v0": v0_real.tolist(),
                "ball_speed": serve_cfg.get("ball_speed"),
                "is_position_mode": is_position_mode,
            },
            is_position_mode=is_position_mode,
        )
        hooks.append(recorder.make_hook())
    hooks.extend(post_exec_hooks)

    runner = EpisodeRunner(
        mpc=mpc,
        perception=perception,
        safety=safety,
        executor=sim_component,
        post_exec_hooks=list(hooks),
    )

    # ── 9. 运行回合（旧 L984-998）──────────────────────────────────────
    total_steps = total_horizon + follow_through_steps
    t_total_start = time.perf_counter()
    run_metrics = runner.run(
        max_steps=total_steps, first_plan_flight=first_plan_flight_value,
    )
    t_mpc_end = time.perf_counter()

    # ── 10. 击打后继续仿真（旧 L1000-1031）──────────────────────────────
    # R01：只有 MPC 正常完成/达到步数上限才执行击打后保持段。安全拒绝或
    # 球状态不可用时立即截断，绝不用 step_full 补任何未滤波制动。
    stop_reason = str(run_metrics.get("stop_reason") or "")
    post_hit_steps, post_hit_skipped_reason = _apply_post_hit_phase(
        run_metrics=run_metrics,
        execution=execution,
        sim_component=sim_component,
        is_position_mode=is_position_mode,
        dt=dt,
        total_horizon=total_horizon,
        follow_through_steps=follow_through_steps,
        recorder=recorder,
        ball_unreachable=bool(mpc.ball_unreachable),
    )
    x_current = np.asarray(execution.get_arm_state(), dtype=np.float64).copy()

    # ── 11. 评估（旧 L1033-1135）───────────────────────────────────────
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
        ball_unreachable=bool(mpc.ball_unreachable),
        stop_reason=stop_reason or None,
    )

    metrics: dict[str, Any] = {
        "total_steps": int(run_metrics.get("total_steps", 0)),
        "safe_steps": int(run_metrics.get("safe_steps", 0)),
        "mpc_done": bool(run_metrics.get("mpc_done", False)),
        "stop_reason": stop_reason or None,
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
        "is_position_mode": is_position_mode,
        "n_mpc_steps": int(len(U_arr)),
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
            "mode": str(serve_cfg.get("mode")),
            "ball_speed": serve_cfg.get("ball_speed"),
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
                "stop_reason": stop_reason or None,
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
                "n_mpc_steps": int(len(U_arr)),
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


# ---------------------------------------------------------------------------
# 辅助：限位覆盖 / 几何 ID / 不可达早退
# ---------------------------------------------------------------------------


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


def _seed_ball_history(
    sim_component: SimComponent,
    *,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
) -> None:
    """把球位置/速度 history 的首帧设为环境初态（R07）。

    调用方必须传入写入 ``execution.set_ball_state`` 的同一对初态
    （``p0_real``/``v0_real``，已含 ``ball_speed_perturb_pct`` 扰动），
    保证产物 ``ball_initial`` 与轨迹首帧同源。

    Args:
        sim_component: 轨迹 history 宿主。
        ball_pos: 环境实际初态位置（3,）。
        ball_vel: 环境实际初态速度（3,）。
    """
    sim_component.ball_pos_history = [
        np.asarray(ball_pos, dtype=np.float64).copy()
    ]
    sim_component.ball_vel_history = [
        np.asarray(ball_vel, dtype=np.float64).copy()
    ]


def _apply_post_hit_phase(
    *,
    run_metrics: Mapping[str, Any],
    execution: Any,
    sim_component: SimComponent,
    is_position_mode: bool,
    dt: float,
    total_horizon: int,
    follow_through_steps: int,
    recorder: Any,
    ball_unreachable: bool,
) -> tuple[int, str | None]:
    """按 ``stop_reason`` 门控击打后保持段（R01）。

    只有 ``mpc_done``/``max_steps`` 且非 ``ball_unreachable`` 才执行未滤波的
    保持控制；``unsafe``/``no_ball``/不可达（及未来任何非正常终止）直接截断，
    返回 0 步与跳过原因。不可达时即使 ``mpc_done`` 为 True 也不得继续甩动。

    Args:
        run_metrics: ``EpisodeRunner.run`` 返回的指标（读 ``stop_reason``）。
        execution: 执行环境。
        sim_component: 轨迹 history 宿主。
        is_position_mode: 位置模式/力矩模式。
        dt: 控制周期（秒）。
        total_horizon: 主评估段步数。
        follow_through_steps: 随挥步数。
        recorder: 轨迹记录器或 ``None``。
        ball_unreachable: 规划侧是否判定来球不可达（``mpc.ball_unreachable``）。

    Returns:
        ``(post_hit_steps, post_hit_skipped_reason)``；执行时为
        ``(_POST_HIT_STEPS, None)``，截断时为 ``(0, 跳过原因)``。
    """
    stop_reason = str(run_metrics.get("stop_reason") or "")
    if ball_unreachable:
        logger.warning("击打后保持段跳过（ball_unreachable=True）：不执行未滤波控制")
        return 0, "ball_unreachable"
    if stop_reason not in _POST_HIT_ALLOWED_STOP_REASONS:
        skipped_reason = stop_reason or "unknown"
        logger.warning(
            "击打后保持段跳过（stop_reason=%s）：不执行未滤波控制",
            skipped_reason,
        )
        return 0, skipped_reason
    steps = _run_post_hit_hold(
        execution=execution,
        sim_component=sim_component,
        is_position_mode=is_position_mode,
        dt=dt,
        total_horizon=total_horizon,
        follow_through_steps=follow_through_steps,
        recorder=recorder,
    )
    return steps, None


def _run_post_hit_hold(
    *,
    execution: Any,
    sim_component: SimComponent,
    is_position_mode: bool,
    dt: float,
    total_horizon: int,
    follow_through_steps: int,
    recorder: Any,
) -> int:
    """击打后保持段：PD 保持当前臂形并推进 ``_POST_HIT_STEPS`` 步（R01）。

    调用方必须先按 ``stop_reason`` 门控：只有 MPC 正常完成/达到步数上限
    才允许进入本函数；安全拒绝与球状态不可用时不得用 ``step_full``
    补未滤波制动。本函数内的控制量与 append 顺序与旧内联循环逐字一致。

    Args:
        execution: 执行环境（需 ``NQ``/``NU``/``step_full``/``get_arm_state``）。
        sim_component: 轨迹 history 宿主（X/U/ball_pos/ball_vel）。
        is_position_mode: 位置模式用关节角保持，否则用力矩 PD。
        dt: 控制周期（秒）。
        total_horizon: 主评估段步数（用于 recorder 时间戳）。
        follow_through_steps: 随挥步数（用于 recorder 时间戳）。
        recorder: 轨迹记录器或 ``None``。

    Returns:
        实际执行步数（未截断时为 ``_POST_HIT_STEPS``）。
    """
    nq = int(execution.NQ)
    nu = int(execution.NU)
    ctrl_range = execution.model.actuator_ctrlrange
    x_current = np.asarray(execution.get_arm_state(), dtype=np.float64).copy()
    for i in range(_POST_HIT_STEPS):
        q_hold = x_current[:nq].copy()
        if is_position_mode:
            u_hold = q_hold.copy()
        else:
            u_hold = 100.0 * (q_hold - x_current[:nq]) \
                - 10.0 * x_current[nq:]
            u_hold = np.clip(
                u_hold,
                ctrl_range[:nu, 0],
                ctrl_range[:nu, 1],
            )
        x_current, ball_pos_post, _ = execution.step_full(u_hold)
        x_current = np.asarray(x_current, dtype=np.float64)
        sim_component.X_history.append(x_current.copy())
        sim_component.U_history.append(np.asarray(u_hold, dtype=np.float64).copy())
        sim_component.ball_pos_history.append(
            np.asarray(ball_pos_post, dtype=np.float64).copy()
        )
        sim_component.ball_vel_history.append(
            np.asarray(execution.get_ball_vel(), dtype=np.float64).copy()
        )
        if recorder is not None:
            step_idx = total_horizon + follow_through_steps + i
            recorder.record(
                q_desired=q_hold.copy() if is_position_mode else None,
                q_actual=x_current[:nq].copy(),
                timestamp=step_idx * dt,
                tcp_pos=np.asarray(execution.get_ee_pos(), dtype=np.float64),
                ball_pos=np.asarray(ball_pos_post, dtype=np.float64),
                u=np.asarray(u_hold, dtype=np.float64).copy(),
            )
    return _POST_HIT_STEPS


def _resolve_geom_ids(model: object, names: tuple[str, ...]) -> tuple[int, ...]:
    """把 geom 名元组解析成 MuJoCo geom ID 元组。"""
    import mujoco

    return tuple(
        int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in names
    )


def _resolve_geom_pairs(
    model: object, pairs: tuple[object, ...],
) -> tuple[tuple[int, int], ...]:
    """把禁止接触对（geom 名或已解析 ID）统一解析为 ID 对。

    任务配置（``SimTaskConfig.protected_geom_pairs``）以 **geom 名** 表达
    保护语义，``ForbiddenContactChecker`` 只接受已解析的 ID 对；这里按
    ``bundle`` 使用的执行模型解析，保持与 R1D 场景装配（tennis_scene）
    相同的名称→ID 映射。

    Args:
        model: MuJoCo 模型（执行侧，含保护碰撞掩码）。
        pairs: 每项 ``(geom1, geom2)``，元素为 geom 名（str）或 ID（int）。

    Returns:
        已解析的 ``((id1, id2), ...)``。

    Raises:
        ValueError: 名称无法解析（``mj_name2id`` 返回 -1）。
    """
    import mujoco

    resolved: list[tuple[int, int]] = []
    for pair in pairs:
        g1, g2 = pair[0], pair[1]  # type: ignore[index]
        ids: list[int] = []
        for item in (g1, g2):
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
    """构造「球不在工作空间」早退结果（旧脚本打印后 return 的等价物）。"""
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
