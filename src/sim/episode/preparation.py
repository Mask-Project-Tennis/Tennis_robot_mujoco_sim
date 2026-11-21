"""V12 回合的准备阶段（配置解析 → 机器人事实 → 发球 → 击打点/Tube）。

本模块承担原 ``run_v12_episode`` 前半段（约 1–5 步）的全部职责，并把结果
收敛为不可变的 :class:`EpisodePreparation`：

1. 收集显式运行时覆盖、解析 ``ResolvedV12Config``、构建机器人 bundle 并
   把限位覆盖真正落到 bundle；
2. 解析发球模式所需的场景/发球事实（shoulder_pos、工作空间、重力等）；
3. 创建唯一 RNG 并按旧顺序生成来球初态；
4. 计算击打点、IK 可达性后过滤、固定击打方向/随挥方向、阶段阈值；
5. 搜索候选击打窗口并构建 initial Tube；
6. 应用球速耦合扰动、重置环境并写入球初态，得到 ``p0_real/v0_real/x0``。

约定：

* 本模块不构造 MPC/组件/runner（由 ``assembly`` 负责），也不写任何文件；
* RNG 只在这里创建一次，发球分支的消费次数/顺序与旧实现逐字一致；
* 早退（``hit_info is None``）时返回 ``hit_info=None`` 的实例，由
  ``execution`` 阶段转交 ``results.build_unreachable_result``；
* 状态所有权：``bundle``/``execution``/``planning``/``rng`` 由本阶段创建，
  随后交给 :class:`EpisodePreparation` 的唯一持有者（execution 编排层）。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np

from src.ilqt.tube_builder import build_hitting_tube, search_hit_window
from src.ilqt.tube_types import HittingTube, HitWindow, TubeConfig
from src.robot.constants import INIT_Q_LEFT
from src.sim.configuration.types import ResolvedV12Config
from src.sim.episode_runtime import (
    UNSET,
    UnsetType,
    apply_runtime_limits,
    resolve_runtime_options,
)
from src.sim.v12_robot_adapter import V12RobotBundle, build_v12_robot
from src.tennis.ball import (
    generate_ball_from_serve_box,
    generate_ball_to_target_box,
)
from src.tennis.hitting import find_hitting_point_physics

# 固定击打方向的兜底值（对应 ``configs/default.yaml`` 的
# ``hitting.hit_direction`` 缺省 [0, -1, 0.3]）。仅在 ``--fixed-direction``
# 且 runtime 与 ``hitting`` 配置都未给出方向时使用；完整优先级见
# :func:`_resolve_hit_direction`。
_DEFAULT_HIT_DIRECTION = (0.0, -1.0, 0.3)


# ---------------------------------------------------------------------------
# 机器人事实 / 配置辅助
# ---------------------------------------------------------------------------


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
    import mujoco

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
    mpc: Any | None = None,
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
        raise ValueError(  # noqa: TRY004 - 历史契约：类型校验也报 ValueError
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


# ---------------------------------------------------------------------------
# 准备阶段结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EpisodePreparation:
    """一次 V12 回合准备阶段产出的全部事实（不可变快照）。

    该对象是准备阶段唯一的输出契约：``execution`` 阶段只读地消费它，
    ``assembly`` 阶段从中取 bundle/limits/MPC 参数，``results`` 阶段从中取
    初态/击打事实/配置快照。任何字段都不在阶段之间二次写回。

    Attributes:
        config: 已解析并应用运行时限位覆盖的配置。
        bundle: 机器人 bundle（含 execution/planning/init_q/robot_limits）。
        execution: 执行环境（物理步进与 FK）。
        planning: 规划环境（MPC 线性化使用）。
        dt: 控制周期（秒）。
        robot_id: 机器人标识（``rm65`` / R1D side）。
        is_position_mode: 是否位置模式。
        runtime: 应用覆盖后的 ``runtime_overrides`` 副本。
        init_q: 初始关节角副本。
        limits: 机器人限位（与 bundle.robot_limits 同一对象）。
        backend: ``{"requested": ..., "effective": ...}`` 后端事实。
        first_plan_flight_value: 首规划飞行开关实际生效值。
        always_collision_value: 球-拍碰撞常开实际生效值。
        ball_speed_perturb_value: 球速耦合扰动百分比实际生效值。
        fixed_direction_value: 固定随挥方向实际生效值。
        hit_direction_vec: 配置/覆盖解析出的单位方向向量。
        hit_direction_config_source: 该向量的来源标签。
        scene_cfg: ``scene`` 配置节副本。
        serve_cfg: ``serve`` 配置节副本。
        mpc_cfg: ``mpc`` 配置节副本（含派生阈值写回）。
        cost_cfg: ``cost`` 配置节副本。
        total_horizon: 主评估段步数。
        total_steps: 主评估段 + 随挥段总步数。
        workspace_radius: 工作空间半径。
        target_center: 目标中心副本。
        target_offset: 目标偏移。
        g: 重力（或 ``None``，由发球生成器取默认）。
        tube_cfg: Tube 配置。
        rng: 本回合唯一 RNG（发球分支已消费）。
        p_racket_init: 初始拍心位置副本。
        shoulder_pos: 肩部位置副本。
        shoulder_pos_source: 肩点来源标签。
        p0: 发球生成器原始球初态位置。
        v0: 发球生成器原始球初态速度。
        hit_info: 击打点字典；``None`` 表示球不可达（早退）。
        k_hit_total: 击打步（命中前过滤后的最终值）。
        p_hit: 击打点位置。
        v_ball_hit: 击球点来球速度。
        n_des_single: 单候选期望拍面法向。
        near_threshold: near 阶段阈值（已写回 ``mpc_cfg``）。
        far_threshold: far 阶段阈值（已写回 ``mpc_cfg``）。
        d_follow: 随挥方向单位向量。
        hit_direction_source: 随挥方向实际来源标签。
        target_speed: 期望来球反方向速度标量。
        v_hit_at_contact: 期望击球速度向量。
        follow_through_steps: 随挥步数。
        racket_speed: 球拍速度阈值。
        ablation_mode: 消融模式字符串。
        need_candidates: 是否需要候选窗口/Tube。
        initial_hitting_tube: 初始 Tube（未搜索到时为 ``None``）。
        hit_window: 候选击打窗口（未搜索到时为 ``None``）。
        p0_real: 应用球速扰动后的环境实际初态位置。
        v0_real: 应用球速扰动后的环境实际初态速度。
        x0: 初始状态向量（前 NQ 为 ``init_q``）。
    """

    config: ResolvedV12Config
    bundle: V12RobotBundle
    execution: Any
    planning: Any
    dt: float
    robot_id: str
    is_position_mode: bool
    runtime: dict[str, Any]
    init_q: np.ndarray
    limits: Any
    backend: dict[str, str]
    first_plan_flight_value: bool
    always_collision_value: bool
    ball_speed_perturb_value: float
    fixed_direction_value: bool
    hit_direction_vec: np.ndarray
    hit_direction_config_source: str
    scene_cfg: dict[str, Any]
    serve_cfg: dict[str, Any]
    mpc_cfg: dict[str, Any]
    cost_cfg: dict[str, Any]
    total_horizon: int
    total_steps: int
    workspace_radius: float
    target_center: np.ndarray
    target_offset: float
    g: Any
    tube_cfg: TubeConfig
    rng: np.random.Generator
    p_racket_init: np.ndarray
    shoulder_pos: np.ndarray
    shoulder_pos_source: str
    p0: np.ndarray
    v0: np.ndarray
    hit_info: Mapping[str, Any] | None = None
    k_hit_total: int = 0
    p_hit: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v_ball_hit: np.ndarray = field(default_factory=lambda: np.zeros(3))
    n_des_single: np.ndarray = field(default_factory=lambda: np.zeros(3))
    near_threshold: int = 50
    far_threshold: int = 50
    d_follow: np.ndarray = field(default_factory=lambda: np.zeros(3))
    hit_direction_source: str = "reflect"
    target_speed: float = 1.8
    v_hit_at_contact: np.ndarray = field(default_factory=lambda: np.zeros(3))
    follow_through_steps: int = 0
    racket_speed: float = 1.8
    ablation_mode: str = "full"
    need_candidates: bool = True
    initial_hitting_tube: HittingTube | None = None
    hit_window: HitWindow | None = None
    p0_real: np.ndarray = field(default_factory=lambda: np.zeros(3))
    v0_real: np.ndarray = field(default_factory=lambda: np.zeros(3))
    x0: np.ndarray = field(default_factory=lambda: np.zeros(0))


# ---------------------------------------------------------------------------
# 阶段入口
# ---------------------------------------------------------------------------


def prepare_episode(
    config: ResolvedV12Config,
    *,
    first_plan_flight: bool | UnsetType = UNSET,
    always_collision: bool | UnsetType = UNSET,
    ball_speed_perturb_pct: float | UnsetType = UNSET,
    fixed_direction: bool | UnsetType = UNSET,
    hit_direction: tuple[float, float, float] | None | UnsetType = UNSET,
    limits_config: Path | None | UnsetType = UNSET,
    max_tcp: float | None | UnsetType = UNSET,
    terminal_exempt_steps: int | None | UnsetType = UNSET,
    dq_max_fraction: float | None | UnsetType = UNSET,
    robot_builder: Callable[[ResolvedV12Config], V12RobotBundle] = build_v12_robot,
) -> EpisodePreparation:
    """执行 V12 准备阶段并返回不可变快照。

    调用顺序与原 ``run_v12_episode`` 前半段完全一致：
    ``resolve_runtime_options`` → ``robot_builder`` → ``apply_runtime_limits``
    → 配置/场景事实 → RNG/发球 → 击打点/IK 过滤 → 阈值/方向 → 候选窗口
    → 球速扰动 → 环境重置/球初态。

    行数说明（contracts §1.7）：本函数约 350 行，是刻意保留的单一路径协议
    函数。长度来自上述严格顺序中共享的局部状态（config/bundle/rng/p0/ik 结果/
    窗口）；抽段需要可变上下文或 30+ 参数透传，且任何顺序漂移都会改变 RNG
    消费，风险高于收益。可独立复用的纯逻辑已抽为模块级 helper
    （``_build_tube_config``、``_resolve_serve_box_*``、``_effective_backend`` 等）。

    Args:
        config: 已解析配置（本函数只接受 ``ResolvedV12Config``，类型校验在
            ``run_v12_episode_impl`` 完成，以保持原报错文案）。
        first_plan_flight: 首规划飞行覆盖（``UNSET`` 继承已保存值）。
        always_collision: 球-拍碰撞常开覆盖（``UNSET`` 继承）。
        ball_speed_perturb_pct: 球速耦合扰动覆盖（``UNSET`` 继承）。
        fixed_direction: 固定随挥方向覆盖（``UNSET`` 继承）。
        hit_direction: 固定击打方向覆盖（``None`` 回落到配置/默认）。
        limits_config: 显式限位文件（``None`` 清除已保存文件层）。
        max_tcp: TCP 上限覆盖（``None`` 清除已保存覆盖）。
        terminal_exempt_steps: 终段豁免步数覆盖（``None`` 清除）。
        dq_max_fraction: 关节增量分数覆盖（``None`` 清除）。
        robot_builder: 机器人构建函数；facade 在调用时注入当前模块属性，使
            monkeypatch 生效。

    Returns:
        :class:`EpisodePreparation`；``hit_info is None`` 表示球不可达。
    """
    from src.sim.episode.results import _explicit_runtime_options

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
    bundle = robot_builder(config)
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

    common: dict[str, Any] = {
        "config": config,
        "bundle": bundle,
        "execution": execution,
        "planning": planning,
        "dt": dt,
        "robot_id": robot_id,
        "is_position_mode": is_position_mode,
        "runtime": runtime,
        "init_q": init_q,
        "limits": limits,
        "backend": backend,
        "first_plan_flight_value": first_plan_flight_value,
        "always_collision_value": always_collision_value,
        "ball_speed_perturb_value": ball_speed_perturb_value,
        "fixed_direction_value": fixed_direction_value,
        "hit_direction_vec": hit_direction_vec,
        "hit_direction_config_source": hit_direction_config_source,
        "scene_cfg": scene_cfg,
        "serve_cfg": serve_cfg,
        "mpc_cfg": mpc_cfg,
        "cost_cfg": cost_cfg,
        "total_horizon": total_horizon,
        "total_steps": total_horizon,
        "workspace_radius": workspace_radius,
        "target_center": target_center,
        "target_offset": target_offset,
        "g": g,
        "tube_cfg": tube_cfg,
        "rng": rng,
        "p_racket_init": p_racket_init,
        "shoulder_pos": shoulder_pos,
        "shoulder_pos_source": shoulder_pos_source,
        "p0": p0,
        "v0": v0,
    }

    # ── 3. 击打点 + IK 可达性后过滤（旧 L716-758）───────────────────────
    hit_info = find_hitting_point_physics(
        execution, p0, v0, shoulder_pos, workspace_radius, total_horizon,
    )
    if hit_info is None:
        return EpisodePreparation(**common)

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

    common["total_steps"] = total_horizon + follow_through_steps
    return EpisodePreparation(
        **common,
        hit_info=hit_info,
        k_hit_total=k_hit_total,
        p_hit=p_hit,
        v_ball_hit=v_ball_hit,
        n_des_single=n_des_single,
        near_threshold=near_threshold,
        far_threshold=far_threshold,
        d_follow=d_follow,
        hit_direction_source=hit_direction_source,
        target_speed=target_speed,
        v_hit_at_contact=v_hit_at_contact,
        follow_through_steps=follow_through_steps,
        racket_speed=racket_speed,
        ablation_mode=ablation_mode,
        need_candidates=need_candidates,
        initial_hitting_tube=initial_hitting_tube,
        hit_window=hit_window,
        p0_real=np.asarray(p0_real, dtype=np.float64).copy(),
        v0_real=np.asarray(v0_real, dtype=np.float64).copy(),
        x0=np.asarray(x0, dtype=np.float64),
    )
