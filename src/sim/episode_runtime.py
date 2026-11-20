"""V12 运行时覆盖的解析与限位恢复（F01 修复线 A）。

本模块把「显式参数 → 已保存覆盖 → 当前默认」的优先级集中到一处，避免
``run_v12_episode`` 只把本次 kwargs 写回配置、却完全不消费 artifact 里已
保存的覆盖（F01 根因）：

* :func:`resolve_runtime_options`：合并显式参数与 ``config.runtime_overrides``，
  得到本次运行「请求」的运行时覆盖（含已保存 limits 文件内容快照）；
* :func:`apply_runtime_limits`：在 ``build_v12_robot`` 之后把请求真正写进
  bundle（``robot_limits`` / ``init_q``），并返回记录实际生效值的新配置。

设计约束：

* 不得 import :mod:`src.sim.episode_artifacts` —— 产物层依赖本模块，反向
  引用会形成循环；
* 恢复已保存 limits 时只消费 artifact 里保存的 ``content`` / ``sha256``，
  **不要求原绝对文件仍然存在**；
* 只有显式给出新的 ``limits_config`` 才重新读文件并完全重算限位，不能把
  旧 artifact 的 effective 值再叠一次；
* ``False`` / ``0`` 是显式覆盖；可空字段（``hit_direction`` /
  ``limits_config`` / ``max_tcp`` / ``terminal_exempt_steps`` /
  ``dq_max_fraction``）的 ``None`` 表示清除该覆盖并回落到当前默认。
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import numpy as np

if TYPE_CHECKING:
    # 仅类型标注：避免在纯解析测试中提前拉起 MuJoCo 模型依赖
    from src.sim.v12_config import ResolvedV12Config
    from src.sim.v12_robot_adapter import V12RobotBundle

logger = logging.getLogger(__name__)

__all__ = [
    "UNSET",
    "UnsetType",
    "apply_runtime_limits",
    "resolve_runtime_options",
]


# ---------------------------------------------------------------------------
# 显式覆盖哨兵
# ---------------------------------------------------------------------------


class UnsetType:
    """显式覆盖哨兵的类型：区分「未传参」与「显式 False/0/None」。"""

    __slots__ = ()

    def __repr__(self) -> str:
        """返回稳定可读表示（日志/断言用）。"""
        return "UNSET"


#: 模块级单例哨兵（``run_v12_episode`` 的 9 个覆盖参数缺省值）。
UNSET: Final[UnsetType] = UnsetType()


# ---------------------------------------------------------------------------
# 键集合与缺省值
# ---------------------------------------------------------------------------

#: 运行选项键（与 ``run_v12_episode`` 参数同名，也写入 runtime_overrides）。
_OPTION_KEYS: Final[tuple[str, ...]] = (
    "ball_speed_perturb_pct",
    "always_collision",
    "first_plan_flight",
    "fixed_direction",
    "hit_direction",
)

#: 限位请求键（每项在 runtime_overrides 里是 requested/effective 记录）。
_LIMIT_REQUEST_KEYS: Final[tuple[str, ...]] = (
    "max_tcp",
    "terminal_exempt_steps",
    "dq_max_fraction",
)

#: 允许显式 ``None`` 清除的键（其余键的 None 视为「未提供」）。
_CLEARABLE_KEYS: Final[frozenset[str]] = frozenset(
    {"hit_direction", "limits_config", *_LIMIT_REQUEST_KEYS}
)

#: 本模块认识的 9 个运行时覆盖键。
_KNOWN_KEYS: Final[frozenset[str]] = frozenset(
    {*_OPTION_KEYS, "limits_config", *_LIMIT_REQUEST_KEYS}
)

#: 选项缺省值（旧调用 ``run_v12_episode(config)`` 的行为）。
_OPTION_DEFAULTS: Final[dict[str, Any]] = {
    "ball_speed_perturb_pct": 0.0,
    "always_collision": False,
    "first_plan_flight": False,
    "fixed_direction": False,
    "hit_direction": None,
}

#: ``RealRobotConfig`` 中以 ndarray 存储、需要从 JSON list 还原的字段。
_REAL_ARRAY_FIELDS: Final[frozenset[str]] = frozenset(
    {"max_qdot", "q_lower", "q_upper", "M_diag", "joint_zero_offset"}
)

#: ``max_tcp=0``（不限速）在产物里的固定说明。
_MAX_TCP_UNLIMITED_NOTE: Final[str] = "max_tcp=0 → 不限速（effective=null）"


# ---------------------------------------------------------------------------
# 轻量类型转换
# ---------------------------------------------------------------------------


def _as_float(value: Any) -> float | None:
    """把 JSON 标量转为 float；None 原样返回。"""
    if value is None:
        return None
    return float(value)


def _as_int(value: Any) -> int | None:
    """把 JSON 标量转为 int；None 原样返回。"""
    if value is None:
        return None
    return int(value)


def _as_float_list(value: Any) -> list[float] | None:
    """把数组/列表转为 float 列表；None 返回 None。"""
    if value is None:
        return None
    return [float(x) for x in np.asarray(value, dtype=np.float64).ravel()]


def _record_field(record: Any, name: str) -> Any:
    """读取 requested/effective 记录字段（兼容旧 artifact 的裸标量）。"""
    if isinstance(record, Mapping):
        return record.get(name)
    return record if name == "requested" else None


def _coerce_option(key: str, value: Any) -> Any:
    """把运行选项规范化为 JSON 原生类型。

    Args:
        key: 选项键（见 ``_OPTION_KEYS``）。
        value: 原始值（可能来自显式参数或 artifact）。

    Returns:
        ``hit_direction`` 为 ``list[float] | None``；扰动百分比为 float；
        其余布尔选项为 bool。

    Raises:
        ValueError: 方向向量不是三维。
    """
    if key == "hit_direction":
        if value is None:
            return None
        array = np.asarray(value, dtype=np.float64).ravel()
        if array.size != 3:
            raise ValueError(
                "hit_direction 必须是三维向量，收到长度 "
                f"{int(array.size)} 的值: {value!r}"
            )
        return [float(x) for x in array]
    if key == "ball_speed_perturb_pct":
        return float(value)
    return bool(value)


def _coerce_request(key: str, value: Any) -> float | int | None:
    """把限位请求值规范化为 JSON 原生类型。"""
    if key == "terminal_exempt_steps":
        return _as_int(value)
    return _as_float(value)


def _coerce_effective(key: str, value: Any) -> float | int | None:
    """把限位 effective 值规范化（``None`` 表示不限速）。"""
    return _coerce_request(key, value)


# ---------------------------------------------------------------------------
# 已保存 limits 内容：快照 / 重建（不落盘）
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """计算文件 SHA256（与产物层同口径）。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def real_config_snapshot(config: Any) -> dict[str, Any]:
    """把 ``RealRobotConfig`` 拍平成 JSON 原生类型快照。

    与 ``v12_episode._real_config_snapshot`` 保持同一口径：数组与列表统一转
    float 列表，标量原样保留。恢复时用该快照重建 ``RealRobotConfig``，
    避免 ``from_yaml`` 对已是弧度的 q_lower/q_upper 二次转度。

    Args:
        config: ``RealRobotConfig`` 实例（鸭子类型即可）。

    Returns:
        字段名 → JSON 原生值的字典。
    """
    snapshot: dict[str, Any] = {}
    for item in fields(config):  # type: ignore[arg-type]
        value = getattr(config, item.name)
        if isinstance(value, np.ndarray):
            snapshot[item.name] = [
                float(x) for x in np.asarray(value, dtype=np.float64).ravel()
            ]
        elif isinstance(value, (list, tuple)):
            snapshot[item.name] = [
                float(x) if isinstance(x, (int, float)) else x for x in value
            ]
        elif value is None or isinstance(value, (bool, int, float, str)):
            snapshot[item.name] = value
        else:
            snapshot[item.name] = str(value)
    return snapshot


def _coerce_real_config_value(name: str, value: Any) -> Any:
    """按 ``RealRobotConfig`` 字段类型还原单个 JSON 值。"""
    if name in _REAL_ARRAY_FIELDS:
        if value is None:
            return value
        return np.asarray(value, dtype=np.float64).ravel()
    return value


def _real_config_from_content(content: Mapping[str, Any]) -> Any:
    """从已保存的内容快照重建 ``RealRobotConfig``（不读文件）。

    Args:
        content: ``runtime_overrides["limits_config"]["content"]``。

    Returns:
        ``RealRobotConfig`` 实例（字段值由快照还原）。

    Raises:
        ValueError: 快照缺少可识别的字段（内容损坏）。
    """
    # 延迟 import：仅恢复真机限位路径需要，避免模块级拉入 src.real
    from src.real.config import RealRobotConfig

    allowed = {item.name for item in fields(RealRobotConfig)}
    kwargs: dict[str, Any] = {}
    unknown: list[str] = []
    for raw_name, value in content.items():
        name = str(raw_name)
        if name not in allowed:
            unknown.append(name)
            continue
        kwargs[name] = _coerce_real_config_value(name, value)
    if unknown:
        logger.warning(
            "limits_config 内容含未知字段（已忽略）: %s", ", ".join(sorted(unknown))
        )
    if not kwargs:
        raise ValueError("limits_config 内容为空，无法重建 RealRobotConfig")
    return RealRobotConfig(**kwargs)


def _build_limits_layer(path: Path) -> dict[str, Any]:
    """读取显式 limits 文件并生成路径+sha256+内容快照。

    Args:
        path: 用户显式给出的 ``--limits-config`` 路径。

    Returns:
        ``{"path": 绝对路径, "sha256": 文件哈希, "content": 字段快照}``。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    from src.real.config import RealRobotConfig

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"限位配置文件不存在: {resolved}")
    real_cfg = RealRobotConfig.from_yaml(resolved)
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "content": real_config_snapshot(real_cfg),
    }


def _resolve_limits_layer(
    saved: Mapping[str, Any], explicit: Mapping[str, Any]
) -> tuple[dict[str, Any] | None, bool]:
    """确定本次运行的 limits 文件层及其来源。

    Returns:
        ``(layer, layer_explicit)``。``layer`` 为 None 表示回落到 bundle 自身
        的 RobotLimits；``layer_explicit`` 为 True 表示用户显式给出新文件
        （含显式 None 清除），此时旧 artifact 的衍生 effective 值必须丢弃。
    """
    if "limits_config" in explicit:
        value = explicit["limits_config"]
        if value is None:
            return None, True
        return _build_limits_layer(Path(value)), True

    raw = saved.get("limits_config")
    if not isinstance(raw, Mapping):
        return None, False
    content = raw.get("content")
    if not isinstance(content, Mapping) or not content:
        logger.warning(
            "已保存的 limits_config 缺少 content 快照，无法恢复限位；"
            "回落到 bundle 当前限位"
        )
        return None, False
    return {
        "path": None if raw.get("path") is None else str(raw.get("path")),
        "sha256": None if raw.get("sha256") is None else str(raw.get("sha256")),
        "content": {str(k): v for k, v in content.items()},
    }, False


# ---------------------------------------------------------------------------
# 解析：显式 > 已保存 > 默认
# ---------------------------------------------------------------------------


def _resolve_option(
    key: str, explicit: Mapping[str, Any], saved: Mapping[str, Any]
) -> Any:
    """按优先级解析单个运行选项。"""
    if key in explicit:
        value = explicit[key]
        if value is not None:
            return _coerce_option(key, value)
        if key in _CLEARABLE_KEYS:
            # 显式 None = 清除该覆盖，回落默认
            return _OPTION_DEFAULTS[key]
        # 非可空字段的 None 视为「未提供」，继续继承已保存值/默认
        logger.debug("运行选项 %s 收到 None，按未提供处理", key)
    if key in saved and saved[key] is not None:
        return _coerce_option(key, saved[key])
    return _OPTION_DEFAULTS[key]


def _resolve_limit_record(
    key: str,
    explicit: Mapping[str, Any],
    saved: Mapping[str, Any],
    layer_explicit: bool,
) -> dict[str, Any]:
    """解析单个限位请求记录。

    ``effective`` 非 None 表示「按 artifact 里实际生效值恢复」；显式请求
    （或显式新 limits 文件）时置 None，由 :func:`apply_runtime_limits` 按
    requested 重新计算，避免把旧 effective 再叠一次。
    """
    if key in explicit:
        raw = explicit[key]
        return {
            "requested": None if raw is None else _coerce_request(key, raw),
            "effective": None,
            "note": None,
        }
    record = saved.get(key)
    if record is None:
        return {"requested": None, "effective": None, "note": None}
    requested = _coerce_request(key, _record_field(record, "requested"))
    if layer_explicit:
        effective: float | int | None = None
        note: str | None = None
    else:
        effective = _coerce_effective(key, _record_field(record, "effective"))
        note = record.get("note") if isinstance(record, Mapping) else None
    return {"requested": requested, "effective": effective, "note": note}


def resolve_runtime_overrides(
    explicit: Mapping[str, Any],
    saved: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """把显式参数与已保存覆盖合并为本次运行的覆盖字典。

    Args:
        explicit: 仅包含用户真正给出的键（值可为 None 表示清除；
            ``limits_config`` 可为 Path）。
        saved: ``ResolvedV12Config.runtime_overrides``（旧 artifact 可能为
            空字典或缺少新键）。

    Returns:
        JSON 原生的运行时覆盖字典（键顺序与旧产物一致：limits_config、
        max_tcp、terminal_exempt_steps、dq_max_fraction、qdot_max、dq_max、
        init_q，随后是 5 个运行选项）。

    Raises:
        ValueError: 显式键未知，或 hit_direction 不是三维向量。
    """
    unknown = sorted(set(explicit) - _KNOWN_KEYS)
    if unknown:
        raise ValueError(
            "未知的运行时覆盖键: " + ", ".join(unknown)
            + "；允许的键: " + ", ".join(sorted(_KNOWN_KEYS))
        )
    saved_map = dict(saved or {})
    layer, layer_explicit = _resolve_limits_layer(saved_map, explicit)

    resolved: dict[str, Any] = {"limits_config": layer}
    for key in _LIMIT_REQUEST_KEYS:
        resolved[key] = _resolve_limit_record(
            key, explicit, saved_map, layer_explicit
        )

    # 已保存的衍生值（M04）：只要没有显式给出新 limits 文件/显式清除，就按
    # artifact 里实际生效的值恢复，**与是否使用 limits 文件无关**。普通仿真
    # 产物 ``limits_config=None`` 同样保存了 qdot_max/dq_max 实际向量；若只在
    # 「存在 limits 内容层」时恢复，当前默认配置一变，同一产物就会静默改用新
    # 默认，恢复执行/回放出现 qdot/dq 漂移。
    # 优先级：显式请求 > 已保存 effective > 当前默认。显式 qdot_max 或
    # dq_max_fraction 时不得再注入对应 saved hint（fraction 由
    # ``_apply_limit_requests`` 按恢复后的 qdot_max 重算 dq）。
    # 注：``_KNOWN_KEYS`` 目前不含 qdot_max/init_q（显式传入会先报未知键），
    # 这里的守卫是防御性的：将来若开放显式向量请求也不会被 saved hint 覆盖。
    hints: dict[str, Any] = {}
    if not layer_explicit:
        if "qdot_max" not in explicit:
            qdot_hint = _as_float_list(saved_map.get("qdot_max"))
            if qdot_hint:
                hints["qdot_max"] = qdot_hint
        if "dq_max_fraction" not in explicit and "dq_max" in saved_map:
            hints["dq_max"] = _as_float_list(saved_map.get("dq_max"))
    if not layer_explicit and "init_q" not in explicit:
        init_q_hint = _as_float_list(saved_map.get("init_q"))
        if init_q_hint:
            hints["init_q"] = init_q_hint
    resolved.update(hints)

    for key in _OPTION_KEYS:
        resolved[key] = _resolve_option(key, explicit, saved_map)
    return resolved


def resolve_runtime_options(
    config: ResolvedV12Config, explicit: Mapping[str, Any]
) -> ResolvedV12Config:
    """解析本次运行实际生效的运行时覆盖（R01 冻结接口）。

    Args:
        config: 已解析的 V12 配置（可能来自 artifact 反序列化）。
        explicit: 仅包含调用方真正给出的覆盖键；缺省的键继承
            ``config.runtime_overrides``，再回落到当前默认。

    Returns:
        带解析后 ``runtime_overrides`` 的新配置（后续 apply/执行/结果/快照
        必须使用该返回值）。
    """
    resolved = resolve_runtime_overrides(explicit, config.runtime_overrides)
    return replace(config, runtime_overrides=resolved)


# ---------------------------------------------------------------------------
# 应用：真正写进 bundle
# ---------------------------------------------------------------------------


def _restore_limits_from_content(
    runtime: Mapping[str, Any], bundle: V12RobotBundle
) -> bool:
    """若运行时含 limits 内容快照，则完全重建 bundle 限位与真机初值。

    Args:
        runtime: 运行时覆盖字典（可能来自 artifact 反序列化）。
        bundle: 机器人事实 bundle（原地替换限位与初值）。

    Returns:
        是否从内容快照重建（True 表示 bundle.robot_limits 已被替换）。
    """
    layer = runtime.get("limits_config")
    if not isinstance(layer, Mapping):
        return False
    content = layer.get("content")
    if not isinstance(content, Mapping) or not content:
        return False

    # 延迟 import：仅真机限位恢复路径需要
    from src.real.runner_factory import build_robot_limits

    real_cfg = _real_config_from_content(content)
    nu = int(getattr(bundle.execution, "NU"))
    q_lower = np.asarray(real_cfg.q_lower, dtype=np.float64).ravel()
    if q_lower.size != nu:
        raise ValueError(
            "limits_config 的关节数与机器人自由度不匹配："
            f"文件 {int(q_lower.size)} vs 机器人 {nu}"
        )
    bundle.robot_limits = build_robot_limits(bundle.execution, real_cfg)
    from src.robot.constants import INIT_Q_REAL

    bundle.init_q = np.asarray(INIT_Q_REAL, dtype=np.float64).copy()
    return True


def _apply_saved_hints(
    bundle: V12RobotBundle, runtime: Mapping[str, Any]
) -> None:
    """恢复 artifact 里保存的 qdot_max / dq_max / init_q 实际生效值。"""
    limits = bundle.robot_limits
    qdot_hint = runtime.get("qdot_max")
    if qdot_hint is not None:
        qdot = np.asarray(limits.qdot_max, dtype=np.float64).ravel()
        saved_qdot = np.asarray(_as_float_list(qdot_hint), dtype=np.float64).ravel()
        if saved_qdot.size != qdot.size:
            raise ValueError(
                "runtime_overrides['qdot_max'] 与机器人关节数不匹配："
                f"{int(saved_qdot.size)} vs {int(qdot.size)}"
            )
        if not np.allclose(saved_qdot, qdot, rtol=0.0, atol=1e-12):
            logger.warning(
                "保存的 qdot_max 与 limits_config 重建值不一致；"
                "按 artifact 实际生效值恢复（前者优先）"
            )
        limits.qdot_max = saved_qdot.copy()

    if "dq_max" in runtime:
        raw_dq = runtime.get("dq_max")
        if raw_dq is None:
            limits.dq_max = None
        else:
            dq = np.asarray(_as_float_list(raw_dq), dtype=np.float64).ravel()
            if dq.size != np.asarray(limits.qdot_max).size:
                raise ValueError(
                    "runtime_overrides['dq_max'] 与机器人关节数不匹配："
                    f"{int(dq.size)} vs {int(np.asarray(limits.qdot_max).size)}"
                )
            limits.dq_max = dq.copy()

    raw_init = runtime.get("init_q")
    if raw_init is not None:
        init_q = np.asarray(_as_float_list(raw_init), dtype=np.float64).ravel()
        current = np.asarray(bundle.init_q, dtype=np.float64).ravel()
        if init_q.size != current.size:
            raise ValueError(
                "runtime_overrides['init_q'] 与机器人关节数不匹配："
                f"{int(init_q.size)} vs {int(current.size)}"
            )
        bundle.init_q = init_q.copy()


def _apply_limit_requests(
    bundle: V12RobotBundle, runtime: Mapping[str, Any], dt: float
) -> None:
    """按 requested/effective 记录把 max_tcp/terminal/dq 写入限位对象。"""
    limits = bundle.robot_limits

    max_tcp_record = runtime.get("max_tcp")
    if max_tcp_record is not None:
        requested = _as_float(_record_field(max_tcp_record, "requested"))
        effective = _as_float(_record_field(max_tcp_record, "effective"))
        if effective is not None:
            limits.max_tcp_speed = float(effective)
        elif requested is not None:
            limits.max_tcp_speed = (
                float("inf") if float(requested) == 0.0 else float(requested)
            )

    terminal_record = runtime.get("terminal_exempt_steps")
    if terminal_record is not None:
        requested = _as_int(_record_field(terminal_record, "requested"))
        effective = _as_int(_record_field(terminal_record, "effective"))
        if effective is not None:
            limits.terminal_exempt_steps = int(effective)
        elif requested is not None:
            limits.terminal_exempt_steps = int(requested)

    # 已保存的 dq_max 向量是实际生效值（见 _apply_saved_hints），优先于
    # 按 fraction 重算；只有没有该 hint 时才消费 dq_max_fraction 请求。
    if "dq_max" not in runtime:
        dq_record = runtime.get("dq_max_fraction")
        requested = (
            None if dq_record is None
            else _as_float(_record_field(dq_record, "requested"))
        )
        if requested is not None:
            limits.dq_max = (
                np.asarray(limits.qdot_max, dtype=np.float64) * float(dt)
                * float(requested)
            )


def _normalize_runtime_overrides(
    bundle: V12RobotBundle,
    runtime: Mapping[str, Any],
    dt: float,
) -> dict[str, Any]:
    """把实际生效值规范化为 JSON 原生覆盖字典（旧产物键顺序不变）。"""
    limits = bundle.robot_limits

    layer = runtime.get("limits_config")
    if isinstance(layer, Mapping):
        content = layer.get("content")
        normalized_layer: dict[str, Any] | None = {
            "path": None if layer.get("path") is None else str(layer.get("path")),
            "sha256": (
                None if layer.get("sha256") is None else str(layer.get("sha256"))
            ),
            "content": (
                {str(k): v for k, v in content.items()}
                if isinstance(content, Mapping) else None
            ),
        }
    else:
        normalized_layer = None

    effective_tcp = float(limits.max_tcp_speed)
    if np.isfinite(effective_tcp):
        max_tcp_effective: float | None = effective_tcp
        max_tcp_note: str | None = None
    else:
        max_tcp_effective = None
        max_tcp_note = _MAX_TCP_UNLIMITED_NOTE

    qdot_max_arr = np.asarray(limits.qdot_max, dtype=np.float64).ravel()
    dq_max_value = getattr(limits, "dq_max", None)
    if dq_max_value is None:
        dq_max_list: list[float] | None = None
        dq_max_fraction_effective: float | None = None
    else:
        dq_max_arr = np.asarray(dq_max_value, dtype=np.float64).ravel()
        dq_max_list = [float(x) for x in dq_max_arr]
        denominator = qdot_max_arr * float(dt)
        fraction = np.divide(
            dq_max_arr,
            denominator,
            out=np.full_like(qdot_max_arr, np.nan),
            where=np.abs(denominator) > 1e-12,
        )
        finite = fraction[np.isfinite(fraction)]
        dq_max_fraction_effective = float(finite[0]) if finite.size else None

    max_tcp_record = runtime.get("max_tcp")
    terminal_record = runtime.get("terminal_exempt_steps")
    dq_record = runtime.get("dq_max_fraction")

    normalized: dict[str, Any] = {
        "limits_config": normalized_layer,
        "max_tcp": {
            "requested": _as_float(_record_field(max_tcp_record, "requested"))
            if max_tcp_record is not None else None,
            "effective": max_tcp_effective,
            "note": max_tcp_note,
        },
        "terminal_exempt_steps": {
            "requested": _as_int(_record_field(terminal_record, "requested"))
            if terminal_record is not None else None,
            "effective": int(limits.terminal_exempt_steps),
        },
        "dq_max_fraction": {
            "requested": _as_float(_record_field(dq_record, "requested"))
            if dq_record is not None else None,
            "effective": dq_max_fraction_effective,
        },
        "qdot_max": [float(x) for x in qdot_max_arr],
        "dq_max": dq_max_list,
        "init_q": [
            float(x)
            for x in np.asarray(bundle.init_q, dtype=np.float64).ravel()
        ],
    }
    for key in _OPTION_KEYS:
        normalized[key] = _coerce_option(
            key, runtime.get(key, _OPTION_DEFAULTS[key])
        )
    return normalized


def apply_runtime_overrides(
    bundle: V12RobotBundle,
    runtime: Mapping[str, Any],
    *,
    dt: float,
) -> dict[str, Any]:
    """把运行时覆盖写入 bundle 并返回规范化后的实际生效记录。

    该函数是 :func:`apply_runtime_limits` 与旧 ``_apply_limits_overrides``
    的共享核心：先按内容快照重建限位（若有），再恢复保存的衍生值，最后
    应用显式请求。

    Args:
        bundle: 机器人事实 bundle（会被原地更新）。
        runtime: 解析后的运行时覆盖（或旧 artifact 的 runtime_overrides）。
        dt: 仿真步长（dq_max 按 ``qdot_max * dt * fraction`` 计算）。

    Returns:
        记录实际生效值的 JSON 原生覆盖字典。
    """
    runtime_map = dict(runtime or {})
    _restore_limits_from_content(runtime_map, bundle)
    _apply_saved_hints(bundle, runtime_map)
    _apply_limit_requests(bundle, runtime_map, float(dt))
    return _normalize_runtime_overrides(bundle, runtime_map, float(dt))


def apply_runtime_limits(
    config: ResolvedV12Config, bundle: V12RobotBundle
) -> ResolvedV12Config:
    """把配置里的运行时覆盖应用到 bundle，并记录实际生效值（R01 冻结接口）。

    必须在 ``build_v12_robot`` 之后、``bundle.reset()`` / 首次 plan 之前调用；
    执行、结果、快照必须使用本函数返回的配置（其 ``runtime_overrides`` 是
    实际生效值，``init_q`` 已按恢复后的 bundle 重新读取）。

    Args:
        config: :func:`resolve_runtime_options` 的返回值（或 artifact 配置）。
        bundle: 机器人事实 bundle（``robot_limits`` / ``init_q`` 会被原地更新）。

    Returns:
        带实际生效 ``runtime_overrides`` 的新配置。
    """
    normalized = apply_runtime_overrides(
        bundle, config.runtime_overrides, dt=float(config.dt)
    )
    return replace(config, runtime_overrides=normalized)
