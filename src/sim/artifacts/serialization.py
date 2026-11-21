# ruff: noqa: I001, PLC0414, UP035, TRY004
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""产物 JSON/NPZ 序列化、哈希与配置 JSON 往返（D1）。

负责 JSON 可序列化转换、规范 JSON、配置字典往返、角色路径解析、输入
哈希、数组映射、payload 构造与 JSON 读取。依赖图位置：``types ←
serialization``。

路径红线：本文件位于 ``<root>/src/sim/artifacts/serialization.py``，
:func:`_repo_root` 使用 ``parents[3]`` 解析仓库根，禁止依赖 CWD。

``build_episode_payload`` 需要资产快照段（由 :mod:`dependencies` 实现），
为避免 ``serialization ↔ dependencies`` 模块级循环，函数内延迟导入。
"""

from __future__ import annotations

import logging
import hashlib
import json
import math
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from src.sim.episode.types import V12EpisodeResult

from src.sim.artifacts.types import (
    EOL_NORMALIZED_ROLES as EOL_NORMALIZED_ROLES,
    EPISODE_SCHEMA as EPISODE_SCHEMA,
    INPUT_SHA_ROLES as INPUT_SHA_ROLES,
    UNITS as UNITS,
    _CONFIG_ARRAY_FIELDS as _CONFIG_ARRAY_FIELDS,
    _CONFIG_PATH_FIELDS as _CONFIG_PATH_FIELDS,
)

_LOGGER = logging.getLogger(__name__)

def _repo_root() -> Path:
    """返回仓库根目录（本文件为 ``<root>/src/sim/artifacts/serialization.py``）。"""
    return Path(__file__).resolve().parents[3]
def _sha256_bytes(data: bytes) -> str:
    """计算字节串的 SHA-256 十六进制摘要。"""
    return hashlib.sha256(data).hexdigest()
def _sha256_file(path: Path) -> str:
    """计算文件的 SHA-256 十六进制摘要。"""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
def _is_eol_normalized_role(role: str) -> bool:
    """判断角色是否按 LF 归一化计算哈希。"""
    return role in EOL_NORMALIZED_ROLES
def _read_normalized_text(path: Path) -> bytes:
    """读取文本配置文件并把 CRLF 归一化为 LF（其余字节保持不变）。"""
    return Path(path).read_bytes().replace(b"\r\n", b"\n")
def _role_file_digest(path: Path, role: str) -> str:
    """按角色规则计算配置/模型文件的 SHA-256（文本角色归一化行尾）。

    Args:
        path: 待哈希文件。
        role: 快照角色（``robot_profile`` / ``runtime`` / ``task`` /
            ``racket`` / ``model``）。

    Returns:
        十六进制摘要；文本角色对 LF 归一化后的字节求哈希，其余角色
        直接对原始字节求哈希。
    """
    if _is_eol_normalized_role(role):
        return _sha256_bytes(_read_normalized_text(path))
    return _sha256_file(path)
def _jsonable(value: Any) -> Any:
    """递归把 numpy/Path/Mapping 等转成 JSON 可序列化对象。

    非有限浮点（``nan`` / ``inf``）统一转成 ``None``：失败回合里
    ``min_dist`` / ``pos_error`` 等可能为 ``inf``，直接序列化会触发
    ``json.dumps(..., allow_nan=False)`` 的 ``ValueError``。
    """
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable(v) for v in value)
    return value
def _canonical_json(value: Any) -> str:
    """生成稳定的（排序键、紧凑、禁用 NaN）JSON 文本，用于哈希。"""
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
def config_to_dict(config: Any) -> dict[str, Any]:
    """把 ``ResolvedV12Config`` 序列化为 JSON 可序列化字典。"""
    payload: dict[str, Any] = {}
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name in _CONFIG_PATH_FIELDS:
            payload[field.name] = None if value is None else str(Path(value))
        elif field.name in _CONFIG_ARRAY_FIELDS:
            payload[field.name] = None if value is None else _jsonable(value)
        elif field.name == "sources":
            payload[field.name] = {str(k): str(v) for k, v in value.items()}
        elif field.name == "explicit_options":
            payload[field.name] = sorted(str(v) for v in value)
        else:
            payload[field.name] = _jsonable(value)
    return payload
def config_from_dict(payload: Mapping[str, Any]) -> Any:
    """由 :func:`config_to_dict` 的字典重建 ``ResolvedV12Config``。"""
    from src.sim.configuration.types import ResolvedV12Config  # 延迟导入避免环

    kwargs: dict[str, Any] = {}
    for field in fields(ResolvedV12Config):
        if field.name not in payload:
            continue
        value = payload[field.name]
        if field.name in _CONFIG_PATH_FIELDS:
            kwargs[field.name] = None if value is None else Path(value)
        elif field.name in _CONFIG_ARRAY_FIELDS:
            kwargs[field.name] = None if value is None else np.asarray(value, dtype=float)
        elif field.name == "explicit_options":
            kwargs[field.name] = frozenset(value)
        else:
            kwargs[field.name] = value
    return ResolvedV12Config(**kwargs)
def _profile_path_for_config(config: Any) -> Path | None:
    """返回机器人描述文件路径（legacy RM65 缺省回退到 rm65.yaml）。"""
    path = getattr(config, "robot_profile_path", None)
    if path is not None:
        return Path(path)
    from src.sim.v12_robot_adapter import default_rm65_profile_path

    return Path(default_rm65_profile_path())
def _model_path_for_profile(profile_path: Path) -> Path:
    """解析机器人描述里 ``model_path`` 的绝对路径。"""
    from src.robot.robot_profile import load_robot_profile

    return Path(load_robot_profile(profile_path).model_path)
def _role_paths(config: Any) -> dict[str, Path | None]:
    """按快照角色解析配置源文件路径（未配置的角色为 None）。"""
    profile_path = _profile_path_for_config(config)
    try:
        model_path: Path | None = (
            _model_path_for_profile(profile_path) if profile_path else None
        )
    except (OSError, ValueError) as exc:  # pragma: no cover - 配置损坏时防御
        _LOGGER.warning("解析机器人模型路径失败：%s", exc)
        model_path = None
    return {
        "robot_profile": profile_path,
        "runtime": getattr(config, "runtime_path", None),
        "task": getattr(config, "task_path", None),
        "racket": getattr(config, "racket_path", None),
        "model": model_path,
    }
def _role_paths_from_payload(payload: Mapping[str, Any]) -> dict[str, Path | None]:
    """从 payload 的 config 快照重建角色路径（用于快照/回放校验）。"""
    from src.sim.configuration.types import ResolvedV12Config  # noqa: F401  (类型校验)

    config = config_from_dict(payload["config"])
    return _role_paths(config)
def _assets_digest(entries: Sequence[Mapping[str, Any]]) -> str:
    """把依赖资产（相对路径 + sha256，排序）聚合为 canonical JSON 的 sha256。"""
    canonical = [
        {
            "relative_path": str(entry["relative_path"]),
            "sha256": str(entry["sha256"]),
        }
        for entry in sorted(entries, key=lambda item: str(item["relative_path"]))
    ]
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()
def _arrays_from_result(result: V12EpisodeResult) -> dict[str, np.ndarray]:
    """从回合结果提取并校验产物的全部数组（形状/对齐）。"""
    joint_names = tuple(str(name) for name in result.joint_names)
    n = len(joint_names)
    X = np.asarray(result.X, dtype=float)
    U = np.asarray(result.U, dtype=float)
    ball_pos = np.asarray(result.ball_pos, dtype=float)
    ball_vel = np.asarray(result.ball_vel, dtype=float)
    time_arr = np.asarray(result.time, dtype=float)
    racket_pos = np.asarray(result.racket_pos, dtype=float)

    if X.ndim != 2 or X.shape[1] != 2 * n:
        raise ValueError(
            f"X 形状 {X.shape} 与 joint_names({n}) 不匹配，期望 (N+1, {2 * n})"
        )
    if U.ndim != 2 or U.shape[1] != n:
        raise ValueError(
            f"U 形状 {U.shape} 与 joint_names({n}) 不匹配，期望 (N, {n})"
        )
    n_steps = int(U.shape[0])
    if int(X.shape[0]) != n_steps + 1:
        raise ValueError(
            f"X 行数({X.shape[0]}) 必须等于 U 行数({n_steps}) + 1"
        )
    for name, arr in (
        ("ball_pos", ball_pos),
        ("ball_vel", ball_vel),
        ("time", time_arr),
        ("racket_pos", racket_pos),
    ):
        expected_rows = n_steps + 1
        if arr.shape[0] != expected_rows:
            raise ValueError(
                f"{name} 行数({arr.shape[0]}) 必须等于 N+1({expected_rows})"
            )
    if ball_pos.shape[1:] != (3,):
        raise ValueError(f"ball_pos 形状 {ball_pos.shape} 期望 (N+1, 3)")
    if ball_vel.shape[1:] != (3,):
        raise ValueError(f"ball_vel 形状 {ball_vel.shape} 期望 (N+1, 3)")
    if racket_pos.shape[1:] != (3,):
        raise ValueError(f"racket_pos 形状 {racket_pos.shape} 期望 (N+1, 3)")
    for name, arr in (
        ("X", X),
        ("U", U),
        ("ball_pos", ball_pos),
        ("ball_vel", ball_vel),
        ("time", time_arr),
        ("racket_pos", racket_pos),
    ):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} 含非有限值（NaN/Inf），拒绝写入产物")
    return {
        "X": X,
        "U": U,
        "ball_pos": ball_pos,
        "ball_vel": ball_vel,
        "time": time_arr,
        "racket_pos": racket_pos,
    }
def _scene_snapshot(result: V12EpisodeResult) -> dict[str, Any]:
    """构造统一场景快照（R1D 取 r1d_artifact.fixed_scene_params）。"""
    config = result.config
    metrics = result.metrics or {}
    r1d_artifact = metrics.get("r1d_artifact")
    source = "metrics.r1d_artifact.fixed_scene_params"
    if isinstance(r1d_artifact, Mapping) and "fixed_scene_params" in r1d_artifact:
        parameters = _jsonable(r1d_artifact["fixed_scene_params"])
    else:
        source = "config.scene_config"
        parameters = _jsonable(getattr(config, "scene_config", {}) or {})
    return {
        "source": source,
        "parameters": parameters,
        "target_center": _jsonable(getattr(config, "target_center", None)),
        "target_offset": _jsonable(getattr(config, "target_offset", None)),
        "workspace_radius": _jsonable(getattr(config, "workspace_radius", None)),
        "ball_initial": {
            "position": _jsonable(np.asarray(result.ball_pos[0], dtype=float)),
            "velocity": _jsonable(np.asarray(result.ball_vel[0], dtype=float)),
        },
        "shoulder_pos": _jsonable(metrics.get("shoulder_pos")),
        "shoulder_pos_source": metrics.get("shoulder_pos_source"),
        "proximity": _jsonable(metrics.get("proximity", {})),
        "dt": float(getattr(config, "dt", 0.0)),
        "seed": getattr(config, "seed", None),
    }
_ROBOT_DISPLAY_NAMES: dict[str, str] = {
    "rm65": "RM-65",
    "unitree_r1d": "R1-A7-D",
}
def _robot_display_name(robot_id: Any) -> str:
    """返回产物 ``hit.message`` 使用的机器人显示名（评审 N1 修复）。"""
    key = str(robot_id or "").strip()
    if key in _ROBOT_DISPLAY_NAMES:
        return _ROBOT_DISPLAY_NAMES[key]
    return key or "RM-65"
def _hit_section(result: V12EpisodeResult) -> dict[str, Any]:
    """构造 ``payload["hit"]``（legacy 与 physical 口径独立）。"""
    from src.sim.hit_detection import classify_hit_result

    metrics = result.metrics or {}
    legacy_hit = bool(metrics.get("legacy_hit", result.hit_type in ("active", "passive")))
    contact_events = _jsonable(metrics.get("contact_events", []))
    divergence = metrics.get("legacy_vs_physical_divergence")
    hit_step = result.hit_step
    if hit_step is None or int(hit_step) < 0:
        time_s = None
    else:
        time_s = float(int(hit_step)) * float(getattr(result.config, "dt", 0.0))
    pos_error = metrics.get("pos_error")
    message = metrics.get("hit_message")
    if message is None:
        try:
            message = classify_hit_result(
                bool(result.active_contact),
                bool(result.passive_contact),
                float(pos_error) if pos_error is not None else 0.0,
                robot_name=_robot_display_name(result.robot_id),
            )
        except (TypeError, ValueError):  # pragma: no cover - 防御异常值
            message = None
    return {
        "legacy_hit": legacy_hit,
        "legacy_hit_type": result.hit_type,
        "legacy_hit_rule": metrics.get(
            "legacy_hit_rule", "first_ball_racket_contact_in_collision_window"
        ),
        "physical_hit": bool(metrics.get("physical_hit", False)),
        "physical_hit_type": metrics.get("physical_hit_type", "miss"),
        "physical_hit_rule": metrics.get(
            "physical_hit_rule", "first_physical_face_contact_face_speed_0p3"
        ),
        "physical_hit_step": metrics.get("physical_hit_step", -1),
        "legacy_vs_physical_divergence": _jsonable(divergence),
        "contact_events": contact_events,
        "step": None if hit_step is None else int(hit_step),
        "time_s": time_s,
        "pos_error": _jsonable(metrics.get("pos_error")),
        "min_dist": _jsonable(metrics.get("min_dist")),
        "v_racket_at_hit": _jsonable(result.v_ee_at_hit),
        "ball_out_speed": _jsonable(result.ball_out_speed),
        "ball_out_vy": _jsonable(result.ball_out_vy),
        "message": message,
    }
def build_episode_payload(result: V12EpisodeResult) -> dict[str, Any]:
    """把回合结果构造成完整、JSON 可序列化的 v2 payload（不落盘）。

    Args:
        result: 任意来源的 :class:`V12EpisodeResult`（真实运行或测试构造）。

    Returns:
        ``v12_episode_v2`` payload 字典（含 ``input_sha256`` 与
        ``scene_snapshot``）。
    """
    # 延迟导入避免 serialization ↔ dependencies 模块级循环
    from src.sim.artifacts.dependencies import _assets_section

    metrics = result.metrics or {}
    config = result.config
    joint_names = [str(name) for name in result.joint_names]
    mode = str(result.mode)
    scene_snapshot = _scene_snapshot(result)
    role_paths = _role_paths(config)
    input_sha256 = {role: "" for role in INPUT_SHA_ROLES}
    for role in ("robot_profile", "runtime", "task", "racket", "model"):
        path = role_paths.get(role)
        if path is not None and Path(path).is_file():
            input_sha256[role] = _role_file_digest(Path(path), role)
    input_sha256["scene_snapshot"] = hashlib.sha256(
        _canonical_json(scene_snapshot).encode("utf-8")
    ).hexdigest()
    # 依赖资产（模型 XML 引用的 mesh/texture）逐文件哈希 + 聚合哈希；
    # 不复制资产文件，回放时按仓库路径重新校验，避免 mesh 被改而模型
    # 哈希不变导致回放几何悄悄变化（T6 追加，设计 §6）。
    assets = _assets_section(role_paths)
    input_sha256["assets"] = str(assets["digest"])

    backend = metrics.get("backend")
    if isinstance(backend, Mapping) and isinstance(backend.get("effective"), str):
        effective = str(backend["effective"])
        finite_difference = effective.endswith("fd")
    else:
        effective = None
        finite_difference = None

    return {
        "schema": EPISODE_SCHEMA,
        "simulation_only": True,
        "robot_id": result.robot_id,
        "mode": mode,
        "racket_id": result.racket_id,
        "joint_names": joint_names,
        "dt": float(getattr(config, "dt", 0.0)),
        "seed": getattr(config, "seed", None),
        "config": config_to_dict(config),
        "input_sha256": input_sha256,
        "scene_snapshot": scene_snapshot,
        "assets": assets,
        "target": _jsonable(getattr(config, "target_center", None)),
        "workspace_radius": _jsonable(getattr(config, "workspace_radius", None)),
        "ball_initial": {
            "position": _jsonable(np.asarray(result.ball_pos[0], dtype=float)),
            "velocity": _jsonable(np.asarray(result.ball_vel[0], dtype=float)),
        },
        "backend": {
            "requested": backend.get("requested") if isinstance(backend, Mapping) else None,
            "effective": effective,
            "solver_backend": getattr(config, "solver_backend", None),
            "solver_config": _jsonable(getattr(config, "solver_config", None)),
            "finite_difference": finite_difference,
        },
        "phase_timings": _jsonable(metrics.get("phase_timings", {})),
        "wall_time_s": _jsonable(metrics.get("wall_time_s")),
        "failure_reason": metrics.get("failure_reason"),
        "episode": {
            "total_steps": metrics.get("total_steps"),
            "safe_steps": metrics.get("safe_steps"),
            "mpc_done": metrics.get("mpc_done"),
            "first_plan_ms": metrics.get("first_plan_ms"),
            "wall_time_s": metrics.get("wall_time_s"),
            "failure_reason": metrics.get("failure_reason"),
        },
        "hit": _hit_section(result),
        "units": dict(UNITS),
        "control": {
            "mode": mode,
            "units": UNITS["control_torque"] if mode == "torque" else UNITS["control_position"],
        },
        "proximity": _jsonable(metrics.get("proximity", {})),
        "note": (
            "hit.legacy_* 为旧版第一接触口径（V11/RM65 行为保持）；hit.physical_* "
            "为 T6 新增的口径（首帧拍面真实接触 + 0.3 m/s 拍速阈值）。"
        ),
    }
def _read_json(path: Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件并断言顶层为对象。"""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层必须是对象：{path}")
    return data
