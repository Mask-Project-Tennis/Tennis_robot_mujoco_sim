"""V12 配置解析：机器人身份选择与 R1D 显式注册表。

从 ``src/sim/v12_config.py`` 迁入的 R1D 注册表、显式路径解析、旧 sim-task
兼容校验与 RM65/R1D 机器人选择。注册表在模块 import 时即需要绝对路径，
因此本模块显式派生仓库 ``configs/`` 目录（本文件位于
``src/sim/configuration/``，向上三级为仓库根），不依赖 CWD。
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    load_runtime_parameters,
)
from src.sim.configuration.defaults import _arg, _is_explicit
from src.sim.configuration.validation import _validate_overrides

# 注册表需要的是仓库内 ``configs/`` 的绝对路径；显式派生而非读取 CWD。
_CONFIGS_DIR = Path(__file__).resolve().parents[3] / "configs"
_R1D_TASK_CONFIG_PATH = _CONFIGS_DIR / "r1d_v12.yaml"
_R1D_RIGHT_PROFILE_PATH = _CONFIGS_DIR / "robots" / "r1d_right.yaml"




@dataclass(frozen=True)
class R1DConfigRegistry:
    """R1D 显式配置注册表条目。

    只登记“机器人描述 → runtime / 任务覆盖 / 球拍配置”的显式三元组，
    不做文件名推断。字段值均为绝对路径。

    Attributes:
        side: 侧别（``"right"`` / ``"left"``），用于校验关节名前缀。
        runtime_path: R1D V12 运行参数 YAML 路径。
        task_path: R1D V12 任务覆盖 YAML 路径。
        racket_path: 球拍配置 YAML 路径（由 T1 交付，本层只记录路径）。
    """

    side: str
    runtime_path: Path
    task_path: Path
    racket_path: Path


#: R1D 显式注册表：键为解析后的机器人描述绝对路径。
#:
#: 当前只登记右臂：右臂是 T5/T7 的验收臂；左臂尚未提供 V12 专用 runtime /
#: 球拍配置，因此不登记，必须由调用方显式传三个路径，避免“按文件名猜侧别”。
R1D_V12_REGISTRY: Mapping[Path, R1DConfigRegistry] = MappingProxyType(
    {
        _R1D_RIGHT_PROFILE_PATH.resolve(): R1DConfigRegistry(
            side="right",
            runtime_path=(
                _CONFIGS_DIR / "robots" / "r1d_v12_runtime_right.yaml"
            ).resolve(),
            task_path=_R1D_TASK_CONFIG_PATH.resolve(),
            racket_path=(
                _CONFIGS_DIR / "rackets" / "blade100_v9_reference.yaml"
            ).resolve(),
        )
    }
)

# ---------------------------------------------------------------------------
# 机器人选择：RM65 旧路径 / R1D 显式路径
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RobotSelection:
    """机器人选择结果（内部结构，不对外暴露）。"""

    robot_id: str
    ndof: int
    side: str | None
    profile: RobotProfile | None
    profile_path: Path | None
    runtime_path: Path | None
    runtime_params: RobotRuntimeParameters | None
    task_path: Path | None
    racket_path: Path | None
    task_mode: str
    overlay: dict[str, Any]
    overlay_origin: str | None
    path_sources: dict[str, str]


def _resolve_explicit_path(raw: Any, label: str) -> Path:
    """解析并校验显式给出的文件路径（存在性检查）。"""
    path = Path(str(raw)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} 文件不存在: {path}")
    return path


def _resolve_racket_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    return _resolve_explicit_path(raw, "显式 --racket-config")


def _load_task_overlay(path: Path, profile: RobotProfile) -> dict[str, Any]:
    """加载并校验 R1D V12 任务覆盖 YAML。"""
    if not path.is_file():
        raise FileNotFoundError(f"R1D V12 任务覆盖文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"R1D V12 任务覆盖必须是映射: {path}")  # noqa: TRY004
    allowed = {"schema_version", "robot_id", "overrides"}
    unknown = set(raw.keys()) - allowed
    if unknown:
        raise ValueError(
            f"R1D V12 任务覆盖存在未知顶层键 {sorted(unknown)}（允许 {sorted(allowed)}）"
        )
    if int(raw.get("schema_version", 0)) != 1:
        raise ValueError("R1D V12 任务覆盖 schema_version 必须为 1")
    if raw.get("robot_id") != profile.robot_id:
        raise ValueError(
            "R1D V12 任务覆盖 robot_id 与机器人描述不一致："
            f"{raw.get('robot_id')!r} != {profile.robot_id!r}"
        )
    overrides = raw.get("overrides") or {}
    return _validate_overrides(overrides, origin=path.name)


def _check_compat_task(path: Path, profile: RobotProfile) -> None:
    """校验旧固定球 sim-task YAML 的机器人身份（不导入 sim_config）。"""
    if not path.is_file():
        raise FileNotFoundError(f"显式 --sim-task-config 文件不存在: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"旧 sim-task 配置必须是映射: {path}")  # noqa: TRY004
    if raw.get("robot_id") != profile.robot_id:
        raise ValueError(
            "旧 sim-task 配置 robot_id 与机器人描述不一致："
            f"{raw.get('robot_id')!r} != {profile.robot_id!r}"
        )
    joint_names = raw.get("joint_names")
    if not isinstance(joint_names, (list, tuple)):
        raise ValueError("旧 sim-task 配置缺少 joint_names 列表")  # noqa: TRY004
    if tuple(joint_names) != profile.active_joint_names:
        raise ValueError(
            "旧 sim-task 配置 joint_names 与机器人描述主动关节顺序不一致"
        )
    actuator_mode = raw.get("actuator_mode")
    if actuator_mode is not None and actuator_mode not in ("torque", "position"):
        raise ValueError(
            f"旧 sim-task 配置 actuator_mode 非法: {actuator_mode!r}"
        )


_FIXED_BALL_CONFLICT_FLAGS: tuple[tuple[str, str], ...] = (
    ("serve_box", "--serve-box"),
    ("ball_speed", "--ball-speed"),
    ("ball_distance", "--ball-distance"),
    ("approach_angle", "--approach-angle"),
    ("no_bounce", "--no-bounce"),
    ("serve_distance", "--serve-distance"),
    ("serve_height", "--serve-height"),
    ("serve_x_size", "--serve-x-size"),
    ("serve_y_size", "--serve-y-size"),
    ("serve_z_size", "--serve-z-size"),
)

_R1D_FORBIDDEN_FLAGS: tuple[tuple[str, str], ...] = (
    ("limits_config", "--limits-config"),
    ("fix_joint5", "--fix-joint5"),
)


def _reject_r1d_rm65_flags(
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> None:
    """R1D 明确拒绝 RM65 专属旗标（不能接受后忽略）。"""
    for dest, flag in _R1D_FORBIDDEN_FLAGS:
        if _is_explicit(args, dest, explicit):
            raise ValueError(
                f"R1D 不支持 {flag}：该旗标属于 RM65 旧路径，不能接受后忽略"
            )


def _select_robot(
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> _RobotSelection:
    """解析机器人身份并确定 runtime/任务/球拍路径。"""
    robot_raw = _arg(args, "robot_config")
    runtime_raw = _arg(args, "runtime_config")
    sim_task_raw = _arg(args, "sim_task_config")
    racket_raw = _arg(args, "racket_config")

    if robot_raw is None:
        if runtime_raw is not None or sim_task_raw is not None:
            raise ValueError(
                "RM65 旧路径不接受 --runtime-config/--sim-task-config："
                "这两个参数只用于 R1D 显式配置"
            )
        racket_path = _resolve_racket_path(racket_raw)
        path_sources: dict[str, str] = {}
        if racket_path is not None:
            path_sources["racket_path"] = "cli:--racket-config"
        return _RobotSelection(
            robot_id="rm65",
            ndof=6,
            side=None,
            profile=None,
            profile_path=None,
            runtime_path=None,
            runtime_params=None,
            task_path=None,
            racket_path=racket_path,
            task_mode="legacy_cli",
            overlay={},
            overlay_origin=None,
            path_sources=path_sources,
        )

    profile_path = _resolve_explicit_path(robot_raw, "显式 --robot-config")
    profile = load_robot_profile(profile_path)

    if profile.robot_id == "rm65":
        if runtime_raw is not None or sim_task_raw is not None:
            raise ValueError(
                "RM65 旧路径不接受 --runtime-config/--sim-task-config："
                "这两个参数只用于 R1D 显式配置"
            )
        if profile.ndof != 6:
            raise ValueError(
                f"rm65 机器人描述主动关节数应为 6，实际为 {profile.ndof}"
            )
        racket_path = _resolve_racket_path(racket_raw)
        path_sources = {"robot_profile_path": "cli:--robot-config"}
        if racket_path is not None:
            path_sources["racket_path"] = "cli:--racket-config"
        return _RobotSelection(
            robot_id="rm65",
            ndof=6,
            side=None,
            profile=profile,
            profile_path=profile_path,
            runtime_path=None,
            runtime_params=None,
            task_path=None,
            racket_path=racket_path,
            task_mode="legacy_cli",
            overlay={},
            overlay_origin=None,
            path_sources=path_sources,
        )

    if profile.robot_id != "unitree_r1d":
        raise ValueError(
            f"未知 robot_id={profile.robot_id!r}：当前只支持 'rm65' 与 "
            "'unitree_r1d'（R1D 需要显式注册表或显式 runtime/task 路径）"
        )

    return _select_r1d(
        profile=profile,
        profile_path=profile_path,
        runtime_raw=runtime_raw,
        sim_task_raw=sim_task_raw,
        racket_raw=racket_raw,
        args=args,
        explicit=explicit,
    )


def _select_r1d(
    profile: RobotProfile,
    profile_path: Path,
    runtime_raw: Any,
    sim_task_raw: Any,
    racket_raw: Any,
    args: argparse.Namespace,
    explicit: frozenset[str],
) -> _RobotSelection:
    """解析 R1D 选择：显式 runtime/task 优先，其次显式注册表。"""
    if profile.ndof != 7:
        raise ValueError(
            f"unitree_r1d 机器人描述主动关节数应为 7，实际为 {profile.ndof}"
        )
    _reject_r1d_rm65_flags(args, explicit)

    registry = R1D_V12_REGISTRY.get(profile_path)
    if registry is not None:
        prefix = f"{registry.side}_"
        if not all(
            name.startswith(prefix) for name in profile.active_joint_names
        ):
            raise ValueError(
                "R1D 注册表侧别与机器人描述主动关节名前缀不一致："
                f"注册表侧别 {registry.side!r}，关节名 {profile.active_joint_names}"
            )

    path_sources: dict[str, str] = {"robot_profile_path": "cli:--robot-config"}

    # runtime：显式优先；否则取注册表；注册表未登记则留空（后续失败）
    if runtime_raw is not None:
        runtime_path = _resolve_explicit_path(runtime_raw, "显式 runtime")
        try:
            runtime_params = load_runtime_parameters(runtime_path, profile)
        except (TypeError, ValueError, FileNotFoundError) as exc:
            raise ValueError(
                "显式 runtime 校验失败：robot_id 或关节顺序与机器人描述不匹配"
                f"（{runtime_path.name}）：{exc}"
            ) from exc
        runtime_source = "cli:--runtime-config"
    elif registry is not None:
        runtime_path = registry.runtime_path
        runtime_params = None
        runtime_source = f"registry:r1d_{registry.side}"
    else:
        runtime_path = None
        runtime_params = None
        runtime_source = ""
    if runtime_source:
        path_sources["runtime_path"] = runtime_source

    # 任务：显式 sim-task → 固定球兼容模式；否则注册表任务覆盖 → R1D overlay
    if sim_task_raw is not None:
        task_path = _resolve_explicit_path(sim_task_raw, "显式 --sim-task-config")
        _check_compat_task(task_path, profile)
        for dest, flag in _FIXED_BALL_CONFLICT_FLAGS:
            if _is_explicit(args, dest, explicit):
                raise ValueError(
                    f"固定球兼容模式不允许 {flag}：来球由旧 sim-task 文件决定，"
                    "不能同时给 V12 发球参数"
                )
        task_mode = "r1d_fixed_ball_compat"
        overlay: dict[str, Any] = {}
        overlay_origin: str | None = None
        path_sources["task_path"] = "cli:--sim-task-config"
    else:
        if registry is None:
            raise ValueError(
                "该 R1D 机器人描述未在 R1D V12 注册表中登记：请显式提供 "
                "--runtime-config 与 --sim-task-config，或使用已注册的 "
                "configs/robots/r1d_right.yaml"
            )
        task_path = registry.task_path
        overlay = _load_task_overlay(task_path, profile)
        overlay_origin = task_path.name
        task_mode = "r1d_overlay"
        path_sources["task_path"] = f"registry:r1d_{registry.side}"

    # 固定球兼容模式必须同时有 runtime：旧入口用 runtime 重建模型/限位，
    # 不能在 resolve 阶段放过、留到 T5 才失败（复审 Q1）。
    if task_mode == "r1d_fixed_ball_compat" and runtime_path is None:
        raise ValueError(
            "固定球兼容模式（--sim-task-config）必须同时提供 --runtime-config："
            "旧入口依赖 runtime 重建模型与限位，不能留到 T5 才失败"
        )

    # 球拍：显式优先，其次注册表（本层只记录路径，不要求文件已交付）
    if racket_raw is not None:
        racket_path = _resolve_explicit_path(racket_raw, "显式 --racket-config")
        path_sources["racket_path"] = "cli:--racket-config"
    elif registry is not None:
        racket_path = registry.racket_path
        path_sources["racket_path"] = f"registry:r1d_{registry.side}"
    else:
        racket_path = None

    return _RobotSelection(
        robot_id="unitree_r1d",
        ndof=7,
        side=registry.side if registry is not None else None,
        profile=profile,
        profile_path=profile_path,
        runtime_path=runtime_path,
        runtime_params=runtime_params,
        task_path=task_path,
        racket_path=racket_path,
        task_mode=task_mode,
        overlay=overlay,
        overlay_origin=overlay_origin,
        path_sources=path_sources,
    )
