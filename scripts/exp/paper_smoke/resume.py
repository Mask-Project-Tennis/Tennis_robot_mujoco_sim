"""续跑与产物校验：读取历史 runs.csv、保留指标、回填实际路径、
校验已完成任务的 v12/r1d 产物完整性，并汇总 runs.csv。
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.exp.batch_runner import CSV_FIELDS

from .artifacts import (
    _artifact_metrics,
    _config_json_with_output,
    _load_npz,
    _npz_scalar_text,
    _portable_run_row,
    _resolve_run_artifact_dir,
    _resolve_run_log_path,
)
from .identity import _record_key, _records_index, _resolved_task_index
from .manifest import (
    ARTIFACT_FIELDS,
    PROJECT,
    R1D_EPISODE_NPZ_NAME,
    RUNS_HIT_SEMANTICS,
    V12_EPISODE_NAME,
    V12_EPISODE_NPZ_NAME,
    V12_NPZ_KEYS,
    _display_path,
)
from .tasks import _run_root_relative

_OVERRIDE_SECTION_FIELDS: dict[str, str] = {
    "mpc": "mpc_config",
    "solver": "solver_config",
    "cost": "cost_config",
    "perception": "perception_config",
    "hitting": "hitting_config",
    "serve": "serve_config",
}

_TARGET_FIELD_ALIASES: dict[str, str] = {
    "center": "target_center",
    "offset": "target_offset",
    "workspace_radius": "workspace_radius",
}

_SERVE_SCENE_ALIASES = frozenset({"use_bounce", "bounce_restitution"})

_MISSING = object()


def _read_previous_runs(
    path: Path,
) -> dict[tuple[str, str, int], dict[str, str]]:
    """读取即将被覆盖的 runs.csv，按续跑键索引旧行（R01）。

    Args:
        path: ``<run_root>/runs.csv``（可能不存在）。

    Returns:
        续跑键到旧行的映射；文件缺失或无法解析时返回空字典。
    """
    if not path.is_file():
        return {}
    index: dict[tuple[str, str, int], dict[str, str]] = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for raw in csv.DictReader(fh):
                key = _record_key(raw)
                if key is not None:
                    index[key] = dict(raw)
    except (OSError, csv.Error):
        return {}
    return index


def _preserve_previous_metrics(
    row: dict[str, Any],
    previous: Mapping[str, Any] | None,
) -> bool:
    """新提取失败时沿用旧 runs.csv 的有效物理指标（R01）。

    只有旧行本身没有 ``artifact_error`` 且至少有一个非空指标时才保留：
    旧行若也标记损坏，不能用它冒充有效数据。保留后清空 ``artifact_error``，
    让论文统计继续按有效样本计数；产物缺失仍由后置校验报错并 exit 1。

    Args:
        row: 本次聚合行（``artifact_error`` 非空）。
        previous: 同一续跑键的旧 runs.csv 行（可能为 None）。

    Returns:
        是否保留了旧指标。
    """
    if not previous:
        return False
    if str(previous.get("artifact_error") or "").strip():
        return False
    preserved = False
    for field in ARTIFACT_FIELDS:
        if field == "artifact_error":
            continue
        value = previous.get(field)
        if value is None or not str(value).strip():
            continue
        row[field] = value
        preserved = True
    if not preserved:
        return False
    row["artifact_error"] = ""
    return True


def _collect_runs(resolved: Mapping[str, Any], batch_root: Path,
                  results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """把各实验 results.csv 汇总为 runs.csv，并追加产物提取列。

    R01：先按 ``(experiment, config_id, seed)`` 找到 ``resolved`` 中的任务，
    用任务在当前 run_root 下的计划 ``artifact_dir``/``log_path`` 定位真实
    产物；行内旧绝对路径仍存在时保留其优先级。写盘前读取旧 ``runs.csv``，
    对提取失败的行沿用旧的非空指标，避免整树搬移后的再次 resume 把已验证
    的 physical/hit 列清空。

    Args:
        resolved: ``resolve_manifest`` 的结果（含 run_root/tasks）。
        batch_root: batch 数据根（当前仅由 batch_runner 使用，保留签名）。
        results: ``batch_runner.run_batch`` 返回的单元摘要列表。

    Returns:
        含 ``runs_csv``/``runs``/``hit_semantics``/``records`` 的汇总字典；
        ``records`` 保留解析后的绝对路径，供路径回填与后置校验复用。
    """
    run_root = Path(resolved["run_root"])
    tasks = _resolved_task_index(resolved)
    previous = _read_previous_runs(run_root / "runs.csv")
    rows: list[dict[str, Any]] = []
    for summary in results:
        csv_path = Path(str(summary.get("csv", "")))
        if not csv_path.is_absolute():
            csv_path = PROJECT / csv_path
        if not csv_path.is_file():
            continue
        with open(csv_path, "r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                row: dict[str, Any] = dict(raw)
                key = _record_key(row)
                task = tasks.get(key) if key is not None else None
                artifact_dir = _resolve_run_artifact_dir(row, task, run_root)
                if artifact_dir is not None:
                    row["artifact_dir"] = str(artifact_dir)
                    if "config_json" in row:
                        row["config_json"] = _config_json_with_output(
                            row.get("config_json"), artifact_dir)
                log_path = _resolve_run_log_path(row, task, run_root)
                if log_path is not None:
                    row["log_path"] = str(log_path)
                row.update(_artifact_metrics(row.get("artifact_dir", "")))
                # R02：提取成功不代表产物可信（NPZ 截断、日志为空、配置
                # 错配都可能仍能读出 hit）。这里先按任务做完备校验，让错配
                # 行也走“保留旧指标 + 后置报错”的路径，避免把错误数值写进
                # 论文统计；后置校验仍会输出完整问题与恢复指引。
                if (task is not None and artifact_dir is not None
                        and artifact_dir.is_dir()):
                    check_issues = _artifact_issues(
                        task, artifact_dir, row, run_root)
                    if (check_issues
                            and not str(row.get("artifact_error") or "").strip()):
                        row["artifact_error"] = (
                            f"artifact_validation: {check_issues[0]}")
                if str(row.get("artifact_error") or "").strip():
                    _preserve_previous_metrics(
                        row, previous.get(key) if key is not None else None)
                rows.append(row)
    runs_path = run_root / "runs.csv"
    fieldnames = CSV_FIELDS + [key for key in ARTIFACT_FIELDS if key not in CSV_FIELDS]
    if "hit_semantics" not in fieldnames:
        fieldnames = fieldnames + ["hit_semantics"]
    runs_path.parent.mkdir(parents=True, exist_ok=True)
    with open(runs_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            # hit 是 batch_runner legacy 口径（active|passive）；论文统计用
            # physical_hit / physical_hit_type，禁止把 hit 当物理命中。
            # 落盘行相对化 run_root 内路径，records 保留原绝对路径：
            # `_apply_actual_paths` 用它回填旧 output 的真实产物位置（N01）。
            portable = _portable_run_row(row, run_root)
            portable["hit_semantics"] = RUNS_HIT_SEMANTICS
            writer.writerow(portable)
    return {"runs_csv": _display_path(runs_path), "runs": len(rows),
            "hit_semantics": RUNS_HIT_SEMANTICS,
            # 原始行（含 artifact_dir/log_path）：用于换 output 续跑时把
            # manifest 的计划路径回填为旧 CSV 的真实产物位置（N01）。
            "records": rows}


def _apply_actual_paths(
    resolved: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    run_root: Path,
) -> int:
    """把已完成任务的真实产物/日志路径回填到 manifest 任务清单（N01）。

    换 ``--output`` 续跑时 batch 命中旧 CSV、``launched=0``，旧产物与日志仍在
    原 output 目录。按 ``(experiment, config_id, seed)`` 找到 CSV 行，用其中
    真实路径覆盖任务计划值：位于新 run_root 内→相对路径，否则保留绝对路径，
    保证旧文件始终可解析。``planned_artifact_dir``/``planned_log_path`` 保留
    本次 output 的计划位置，不被覆盖。

    Args:
        resolved: ``resolve_manifest`` 的可变结果（就地更新 ``tasks``）。
        records: 富化后的 CSV 行（``_collect_runs`` 的 ``records``）。
        run_root: 本次 output 根目录。

    Returns:
        被回填的任务数（0 表示没有可复用的已完成记录）。
    """
    index = _records_index(records)
    updated = 0
    for task in resolved.get("tasks", []):
        key = _record_key(task)
        if key is None:
            continue
        row = index.get(key)
        if row is None:
            continue
        changed = False
        for field in ("artifact_dir", "log_path"):
            value = str(row.get(field) or "").strip()
            if not value:
                continue
            # M06 交叉：整树搬移后旧 CSV 的绝对路径已不存在；此时保留本次
            # output 的计划路径（相对当前 run_root），避免把旧根写回 manifest。
            candidate = _row_path(run_root, value)
            if candidate is None or not candidate.exists():
                continue
            task[field] = _run_root_relative(run_root, value)
            changed = True
        if changed:
            updated += 1
    return updated


def _row_path(run_root: Path, value: Any) -> Path | None:
    """把 runs.csv 中的路径字段解析为绝对路径。

    Args:
        run_root: 本次 output 根目录（相对路径以其为基准）。
        value: CSV 字段原值（可能为空、相对或绝对）。

    Returns:
        绝对路径；字段为空时为 None。
    """
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    return path if path.is_absolute() else Path(run_root) / path


def _expected_output_matches(artifact_dir: Path, pattern: str) -> list[Path]:
    """在 ``artifact_dir`` 下匹配一个 ``expected_outputs`` 模式。

    支持普通文件/目录与 ``glob`` 通配（``*``/``?``/``[``）。

    Args:
        artifact_dir: 单次运行的产物目录（绝对）。
        pattern: manifest 中相对 ``artifact_dir`` 的模式。

    Returns:
        匹配到的路径列表（普通路径存在时为单元素，否则为空）。
    """
    target = artifact_dir / pattern
    if any(char in pattern for char in "*?["):
        parent = target.parent
        if not parent.is_dir():
            return []
        return sorted(parent.glob(target.name))
    return [target] if target.exists() else []


def _is_completed_record(row: Mapping[str, Any]) -> bool:
    """判断 runs.csv 行是否为已完成（非 error/非 timeout）记录。

    Args:
        row: CSV 行。

    Returns:
        ``error`` 为空且未标记 ``timed_out`` 时为 True。
    """
    if str(row.get("error") or "").strip():
        return False
    return str(row.get("timed_out") or "").strip() not in ("1", "true", "True")


def _v12_identity_issues(
    payload: Mapping[str, Any],
    task: Mapping[str, Any],
    row: Mapping[str, Any],
) -> list[str]:
    """校验 ``v12_episode.json`` 的 robot_id/mode/seed 与任务行一致。

    Args:
        payload: ``v12_episode.json`` 解析后的字典。
        task: manifest 任务。
        row: runs.csv 行。

    Returns:
        身份不符的描述列表（空表示一致）。
    """
    identity = task.get("identity") or {}
    expected: dict[str, Any] = {
        "robot_id": str(identity.get("robot_id") or row.get("robot_id") or ""),
        "mode": str(task.get("mode") or identity.get("mode") or ""),
        "seed": task.get("seed"),
    }
    issues: list[str] = []
    for field, want in expected.items():
        have = payload.get(field)
        if field == "seed":
            try:
                matched = int(have) == int(want)
            except (TypeError, ValueError):
                matched = False
        else:
            matched = str(have) == str(want)
        if not matched:
            issues.append(
                f"v12_episode.json {field} 不符（产物={have!r}，期望={want!r}）")
    return issues


def _override_config_path(key: str) -> tuple[str, ...] | None:
    """把 ``--task-overrides`` 点路径映射为 ``payload["config"]`` 路径（R02）。

    映射规则与 ``src/sim/v12_config.py::_set_value`` 一致：``actuator.*``
    折叠为顶层字段，``target.*`` 用同义顶层字段，``serve.use_bounce`` 等
    落在 ``scene_config``，其余按 ``solver/mpc/cost/perception/hitting/serve``
    节前缀映射到对应的 ``*_config``。

    Args:
        key: override 键（如 ``cost.tube.window_half_ms``）。

    Returns:
        config 内的字段路径；无法映射时返回 None。
    """
    if key == "seed":
        return ("seed",)
    if key == "actuator.mode":
        return ("actuator_mode",)
    if key == "actuator.feedforward":
        return ("feedforward",)
    if key in ("actuator.kp", "actuator.kd"):
        return (key.rsplit(".", 1)[1],)
    section, _, rest = key.partition(".")
    if section == "target" and rest in _TARGET_FIELD_ALIASES:
        return (_TARGET_FIELD_ALIASES[rest],)
    if section == "target" and rest == "shoulder_pos":
        return ("scene_config", "shoulder_pos")
    if section == "serve" and rest in _SERVE_SCENE_ALIASES:
        return ("scene_config", rest)
    if section in _OVERRIDE_SECTION_FIELDS and rest:
        return (_OVERRIDE_SECTION_FIELDS[section], *rest.split("."))
    return None


def _lookup_config_path(config: Mapping[str, Any], path: Sequence[str]) -> Any:
    """按点路径读取 config；任一层缺失时返回 ``_MISSING``。

    Args:
        config: ``v12_episode.json`` 的 ``config`` 段。
        path: :func:`_override_config_path` 生成的字段路径。

    Returns:
        叶子值或 ``_MISSING``。
    """
    node: Any = config
    for part in path:
        if not isinstance(node, Mapping) or part not in node:
            return _MISSING
        node = node[part]
    return node


def _numeric_value_matches(actual: Any, expected: Any) -> bool:
    """按浮点容差比较数值（bool 不参与数值比较）。"""
    if isinstance(actual, bool) or isinstance(expected, bool):
        return False
    try:
        return math.isclose(
            float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-12)
    except (TypeError, ValueError):
        return False


def _override_value_matches(actual: Any, text: str) -> bool:
    """把 override 文本按产物值类型归一化后比较（R02）。

    Args:
        actual: ``payload["config"]`` 中的实际值。
        text: ``key=value`` 中 ``_format_override`` 写出的文本。

    Returns:
        bool 严格相等、数值按容差、字符串逐字相等；无法解析为 False。
    """
    token = text.strip()
    if isinstance(actual, bool):
        lowered = token.lower()
        if lowered in ("true", "1", "yes", "on"):
            return actual is True
        if lowered in ("false", "0", "no", "off"):
            return actual is False
        return False
    if isinstance(actual, (int, float)):
        try:
            return math.isclose(
                float(actual), float(token), rel_tol=1e-9, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if actual is None:
        return token.lower() in ("none", "null")
    return str(actual) == token


def _v12_config_issues(
    payload: Mapping[str, Any],
    task: Mapping[str, Any],
) -> list[str]:
    """校验 ``payload["config"]`` 与任务命令行一致（R02 科学配置）。

    至少覆盖 ``--ball-speed``、执行模式、``--no-feedforward`` 和
    ``--task-overrides`` 的每个点路径；同 robot/mode/seed 但球速不同的
    完整 JSON 放入另一任务目录时必须失败。

    Args:
        payload: ``v12_episode.json`` 解析后的字典。
        task: manifest 任务（提供 argv/mode）。

    Returns:
        问题描述列表（空表示一致）。
    """
    config = payload.get("config")
    if not isinstance(config, Mapping):
        return ["v12_episode.json 缺少 config 段，无法校验科学配置"]
    argv = task.get("argv") or {}
    issues: list[str] = []
    # 真实 writer 的 config 一定带 sources/explicit_options 溯源字段；只有它们
    # 存在时才要求“每个任务字段都在 config 里”（缺失即失败）。合成/旧式最小
    # 产物没有溯源字段，只能核对已声明的字段是否一致（不把未知当错误，否则
    # 会拒绝历史最小夹具）。已声明但不一致的情况两种情况都必须报错。
    complete_config = (
        isinstance(config.get("sources"), Mapping)
        and "explicit_options" in config
    )
    if "--ball-speed" in argv:
        serve = config.get("serve_config")
        if isinstance(serve, Mapping):
            actual = serve.get("ball_speed")
        else:
            # 兼容合成/旧式最小产物：球速直接写在 config 顶层。
            actual = config.get("ball_speed")
        if not _numeric_value_matches(actual, argv["--ball-speed"]):
            issues.append(
                "config.serve_config.ball_speed 不符"
                f"（产物={actual!r}，命令={argv['--ball-speed']!r}）")
    if "seed" in config:
        try:
            seed_matched = int(config["seed"]) == int(task.get("seed"))
        except (TypeError, ValueError):
            seed_matched = False
        if not seed_matched:
            issues.append(
                "config.seed 不符"
                f"（产物={config['seed']!r}，任务 seed={task.get('seed')!r}）")
    elif complete_config:
        issues.append("config 缺少 seed（完整产物必须声明随机种子）")
    want_mode = (
        "position"
        if str(task.get("mode") or "").strip().lower() == "position"
        else "torque")
    actual_mode = config.get("actuator_mode")
    if actual_mode is None:
        if complete_config:
            issues.append("config 缺少 actuator_mode（完整产物必须声明执行模式）")
    elif str(actual_mode) != want_mode:
        issues.append(
            "config.actuator_mode 不符"
            f"（产物={actual_mode!r}，任务 mode={want_mode!r}）")
    if "--no-feedforward" in argv:
        if "feedforward" not in config:
            if complete_config:
                issues.append("config 缺少 feedforward（完整产物必须声明前馈开关）")
        elif config.get("feedforward") is not False:
            issues.append(
                "config.feedforward 应为 False（命令含 --no-feedforward，"
                f"产物={config.get('feedforward')!r}）")
    for item in argv.get("--task-overrides") or []:
        key, sep, text = str(item).partition("=")
        if not sep:
            issues.append(f"--task-overrides 项缺少 '='：{item!r}")
            continue
        path = _override_config_path(key)
        if path is None:
            issues.append(f"--task-overrides 无法映射到 config：{key!r}")
            continue
        actual = _lookup_config_path(config, path)
        dotted = ".".join(path)
        if actual is _MISSING:
            if complete_config:
                issues.append(f"config 缺少覆盖字段 {dotted}（override {key}）")
        elif not _override_value_matches(actual, text):
            issues.append(
                f"config.{dotted} 与 override {key}={text} 不符"
                f"（产物={actual!r}）")
    return issues


def _v12_payload_issues(
    path: Path,
    task: Mapping[str, Any],
    row: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, list[str]]:
    """校验 ``v12_episode.json`` 可解析、schema/身份/科学配置正确。

    Args:
        path: ``v12_episode.json`` 绝对路径。
        task: manifest 任务。
        row: runs.csv 行。

    Returns:
        ``(payload, issues)``：payload 供同目录 NPZ 校验取 ``joint_names``；
        解析失败时 payload 为 None。
    """
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return None, [f"v12_episode.json 无法解析: {exc}"]
    if not isinstance(payload, Mapping):
        return None, ["v12_episode.json 顶层不是 JSON 对象"]
    issues: list[str] = []
    schema = payload.get("schema")
    if schema != "v12_episode_v2":
        issues.append(f"v12_episode.json schema 不是 v12_episode_v2（实为 {schema!r}）")
    issues.extend(_v12_identity_issues(payload, task, row))
    issues.extend(_v12_config_issues(payload, task))
    return payload, issues


def _v12_npz_issues(
    path: Path,
    payload: Mapping[str, Any] | None,
) -> list[str]:
    """校验共享 v12 NPZ 的 schema/数组形状/finite（R02）。

    Args:
        path: ``v12_episode.npz`` 绝对路径。
        payload: 同目录 ``v12_episode.json`` 的解析结果（可取 joint_names）。

    Returns:
        问题描述列表（空表示通过）。
    """
    arrays, issues = _load_npz(path)
    if arrays is None:
        return issues
    # 显式 synthetic 产物是测试/集成夹具（非真实 writer 输出），允许省略
    # schema 数组；其余形状/finite/行关系检查照常执行。真实产物仍强制 schema。
    synthetic = isinstance(payload, Mapping) and payload.get("synthetic") is True
    required = [
        name for name in V12_NPZ_KEYS
        if not (synthetic and name == "schema")
    ]
    missing = [name for name in required if name not in arrays]
    if missing:
        return issues + [f"v12_episode.npz 缺少数组 {', '.join(missing)}"]
    schema = (
        _npz_scalar_text(arrays["schema"])
        if "schema" in arrays else "v12_episode_v2")
    if schema != "v12_episode_v2":
        issues.append(
            f"v12_episode.npz schema 不是 v12_episode_v2（实为 {schema!r}）")
    for name in V12_NPZ_KEYS[1:]:
        array = arrays[name]
        if not np.issubdtype(array.dtype, np.number):
            issues.append(
                f"v12_episode.npz 数组 {name} dtype 非数值（{array.dtype}）")
            continue
        if not np.all(np.isfinite(array)):
            issues.append(f"v12_episode.npz 数组 {name} 含 NaN/Inf")
    x = arrays["X"]
    u = arrays["U"]
    if x.ndim != 2:
        issues.append(f"v12_episode.npz X 不是二维（shape={x.shape}）")
    if u.ndim != 2:
        issues.append(f"v12_episode.npz U 不是二维（shape={u.shape}）")
    if x.ndim != 2 or u.ndim != 2:
        return issues
    if u.shape[1] > 0 and x.shape[1] != 2 * u.shape[1]:
        issues.append(
            "v12_episode.npz X 列数应等于 2×U 列数"
            f"（X={x.shape}，U={u.shape}）")
    if x.shape[0] != u.shape[0] + 1:
        issues.append(
            "v12_episode.npz X 行数应等于 U 行数+1"
            f"（X={x.shape}，U={u.shape}）")
    joint_names = payload.get("joint_names") if isinstance(payload, Mapping) else None
    if (isinstance(joint_names, Sequence) and not isinstance(joint_names, str)
            and len(joint_names) > 0):
        if u.shape[1] != len(joint_names):
            issues.append(
                "v12_episode.npz U 列数与 joint_names 不符"
                f"（U={u.shape}，joint_names={len(joint_names)}）")
        if x.shape[1] != 2 * len(joint_names):
            issues.append(
                "v12_episode.npz X 列数与 joint_names 不符"
                f"（X={x.shape}，joint_names={len(joint_names)}）")
    expected_rows = u.shape[0] + 1
    for name in ("ball_pos", "ball_vel", "racket_pos"):
        array = arrays[name]
        if array.shape != (expected_rows, 3):
            issues.append(
                f"v12_episode.npz {name} 形状应为 {(expected_rows, 3)}"
                f"（实为 {array.shape}）")
    time_array = arrays["time"]
    if time_array.shape != (expected_rows,):
        issues.append(
            "v12_episode.npz time 形状应为 "
            f"{(expected_rows,)}（实为 {time_array.shape}）")
    return issues


def _r1d_npz_issues(path: Path) -> list[str]:
    """校验旧 R1D 兼容 NPZ 可加载、非空、X/U 行关系与 finite（R02）。

    Args:
        path: ``r1d_episode.npz`` 绝对路径。

    Returns:
        问题描述列表（空表示通过）。
    """
    arrays, issues = _load_npz(path)
    if arrays is None:
        return issues
    missing = [name for name in ("X", "U") if name not in arrays]
    if missing:
        return issues + [f"r1d_episode.npz 缺少数组 {', '.join(missing)}"]
    x = arrays["X"]
    u = arrays["U"]
    for name in ("X", "U"):
        array = arrays[name]
        if array.size == 0:
            issues.append(f"r1d_episode.npz 数组 {name} 为空")
        if not np.issubdtype(array.dtype, np.number):
            issues.append(
                f"r1d_episode.npz 数组 {name} dtype 非数值（{array.dtype}）")
            continue
        if not np.all(np.isfinite(array)):
            issues.append(f"r1d_episode.npz 数组 {name} 含 NaN/Inf")
    if x.ndim != 2:
        issues.append(f"r1d_episode.npz X 不是二维（shape={x.shape}）")
    if u.ndim != 2:
        issues.append(f"r1d_episode.npz U 不是二维（shape={u.shape}）")
    if x.ndim != 2 or u.ndim != 2:
        return issues
    if x.shape[0] != u.shape[0] + 1:
        issues.append(
            "r1d_episode.npz X 行数应等于 U 行数+1"
            f"（X={x.shape}，U={u.shape}）")
    if u.shape[1] > 0 and x.shape[1] != 2 * u.shape[1]:
        issues.append(
            "r1d_episode.npz X 列数应等于 2×U 列数"
            f"（X={x.shape}，U={u.shape}）")
    expected_rows = u.shape[0] + 1
    for name, expected in (
            ("ball_pos", (expected_rows, 3)),
            ("ball_vel", (expected_rows, 3)),
            ("t", (expected_rows,))):
        if name not in arrays:
            continue
        if arrays[name].shape != expected:
            issues.append(
                f"r1d_episode.npz {name} 形状应为 {expected}"
                f"（实为 {arrays[name].shape}）")
    return issues


def _log_issues(path: Path, task: Mapping[str, Any]) -> list[str]:
    """校验日志非空且带 ``# config_id:`` 头部（R02）。

    Args:
        path: 已确认存在的日志文件。
        task: manifest 任务（提供 ``config_id``）。

    Returns:
        问题描述列表（空表示通过）。
    """
    try:
        if path.stat().st_size <= 0:
            return [f"日志为空 {path.as_posix()}"]
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return [f"日志无法读取 {path.as_posix()}: {exc}"]
    config_id = str(task.get("config_id") or "").strip()
    if config_id and f"# config_id: {config_id}" not in text:
        return [(
            "日志缺少 config_id 头部"
            f"（期望 '# config_id: {config_id}'，文件 {path.as_posix()}）")]
    return []


def _artifact_issues(
    task: Mapping[str, Any],
    artifact_dir: Path,
    row: Mapping[str, Any],
    run_root: Path,
) -> list[str]:
    """检查单个已完成任务的产物/日志是否齐全可解析（M09）。

    Args:
        task: manifest 任务（提供 ``expected_outputs``）。
        artifact_dir: 该任务的产物目录（绝对）。
        row: runs.csv 行（提供真实 ``log_path``）。
        run_root: 本次 output 根目录。

    Returns:
        问题描述列表（空表示通过）。
    """
    issues: list[str] = []
    expected = [str(item) for item in task.get("expected_outputs") or []]
    if not expected:
        issues.append("manifest 任务缺少 expected_outputs，无法校验产物")
    v12_payload: Mapping[str, Any] | None = None
    for pattern in expected:
        matches = _expected_output_matches(artifact_dir, pattern)
        if not matches:
            issues.append(f"缺少产物 {pattern}（目录 {artifact_dir.as_posix()}）")
            continue
        name = Path(pattern).name
        if name == V12_EPISODE_NAME:
            v12_payload, payload_issues = _v12_payload_issues(
                matches[0], task, row)
            issues.extend(payload_issues)
        elif name == V12_EPISODE_NPZ_NAME:
            issues.extend(_v12_npz_issues(matches[0], v12_payload))
        elif name == R1D_EPISODE_NPZ_NAME:
            issues.extend(_r1d_npz_issues(matches[0]))
    # 整树搬移后 runs.csv 中的绝对日志路径仍指向旧根；优先使用本次 manifest
    # 计划路径（相对当前 run_root 解析），仅当它不存在时才回退旧记录。
    task_log = _row_path(run_root, task.get("log_path"))
    row_log = _row_path(run_root, row.get("log_path"))
    if task_log is not None and task_log.is_file():
        log_path = task_log
    elif row_log is not None and row_log.is_file():
        log_path = row_log
    else:
        log_path = task_log or row_log
    if log_path is None:
        issues.append("缺少 log_path（runs.csv 与 manifest 均为空）")
    elif not log_path.is_file():
        issues.append(f"日志不存在 {log_path.as_posix()}")
    else:
        issues.extend(_log_issues(log_path, task))
    return issues


def _validate_completed_artifacts(
    resolved: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    run_root: Path,
) -> list[str]:
    """校验续跑跳过的已完成任务产物是否真实完好（M09）。

    ``batch_runner`` 只按 CSV 的 ``error``/``timed_out`` 判定完成：产物目录被
    移走、JSON 缺失/损坏/身份不符时仍会 ``launched=0`` 并以 0 退出，论文数据
    因此静默缺失。这里对每条完成记录按 manifest 的 ``expected_outputs`` 与
    ``v12_episode.json`` 身份做独立校验，返回按任务汇总的问题描述。

    Args:
        resolved: ``resolve_manifest`` 的结果（用于取任务期望产物）。
        records: ``_collect_runs`` 返回的原始记录（绝对路径）。
        run_root: 本次 output 根目录。

    Returns:
        每个存在问题的任务一条描述；空列表表示全部通过（含空 records）。
    """
    if not records:
        return []
    tasks = _resolved_task_index(resolved)
    problems: list[str] = []
    for row in records:
        if not _is_completed_record(row):
            continue
        key = _record_key(row)
        if key is None:
            continue
        task = tasks.get(key)
        if task is None:
            # 不属于本次 manifest 的历史行（配置已变更）不阻塞本次运行。
            continue
        # M06 交叉：整树搬移后 runs.csv 的 artifact_dir 绝对路径指向旧根。
        # 优先用本次 manifest 的计划目录（相对当前 run_root），不存在再回退
        # 旧记录；两者都不存在时以计划目录给出恢复指引。
        task_dir = _row_path(run_root, task.get("artifact_dir"))
        row_dir = _row_path(run_root, row.get("artifact_dir"))
        if task_dir is not None and task_dir.is_dir():
            artifact_dir = task_dir
        elif row_dir is not None and row_dir.is_dir():
            artifact_dir = row_dir
        else:
            artifact_dir = task_dir or row_dir
        if artifact_dir is None:
            issues = ["runs.csv 行缺少 artifact_dir"]
        else:
            issues = _artifact_issues(task, artifact_dir, row, run_root)
        if not issues:
            continue
        problems.append(
            f"{key[0]} / config_id={key[1][:12]} / seed={key[2]} "
            f"artifact={artifact_dir.as_posix() if artifact_dir else '<空>'}\n"
            + "\n".join(f"    - {issue}" for issue in issues))
    return problems


def _print_artifact_problems(problems: Sequence[str]) -> None:
    """打印续跑产物校验失败明细与恢复指引（M09）。

    Args:
        problems: :func:`_validate_completed_artifacts` 返回的问题描述。
    """
    if not problems:
        return
    print(f"[r1d-smoke] 续跑产物校验失败：{len(problems)} 条已完成任务的产物"
          f"缺失或损坏（runs.csv 仍保留这些行）", file=sys.stderr)
    for problem in problems:
        print(f"[r1d-smoke] ✗ {problem}", file=sys.stderr)
    print("[r1d-smoke] 恢复指引：把移走的产物目录/文件放回上述 artifact 路径"
          "（或恢复缺失日志）后重跑；若确认该任务作废，删除 runs.csv 中对应行"
          "并用 --no-resume 重跑（旧行请另存保留，便于追溯）。",
          file=sys.stderr)
