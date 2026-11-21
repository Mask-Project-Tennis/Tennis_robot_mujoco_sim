"""产物路径与 metrics 辅助：v12_episode.json 指标提取、可搬移路径、NPZ 读取。
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .manifest import ARTIFACT_FIELDS
from .tasks import _run_root_relative


def _artifact_metrics(artifact_dir: str | Path) -> dict[str, Any]:
    """从单次 R1D 产物 ``v12_episode.json`` 提取 runs.csv 追加列。"""
    metrics: dict[str, Any] = {key: "" for key in ARTIFACT_FIELDS}
    if not artifact_dir:
        metrics["artifact_error"] = "missing_artifact_dir"
        return metrics
    path = Path(artifact_dir) / "v12_episode.json"
    if not path.is_file():
        metrics["artifact_error"] = "missing_v12_episode"
        return metrics
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        metrics["artifact_error"] = f"invalid_v12_episode: {exc}"
        return metrics
    hit = payload.get("hit") or {}
    episode = payload.get("episode") or {}
    metrics.update({
        "physical_hit": hit.get("physical_hit", ""),
        "physical_hit_type": hit.get("physical_hit_type", ""),
        "physical_hit_step": hit.get("physical_hit_step", ""),
        "v_racket_at_hit": hit.get("v_racket_at_hit", ""),
        "ball_out_speed": hit.get("ball_out_speed", ""),
        "ball_out_vy": hit.get("ball_out_vy", ""),
        "contact_event_count": len(hit.get("contact_events") or []),
        "artifact_error": "",
    })
    # 结果行缺失但产物已写的字段由 artifact 补齐（保留 batch CSV 已有值）
    if not metrics["physical_hit_type"]:
        metrics["physical_hit_type"] = hit.get("legacy_hit_type", "")
    if "total_steps" in metrics:
        metrics["total_steps"] = episode.get("total_steps", "")
    return metrics


def _is_absolute_path_text(value: str) -> bool:
    """判断字符串是否像绝对路径（兼顾 Windows 盘符与 posix 根）。

    Args:
        value: 待判断字符串。

    Returns:
        是绝对路径文本时为 True。
    """
    return Path(value).is_absolute() or value.startswith("/")


def _config_json_with_output(text: Any, artifact_dir: Path | None) -> str:
    """把 ``config_json`` 的 ``--output`` 指向解析后的产物目录（R01）。

    整树搬移后旧行的 ``config_json`` 仍记录搬移前的绝对 ``--output``；若只
    相对化 run_root 内路径，该字段会永久指向已不存在的旧根。这里按 R01
    解析出的真实 ``artifact_dir`` 重写，随后 ``_portable_run_row`` 会把它
    相对化到当前 run_root，使 runs.csv 不再残留旧绝对 artifact 路径。

    Args:
        text: 原 config_json 文本（可能为空或损坏）。
        artifact_dir: 解析后的产物目录（绝对）；None 表示不改写。

    Returns:
        重写后的 JSON 文本；无法解析或无需改写时原样返回。
    """
    raw = str(text or "")
    if artifact_dir is None or not raw.strip():
        return raw
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    if not isinstance(payload, dict) or "--output" not in payload:
        return raw
    payload["--output"] = str(artifact_dir)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _portable_config_json(text: str, run_root: Path) -> str:
    """把 ``config_json`` 中位于 run_root 内的绝对路径改写成相对路径（M06）。

    ``runs.csv`` 整树搬移后仍要能读懂每行的真实命令；``--output``、
    ``--runtime-config``（快照）等 run_root 内路径写相对值，run_root 外
    路径（数据根/仓库路径）保持绝对。无法解析的 JSON 原样返回，避免
    为迁移而丢弃历史行。

    Args:
        text: batch_runner 写出的配置 JSON 文本。
        run_root: 本次 output 根目录（绝对路径）。

    Returns:
        改写后的 JSON 文本（按键排序、不转义中文）。
    """
    if not text.strip():
        return text
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text
    if not isinstance(payload, dict):
        return text

    def _rewrite(value: Any) -> Any:
        """递归改写字符串叶子（列表/字典保持结构）。"""
        if isinstance(value, str):
            if not _is_absolute_path_text(value):
                return value
            return _run_root_relative(run_root, value)
        if isinstance(value, list):
            return [_rewrite(item) for item in value]
        if isinstance(value, dict):
            return {str(key): _rewrite(item) for key, item in value.items()}
        return value

    return json.dumps(_rewrite(payload), ensure_ascii=False, sort_keys=True)


def _portable_run_row(
    row: Mapping[str, Any], run_root: Path,
) -> dict[str, Any]:
    """生成写入 ``runs.csv`` 的可搬移行（records 仍保留原绝对路径）。

    Args:
        row: ``_collect_runs`` 聚合出的原始记录（路径为 batch CSV 原值）。
        run_root: 本次 output 根目录。

    Returns:
        浅拷贝并相对化 ``artifact_dir``/``log_path``/``config_json`` 的行。
    """
    portable = dict(row)
    for field in ("artifact_dir", "log_path"):
        value = str(portable.get(field) or "").strip()
        if value:
            portable[field] = _run_root_relative(run_root, value)
    config_json = str(portable.get("config_json") or "")
    if config_json.strip():
        portable["config_json"] = _portable_config_json(config_json, run_root)
    return portable


def _resolve_run_artifact_dir(
    row: Mapping[str, Any],
    task: Mapping[str, Any] | None,
    run_root: Path,
) -> Path | None:
    """解析已完成行在当前 run_root 下的真实产物目录（R01）。

    整树搬移后 ``batch/results.csv`` 里仍是旧绝对路径，直接按它提取
    指标会把有效行清空。这里先看行内路径是否仍存在（换 ``--output`` 续跑
    时仍指向旧产物），否则按 ``(experiment, config_id, seed)`` 对应的
    manifest 任务，用它在当前 run_root 下的计划 ``artifact_dir`` 定位。

    Args:
        row: batch results.csv 行。
        task: 对应 manifest 任务（历史行可能为 None）。
        run_root: 本次 output 根目录。

    Returns:
        真实存在的目录；都不存在时返回计划目录（供提取器显式报缺失）。
    """
    from .resume import _row_path  # 延迟导入：避免 artifacts ↔ resume 循环依赖
    row_dir = _row_path(run_root, row.get("artifact_dir"))
    if row_dir is not None and row_dir.is_dir():
        return row_dir
    task_dir = _row_path(run_root, task.get("artifact_dir")) if task else None
    if task_dir is not None and task_dir.is_dir():
        return task_dir
    return task_dir or row_dir


def _resolve_run_log_path(
    row: Mapping[str, Any],
    task: Mapping[str, Any] | None,
    run_root: Path,
) -> Path | None:
    """解析已完成行在当前 run_root 下的真实日志路径（R01）。

    优先级与产物目录一致：行内路径存在（旧 output 续跑）→ 任务计划路径
    （整树搬移后的新位置）→ 计划路径（供校验显式报缺失）。

    Args:
        row: batch results.csv 行。
        task: 对应 manifest 任务（历史行可能为 None）。
        run_root: 本次 output 根目录。

    Returns:
        日志绝对路径；行内与计划均为空时返回 None。
    """
    from .resume import _row_path  # 延迟导入：避免 artifacts ↔ resume 循环依赖
    row_log = _row_path(run_root, row.get("log_path"))
    if row_log is not None and row_log.is_file():
        return row_log
    task_log = _row_path(run_root, task.get("log_path")) if task else None
    if task_log is not None and task_log.is_file():
        return task_log
    return task_log or row_log


def _load_npz(path: Path) -> tuple[dict[str, np.ndarray] | None, list[str]]:
    """实际解包 NPZ（``allow_pickle=False``），截断/损坏文件必须报错。

    Args:
        path: NPZ 绝对路径。

    Returns:
        ``(arrays, issues)``；无法加载时 arrays 为 None。
    """
    try:
        with np.load(path, allow_pickle=False) as data:
            arrays = {name: np.asarray(data[name]) for name in data.files}
    except (OSError, ValueError, EOFError, KeyError, zipfile.BadZipFile) as exc:
        return None, [f"{path.name} 无法加载: {exc}"]
    if not arrays:
        return None, [f"{path.name} 不含任何数组"]
    return arrays, []


def _npz_scalar_text(value: np.ndarray) -> str:
    """把 savez 写入的 0 维字符串数组还原为文本（兼容 bytes）。"""
    array = np.asarray(value)
    if array.size != 1:
        return ""
    item = array.reshape(-1)[0].item()
    if isinstance(item, bytes):
        return item.decode("utf-8", "replace")
    return str(item)
