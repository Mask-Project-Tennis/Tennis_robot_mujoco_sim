#!/usr/bin/env python
"""R1D 论文 smoke 单 case 统计提取（T8b，T8a manifest ``extract_argv`` 指向本脚本）。

用途
----
把 R1D 共享 CLI 的一个 ``(experiment, case)`` 在若干 seed 上的产物，汇总成 T8c 可直接
消费的稳定 JSON。入口参数与 T8a ``run_r1d_paper_smoke.py`` 解析后的 ``extract_argv``
一致::

    python scripts/extract/extract_r1d_smoke_stats.py \\
        --experiment exp15_speed_v2 --case speed7 \\
        --data-root results/r1d_v12_parity/t8a/plumbing/batch \\
        --output results/r1d_v12_parity/t8b/stats

数据定位与执行模式（F03 / F04）
----------------------------------
- ``--catalog PATH``：从 ``catalog.json`` 解析 run_root / data_root / manifest
  （顶层路径相对 catalog 文件目录，行内 artifact_dir 相对 run_root）；显式坏 catalog
  直接退出码 2，绝不回退旧目录推断。
- ``--mode torque|position|both``：同一实验/案例出现多种执行模式时未显式选择即报错；
  显式模式时产物写 ``<output>/<mode>/<experiment>/<case>.json``，未给模式保持旧布局
  ``<output>/<experiment>/<case>.json``。

口径（两种命中语义必须分开报告，禁止混算）
----------------------------------------
- ``physical``：``v12_episode.json`` 的 ``hit.physical_hit`` / ``hit.physical_hit_type``
  （首个物理拍面接触，拍面速度阈值 0.3 m/s；v2 schema）。
- ``legacy``：旧口径 ``hit_type == "active"``（``v12_episode.json`` 的
  ``hit.legacy_hit_type`` / ``r1d_episode.json`` 的 ``hit.type`` / ``runs.csv`` 的
  ``hit_type``）；``passive`` 接触不算 active。

缺口显式化（禁止伪造 0.0 / 禁止删失败行）
----------------------------------------
- 缺数据、缺度量列、全 miss、无共同 seed、超时、run error、no-verdict 全部保留在
  ``per_seed``；不可估计时 ``*_active_rate`` 为 ``null``，并给出 ``unavailable_reason``
  与 ``unavailable_reasons.<semantics>``，同时在 ``*_active_denominator`` 记录真实分母。
- ``pos_error`` / ``min_dist`` 再小也不算命中：只认显式 hit 字段。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "r1d_smoke_case_stats_v1"
JSON_ARTIFACTS = ("v12_episode.json", "r1d_episode.json")
SEED_DIR_RE = re.compile(r"^seed(?P<seed>-?\d+)$")
HIT_SEMANTICS = {
    "physical": "v12_episode_v2: hit.physical_hit / hit.physical_hit_type（首个物理拍面接触，face_speed>0.3 m/s）",
    "legacy": "legacy: hit_type=='active'（passive 接触不计 active；来源 v12 hit.legacy_hit_type / r1d hit.type / runs.csv hit_type）",
}
STATUSES = ("ok", "error", "timeout", "no_verdict", "missing")
VALID_MODES: tuple[str, ...] = ("torque", "position")
MODE_CHOICES: tuple[str, ...] = (*VALID_MODES, "both")


class ExtractError(RuntimeError):
    """提取链路的可预期错误（CLI 以退出码 2 报告，绝不静默回退）。"""


def _load_catalog(catalog_path: Path) -> Any:
    """加载 ``catalog.json``（``scripts.exp.run_catalog`` 是契约的唯一事实来源）。"""
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from scripts.exp import run_catalog

    try:
        return run_catalog.load_catalog(Path(catalog_path))
    except run_catalog.CatalogError as exc:
        raise ExtractError(f"catalog 加载失败（{catalog_path}）：{exc}") from exc


# ---------------------------------------------------------------------------
# 通用小工具
# ---------------------------------------------------------------------------


def _text_or_none(value: Any) -> str | None:
    """把空字符串/None 归一为 None，其余转 ``str``。"""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_or_none(value: Any) -> bool | None:
    """把 True/False/"true"/"1"/"yes" 解析为布尔；无法判定返回 None。"""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 1:
            return True
        if value == 0:
            return False
        return None
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _float_or_none(value: Any) -> float | None:
    """把数值/数字串解析为 float；空值或非法值返回 None。"""
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _int_or_none(value: Any) -> int | None:
    """把数值/数字串解析为 int；空值或非法值返回 None。"""
    number = _float_or_none(value)
    return None if number is None else int(number)


def _wilson(hits: int, n: int, z: float = 1.96) -> list[float] | None:
    """Wilson 95% 置信区间（百分数 pp；与 paired_stats.wilson 语义一致）。"""
    if n <= 0:
        return None
    p = hits / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return [max(0.0, (centre - half) * 100.0), min(100.0, (centre + half) * 100.0)]


def _read_json(path: Path) -> dict[str, Any] | None:
    """读 JSON 文件；不存在或解析失败返回 None（失败由调用方记入 errors）。"""
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _backend_value(payload: dict[str, Any] | None) -> Any:
    """返回产物中的 backend 原值（dict 保留结构；字符串归一化）。"""
    if not payload:
        return None
    value = payload.get("backend")
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    return _text_or_none(value)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """读 CSV 为 dict 行列表；文件不存在返回空列表。"""
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def _artifact_dir_key(artifact_dir: str) -> tuple[str, str, int] | None:
    """从 ``.../<experiment>/<case>/seed<N>`` 路径解析 (experiment, case, seed)。"""
    parts = Path(artifact_dir).parts
    for index, part in enumerate(parts):
        match = SEED_DIR_RE.match(part)
        if match and index >= 2:
            return parts[index - 2], parts[index - 1], int(match.group("seed"))
    return None


def _row_artifact_dir(row: dict[str, str]) -> str:
    """取一行 CSV 记录的产物目录：``artifact_dir`` 列优先，其次 ``config_json`` 的 ``--output``。"""
    explicit = _text_or_none(row.get("artifact_dir"))
    if explicit:
        return explicit
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        return ""
    if isinstance(cfg, dict):
        output = cfg.get("--output")
        if isinstance(output, str) and output.strip():
            return output
    return ""


def _row_case_key(row: dict[str, str]) -> tuple[str, str, int] | None:
    """从一行 CSV 解析 (experiment, case, seed)；无法解析返回 None。"""
    return _artifact_dir_key(_row_artifact_dir(row))


# ---------------------------------------------------------------------------
# manifest / 数据根定位
# ---------------------------------------------------------------------------


def _manifest_tasks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """从 resolved manifest 或 source manifest 中取出任务列表（兼容两种 schema）。"""
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    experiments = payload.get("experiments")
    rows: list[dict[str, Any]] = []
    if isinstance(experiments, list):
        for exp in experiments:
            if not isinstance(exp, dict):
                continue
            exp_id = exp.get("experiment") or exp.get("id")
            for case in exp.get("cases") or []:
                if not isinstance(case, dict):
                    continue
                for seed in case.get("seeds") or []:
                    rows.append(
                        {
                            "experiment": exp_id,
                            "case_id": case.get("case") or case.get("case_id") or case.get("id"),
                            "seed": seed,
                            "mode": case.get("mode"),
                            "argv": case.get("argv"),
                            "identity": payload.get("identity"),
                        }
                    )
    return rows


def load_manifest(path: Path, experiment: str, case: str) -> dict[str, Any]:
    """读出该 (experiment, case) 的预期 seed、mode、identity 与 argv。

    Args:
        path: manifest JSON 路径。
        experiment: 实验目录名（如 ``exp15_speed_v2``）。
        case: case id（如 ``speed7``）。

    Returns:
        含 ``path`` / ``exists`` / ``seeds`` / ``argv_by_seed`` / ``mode_by_seed`` / ``mode`` /
        ``identity`` 的字典；manifest 缺失时 ``exists=False``。
    """
    payload = _read_json(path) if path else None
    result: dict[str, Any] = {
        "path": str(path) if path else None,
        "exists": payload is not None,
        "seeds": [],
        "argv_by_seed": {},
        "mode_by_seed": {},
        "mode": None,
        "identity": None,
    }
    if payload is None:
        return result
    seeds: list[int] = []
    argv_by_seed: dict[int, dict[str, Any]] = {}
    mode_by_seed: dict[int, str] = {}
    modes: list[str] = []
    identity = None
    for task in _manifest_tasks(payload):
        if str(task.get("experiment")) != experiment:
            continue
        task_case = task.get("case_id") or task.get("case")
        if task_case is not None and str(task_case) != case:
            continue
        seed = _int_or_none(task.get("seed"))
        if seed is None:
            continue
        seeds.append(seed)
        argv = task.get("argv")
        if isinstance(argv, dict):
            argv_by_seed[seed] = argv
        mode = _text_or_none(task.get("mode"))
        if mode:
            modes.append(mode)
            mode_by_seed[seed] = mode.strip().lower()
        if identity is None and isinstance(task.get("identity"), dict):
            identity = task["identity"]
    result["seeds"] = sorted(set(seeds))
    result["argv_by_seed"] = argv_by_seed
    result["mode_by_seed"] = mode_by_seed
    result["mode"] = modes[0] if modes else None
    result["identity"] = identity or payload.get("identity")
    return result


def _row_matches(row: dict[str, str], experiment: str, case: str, argv: dict[str, Any] | None) -> bool:
    """判断一行 CSV 是否属于目标 (experiment, case)。

    优先用 ``artifact_dir`` 的 ``<experiment>/<case>/seed<N>`` 结构；没有该列时退回
    与 manifest ``argv`` 的 config 对比（忽略 ``--output`` / ``--seed`` 等运行期路径），
    再退回接受（旧日志/无 artifact_dir 的 CSV，由 ``_collect_rows`` 统一处理歧义）。
    """
    if _text_or_none(row.get("experiment")) not in (None, experiment):
        return False
    key = _row_case_key(row)
    if key is not None:
        return key[0] == experiment and key[1] == case
    if not argv:
        return True
    try:
        row_cfg = json.loads(row.get("config_json") or "{}")
    except json.JSONDecodeError:
        return True
    if not isinstance(row_cfg, dict):
        return True
    for name, value in argv.items():
        if name in {"--output", "--seed", "_timeout"}:
            continue
        if row_cfg.get(name) != value:
            return False
    return True


def _collect_rows(
    run_root: Path,
    data_root: Path,
    experiment: str,
    case: str,
    argv_by_seed: dict[int, dict[str, Any]],
    *,
    mode: str | None = None,
) -> tuple[dict[int, dict[str, str]], dict[str, Any]]:
    """读取 runs.csv 与每个实验 results.csv，仅保留属于该 case 的行（runs.csv 优先）。

    若某个 CSV 里没有任何一行能通过 ``artifact_dir`` 归属到该 case（老日志/改名行），
    则视为该 case CSV 全部兼容；否则只保留能识别的行，无法识别的行计入
    ``sources.<name>.skipped_ambiguous``，绝不静默串档。显式 ``mode`` 时丢弃模式
    明确不同的行（模式未知的行保留，交给后续身份校验），丢弃数计入
    ``sources.<name>.skipped_mode``。
    """
    runs_path = run_root / "runs.csv"
    exp_csv = data_root / experiment / "results.csv"
    rows: dict[int, dict[str, str]] = {}
    sources: dict[str, Any] = {}
    for name, path in (("runs_csv", runs_path), ("experiment_csv", exp_csv)):
        raw = _read_csv(path)
        keyed = [row for row in raw if _row_case_key(row) is not None]
        multi_case = bool(keyed)
        matched = 0
        skipped_ambiguous = 0
        skipped_mode = 0
        for row in raw:
            seed = _int_or_none(row.get("seed"))
            if seed is None:
                continue
            row_mode = (row.get("mode") or "").strip().lower()
            if mode is not None and row_mode and row_mode != mode:
                skipped_mode += 1
                continue
            if multi_case and _row_case_key(row) is None:
                skipped_ambiguous += 1
                continue
            if not _row_matches(row, experiment, case, argv_by_seed.get(seed)):
                continue
            matched += 1
            if seed not in rows:
                rows[seed] = row
        sources[name] = {
            "path": str(path),
            "exists": path.is_file(),
            "rows": len(raw),
            "matched_rows": matched,
            "skipped_ambiguous": skipped_ambiguous,
            "skipped_mode": skipped_mode,
        }
    return rows, sources


def _row_artifact_path(row: dict[str, str], run_root: Path) -> Path | None:
    """把行内 artifact_dir（相对 run_root 或绝对）解析为真实目录。"""
    value = _row_artifact_dir(row)
    if not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = run_root / path
    return path


def _artifact_mode(artifact_dir: Path) -> str | None:
    """读产物 JSON 声明的执行模式，用于拒绝把另一模式的产物算进本模式。"""
    for name in JSON_ARTIFACTS:
        payload = _read_json(artifact_dir / name)
        if payload:
            value = _text_or_none(payload.get("mode"))
            if value:
                return value.strip().lower()
    return None


def _find_artifacts(run_root: Path, experiment: str, case: str) -> dict[int, Path]:
    """定位 ``<run_root>/artifacts/<experiment>/<case>/seed<N>/`` 目录。"""
    root = run_root / "artifacts" / experiment / case
    found: dict[int, Path] = {}
    if not root.is_dir():
        return found
    for child in sorted(root.iterdir()):
        match = SEED_DIR_RE.match(child.name)
        if match and child.is_dir():
            found[int(match.group("seed"))] = child
    return found


# ---------------------------------------------------------------------------
# 单 seed 判定
# ---------------------------------------------------------------------------


def _physical_from_dict(hit: dict[str, Any]) -> tuple[bool | None, str | None, int | None]:
    """从 v12 ``hit`` 段取 physical 口径三元组。"""
    return (
        _bool_or_none(hit.get("physical_hit")),
        _text_or_none(hit.get("physical_hit_type")),
        _int_or_none(hit.get("physical_hit_step")),
    )


def _legacy_type_from_r1d(payload: dict[str, Any] | None) -> tuple[bool | None, str | None]:
    """从 ``r1d_episode.json`` 的 ``hit.type`` 推 legacy 布尔与类型。"""
    if not payload:
        return None, None
    hit = payload.get("hit")
    if not isinstance(hit, dict):
        return None, None
    hit_type = _text_or_none(hit.get("type"))
    if hit_type is None:
        return None, None
    return hit_type in {"active", "passive"}, hit_type


def _parse_artifact(artifact_dir: Path) -> dict[str, Any]:
    """读 seed 目录下的 v2 / r1d episode JSON，返回判定所需字段与错误列表。"""
    parsed: dict[str, Any] = {
        "files": [],
        "errors": [],
        "v12": None,
        "r1d": None,
    }
    for name in JSON_ARTIFACTS:
        path = artifact_dir / name
        if not path.is_file():
            continue
        parsed["files"].append(name)
        payload = _read_json(path)
        if payload is None:
            parsed["errors"].append(f"{name}: JSON 解析失败")
            continue
        if name == "v12_episode.json":
            parsed["v12"] = payload
        else:
            parsed["r1d"] = payload
    return parsed


def _seed_entry(seed: int, row: dict[str, str] | None, artifact_dir: Path | None) -> dict[str, Any]:
    """构造单个 seed 的判定记录（保留失败/超时/缺数据行）。"""
    row = row or {}
    parsed = _parse_artifact(artifact_dir) if artifact_dir is not None else {"files": [], "errors": [], "v12": None, "r1d": None}
    v12 = parsed["v12"] or {}
    hit12 = v12.get("hit") if isinstance(v12.get("hit"), dict) else {}
    notes: list[str] = []

    # physical 口径：artifact 优先，其次 CSV 列
    physical_hit, physical_type, physical_step = _physical_from_dict(hit12)
    if physical_hit is None and physical_type is None:
        physical_hit = _bool_or_none(row.get("physical_hit"))
        physical_type = _text_or_none(row.get("physical_hit_type"))
        physical_step = _int_or_none(row.get("physical_hit_step"))

    # legacy 口径：v12 legacy 字段优先，其次 r1d hit.type，最后 CSV hit_type
    legacy_hit = _bool_or_none(hit12.get("legacy_hit"))
    legacy_type = _text_or_none(hit12.get("legacy_hit_type"))
    if legacy_type is None and legacy_hit is None:
        legacy_hit, legacy_type = _legacy_type_from_r1d(parsed["r1d"])
    if legacy_type is None:
        legacy_type = _text_or_none(row.get("hit_type"))
    if legacy_hit is None and legacy_type is not None:
        legacy_hit = legacy_type in {"active", "passive"}
    if legacy_hit is None and row.get("hit"):
        legacy_hit = _bool_or_none(row.get("hit"))
        if legacy_hit is not None and legacy_type is None:
            legacy_type = "active" if legacy_hit else "miss"

    timed_out = _bool_or_none(row.get("timed_out")) is True
    error_text = _text_or_none(row.get("error")) or _text_or_none(row.get("artifact_error"))
    artifact_error = "; ".join(parsed["errors"]) if parsed["errors"] else None
    failure_reason = _text_or_none(row.get("failure_reason")) or _text_or_none(v12.get("failure_reason"))

    has_verdict = physical_type is not None or physical_hit is not None or legacy_type is not None
    if timed_out:
        status = "timeout"
    elif error_text or artifact_error:
        status = "error"
    elif has_verdict:
        status = "ok"
    elif artifact_dir is not None and artifact_dir.is_dir():
        status = "no_verdict"
    else:
        status = "missing"
        notes.append("artifact 目录缺失" if artifact_dir is not None else "未找到产物目录/CSV 行")

    backend = _backend_value(v12)
    if backend is None and parsed["r1d"]:
        backend = _backend_value(parsed["r1d"])
    total_steps = None
    episode = v12.get("episode")
    if isinstance(episode, dict):
        total_steps = _int_or_none(episode.get("total_steps"))
    contact_events = hit12.get("contact_events")
    contact_count = _int_or_none(row.get("contact_event_count"))
    if contact_count is None and isinstance(contact_events, list):
        contact_count = len(contact_events)

    return {
        "seed": seed,
        "status": status,
        "valid": status == "ok",
        "source": "artifact" if parsed["files"] else ("runs_csv" if row else None),
        "artifact_dir": str(artifact_dir) if artifact_dir is not None else "",
        "artifact_files": parsed["files"],
        "physical_hit": physical_hit,
        "physical_hit_type": physical_type,
        "physical_hit_step": physical_step,
        "legacy_hit": legacy_hit,
        "legacy_hit_type": legacy_type,
        "backend": backend,
        "failure_reason": failure_reason,
        "timed_out": timed_out,
        "error": error_text or artifact_error,
        "exit_code": _int_or_none(row.get("exit_code")),
        "pos_error": _float_or_none(hit12.get("pos_error")) if hit12.get("pos_error") is not None else _float_or_none(row.get("pos_error")),
        "v_racket_at_hit": _float_or_none(hit12.get("v_racket_at_hit")) if hit12.get("v_racket_at_hit") is not None else _float_or_none(row.get("v_racket_at_hit")),
        "ball_out_speed": _float_or_none(hit12.get("ball_out_speed")) if hit12.get("ball_out_speed") is not None else _float_or_none(row.get("ball_out_speed")),
        "ball_out_vy": _float_or_none(hit12.get("ball_out_vy")) if hit12.get("ball_out_vy") is not None else _float_or_none(row.get("ball_out_vy")),
        "contact_event_count": contact_count,
        "total_steps": total_steps,
        "wall_time_s": _float_or_none(row.get("wall_time")),
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def _count_semantics(entries: list[dict[str, Any]], kind: str) -> dict[str, int]:
    """统计物理/legacy 口径的 active/passive/miss/分母。"""
    denominator = 0
    active = 0
    passive = 0
    miss = 0
    for entry in entries:
        hit = entry[f"{kind}_hit"]
        hit_type = entry[f"{kind}_hit_type"]
        if hit is None and hit_type is None:
            continue
        denominator += 1
        if hit_type == "active":
            active += 1
        elif hit_type == "passive":
            passive += 1
        elif hit_type == "miss":
            miss += 1
        elif hit is True:
            active += 1
        elif hit is False:
            miss += 1
    return {"denominator": denominator, "active": active, "passive": passive, "miss": miss}


def _semantics_reason(kind: str, stats: dict[str, int], entries: list[dict[str, Any]], no_common_seed: bool) -> str | None:
    """给出某口径为何不可估计的原因码；可估计返回 None。"""
    other = "legacy" if kind == "physical" else "physical"
    other_present = any(entry[f"{other}_hit"] is not None or entry[f"{other}_hit_type"] is not None for entry in entries)
    if not entries:
        return f"{kind}_missing_data"
    if no_common_seed:
        return "no_common_seed"
    if stats["denominator"] == 0:
        statuses = [entry["status"] for entry in entries]
        if other_present:
            return f"{kind}_metrics_unavailable"
        if statuses and all(status in {"timeout", "error"} for status in statuses):
            return "all_failed"
        if any(status == "no_verdict" for status in statuses):
            return "no_verdict"
        return f"{kind}_missing_data"
    if stats["active"] == 0:
        if stats["miss"] == stats["denominator"]:
            return "all_miss"
        return "no_active_verdict"
    return None


def aggregate(entries: list[dict[str, Any]], *, no_common_seed: bool = False) -> dict[str, Any]:
    """把 per-seed 记录聚合为 case 级统计（不可估计处一律 ``None`` + 原因）。"""
    attempted = len(entries)
    valid_entries = [entry for entry in entries if entry["valid"] and not no_common_seed]
    for entry in entries:
        entry["correlated"] = not no_common_seed
    physical = _count_semantics(valid_entries, "physical")
    legacy = _count_semantics(valid_entries, "legacy")
    status_counts = {status: sum(1 for entry in entries if entry["status"] == status) for status in STATUSES}
    physical_reason = _semantics_reason("physical", physical, entries, no_common_seed)
    legacy_reason = _semantics_reason("legacy", legacy, entries, no_common_seed)

    def _rate(stats: dict[str, int]) -> float | None:
        if stats["denominator"] == 0 or stats["active"] == 0:
            return None
        return 100.0 * stats["active"] / stats["denominator"]

    reasons: list[str] = []
    for reason in (physical_reason, legacy_reason):
        if reason and reason not in reasons:
            reasons.append(reason)

    return {
        "n_attempted": attempted,
        "n_valid": len(valid_entries),
        "n_physical_active": physical["active"],
        "n_physical_passive": physical["passive"],
        "n_physical_miss": physical["miss"],
        "n_legacy_active": legacy["active"],
        "n_legacy_passive": legacy["passive"],
        "n_legacy_miss": legacy["miss"],
        "n_active": physical["active"],
        "n_passive": physical["passive"],
        "n_miss": physical["miss"],
        "n_error": status_counts["error"],
        "n_timeout": status_counts["timeout"],
        "n_no_verdict": status_counts["no_verdict"],
        "n_missing": status_counts["missing"],
        "physical_active_rate": _rate(physical),
        "physical_active_denominator": physical["denominator"],
        "physical_ci95": _wilson(physical["active"], physical["denominator"]),
        "legacy_active_rate": _rate(legacy),
        "legacy_active_denominator": legacy["denominator"],
        "legacy_ci95": _wilson(legacy["active"], legacy["denominator"]),
        "unavailable_reasons": {"physical": physical_reason, "legacy": legacy_reason},
        "unavailable_reason": "; ".join(reasons) if reasons else None,
        "physical_hit_semantics_available": physical["denominator"] > 0,
        "legacy_hit_semantics_available": legacy["denominator"] > 0,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def default_output_dir(data_root: Path) -> Path:
    """默认输出目录：``<run_root>/stats``（data_root 形如 ``<run_root>/batch``）。"""
    return data_root.parent / "stats"


def _experiment_modes(run_root: Path, data_root: Path, experiment: str) -> list[str]:
    """列出该 experiment 在 runs.csv / results.csv 中声明的执行模式（去重排序）。

    用于回答“这个实验到底跑了哪些执行模式”：只要同一实验下同时出现 ``torque`` 与
    ``position``，未显式选择模式的提取就必须报错，禁止把两种模式合成一个样本池。
    """
    modes: set[str] = set()
    for path in (run_root / "runs.csv", data_root / experiment / "results.csv"):
        for row in _read_csv(path):
            key = _row_case_key(row)
            row_experiment = _text_or_none(row.get("experiment"))
            belongs = row_experiment == experiment or (key is not None and key[0] == experiment)
            if not belongs:
                continue
            value = (row.get("mode") or "").strip().lower()
            if value:
                modes.add(value)
    return sorted(modes)


def collect_case_stats(
    *,
    experiment: str,
    case: str,
    data_root: Path | None = None,
    manifest_path: Path | None = None,
    catalog_path: Path | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """收集并聚合一个 (experiment, case) 的统计 JSON 内容。

    显式 ``catalog_path`` 时以 catalog 声明的 ``run_root`` / ``data_root`` /
    ``manifest_path`` 为准（``data_root`` 显式传入可覆盖数据根）；坏 catalog 抛
    ``ExtractError``，绝不回退旧路径推断。``mode`` 为空且该实验同时含多种执行模式
    时抛 ``ExtractError``，要求调用方显式选择。

    Args:
        experiment: 实验目录名。
        case: case id。
        data_root: R1D 数据根（``<run_root>/batch``）或旧 ``experiment_data``。
        manifest_path: 可选 resolved/source manifest，用于补预期 seed/identity。
        catalog_path: 可选 ``catalog.json``（F03 契约的唯一路径来源）。
        mode: ``torque`` / ``position``；None 表示按数据推断（混合则报错）。

    Returns:
        符合 ``r1d_smoke_case_stats_v1`` 的字典（未写盘）。

    Raises:
        ExtractError: catalog 非法、模式混合未指定、或显式模式与数据冲突。
    """
    if mode is not None and mode not in VALID_MODES:
        raise ExtractError(f"未知 --mode {mode!r}；可选 {', '.join(MODE_CHOICES)}")
    catalog = _load_catalog(catalog_path) if catalog_path is not None else None
    if catalog is not None:
        root = Path(data_root) if data_root is not None else Path(catalog.data_root)
        run_root = Path(catalog.run_root)
        if manifest_path is None:
            manifest_path = Path(catalog.manifest_path)
        synthetic = bool(catalog.synthetic)
        catalog_text: str | None = str(catalog.catalog_path)
    else:
        root = Path(data_root) if data_root is not None else PROJECT / "experiment_data"
        run_root = root.parent
        synthetic = False
        catalog_text = None

    declared_modes = _experiment_modes(run_root, root, experiment)
    if mode is None and len(set(declared_modes) & set(VALID_MODES)) > 1:
        raise ExtractError(
            f"实验 {experiment} 同时含多种执行模式 {declared_modes}，"
            "必须显式 --mode torque / --mode position / --mode both，禁止合成样本池"
        )
    resolved_mode = mode
    if resolved_mode is None and len(declared_modes) == 1 and declared_modes[0] in VALID_MODES:
        resolved_mode = declared_modes[0]

    manifest = load_manifest(manifest_path, experiment, case) if manifest_path else {
        "path": None,
        "exists": False,
        "seeds": [],
        "argv_by_seed": {},
        "mode_by_seed": {},
        "mode": None,
        "identity": None,
    }
    rows_by_seed, sources = _collect_rows(
        run_root, root, experiment, case, manifest["argv_by_seed"], mode=resolved_mode
    )
    sources["manifest"] = {
        "path": manifest["path"],
        "exists": manifest["exists"],
        "expected_seeds": manifest["seeds"],
    }
    artifact_dirs = _find_artifacts(run_root, experiment, case)
    for seed, row in rows_by_seed.items():
        path = _row_artifact_path(row, run_root)
        if path is not None and path.is_dir():
            artifact_dirs[seed] = path
    if resolved_mode is not None:
        artifact_dirs = {
            seed: path
            for seed, path in artifact_dirs.items()
            if _artifact_mode(path) in (None, resolved_mode)
        }

    expected_seeds = set(manifest["seeds"])
    if resolved_mode is not None and manifest.get("mode_by_seed"):
        mode_by_seed: dict[int, str] = manifest["mode_by_seed"]
        expected_seeds = {
            seed for seed in expected_seeds
            if mode_by_seed.get(seed) in (None, resolved_mode)
        }
    csv_seeds = set(rows_by_seed)
    artifact_seeds = set(artifact_dirs)
    all_seeds = sorted(expected_seeds | csv_seeds | artifact_seeds)
    no_common_seed = bool(csv_seeds and artifact_seeds and not (csv_seeds & artifact_seeds))

    entries = [
        _seed_entry(seed, rows_by_seed.get(seed), artifact_dirs.get(seed))
        for seed in all_seeds
    ]
    summary = aggregate(entries, no_common_seed=no_common_seed)

    identity = {
        key: value
        for key, value in (manifest.get("identity") or {}).items()
        if key in {"robot_id", "mode", "racket_id", "config_hash", "metric_schema"}
    }
    row0 = next(iter(rows_by_seed.values()), {})
    row_mode = _text_or_none(row0.get("mode"))
    if resolved_mode is None:
        resolved_mode = (
            (row_mode or "").strip().lower()
            or (declared_modes[0] if len(declared_modes) == 1 else None)
            or (identity.get("mode") or "").strip().lower()
            or manifest.get("mode")
        )
    robot_id = _text_or_none(row0.get("robot_id")) or _text_or_none(identity.get("robot_id"))
    racket_id = _text_or_none(row0.get("racket_id")) or _text_or_none(identity.get("racket_id"))
    config_hash = _text_or_none(row0.get("config_hash")) or _text_or_none(identity.get("config_hash"))
    metric_schema = _text_or_none(row0.get("metric_schema")) or _text_or_none(identity.get("metric_schema"))
    if catalog is not None and robot_id and robot_id != catalog.robot_id:
        raise ExtractError(
            f"catalog robot_id={catalog.robot_id!r} 与数据 robot_id={robot_id!r} 不一致"
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": experiment,
        "case_id": case,
        "mode": resolved_mode,
        "racket_id": racket_id,
        "robot_id": robot_id,
        "config_hash": config_hash,
        "metric_schema": metric_schema,
        "hit_semantics": dict(HIT_SEMANTICS),
        "data_root": str(root.resolve()),
        "run_root": str(run_root.resolve()),
        "catalog": catalog_text,
        "synthetic": synthetic,
        "sources": sources,
        "seeds_expected": sorted(expected_seeds),
        "seeds_found": all_seeds,
        "no_common_seed": no_common_seed,
        **summary,
        "summary": summary,
        "per_seed": entries,
    }
    return payload


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="R1D 论文 smoke 单 case 统计提取（physical / legacy 口径分开，缺口显式）"
    )
    parser.add_argument("--experiment", required=True, help="实验目录名，如 exp15_speed_v2")
    parser.add_argument("--case", required=True, help="case id，如 speed7")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help=f"数据根（默认 {PROJECT / 'experiment_data'}；R1D 用 <run_root>/batch）",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出目录（T8a 传 <run_root>/stats；未给则只打印不写盘）",
    )
    parser.add_argument("--manifest", type=Path, default=None, help="可选 manifest JSON（补预期 seed / identity）")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help="可选 catalog.json（解析 run_root/data_root/manifest；坏 catalog 退出码 2）",
    )
    parser.add_argument(
        "--mode",
        choices=MODE_CHOICES,
        default=None,
        help="执行模式；未给时按数据推断，混合模式必须显式选择（both 分模式各写一份）",
    )
    return parser


def _report(payload: dict[str, Any], experiment: str, case: str) -> None:
    """打印单次提取的摘要行（stdout 是运维界面）。"""
    physical_rate = payload["physical_active_rate"]
    legacy_rate = payload["legacy_active_rate"]
    physical_text = "null" if physical_rate is None else f"{physical_rate:.1f}%"
    legacy_text = "null" if legacy_rate is None else f"{legacy_rate:.1f}%"
    print(
        f"[r1d-smoke-stats] {experiment}/{case} mode={payload.get('mode')} "
        f"seeds={payload['n_attempted']} valid={payload['n_valid']} "
        f"physical_active_rate={physical_text} "
        f"(n={payload['physical_active_denominator']}) legacy_active_rate={legacy_text} "
        f"(n={payload['legacy_active_denominator']}) unavailable_reason={payload['unavailable_reason']}"
    )


def main(argv: list[str] | None = None) -> int:
    """入口：收集统计并写出 ``<output>[/<mode>]/<experiment>/<case>.json``。"""
    args = build_parser().parse_args(argv)
    try:
        catalog = _load_catalog(args.catalog) if args.catalog is not None else None
        data_root: Path = (
            args.data_root
            if args.data_root is not None
            else (Path(catalog.data_root) if catalog is not None else PROJECT / "experiment_data")
        )
        if args.mode == "both":
            modes: tuple[str | None, ...] = VALID_MODES
        elif args.mode is not None:
            modes = (args.mode,)
        else:
            modes = (None,)
        output_dir: Path = args.output if args.output is not None else default_output_dir(data_root)
        for requested in modes:
            payload = collect_case_stats(
                experiment=args.experiment,
                case=args.case,
                data_root=data_root,
                manifest_path=args.manifest,
                catalog_path=args.catalog,
                mode=requested,
            )
            if args.mode == "both" and payload["n_attempted"] == 0:
                continue
            if args.output is not None:
                if requested is not None:
                    target = output_dir / requested / args.experiment / f"{args.case}.json"
                else:
                    target = output_dir / args.experiment / f"{args.case}.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            else:
                target = None
            _report(payload, args.experiment, args.case)
            if args.output is None:
                print("[r1d-smoke-stats] 未给 --output：只打印，不写文件")
            elif target is not None:
                print(f"[r1d-smoke-stats] wrote {target}")
    except ExtractError as exc:
        print(f"[r1d-smoke-stats] 错误：{exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())


if __name__ == "__main__":
    sys.exit(main())
