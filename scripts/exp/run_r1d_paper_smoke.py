"""R1D 论文 smoke 运行入口（T8a 交付）。

读取 ``configs/experiments/r1d_paper_smoke.yaml``，完成：

1. manifest 校验（schema / 身份 / seed 小样本规则 / override 白名单 /
   R1D 共享 CLI 面 / 依赖引用 / 每 case 的 10 个计划字段）；
2. 解析出本次运行根下的 ``manifest_resolved.json``（每条命令、身份、
   config_hash、seed、状态；每个 case 的 ``source_paper_item/experiment_id/
   argv/extract_argv/plot_argv/expected_outputs/seed/timeout_s``；config_hash
   在运行时对冻结配置文件计算，缺文件即报错，不伪造）；case 级
   ``runtime_overrides`` 会生成 ``<run_root>/runtime_overrides/<unit>/<case>.yaml``
   快照并以稳定 token 参与 config_id（N15，换 --output 不改变身份）；
3. 可选执行：复用 ``scripts/exp/batch_runner.py`` 的 run_batch 逐实验运行，
   R1D 强制 workers=1；工具型额外依赖（如 exp18_fig_assets）按 manifest
   命令模板执行并校验 ``expected_outputs``；
4. 汇总 ``runs.csv``（保留 batch_runner 旧列 + 身份列 + 状态列 +
   从 ``v12_episode.json`` 提取的物理命中/出球质量列 + ``hit_semantics=legacy``）。

``--dry-run`` 只写 ``manifest_resolved.json`` 与 runtime 快照，并逐 case 打印
10 个计划字段（blocked case 显式 null），供 T8c 在执行前核对。

正式 19 组 smoke 由 T8c 在 T7 冻结后执行；T8a 只保证 dry-run 与运行层正确。
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.exp import batch_runner  # noqa: E402
from scripts.exp.batch_runner import (  # noqa: E402
    CSV_FIELDS,
    ExperimentSpec,
    RunIdentity,
    compute_config_hash,
    make_config_id,
)

PROJECT = batch_runner.PROJECT
MANIFEST_SCHEMA = "r1d_paper_smoke_v1"
RESOLVED_SCHEMA = "r1d_paper_smoke_resolved_v1"
DEFAULT_MANIFEST_PATH = PROJECT / "configs" / "experiments" / "r1d_paper_smoke.yaml"
DEFAULT_RUN_ROOT = PROJECT / "experiment_data" / "r1d_v12_parity" / "paper_smoke"

#: catalog 的数据来源集合（论文名义机制数据 = exp17d 噪声对照组的机制组）
SOURCE_SETS: dict[str, str] = {"nominal_mechanism": "exp17d_mechanism"}

#: batch 条件允许的 argv 键（R1D 共享 CLI 面；机器人配置由 manifest 注入）
ALLOWED_CASE_ARGV = frozenset(
    {"--serve-box", "--no-plot", "--ball-speed", "--position-mode"})
#: RM65 专属开关，出现在 R1D manifest 的 case argv 中必须拒绝（避免静默失效）
FORBIDDEN_CASE_ARGV = frozenset({
    "--always-collision", "--no-freeze-first-plan", "--max-tcp", "--limits-config",
    "--ablation", "--random-perturb", "--perturb-sign", "--obs-noise-pos",
    "--obs-noise-vel", "--obs-use-kf", "--dump-trajectory", "--dump-step-timing",
    "--dump-step-latencies", "--async-replan", "--sim-limits",
    "--time-perturb-ms", "--space-perturb-m",
})
VALID_MODES = ("torque", "position")
VALID_CASE_STATUS = ("ready", "adapted", "blocked")
VALID_UNIT_KINDS = ("batch", "derived", "tool")
VALID_MODE_FILTERS = ("torque", "position", "both")

#: 每个 case 必须能直接读出的计划字段（T8 N14，计划 Interfaces 字面要求）。
#: 前 5 项是论文计划/追溯信息，后 5 项是运行计划；blocked case 对不适用的
#: 运行计划字段显式写 null。
CASE_SPEC_FIELDS = (
    "id",
    "source_paper_item",
    "experiment_id",
    "argv",
    "extract_argv",
    "plot_argv",
    "expected_outputs",
    "mode",
    "seed",
    "timeout_s",
)
#: manifest.case_defaults 允许的默认键（id/mode/argv 等 case 自身字段不可默认）。
CASE_DEFAULT_KEYS = (
    "source_paper_item",
    "experiment_id",
    "extract_argv",
    "plot_argv",
    "expected_outputs",
    "seed",
    "timeout_s",
)

#: runs.csv 的 ``hit`` 列沿用 batch_runner legacy 口径（active|passive），
#: 论文统计必须用物理命中列（``physical_hit`` / ``hit_type == "active"``）。
RUNS_HIT_SEMANTICS = "legacy"
#: case 级 runtime 覆盖生成的 YAML 快照目录（相对 run_root）
RUNTIME_SNAPSHOT_DIR = "runtime_overrides"
#: 每个 R1D 任务的统一 v2 产物 JSON（M09 续跑校验的核心对象）
V12_EPISODE_NAME = "v12_episode.json"
#: 共享 v2 产物 NPZ（R02：必须真实解包校验，不能只看文件存在）
V12_EPISODE_NPZ_NAME = "v12_episode.npz"
#: 旧 R1D 兼容 NPZ（``r1d_sim_episode_v1``）
R1D_EPISODE_NPZ_NAME = "r1d_episode.npz"
#: v12 NPZ 必须包含的数组（与 ``write_episode_artifacts`` 的 savez 键一致）
V12_NPZ_KEYS = (
    "schema", "X", "U", "ball_pos", "ball_vel", "time", "racket_pos",
)

#: 从 v12_episode.json 提取并追加到 runs.csv 的列（R1D 结果行不含出球质量）
ARTIFACT_FIELDS = [
    "physical_hit",
    "physical_hit_type",
    "physical_hit_step",
    "v_racket_at_hit",
    "ball_out_speed",
    "ball_out_vy",
    "contact_event_count",
    "artifact_error",
]
#: 论文 smoke 的 R1D 要求（串行执行，避免多进程 MuJoCo 段错误）
REQUIRED_WORKERS = 1

_OVERRIDE_KEYS_CACHE: frozenset[str] | None = None


class ManifestError(ValueError):
    """manifest 结构或语义不合法（运行前即失败，不产生部分产物）。"""


# ---------------------------------------------------------------------------
# 路径与通用小工具
# ---------------------------------------------------------------------------


def _resolve_project_path(value: str | Path) -> Path:
    """把相对路径按项目根解析为绝对路径。"""
    path = Path(value)
    return path if path.is_absolute() else PROJECT / path


def _display_path(path: str | Path) -> str:
    """尽量相对项目根显示路径（越界时返回绝对路径）。"""
    path = Path(path)
    try:
        return str(path.resolve().relative_to(PROJECT))
    except ValueError:
        return str(path)


def _format_override(value: Any) -> str:
    """把 task_overrides 的值格式化为 YAML 可解析字符串（V12 用 yaml.safe_load）。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, float):
        return repr(value)
    return str(value)


def _file_sha256(path: Path) -> str:
    """计算文件 sha256（16 进制全串）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allowed_override_keys() -> frozenset[str]:
    """延迟加载 V12 overlay 白名单（避免 import 阶段拉起重依赖）。

    Raises:
        ManifestError: 无法导入白名单（视为环境损坏，不做降级猜测）。
    """
    global _OVERRIDE_KEYS_CACHE
    if _OVERRIDE_KEYS_CACHE is None:
        try:
            from src.sim.v12_config import _OVERRIDE_VALIDATORS
        except Exception as exc:  # pragma: no cover - 环境异常路径
            raise ManifestError(
                f"无法加载 V12 override 白名单（src.sim.v12_config）: {exc}") from exc
        _OVERRIDE_KEYS_CACHE = frozenset(_OVERRIDE_VALIDATORS)
    return _OVERRIDE_KEYS_CACHE


# ---------------------------------------------------------------------------
# manifest 读取与校验
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """读取 smoke manifest（YAML）。

    Args:
        path: manifest 路径（默认 ``configs/experiments/r1d_paper_smoke.yaml``）。

    Returns:
        解析后的 manifest 字典（未做语义校验）。

    Raises:
        ManifestError: 文件不存在或不是 YAML 映射。
    """
    manifest_path = _resolve_project_path(path or DEFAULT_MANIFEST_PATH)
    if not manifest_path.is_file():
        raise ManifestError(f"manifest 不存在: {manifest_path}")
    try:
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ManifestError(f"manifest YAML 解析失败: {exc}") from exc
    if not isinstance(data, Mapping):
        raise ManifestError(f"manifest 顶层必须是映射: {manifest_path}")
    return dict(data)


def _validate_seeds(raw: Any, where: str) -> list[int]:
    """校验 seed 列表：非空、正整数、无重复、至少 2 个（小样本规则）。"""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ManifestError(f"{where}: seeds 必须是列表")
    seeds: list[int] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ManifestError(f"{where}: seed 必须是整数，得到 {item!r}")
        if item < 0:
            raise ManifestError(f"{where}: seed 不能为负，得到 {item}")
        seeds.append(item)
    if len(set(seeds)) != len(seeds):
        raise ManifestError(f"{where}: seeds 有重复: {seeds}")
    if len(seeds) < 2:
        raise ManifestError(
            f"{where}: 小样本规则要求每个比较 cell ≥2 个共同 seed，"
            f"当前只有 {seeds}；单 seed 调试请直接用 V12 命令，不走 paper smoke")
    return seeds


def _validate_override_mapping(raw: Any, where: str) -> dict[str, Any]:
    """校验 task_overrides 映射：键必须在 V12 overlay 白名单内。"""
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: task_overrides 必须是映射")
    allowed = _allowed_override_keys()
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ManifestError(f"{where}: override 键必须是字符串: {key!r}")
        if key not in allowed:
            raise ManifestError(
                f"{where}: override 键 {key!r} 不在 V12 overlay 白名单内"
                f"（见 src/sim/v12_config.py::_OVERRIDE_VALIDATORS）")
        out[key] = value
    return out


def _load_base_runtime_keys(manifest: Mapping[str, Any]) -> frozenset[str]:
    """读取冻结 runtime YAML 的顶层键（校验 case 级 runtime 覆盖键）。

    Raises:
        ManifestError: runtime 配置文件缺失/不可解析。
    """
    rel = str(manifest["configs"]["runtime"])
    path = _resolve_project_path(rel)
    if not path.is_file():
        raise ManifestError(
            f"manifest.configs.runtime 文件不存在（无法校验 runtime_overrides）: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"读取 runtime 配置失败: {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ManifestError(f"runtime 配置必须是映射: {path}")
    return frozenset(str(key) for key in payload)


def _validate_runtime_overrides(
    raw: Any,
    where: str,
    *,
    runtime_keys: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """校验 case 级 runtime 覆盖（如 {"max_tcp_speed": 1.0}）。

    R1D 的 ``--runtime-config`` 是 max_tcp_speed 等运行参数的唯一来源；
    这里只允许覆盖冻结 runtime YAML 已存在的顶层键，避免拼错静默失效。
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: runtime_overrides 必须是映射")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.strip():
            raise ManifestError(f"{where}: runtime 覆盖键必须是非空字符串: {key!r}")
        if runtime_keys and key not in runtime_keys:
            raise ManifestError(
                f"{where}: runtime 覆盖键 {key!r} 不在冻结 runtime YAML 顶层"
                f"（可用: {sorted(runtime_keys)}）")
        if isinstance(value, (Mapping, list, tuple)) or value is None:
            raise ManifestError(
                f"{where}: runtime 覆盖 {key!r} 只支持标量（bool/int/float/str），"
                f"得到 {type(value).__name__}")
        out[key] = value
    return out


def _validate_case_defaults(raw: Any) -> dict[str, Any]:
    """校验 ``manifest.case_defaults``：每个 case 计划字段的默认值。"""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ManifestError("manifest.case_defaults 必须是映射")
    unknown = sorted(str(key) for key in raw if key not in CASE_DEFAULT_KEYS)
    if unknown:
        raise ManifestError(
            f"manifest.case_defaults 含未知键 {unknown}（可用: {list(CASE_DEFAULT_KEYS)}）")
    return {str(key): value for key, value in raw.items()}


def _case_token_list(value: Any, where: str) -> list[str]:
    """校验 argv 模板列表（str 列表）。"""
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ManifestError(f"{where} 必须是字符串列表（argv 模板）")
    tokens = [str(token) for token in value]
    if not tokens:
        raise ManifestError(f"{where} 不能为空列表")
    return tokens


def _validate_case_plan_fields(
    case: Mapping[str, Any],
    *,
    defaults: Mapping[str, Any],
    where: str,
    unit_id: str,
    default_seeds: list[int],
    status: str,
) -> dict[str, Any]:
    """解析单个 case 的计划字段（N14：计划 Interfaces 要求的 9 字段）。

    字段可继承 ``manifest.case_defaults``（或以后的 experiment 级默认），
    case 显式值优先；blocked case 允许对运行计划字段显式写 null。
    """
    def pick(key: str) -> Any:
        return case[key] if key in case else defaults.get(key)

    raw_experiment = pick("experiment_id")
    if raw_experiment is None or raw_experiment == "{experiment_id}":
        experiment_id = unit_id
    elif isinstance(raw_experiment, str) and raw_experiment.strip():
        experiment_id = raw_experiment.replace("{experiment_id}", unit_id)
    else:
        raise ManifestError(f"{where}.experiment_id 必须是非空字符串")

    raw_source = pick("source_paper_item")
    if not isinstance(raw_source, str) or not raw_source.strip():
        raise ManifestError(
            f"{where}.source_paper_item 必须是非空字符串（指向论文矩阵条目）")
    source_paper_item = raw_source.replace("{experiment_id}", experiment_id)

    # seed：支持 int 或 int 列表；兼容旧键 ``seeds``；blocked 允许 null。
    raw_seed = pick("seed")
    if "seeds" in case and raw_seed is None:
        case_seeds = _validate_seeds(case["seeds"], f"{where}.seeds")
    elif raw_seed is None:
        case_seeds = [] if status == "blocked" else list(default_seeds)
    elif isinstance(raw_seed, int) and not isinstance(raw_seed, bool):
        case_seeds = [int(raw_seed)]
    elif isinstance(raw_seed, Sequence) and not isinstance(raw_seed, (str, bytes)):
        case_seeds = _validate_seeds(list(raw_seed), f"{where}.seed")
    else:
        raise ManifestError(
            f"{where}.seed 必须是 int 或 int 列表，得到 {type(raw_seed).__name__}")
    if status == "blocked":
        case_seeds = []  # blocked 不建任务；seed 字段在 resolved 中显式 null

    raw_timeout = pick("timeout_s")
    timeout_s: float | None
    if raw_timeout is None:
        if status == "blocked":
            timeout_s = None
        else:
            timeout_s = float(defaults.get("timeout_s", 0.0) or 0.0) or None
    else:
        try:
            timeout_s = float(raw_timeout)
        except (TypeError, ValueError) as exc:
            raise ManifestError(f"{where}.timeout_s 必须是数字: {raw_timeout!r}") from exc
        if timeout_s <= 0:
            raise ManifestError(f"{where}.timeout_s 必须 > 0，得到 {timeout_s}")

    raw_argv = case.get("argv")
    argv: dict[str, Any] | None
    if status == "blocked" and raw_argv is None:
        argv = None  # blocked case 不适用处显式 null（N14）
    elif raw_argv is None:
        argv = {}
    else:
        argv = _validate_argv(raw_argv, f"{where}.argv")

    raw_expected = pick("expected_outputs")
    if raw_expected is None:
        if status != "blocked":
            raise ManifestError(
                f"{where}.expected_outputs 必须给出（非 blocked case 不可为 null）")
        expected_outputs: list[str] | None = None
    elif not isinstance(raw_expected, Sequence) or isinstance(raw_expected, (str, bytes)):
        raise ManifestError(f"{where}.expected_outputs 必须是字符串列表")
    else:
        expected_outputs = [str(item) for item in raw_expected]
        if not expected_outputs:
            raise ManifestError(f"{where}.expected_outputs 不能为空列表")

    def _plan_argv(key: str) -> list[str] | None:
        raw = pick(key)
        if raw is None:
            if status != "blocked":
                raise ManifestError(f"{where}.{key} 必须给出（非 blocked case 不可为 null）")
            return None
        return _case_token_list(raw, f"{where}.{key}")

    return {
        "experiment_id": experiment_id,
        "source_paper_item": source_paper_item,
        "argv": argv,
        "extract_argv": _plan_argv("extract_argv"),
        "plot_argv": _plan_argv("plot_argv"),
        "expected_outputs": expected_outputs,
        "seed": case_seeds if case_seeds else None,
        "seeds": case_seeds,
        "timeout_s": timeout_s,
    }


def _validate_argv(raw: Any, where: str) -> dict[str, Any]:
    """校验 case argv：只允许 R1D 共享面，禁止 RM65 专属开关。"""
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ManifestError(f"{where}: argv 必须是映射")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not key.startswith("--"):
            raise ManifestError(f"{where}: argv 键必须形如 '--flag': {key!r}")
        if key in FORBIDDEN_CASE_ARGV:
            raise ManifestError(
                f"{where}: {key!r} 是 RM65 专属开关，R1D 共享 CLI 面会拒绝"
                f"（见 scripts/rm65_mpc_v12.py::_collect_rejected_rm65_flags）")
        if key not in ALLOWED_CASE_ARGV:
            raise ManifestError(f"{where}: 未在允许列表内的 argv 键: {key!r}")
        out[key] = value
    return out


def _validate_cases(
    cases: Any,
    where: str,
    *,
    ablation_profiles: Mapping[str, Mapping[str, Any]],
    default_seeds: list[int],
    runtime_keys: frozenset[str] = frozenset(),
    case_defaults: Mapping[str, Any] | None = None,
    unit_id: str = "",
) -> list[dict[str, Any]]:
    """校验并规范化一个单元（实验/依赖）的 cases 列表。"""
    if not isinstance(cases, Sequence) or isinstance(cases, (str, bytes)):
        raise ManifestError(f"{where}: cases 必须是列表")
    defaults = dict(case_defaults or {})
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        cwhere = f"{where}.cases[{index}]"
        if not isinstance(case, Mapping):
            raise ManifestError(f"{cwhere}: case 必须是映射")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ManifestError(f"{cwhere}: 缺少非空 id")
        if case_id in seen:
            raise ManifestError(f"{cwhere}: case id 重复: {case_id!r}")
        seen.add(case_id)
        mode = case.get("mode")
        if mode not in VALID_MODES:
            raise ManifestError(f"{cwhere}: mode 必须是 {VALID_MODES}，得到 {mode!r}")
        tier = case.get("tier")
        if tier is not None and tier not in ablation_profiles:
            raise ManifestError(
                f"{cwhere}: tier {tier!r} 不在 ablation_profiles 中"
                f"（可用: {sorted(ablation_profiles)}）")
        status = case.get("status", "ready")
        if status not in VALID_CASE_STATUS:
            raise ManifestError(
                f"{cwhere}: status 必须是 {VALID_CASE_STATUS}，得到 {status!r}")
        reason = case.get("blocked_reason", "")
        if status == "blocked" and not str(reason).strip():
            raise ManifestError(f"{cwhere}: blocked case 必须给出 blocked_reason")
        plan = _validate_case_plan_fields(
            case, defaults=defaults, where=cwhere, unit_id=unit_id,
            default_seeds=default_seeds, status=status)
        overrides = _validate_override_mapping(
            case.get("task_overrides", {}), f"{cwhere}.task_overrides")
        runtime_overrides = _validate_runtime_overrides(
            case.get("runtime_overrides"), f"{cwhere}.runtime_overrides",
            runtime_keys=runtime_keys)
        normalized.append({
            "id": case_id,
            "mode": mode,
            "tier": tier,
            "status": status,
            "blocked_reason": str(reason),
            "argv": plan["argv"],
            "extract_argv": plan["extract_argv"],
            "plot_argv": plan["plot_argv"],
            "expected_outputs": plan["expected_outputs"],
            "experiment_id": plan["experiment_id"],
            "source_paper_item": plan["source_paper_item"],
            "seed": plan["seed"],
            "seeds": plan["seeds"],
            "timeout_s": plan["timeout_s"],
            "task_overrides": overrides,
            "runtime_overrides": runtime_overrides,
        })
    return normalized


def _check_seed_rule(unit: Mapping[str, Any], where: str) -> None:
    """校验每比较 cell ≥2 共同 seed（小样本规则，只看非 blocked case）。"""
    active = [c for c in unit["cases"] if c["status"] != "blocked"]
    if not active:
        return  # 整组 blocked 时允许（由 T8c 决定是否补条件）
    for case in active:
        if len(case["seeds"]) < 2:
            raise ManifestError(
                f"{where}: case {case['id']!r} 只有 {case['seeds']} 个 seed，"
                f"违反小样本规则（每比较 cell ≥2 共同 seed）")
    common = set(active[0]["seeds"])
    for case in active[1:]:
        common &= set(case["seeds"])
    if len(common) < 2:
        raise ManifestError(
            f"{where}: 非 blocked cases 的共同 seed 少于 2 个"
            f"（{sorted(common)}），无法做配对比较")


def _validate_units(
    units: Any,
    key: str,
    *,
    known_ids: set[str],
    ablation_profiles: Mapping[str, Mapping[str, Any]],
    default_seeds: list[int],
    runtime_keys: frozenset[str] = frozenset(),
    case_defaults: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """校验 experiments / extra_dependencies 列表并返回规范化单元。"""
    if not isinstance(units, Sequence) or isinstance(units, (str, bytes)):
        raise ManifestError(f"{key} 必须是列表")
    seen: set[str] = set()
    normalized: list[dict[str, Any]] = []
    for index, unit in enumerate(units):
        where = f"{key}[{index}]"
        if not isinstance(unit, Mapping):
            raise ManifestError(f"{where}: 单元必须是映射")
        unit_id = unit.get("id")
        if not isinstance(unit_id, str) or not unit_id.strip():
            raise ManifestError(f"{where}: 缺少非空 id")
        if unit_id in seen or unit_id in known_ids:
            raise ManifestError(f"{where}: id 重复或与 experiments 冲突: {unit_id!r}")
        seen.add(unit_id)
        kind = unit.get("kind", "batch")
        if key == "extra_dependencies":
            if kind not in ("tool", "batch"):
                raise ManifestError(f"{where}: 依赖 kind 必须是 tool/batch，得到 {kind!r}")
        elif kind not in ("batch", "derived"):
            raise ManifestError(f"{where}: 实验 kind 必须是 batch/derived，得到 {kind!r}")
        cases = _validate_cases(
            unit.get("cases", []) if kind != "tool" else [],
            where, ablation_profiles=ablation_profiles,
            default_seeds=default_seeds, runtime_keys=runtime_keys,
            case_defaults=case_defaults, unit_id=unit_id)
        entry: dict[str, Any] = {
            "id": unit_id,
            "kind": kind,
            "title": str(unit.get("title", "")),
            "paper_ref": unit.get("paper_ref", {}) or {},
            "matrix_ref": str(unit.get("matrix_ref", "")),
            "depends_on": [str(x) for x in unit.get("depends_on", [])],
            "expected_fields": [str(x) for x in unit.get("expected_fields", [])],
            "t8b_consumer": str(unit.get("t8b_consumer", "")),
            "notes": str(unit.get("notes", "")),
            "cases": cases,
            "status": "ready",
        }
        if kind == "derived":
            if not cases:
                entry["cases"] = []
            derived_from = unit.get("derived_from")
            if not isinstance(derived_from, str) or not derived_from.strip():
                raise ManifestError(f"{where}: derived 实验必须给出 derived_from")
            entry["derived_from"] = derived_from
        elif kind == "tool":
            status = unit.get("status", "ready")
            if status not in ("ready", "blocked"):
                raise ManifestError(f"{where}: tool 依赖 status 只能是 ready/blocked")
            reason = str(unit.get("blocked_reason", ""))
            if status == "blocked" and not reason.strip():
                raise ManifestError(f"{where}: blocked tool 依赖必须给出 blocked_reason")
            command = unit.get("command", [])
            if status == "ready":
                if not isinstance(command, Sequence) or not command:
                    raise ManifestError(f"{where}: tool 依赖必须给出 command 列表")
                expected = unit.get("expected_outputs", [])
                if not isinstance(expected, Sequence) or not expected:
                    raise ManifestError(f"{where}: tool 依赖必须给出 expected_outputs")
            entry.update({
                "status": status,
                "blocked_reason": reason,
                "command": [str(tok) for tok in command],
                "expected_outputs": [str(x) for x in unit.get("expected_outputs", [])],
            })
        else:
            _check_seed_rule(entry, where)
            if unit.get("depends_on"):
                entry["depends_on"] = [str(x) for x in unit["depends_on"]]
        normalized.append(entry)
    return normalized


def _check_dependency_refs(
    experiments: Sequence[Mapping[str, Any]],
    dependencies: Sequence[Mapping[str, Any]],
) -> None:
    """校验 depends_on 引用存在且无环。"""
    known = {str(e["id"]) for e in experiments} | {str(d["id"]) for d in dependencies}
    graph: dict[str, list[str]] = {}
    for unit in list(experiments) + list(dependencies):
        unit_id = str(unit["id"])
        refs = [str(x) for x in unit.get("depends_on", [])]
        for ref in refs:
            if ref not in known:
                raise ManifestError(f"{unit_id}: depends_on 引用了未知 id {ref!r}")
            if ref == unit_id:
                raise ManifestError(f"{unit_id}: depends_on 不能自引用")
        if unit.get("kind") == "derived":
            derived_from = str(unit.get("derived_from", ""))
            if derived_from not in known:
                raise ManifestError(
                    f"{unit_id}: derived_from 引用了未知 id {derived_from!r}")
            if derived_from == unit_id:
                raise ManifestError(f"{unit_id}: derived_from 不能自引用")
            refs = refs + [derived_from]
        graph[unit_id] = refs
    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ManifestError(f"depends_on 存在环: {node}")
        visiting.add(node)
        for ref in graph.get(node, []):
            _visit(ref)
        visiting.discard(node)
        visited.add(node)

    for node in graph:
        _visit(node)


def _normalize_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """校验 manifest 并返回规范化副本（cases/依赖带默认值与种子）。

    规范化后的结构是后续 resolve/执行唯一依赖的形态：
    ``seeds`` 为 int 列表、``ablation_profiles`` 已校验、``experiments`` /
    ``extra_dependencies`` 的 case 含 status/argv/task_overrides/seeds。

    Raises:
        ManifestError: 任一项不合法。
    """
    if not isinstance(manifest, Mapping):
        raise ManifestError("manifest 必须是映射")
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ManifestError(
            f"schema 必须是 {MANIFEST_SCHEMA!r}，得到 {manifest.get('schema')!r}")
    seeds = _validate_seeds(manifest.get("seeds"), "manifest")
    configs = manifest.get("configs")
    if not isinstance(configs, Mapping):
        raise ManifestError("manifest.configs 必须是映射")
    for role in ("robot", "runtime", "racket", "task_overlay"):
        if not isinstance(configs.get(role), str) or not configs[role]:
            raise ManifestError(f"manifest.configs.{role} 必须是非空路径字符串")
    if configs.get("sim_task_config") is not None:
        raise ManifestError(
            "manifest.configs.sim_task_config 必须为 null：paper smoke 走 R1D overlay 模式"
            "（显式 --sim-task-config 会切到固定球兼容模式并拒绝 serve-box 等参数）")
    identity_defaults = manifest.get("identity_defaults")
    if not isinstance(identity_defaults, Mapping):
        raise ManifestError("manifest.identity_defaults 必须是映射")
    for key in ("robot_id", "racket_id", "metric_schema"):
        if not isinstance(identity_defaults.get(key), str) or not identity_defaults[key]:
            raise ManifestError(f"manifest.identity_defaults.{key} 必须是非空字符串")
    base_argv = _validate_argv(manifest.get("base_argv", {}), "manifest.base_argv")
    profiles_raw = manifest.get("ablation_profiles")
    if not isinstance(profiles_raw, Mapping):
        raise ManifestError("manifest.ablation_profiles 必须是映射")
    profiles: dict[str, dict[str, Any]] = {}
    for name, overrides in profiles_raw.items():
        profiles[str(name)] = _validate_override_mapping(
            overrides or {}, f"manifest.ablation_profiles.{name}")
    script = manifest.get("script")
    if not isinstance(script, str) or not script:
        raise ManifestError("manifest.script 必须是非空字符串")
    case_defaults = _validate_case_defaults(manifest.get("case_defaults"))
    runtime_keys = _load_base_runtime_keys(manifest)
    experiments = _validate_units(
        manifest.get("experiments"), "experiments", known_ids=set(),
        ablation_profiles=profiles, default_seeds=seeds,
        runtime_keys=runtime_keys, case_defaults=case_defaults)
    if not experiments:
        raise ManifestError("manifest.experiments 不能为空")
    dependencies = _validate_units(
        manifest.get("extra_dependencies", []), "extra_dependencies",
        known_ids={str(e["id"]) for e in experiments},
        ablation_profiles=profiles, default_seeds=seeds,
        runtime_keys=runtime_keys, case_defaults=case_defaults)
    _check_dependency_refs(experiments, dependencies)
    normalized = dict(manifest)
    normalized.update({
        "seeds": seeds,
        "configs": dict(configs),
        "identity_defaults": dict(identity_defaults),
        "base_argv": base_argv,
        "ablation_profiles": profiles,
        "case_defaults": case_defaults,
        "experiments": experiments,
        "extra_dependencies": dependencies,
    })
    return normalized


def validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """校验 smoke manifest 并返回规范化副本（不合法即抛 ManifestError）。

    校验项：schema、seeds 小样本规则、configs/identity_defaults 完整性、
    ablation_profiles 与 override 白名单、cases 的 argv/status/tier、
    每比较 cell ≥2 共同 seed、依赖引用与环、tool 依赖的输出声明。

    Args:
        manifest: ``load_manifest`` 返回的字典。

    Returns:
        规范化 manifest（case 带默认 status/argv/task_overrides/seeds）。

    Raises:
        ManifestError: 任一项不合法。
    """
    return _normalize_manifest(manifest)


# ---------------------------------------------------------------------------
# manifest 解析（命令/身份/config_hash/seed）
# ---------------------------------------------------------------------------


def _resolved_configs(manifest: Mapping[str, Any]) -> dict[str, str]:
    """返回 manifest 冻结配置文件（角色 → 项目内相对路径）。"""
    configs = manifest["configs"]
    return {
        "robot": str(configs["robot"]),
        "runtime": str(configs["runtime"]),
        "racket": str(configs["racket"]),
        "task": str(configs["task_overlay"]),
    }


def _config_hash(manifest: Mapping[str, Any]) -> tuple[str, dict[str, dict[str, str]]]:
    """计算冻结配置的 config_hash，并返回每个角色文件的 sha256。

    Raises:
        ManifestError: 任一冻结配置文件缺失（T7 未冻结/路径错误）。
    """
    role_paths = _resolved_configs(manifest)
    details: dict[str, dict[str, str]] = {}
    for role, rel in role_paths.items():
        path = _resolve_project_path(rel)
        if not path.is_file():
            raise ManifestError(
                f"冻结配置文件缺失（T7 未冻结或路径错误）: {role} -> {path}")
        details[role] = {"path": rel, "sha256": _file_sha256(path)}
    try:
        digest = compute_config_hash(role_paths, extra={"manifest_schema": MANIFEST_SCHEMA})
    except FileNotFoundError as exc:  # pragma: no cover - 上面已显式检查
        raise ManifestError(str(exc)) from exc
    return digest, details


def _runtime_effective_config(
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    case: Mapping[str, Any],
    run_root: Path,
) -> tuple[str, dict[str, Any]]:
    """解析 case 级 runtime 覆盖，返回 (``--runtime-config`` 值, 快照信息)。

    R1D 的 max_tcp_speed 只在 ``--runtime-config`` 指向的 YAML 里；不能用
    ``--max-tcp``（R1D 共享面拒绝）。因此按 base runtime + case 覆盖生成
    快照文件，并在 resolved/CSV 记录 base 路径、覆盖字典与快照 sha256。

    Args:
        manifest: 规范化 manifest（含 configs.runtime）。
        unit: 实验/依赖单元。
        case: 规范化 case（含 runtime_overrides）。
        run_root: 本次运行根目录（快照写在 run_root/runtime_overrides/ 下）。

    Returns:
        (传给 --runtime-config 的路径字符串, runtime 信息字典)。
    """
    base_rel = str(manifest["configs"]["runtime"])
    overrides = dict(case.get("runtime_overrides") or {})
    base_path = _resolve_project_path(base_rel)
    base_sha = _file_sha256(base_path)
    if not overrides:
        return base_rel, {
            "base": base_rel,
            "base_sha256": base_sha,
            "overrides": {},
            "snapshot": None,
            "sha256": base_sha,
        }
    payload = yaml.safe_load(base_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ManifestError(f"runtime 配置必须是映射，无法应用覆盖: {base_path}")
    effective = dict(payload)
    effective.update(overrides)
    snapshot_dir = run_root / RUNTIME_SNAPSHOT_DIR / str(unit["id"])
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"{case['id']}.yaml"
    snapshot_path.write_text(
        yaml.safe_dump(dict(effective), sort_keys=False, allow_unicode=True),
        encoding="utf-8")
    rel = _display_path(snapshot_path)
    return rel, {
        "base": base_rel,
        "base_sha256": base_sha,
        "overrides": dict(overrides),
        "snapshot": rel,
        "sha256": _file_sha256(snapshot_path),
    }


def _stable_config_id_argv(
    argv: Mapping[str, Any], runtime_info: Mapping[str, Any],
) -> dict[str, Any]:
    """生成用于 config_id 的稳定 argv（N15）。

    runtime 覆盖任务的 ``--runtime-config`` 指向本次 run_root 下的快照文件，
    直接进入 config_id 会让同一条任务在换 ``--output`` 后身份变化。这里用
    ``<base_stem>#<runtime_sha8>`` 稳定 token 取代快照路径；``--output`` 与
    ``--seed`` 已由 ``make_config_id`` 忽略。
    """
    stable = dict(argv)
    snapshot = runtime_info.get("snapshot")
    if snapshot:
        stem = Path(str(runtime_info.get("base", ""))).stem
        sha8 = str(runtime_info.get("sha256", ""))[:8]
        stable["--runtime-config"] = f"<{stem}#{sha8}>"
    return stable


def _task_log_path(
    run_root: Path, experiment: str, config_id: str, seed: int,
) -> Path:
    """构造任务的计划日志路径（与 batch 实际写日志共用同一函数，N01）。"""
    return batch_runner.task_log_path(
        Path(run_root) / "logs", experiment, config_id, seed)


def _run_root_relative(run_root: Path, value: str | Path) -> str:
    """把路径改写为相对 ``run_root`` 的 posix 字符串（越界则保留绝对路径）。

    与 ``_display_path`` 不同：后者相对 ``PROJECT``，无法在换 output 后由
    ``run_root`` 正确解析，故 manifest 内路径一律用本函数。

    Args:
        run_root: 本次 output 根目录。
        value: 已解析的绝对路径字符串（或相对路径，原样返回）。

    Returns:
        位于 run_root 内→相对 posix 路径；否则→绝对路径字符串。
    """
    path = Path(value)
    if not path.is_absolute():
        return path.as_posix()
    try:
        return path.resolve().relative_to(Path(run_root).resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


#: manifest 落盘时需要相对化的任务路径字段（`argv` 保留真实绝对命令）。
PORTABLE_TASK_PATH_FIELDS: tuple[str, ...] = (
    "artifact_dir",
    "log_path",
    "planned_artifact_dir",
    "planned_log_path",
)


def _iter_resolved_tasks(resolved: Mapping[str, Any]) -> Iterator[dict[str, Any]]:
    """遍历 resolved 的顶层 ``tasks`` 与各 unit 的 ``tasks``（去重）。

    两者通常指向同一批任务对象；用 ``id`` 去重避免同一路径被重复处理，
    也兼容未来只在一处保存任务的实现。

    Args:
        resolved: ``resolve_manifest`` 的结果。

    Yields:
        任务字典（可变，供就地改写路径）。
    """
    seen: set[int] = set()
    groups: list[Any] = [resolved.get("tasks") or []]
    groups.extend(unit.get("tasks") or [] for unit in resolved.get("units") or []
                  if isinstance(unit, Mapping))
    for group in groups:
        for task in group:
            if not isinstance(task, dict) or id(task) in seen:
                continue
            seen.add(id(task))
            yield task


def _portable_resolved(
    resolved: Mapping[str, Any], run_root: Path,
) -> dict[str, Any]:
    """生成可整树搬移的 ``manifest_resolved.json`` 载荷（M06）。

    落盘版本把 ``run_root`` 写成 ``"."``，``data_root`` 与任务的
    ``artifact_dir``/``log_path``/``planned_*`` 在 run_root 内改写成相对
    posix 路径（run_root 外保持绝对路径），因此整棵 output 目录移动后
    catalog/统计消费端仍能解析。内存中的 ``resolved`` 继续保留绝对路径，
    producer 执行任务与回填不受影响；``schema``/``task.argv``（真实绝对
    命令）按冻结约定不改写。

    Args:
        resolved: ``resolve_manifest``（可经 ``_apply_actual_paths`` 回填）。
        run_root: 本次 output 根目录（绝对路径）。

    Returns:
        深拷贝后的可序列化字典，可直接交给 :func:`_write_json`。
    """
    root = Path(run_root).resolve()
    portable = copy.deepcopy(dict(resolved))
    portable["run_root"] = "."
    portable["data_root"] = _run_root_relative(root, str(resolved["data_root"]))
    for task in _iter_resolved_tasks(portable):
        for field in PORTABLE_TASK_PATH_FIELDS:
            value = task.get(field)
            if value is None or not str(value).strip():
                continue
            task[field] = _run_root_relative(root, str(value))
    return portable


def _build_task(
    *,
    manifest: Mapping[str, Any],
    unit: Mapping[str, Any],
    case: Mapping[str, Any],
    seed: int,
    identity: RunIdentity,
    run_root: Path,
    data_root: Path,
    profiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """构造一条 ready/adapted 任务（argv、command、身份、artifact 目录）。

    per-task ``config_hash`` 由有效配置（含 case 级 runtime/task 覆盖）计算，
    保证同一实验内 TCP 1.0/1.8 等变体在续跑时不会混身份。
    """
    overrides: dict[str, Any] = dict(profiles.get(case.get("tier") or "", {}))
    overrides.update(case.get("task_overrides", {}))
    runtime_rel, runtime_info = _runtime_effective_config(
        manifest, unit, case, run_root)
    argv: dict[str, Any] = dict(manifest.get("base_argv", {}))
    argv.update(case.get("argv") or {})
    argv.update({
        "--robot-config": str(manifest["configs"]["robot"]),
        "--runtime-config": runtime_rel,
        "--racket-config": str(manifest["configs"]["racket"]),
    })
    artifact_dir = run_root / "artifacts" / str(unit["id"]) / str(case["id"]) / f"seed{seed}"
    argv["--output"] = str(artifact_dir)
    argv["--seed"] = seed
    if case["mode"] == "position":
        argv["--position-mode"] = None
    if overrides:
        argv["--task-overrides"] = [
            f"{key}={_format_override(value)}" for key, value in sorted(overrides.items())]
    # config_id 用稳定 token（不含 run_root 快照路径），保证换 --output 后仍可续跑（N15）。
    config_id = make_config_id(_stable_config_id_argv(argv, runtime_info))
    plan_tokens = {
        "{experiment_id}": str(unit["id"]),
        "{case_id}": str(case["id"]),
        "{seed}": str(seed),
        # 混合实验（同一 case 两种执行模式）必须显式传 mode，
        # 否则 extractor 无法区分样本池；plot 也按 mode 分层读写。
        "{mode}": str(case["mode"]),
    }
    extract_argv = _substitute(
        case.get("extract_argv") or [], run_root, data_root, extra=plan_tokens)
    plot_argv = _substitute(
        case.get("plot_argv") or [], run_root, data_root, extra=plan_tokens)
    metric_schema = str(manifest["identity_defaults"]["metric_schema"])
    runtime_for_hash = (runtime_info["snapshot"]
                        if runtime_info["snapshot"] else str(runtime_info["base"]))
    task_hash = compute_config_hash(
        {
            "robot": str(manifest["configs"]["robot"]),
            "runtime": str(runtime_for_hash),
            "racket": str(manifest["configs"]["racket"]),
            "task": str(manifest["configs"]["task_overlay"]),
        },
        extra={
            "manifest_schema": MANIFEST_SCHEMA,
            "metric_schema": metric_schema,
            "runtime_overrides": dict(runtime_info["overrides"]),
            "task_overrides": overrides,
        },
    )
    task_identity = RunIdentity(
        robot_id=str(identity.robot_id),
        mode=str(case["mode"]),
        racket_id=str(identity.racket_id),
        config_hash=task_hash,
        metric_schema=metric_schema,
    )
    return {
        "experiment": str(unit["id"]),
        "case_id": str(case["id"]),
        "seed": seed,
        "mode": case["mode"],
        "case_status": case["status"],
        "tier": case.get("tier"),
        "config_id": config_id,
        "config_hash": task_hash,
        "runtime": runtime_info,
        "identity": task_identity.as_dict(),
        # N14：每个任务携带可追溯的论文计划字段（T8b/T8c 直接消费）。
        "experiment_id": str(case.get("experiment_id", unit["id"])),
        "source_paper_item": str(case.get("source_paper_item", "")),
        "extract_argv": extract_argv,
        "plot_argv": plot_argv,
        "expected_outputs": [str(x) for x in case.get("expected_outputs") or []],
        "timeout_s": float(case.get("timeout_s") or manifest.get("_timeout_s_default", 900.0)),
        "argv": argv,
        "command": batch_runner.build_command(str(manifest["script"]), argv),
        # 计划路径（绝对）：`artifact_dir`/`log_path` 可能在续跑回填后被替换为
        # 旧 output 的真实路径，`planned_*` 始终保留本次 output 的计划位置（N01）。
        "artifact_dir": str(artifact_dir),
        "log_path": str(_task_log_path(run_root, str(unit["id"]), config_id, seed)),
        "planned_artifact_dir": str(artifact_dir),
        "planned_log_path": str(
            _task_log_path(run_root, str(unit["id"]), config_id, seed)),
    }


def _filter_tasks_by_mode(
    tasks: list[dict[str, Any]], mode_filter: str) -> list[dict[str, Any]]:
    """按 --mode 过滤任务（torque/position/both）。"""
    if mode_filter == "both":
        return tasks
    return [task for task in tasks if task["mode"] == mode_filter]


def resolve_manifest(
    manifest: Mapping[str, Any],
    *,
    manifest_path: str | Path | None = None,
    data_root: str | Path | None = None,
    output: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    mode: str = "torque",
    limit: int | None = None,
    timeout_s: float | None = None,
    allow_single_seed: bool = False,
) -> dict[str, Any]:
    """把 manifest 解析为本次运行的完整任务清单（不执行子进程）。

    Args:
        manifest: 已加载的 manifest 字典。
        manifest_path: 原始 manifest 路径（写入 resolved JSON 的 sha256）。
        data_root: batch CSV 数据根（默认 ``<run_root>/batch``）。
        output: 本次运行根目录（默认 manifest.run_root_default）。
        seeds: 覆盖 manifest seeds（仍受小样本规则约束，≥2）。
        mode: torque/position/both 过滤。
        limit: 只保留前 N 条任务（跨实验统一截断）。
        timeout_s: 覆盖 manifest 默认单 run 超时（秒）。
        allow_single_seed: 允许 --limit 截断出单 seed cell（仅调试，
            默认 False 时任何 <2 seed 的比较 cell 都会被拒绝）。

    Returns:
        可直接 JSON 序列化的 resolved 字典（含 units / tasks / blocked / counts）。

    Raises:
        ManifestError: manifest 不合法、配置缺失、或工具有未声明引用。
    """
    manifest = validate_manifest(manifest)
    if mode not in VALID_MODE_FILTERS:
        raise ManifestError(f"mode 必须是 {VALID_MODE_FILTERS}，得到 {mode!r}")
    manifest_file = _resolve_project_path(
        manifest_path or DEFAULT_MANIFEST_PATH)
    run_root = _resolve_project_path(output) if output is not None else _resolve_project_path(
        manifest.get("run_root_default", str(DEFAULT_RUN_ROOT)))
    batch_root = (_resolve_project_path(data_root) if data_root is not None
                  else run_root / "batch")

    manifest_seeds = _validate_seeds(seeds, "--seeds") if seeds is not None else list(
        _validate_seeds(manifest.get("seeds"), "manifest"))
    config_hash, config_details = _config_hash(manifest)
    identity = RunIdentity(
        robot_id=str(manifest["identity_defaults"]["robot_id"]),
        mode="",
        racket_id=str(manifest["identity_defaults"]["racket_id"]),
        config_hash=config_hash,
        metric_schema=str(manifest["identity_defaults"]["metric_schema"]),
    )
    profiles: dict[str, dict[str, Any]] = {
        str(name): dict(overrides or {})
        for name, overrides in manifest.get("ablation_profiles", {}).items()}
    timeout_default = float(
        timeout_s if timeout_s is not None
        else manifest.get("timeout_s_default", 900.0))
    manifest_for_tasks = dict(manifest)
    manifest_for_tasks["_timeout_s_default"] = timeout_default

    units: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    experiments = manifest.get("experiments", [])
    dependencies = manifest.get("extra_dependencies", [])
    for unit in list(experiments) + list(dependencies):
        unit_id = str(unit["id"])
        kind = str(unit.get("kind", "batch"))
        entry: dict[str, Any] = {
            "id": unit_id,
            "kind": kind,
            "title": str(unit.get("title", "")),
            "paper_ref": unit.get("paper_ref", {}) or {},
            "matrix_ref": str(unit.get("matrix_ref", "")),
            "depends_on": [str(x) for x in unit.get("depends_on", [])],
            "expected_fields": [str(x) for x in unit.get("expected_fields", [])],
            "t8b_consumer": str(unit.get("t8b_consumer", "")),
            "notes": str(unit.get("notes", "")),
            "status": "blocked" if str(unit.get("status", "")) == "blocked" else "ready",
            "cases": [],
            "task_count": 0,
        }
        if kind == "derived":
            entry["derived_from"] = str(unit.get("derived_from", ""))
            entry["status"] = "derived"
            units.append(entry)
            continue
        if kind == "tool":
            entry["expected_outputs"] = [str(x) for x in unit.get("expected_outputs", [])]
            entry["command"] = [str(tok) for tok in unit.get("command", [])]
            if entry["status"] == "blocked":
                entry["blocked_reason"] = str(unit.get("blocked_reason", ""))
            units.append(entry)
            continue
        unit_tasks: list[dict[str, Any]] = []
        active_case_count = 0
        for case in unit.get("cases", []):
            blocked_case = case["status"] == "blocked"
            case_entry = {
                "id": str(case["id"]),
                "mode": str(case["mode"]),
                "tier": case.get("tier"),
                "status": str(case["status"]),
                "runtime_overrides": dict(case.get("runtime_overrides") or {}),
                "seeds": list(case["seeds"]),
                # N14：blocked case 也显式给出计划字段（不适用处为 null）。
                "experiment_id": case.get("experiment_id") or str(unit_id),
                "source_paper_item": case.get("source_paper_item"),
                "argv": (None if blocked_case
                         else dict(case.get("argv") or {})),
                "extract_argv": (None if blocked_case
                                 else list(case.get("extract_argv") or [])),
                "plot_argv": (None if blocked_case
                              else list(case.get("plot_argv") or [])),
                "expected_outputs": (None if blocked_case
                                     else [str(x) for x in case.get("expected_outputs") or []]),
                "seed": (None if blocked_case else list(case["seeds"])),
                "timeout_s": (None if blocked_case else case.get("timeout_s")),
            }
            entry["cases"].append(case_entry)
            if case["status"] == "blocked":
                blocked.append({
                    "unit": unit_id,
                    "case_id": str(case["id"]),
                    "reason": str(case.get("blocked_reason", "")),
                })
                case_entry["blocked_reason"] = str(case.get("blocked_reason", ""))
                continue
            active_case_count += 1
            case_seeds = (list(manifest_seeds) if seeds is not None
                          else list(case["seeds"]))
            for seed in case_seeds:
                unit_tasks.append(_build_task(
                    manifest=manifest_for_tasks, unit=unit, case=case, seed=seed,
                    identity=identity, run_root=run_root, data_root=batch_root,
                    profiles=profiles))
        unit_tasks = _filter_tasks_by_mode(unit_tasks, mode)
        entry["task_count"] = len(unit_tasks)
        if unit_tasks:
            entry["status"] = "ready"
        elif active_case_count == 0:
            entry["status"] = "blocked"
        else:
            # 有可执行 case 但被 --mode/--limit 过滤掉：显式标记，避免误读为 blocked
            entry["status"] = "filtered"
        entry["tasks"] = unit_tasks
        tasks.extend(unit_tasks)
        units.append(entry)

    if limit is not None:
        if limit < 0:
            raise ManifestError(f"limit 不能为负: {limit}")
        tasks = tasks[:limit]
        kept: dict[str, int] = {}
        for task in tasks:
            kept[task["experiment"]] = kept.get(task["experiment"], 0) + 1
        for unit in units:
            unit["tasks"] = [t for t in unit.get("tasks", [])
                             if t["experiment"] in kept][: kept.get(unit["id"], 0)]
            unit["task_count"] = len(unit["tasks"])
        if not allow_single_seed:
            by_cell: dict[tuple[str, str], set[int]] = {}
            for task in tasks:
                by_cell.setdefault(
                    (str(task["experiment"]), str(task["case_id"])), set()).add(
                        int(task["seed"]))
            short = [
                f"{unit_id}/{case_id}: {sorted(seeds_in_cell)}"
                for (unit_id, case_id), seeds_in_cell in sorted(by_cell.items())
                if len(seeds_in_cell) < 2]
            if short:
                raise ManifestError(
                    "--limit 截断后出现单 seed 比较 cell（小样本规则要求每 cell ≥2 "
                    f"共同 seed）: {'; '.join(short)}。请增大 --limit；仅在调试时"
                    "显式加 --allow-single-seed。")

    ready_tasks = [t for t in tasks]
    resolved = {
        "schema": RESOLVED_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": {
            "path": _display_path(manifest_file),
            "sha256": _file_sha256(manifest_file) if manifest_file.is_file() else "",
        },
        "script": str(manifest["script"]),
        "run_root": str(run_root),
        "data_root": str(batch_root),
        "configs": config_details,
        "config_hash": config_hash,
        "identity": identity.as_dict(),
        "seeds": manifest_seeds,
        "timeout_s_default": timeout_default,
        "mode_filter": mode,
        "limit": limit,
        "allow_single_seed": bool(allow_single_seed),
        "units": units,
        "tasks": ready_tasks,
        "blocked": blocked,
        "counts": {
            "experiments": len(experiments),
            "dependencies": len(dependencies),
            "batch_tasks": len(ready_tasks),
            "torque_tasks": sum(1 for t in ready_tasks if t["mode"] == "torque"),
            "position_tasks": sum(1 for t in ready_tasks if t["mode"] == "position"),
            "blocked_cases": len(blocked),
            "derived_units": sum(1 for u in units if u["kind"] == "derived"),
            "tool_units": sum(1 for u in units if u["kind"] == "tool"),
        },
    }
    return resolved


def build_specs_from_manifest(
    manifest_path: str | Path | None = None,
    *,
    data_root: str | Path | None = None,
    output: str | Path | None = None,
    seeds: Sequence[int] | None = None,
    mode: str = "torque",
    limit: int | None = None,
    timeout_s: float | None = None,
    workers: int = REQUIRED_WORKERS,
    allow_single_seed: bool = False,
) -> list[ExperimentSpec]:
    """构建 R1D 论文 smoke 的 ExperimentSpec 列表（供 overnight 工厂复用）。

    每个 manifest 实验（及 batch 型额外依赖）生成一个 spec，grid 为解析后的
    任务参数；``_timeout`` meta 写入任务级超时。R1D 强制 workers=1。

    Args:
        manifest_path: manifest 路径（默认 configs/experiments/r1d_paper_smoke.yaml）。
        data_root: batch CSV 数据根（默认 ``<output>/batch``）。
        output: 本次运行根目录（默认 manifest.run_root_default）。
        seeds: 覆盖 manifest seeds。
        mode: torque/position/both。
        limit: 只保留前 N 条任务。
        timeout_s: 覆盖默认超时。
        workers: 预期并行度（R1D 必须为 1）。
        allow_single_seed: 允许 --limit 截断出单 seed cell（仅调试）。

    Returns:
        ExperimentSpec 列表（name=实验/依赖 id；identity 携带 config_hash）。

    Raises:
        ManifestError: manifest 不合法或配置缺失。
        ValueError: workers != 1。
    """
    if workers != REQUIRED_WORKERS:
        raise ValueError(
            f"R1D 论文 smoke 要求 workers={REQUIRED_WORKERS}（串行，避免多进程 MuJoCo 问题），"
            f"得到 workers={workers}")
    manifest = load_manifest(manifest_path)
    resolved = resolve_manifest(
        manifest, manifest_path=manifest_path, data_root=data_root, output=output,
        seeds=seeds, mode=mode, limit=limit, timeout_s=timeout_s,
        allow_single_seed=allow_single_seed)
    identity = RunIdentity(
        robot_id=str(resolved["identity"]["robot_id"]),
        mode="",
        racket_id=str(resolved["identity"]["racket_id"]),
        config_hash=str(resolved["identity"]["config_hash"]),
        metric_schema=str(resolved["identity"]["metric_schema"]),
    )
    specs: list[ExperimentSpec] = []
    for unit in resolved["units"]:
        if unit["kind"] != "batch" or not unit.get("tasks"):
            continue
        grid: list[dict[str, Any]] = []
        for task in unit["tasks"]:
            params = dict(task["argv"])
            params["_timeout"] = task["timeout_s"]
            # per-task 有效配置 hash（含 runtime/task 覆盖）：batch_runner 用它
            # 覆盖身份，保证 TCP 1.0/1.8 或 ablation 档不会混续跑。
            params["_config_hash"] = str(task["config_hash"])
            # 保留 manifest 已算好的稳定 config_id（F05）：batch 不得按含绝对
            # --output/快照路径的 argv 重算，否则换 output 后 resume 失效。
            params["_config_id"] = str(task["config_id"])
            grid.append(params)
        specs.append(ExperimentSpec(
            name=str(unit["id"]),
            script=str(resolved["script"]),
            grid=grid,
            report_ref=f"paper_smoke:{unit['id']}",
            identity=identity,
        ))
    return specs


# ---------------------------------------------------------------------------
# 执行与汇总
# ---------------------------------------------------------------------------


def _substitute(
    tokens: Sequence[str],
    run_root: Path,
    data_root: Path,
    extra: Mapping[str, str] | None = None,
) -> list[str]:
    """替换工具依赖/计划命令模板中的占位符。

    Args:
        tokens: 命令 token 列表（str/Path 等可转字符串值）。
        run_root: 本次运行根目录，替换 ``{run_root}``。
        data_root: batch 数据根，替换 ``{data_root}``。
        extra: 额外占位符映射（如 ``{experiment_id}``/``{case_id}``/``{seed}``）。

    Returns:
        替换后的字符串列表（顺序与输入一致）。
    """
    mapping = {
        "{python}": sys.executable,
        "{run_root}": str(run_root),
        "{data_root}": str(data_root),
        "{project}": str(PROJECT),
        # D 线（提取/绘图）消费端通过 --catalog 解析路径，不再猜父目录（F03）。
        "{catalog}": str(Path(run_root) / "catalog.json"),
    }
    if extra:
        mapping.update({str(key): str(value) for key, value in extra.items()})
    out: list[str] = []
    for token in tokens:
        value = str(token)
        for key, replacement in mapping.items():
            value = value.replace(key, replacement)
        out.append(value)
    return out


def _check_tool_outputs(
    patterns: Sequence[str], run_root: Path,
) -> list[str]:
    """校验工具依赖声明的 expected_outputs 是否真的生成。

    支持目录、文件与 ``*``/``?``/``[`` 通配；相对路径按 run_root 解析。
    返回缺失的模式列表（空列表表示全部存在）。
    """
    missing: list[str] = []
    for pattern in patterns:
        rel = str(pattern)
        target = Path(rel)
        if not target.is_absolute():
            target = run_root / rel
        if any(ch in rel for ch in "*?["):
            parent = target.parent
            match = None
            if parent.is_dir():
                match = next(parent.glob(target.name), None)
            if match is None:
                missing.append(rel)
        elif not target.exists():
            missing.append(rel)
    return missing


def _run_tool_dependency(
    unit: Mapping[str, Any], run_root: Path, data_root: Path,
) -> dict[str, Any]:
    """执行单个 ready 的工具型额外依赖，返回状态摘要。

    退出码 0 之外还校验 ``expected_outputs``：工具脚本可能“成功退出但
    没产出”（如 collect_fig_assets 跳过某目标），必须显式失败（T8a 评审 N5）。
    """
    unit_id = str(unit["id"])
    if unit.get("status") == "blocked":
        return {"id": unit_id, "status": "blocked",
                "blocked_reason": str(unit.get("blocked_reason", ""))}
    command = _substitute(unit.get("command", []), run_root, data_root)
    log_dir = run_root / "logs" / "_dependencies"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{unit_id}.log"
    timeout = float(unit.get("timeout_s", 3600.0))
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout,
            cwd=str(PROJECT), encoding="utf-8", errors="replace")
        text = "# command: " + " ".join(command) + "\n\n" + result.stdout + result.stderr
        log_path.write_text(text, encoding="utf-8", errors="replace")
        status = "ok" if result.returncode == 0 else "error"
        payload: dict[str, Any] = {
            "id": unit_id,
            "status": status,
            "returncode": result.returncode,
            "command": command,
            "log_path": _display_path(log_path),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }
        if status == "ok":
            missing = _check_tool_outputs(
                [str(p) for p in unit.get("expected_outputs", [])], run_root)
            if missing:
                payload["status"] = "invalid_output"
                payload["missing_outputs"] = missing
        return payload
    except subprocess.TimeoutExpired:
        log_path.write_text(
            "# command: " + " ".join(command) + f"\n\nTIMEOUT after {timeout}s\n",
            encoding="utf-8")
        return {
            "id": unit_id,
            "status": "timeout",
            "returncode": -1,
            "command": command,
            "log_path": _display_path(log_path),
            "elapsed_s": round(time.perf_counter() - started, 2),
        }


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
    row_log = _row_path(run_root, row.get("log_path"))
    if row_log is not None and row_log.is_file():
        return row_log
    task_log = _row_path(run_root, task.get("log_path")) if task else None
    if task_log is not None and task_log.is_file():
        return task_log
    return task_log or row_log


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


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """写契约 JSON（UTF-8、缩进 2、不转义中文）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _record_key(row: Mapping[str, Any]) -> tuple[str, str, int] | None:
    """从 runs.csv 行提取 ``(experiment, config_id, seed)`` 续跑键。

    Args:
        row: CSV 行（字段可能缺失或为空）。

    Returns:
        完整键；缺 experiment/config_id 或 seed 非法时为 None。
    """
    experiment = str(row.get("experiment") or "")
    config_id = str(row.get("config_id") or "")
    try:
        seed = int(row.get("seed"))
    except (TypeError, ValueError):
        return None
    if not experiment or not config_id:
        return None
    return (experiment, config_id, seed)


def _records_index(
    records: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """按续跑键索引 runs.csv 记录（后写覆盖先写）。

    Args:
        records: ``_collect_runs`` 返回的原始记录（绝对路径）。

    Returns:
        续跑键到 CSV 行的映射。
    """
    index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for row in records:
        key = _record_key(row)
        if key is not None:
            index[key] = row
    return index


def _resolved_task_index(
    resolved: Mapping[str, Any],
) -> dict[tuple[str, str, int], Mapping[str, Any]]:
    """按续跑键索引 manifest 任务。

    Args:
        resolved: ``resolve_manifest`` 的结果。

    Returns:
        续跑键到任务字典的映射。
    """
    index: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for task in _iter_resolved_tasks(resolved):
        key = _record_key(task)
        if key is not None:
            index[key] = task
    return index


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


#: ``--task-overrides`` 前缀到 payload["config"] 节的映射（R02）
_OVERRIDE_SECTION_FIELDS: dict[str, str] = {
    "mpc": "mpc_config",
    "solver": "solver_config",
    "cost": "cost_config",
    "perception": "perception_config",
    "hitting": "hitting_config",
    "serve": "serve_config",
}
#: target.* 覆盖在 config 顶层的同义字段（``shoulder_pos`` 单独落在 scene）
_TARGET_FIELD_ALIASES: dict[str, str] = {
    "center": "target_center",
    "offset": "target_offset",
    "workspace_radius": "workspace_radius",
}
#: ``serve.use_bounce``/``serve.bounce_restitution`` 实际写入 scene 节
_SERVE_SCENE_ALIASES = frozenset({"use_bounce", "bounce_restitution"})
#: 在 config 中找不到某个点路径时的哨兵
_MISSING = object()


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
        return [
            "日志缺少 config_id 头部"
            f"（期望 '# config_id: {config_id}'，文件 {path.as_posix()}）"]
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


def _catalog_payload(
    resolved: Mapping[str, Any], run_root: Path,
) -> dict[str, Any]:
    """构造 output 根的 ``catalog.json`` 声明（schema_version=1）。

    路径相对 catalog 文件本身声明，使整棵 run_root 可移动；``data_root``
    位于 run_root 内时写相对路径、外部时写绝对路径（不搬旧数据）。
    """
    source_manifest = resolved.get("source_manifest") or {}
    data_root = Path(str(resolved["data_root"]))
    identity = resolved.get("identity") or {}
    return {
        "schema_version": 1,
        "synthetic": False,
        "robot_id": str(identity.get("robot_id", "")),
        "run_root": ".",
        "data_root": _run_root_relative(run_root, data_root),
        "runs_csv": "runs.csv",
        "manifest_path": "manifest_resolved.json",
        "path_base": "run_root",
        "source_sets": dict(SOURCE_SETS),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_manifest": {
            "path": str(source_manifest.get("path", "")),
            "sha256": str(source_manifest.get("sha256", "")),
        },
        "config_hash": str(resolved.get("config_hash", "")),
        "seeds": [int(seed) for seed in resolved.get("seeds", [])],
    }


def _write_catalog(run_root: Path, resolved: Mapping[str, Any]) -> Path:
    """在 output 根写 ``catalog.json``，返回其路径。"""
    path = Path(run_root) / "catalog.json"
    _write_json(path, _catalog_payload(resolved, Path(run_root)))
    return path


def _print_resolved_summary(resolved: Mapping[str, Any], *, dry_run: bool) -> None:
    """打印解析结果摘要（dry-run 时打印完整任务命令）。"""
    counts = resolved["counts"]
    print(f"[r1d-smoke] manifest={resolved['source_manifest']['path']} "
          f"manifest_sha256={resolved['source_manifest']['sha256'][:12]}…")
    print(f"[r1d-smoke] run_root={_display_path(resolved['run_root'])} "
          f"data_root={_display_path(resolved['data_root'])}")
    print(f"[r1d-smoke] config_hash={resolved['config_hash']} "
          f"identity={resolved['identity']}")
    print(f"[r1d-smoke] seeds={resolved['seeds']} mode={resolved['mode_filter']} "
          f"timeout_s={resolved['timeout_s_default']:g}")
    print(f"[r1d-smoke] 实验 {counts['experiments']} 个 + 额外依赖 "
          f"{counts['dependencies']} 个；batch 任务 {counts['batch_tasks']} 条"
          f"（torque {counts['torque_tasks']} / position {counts['position_tasks']}）；"
          f"blocked case {counts['blocked_cases']} 个；derived {counts['derived_units']}；"
          f"tool {counts['tool_units']}")
    for unit in resolved["units"]:
        kind = unit["kind"]
        if kind == "batch":
            print(f"  [{unit['status']:8s}] {unit['id']:34s} "
                  f"cases={len(unit['cases']):2d} tasks={unit['task_count']:3d}")
        elif kind == "derived":
            print(f"  [derived ] {unit['id']:34s} derived_from={unit.get('derived_from', '')}")
        else:
            print(f"  [{unit['status']:8s}] {unit['id']:34s} tool")
    for blocked in resolved["blocked"]:
        print(f"  [blocked] {blocked['unit']}/{blocked['case_id']}: {blocked['reason']}")
    if dry_run:
        print("[r1d-smoke] case 计划字段（N14：每 case 可核对 experiment_id/"
              "source_paper_item/argv/extract_argv/plot_argv/expected_outputs/"
              "mode/seed/timeout_s；blocked case 显式 null）：")

        def _show(value: Any) -> str:
            """把 case 计划字段渲染为单行 JSON（None 显式输出 null）。"""
            return "null" if value is None else json.dumps(
                value, ensure_ascii=False, sort_keys=True)

        for unit in resolved["units"]:
            if unit["kind"] != "batch":
                continue
            for case in unit["cases"]:
                print(f"  CASE {unit['id']}/{case['id']}: "
                      f"status={case['status']} "
                      f"experiment_id={case.get('experiment_id')} "
                      f"source_paper_item={case.get('source_paper_item')} "
                      f"mode={case['mode']} seed={_show(case.get('seed'))} "
                      f"timeout_s={case.get('timeout_s')} "
                      f"argv={_show(case.get('argv'))} "
                      f"expected_outputs={_show(case.get('expected_outputs'))} "
                      f"extract_argv={_show(case.get('extract_argv'))} "
                      f"plot_argv={_show(case.get('plot_argv'))}")
        for task in resolved["tasks"]:
            print(f"  DRY-RUN [{task['experiment']}/{task['case_id']}] "
                  f"seed={task['seed']} mode={task['mode']}: "
                  + " ".join(str(x) for x in task["command"]))
            print("           extract_argv: "
                  + " ".join(str(x) for x in task["extract_argv"]))
            print("           plot_argv: "
                  + " ".join(str(x) for x in task["plot_argv"]))
        for unit in resolved["units"]:
            if unit["kind"] == "tool" and unit["status"] == "ready":
                command = _substitute(unit["command"], Path(resolved["run_root"]),
                                      Path(resolved["data_root"]))
                print(f"  DRY-RUN [tool {unit['id']}]: " + " ".join(command))


def main(argv: Sequence[str] | None = None) -> int:
    """CLI 入口：校验并（可选）执行 R1D 论文 smoke。

    Args:
        argv: 参数列表（None 时用 ``sys.argv[1:]``）。

    Returns:
        进程退出码：0 成功（含 dry-run）；1 存在 batch error/timeout、工具失败
        或续跑产物校验失败；2 manifest/参数错误。
    """
    parser = argparse.ArgumentParser(
        description="R1D 论文 smoke 运行入口（manifest 驱动，T8a）")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH,
                        help="smoke manifest 路径")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="batch CSV 数据根（默认 <output>/batch）")
    parser.add_argument("--output", type=Path, default=None,
                        help="本次运行根目录（默认 experiment_data/r1d_v12_parity/paper_smoke）")
    parser.add_argument("--workers", type=int, default=None,
                        help="并行进程数；R1D 论文 smoke 强制为 1")
    parser.add_argument("--timeout", type=float, default=None,
                        help="覆盖 manifest 默认单 run 超时（秒）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只校验/解析并打印完整任务清单，不执行子进程")
    parser.add_argument("--limit", type=int, default=None,
                        help="只保留前 N 条任务（调试用；若截断出单 seed cell 会失败，"
                             "除非显式加 --allow-single-seed）")
    parser.add_argument("--allow-single-seed", action="store_true",
                        help="仅调试：允许 --limit 截断出单 seed 的比较 cell（默认拒绝）")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="按身份命名空间续跑（默认）")
    parser.add_argument("--no-resume", dest="resume", action="store_false",
                        help="忽略已完成的 runs 重新执行")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="覆盖 manifest seeds（小样本规则要求 ≥2 个共同 seed）")
    parser.add_argument("--mode", choices=VALID_MODE_FILTERS, default="torque",
                        help="只跑 torque / position / both（默认 torque）")
    parser.add_argument("--legacy-resume", action="store_true",
                        help="允许续跑旧 21 列 CSV（仅旧 RM65 身份；R1D 不应使用）")
    args = parser.parse_args(argv)

    try:
        manifest = load_manifest(args.manifest)
        workers = (args.workers if args.workers is not None
                   else int(manifest.get("workers_default", REQUIRED_WORKERS)))
        if workers != REQUIRED_WORKERS:
            parser.error(
                f"R1D 论文 smoke 必须 --workers {REQUIRED_WORKERS}"
                f"（串行执行，避免多进程 MuJoCo 段错误），得到 {workers}")
        resolved = resolve_manifest(
            manifest, manifest_path=args.manifest, data_root=args.data_root,
            output=args.output, seeds=args.seeds, mode=args.mode,
            limit=args.limit, timeout_s=args.timeout,
            allow_single_seed=args.allow_single_seed)
    except ManifestError as exc:
        print(f"[r1d-smoke] manifest 校验失败: {exc}", file=sys.stderr)
        return 2

    run_root = Path(resolved["run_root"])
    resolved_path = run_root / "manifest_resolved.json"
    # M06：manifest 落盘用可搬移版本（run_root="."、内部路径相对），
    # 内存 resolved 继续保留绝对路径供执行与回填。
    _write_json(resolved_path, _portable_resolved(resolved, run_root))
    print(f"[r1d-smoke] 已写 {_display_path(resolved_path)}")
    _print_resolved_summary(resolved, dry_run=args.dry_run)
    if args.dry_run:
        print("[r1d-smoke] dry-run 完成：未执行任何子进程、未写 artifacts"
              "（仅写 manifest_resolved.json 与 runtime 覆盖快照）")
        return 0

    batch_root = Path(resolved["data_root"])
    specs = build_specs_from_manifest(
        args.manifest, data_root=args.data_root, output=args.output,
        seeds=args.seeds, mode=args.mode, limit=args.limit,
        timeout_s=args.timeout, workers=workers,
        allow_single_seed=args.allow_single_seed)
    spec_by_name = {spec.name: spec for spec in specs}
    summaries: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    started = time.perf_counter()
    for unit in resolved["units"]:
        if unit["kind"] == "tool":
            result = _run_tool_dependency(unit, run_root, batch_root)
            tool_results.append(result)
            print(f"[r1d-smoke] 工具依赖 {unit['id']}: {result['status']} "
                  f"log={result.get('log_path', '')}")
            continue
        if unit["kind"] != "batch" or unit["task_count"] == 0:
            continue
        spec = spec_by_name.get(str(unit["id"]))
        if spec is None:  # pragma: no cover - 解析与构建不一致属内部错误
            raise RuntimeError(f"内部错误：任务清单与 spec 不一致: {unit['id']}")
        try:
            summary = batch_runner.run_batch(
                spec,
                workers=workers,
                resume=args.resume,
                limit=None,
                # R10：只传 data_root（--data-root 优先，缺省 <run_root>/batch）。
                # 不再传优先级更高的 output=，否则 --data-root 会被静默覆盖，
                # 依赖/resume/_collect_runs 也会落到不同根。
                data_root=batch_root,
                timeout_s=resolved["timeout_s_default"],
                dry_run=False,
                allow_legacy_resume=args.legacy_resume,
                log_dir=run_root / "logs",
            )
        except ValueError as exc:
            # 身份/CSV 隔离契约拒绝混写：非零退出，不写 catalog/summary 掩盖冲突。
            print(f"[r1d-smoke] batch 拒绝执行 {unit['id']}: {exc}",
                  file=sys.stderr)
            return 2
        summaries.append(summary)
        print(f"[r1d-smoke] {spec.name}: launched={summary['launched']} "
              f"done_before={summary['done_before']} hits={summary['hits_this_run']} "
              f"errors={summary['errors']} timeouts={summary['timeouts']} "
              f"elapsed={summary['elapsed_s']}s")
    runs = _collect_runs(resolved, batch_root, summaries)
    # N01：换 output 续跑（launched=0）时，把 manifest 的计划路径替换为旧 CSV
    # 里真实存在的 artifact/log；本次计划位置保留在 planned_* 字段。
    updated = _apply_actual_paths(resolved, runs.get("records") or [], run_root)
    # M09：续跑跳过的已完成行必须产物真实完好，否则静默 0 退出会掩盖数据缺失。
    artifact_problems = _validate_completed_artifacts(
        resolved, runs.get("records") or [], run_root)
    if updated:
        _write_json(resolved_path, _portable_resolved(resolved, run_root))
        print(f"[r1d-smoke] 已按旧 CSV 实际路径回填 {updated} 条任务并重写 "
              f"{_display_path(resolved_path)}")
    _print_artifact_problems(artifact_problems)
    catalog_path = _write_catalog(run_root, resolved)
    print(f"[r1d-smoke] 已写 {_display_path(catalog_path)}")
    totals = {
        "hits": sum(int(s.get("hits_this_run", 0)) for s in summaries),
        "errors": sum(int(s.get("errors", 0)) for s in summaries),
        "timeouts": sum(int(s.get("timeouts", 0)) for s in summaries),
        "tool_failures": sum(1 for r in tool_results
                             if r["status"] not in ("ok", "blocked")),
        # M09：续跑跳过但产物缺失/损坏的任务数（与 batch errors 分开统计）。
        "artifact_errors": len(artifact_problems),
        "elapsed_s": round(time.perf_counter() - started, 2),
    }
    summary_payload = {
        "schema": "r1d_paper_smoke_summary_v1",
        "run_root": resolved["run_root"],
        "config_hash": resolved["config_hash"],
        "seeds": resolved["seeds"],
        "mode_filter": resolved["mode_filter"],
        "batch": {"runs_csv": runs["runs_csv"], "runs": runs["runs"],
                  "hit_semantics": runs["hit_semantics"]},
        # F03：output 根的 catalog 声明（提取/绘图用 --catalog 消费）。
        "catalog": _display_path(catalog_path),
        "experiments": summaries,
        "tool_dependencies": tool_results,
        "totals": totals,
    }
    _write_json(run_root / "run_summary.json", summary_payload)
    print(f"[r1d-smoke] runs.csv={runs['runs_csv']} rows={runs['runs']}；"
          f"run_summary.json 已写")
    print(f"[r1d-smoke] 合计 hits={totals['hits']} errors={totals['errors']} "
          f"timeouts={totals['timeouts']} tool_failures={totals['tool_failures']} "
          f"artifact_errors={totals['artifact_errors']} "
          f"elapsed={totals['elapsed_s']}s")
    return 1 if (totals["errors"] or totals["timeouts"]
                 or totals["tool_failures"]
                 or totals["artifact_errors"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
