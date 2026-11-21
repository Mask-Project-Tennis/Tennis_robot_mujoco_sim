"""manifest 读取、校验与规范化（R1D 论文 smoke 的数据契约）。

迁移自 ``scripts/exp/run_r1d_paper_smoke.py``（C1 拆分），行为逐字保持：
ManifestError、路径解析、override 白名单、seeds/units/cases 校验与
``validate_manifest`` 的规范化输出。
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from scripts.exp import batch_runner

_ROOT = Path(__file__).resolve().parents[3]  # 仓库根目录


PROJECT = batch_runner.PROJECT

MANIFEST_SCHEMA = "r1d_paper_smoke_v1"

RESOLVED_SCHEMA = "r1d_paper_smoke_resolved_v1"

DEFAULT_MANIFEST_PATH = PROJECT / "configs" / "experiments" / "r1d_paper_smoke.yaml"

DEFAULT_RUN_ROOT = PROJECT / "experiment_data" / "r1d_v12_parity" / "paper_smoke"

SOURCE_SETS: dict[str, str] = {"nominal_mechanism": "exp17d_mechanism"}

ALLOWED_CASE_ARGV = frozenset(
    {"--serve-box", "--no-plot", "--ball-speed", "--position-mode"})

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

CASE_DEFAULT_KEYS = (
    "source_paper_item",
    "experiment_id",
    "extract_argv",
    "plot_argv",
    "expected_outputs",
    "seed",
    "timeout_s",
)

RUNS_HIT_SEMANTICS = "legacy"

RUNTIME_SNAPSHOT_DIR = "runtime_overrides"

V12_EPISODE_NAME = "v12_episode.json"

V12_EPISODE_NPZ_NAME = "v12_episode.npz"

R1D_EPISODE_NPZ_NAME = "r1d_episode.npz"

V12_NPZ_KEYS = (
    "schema", "X", "U", "ball_pos", "ball_vel", "time", "racket_pos",
)

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

REQUIRED_WORKERS = 1

_OVERRIDE_KEYS_CACHE: frozenset[str] | None = None


class ManifestError(ValueError):
    """manifest 结构或语义不合法（运行前即失败，不产生部分产物）。"""


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
            from src.sim.configuration.validation import allowed_override_keys
        except Exception as exc:  # pragma: no cover - 环境异常路径
            raise ManifestError(
                f"无法加载 V12 override 白名单（src.sim.configuration.validation）: {exc}") from exc
        _OVERRIDE_KEYS_CACHE = allowed_override_keys()
    return _OVERRIDE_KEYS_CACHE


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
                f"（见 src/sim/configuration/validation.py::allowed_override_keys）")
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
