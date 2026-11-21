"""统计口径：Wilson 区间、McNemar 精确检验、配对 bootstrap 与 DiD 基础量。

命中判定统一走 ``physical_of`` / ``HitPolicy``；tier 解析走 identity，本模块
只做数值与配对索引，不改动输入行。
"""
from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Callable, Hashable
from typing import Any

from .identity import tier_of
from .types import HitPolicy, _to_bool_or_none, cfg_of, is_error

# ----------------------------------------------------------------------------
# 基础统计工具
# ----------------------------------------------------------------------------

def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 置信区间（返回百分比 pp）。"""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d




def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确检验（双侧，二项分布）。

    Args:
        b: A 命中且 B 未命中的配对数。
        c: A 未命中且 B 命中的配对数。

    Returns:
        双侧 p 值。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(2 * tail, 1.0)




def paired_bootstrap(
    hits_a: list[int], hits_b: list[int], n_boot: int = 20000, seed: int = 0
) -> tuple[float, float]:
    """配对 bootstrap：对 seed 重采样，返回命中率差的 95% 分位区间（pp）。"""
    rng = random.Random(seed)
    n = len(hits_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        pa = sum(hits_a[i] for i in idx)
        pb = sum(hits_b[i] for i in idx)
        diffs.append((pa - pb) / n * 100)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]




def sign_test(dids: list[int]) -> float:
    """非零配对差上的符号检验（DiD 用）。"""
    nz = [d for d in dids if d != 0]
    if not nz:
        return 1.0
    pos = sum(1 for d in nz if d > 0)
    k = min(pos, len(nz) - pos)
    tail = sum(math.comb(len(nz), i) for i in range(k + 1)) / 2 ** len(nz)
    return min(2 * tail, 1.0)




def physical_of(row: dict[str, str]) -> int | None:
    """physical 口径命中：1 / 0；无判定（缺失或不可用）返回 None。

    R03：以 ``physical_hit_type`` 为准 —— 只有 ``active`` 计命中，
    ``passive``/``miss`` 一律不计；``physical_hit`` 布尔仅在类型缺失或未知时兜底，
    避免"被动接触被布尔列误计为主动命中"。
    """
    ptype = (row.get("physical_hit_type") or "").strip().lower()
    if ptype == "active":
        return 1
    if ptype in ("passive", "miss"):
        return 0
    hit = _to_bool_or_none(row.get("physical_hit"))
    if hit is not None:
        return 1 if hit else 0
    return None




def cell_stats(rows: list[dict[str, str]], hitfn: HitPolicy) -> dict[str, Any]:
    """统计一格的样本构成与命中率（含 Wilson CI、passive 计数）。"""
    errs: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for r in rows:
        if is_error(r):
            errs[(r.get("error") or "").strip()] += 1
        else:
            types[(r.get("hit_type") or "").strip()] += 1
    n_valid = sum(types.values())
    hits = sum(hitfn(r) for r in rows if not is_error(r))
    lo, hi = wilson(hits, n_valid)
    return {
        "rows": len(rows),
        "n_valid": n_valid,
        "hits": hits,
        "rate": 100 * hits / n_valid if n_valid else float("nan"),
        "ci95": [lo, hi],
        "active": types.get("active", 0),
        "passive": types.get("passive", 0),
        "miss": types.get("miss", 0),
        "passive_share": 100 * types.get("passive", 0) / n_valid if n_valid else 0.0,
        "errors": dict(errs),
    }




def physical_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """physical 口径统计：有判定即给真实 rate（0/N 记 0.0），无判定才 rate=null。

    R03：``physical_hit=True + type=passive`` 不计主动命中，该 cell 得到
    ``n_active=0, n_passive=1, rate=0.0``，并保留 ``unavailable_reason="all_miss"``
    供表格标注。0/N 是测得的 0%，不是"缺数据"，故不再返回 null；只有完全
    没有 physical 判定（如旧 RM65 无该列）时 rate 才为 null，不伪造 0.0。
    CI 仅在有命中时报告（全 miss 与 R06 配对口径一致，不给零命中区间）。
    """
    verdicts = [physical_of(r) for r in rows if not is_error(r)]
    seen = [v for v in verdicts if v is not None]
    n_valid = len(verdicts)
    n_active = sum(seen)
    n_passive = sum(
        1 for r in rows
        if not is_error(r) and physical_of(r) == 0
        and (r.get("physical_hit_type") or "").strip().lower() == "passive"
    )
    reason = None
    rate = None
    if not seen:
        reason = "no_physical_verdict"
    else:
        rate = 100 * n_active / len(seen)
        if n_active == 0:
            reason = "all_miss"
    return {
        "rate": rate,
        "denominator": len(seen),
        "n_valid": n_valid,
        "n_missing_verdict": n_valid - len(seen),
        "n_active": n_active,
        "n_passive": n_passive,
        "n_miss": len(seen) - n_active - n_passive,
        "ci95": list(wilson(n_active, len(seen))) if n_active else None,
        "unavailable_reason": reason,
        "hit_semantics": "physical",
    }




def _unique_index(
    rows: list[dict[str, str]],
    keyfn: Callable[[dict[str, str]], Hashable],
    valuefn: Callable[[dict[str, str]], int | None],
) -> tuple[dict[Hashable, int], list[Hashable]]:
    """按配对键建立去重取值索引（M02：行序无关，禁止后写覆盖）。

    同一个 key 出现多次时：取值全相同则去重（保留一份）；取值不同则把该 key
    整体排除（无法判定哪一行才是真值），并在返回值里报告。

    Args:
        rows: 已过滤的候选行。
        keyfn: 配对键（``paired_test`` 用 seed 字符串，DiD 用 seed 整数；只要求可哈希）。
        valuefn: 取值函数；返回 None 的行直接跳过（例如缺 physical 判定）。

    Returns:
        ``(index, conflicts)``：index 为 key → 取值；conflicts 为因取值冲突被
        排除的 key 列表（升序）。
    """
    groups: dict[Hashable, set[int]] = {}
    for row in rows:
        value = valuefn(row)
        if value is None:
            continue
        groups.setdefault(keyfn(row), set()).add(value)
    index: dict[Hashable, int] = {}
    conflicts: list[Hashable] = []
    for key in sorted(groups):
        values = groups[key]
        if len(values) == 1:
            index[key] = next(iter(values))
        else:
            conflicts.append(key)
    return index, conflicts




def paired_test(
    rows: list[dict[str, str]],
    hitfn: HitPolicy,
    tier_a: str,
    tier_b: str,
    keyfn: Callable[[dict[str, str]], str] = lambda r: r["seed"],
    filterfn: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, Any]:
    """same-seed 配对比较：McNemar 精确检验 + 配对 bootstrap 95% CI。

    M02：同一 seed 在同一档重复出现时不再后写覆盖——同值去重、异值整键排除，
    因此结果与 CSV 行序无关；被排除的键在 ``excluded_duplicate_keys`` 中报告。

    Returns:
        含 n_paired / diff_pp / ci95 / p_mcnemar / 各档率的字典；无共同 seed 时
        ``unavailable_reason`` 为 ``no_common_seed`` 或 ``duplicate_seed_conflict``。
    """
    sel = [r for r in rows if (filterfn(r) if filterfn else True) and not is_error(r)]
    rows_a = [r for r in sel if tier_of(r) == tier_a]
    rows_b = [r for r in sel if tier_of(r) == tier_b]
    da, conflicts_a = _unique_index(rows_a, keyfn, hitfn)
    db, conflicts_b = _unique_index(rows_b, keyfn, hitfn)
    excluded = sorted(set(conflicts_a) | set(conflicts_b))
    keys = sorted(set(da) & set(db))
    if not keys:
        out: dict[str, Any] = {
            "n_paired": 0,
            "unavailable_reason": (
                "duplicate_seed_conflict" if excluded else "no_common_seed"
            ),
        }
        if excluded:
            out["excluded_duplicate_keys"] = excluded
        return out
    ha = [da[k] for k in keys]
    hb = [db[k] for k in keys]
    b = sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1)
    lo, hi = paired_bootstrap(ha, hb)
    result: dict[str, Any] = {
        "n_paired": len(keys),
        "rate_a": 100 * sum(ha) / len(keys),
        "rate_b": 100 * sum(hb) / len(keys),
        "diff_pp": 100 * (sum(ha) - sum(hb)) / len(keys),
        "ci95": [lo, hi],
        "p_mcnemar": mcnemar_exact(b, c),
        "discordant": {"a_hit_b_miss": b, "a_miss_b_hit": c},
    }
    if excluded:
        result["excluded_duplicate_keys"] = excluded
    return result




def _physical_pair_summary(ha: list[int], hb: list[int]) -> dict[str, Any]:
    """physical 口径配对结果汇总（R06：两档共同 seed 全 miss 时显式不可估计）。

    Args:
        ha: 档 A 每个共同 seed 的 physical 命中（0/1）。
        hb: 档 B 每个共同 seed 的 physical 命中（0/1）。

    Returns:
        n_paired / rate_a / rate_b / diff_pp / discordant / hit_semantics 恒有；
        ci95 / p_mcnemar 在可估计时为数值，两档全 miss 时为 None 并给
        ``unavailable_reason="all_miss"``（不输出 [0,0] 零宽区间、不伪造 p=1）。
    """
    n = len(ha)
    total_a = sum(ha)
    total_b = sum(hb)
    b = sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1)
    out: dict[str, Any] = {
        "n_paired": n,
        "rate_a": 100 * total_a / n,
        "rate_b": 100 * total_b / n,
        "diff_pp": 100 * (total_a - total_b) / n,
        "discordant": {"a_hit_b_miss": b, "a_miss_b_hit": c},
        "hit_semantics": "physical",
    }
    if total_a == 0 and total_b == 0:
        out["ci95"] = None
        out["p_mcnemar"] = None
        out["unavailable_reason"] = "all_miss"
        return out
    out["ci95"] = list(paired_bootstrap(ha, hb))
    out["p_mcnemar"] = mcnemar_exact(b, c)
    return out




def paired_test_physical(
    rows: list[dict[str, str]],
    tier_a: str,
    tier_b: str,
    keyfn: Callable[[dict[str, str]], str] = lambda r: r["seed"],
    filterfn: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, Any]:
    """physical 口径的同 seed 配对比较（缺判定/无共同 seed/全 miss 显式不可估计）。

    M02：同一 seed 重复出现时同值去重、异值整键排除，结果与行序无关；
    冲突键在 ``excluded_duplicate_keys`` 中报告。
    """
    sel = [r for r in rows if (filterfn(r) if filterfn else True) and not is_error(r)]
    rows_a = [r for r in sel if tier_of(r) == tier_a]
    rows_b = [r for r in sel if tier_of(r) == tier_b]
    da, conflicts_a = _unique_index(rows_a, keyfn, physical_of)
    db, conflicts_b = _unique_index(rows_b, keyfn, physical_of)
    excluded = sorted(set(conflicts_a) | set(conflicts_b))
    keys = sorted(set(da) & set(db))
    if not keys:
        if excluded:
            return {
                "n_paired": 0,
                "unavailable_reason": "duplicate_seed_conflict",
                "excluded_duplicate_keys": excluded,
            }
        return {
            "n_paired": 0,
            "unavailable_reason": "no_common_seed" if (da or db) else "no_physical_verdict",
        }
    summary = _physical_pair_summary([da[k] for k in keys], [db[k] for k in keys])
    if excluded:
        summary["excluded_duplicate_keys"] = excluded
    return summary




def baseline_cell(row: dict[str, str]) -> bool:
    """exp17a 的「无噪声、无 KF」基线条件（空配置）。"""
    c = cfg_of(row)
    return (
        not c.get("--obs-noise-pos")
        and not c.get("--obs-noise-vel")
        and "--obs-use-kf" not in c
    )


__all__ = [
    "_physical_pair_summary",
    "_unique_index",
    "baseline_cell",
    "cell_stats",
    "mcnemar_exact",
    "paired_bootstrap",
    "paired_test",
    "paired_test_physical",
    "physical_of",
    "physical_stats",
    "sign_test",
    "wilson",
]
