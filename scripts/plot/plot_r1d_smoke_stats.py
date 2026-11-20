#!/usr/bin/env python
"""R1D 论文 smoke 单 case 统计绘图（T8c 后续集成，T8a manifest ``plot_argv`` 指向本脚本）。

用途
----
读取 T8b ``extract_r1d_smoke_stats.py`` 写出的单 case 统计 JSON
（``schema == "r1d_smoke_case_stats_v1"``），画一张 physical / legacy 双口径的
命中率概览图，并写出同名 JSON 摘要；供 ``run_r1d_paper_smoke.py`` 的
``plot_argv`` 在每次 smoke 任务后调用。

硬约束
------
- **不补数、不伪造**：``physical_active_rate`` / ``legacy_active_rate`` 为 ``None``
  时，图中对应位置只标注 ``不可估计: <reason>``，绝不画成 0%；``per_seed`` 中的
  error/timeout/no_verdict 同样只计数、不参与命中率。
- 只读取 ``--stats-root`` 下 ``<experiment>/<case>.json``，不读旧
  ``paper/planning/`` 或旧 ``experiment_data/exp*``。
- ``--output`` 未给时只打印摘要、不写盘；给了才创建目录并写 PNG + JSON。
- 输出文件名带 experiment/case，避免多个 case 共用同一输出目录时互相覆盖。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT = Path(__file__).resolve().parents[2]
SCHEMA = "r1d_smoke_case_stats_v1"
PLOT_SCHEMA = "r1d_smoke_case_plot_v1"
SMOKE_NOTE = "R1D smoke small-sample link-preview only; not a paper statistic"


@dataclass(frozen=True)
class CaseRates:
    """单口径命中率摘要（不可估计时 rate 为 None，denominator 仍记录真实分母）。"""

    semantics: str
    available: bool
    rate: float | None
    denominator: int
    active: int
    passive: int
    miss: int
    ci95: tuple[float, float] | None
    unavailable_reason: str | None


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器。"""
    parser = argparse.ArgumentParser(
        description="R1D 论文 smoke 单 case 统计绘图（physical / legacy 双口径，缺口显式）"
    )
    parser.add_argument("--stats-root", type=Path, required=True,
                        help="T8b 统计根目录（<root>/<experiment>/<case>.json）")
    parser.add_argument("--experiment", required=True, help="实验目录名，如 exp15_speed_v2")
    parser.add_argument("--case", required=True, help="case id，如 speed7")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录；未给则只打印摘要、不写盘")
    return parser


def load_case_stats(stats_root: Path, experiment: str, case: str) -> dict[str, Any]:
    """加载并校验 T8b 单 case 统计 JSON。

    Args:
        stats_root: T8b 统计根目录。
        experiment: 实验目录名。
        case: case id。

    Returns:
        解析后的统计字典。

    Raises:
        FileNotFoundError: 文件不存在（附候选路径与生成命令提示）。
        ValueError: schema 不符或缺少必需字段。
    """
    path = stats_root / experiment / f"{case}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"找不到单 case 统计 JSON: {path}；"
            f"请先运行 scripts/extract/extract_r1d_smoke_stats.py "
            f"--experiment {experiment} --case {case} --data-root <batch> "
            f"--output {stats_root}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"统计 JSON 顶层必须是对象: {path}")
    if payload.get("schema") != SCHEMA:
        raise ValueError(
            f"统计 JSON schema 不符: {path}（期望 {SCHEMA}，实际 {payload.get('schema')!r}）"
        )
    required = (
        "experiment_id", "case_id", "n_attempted", "n_valid",
        "n_error", "n_timeout", "n_no_verdict", "n_missing",
        "physical_active_rate", "physical_active_denominator",
        "legacy_active_rate", "legacy_active_denominator",
        "unavailable_reasons", "per_seed",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"统计 JSON 缺少必需字段 {missing}: {path}")
    return payload


def _optional_rate(value: Any, denominator: Any) -> float | None:
    """把 JSON 的 rate/denominator 规范成 0–100 的百分比；不可估计返回 None。"""
    if value is None or denominator in (None, 0):
        return None
    try:
        rate = float(value)
        denom = int(denominator)
    except (TypeError, ValueError):
        return None
    if denom <= 0 or not math.isfinite(rate):
        return None
    return rate


def build_case_rates(payload: dict[str, Any]) -> tuple[CaseRates, CaseRates]:
    """从统计 JSON 构造 physical / legacy 两个口径摘要。"""
    reasons = payload.get("unavailable_reasons") or {}
    if not isinstance(reasons, dict):
        reasons = {}

    physical = CaseRates(
        semantics="physical",
        available=_optional_rate(
            payload.get("physical_active_rate"), payload.get("physical_active_denominator")
        ) is not None,
        rate=_optional_rate(
            payload.get("physical_active_rate"), payload.get("physical_active_denominator")
        ),
        denominator=int(payload.get("physical_active_denominator") or 0),
        active=int(payload.get("n_physical_active") or 0),
        passive=int(payload.get("n_physical_passive") or 0),
        miss=int(payload.get("n_physical_miss") or 0),
        ci95=_parse_ci(payload.get("physical_ci95")),
        unavailable_reason=reasons.get("physical") or payload.get("unavailable_reason"),
    )
    legacy = CaseRates(
        semantics="legacy",
        available=_optional_rate(
            payload.get("legacy_active_rate"), payload.get("legacy_active_denominator")
        ) is not None,
        rate=_optional_rate(
            payload.get("legacy_active_rate"), payload.get("legacy_active_denominator")
        ),
        denominator=int(payload.get("legacy_active_denominator") or 0),
        active=int(payload.get("n_legacy_active") or 0),
        passive=int(payload.get("n_legacy_passive") or 0),
        miss=int(payload.get("n_legacy_miss") or 0),
        ci95=_parse_ci(payload.get("legacy_ci95")),
        unavailable_reason=reasons.get("legacy") or payload.get("unavailable_reason"),
    )
    return physical, legacy


def _parse_ci(value: Any) -> tuple[float, float] | None:
    """解析 CI95；缺省/非法返回 None（不编造区间）。"""
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        low, high = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(low) and math.isfinite(high)):
        return None
    return low, high


def _status_counts(payload: dict[str, Any]) -> dict[str, int]:
    """统计 per_seed 状态；error/timeout/no_verdict 绝不并入 miss。"""
    counts = {"ok": 0, "error": 0, "timeout": 0, "no_verdict": 0, "missing": 0}
    for row in payload.get("per_seed") or []:
        if not isinstance(row, dict):
            continue
        status = str(row.get("status") or "missing")
        counts[status if status in counts else "missing"] += 1
    return counts


def _safe_stem(experiment: str, case: str) -> str:
    """把 experiment/case 变成安全文件名片段。"""
    text = f"{experiment}__{case}"
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in text)


def render_case_figure(
    payload: dict[str, Any], physical: CaseRates, legacy: CaseRates, output_dir: Path
) -> tuple[Path, Path]:
    """渲染单 case 概览图并写 JSON 摘要。

    Args:
        payload: T8b 统计 JSON。
        physical: physical 口径摘要。
        legacy: legacy 口径摘要。
        output_dir: 输出目录（不存在则创建）。

    Returns:
        ``(png_path, json_path)``。
    """
    # 延迟导入 matplotlib：CLI --help / 无输出模式不拉重依赖。
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(str(payload.get("experiment_id") or ""),
                      str(payload.get("case_id") or ""))
    png_path = output_dir / f"{stem}.png"
    json_path = output_dir / f"{stem}.json"

    labels = ["physical", "legacy"]
    rates = [physical, legacy]
    colors = ["#2a6fdb", "#d9822b"]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x_positions = [0.0, 1.0]
    for x, item, color in zip(x_positions, rates, colors):
        if item.available and item.rate is not None:
            ax.bar(x, item.rate, width=0.5, color=color, alpha=0.85)
            ax.text(x, item.rate + 2.0, f"{item.rate:.1f}%", ha="center", va="bottom",
                    fontsize=10)
        else:
            reason = item.unavailable_reason or "unavailable"
            ax.text(x, 50.0, f"unavailable\\n{reason}", ha="center", va="center",
                    fontsize=10, color="#b03030")
        ax.text(x, -8.0, f"n={item.denominator}\nactive={item.active} passive={item.passive}",
                ha="center", va="top", fontsize=8.5, color="#444444")

    ax.set_xticks(x_positions, labels)
    ax.set_ylim(-18.0, 112.0)
    ax.set_ylabel("active hit rate [%]")
    ax.axhline(0.0, color="#999999", linewidth=0.8)
    status = _status_counts(payload)
    title = (
        f"{payload.get('experiment_id')}/{payload.get('case_id')}  "
        f"mode={payload.get('mode')}  attempted={payload.get('n_attempted')} "
        f"valid={payload.get('n_valid')}"
    )
    ax.set_title(title, fontsize=10)
    ax.text(0.0, 1.02, SMOKE_NOTE, transform=ax.transAxes, fontsize=8.5,
            color="#666666", ha="left", va="bottom")
    ax.text(0.0, -0.30,
            f"error={status['error']} timeout={status['timeout']} "
            f"no_verdict={status['no_verdict']} missing={status['missing']}; unavailable rates are not drawn as 0%",
            transform=ax.transAxes, fontsize=8.5, color="#666666", ha="left", va="top")
    fig.tight_layout()
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    summary = {
        "schema": PLOT_SCHEMA,
        "experiment_id": payload.get("experiment_id"),
        "case_id": payload.get("case_id"),
        "mode": payload.get("mode"),
        "robot_id": payload.get("robot_id"),
        "metric_schema": payload.get("metric_schema"),
        "n_attempted": payload.get("n_attempted"),
        "n_valid": payload.get("n_valid"),
        "status_counts": status,
        "physical": _rates_to_dict(physical),
        "legacy": _rates_to_dict(legacy),
        "note": SMOKE_NOTE,
        "figure": str(png_path),
    }
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    return png_path, json_path


def _rates_to_dict(item: CaseRates) -> dict[str, Any]:
    """把 CaseRates 转成 JSON 友好字典（None 保持 None，不改为 0）。"""
    return {
        "available": item.available,
        "active_rate": item.rate,
        "active_denominator": item.denominator,
        "active": item.active,
        "passive": item.passive,
        "miss": item.miss,
        "ci95": list(item.ci95) if item.ci95 is not None else None,
        "unavailable_reason": item.unavailable_reason,
    }


def print_summary(payload: dict[str, Any], physical: CaseRates, legacy: CaseRates) -> None:
    """打印一行摘要，便于 manifest 子进程日志审计。"""
    def _fmt(item: CaseRates) -> str:
        if item.available and item.rate is not None:
            return f"{item.rate:.1f}% (n={item.denominator})"
        return f"null (n={item.denominator}, {item.unavailable_reason or 'unavailable'})"

    print(
        f"[r1d-smoke-plot] {payload.get('experiment_id')}/{payload.get('case_id')} "
        f"physical={_fmt(physical)} legacy={_fmt(legacy)} "
        f"attempted={payload.get('n_attempted')} valid={payload.get('n_valid')}"
    )


def main(argv: list[str] | None = None) -> int:
    """入口：加载统计、渲染图、按需写盘。"""
    args = build_parser().parse_args(argv)
    payload = load_case_stats(args.stats_root, args.experiment, args.case)
    physical, legacy = build_case_rates(payload)
    print_summary(payload, physical, legacy)
    if args.output is None:
        print("[r1d-smoke-plot] 未给 --output：只打印，不写文件")
        return 0
    png_path, json_path = render_case_figure(payload, physical, legacy, args.output)
    print(f"[r1d-smoke-plot] wrote {png_path} + {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
