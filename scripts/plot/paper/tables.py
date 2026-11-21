"""LaTeX 表格：Table I/II（active-hit 率与配对差）与 table3 共享输入预览。

空值契约保持不变：缺档/不可估计一律渲染 ``--``，并在表下注明原因，
绝不显示 0；不可估计原因先经 :func:`_tex_escape` 转义再进正文。
"""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

from . import data as _data
from .style import fmt_pv, pyplot

MODE_LABELS = {"full": "full", "tube_only": "tube-only",
               "softmin_only": "candidate-set-only", "none": "point-target"}
MODE_ORDER = ["full", "tube_only", "softmin_only", "none"]


# table3 隔离预览的固定标注：数据来自 R1D smoke 链路时只能作链路预览，不得当论文统计
TABLE3_ANNOTATION = (
    "R1D smoke 小样本，仅链路预览，不作论文统计结论"
    "（缺格显式留空；legacy / physical 两种口径分表呈现）")
# PDF 面板用英文标注：matplotlib 默认 STIX 字体不含 CJK 字形，直接画中文会出豆腐块
TABLE3_ANNOTATION_EN = (
    "R1D smoke, small n; link-preview only; not a paper statistic "
    "(missing cells left blank, no imputation).")

# column_map 不可用时的 tier 兜底顺序（与 paired_stats.build_tables 保持一致）
_TABLE3_TIERS_FALLBACK = (
    ("full", "full"),
    ("tube_only", "tube-only"),
    ("softmin_only", "candidate-set-only"),
    ("none", "point-target"),
)


def _fmt_dash(value: object, *, digits: int = 1, signed: bool = True) -> str:
    """渲染可空数值：None / NaN / 非数值统一为 ``--``，绝不把缺失伪装成 0。"""
    if value is None:
        return "--"
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "--"
    if math.isnan(number) or math.isinf(number):
        return "--"
    return f"{number:+.{digits}f}" if signed else f"{number:.{digits}f}"


def _fmt_ci_dash(ci: object) -> str:
    """渲染可空 95% 区间 ``[lo,hi]``；不可估计（None/NaN/坏结构）返回 ``--``。"""
    if not isinstance(ci, (list, tuple)) or len(ci) != 2:
        return "--"
    lo, hi = ci[0], ci[1]
    if lo is None or hi is None:
        return "--"
    try:
        lo_f, hi_f = float(lo), float(hi)
    except (TypeError, ValueError):
        return "--"
    if math.isnan(lo_f) or math.isnan(hi_f):
        return "--"
    return f"[{lo_f:.1f},{hi_f:.1f}]"


def _fmt_p_dash(value: object) -> str:
    """渲染可空 p 值：None → ``--``；否则沿用 ``fmt_pv`` 既有格式。"""
    if value is None:
        return "--"
    try:
        return fmt_pv(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "--"


def _tex_escape(text: str) -> str:
    """转义进入 LaTeX 正文的动态文本（M10）。

    不可估计原因（``all_miss`` / ``no_common_seed`` 等）含裸下划线，直接拼进脚注会
    让 xelatex 报 ``Missing $ inserted``。本函数只用于**动态文本**（原因、mode 名等），
    表内固定文本与 JSON 原始值一律不改。

    处理顺序：先把反斜杠换成占位符（避免后续替换引入的反斜杠被二次转义），再处理
    ``& % $ # _ { }``，最后用文本命令表达 ``~`` 与 ``^``；占位符最后还原成
    ``\\textbackslash{}``。

    Args:
        text: 任意动态文本（可含中文）。

    Returns:
        可直接嵌入 LaTeX 正文的转义字符串。
    """
    placeholder = "\x00"
    escaped = str(text).replace("\\", placeholder)
    for char, replacement in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        escaped = escaped.replace(char, replacement)
    escaped = escaped.replace("~", r"\textasciitilde{}")
    escaped = escaped.replace("^", r"\textasciicircum{}")
    return escaped.replace(placeholder, r"\textbackslash{}")


def _table_section(stats: dict, hit_semantics: str) -> dict:
    """按命中口径选择表格数据源：legacy 用顶层，physical 用 ``stats['physical']``。

    Args:
        stats: paired_stats 生成的统计 JSON。
        hit_semantics: ``legacy`` 或 ``physical``。

    Returns:
        供表格消费的统计字典（physical 时为物理口径段落）。

    Raises:
        FileNotFoundError: 请求 physical 但统计 JSON 没有 physical 段落。
        ValueError: 未知命中口径。
    """
    if hit_semantics == "legacy":
        return stats
    if hit_semantics != "physical":
        raise ValueError(f"未知命中口径：{hit_semantics!r}（可选 legacy / physical）")
    physical = stats.get("physical")
    if not isinstance(physical, dict) or "E8_limit_mechanism" not in physical:
        raise FileNotFoundError(
            "统计 JSON 缺少 physical 段落：请先用 paired_stats --mode <mode> 生成分模式统计，"
            "再以 --hit-semantics physical 出表（不回退到混合口径）"
        )
    return physical


def tables(stats: dict, out_dir: Path | None = None, *, hit_semantics: str = "legacy") -> None:
    """生成 Table I（各档 active-hit 率与 n）与 Table II（配对差、McNemar p、DiD）。

    空值契约：该档没有有效样本（``n_valid == 0``）或推断不可估计（``ci95`` /
    ``p_sign`` 为 None）时，TeX 一律渲染 ``--``，并在表下注明原因，绝不显示 0。

    Args:
        stats: paired_stats 生成的统计 JSON。
        out_dir: 输出目录，默认 ``OUT/table_data``。
        hit_semantics: ``legacy``（默认，保持旧 RM65 输出）或 ``physical``
            （读取 ``stats["physical"]``，缺失时抛 ``FileNotFoundError`` 由上层跳过）。

    Raises:
        FileNotFoundError: physical 口径缺少 ``physical`` 段落，或表格所需小节缺失。
    """
    source = _table_section(stats, hit_semantics)
    try:
        e3 = source["E3_nominal"]
        e4 = source["E4_grid"]
        e7 = source["E7_corners"]
        e8 = source["E8_limit_mechanism"]
    except KeyError as exc:
        raise FileNotFoundError(f"统计 JSON 缺少表格所需小节：{exc}") from exc
    out_dir = out_dir if out_dir is not None else _data.OUT / "table_data"
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_text = str(stats.get("mode") or "未标注")

    def _cells(kind: str) -> dict:
        if kind == "E3":
            return e3.get("cells") or {}
        if kind == "E4":
            return e4.get("merged_20cell") or {}
        return e7.get("cells") or {}

    def cell(kind: str, key: str) -> float | None:
        """取一格的命中率；缺档或 n_valid==0 返回 None（表格渲染为 --）。"""
        entry = _cells(kind).get(key)
        if not isinstance(entry, dict) or not entry.get("n_valid"):
            return None
        rate = entry.get("rate")
        return None if rate is None else float(rate)

    def n_of(kind: str, key: str) -> int | None:
        """取一格的有效样本数；缺档返回 None。"""
        entry = _cells(kind).get(key)
        if not isinstance(entry, dict) or not entry.get("n_valid"):
            return None
        return int(entry["n_valid"])

    rows12 = [
        ("Nominal 7 m/s", "E3", *[f"7|{m}" for m in MODE_ORDER]),
        ("Nominal 9 m/s", "E3", *[f"9|{m}" for m in MODE_ORDER]),
        ("Nominal 12 m/s", "E3", *[f"12|{m}" for m in MODE_ORDER]),
    ]
    rows_grid = [("Perturbed grid (20 cells, 9 m/s)", "E4",
                  *[m for m in MODE_ORDER])]
    rows_corner = [("Corner 12 m/s, $t{=}0$, $s{=}0.2$ m", "E7",
                    *[f"12|s0.2|t0|{m}" for m in MODE_ORDER])]
    rows_all = rows12 + rows_grid + rows_corner

    # chat21 的行内注释压缩后已随 caption 一起删去，避免行内注释过大
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Active-hit rate (\%) of the four configurations by "
             r"condition; $n$ counts valid runs per configuration and is given "
             r"as a range over the four (the perturbation-grid row pools its "
             r"$20$ cells). Exact intervals are in the artifact.}",
             r"\label{tab:ablation}",
             r"\begin{tabular}{lccccrc}", r"\toprule",
             r"Condition & full & tube-only & candidate-set-only & point-target & $n$ & Design \\",
             r"\midrule"]
    kind_label = {"E3": "nominal four-tier", "E4": "perturbation grid",
                  "E7": "high-power corners"}
    for label, kind, *keys in rows_all:
        vals = " & ".join(_fmt_dash(cell(kind, k), signed=False) for k in keys)
        ns = [n for n in (n_of(kind, k) for k in keys) if n is not None]
        n_str = "--" if not ns else (f"{min(ns)}" if min(ns) == max(ns) else f"{min(ns)}--{max(ns)}")
        lines.append(f"{label} & {vals} & {n_str} & {kind_label[kind]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if hit_semantics == "physical":
        lines.append(
            r"\par\footnotesize 命中口径：physical（v12 首个物理拍面接触，face speed "
            r"$>0.3$\,m/s），mode=" + _tex_escape(mode_text)
            + r"；\texttt{--} 表示该档无有效样本、未估计。"
        )
        lines.append(r"\end{table*}")
    else:
        lines.append(r"\end{table*}")
    lines.append("")
    (out_dir / "table1_comparison.tex").write_text("\n".join(lines), encoding="utf-8")

    # Table II：TCP 限速的配对差 + p + 双差 DiD；缺失/不可估计一律 -- 并注明原因
    lines = [r"\begin{table}[t]", r"\centering",
             r"\caption{Paired gain ({\upshape pp}) over point-target at "
             r"$7$\,{\upshape m/s} ($n{=}394$ seeds); $p$: unadjusted "
             r"McNemar. Lower block: between-cap shifts.}",
             r"\label{tab:limits}",
             r"\setlength{\tabcolsep}{3.5pt}",
             # 以下两行是写给 TeX 的注释（chat22 恢复版）：保持原行，
             # 不要因 tables() 改动顺手删除或改写
             r"% chat22：将 1.0 改 0.92，约省 8 pt，用于 Fig.5(c) 高度占用的页底空间补偿。",
             r"% 验证：5 页以内 与V-E 两不超（超版）时约 ~35 pt 空白",
             r"\renewcommand{\arraystretch}{0.92}",
             r"\begin{tabular}{lcccr}", r"\toprule",
             r"Config. & \multicolumn{2}{c}{Gain over point-target (pp)} & "
             r"\multicolumn{2}{c}{$p$} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r" & TCP 1.8 & TCP 1.0 & TCP 1.8 & TCP 1.0 \\", r"\midrule"]
    paired = e8.get("paired") or {}
    for mode in ("tube_only", "softmin_only", "full"):
        p18 = paired.get(f"TCP1.8|{mode}_vs_none") or {}
        p10 = paired.get(f"TCP1.0|{mode}_vs_none") or {}
        lines.append(
            f"{MODE_LABELS[mode]} & ${_fmt_dash(p18.get('diff_pp'))}$ & "
            f"${_fmt_dash(p10.get('diff_pp'))}$ & ${_fmt_p_dash(p18.get('p_mcnemar'))}$ & "
            f"${_fmt_p_dash(p10.get('p_mcnemar'))}$ \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", "",
              r"\begin{tabular}{lccc}", r"\toprule",
              r"DiD (TCP 1.8 $\to$ TCP 1.0) & DiD (pp) & $95\%$ CI & sign-test $p$ \\",
              r"\midrule"]
    did = e8.get("did") or {}
    notes: list[str] = []
    for mode in ("tube_only", "softmin_only", "full"):
        for label in ("TCP1.8", "TCP1.0"):
            pair = paired.get(f"{label}|{mode}_vs_none") or {}
            if pair.get("unavailable_reason"):
                notes.append(
                    f"{MODE_LABELS[mode]}@{label}: {pair['unavailable_reason']}"
                    f"（n={pair.get('n_paired', 0)}，观察差 {_fmt_dash(pair.get('diff_pp'))} pp）"
                )
    for mode in ("tube_only", "softmin_only", "full"):
        d = did.get(mode) or {}
        reason = d.get("unavailable_reason")
        ci_text = _fmt_ci_dash(d.get("ci95"))
        lines.append(
            f"{MODE_LABELS[mode]} & ${_fmt_dash(d.get('mean_did_pp'))}$ & "
            f"${ci_text}$ & ${_fmt_p_dash(d.get('p_sign'))}$ \\\\")
        if reason:
            notes.append(
                f"{MODE_LABELS[mode]}: {reason}（n={d.get('n_pairs', 0)}）"
            )
    lines += [r"\bottomrule", r"\end{tabular}"]
    if notes:
        # 不可估计原因写进 TeX 注释与表下脚注，保证 -- 可追溯（M10：正文必须转义）
        for note in notes:
            # 注释行保留原始 reason，但折叠换行，避免下一行脱离 % 注释泄漏进正文
            lines.append("% unavailable: " + " ".join(str(note).split()))
        lines.append(
            r"\par\footnotesize \texttt{--} 表示不可估计（观察差值仍保留样本数，"
            r"不做零假设推断）：" + _tex_escape("；".join(notes)) + "。"
        )
    if hit_semantics == "physical":
        lines.append(
            r"\par\footnotesize 命中口径：physical，mode=" + _tex_escape(mode_text) + "。"
        )
    lines += [r"\end{table}", ""]
    (out_dir / "table2_ablation.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"已保存 {out_dir}/table1_comparison.tex + table2_ablation.tex")


def _fmt_pct(value: object) -> str:
    """百分比数值格式化：None / NaN / 非数值 → `--`（缺格不画 0）。"""
    if value is None:
        return "--"
    try:
        v = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(v):
        return "--"
    return f"{v:.1f}"


def _fmt_p_raw(value: object) -> str:
    """配对 p 值格式化：None / NaN / 非数值 → `--`。"""
    if value is None:
        return "--"
    try:
        p = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "--"
    if not np.isfinite(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def build_table3_preview(stats: dict) -> dict:
    """从 T8b 统计构建 table3 隔离预览数据（不触碰 E1–E8）。

    设计要点：
    - 只依赖 `tables.table3_shared_input`（rows[] 与 column_map），因此 R1D
      smoke 小样本即使没有 exp22/exp24 数据也能出「链路预览」表；
    - legacy / physical 两种口径分开返回，缺格按 `present=False` 或
      None/NaN 显示 `--`，绝不补 0；
    - tier 的列顺序与显示名从 column_map 的
      `tables.table3_shared_input.rows[].cells.<tier>.legacy_rate` 反推；
      column_map 缺失时退回 paired_stats 的固定 tier 顺序。

    Args:
        stats: T8b 统计 JSON（含 tables.table3_shared_input）。

    Returns:
        {"annotation", "description", "tiers", "rows"}；每行含
        `legacy` / `physical` 两套格式化值、`present` 标记与 `missing` tier 列表。

    Raises:
        ValueError: 缺 `tables.table3_shared_input` 或 rows 为空（清晰报错，
            而非裸 KeyError）。
    """
    tables = stats.get("tables") if isinstance(stats, dict) else None
    table3 = (tables or {}).get("table3_shared_input")
    if not isinstance(table3, dict):
        raise ValueError(
            "统计 JSON 缺少 tables.table3_shared_input：table3 隔离预览需要 T8b 统计"
            "（先运行 scripts/extract/paired_stats.py 生成 paper_stats_active.json）")
    rows_in = table3.get("rows") or []
    if not rows_in:
        raise ValueError("tables.table3_shared_input.rows 为空：没有可预览的 (t,s) 格")

    column_map = (tables or {}).get("column_map") or {}
    columns = (column_map.get("table3_shared_input.tex") or {}).get("columns") or {}
    tiers: list[dict[str, str]] = []
    for label, json_path in columns.items():
        m = re.search(r"\.cells\.([A-Za-z0-9_]+)\.legacy_rate$", str(json_path))
        if m:
            tiers.append({"key": m.group(1), "label": str(label)})
    if not tiers:
        tiers = [{"key": k, "label": lab} for k, lab in _TABLE3_TIERS_FALLBACK]

    rows = []
    for src in rows_in:
        cells = src.get("cells") or {}
        legacy: dict[str, str] = {}
        physical: dict[str, str] = {}
        present: dict[str, bool] = {}
        for tier in tiers:
            key = tier["key"]
            cell = cells.get(key) or {}
            is_present = bool(cell.get("present"))
            present[key] = is_present
            legacy[key] = _fmt_pct(cell.get("legacy_rate")) if is_present else "--"
            physical[key] = _fmt_pct(cell.get("physical_rate")) if is_present else "--"
        rows.append({
            "cell": str(src.get("cell", "")),
            "t": src.get("t"),
            "s": src.get("s"),
            "legacy": legacy,
            "physical": physical,
            "present": present,
            "p_raw_legacy": _fmt_p_raw(src.get("p_raw_legacy")),
            "p_raw_physical": _fmt_p_raw(src.get("p_raw_physical")),
            "n_paired_legacy": src.get("n_paired_legacy"),
            "n_paired_physical": src.get("n_paired_physical"),
            "missing": list(src.get("missing") or []),
        })
    return {"annotation": TABLE3_ANNOTATION,
            "description": str(table3.get("description", "")),
            "tiers": tiers, "rows": rows}


def _table3_tex(preview: dict) -> str:
    """把隔离预览渲染成 LaTeX（legacy / physical 上下两张表，缺格 `--`）。"""
    tiers = preview["tiers"]
    header = " & ".join(t["label"] for t in tiers)
    lines = [
        "% 自动生成（T8c table3 数据驱动隔离版；勿手改）",
        f"% {preview['annotation']}",
        f"% {preview['description']}",
        r"\begin{table*}[t]", r"\centering",
        r"\caption{Shared-input protocol preview (R1D smoke 小样本，仅链路预览，"
        r"不作论文统计结论)。缺格以 \texttt{--} 显式留空（不补 0）；"
        r"上表 legacy 口径，下表 physical 口径。}",
        r"\label{tab:shared_input_preview}",
        r"{\small Legacy hit rate (\%)} \\[2pt]",
        r"\begin{tabular}{l" + "c" * len(tiers) + "r}", r"\toprule",
        f"$ (t,s) $ cell & {header} & $p_{{raw}}$ \\\\", r"\midrule",
    ]
    for row in preview["rows"]:
        vals = " & ".join(row["legacy"][t["key"]] for t in tiers)
        lines.append(f"{row['cell']} & {vals} & {row['p_raw_legacy']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\vspace{6pt}",
              r"{\small Physical hit rate (\%)} \\[2pt]",
              r"\begin{tabular}{l" + "c" * len(tiers) + "r}", r"\toprule",
              f"$ (t,s) $ cell & {header} & $p_{{raw}}$ \\\\", r"\midrule"]
    for row in preview["rows"]:
        vals = " & ".join(row["physical"][t["key"]] for t in tiers)
        lines.append(f"{row['cell']} & {vals} & {row['p_raw_physical']} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    return "\n".join(lines)


def _table3_pdf(preview: dict, path: Path) -> None:
    """把隔离预览渲染成 PDF（matplotlib 两面板表 + 固定标注）。"""
    plt = pyplot()
    tiers = preview["tiers"]
    n_rows = len(preview["rows"])
    fig_h = max(2.8, 1.6 + 0.30 * (2 * n_rows + 4))
    fig = plt.figure(figsize=(7.16, fig_h))
    fig.suptitle("Shared-input protocol preview — R1D smoke (link-only, small n)",
                 fontsize=9)
    for panel, metric in enumerate(("legacy", "physical")):
        ax = fig.add_axes([0.02, 0.50 - panel * 0.47, 0.96, 0.40])
        ax.axis("off")
        ax.set_title(f"({'a' if panel == 0 else 'b'}) {metric.capitalize()} "
                     "hit rate (%)", fontsize=8, loc="left")
        col_labels = ["(t,s)"] + [t["label"] for t in tiers] + ["p_raw"]
        cell_text = [
            [row["cell"]]
            + [row[metric][t["key"]] for t in tiers]
            + [row[f"p_raw_{metric}"]]
            for row in preview["rows"]
        ]
        tab = ax.table(cellText=cell_text, colLabels=col_labels,
                       loc="upper center", cellLoc="center")
        tab.auto_set_font_size(False)
        tab.set_fontsize(7)
        tab.scale(1.0, 1.22)
    fig.text(0.02, 0.012, TABLE3_ANNOTATION_EN, fontsize=7, color="0.35")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def table3_shared_input(stats: dict, out_dir: Path | None = None) -> dict[str, Path]:
    """生成 table3 隔离预览三件套（tex + csv + pdf）。

    Args:
        stats: T8b 统计 JSON（含 tables.table3_shared_input）。
        out_dir: 输出目录；None → `OUT/table_data`（与 Table I/II 同目录约定）。

    Returns:
        {"tex": Path, "csv": Path, "pdf": Path}（绝对路径）。

    Raises:
        ValueError: 统计 JSON 缺 table3 键（由 build_table3_preview 抛出）。
    """
    preview = build_table3_preview(stats)
    base = Path(out_dir) if out_dir is not None else _data.OUT / "table_data"
    base.mkdir(parents=True, exist_ok=True)
    tex_path = (base / "table3_shared_input_preview.tex").resolve()
    csv_path = (base / "table3_shared_input_preview.csv").resolve()
    pdf_path = (base / "table3_shared_input_preview.pdf").resolve()

    tex_path.write_text(_table3_tex(preview), encoding="utf-8")

    tiers = preview["tiers"]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["cell", "t", "s"]
                        + [f"{t['key']}_{metric}" for t in tiers
                           for metric in ("legacy", "physical")]
                        + ["p_raw_legacy", "p_raw_physical", "missing_tiers"])
        for row in preview["rows"]:
            values: list[object] = [row["cell"], row["t"], row["s"]]
            for tier in tiers:
                for metric in ("legacy", "physical"):
                    values.append("" if row[metric][tier["key"]] == "--"
                                  else row[metric][tier["key"]])
            values += ["" if row["p_raw_legacy"] == "--" else row["p_raw_legacy"],
                       "" if row["p_raw_physical"] == "--" else row["p_raw_physical"],
                       ";".join(row["missing"])]
            writer.writerow(values)

    _table3_pdf(preview, pdf_path)
    print(f"已保存 {tex_path} + {csv_path} + {pdf_path}")
    return {"tex": tex_path, "csv": csv_path, "pdf": pdf_path}
