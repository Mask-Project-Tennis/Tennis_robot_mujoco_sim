"""论文统计口径统一提取：active-hit 主指标 + same-seed 配对检验（RM65 / R1D 同源）。

背景（2026-09-12 审稿意见核实后确立的口径）：
1. 主指标改为 **active-hit rate**：`hit_type == "active"`（击球瞬间球拍面速度 > 0.3 m/s）；
   被动接触（0 < v_r <= 0.3 m/s）单独计数、不进入主指标；
2. 全部实验为 same-seed 配对设计（同 seed 同发球同扰动交给各档），
   因此显著性检验改用 **配对 McNemar 精确检验** + **配对 bootstrap 95% CI**，
   取代此前提取脚本中的两独立样本比例 z 检验（忽略配对，口径偏保守）；
3. 排除行（`error` 非空，均为 `no_result`）逐档计数并报告，验证排除与方法无关；
4. 限速层 × 机制档的交互效应用 **配对 DiD**（每 seed 的增益差）报告。

命中语义（T8b 起显式区分，二者都报告、互不替代）：
- **legacy**：旧口径 `hit_type == "active"`（RM65 历史 CSV，亦为 R1D batch CSV 的
  `hit_type`；`passive` 不计入 active；可 `--include-passive` 复现旧合并口径）；
- **physical**：R1D v2 产物口径 `v12_episode.json` 的 `hit.physical_hit_type ==
  "active"`（首次物理拍面接触且拍速 > 0.3 m/s；`passive`/`miss` 不计命中），
  由 T8a 汇总进 `<run_root>/runs.csv` 的 `physical_hit` / `physical_hit_type` 列；
  `physical_hit` 布尔仅在类型缺失/未知时兜底。旧 RM65 根没有该列 → physical
  一律 `rate=null` + `unavailable_reason`，**不伪造 0.0**，也不拿 legacy 冒充
  physical；有判定但全 miss 时 `rate=0.0` 且标 `all_miss`（0/N 是测得的 0%）。

数据根与输出（T8b / F03 / F04）：
- `--data-root` 指定实验数据根（默认 `experiment_data/`，旧行为不变）；
- `--catalog` 指定 `run_catalog` 产出的 `catalog.json`：据此解析真实 artifact 位置
  （output 与 data-root 分离、跨目录时不靠猜数据根）；显式坏 catalog 直接退出码 2，
  **不回退**旧的数据根推断；无 catalog 的旧 RM65 根保持旧行为；
- `--mode torque|position|both` 指定执行模式：同一次统计只允许一种模式进入同一个
  样本池（torque/position 合成会掩盖两模式各自的命中率）；无 `--mode` 时仅在数据
  恰好只有一种模式时自动推断，混合模式直接非零退出并要求显式选择；
  `--mode both --output <dir>` 分别写 `<dir>/torque/` 与 `<dir>/position/` 下的
  `06-stats-active.json` / `paper_stats_active.json`，并写 `<dir>/modes.json`
  （schema `paired_stats_multi_mode_v1`，只作索引与验证状态，不重复统计内容）；
- `--output` 指定输出目录，默认 `<data_root>/../stats/`（不再硬写 `paper/planning/`）；
  单模式时写出 `06-stats-active.json` 与 `paper_stats_active.json` 两份内容相同的产物；
- `--dry-run` 只打印不写盘（写盘前必先打印目标路径）；
- `E1`–`E8` 顶层键与字段保持旧定义不变（旧绘图键兼容），新增内容一律追加在
  `hit_semantics` / `physical` / `E9_shared_input` / `tables` / `data_root` /
  `unavailable` 等新顶层键下，避免旧消费者把新键误当数据键。

用法:
    python scripts/extract/paired_stats.py [--include-passive]
    python scripts/extract/paired_stats.py --data-root <new_root>/batch \\
        --output results/.../stats
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .report import (
    _multi_mode_index,
    _print_report,
    _reject_physical_conflicts,
    _write_stats,
    compute_summary,
    default_output_dir,
)
from .types import (
    DEFAULT_DATA_ROOT,
    MODE_CHOICES,
    MULTI_MODE_FILENAME,
    VALID_MODES,
    StatsError,
)

# 新包深度：cli.py -> paired_analysis -> extract -> scripts -> 仓库根。
PROJECT = Path(__file__).resolve().parents[3]



def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--catalog / --mode / --data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-passive", action="store_true",
                        help="把被动接触也计入命中（复现旧口径，用于对照）")
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="catalog.json 路径；显式坏 catalog 退出码 2（不回退旧逻辑）")
    parser.add_argument("--mode", choices=MODE_CHOICES, default=None,
                        help="执行模式 torque/position/both；混合数据必须显式选择")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（给出时写 <output>/06-stats-active.json 与 "
                             "paper_stats_active.json；--mode both 时写 <output>/<mode>/）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算并打印，不写任何文件")
    return parser




def main(argv: list[str] | None = None) -> int:
    """跑完全部统计并打印报告；仅在显式 --output 时写出 JSON（--dry-run 不写盘）。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    out_dir: Path = args.output if args.output is not None else default_output_dir(data_root)
    try:
        summaries: dict[str, dict[str, Any]] = {}
        if args.mode == "both":
            for mode in VALID_MODES:
                summaries[mode] = compute_summary(
                    args.data_root,
                    include_passive=args.include_passive,
                    catalog_path=args.catalog,
                    mode=mode,
                )
        else:
            single = compute_summary(
                args.data_root,
                include_passive=args.include_passive,
                catalog_path=args.catalog,
                mode=args.mode,
            )
            summaries[single.get("mode") or "unspecified"] = single
    except StatsError as exc:
        print(f"[paired_stats] 错误：{exc}", file=sys.stderr)
        return 2

    for mode, summary in summaries.items():
        if args.mode == "both":
            print(f"===== mode={mode} =====")
        _print_report(summary)
    if args.dry_run:
        print(f"\n[dry-run] 不写文件（目标目录 {out_dir}）")
        return _reject_physical_conflicts(summaries)
    if args.output is None:
        # 未显式给 --output：只打印，绝不写默认根/paper（旧 paper/planning 不被覆盖）
        print(f"\n[paired_stats] 未给 --output：只打印，不写文件（默认目标 {out_dir}）")
        return _reject_physical_conflicts(summaries)
    if args.mode == "both":
        print(f"\n写出目标目录: {out_dir}")
        for mode, summary in summaries.items():
            for target in _write_stats(out_dir / mode, summary):
                print(f"  已写出 {target}")
        index_path = out_dir / MULTI_MODE_FILENAME
        index_path.write_text(
            json.dumps(_multi_mode_index(out_dir, summaries), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  已写出 {index_path}")
        return _reject_physical_conflicts(summaries)
    summary = next(iter(summaries.values()))
    print(f"\n写出目标目录: {out_dir}")
    for target in _write_stats(out_dir, summary):
        print(f"  已写出 {target}")
    return _reject_physical_conflicts(summaries)


__all__ = [
    "build_parser",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - 供 python -m scripts.extract.paired_analysis.cli 使用
    raise SystemExit(main())
