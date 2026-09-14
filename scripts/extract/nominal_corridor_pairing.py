#!/usr/bin/env python3
"""nominal 条件 corridor-only 配对统计（chat12 审稿必改项）。

审稿意见：结论声称 corridor 的独立价值 "only" 出现在时间裕量紧张/空间误差
条件下，但 Table II 标称 12 m/s 显示 corridor-only 比 point-target 高
+21.4 pp（74.5 vs 53.1）——结论与自家表格矛盾。本脚本把 nominal 条件的
corridor-only 配对比较正式纳入 RQ2：same-seed 配对差值 + bootstrap 95% CI
+ McNemar 精确检验 + 不一致配对计数。

数据来源（与 Table II 同源）：
    exp17a_noise    full / none   标称格（σ=0, kf=off）@ 7/9/12 m/s
    exp17d_mechanism tube_only / softmin_only         @ 7/9/12 m/s

用法:
    python scripts/extract/nominal_corridor_pairing.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.extract.paired_stats import (
    DATA,
    HitPolicy,
    baseline_cell,
    cfg_of,
    load,
    paired_test,
)

OUT = DATA / "nominal_corridor_pairing.json"
SPEEDS = [7, 9, 12]
# (tier_a, tier_b, 说明) —— 论文 RQ2 需要走廊独立价值 + 窗口独立价值对照
PAIRS = [
    ("tube_only", "none", "corridor-only − point-target"),
    ("softmin_only", "none", "candidate-set-only − point-target"),
    ("full", "softmin_only", "full − candidate-set-only"),
]


def main() -> None:
    """重算 nominal 条件的配对统计并输出 JSON。"""
    hitfn = HitPolicy()
    rows_a = load("exp17a_noise")
    rows_d = load("exp17d_mechanism")

    result: dict = {}
    for speed in SPEEDS:
        result[str(speed)] = {}
        for tier_a, tier_b, desc in PAIRS:
            # tier 可能来自两个数据源：none/full 在 exp17a，tube_only/softmin_only 在 exp17d
            def _speed_filter(row: dict) -> bool:
                cfg = cfg_of(row)
                return (int(cfg.get("--ball-speed", 0)) == speed
                        and baseline_cell(row))

            pools: dict[str, list] = {"exp17a_noise": rows_a, "exp17d_mechanism": rows_d}
            combined: list[dict] = []
            # exp17a 只含 full/none，exp17d 只含 tube_only/softmin_only —— 各自过滤速度
            for name, rows in pools.items():
                for r in rows:
                    if cfg_of(r).get("--ball-speed") == speed and baseline_cell(r):
                        combined.append(r)
            stats = paired_test(combined, hitfn, tier_a, tier_b)
            if stats.get("n_paired", 0) == 0:
                print(f"[{speed} m/s] {desc}: 无配对数据！")
                continue
            result[str(speed)][f"{tier_a}_vs_{tier_b}"] = stats
            print(
                f"[{speed} m/s] {desc}: "
                f"{stats['rate_a']:.1f} vs {stats['rate_b']:.1f}% "
                f"(diff {stats['diff_pp']:+.1f} pp, "
                f"CI [{stats['ci95'][0]:+.1f}, {stats['ci95'][1]:+.1f}], "
                f"McNemar p={stats['p_mcnemar']:.3g}, "
                f"n={stats['n_paired']}, "
                f"discordant {stats['discordant']})"
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n已写入 {OUT}")


if __name__ == "__main__":
    main()
