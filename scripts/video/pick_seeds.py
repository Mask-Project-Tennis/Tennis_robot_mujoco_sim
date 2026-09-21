#!/usr/bin/env python
"""随稿视频演示种子挑选与核验工具。

从 exp17a_noise / exp17d_mechanism 的 per-seed CSV 中挑选两类演示种子：
1. 分歧种子（12 m/s）：point-target(none)=miss 且 full=active 的同 seed 配对，
   理想情况下 softmin_only 也为 active（对应旁白"candidate window alone"）；
2. 干净命中种子（7 m/s）：full=active 且命中位置误差最小的 seed（开场/收尾镜头）。

核验模式（--verify）：按 config_json 精确重建 CLI 重跑 v12，比对 hit_type 是否
与 CSV 一致；并对首选分歧种子跑满四变体 --dump-trajectory，校验 p0/v0 逐位一致
（同 seed 四变体配对的前提）。

用法（mujoco_tennis 环境，仓库根目录）：
    python scripts/video/pick_seeds.py                 # 仅筛选，写 assets/seeds.json
    python scripts/video/pick_seeds.py --verify        # 筛选 + v12 重跑核验
    python scripts/video/pick_seeds.py --top 3         # 备选分歧种子数量（默认 3）
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
VIDEO_ASSETS = REPO / "paper" / "video" / "assets"
ENV_PY = Path("/home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python")

# nominal 四档的 (实验目录, config_id) 映射：full/none 来自 exp17a，其余来自 exp17d
NOMINAL_CELLS: dict[tuple[str, int], tuple[str, str]] = {}
for _speed in (7, 9, 12):
    NOMINAL_CELLS[("full", _speed)] = (
        "exp17a_noise", f"ablationfull__ballspeed{_speed}__noplot__servebox")
    NOMINAL_CELLS[("none", _speed)] = (
        "exp17a_noise", f"ablationnone__ballspeed{_speed}__noplot__servebox")
    for _mode in ("softmin_only", "tube_only"):
        NOMINAL_CELLS[(_mode, _speed)] = (
            "exp17d_mechanism", f"ablation{_mode}__ballspeed{_speed}__noplot__servebox")

# 已知系统性 no_result 的种子，演示素材必须避开
BAD_SEEDS = {20, 21, 27}

RESULT_RE = re.compile(r"__RESULT__:.*hit_type=(\w+)")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("pick_seeds")


def load_cell(mode: str, speed: int) -> pd.DataFrame:
    """读取某个 (ablation, 球速) 格的全部 per-seed 行。

    Args:
        mode: ablation 变体名（full/none/softmin_only/tube_only）。
        speed: 来球速度 m/s。

    Returns:
        以 seed 为索引的 DataFrame，仅保留有效结果行。
    """
    exp_dir, config_id = NOMINAL_CELLS[(mode, speed)]
    df = pd.read_csv(REPO / "experiment_data" / exp_dir / "results.csv")
    cell = df[df["config_id"] == config_id].copy()
    if cell.empty:
        raise ValueError(f"未找到 config_id={config_id}")
    return cell.set_index("seed")


def pick_discordant(top: int = 3) -> list[dict]:
    """挑选 12 m/s 下 none=miss 且 full=active 的分歧种子，按演示价值排序。

    排序键：softmin_only 也命中（加分） > none 未中距离更远 > full 命中误差更小。

    Args:
        top: 保留前 N 个候选。

    Returns:
        候选列表，每项含 seed、各变体 hit_type、none.min_dist、full.hit_pos_error。
    """
    cells = {m: load_cell(m, 12) for m in ("none", "full", "softmin_only", "tube_only")}
    common = set.intersection(*(set(c.index) for c in cells.values())) - BAD_SEEDS
    rows = []
    for s in sorted(common):
        none_row = cells["none"].loc[s]
        full_row = cells["full"].loc[s]
        if none_row["hit_type"] != "miss" or full_row["hit_type"] != "active":
            continue
        soft = cells["softmin_only"].loc[s]
        tube = cells["tube_only"].loc[s]
        rows.append({
            "seed": int(s),
            "none_hit_type": str(none_row["hit_type"]),
            "full_hit_type": str(full_row["hit_type"]),
            "softmin_only_hit_type": str(soft["hit_type"]),
            "tube_only_hit_type": str(tube["hit_type"]),
            "none_min_dist": float(none_row["min_dist"]),
            "full_hit_pos_error": float(full_row["hit_pos_error"]),
            "config_json": str(cells["none"].loc[s]["config_json"]),
        })
    # softmin 命中优先；miss 距离大更直观；hit 误差小更干净
    rows.sort(key=lambda r: (
        0 if r["softmin_only_hit_type"] == "active" else 1,
        -r["none_min_dist"],
        r["full_hit_pos_error"],
    ))
    logger.info("12 m/s 分歧种子（none=miss & full=active）共 %d 个，取前 %d", len(rows), top)
    return rows[:top]


def pick_clean_hit() -> dict:
    """挑选 7 m/s 下 full=active 且命中位置误差最小的干净命中种子。

    Returns:
        含 seed、hit_pos_error、config_json 的字典。
    """
    cell = load_cell("full", 7)
    cand = cell[(cell["hit_type"] == "active") & (~cell.index.isin(BAD_SEEDS))]
    best = cand.sort_values("hit_pos_error").index[0]
    row = cand.loc[best]
    logger.info("7 m/s 干净命中种子: seed=%s hit_pos_error=%.4f m", best, row["hit_pos_error"])
    return {
        "seed": int(best),
        "full_hit_type": "active",
        "full_hit_pos_error": float(row["hit_pos_error"]),
        "config_json": str(row["config_json"]),
    }


def build_argv(config_json: str, seed: int, extra: list[str] | None = None) -> list[str]:
    """把 CSV 里的 config_json 还原为 v12 命令行参数列表。

    Args:
        config_json: CSV config_json 列的原始字符串。
        seed: 演示种子号。
        extra: 追加参数（如 --dump-trajectory 路径）。

    Returns:
        形如 ["--ablation", "none", ...] 的参数列表。
    """
    argv: list[str] = []
    for key, val in json.loads(config_json).items():
        argv.append(key)
        if val is not None:  # store_true 标志值为 null
            argv.append(str(val))
    argv.append(f"--seed={seed}")
    if extra:
        argv.extend(extra)
    return argv


def run_v12(argv: list[str]) -> str:
    """跑一次 v12 并从 stdout 解析 hit_type。

    Args:
        argv: v12 CLI 参数列表（不含脚本路径）。

    Returns:
        hit_type 字符串（active/passive/miss）；解析失败返回 "no_result"。
    """
    proc = subprocess.run(
        [str(ENV_PY), "scripts/rm65_mpc_v12.py", *argv],
        cwd=REPO, capture_output=True, text=True, timeout=120, check=False)
    m = RESULT_RE.search(proc.stdout)
    return m.group(1) if m else "no_result"


def verify(plan: dict, top_seed_all_variants: bool = True) -> None:
    """重跑核验：hit_type 与 CSV 一致 + 首选种子四变体 p0/v0 逐位一致。

    Args:
        plan: pick 阶段产出的计划字典（写入 seeds.json 的同构数据）。
        top_seed_all_variants: 是否对首选分歧种子跑满四变体做配对校验。
    """
    all_ok = True
    # 1) 干净命中种子：full @7
    c = plan["clean_hit_7ms"]
    argv = build_argv(c["config_json"], c["seed"], ["--no-plot"])
    got = run_v12(argv)
    ok = got == "active"
    all_ok &= ok
    logger.info("[核验] 7 m/s seed=%d full → %s（期望 active）%s",
                c["seed"], got, "✓" if ok else "✗ 不一致")

    # 2) 分歧种子：none/full @12（首选加跑 softmin_only/tube_only）
    for i, d in enumerate(plan["discordant_12ms"]):
        for mode, csv_key in (("none", "none_hit_type"), ("full", "full_hit_type")):
            cfg = json.loads(d["config_json"]) | {"--ablation": mode}
            argv = build_argv(json.dumps(cfg), d["seed"], ["--no-plot"])
            got = run_v12(argv)
            ok = got == d[csv_key]
            all_ok &= ok
            logger.info("[核验] 12 m/s seed=%d %-13s → %s（期望 %s）%s",
                        d["seed"], mode, got, d[csv_key], "✓" if ok else "✗ 不一致")
        if i == 0 and top_seed_all_variants:
            p0s = verify_p0_pairing(d)
            all_ok &= p0s

    logger.info("核验总结: %s", "全部通过 ✓" if all_ok else "存在不一致 ✗（换种子重试）")


def verify_p0_pairing(d: dict) -> bool:
    """同 seed 四变体 --dump-trajectory 后校验发球 p0/v0 逐位一致。

    Args:
        d: 分歧种子条目。

    Returns:
        四变体 p0/v0 是否全部一致。
    """
    base_cfg = json.loads(d["config_json"])
    p0_ref, v0_ref = None, None
    ok = True
    for mode in ("none", "softmin_only", "tube_only", "full"):
        npz_path = VIDEO_ASSETS / f"paircheck_s{d['seed']}_{mode}.npz"
        argv = build_argv(json.dumps(base_cfg | {"--ablation": mode}), d["seed"],
                          [f"--dump-trajectory={npz_path}", "--no-plot"])
        got = run_v12(argv)
        with np.load(npz_path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
        p0, v0 = np.array(meta["p0"]), np.array(meta["v0"])
        same = (p0_ref is None) or (np.array_equal(p0, p0_ref) and np.array_equal(v0, v0_ref))
        p0_ref, v0_ref = p0, v0
        ok &= same
        logger.info("[配对] seed=%d %-13s hit=%s p0/v0 %s", d["seed"], mode, got,
                    "逐位一致 ✓" if same else "✗ 与首个变体不同")
    return ok


def main() -> None:
    """入口：筛选 → 写 seeds.json →（可选）v12 重跑核验。"""
    parser = argparse.ArgumentParser(description="随稿视频演示种子挑选与核验")
    parser.add_argument("--top", type=int, default=3, help="保留分歧种子数（默认 3）")
    parser.add_argument("--verify", action="store_true", help="重跑 v12 核验 hit_type 与 p0/v0")
    args = parser.parse_args()

    VIDEO_ASSETS.mkdir(parents=True, exist_ok=True)
    plan = {
        "picked_at": "2026-09-20",
        "discordant_12ms": pick_discordant(args.top),
        "clean_hit_7ms": pick_clean_hit(),
    }
    out = VIDEO_ASSETS / "seeds.json"
    out.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("已写入 %s", out)

    for d in plan["discordant_12ms"]:
        logger.info("分歧种子 seed=%d: none=%s full=%s softmin=%s tube=%s "
                    "(none_min_dist=%.3f m, full_err=%.3f m)",
                    d["seed"], d["none_hit_type"], d["full_hit_type"],
                    d["softmin_only_hit_type"], d["tube_only_hit_type"],
                    d["none_min_dist"], d["full_hit_pos_error"])

    if args.verify:
        verify(plan)


if __name__ == "__main__":
    main()
