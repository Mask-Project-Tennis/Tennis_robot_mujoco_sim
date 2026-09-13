"""aux_stats 辅助统计复现测试（敏感性 + TCP 超限诊断）。

预期值来自 paper/sections/results.tex 的实测数字，锁定回归。
"""
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from scripts.extract.aux_stats import compute_aux


def test_sensitivity_matches_paper() -> None:
    """敏感性 sweep 复现 results.tex 的 93.9–94.9% 与 84.8%。"""
    aux = compute_aux()
    sens = aux["sensitivity"]
    full = [r for r in sens if r["tier"] == "full" and r["knob"] != "nominal"]
    assert min(r["rate"] for r in full) == 93.9
    assert max(r["rate"] for r in full) == 94.9
    tube = [r for r in sens if r["tier"] == "tube_only"]
    assert len(tube) == 2
    assert all(r["rate"] == 84.8 for r in tube)


def test_tcp_exceed_matches_paper() -> None:
    """TCP 超限复现：E2 19.2%/mean 1.09/p90 1.01；E8 每档 19/8/19/6%。"""
    aux = compute_aux()
    e2 = aux["tcp_exceed"]["E2"]
    assert round(e2["exceed_pct"], 1) == 19.2
    assert round(e2["mean"], 2) == 1.09
    assert round(e2["p90"], 2) == 1.01
    e8 = {r["tier"]: round(r["exceed_pct"]) for r in aux["tcp_exceed"]["E8"]}
    assert e8 == {"full": 19, "tube_only": 8, "softmin_only": 19, "none": 6}


def test_exempt_window_within_cap() -> None:
    """豁免窗内 per-step 峰值不超过 1.0 m/s。"""
    aux = compute_aux()
    assert aux["exempt_window"]["max_in_window"] <= 1.0
