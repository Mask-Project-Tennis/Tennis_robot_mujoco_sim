#!/usr/bin/env python3
"""overnight_gate —— 连夜重跑的 P0 门禁：环境验证 + 复现性抽查自动判定。

门禁项（全部通过才放行后续实验）:
1. cpp    C++ iLQR 扩展可导入
2. repro  V12 标称性能抽查: 9m/s 与 7m/s 各 10 seeds, 命中 ≥7/10
          （对照 exp15 报告 90% @9m/s, exp13 报告 85.7% @7m/s;
           允许 ±10pp 漂移 —— 6/18 报告后代码仍有提交, 不要求逐位复现）
3. limits exp16 真机限位配置生效验证: real 组 max_tcp ≈ 1.0（sim 默认 ≈ 1.8）

输出: experiment_data/_driver_state/gate.json + stdout 明细, 退出码 0=PASS。

用法:
    python scripts/exp/overnight_gate.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))
STATE_DIR = PROJECT / "experiment_data" / "_driver_state"
PY = sys.executable


def check_cpp() -> dict:
    """验证 C++ iLQR 扩展可导入。

    经 src.cpp.solver_cpp（python wrapper）加载 —— 与仿真代码实际路径
    一致; 不直接 import .so（那需要 conda activate 的 LD_LIBRARY_PATH，
    批量场景不成立, 属误报路径）。
    """
    try:
        proc = subprocess.run(
            [PY, "-c",
             "import src.cpp.solver_cpp as m; print('iLQR C++ 模块已加载')"],
            capture_output=True, text=True, timeout=60, cwd=str(PROJECT))
        ok = proc.returncode == 0
        return {"name": "cpp", "pass": ok,
                "detail": proc.stdout.strip() or proc.stderr.strip()[-200:]}
    except Exception as exc:  # noqa: BLE001 —— 门禁需捕获一切异常并落盘
        return {"name": "cpp", "pass": False, "detail": repr(exc)}


def _hit_rate(speed: int, seeds: list[int]) -> dict:
    """跑一批 V12 力矩模式 run，返回命中率明细。"""
    from src.sim.hit_detection import determine_hit_from_type
    from scripts.exp.batch_runner import parse_result_line

    hits, errors, details = 0, 0, []
    for seed in seeds:
        cmd = [PY, str(PROJECT / "scripts/rm65_mpc_v12.py"),
               "--serve-box", "--ball-speed", str(speed),
               "--seed", str(seed), "--no-plot"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=90, cwd=str(PROJECT))
            fields = parse_result_line(proc.stdout + proc.stderr)
        except subprocess.TimeoutExpired:
            fields = {"error": "timeout"}
        if "error" in fields:
            errors += 1
        elif determine_hit_from_type(str(fields.get("hit_type", "miss"))):
            hits += 1
        details.append({"seed": seed, "hit": "error" not in fields
                        and bool(determine_hit_from_type(
                            str(fields.get("hit_type", "miss")))),
                        "error": fields.get("error", ""),
                        "pos_error": fields.get("pos_error")})
    return {"hits": hits, "n": len(seeds), "errors": errors, "details": details}


def check_repro() -> dict:
    """V12 标称性能抽查: 9m/s 与 7m/s 各 10 seeds, 双双 ≥7 命中才 PASS。"""
    results = {}
    for speed, report_rate in [(9, "90.0%"), (7, "85.7%")]:
        t0 = time.perf_counter()
        r = _hit_rate(speed, list(range(1, 11)))
        results[f"speed{speed}"] = {**r, "report_ref": report_rate,
                                    "elapsed_s": round(time.perf_counter() - t0, 1)}
    ok = (results["speed9"]["hits"] >= 7 and results["speed7"]["hits"] >= 7)
    return {"name": "repro", "pass": ok,
            "detail": f"9m/s {results['speed9']['hits']}/10 (报告 90%), "
                      f"7m/s {results['speed7']['hits']}/10 (报告 85.7%)",
            "results": results}


def check_limits() -> dict:
    """exp16 真机限位配置生效验证: real 组 max_tcp 应 ≈1.0 而非默认 1.8。"""
    from scripts.exp.batch_runner import parse_result_line

    max_tcps = []
    for seed in [1, 2, 3]:
        cmd = [PY, str(PROJECT / "scripts/rm65_mpc_v12.py"),
               "--serve-box", "--ball-speed", "7", "--seed", str(seed),
               "--no-plot", "--limits-config", "configs/real_robot.yaml"]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=90, cwd=str(PROJECT))
            fields = parse_result_line(proc.stdout + proc.stderr)
            if "max_tcp" in fields:
                max_tcps.append(float(fields["max_tcp"]))
        except subprocess.TimeoutExpired:
            pass
    # real 限位下 max_tcp 不应超过 1.05（报告: real 均值 0.99, sim 1.58-1.8）
    ok = bool(max_tcps) and max(max_tcps) < 1.05
    return {"name": "limits", "pass": ok,
            "detail": f"real 组 max_tcp={max_tcps}（预期 <1.05; sim 默认约 1.8）"}


def main() -> None:
    """跑三项门禁并写状态文件, 退出码 0=全部通过。"""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    checks = [check_cpp(), check_repro(), check_limits()]
    state = {
        "gate_pass": all(c["pass"] for c in checks),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "checks": checks,
    }
    (STATE_DIR / "gate.json").write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    for c in checks:
        print(f"  [{'PASS' if c['pass'] else 'FAIL'}] {c['name']}: {c['detail']}")
    print(f"==> GATE {'PASS' if state['gate_pass'] else 'FAIL（停止, 不跑后续实验）'}")
    sys.exit(0 if state["gate_pass"] else 1)


if __name__ == "__main__":
    main()
