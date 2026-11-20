"""R1D 论文 smoke 的 --data-root 贯穿测试（R10）。

旧实现同时传 ``data_root=batch_root`` 与优先级更高的
``output=<run_root>/batch/<spec>``，导致 ``--data-root`` 被静默覆盖、
CSV/依赖/resume 落到不同根。这里 monkeypatch 掉真实仿真执行，只验证
``main`` 传给 ``batch_runner.run_batch`` 的关键字参数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.exp import batch_runner, run_r1d_paper_smoke


def _fake_summary(name: str, **kwargs: Any) -> dict[str, Any]:
    """run_batch 的最小合法返回值（main 只读这些键）。"""
    return {
        "experiment": name,
        "total": 0,
        "done_before": 0,
        "launched": 0,
        "hits_this_run": 0,
        "errors": 0,
        "timeouts": 0,
        "elapsed_s": 0.0,
        "csv": "",
        "data_root": str(kwargs.get("data_root", "")),
        "dry_run": False,
    }


@pytest.fixture()
def captured(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """把 run_batch / 工具依赖 / runs.csv 汇总替换为纯记录桩。"""
    calls: list[dict[str, Any]] = []

    def fake_run_batch(spec: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append({"spec": spec, **kwargs})
        return _fake_summary(spec.name, **kwargs)

    def fake_collect_runs(resolved: Any, batch_root: Path,
                          results: Any) -> dict[str, Any]:
        return {"runs_csv": "runs.csv", "runs": 0,
                "hit_semantics": {"physical": "physical"}}

    def fake_tool(unit: Any, run_root: Path,
                  batch_root: Path) -> dict[str, Any]:
        return {"status": "ok", "log_path": ""}

    monkeypatch.setattr(batch_runner, "run_batch", fake_run_batch)
    monkeypatch.setattr(run_r1d_paper_smoke.batch_runner, "run_batch",
                        fake_run_batch)
    monkeypatch.setattr(run_r1d_paper_smoke, "_collect_runs", fake_collect_runs)
    monkeypatch.setattr(run_r1d_paper_smoke, "_run_tool_dependency", fake_tool)
    return calls


def _run_main(tmp_path: Path, extra: list[str]) -> int:
    """以 tmp 作为输出根执行 smoke main（不跑真实仿真）。"""
    run_root = tmp_path / "run"
    argv = ["--output", str(run_root), "--limit", "2",
            "--allow-single-seed", *extra]
    return run_r1d_paper_smoke.main(argv)


class TestPaperSmokeDataRoot:
    """R10: --data-root 必须贯穿到 batch 输出根。"""

    def test_custom_data_root_reaches_run_batch(
            self, tmp_path: Path, captured: list[dict[str, Any]]) -> None:
        """--data-root <custom> 时 run_batch 收到该根，且不再收到 output。"""
        custom = tmp_path / "custom_csv"
        code = _run_main(tmp_path, ["--data-root", str(custom)])
        assert code == 0
        assert captured, "至少应有一个 batch 单元被执行（桩）"
        for call in captured:
            assert Path(call["data_root"]) == custom
            assert "output" not in call, (
                "R10: output= 会覆盖 data_root，禁止再传")
        # batch CSV 不应再落到 <run>/batch 下
        batch_dir = tmp_path / "run" / "batch"
        assert not batch_dir.exists() or not any(batch_dir.iterdir())

    def test_default_data_root_is_run_root_batch(
            self, tmp_path: Path, captured: list[dict[str, Any]]) -> None:
        """不传 --data-root 时仍为 <run_root>/batch。"""
        code = _run_main(tmp_path, [])
        assert code == 0
        assert captured
        expected = tmp_path / "run" / "batch"
        for call in captured:
            assert Path(call["data_root"]) == expected
            assert "output" not in call
