"""M10 回归：表格脚注里的不可估计原因必须先转义再进 LaTeX 正文。

复审证据：``scripts/plot/paper_figs.py`` 的 Table II 脚注把 ``all_miss`` /
``no_common_seed`` 原样拼进 ``\\par\\footnotesize``，xelatex 因裸下划线报
``Missing $ inserted``。本文件同时验证：

1. 转义函数本身覆盖 ``& % $ # _ { } ~ ^ \\``；
2. 正文（非 ``%`` 注释行）不含裸下划线，注释行保留原始原因；
3. 含换行 / ``%`` 的原因不会截断正文或破坏注释行；
4. JSON 原始值不被改写；
5. 有 xelatex 时真实编译 Table I（physical 口径，动态 mode 名）与 Table II，
   无 xelatex 时如实 skip（不做假编译）。
"""

from __future__ import annotations

import copy
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PROJECT = Path(__file__).resolve().parents[1]
PLOT_DIR = PROJECT / "scripts" / "plot"
if str(PLOT_DIR) not in sys.path:
    sys.path.insert(0, str(PLOT_DIR))

import paper_figs  # noqa: E402  （被测模块）

_MODES = ("tube_only", "softmin_only", "full")
_XELATEX_CANDIDATES = (
    Path("D:/latex/miktex/bin/x64/xelatex.exe"),
    Path("C:/Program Files/MiKTeX/miktex/bin/x64/xelatex.exe"),
)


def _xelatex() -> str | None:
    """定位 xelatex：PATH 优先，其次本机 MiKTeX 常见安装位置。"""
    found = shutil.which("xelatex")
    if found:
        return found
    for candidate in _XELATEX_CANDIDATES:
        if candidate.is_file():
            return str(candidate)
    return None


def _all_miss_stats(*, mode: str = "torque", reason: str = "all_miss") -> dict[str, Any]:
    """构造四档全 miss 的最小统计 JSON（只含 ``tables()`` 必需小节）。

    Args:
        mode: 顶层 ``mode`` 字段（会进入 Table I 的 physical 脚注）。
        reason: 配对不可估计原因，模拟真实 ``all_miss``。

    Returns:
        顶层与 ``physical`` 段落结构一致的最小统计字典。
    """
    paired = {
        f"{tcp}|{variant}_vs_none": {"n_paired": 0, "unavailable_reason": reason}
        for tcp in ("TCP1.8", "TCP1.0")
        for variant in _MODES
    }
    did = {
        variant: {"n_pairs": 0, "unavailable_reason": "no_common_seed"}
        for variant in _MODES
    }
    section: dict[str, Any] = {
        "E3_nominal": {"cells": {}},
        "E4_grid": {"merged_20cell": {}},
        "E7_corners": {"cells": {}},
        "E8_limit_mechanism": {"paired": paired, "did": did},
    }
    stats: dict[str, Any] = {"mode": mode, **copy.deepcopy(section)}
    stats["physical"] = copy.deepcopy(section)
    return stats


def _body_text(tex: str) -> str:
    """取 TeX 里真正进入正文的行（排除 ``%`` 注释行）。"""
    return "\n".join(
        line for line in tex.splitlines() if not line.lstrip().startswith("%")
    )


def _compile_table(tmp_path: Path, tex: str) -> subprocess.CompletedProcess[str]:
    """把表格片段包进最小文档，真实调用 xelatex 编译。

    Args:
        tmp_path: 编译工作目录。
        tex: 表格片段（可含 ``table`` / ``table*`` 环境）。

    Returns:
        编译进程结果（``returncode`` 为 0 表示无错误）。
    """
    document = "\n".join(
        [
            r"\documentclass{article}",
            r"\usepackage{booktabs}",
            r"\begin{document}",
            tex,
            r"\end{document}",
            "",
        ]
    )
    path = tmp_path / "synthetic_table.tex"
    path.write_text(document, encoding="utf-8")
    exe = _xelatex()
    assert exe is not None, "调用方应先检查 xelatex 是否存在"
    return subprocess.run(
        [
            exe,
            "-disable-installer",
            "-halt-on-error",
            "-interaction=nonstopmode",
            path.name,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        errors="replace",
        timeout=180,
        check=False,
    )


class TestTexEscape:
    """``_tex_escape`` 的字符覆盖与顺序语义。"""

    def test_escape_specials(self) -> None:
        """下划线/百分号/反斜杠/花括号均被转义成可编译文本。"""
        assert paper_figs._tex_escape("all_miss") == r"all\_miss"
        assert paper_figs._tex_escape("50%") == r"50\%"
        assert paper_figs._tex_escape("a&b") == r"a\&b"
        assert paper_figs._tex_escape("$x_1$") == r"\$x\_1\$"
        assert paper_figs._tex_escape("#1") == r"\#1"
        assert paper_figs._tex_escape("{a}") == r"\{a\}"
        assert paper_figs._tex_escape("\\") == r"\textbackslash{}"
        assert paper_figs._tex_escape("a~b") == r"a\textasciitilde{}b"
        assert paper_figs._tex_escape("a^b") == r"a\textasciicircum{}b"

    def test_backslash_not_double_escaped(self) -> None:
        """先保护反斜杠：``\\_`` 只得到 ``\\textbackslash{}\\_``，不产生二次转义。"""
        assert paper_figs._tex_escape("\\_") == r"\textbackslash{}\_"


class TestTableTexBody:
    """表格产物：正文转义、注释保留原文、JSON 不被修改。"""

    def test_table2_body_has_no_bare_underscore(self, tmp_path: Path) -> None:
        """Table II 正文不含裸下划线，原因以 ``all\\_miss`` 形式出现。"""
        paper_figs.tables(_all_miss_stats(), tmp_path, hit_semantics="legacy")
        tex = (tmp_path / "table2_ablation.tex").read_text(encoding="utf-8")
        body = _body_text(tex)
        assert "_" not in body.replace("\\_", "")
        assert r"all\_miss" in body

    def test_comments_keep_raw_reason(self, tmp_path: Path) -> None:
        """``% unavailable:`` 注释行保留原始原因（便于 grep），且每行一条。"""
        paper_figs.tables(_all_miss_stats(), tmp_path, hit_semantics="legacy")
        tex = (tmp_path / "table2_ablation.tex").read_text(encoding="utf-8")
        comments = [
            line for line in tex.splitlines() if line.startswith("% unavailable:")
        ]
        assert len(comments) == 9  # 3 档 × 2 TCP + 3 档 DiD
        assert all("all_miss" in line for line in comments[:6])
        assert all("no_common_seed" in line for line in comments[6:])

    def test_newline_and_percent_reason_do_not_break_comments(
        self, tmp_path: Path
    ) -> None:
        """原因含换行与 ``%`` 时：注释折叠成单行，正文百分号被转义。"""
        stats = _all_miss_stats(reason="all_miss\n50%_conflict")
        paper_figs.tables(stats, tmp_path, hit_semantics="legacy")
        tex = (tmp_path / "table2_ablation.tex").read_text(encoding="utf-8")
        comments = [
            line for line in tex.splitlines() if line.startswith("% unavailable:")
        ]
        assert len(comments) == 9
        assert any("all_miss 50%_conflict" in line for line in comments)
        body = _body_text(tex)
        assert r"50\%\_conflict" in body
        # ``%`` 若未转义会把同处正文全部注释掉：转义后 50 仍在可见正文里
        assert "50" in body

    def test_json_payload_not_mutated(self, tmp_path: Path) -> None:
        """出表只读统计 JSON：原始值（含裸下划线）保持不变。"""
        stats = _all_miss_stats()
        before = copy.deepcopy(stats)
        paper_figs.tables(stats, tmp_path, hit_semantics="legacy")
        assert stats == before
        reason = stats["physical"]["E8_limit_mechanism"]["paired"][
            "TCP1.8|full_vs_none"
        ]["unavailable_reason"]
        assert reason == "all_miss"

    def test_table1_physical_footnote_escapes_mode(self, tmp_path: Path) -> None:
        """Table I physical 脚注的动态 mode 名同样转义。"""
        stats = _all_miss_stats(mode="torque_position")
        paper_figs.tables(stats, tmp_path, hit_semantics="physical")
        tex = (tmp_path / "table1_comparison.tex").read_text(encoding="utf-8")
        assert r"mode=torque\_position" in tex
        assert "_" not in _body_text(tex).replace("\\_", "")


class TestXelatexCompile:
    """真实编译（无 xelatex 时 skip，绝不假编译）。"""

    def test_tables_compile_with_xelatex(self, tmp_path: Path) -> None:
        """physical 口径的 Table I 与 Table II 都能被 xelatex 编译通过。"""
        if _xelatex() is None:
            pytest.skip("未安装 xelatex，跳过真实编译")
        stats = _all_miss_stats(mode="torque_position")
        out_dir = tmp_path / "table_data"
        paper_figs.tables(stats, out_dir, hit_semantics="physical")
        build_dir = tmp_path / "build"
        build_dir.mkdir()
        for name in ("table1_comparison.tex", "table2_ablation.tex"):
            tex = (out_dir / name).read_text(encoding="utf-8")
            result = _compile_table(build_dir, tex)
            tail = ((result.stdout or "") + (result.stderr or ""))[-2000:]
            assert result.returncode == 0, f"{name} 编译失败：\n{tail}"
