# scripts/plot/paper —— 论文图表子包

`scripts/plot/paper_figs.py`（原 2091 行）按职责拆分到本子包；旧文件仅保留
兼容 facade（重导出 + 可变全局转发），**旧导入方式与行为不变**。

## 模块职责

| 文件 | 内容 | matplotlib |
|------|------|-----------|
| `data.py` | `PROJECT`/`DATA`/`OUT` 三个可变全局；`resolve_data_root`、`_data_path`、`resolve_stats_json`、`load_stats` | 不 import |
| `style.py` | `C` 色表、`_RC_PARAMS`、`apply_style()`、`pyplot()`、`style_ax`、`ci_half`、`fmt_p`、`fmt_pv`、`save` | 仅函数内延迟 import |
| `trajectory.py` | `_corridor_band`、`fig1_hero`、`fig3_3d`、`fig3_alt_2d`、`_fig3_alt_2d_old`、`fig4`、`fig3_hero` | 函数内 |
| `comparisons.py` | `_exp26_paired_gains`、`_exp26_gains_ci`、`fig5`、`fig5_interaction`、`_holm_sig`、`fig6` | 函数内 |
| `diagnostics.py` | `fig_realtime`（fig8）、`fig_diagnostic`（fig7） | 函数内 |
| `tables.py` | `_tex_escape`、`_table_section`、`tables`、`build_table3_preview`、`_table3_tex`、`_table3_pdf`、`table3_shared_input`、Table I/II/3 常量与数字格式化 | 仅 `pyplot()` |
| `cli.py` | `build_parser()`、`main()` | 经各绘图函数 |
| `__init__.py` | 轻量重导出，无 IO | 不 import |

依赖方向单向：`data ← style / trajectory / comparisons / diagnostics / tables ← cli`。

## 两条不变量

1. **import 不做重活**：导入本包或旧 facade 不会初始化 matplotlib、不读数据、
   不改 `sys.path`；matplotlib 只在明确绘图入口（`style.pyplot()`/`apply_style()`）
   才 `use("Agg")` 并应用论文 rcParams。
2. **可变全局单一事实来源**：`PROJECT`/`DATA`/`OUT` 只存在于 `paper/data.py`；
   绘图模块在运行时通过 `from . import data as _data` 取值，因此
   `main()` 的 `--data-root`/`--out` 对全部图表生效。旧 facade 通过模块类
   `__getattr__`/`__setattr__` 转发这三个名字，`monkeypatch.setattr(paper_figs,
   "DATA", ...)` 与直接读 `paper_figs.DATA` 语义不变。

## 用法

```bash
# 与拆分前一致：显式选择图，输出到临时目录（不写 paper/figures）
python scripts/plot/paper_figs.py --fig 3 4 5 --out /tmp/paper_figs

# 表格与 table3 预览
python scripts/plot/paper_figs.py --fig table table3 --out /tmp/paper_figs

# 旧式路径导入
python -c 'import sys; sys.path.insert(0, "scripts/plot"); import paper_figs; print(paper_figs.fig5.__module__)'
```

无头环境默认 `Agg`；缺数据时仍按原行为打印原因并跳过（不补数、不静默写
`paper/figures/`）。

## 测试与检查

```bash
MPLBACKEND=Agg python -m pytest \
  tests/test_paper_figs_paths.py tests/test_paper_figs_tex_escape.py \
  tests/test_r1d_paper_pipeline.py tests/test_paper_plot_import_contracts.py -q
python -m ruff check scripts/plot/paper scripts/plot/paper_figs.py \
  tests/test_paper_plot_import_contracts.py
```

`tests/test_paper_plot_import_contracts.py` 覆盖：新旧对象 is 相同、import 后
`matplotlib` 不在 `sys.modules`（子进程）、`PROJECT` 与 cwd 无关、统计 JSON 回退
顺序、facade 上 monkeypatch `DATA` 生效、表格 TeX 转义/空值语义、`build_parser`、
源码结构（facade 行数、无反向 import、data/tables 无 matplotlib），以及固定小样本
无头渲染（PDF 落到临时 `OUT`，不触碰 `paper/figures/`）。


## 何时读

改论文图/表、TeX 转义、字号配色或输出路径时；只跑图用 `scripts/plot/paper_figs.py` 或
`python -m scripts.plot.paper.cli`。

## 允许依赖

`paper.data`/`paper.tables` 可依赖标准库、numpy/pandas 与 `src` 公开 API；
`matplotlib` 只在 `paper.style`/绘图入口函数内延迟导入，数据层不得 import matplotlib。
`src/` 不得反向 import 本包。

## 最小用例

```bash
MPLBACKEND=Agg python -m scripts.plot.paper.cli \
  --data-root paper/planning --out /tmp/paper-figs --fig table3
```
