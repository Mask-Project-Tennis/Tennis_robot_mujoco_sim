# scripts/plot — 论文图表与诊断绘图

## 何时读

- 生成/复现论文图表、表格或 R1D per-case smoke 图时。
- 改字号、配色、TeX 转义或图名/输出路径时。
- 需要确认旧 `paper_figs.py` 与 `paper/` 子包边界时。

## 职责

| 路径 | 职责 |
|------|------|
| `paper/` | 论文图表子包：数据解析、样式、轨迹图、对比图、诊断图、表格、CLI |
| `paper_figs.py` | 兼容 facade（旧 `import paper_figs` / CLI 仍可用） |
| `plot_r1d_smoke_stats.py` | R1D smoke 统计图 |
| 其余 `plot_*.py` | 既有诊断/结果图 |

## 稳定入口

```bash
# 新入口
python -m scripts.plot.paper.cli --help

# 旧入口（兼容）
python scripts/plot/paper_figs.py --help

# 只生成 table3
python scripts/plot/paper_figs.py --data-root results --out /tmp/figs --fig table3
```

```python
import sys
sys.path.insert(0, "scripts/plot")
import paper_figs  # 旧式导入仍可用

from scripts.plot.paper import main, tables
```

- 图名、TeX 转义、表格列与输出文件名属冻结契约；改动需对拍旧实现。
- `matplotlib` 只在绘图入口延迟导入；`paper.data` 等数据层不得 import matplotlib。

## 允许依赖

- `scripts/plot/` 可依赖 `src` 公开 API、numpy/pandas/matplotlib 与标准库；`src/` 不得反向 import `scripts/`。
- `paper/data.py` 用 `parents[3]` 定位仓库根；不要用 CWD。
- 不把大图/中间产物写入仓库；用 `--out` 指向临时/结果目录。

## 测试

```bash
MPLBACKEND=Agg python -m pytest \
  tests/test_paper_figs_paths.py \
  tests/test_paper_figs_tex_escape.py \
  tests/test_r1d_paper_pipeline.py \
  tests/test_paper_plot_import_contracts.py -q
```

## 最小用例

```bash
MPLBACKEND=Agg python scripts/plot/paper_figs.py \
  --data-root paper/planning --out /tmp/paper-figs --fig table3
```

更多细节见 `paper/README.md`。
