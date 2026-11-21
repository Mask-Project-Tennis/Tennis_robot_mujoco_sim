# scripts/extract — 结果提取与统计分析

## 何时读

- 从实验日志/产物生成 CSV/JSON 汇总时。
- 改配对统计、attempt/mode 隔离、physical 来源或论文统计口径时。
- 需要确认旧 `paired_stats.py` 的兼容入口与正式子包边界时。

## 职责

| 路径 | 职责 |
|------|------|
| `paired_analysis/` | 论文配对统计：类型/身份/来源/数据集/指标/比较/报告/CLI |
| `paired_stats.py` | 兼容 facade（145 个符号重导出；旧 `--help`/统计输出逐字节保持） |
| `extract_*.py` | 各实验日志/产物的专用提取器 |
| `aux_stats.py` | 辅助统计（保留兼容） |

## 稳定入口

```bash
python scripts/extract/paired_stats.py --help
python -m scripts.extract.paired_analysis.cli --help
```

```python
from scripts.extract.paired_analysis.types import HitPolicy, StatsContext
from scripts.extract.paired_analysis.sources import make_context
from scripts.extract.paired_analysis.metrics import paired_test, paired_test_physical, physical_of, physical_stats
```

- 旧/新入口为同一对象（`is` 相同）；`paired_stats.py` 不再含统计实现。
- 统计口径红线：final attempt 只取最终尝试；mode 隔离；physical 来源冲突显式；配对键/`bootstrap(20000, seed=0)` 不变。

## 允许依赖

- `scripts/extract/` 可依赖 `src` 公开 API、numpy/pandas 与标准库；`src/` 不得反向 import `scripts/`。
- `paired_analysis` 的 `data.py`/`cli.py` 用 `parents[3]` 定位仓库根；不要用 CWD 猜路径。
- 不改输入 CSV/NPZ/JSON；测试先复制到独立临时目录。

## 测试

```bash
python -m pytest \
  tests/test_paired_stats_semantics.py \
  tests/test_paper_stats_catalog.py \
  tests/test_stats_attempt_resume.py \
  tests/test_r1d_statistics.py \
  tests/test_paired_analysis_imports.py -q
```

## 最小用例

```bash
python scripts/extract/paired_stats.py --help
```

更多细节见 `paired_analysis/README.md`。
