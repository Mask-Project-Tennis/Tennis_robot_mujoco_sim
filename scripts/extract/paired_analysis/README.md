# paired_analysis —— 论文统计口径统一提取（模块化实现）

本包是 `scripts/extract/paired_stats.py`（原 2682 行）按职责拆分后的实现。
旧文件已缩为兼容 facade（显式重导出、`main` 直通、`__main__` 守卫），
既有调用方与测试无需修改即可继续使用。

## 模块边界

| 模块 | 行数 | 职责 |
|------|------|------|
| `types.py` | 265 | 异常类（`StatsError`/`ModeSelectionError`/`DuplicateRecordError`/`IdentityError`/`DatasetConflictError`）、数据结构（`StatsContext`/`RecordIdentity`/`PhysicalIndex`/`HitPolicy`）、共享常量（`SCHEMA`/`MODES`/`VALID_MODES`/`HIT_SEMANTICS`/`STATS_FILENAMES`/`MULTI_MODE_*`/`SEED_DIR_RE`/`TIER_LABELS`）、数据根（`DATA`/`DEFAULT_DATA_ROOT`）、纯解析原语（`cfg_of`/`is_error`/`_to_bool_or_none`） |
| `identity.py` | 433 | 行身份解析：`record_identity`/`tier_of`/`case_of`/`mode_of`/`overrides_of`/`speed_of`/`time_perturb_ms_of`/`space_perturb_of`/`tcp_label_of`/`shared_input_of`、config 指纹与 feedforward/base-tier 判定 |
| `sources.py` | 501 | 磁盘读取与来源解析：`_read_csv_rows`/`_runs_csv_paths`/catalog 绑定（`make_context`）、physical 列富化、artifact 去重与来源冲突（`_physical_index`/`_resolve_physical_source`）、`load`/`_load_rows` |
| `dataset.py` | 350 | attempt 收敛与数据集校验：`_group_attempts`/`_check_attempts`/`_final_attempt_rows`/`_validate_dataset`、`_discover_modes`/`resolve_mode` |
| `metrics.py` | 366 | 统计量：`wilson`/`mcnemar_exact`/`paired_bootstrap`/`sign_test`、`cell_stats`/`physical_stats`/`paired_test`/`paired_test_physical`、`baseline_cell`/`physical_of` |
| `comparisons.py` | 616 | 各实验对比（`e1_speed_sweep`…`e9_shared_input`、`exclusion_report`）与配对 DiD 内部实现（`_did`/`_did_reason`/`_tier_cell`） |
| `report.py` | 590 | 报告汇总与写出：`compute_summary`/`build_tables`/`default_output_dir`、`_print_report`/`_write_stats`/`_multi_mode_index`、physical 冲突拒绝与格式化辅助 |
| `cli.py` | 156 | CLI 契约：`PROJECT`（`parents[3]`）、`build_parser`、`main`；模块 docstring 即 `--help` 描述 |

## 允许依赖

- 标准库（`argparse`/`csv`/`json`/`math`/`pathlib`/`statistics` 等）；不依赖 `pandas`、`numpy`、`scipy`、`matplotlib`。
- 同包内按上图单向依赖；`cli.py` 是唯一组合根，旧 facade 只调用本包，本包不反向 import 旧 facade。
- 不依赖 `src/`、其他 `scripts/` 子包或当前工作目录。

## 依赖方向

```text
types（叶）
  ↑
identity   sources
  ↑          ↑
  └─ dataset ┘
  ↑
metrics ← sources/identity
  ↑
comparisons
  ↑
report
  ↑
cli（唯一组合根，供 facade 调用）
```

除以下两个**函数内延迟导入**（避免 `sources ↔ dataset` 模块级环）外，全部依赖为单向：

- `sources._physical_index` → `.dataset._final_attempt_rows`
- `sources._load_rows` → `.dataset._final_attempt_rows`

`types.py` 是唯一计算仓库根的地方（`_repo_root()` 返回 `Path(__file__).resolve().parents[3]`）；
`cli.PROJECT` 按 brief 保留字面量 `parents[3]`。任何模块都不得再各自用 `__file__` 推断路径，
也不得依赖当前工作目录。

## 兼容入口

旧路径 `scripts.extract.paired_stats`（以及 `sys.path` 注入后 `import paired_stats`）
继续导出全部 122 个原顶格符号、`PROJECT`/`DEFAULT_DATA_ROOT`/`DATA` 与 21 个 stdlib 模块属性，
且与子模块中的定义是**同一对象**（`is` 相同），不做包装副本。

`python scripts/extract/paired_stats.py --help` / 直接运行脚本的行为保持不变；
package 内也可用 `python -m scripts.extract.paired_analysis.cli` 调试。

## 关键口径（不可回退）

- 主指标 active-hit：`hit_type == "active"`；`--include-passive` 复现旧合并口径；
- physical 口径来自 `runs.csv` 的 `physical_hit_type`/`physical_hit`，缺列 → `rate=null` + `unavailable_reason`，**不伪造 0.0**；有判定但全 miss → `rate=0.0` + `all_miss`；
- 同一身份至多一条完成尝试，重复完成记录直接报 `DuplicateRecordError`，不做静默去重；
- 模式隔离：同一统计只允许一种执行模式入池，混合数据必须显式 `--mode`；
- 配对检验：McNemar 精确检验、配对 bootstrap（`n_boot=20000, seed=0`）、配对 DiD（`random.Random(42)`）。

## 测试

```bash
python -m pytest tests/test_paired_stats_semantics.py tests/test_paper_stats_catalog.py \
  tests/test_stats_attempt_resume.py tests/test_r1d_statistics.py \
  tests/test_paired_analysis_imports.py -q
python -m ruff check scripts/extract/paired_analysis scripts/extract/paired_stats.py \
  tests/test_paired_analysis_imports.py
```

`tests/test_paired_analysis_imports.py` 覆盖：facade/子模块对象同一性、`PROJECT` 任意 cwd
解析、CLI 参数与退出码、数据结构字段顺序/默认值、8 个子模块独立冷启动导入、AST 依赖图无环。


## 何时读

改配对统计口径、来源优先级、attempt/mode 隔离或论文表格列时；
只跑统计用 `scripts/extract/paired_stats.py`。

## 最小用例

```python
from scripts.extract.paired_analysis.types import HitPolicy, StatsContext
from scripts.extract.paired_analysis.sources import make_context
from scripts.extract.paired_analysis.metrics import paired_test
from scripts.extract.paired_analysis.cli import main

# ctx = make_context(...) 后调用 paired_test(ctx, ...)
print(HitPolicy, StatsContext, callable(paired_test), callable(main))
```
