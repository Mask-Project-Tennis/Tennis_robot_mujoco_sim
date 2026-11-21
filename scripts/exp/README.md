# scripts/exp — 批量实验与论文 smoke 编排

## 何时读

- 新增/运行批量实验（参数矩阵、worker、断点续跑）时。
- 改 R1D 论文 smoke 的 manifest、任务身份、续跑或产物聚合时。
- 需要区分“包装脚本 / 运行器 / 提取器”三层职责时。

## 职责

| 路径 | 职责 |
|------|------|
| `paper_smoke/` | R1D 论文 smoke：manifest、task spec、身份指纹、续跑、产物、catalog、CLI |
| `run_r1d_paper_smoke.py` | paper smoke 兼容入口（委托 `paper_smoke.cli`） |
| `batch_runner.py` | 通用批量运行器（子进程、UTF-8 日志、断点） |
| `run_catalog.py` | 实验/运行目录登记（主代理维护） |
| `_run_expN_*.py` | 单实验包装：monkey-patch 约束 → 构造 `sys.argv` → 调主脚本 |
| `run_expN_batch.py` | 参数矩阵遍历 + 子进程 + 日志 |
| 其余 `exp9`–`exp16` 脚本 | 既有实验设施，保留兼容 |

## 稳定入口

```bash
# 旧入口（推荐给脚本/文档）
python scripts/exp/run_r1d_paper_smoke.py \
  --manifest configs/experiments/r1d_paper_smoke.yaml \
  --output experiment_data/r1d_v12_parity/paper_smoke

# 新入口（等价）
python -m scripts.exp.paper_smoke.cli --help
```

```python
from scripts.exp.paper_smoke import (
    ManifestError,
    build_specs_from_manifest,
    load_manifest,
    resolve_manifest,
    validate_manifest,
)
from scripts.exp.paper_smoke.cli import main
```

- 以上名字在旧/新入口为同一对象；不要复制第二份 manifest 解析。
- `batch_runner.py` / `run_catalog.py` 属主代理独占，改前先确认调用方。

## 允许依赖

- `scripts/exp/` 可依赖 `src` 的公开 API 与标准库；`src/` 不得反向 import `scripts/`。
- `paper_smoke` 不启动真实仿真（测试用 fake）；不在 import 期读 manifest/写文件。
- 不改 `experiment_data/` 原始数据；新产物写到独立 `--output`。

## 测试

```bash
python -m pytest \
  tests/test_paper_smoke_import_contracts.py \
  tests/test_paper_smoke_data_root.py \
  tests/test_r1d_manifest_consistency.py -q
```

## 最小用例

```python
from scripts.exp.paper_smoke import load_manifest, resolve_manifest, build_specs_from_manifest

manifest = load_manifest("configs/experiments/r1d_paper_smoke.yaml")
specs = build_specs_from_manifest(None, seeds=[100, 101])
print(len(specs))
```

更多细节见 `paper_smoke/README.md`。
