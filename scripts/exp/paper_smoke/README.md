# scripts/exp/paper_smoke

R1D 论文 smoke 实验的 manifest 数据契约、任务展开、续跑与 CLI 实现。

C1 拆分自 `scripts/exp/run_r1d_paper_smoke.py`；旧路径现为兼容 facade，行为与符号
（含既有测试使用的私有函数、`subprocess`/`yaml`/`batch_runner` 模块属性）保持不变。

## 模块

| 模块 | 职责 |
|------|------|
| `manifest.py` | `ManifestError`、manifest 读取/校验/规范化、override 白名单、路径解析 |
| `identity.py` | `config_hash`、runtime 覆盖快照、`config_id` 与续跑身份键 |
| `tasks.py` | resolved manifest、任务规格与 argv 构造、`resolve_manifest` / `build_specs_from_manifest` |
| `artifacts.py` | `v12_episode.json` 指标提取、可搬移路径、NPZ 读取 |
| `resume.py` | 历史 `runs.csv` 读取、指标保留、实际路径回填、完成产物校验 |
| `catalog.py` | `catalog.json` 声明 |
| `cli.py` | argparse 入口 `main`、工具依赖执行、退出码 |

公开签名保持不变：

```python
load_manifest(path: str | Path | None = None) -> dict[str, Any]
validate_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]
resolve_manifest(manifest, *, manifest_path=None, data_root=None, output=None,
                 seeds=None, mode="torque", limit=None, timeout_s=None,
                 allow_single_seed=False) -> dict[str, Any]
build_specs_from_manifest(manifest_path=None, *, data_root=None, output=None,
                          seeds=None, mode="torque", limit=None, timeout_s=None,
                          workers=1, allow_single_seed=False) -> list[ExperimentSpec]
main(argv: Sequence[str] | None = None) -> int
```

## 用法

```bash
# 旧入口（推荐给脚本/文档）
python scripts/exp/run_r1d_paper_smoke.py --dry-run

# 新入口（等价）
python -m scripts.exp.paper_smoke.cli --dry-run
```

`import scripts.exp.paper_smoke` 只加载轻量 manifest/tasks 代码，不启动子进程、
不写文件；重型路径请直接导入 `scripts.exp.paper_smoke.cli`。


## 何时读

改 paper smoke 的 manifest 校验、任务身份、续跑键、产物聚合或 CLI 参数时；
只跑实验用根目录 `scripts/exp/run_r1d_paper_smoke.py`。

## 稳定入口

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

## 允许依赖

可依赖标准库、`src` 公开 API 与 `batch_runner` 的纯函数；不得在 import 期启动子进程/
读大 manifest/写文件；`src/` 不得反向 import 本包。

## 测试

```bash
python -m pytest tests/test_paper_smoke_import_contracts.py \
  tests/test_paper_smoke_data_root.py tests/test_r1d_manifest_consistency.py -q
```

## 最小用例

```python
from scripts.exp.paper_smoke import load_manifest, build_specs_from_manifest

manifest = load_manifest("configs/experiments/r1d_paper_smoke.yaml")
specs = build_specs_from_manifest(None, seeds=[100, 101])
print(len(manifest), len(specs))
```
