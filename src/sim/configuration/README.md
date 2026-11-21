# configuration — V12 配置层

B1 迁移完成后，V12 任务配置的**全部解析实现**都位于本包；旧模块
`src/sim/v12_config.py` 只保留同名对象重导出，保证旧导入路径、`__all__`、
对象身份（`is`）与行为不变。

## 模块布局

| 模块 | 职责 |
|------|------|
| `types.py` | `ResolvedV12Config` 唯一类定义、`_ConfigDict`、冻结/深拷贝/递归相等比较 |
| `validation.py` | `_v_*` 校验器、`_OVERRIDE_VALIDATORS` 白名单注册表、任务覆盖展开与校验 |
| `defaults.py` | CLI 缺省值表、机器人配置来源合并、基线值构建、服务箱自动值、`_set_value` |
| `selection.py` | 机器人选择（RM65/R1D）、`R1DConfigRegistry`、`R1D_V12_REGISTRY`、任务覆盖 YAML 校验 |
| `overrides.py` | `normalize_explicit_options`、任务覆盖应用、显式 CLI 覆盖、PD 增益解析 |
| `resolution.py` | 仓库根/配置目录常量、`_finalize` 收尾派生、唯一入口 `resolve_v12_config` |

## 依赖方向（严格单向）

```
types  ←  validation
types  ←  defaults
types  ←  selection  ←  validation, defaults
         overrides  ←  validation, defaults, selection
         resolution ← types, validation, defaults, selection, overrides
```

- `types.py` 不 import 本包任何上层模块，也不 import 旧 facade；
- 下层不得反向 import `resolution`；`defaults.py` 仅以参数注入方式接收
  `configs_dir`，`overrides.py` 仅以参数注入方式接收 `repo_root`，避免环；
- 旧 facade 只做同名对象重导出，不包含第二份解析实现，也不 import MuJoCo/YAML 运行时依赖。

## 仓库根与配置目录

旧 `src/sim/v12_config.py` 的 `_REPO_ROOT = Path(__file__).resolve().parents[2]`
在迁入 `src/sim/configuration/resolution.py` 后改为
`Path(__file__).resolve().parents[3]`：

```
resolution.py → configuration → sim → src → 工作树根
```

`_CONFIGS_DIR = _REPO_ROOT / "configs"` 必须始终存在。`selection.py` 在
import 时就要为 `R1D_V12_REGISTRY` 构造绝对路径，因此单独派生同一份仓库根；
两处都会在 `tests/test_configuration_migration.py` 中被锁定（含从非仓库
cwd 解析的用例）。

## 稳定入口

```python
from src.sim.configuration import (
    R1D_V12_REGISTRY,
    R1DConfigRegistry,
    ResolvedV12Config,
    normalize_explicit_options,
    resolve_v12_config,
)
```

旧路径继续可用且对象同一：

```python
from src.sim.v12_config import resolve_v12_config

assert resolve_v12_config is src.sim.configuration.resolve_v12_config
```

私有兼容名 `_OVERRIDE_VALIDATORS`、`_ConfigDict` 仍在旧 facade 重导出，
供 `scripts/exp/run_r1d_paper_smoke.py` 与测试使用。

## 行为红线（迁移期间保持不变）

- 来源顺序：`default.yaml` → `v5_active_hit.yaml` → `mpc.yaml` → V12 字面量
  默认 → 服务箱自动值 → 显式 CLI；R1D 再叠加 `r1d_v12.yaml` 与任务覆盖；
- `False` / `0` / `None` 三分语义，显式值不被 truthiness 吞掉；
- `sources` 逐字段来源标签、`explicit_options` 仅记录显式项；
- 异常类型与消息口径不变（含 `_validate_overrides` 的历史路径提示文本）。

## 测试

```bash
python -m pytest tests/test_v12_config.py tests/test_configuration_migration.py -q
python -m ruff check src/sim/configuration src/sim/v12_config.py tests/test_configuration_migration.py
```

## 最小 import 用例

```python
from src.sim.configuration import ResolvedV12Config, resolve_v12_config

print(ResolvedV12Config.__module__)  # src.sim.configuration.types
print(resolve_v12_config.__module__)  # src.sim.configuration.resolution
```


## 何时读

改 V12 配置优先级、来源追踪、R1D 注册表或 CLI 显式覆盖时先读本文件；
只跑固定案例时直接看根目录 README / `docs/r1d_v12_usage.md`。
