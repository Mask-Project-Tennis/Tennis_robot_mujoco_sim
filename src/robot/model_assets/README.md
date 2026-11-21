# model_assets — 机器人模型资产依赖解析层

## 当前阶段（A1）

本包承接原 `src/robot/model_factory.py` 中的 **MJCF/URDF 文件依赖闭包解析** 实现：
常量与 XML 扫描在 `references.py`，哈希/仓库相对路径/条目组装与
`collect_model_file_dependencies` 在 `dependencies.py`。旧模块
`src/robot/model_factory.py` 只做显式 `from ... import X as X` 重导出，
新旧导入返回同一对象（`is` 相同），冻结签名与依赖清单字段/排序不变。

仓库根由 `dependencies.py` 的文件位置解析（`parents[3]`：`dependencies.py →
model_assets → robot → src → repo`），因此无论从哪个 cwd 调用，仓库内模型都
保持仓库相对路径。

## 职责

- `references.py`：XML 注释等长掩码、属性值解析、`meshdir`/`texturedir`
  与 `file` 的相对路径候选、两阶段（先收集引用、再解析资产）MJCF 扫描，
  含 include 嵌套深度/环检测与缺失路径记录。
- `dependencies.py`：仓库根解析、文件 SHA256、仓库内/外部相对路径命名、
  依赖条目构造（含 `content_base64` 与 `asset_dirs`），以及公开入口
  `collect_model_file_dependencies` 的组装、去重与排序。
- 稳定导出：`collect_model_file_dependencies`、5 个 `DEPENDENCY_ROLE_*`
  角色常量、`DEPENDENCY_XML_ROLES` 与 `mask_xml_comments`。

## 不负责

- 不加载 MuJoCo 模型、不生成 XML 模板、不装配机器人模型（仍在
  `src/robot/model_factory.py`，A2 再迁移）。
- 不做臂片段合并与 `collect_model_dependencies` 的 profile 级编排（保留在
  旧 facade，A1 只保证其内部文件扫描恰好调用一次）。
- 导入期不做文件 IO、不加载 MuJoCo、不起线程、不 import
  `src.robot.model_factory` 或 `scripts/`。

## 稳定入口

```python
from src.robot.model_assets import (
    DEPENDENCY_ROLE_MESH,
    collect_model_file_dependencies,
    mask_xml_comments,
)
```

旧路径继续可用（对象相同）：

```python
from src.robot.model_factory import collect_model_file_dependencies
from src.robot.model_factory import mask_xml_comments
```

## 依赖方向

```text
references.py（纯 XML 解析，无包内依赖）
    ↑
dependencies.py（+ 哈希 / 仓库根 / 条目构造）
    ↑
__init__.py（轻量重导出）
    ↑
model_factory.py（仅旧路径 facade，不反向被包导入）
```

`_walk_model_file` 通过 keyword-only 参数 `build_entry` 注入条目构造器，
避免 `references → dependencies` 的反向 import。

## 依赖条目字段

- 公共字段：`role`、`relative_path`、`source`、`sha256`、`size`。
- `reference`：该文件在声明处的原始引用（`include`/`mesh`/`texture` 等）。
- `content_base64`：仅 XML 角色（`model`/`include`/`arm_fragment`）附带。
- `asset_dirs`：仅 `model` 条目附带，记录合并后的
  `{"mesh": meshdir, "texture": texturedir}`。

返回条目的排序口径仍为 `(role, relative_path)`；缺失路径单独以 `tuple[str, ...]`
返回，可选 `asset_missing` 列表只收集资产缺失（include 缺失不进入）。

## 测试

```bash
python -m pytest tests/test_model_asset_imports.py tests/test_episode_asset_meshdir.py \
  tests/test_episode_asset_closure.py tests/test_r1d_model.py -q
python -m ruff check src/robot/model_assets src/robot/model_factory.py \
  tests/test_model_asset_imports.py
```

## 最小用例

```python
from pathlib import Path

from src.robot.model_assets import collect_model_file_dependencies

entries, missing = collect_model_file_dependencies(
    Path("src/robot/rm65_model.xml").resolve()
)
print(len(entries), missing)  # 23 ()
```

## 后续

- A2：把 `collect_model_dependencies` / `_arm_fragment_path` 及模型加载、
  XML 组装实现迁入模型构建层；本包保持纯依赖解析，不承担装配职责。


## 何时读

改模型引用解析、依赖闭包、mesh 路径归一化或缺失资源报错时先读本文件；
只做模型装配/加载时转 `../model_building/README.md`。
