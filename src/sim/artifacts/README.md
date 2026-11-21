# `src/sim/artifacts` — V12 回合产物子包（D1）

本包由原 2804 行单文件 `src/sim/episode_artifacts.py` 机械拆分而来。
拆分只移动定义、不改变行为：函数/类实现逐字迁移，旧路径
`src.sim.episode_artifacts` 保留为纯 facade（同名重导出，`is` 相同、
`__module__` 指向新模块）。

## 模块边界与依赖方向

```
types  ←  serialization  ←  dependencies  ←  snapshot
                     ↑            ↑
                validation  ←  materialization  ←  replay
```

依赖是单向的：下层不得 import 上层，任何模块都**禁止**反向 import
`src.sim.episode_artifacts`（旧 facade）或 `src.sim.episode.execution`。

| 模块 | 职责 | 主要符号 |
|------|------|----------|
| `types.py` | 产物类型与 schema/role 常量（唯一类定义处） | `EpisodeArtifacts`、`EpisodeReplayResult`、`EPISODE_SCHEMA`、`ARTIFACT_*`、`SNAPSHOT_*`、`DEPENDENCY_*` |
| `serialization.py` | JSON/哈希/配置往返/角色路径/payload 构造 | `_repo_root`、`_sha256_*`、`_jsonable`、`_canonical_json`、`config_to_dict`、`config_from_dict`、`_role_paths`、`_assets_digest`、`_arrays_from_result`、`_scene_snapshot`、`_hit_section`、`build_episode_payload`、`_read_json` |
| `dependencies.py` | 模型资产/XML 依赖闭包扫描、条目归一化、暂存与复制 | `_model_asset_entries`、`_normalize_dependency_entries`、`_build_dependency_entries`、`_write_dependency_copies`、`resolve_verified_assets`、`_DependencyStager`、`_is_ascii_text`、`_loadable_asset_path` |
| `snapshot.py` | 产物写盘与快照元数据 | `write_episode_artifacts`、`_write_snapshot`、`_write_snapshot_metadata` |
| `validation.py` | 读取、来源优先级、模型/资产哈希校验 | `load_episode_artifacts`、`_snapshot_entries`、`_verify_model_assets`、`_verify_and_resolve_sources`、`_verify_loaded_model`、`_resolved_model_path` |
| `materialization.py` | 快照模型物化（跨盘/非 ASCII 路径）与回放 bundle 组装 | `_materialize_profile`、`_materialize_snapshot_model`、`_prepare_replay_config`、`_build_replay_bundle_with_snapshot`、`_model_origin_dir`、`_MATERIALIZED_MODEL_PREFIX` |
| `replay.py` | 公开回放入口与 viewer 可视化 | `build_replay_bundle`、`replay_episode_artifacts`、`_replay_frames`、`_visualize_replay` |

## 关键契约与实现说明

1. **仓库根解析**：`serialization._repo_root()` 使用
   `Path(__file__).resolve().parents[3]`（文件从 `sim/` 下移一层到
   `sim/artifacts/`，层级必须加一）。全包统一复用它，禁止按 CWD 猜根。
2. **循环依赖**：`serialization.build_episode_payload` 需要
   `dependencies._assets_section`，而 `dependencies` 反向依赖
   `serialization` 的哈希/角色路径 helper；因此在函数内延迟导入
   `_assets_section`，模块级依赖方向保持单向。
3. **类型导入红线**：`V12EpisodeResult` 只从 `src.sim.episode.types`
   以 `TYPE_CHECKING` 或函数内延迟导入方式获取；`V12RobotBundle` 同理
   （`TYPE_CHECKING`）。不得 import `episode.execution`。
4. **brief 归属偏差**：为避免环，`_resolved_model_path` 定义在
   `validation.py`（`_verify_loaded_model` 使用）并由 `materialization.py`
   以 `as` 重导出；`_is_ascii_text` / `_loadable_asset_path` 定义在
   `dependencies.py`（`_DependencyStager` 使用）并由 `validation.py`
   重导出。两者均保持 `is` 相同，模块边界说明见 brief 时以本文件为准。
5. **`_DependencyStager` 分阶段迁移**：未重写算法，按原 `__init__` →
   `stage_model` / `stage_sibling_fragments`（暂存）→ `_asset_entry` /
   `_stage_xml_entry`（清单与重写）→ `_materialize_snapshot_model` 的
   `try/except BaseException` 与 `weakref.finalize`（清理）逐段对照。
6. **清理保证**：物化失败时 `_materialize_snapshot_model` 删除临时目录后
   重新抛异常；成功时通过 `weakref.finalize(bundle, cleanup)` 在 bundle
   释放时清理。
7. **facade 与 `__all__`**：旧模块原本没有 `__all__`，facade 也不定义；
   所有旧可见名（含测试直接使用的私有名）均显式 `as` 重导出。
   `src.sim.artifacts.__init__` 只暴露稳定入口，不做 MuJoCo 等重型 import。
8. **可变属性 patch 兼容垫片**：既有测试会
   `monkeypatch.setattr(artifacts, "resolve_verified_assets", ...)` 或替换
   `REQUIRE_RESULT_DEPENDENCY_MANIFEST` 后调用 facade 入口。拆分后实现读取
   各子模块的模块全局，因此 facade 末尾安装一个 `ModuleType` 子类，把对
   facade 的属性写入同步到定义该名字的子模块（teardown 恢复原值同样同步），
   保持旧 patch 语义。该垫片只做属性转发、不承载逻辑；同时 facade 保留
   `tempfile` 别名（既有测试通过 `artifacts.tempfile` 打桩 `mkdtemp`）。
9. **ruff 说明**：生成文件顶层声明 `# ruff: noqa: I001, PLC0414, UP035`
   —— 契约 §6 要求 `import X as X` 显式重导出，isort 会将其拆成每名一条
   或建议删除别名；`typing` 导入保持原文件风格。另有 3 处拆分前旧代码
   既有的 `BLE001`/`TRY004`/`SIM102` 告警，行为未变，按文件定点抑制
   （详见 `reports/D1.md`）。

## 新增代码放哪里

- 新的产物字段/schema 常量 → `types.py`；
- 新的序列化/哈希逻辑 → `serialization.py`；
- 依赖扫描或暂存行为 → `dependencies.py`；
- 写盘/读盘/物化/回放行为 → 对应阶段模块；
- **不要**把逻辑写回 `src/sim/episode_artifacts.py`。


## 何时读

改 `v12_episode_v2` 产物字段、依赖快照、模型/配置哈希、物化目录或回放时；
只读写已生成产物时用下方稳定入口，不必进入 `dependencies.py`/`materialization.py`。

## 稳定入口

```python
from src.sim.artifacts import (
    EpisodeArtifacts,
    build_replay_bundle,
    load_episode_artifacts,
    replay_episode_artifacts,
    write_episode_artifacts,
)
```

## 允许依赖

`artifacts` 只依赖自身子模块、标准库/numpy/pyyaml、`src.utils.mujoco_loader`，以及
`src.robot.model_assets` / `src.robot.model_building` / `src.robot.robot_profile`
的只读模型闭包与配置 helper（延迟导入）；禁止 import 旧 facade
`src.sim.episode_artifacts`、`src.robot.model_factory` 或 `src.sim.episode.execution`；
MuJoCo 只在回放函数内延迟导入。

## 测试

```bash
python -m pytest tests/test_episode_artifacts.py tests/test_artifact_import_contracts.py -q
```

## 最小用例

```python
from pathlib import Path
from src.sim.artifacts import write_episode_artifacts, load_episode_artifacts

# result 来自 run_v12_episode
out = Path("results/example_episode")
write_episode_artifacts(result, out)
artifacts = load_episode_artifacts(out)
print(artifacts.payload["schema_version"], sorted(artifacts.arrays)[:3])
```
