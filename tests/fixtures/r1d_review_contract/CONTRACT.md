# R0 冻结契约（测试夹具，禁止当作真实实验数据）

本目录由协调者在 R0 写入，供 A/B/C/D/I1 各工作线共用同一套**合成**接口样例。
它不是实验产物，不得放入 `experiment_data/`，也不得被统计脚本当成真实数据引用。
凡是由本目录派生的统计结果，必须在输出 JSON 顶层标记 `"synthetic": true`。

## 1. 有效运行覆盖（A 线，`src/sim/episode_runtime.py`）

```python
resolve_runtime_options(
    config: ResolvedV12Config, explicit: Mapping[str, Any]
) -> ResolvedV12Config
apply_runtime_limits(
    config: ResolvedV12Config, bundle: V12RobotBundle
) -> ResolvedV12Config
```

- `explicit` 只包含调用者真正传入的键（`run_v12_episode` 内部用 `UNSET` 哨兵）。
- 优先级：显式值 > `config.runtime_overrides` 已保存值 > 旧默认。
- 显式 `False` / `0` 是覆盖；可空字段显式 `None` 表示清除覆盖并回到旧默认。
- 允许清除的字段：`hit_direction`、`limits_config`、`max_tcp`、
  `terminal_exempt_steps`、`dq_max_fraction`。
- 保存的 `limits_config` 快照消费内容与 SHA256，不依赖原绝对文件仍存在；
  显式新文件覆盖时不得保留旧 effective 值。
- `apply_runtime_limits` 同时更新 `bundle.robot_limits` 并返回带实际 effective 值的
  新配置；执行、结果、快照使用同一返回值；replay 也必须调用。

## 2. 已验证资源与依赖闭包（B 线）

```python
resolve_verified_assets(
    snapshot: Mapping[str, Any], project_root: Path
) -> dict[str, Path]
```

- key 是 snapshot 内稳定逻辑 ID（`asset_id`），不是某次绝对路径。
- 只返回内容哈希匹配的资源；缺失/损坏按逻辑 ID 报错。
- 物化必须消费同一映射，复制后再验字节，不得重新挑选旧 `source`。
- 快照新增 `dependency_manifest_version: 1`；条目字段：
  `asset_id`、`role`、`relative_path`、`source`、`sha256`，
  XML 片段另含 `content_base64`。
- `RobotModelBundle.dependency_manifest: tuple[dict[str, Any], ...] = ()`；
  `V12EpisodeResult.model_dependency_manifest: list[dict[str, Any]] = []`。
- 新 R1D 生产结果缺清单报错；旧 artifact 无该字段走显式 legacy 分支并警告。

## 3. 任务身份、catalog、日志（C 线）

- `ExperimentSpec` 参数可携带保留元数据 `_config_id`（manifest 稳定 ID）。
  所有 `_` 前缀元数据不进入 CLI、不参与配置内容摘要。
- 旧调用无 `_config_id` 时继续用 `make_config_id(params)`。
- `batch_runner.task_log_path(log_dir, experiment, config_id, seed) -> Path`：
  Windows 合法、总长 ≤ 96 字符、包含稳定摘要；manifest 计划路径与实际写日志
  必须调用同一函数。

`catalog.sample.json` 是 catalog schema 样例（`schema_version: 1`）：

- `catalog_path` 在加载后补入，JSON 中不写。
- 其余路径相对 catalog 文件解析，也接受绝对路径。
- `path_base=run_root`：`runs.csv` 中 `artifact_dir` / `log_path` 的相对路径
  以 `run_root` 为基准。
- `source_sets.nominal_mechanism`：R1D 为 `exp17d_mechanism`；
  旧 RM65 无 catalog 时保持旧逻辑。

`run_catalog.resolve_catalog_path(catalog, value)` 按上述基准解析。
显式 catalog 不存在、声明文件冲突、身份不匹配必须非零退出。

## 4. 分模式统计（D 线）

- 两个提取入口新增 `--catalog PATH`、`--mode torque|position|both`。
- 单模式保持既有 JSON 内部结构；混合模式未指定 `--mode` 必须报错。
- `paired_stats.py --mode both --output stats` 写：
  `stats/torque/{06-stats-active.json,paper_stats_active.json}`、
  `stats/position/...`，以及仅作索引的 `stats/modes.json`
  （`schema: paired_stats_multi_mode_v1`）。
- per-case 提取写 `stats/<mode>/<experiment>/<case>.json`。
- 唯一记录键：`experiment/case/config_id/seed/mode/robot_id/racket_id/来源身份`；
  同来源重复直接报错。
- `runs.csv`（富化视图）与 `results.csv`（基础视图）不得纵向拼接计数；
  各自内部唯一后按完整身份一对一连接。
- 全 miss 的 E8：保留观察差值/样本数，推断字段为 `null`，
  `unavailable_reason="all_miss"`；`paper_figs.py --hit-semantics legacy|physical`
  默认 `legacy`，空值渲染 `--` 并保留原因。

## 5. 文件清单

| 文件 | 用途 |
|---|---|
| `catalog.sample.json` | catalog v1 形状样例（相对路径，path_base=run_root） |
| `tasks.sample.json` | `manifest_resolved.json` 的 4 条任务投影（2 case × 2 mode × 1 seed） |
| `runs.sample.csv` | `runs.csv` 表头 + 4 行合成记录（不含真实 artifact） |

`runs.sample.csv` 只固定列形状与路径基准；需要真实 artifact 的测试必须在
`tmp_path` 下自行写最小 `v12_episode.json`，并设置 `"synthetic": true`。
