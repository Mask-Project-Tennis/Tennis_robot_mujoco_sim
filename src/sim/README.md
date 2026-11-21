# src/sim — 仿真环境、V12 回合与产物

## 何时读

- 改 MuJoCo 仿真环境、V12 配置解析、回合编排、产物 schema/回放时。
- 需要从配置到 episode 结果、再到 JSON/NPZ 与 replay 的完整链路时。
- 排查 R1D/RM65 双机器人地址、位置/力矩模式或固定案例回归时。

## 职责

| 路径 | 职责 |
|------|------|
| `configuration/` | V12 配置：类型、校验、选择、默认值、覆盖、解析 |
| `episode/` | V12 闭环阶段函数：preparation / assembly / hooks / post_hit / results / execution |
| `artifacts/` | v2 产物：序列化、依赖快照、校验、物化、回放 |
| `v12_config.py` | 配置解析兼容 facade |
| `v12_episode.py` | `run_v12_episode` 兼容 facade |
| `episode_artifacts.py` | 产物/回放兼容 facade |
| `v12_robot_adapter.py` | `build_v12_robot` / `V12RobotBundle` 双机器人适配（主代理维护） |
| `r1d_runner.py` | 旧 R1D 单臂闭环入口（兼容） |
| `env.py`、`rm65_env.py` | MuJoCo 环境基类与 RM65 双臂环境 |
| `r1d_*`、`profile_*`、`env_factory.py` | R1D 七维执行/规划环境 |
| `replay.py`、`hit_detection.py` | 轨迹回放与击打判定 |

## 稳定入口

```python
from src.sim.configuration import resolve_v12_config
from src.sim.episode import V12EpisodeResult
from src.sim.v12_episode import run_v12_episode
from src.sim.artifacts import (
    EpisodeArtifacts,
    build_replay_bundle,
    load_episode_artifacts,
    replay_episode_artifacts,
    write_episode_artifacts,
)
from src.sim.v12_robot_adapter import build_v12_robot
```

- `resolve_v12_config(args=None, explicit_options=None) -> ResolvedV12Config`
- `run_v12_episode(config, ...) -> V12EpisodeResult`
- `write_episode_artifacts(result, output_dir) -> dict[str, Any]`
- `load_episode_artifacts(output_dir) -> EpisodeArtifacts`
- `build_replay_bundle(output_dir) -> V12RobotBundle`
- `replay_episode_artifacts(output_dir, *, speed=1.0, viewer=True, bundle=None) -> EpisodeReplayResult`
- 旧路径 `src.sim.v12_config` / `v12_episode` / `episode_artifacts` 与新包导出同一对象。

## 允许依赖

- `configuration` 是叶子：只依赖标准库、numpy、`src.robot` 的 profile/参数类型；不导入 `episode` / `artifacts`。
- `episode` 可依赖 `configuration`、`v12_robot_adapter`、`src.ilqt`、`src.robot`；不反向依赖 `artifacts`。
- `artifacts` 只依赖自身子模块与通用工具；禁止 import 旧 `episode_artifacts` facade 或 `episode.execution`。
- `src.sim` 不得 import `scripts`；import 期不启动仿真、不写文件、不起线程。

## 测试

```bash
python -m pytest \
  tests/test_v12_config.py \
  tests/test_configuration_migration.py \
  tests/test_v12_shared_pipeline.py \
  tests/test_episode_decomposition.py \
  tests/test_v12_runtime_restore.py \
  tests/test_episode_artifacts.py \
  tests/test_artifact_import_contracts.py \
  tests/test_r1d_v12_acceptance.py -q
```

## 最小用例

```python
from pathlib import Path
from src.sim.configuration import resolve_v12_config
from src.sim.v12_episode import run_v12_episode
from src.sim.artifacts import write_episode_artifacts, load_episode_artifacts

config = resolve_v12_config()
result = run_v12_episode(config)
out = Path("results/example_episode")
write_episode_artifacts(result, out)
artifacts = load_episode_artifacts(out)
print(artifacts.payload["schema_version"])
```

更多细节见 `configuration/README.md`、`episode/README.md`、`artifacts/README.md`。
