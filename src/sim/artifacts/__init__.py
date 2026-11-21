# ruff: noqa: I001
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留分组风格。
"""V12 产物/回放子包（D1）。

稳定入口：写入、读取、构建回放 bundle、回放；类型与配置序列化兼容
``src.sim.episode_artifacts`` facade。默认不做重型 import，MuJoCo 只在
回放函数内延迟导入。
"""

from src.sim.artifacts.replay import (
    build_replay_bundle as build_replay_bundle,
    replay_episode_artifacts as replay_episode_artifacts,
)
from src.sim.artifacts.serialization import (
    config_from_dict as config_from_dict,
    config_to_dict as config_to_dict,
)
from src.sim.artifacts.snapshot import (
    write_episode_artifacts as write_episode_artifacts,
)
from src.sim.artifacts.types import (
    ARTIFACT_JSON_NAME as ARTIFACT_JSON_NAME,
    ARTIFACT_NPZ_NAME as ARTIFACT_NPZ_NAME,
    EPISODE_SCHEMA as EPISODE_SCHEMA,
    SNAPSHOT_CONFIG_NAME as SNAPSHOT_CONFIG_NAME,
    EpisodeArtifacts as EpisodeArtifacts,
    EpisodeReplayResult as EpisodeReplayResult,
)
from src.sim.artifacts.validation import (
    load_episode_artifacts as load_episode_artifacts,
)
