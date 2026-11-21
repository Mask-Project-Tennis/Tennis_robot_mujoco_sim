# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""V12 回合产物共享类型与 schema 常量（D1）。

本模块是 :class:`EpisodeArtifacts` / :class:`EpisodeReplayResult` 的**唯一
类定义**所在，同时集中定义产物 schema 版本、文件名、快照角色、依赖清单
角色等只读常量。其他 artifacts 子模块只从本模块导入，不重复定义。

设计约束：

* 不 import 任何 artifacts 内部模块（依赖图最底层）；
* 不 import MuJoCo、不做文件 IO；
* 字段顺序 / 类型 / ``frozen=True`` 与迁移前逐字一致。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)

EPISODE_SCHEMA = "v12_episode_v2"
ARTIFACT_JSON_NAME = "v12_episode.json"
ARTIFACT_NPZ_NAME = "v12_episode.npz"
SNAPSHOT_DIR_NAME = "snapshot"
SNAPSHOT_FILES_DIR_NAME = "files"
SNAPSHOT_MANIFEST_NAME = "manifest.json"
SNAPSHOT_CONFIG_NAME = "resolved_config.json"
SNAPSHOT_SCENE_NAME = "scene.json"
UNITS: dict[str, str] = {
    "position": "m",
    "linear_velocity": "m/s",
    "velocity": "m/s",
    "angle": "rad",
    "angular_velocity": "rad/s",
    "time": "s",
    "distance": "m",
    "control_torque": "N*m",
    "control_position": "rad",
}
SNAPSHOT_ROLES: tuple[str, ...] = (
    "robot_profile",
    "runtime",
    "task",
    "racket",
    "model",
)
INPUT_SHA_ROLES: tuple[str, ...] = (
    "robot_profile",
    "runtime",
    "task",
    "racket",
    "model",
    "assets",
    "scene_snapshot",
)
EOL_NORMALIZED_ROLES: tuple[str, ...] = (
    "robot_profile",
    "runtime",
    "task",
    "racket",
)
ASSET_XML_TAGS: tuple[str, ...] = ("mesh", "texture")
_ASSET_TAG_PATTERN = re.compile(
    r"<(mesh|texture)\b([^>]*?)/?>", re.IGNORECASE | re.DOTALL
)
_ASSET_FILE_ATTR_PATTERN = re.compile(
    r"""\bfile\s*=\s*(?:"([^"]*)"|'([^']*)')""", re.IGNORECASE
)
_COMPILER_OR_INCLUDE_PATTERN = re.compile(
    r"<(?P<tag>compiler|include)\b(?P<attrs>[^>]*?)/?>",
    re.IGNORECASE,
)
_ASSET_DIR_ATTRS: dict[str, str] = {
    "mesh": "meshdir",
    "texture": "texturedir",
}
DEPENDENCY_MANIFEST_VERSION = 1
DEPENDENCY_ROLE_MODEL = "model"
DEPENDENCY_ROLE_INCLUDE = "include"
DEPENDENCY_ROLE_ARM_FRAGMENT = "arm_fragment"
DEPENDENCY_ROLE_MESH = "mesh"
DEPENDENCY_ROLE_TEXTURE = "texture"
DEPENDENCY_XML_ROLES = frozenset(
    {
        DEPENDENCY_ROLE_MODEL,
        DEPENDENCY_ROLE_INCLUDE,
        DEPENDENCY_ROLE_ARM_FRAGMENT,
    }
)
DEPENDENCY_ASSET_ROLES = frozenset(
    {DEPENDENCY_ROLE_MESH, DEPENDENCY_ROLE_TEXTURE}
)
_DEPENDENCY_COPY_DIRNAME = "assets"
_INCLUDE_TAG_PATTERN = re.compile(r"<include\b[^>]*?/?>", re.IGNORECASE)
_MAX_INCLUDE_DEPTH = 8
REQUIRE_RESULT_DEPENDENCY_MANIFEST: bool = True
_CONFIG_PATH_FIELDS = (
    "robot_profile_path",
    "runtime_path",
    "racket_path",
    "task_path",
    "output_dir",
)
_CONFIG_ARRAY_FIELDS = ("target_center", "kp", "kd")
@dataclass(frozen=True)
class EpisodeArtifacts:
    """已落盘的 v2 产物（payload + 数组 + 目录）。"""

    payload: dict[str, Any]
    arrays: dict[str, np.ndarray]
    directory: Path

    #: 便捷访问（与 NPZ 键一一对应）
    @property
    def X(self) -> np.ndarray:
        return self.arrays["X"]

    @property
    def U(self) -> np.ndarray:
        return self.arrays["U"]

    @property
    def ball_pos(self) -> np.ndarray:
        return self.arrays["ball_pos"]

    @property
    def ball_vel(self) -> np.ndarray:
        return self.arrays["ball_vel"]

    @property
    def time(self) -> np.ndarray:
        return self.arrays["time"]

    @property
    def racket_pos(self) -> np.ndarray:
        return self.arrays["racket_pos"]
@dataclass(frozen=True)
class EpisodeReplayResult:
    """回放结果（按名重建状态；不修改产物文件）。"""

    payload: dict[str, Any]
    ball_positions: np.ndarray
    racket_positions: np.ndarray
    speed: float
    viewer_opened: bool
    directory: Path
