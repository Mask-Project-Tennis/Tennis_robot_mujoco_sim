# ruff: noqa: I001, PLC0414
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留分组风格。
"""V12 回合产物（v2 schema）写入、读取、快照与回放 —— 兼容 facade。

D1 拆分后实现位于 :mod:`src.sim.artifacts` 子包：

- :mod:`src.sim.artifacts.types`：产物类型与 schema 常量；
- :mod:`src.sim.artifacts.serialization`：JSON/哈希/配置往返/payload；
- :mod:`src.sim.artifacts.dependencies`：资产扫描、依赖清单与暂存；
- :mod:`src.sim.artifacts.snapshot`：产物写入与快照元数据；
- :mod:`src.sim.artifacts.validation`：读取、来源优先级与哈希校验；
- :mod:`src.sim.artifacts.materialization`：模型快照物化；
- :mod:`src.sim.artifacts.replay`：bundle 构建与回放。

本模块只做同名重导出（``is`` 相同、``__module__`` 指向新模块），
不得在此重新实现任何逻辑；旧调用方与测试继续从本路径导入。
"""

from __future__ import annotations

import logging
import tempfile as tempfile

from src.sim.artifacts.types import (
    ARTIFACT_JSON_NAME as ARTIFACT_JSON_NAME,
    ARTIFACT_NPZ_NAME as ARTIFACT_NPZ_NAME,
    ASSET_XML_TAGS as ASSET_XML_TAGS,
    DEPENDENCY_ASSET_ROLES as DEPENDENCY_ASSET_ROLES,
    DEPENDENCY_MANIFEST_VERSION as DEPENDENCY_MANIFEST_VERSION,
    DEPENDENCY_ROLE_ARM_FRAGMENT as DEPENDENCY_ROLE_ARM_FRAGMENT,
    DEPENDENCY_ROLE_INCLUDE as DEPENDENCY_ROLE_INCLUDE,
    DEPENDENCY_ROLE_MESH as DEPENDENCY_ROLE_MESH,
    DEPENDENCY_ROLE_MODEL as DEPENDENCY_ROLE_MODEL,
    DEPENDENCY_ROLE_TEXTURE as DEPENDENCY_ROLE_TEXTURE,
    DEPENDENCY_XML_ROLES as DEPENDENCY_XML_ROLES,
    EOL_NORMALIZED_ROLES as EOL_NORMALIZED_ROLES,
    EPISODE_SCHEMA as EPISODE_SCHEMA,
    EpisodeArtifacts as EpisodeArtifacts,
    EpisodeReplayResult as EpisodeReplayResult,
    INPUT_SHA_ROLES as INPUT_SHA_ROLES,
    REQUIRE_RESULT_DEPENDENCY_MANIFEST as REQUIRE_RESULT_DEPENDENCY_MANIFEST,
    SNAPSHOT_CONFIG_NAME as SNAPSHOT_CONFIG_NAME,
    SNAPSHOT_DIR_NAME as SNAPSHOT_DIR_NAME,
    SNAPSHOT_FILES_DIR_NAME as SNAPSHOT_FILES_DIR_NAME,
    SNAPSHOT_MANIFEST_NAME as SNAPSHOT_MANIFEST_NAME,
    SNAPSHOT_ROLES as SNAPSHOT_ROLES,
    SNAPSHOT_SCENE_NAME as SNAPSHOT_SCENE_NAME,
    UNITS as UNITS,
    _ASSET_DIR_ATTRS as _ASSET_DIR_ATTRS,
    _ASSET_FILE_ATTR_PATTERN as _ASSET_FILE_ATTR_PATTERN,
    _ASSET_TAG_PATTERN as _ASSET_TAG_PATTERN,
    _COMPILER_OR_INCLUDE_PATTERN as _COMPILER_OR_INCLUDE_PATTERN,
    _CONFIG_ARRAY_FIELDS as _CONFIG_ARRAY_FIELDS,
    _CONFIG_PATH_FIELDS as _CONFIG_PATH_FIELDS,
    _DEPENDENCY_COPY_DIRNAME as _DEPENDENCY_COPY_DIRNAME,
    _INCLUDE_TAG_PATTERN as _INCLUDE_TAG_PATTERN,
    _MAX_INCLUDE_DEPTH as _MAX_INCLUDE_DEPTH,
)

from src.sim.artifacts.serialization import (
    _ROBOT_DISPLAY_NAMES as _ROBOT_DISPLAY_NAMES,
    _arrays_from_result as _arrays_from_result,
    _assets_digest as _assets_digest,
    _canonical_json as _canonical_json,
    _hit_section as _hit_section,
    _is_eol_normalized_role as _is_eol_normalized_role,
    _jsonable as _jsonable,
    _model_path_for_profile as _model_path_for_profile,
    _profile_path_for_config as _profile_path_for_config,
    _read_json as _read_json,
    _read_normalized_text as _read_normalized_text,
    _repo_root as _repo_root,
    _robot_display_name as _robot_display_name,
    _role_file_digest as _role_file_digest,
    _role_paths as _role_paths,
    _role_paths_from_payload as _role_paths_from_payload,
    _scene_snapshot as _scene_snapshot,
    _sha256_bytes as _sha256_bytes,
    _sha256_file as _sha256_file,
    build_episode_payload as build_episode_payload,
    config_from_dict as config_from_dict,
    config_to_dict as config_to_dict,
)

from src.sim.artifacts.dependencies import (
    _DependencyStager as _DependencyStager,
    _asset_id as _asset_id,
    _assets_section as _assets_section,
    _build_dependency_entries as _build_dependency_entries,
    _collect_verified_assets as _collect_verified_assets,
    _copy_verified as _copy_verified,
    _dependency_entries as _dependency_entries,
    _dependency_section as _dependency_section,
    _derive_dependency_entries as _derive_dependency_entries,
    _is_ascii_text as _is_ascii_text,
    _loadable_asset_path as _loadable_asset_path,
    _masked_xml as _masked_xml,
    _model_asset_entries as _model_asset_entries,
    _normalize_dependency_entries as _normalize_dependency_entries,
    _replace_file_attribute as _replace_file_attribute,
    _resolve_xml_reference as _resolve_xml_reference,
    _result_dependency_manifest as _result_dependency_manifest,
    _runtime_arm_fragment as _runtime_arm_fragment,
    _snapshot_relative_path as _snapshot_relative_path,
    _snapshot_root_for as _snapshot_root_for,
    _tag_attribute as _tag_attribute,
    _write_dependency_copies as _write_dependency_copies,
    resolve_verified_assets as resolve_verified_assets,
)

from src.sim.artifacts.snapshot import (
    _write_snapshot as _write_snapshot,
    _write_snapshot_metadata as _write_snapshot_metadata,
    write_episode_artifacts as write_episode_artifacts,
)

from src.sim.artifacts.validation import (
    _external_source_for_role as _external_source_for_role,
    _resolved_model_path as _resolved_model_path,
    _snapshot_entries as _snapshot_entries,
    _verify_and_resolve_sources as _verify_and_resolve_sources,
    _verify_loaded_model as _verify_loaded_model,
    _verify_model_assets as _verify_model_assets,
    load_episode_artifacts as load_episode_artifacts,
)

from src.sim.artifacts.materialization import (
    _MATERIALIZED_MODEL_PREFIX as _MATERIALIZED_MODEL_PREFIX,
    _build_replay_bundle_with_snapshot as _build_replay_bundle_with_snapshot,
    _materialize_profile as _materialize_profile,
    _materialize_snapshot_model as _materialize_snapshot_model,
    _model_origin_dir as _model_origin_dir,
    _prepare_replay_config as _prepare_replay_config,
    _replay_config as _replay_config,
)

from src.sim.artifacts.replay import (
    _replay_frames as _replay_frames,
    _visualize_replay as _visualize_replay,
    build_replay_bundle as build_replay_bundle,
    replay_episode_artifacts as replay_episode_artifacts,
)

_LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 兼容垫片：保持旧 facade 的“可变属性 patch”语义
# ---------------------------------------------------------------------------
# 拆分前测试会用 monkeypatch 替换本模块的模块级属性，例如：
#   monkeypatch.setattr(artifacts, "resolve_verified_assets", sneaky)
#   monkeypatch.setattr(artifacts, "REQUIRE_RESULT_DEPENDENCY_MANIFEST", False)
# 然后调用本模块的公开入口。拆分后实现读取各子模块自己的模块全局，因此这里
# 把对 facade 的属性写入同步到定义该名字的子模块，使旧 patch 语义继续生效；
# teardown 恢复原值时同样会同步回去。垫片只做属性转发，不承载业务逻辑。
import sys as _sys
import types as _module_types

from src.sim.artifacts import dependencies as _dependencies_module
from src.sim.artifacts import materialization as _materialization_module
from src.sim.artifacts import replay as _replay_module
from src.sim.artifacts import serialization as _serialization_module
from src.sim.artifacts import snapshot as _snapshot_module
from src.sim.artifacts import types as _types_module
from src.sim.artifacts import validation as _validation_module

_ARTIFACT_MODULES = (
    _types_module,
    _serialization_module,
    _dependencies_module,
    _snapshot_module,
    _validation_module,
    _materialization_module,
    _replay_module,
)


class _FacadeModule(_module_types.ModuleType):
    """旧 facade 模块类：属性写入同时同步到新子模块。"""

    def __setattr__(self, name: str, value: object) -> None:
        super().__setattr__(name, value)
        for module in _ARTIFACT_MODULES:
            if name in vars(module):
                setattr(module, name, value)


_sys.modules[__name__].__class__ = _FacadeModule
