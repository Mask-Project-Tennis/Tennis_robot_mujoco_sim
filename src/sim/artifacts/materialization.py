# ruff: noqa: I001, PLC0414, UP035
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""模型快照物化与回放配置准备（D1）。

把校验通过的快照模型物化为临时 XML + 临时 profile（Windows 非 ASCII
路径映射到 ASCII 副本），临时目录生命周期绑定 bundle（``weakref.finalize``
兜底）。依赖图位置：``... ← validation ← materialization``。
"""

from __future__ import annotations

import logging
import hashlib
import re
import shutil
import tempfile
import weakref
from pathlib import Path
from typing import Any, Callable, Mapping, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from src.sim.v12_robot_adapter import V12RobotBundle

from src.sim.artifacts.types import (
    DEPENDENCY_MANIFEST_VERSION as DEPENDENCY_MANIFEST_VERSION,
    DEPENDENCY_ROLE_MODEL as DEPENDENCY_ROLE_MODEL,
    EpisodeArtifacts as EpisodeArtifacts,
    SNAPSHOT_DIR_NAME as SNAPSHOT_DIR_NAME,
    SNAPSHOT_MANIFEST_NAME as SNAPSHOT_MANIFEST_NAME,
)
from src.sim.artifacts.serialization import (
    _read_json as _read_json,
    _repo_root as _repo_root,
    config_from_dict as config_from_dict,
)
from src.sim.artifacts.dependencies import (
    _DependencyStager as _DependencyStager,
    _dependency_entries as _dependency_entries,
    _snapshot_root_for as _snapshot_root_for,
    resolve_verified_assets as resolve_verified_assets,
)
from src.sim.artifacts.validation import (
    _external_source_for_role as _external_source_for_role,
    _resolved_model_path as _resolved_model_path,
)

_LOGGER = logging.getLogger(__name__)

_MATERIALIZED_MODEL_PREFIX = "v12_replay_model_"
def _model_origin_dir(payload: Mapping[str, Any], model_path: Path) -> Path:
    """返回模型 XML 原始位置目录（解析相对 mesh/texture 路径的基准）。

    优先读取快照 manifest 中 ``model`` 条目的 ``source``（产物写入时记录
    的原始绝对路径）；其次回退 payload config 解析出的外部模型父目录；
    最后回退待物化模型自身所在目录。
    """
    for parent in Path(model_path).resolve().parents:
        manifest_path = parent / SNAPSHOT_MANIFEST_NAME
        if not manifest_path.is_file():
            continue
        try:
            manifest = _read_json(manifest_path)
        except (OSError, ValueError):  # pragma: no cover - 破损 manifest
            break
        for item in manifest.get("files") or []:
            if not isinstance(item, Mapping) or item.get("role") != "model":
                continue
            source = item.get("source")
            if source:
                return Path(str(source)).parent
        break
    external = _external_source_for_role(payload, "model")
    if external is not None:
        external_path = Path(external)
        if not external_path.is_absolute():
            external_path = _repo_root() / external_path
        return external_path.parent
    return Path(model_path).resolve().parent
def _materialize_profile(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    staging_dir: Path,
    model_xml: Path,
) -> Path:
    """以原 robot profile 文本为模板，仅改 ``model_path`` 指向物化 XML。"""
    profile_path = sources.get("robot_profile")
    if profile_path is None:
        profile_path = _external_source_for_role(payload, "robot_profile")
    if profile_path is None:
        raise ValueError("无法物化模型快照：产物缺少 robot profile 来源")
    try:
        profile_text = Path(profile_path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(
            f"读取 robot profile 失败，无法物化模型：{profile_path}（{exc}）"
        ) from exc
    materialized, count = re.subn(
        r"(?m)^(\s*model_path\s*:\s*).*$",
        lambda match: f"{match.group(1)}'{model_xml.as_posix()}'",
        profile_text,
        count=1,
    )
    if count == 0:
        raise ValueError(
            f"robot profile 缺少 model_path 字段，无法物化模型：{profile_path}"
        )
    temp_profile = staging_dir / "profile.yaml"
    temp_profile.write_text(materialized, encoding="utf-8", newline="\n")
    return temp_profile
def _materialize_snapshot_model(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    model_path: Path,
    *,
    output_dir: Path | None = None,
) -> tuple[Path, Callable[[], None]]:
    """把校验通过的模型快照物化为可加载的临时 XML + 临时 profile。

    快照里 mesh/texture 的路径仍相对原仓库（如 ``../../assets/...``），
    把快照 XML 直接交给 MuJoCo 会解析到快照目录内不存在的资产。这里读入
    **已校验**的快照字节（再次用 ``input_sha256["model"]`` 比对，保证
    "校验的文件 = 消费的文件"），用依赖闭包清单解析出的**同一份**已验证
    路径映射改写每个 ``<mesh>`` / ``<texture>`` 的 ``file``（F02：不再按
    记录来源重新挑选旧文件）；``<include>`` 递归物化；兄弟臂片段写到临时
    模型同目录，使工厂 ``_arm_fragment_path`` 优先命中快照内容（F08）。
    Windows 非 ASCII 路径映射到 ASCII 副本。

    Args:
        payload: v2 产物 payload（提供模型哈希与依赖清单）。
        sources: :func:`_verify_and_resolve_sources` 的校验结果。
        model_path: 将被加载的模型文件（快照或外部回退）。
        output_dir: 产物目录（用于定位快照副本；缺省时从 ``model_path``
            向上查找 ``snapshot/``，仍找不到则只用仓库/记录来源候选）。

    Returns:
        ``(temp_profile_path, cleanup)``；``cleanup`` 为无参可调用对象，
        删除本次物化的临时目录（构建成功/失败都必须调用）。

    Raises:
        ValueError: 模型字节哈希与记录不一致、XML 非 UTF-8、依赖缺失/
            哈希不一致、``<include>`` 未被清单记录，或 profile 无
            ``model_path``。
    """
    resolved_model = Path(model_path).resolve()
    try:
        raw = resolved_model.read_bytes()
    except OSError as exc:
        raise ValueError(f"读取待物化模型失败：{resolved_model}（{exc}）") from exc
    expected = str(payload.get("input_sha256", {}).get("model", "") or "")
    actual = hashlib.sha256(raw).hexdigest()
    if expected and actual != expected:
        raise ValueError(
            "回放模型文件哈希不一致，拒绝物化："
            f"{resolved_model}（期望 {expected}，实际 {actual}）"
        )
    try:
        model_text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"模型快照不是 UTF-8 XML，无法物化：{resolved_model}"
        ) from exc

    entries, legacy = _dependency_entries(payload)
    if not entries:
        raise ValueError(
            "产物缺少装配依赖条目（既无 dependencies 也无 assets.files），"
            f"无法证明装配闭包，拒绝物化：{resolved_model}"
        )
    snapshot_dir = _snapshot_root_for(resolved_model)
    if snapshot_dir is None:
        snapshot_dir = _snapshot_root_for(sources.get("model"))
    if snapshot_dir is None and output_dir is not None:
        candidate_dir = Path(output_dir) / SNAPSHOT_DIR_NAME
        if (candidate_dir / SNAPSHOT_MANIFEST_NAME).is_file():
            snapshot_dir = candidate_dir
    view: dict[str, Any] = {
        "dependencies": {"version": DEPENDENCY_MANIFEST_VERSION, "files": entries},
        "snapshot_dir": str(snapshot_dir) if snapshot_dir is not None else "",
    }
    # F02：校验与消费共用这一份映射；任何缺失/哈希不符都会带 asset_id 抛错
    resolved = resolve_verified_assets(view, _repo_root())

    model_entry = next(
        (
            entry
            for entry in entries
            if str(entry.get("role")) == DEPENDENCY_ROLE_MODEL
            and str(entry.get("sha256")) == (expected or actual)
        ),
        None,
    )
    if model_entry is not None and model_entry.get("source"):
        logical_base = Path(str(model_entry["source"]))
    else:
        logical_base = _model_origin_dir(payload, resolved_model) / resolved_model.name
        if not legacy:
            _LOGGER.warning(
                "依赖清单的模型条目与产物模型哈希不一致（快照模型可能被改动），"
                "按写入时记录的模型位置解析相对引用：%s",
                resolved_model,
            )
    if legacy:
        _LOGGER.warning(
            "旧 artifact 缺少装配依赖清单（legacy 分支）：仅校验角色文件与网格哈希，"
            "不保证装配闭包（兄弟片段可能回落到仓库同名文件）：%s",
            output_dir if output_dir is not None else resolved_model,
        )

    staging_dir = Path(tempfile.mkdtemp(prefix=_MATERIALIZED_MODEL_PREFIX))

    def cleanup() -> None:
        """删除本次物化的临时目录（幂等）。"""
        shutil.rmtree(staging_dir, ignore_errors=True)

    try:
        stager = _DependencyStager(
            entries=entries,
            resolved=resolved,
            staging_dir=staging_dir,
            logical_base=logical_base,
        )
        materialized_text = stager.stage_model(model_text)
        # F08：兄弟片段与模型同目录，工厂优先命中快照内容而非仓库同名文件
        stager.stage_sibling_fragments()
        # 沿用原模型文件名：R1D 组装按 model_path 名/内容推断侧别，避免
        # 无谓改变原有语义。
        xml_path = staging_dir / resolved_model.name
        xml_path.write_text(materialized_text, encoding="utf-8", newline="\n")
        profile_path = _materialize_profile(payload, sources, staging_dir, xml_path)
    except BaseException:
        cleanup()
        raise
    return profile_path, cleanup
def _replay_config(payload: Mapping[str, Any], sources: Mapping[str, Path]) -> Any:
    """由 payload 与校验后的来源重建回放用 ``ResolvedV12Config``。"""
    config = config_from_dict(payload["config"])
    replacements = {
        "robot_profile_path": sources.get("robot_profile"),
        "runtime_path": sources.get("runtime"),
        "task_path": sources.get("task"),
        "racket_path": sources.get("racket"),
    }
    updates = {
        name: path
        for name, path in replacements.items()
        if path is not None
    }
    if not updates:
        return config
    from dataclasses import replace

    return replace(config, **updates)
def _prepare_replay_config(
    payload: Mapping[str, Any],
    sources: Mapping[str, Path],
    config: Any,
    model_path: Path | None,
    *,
    output_dir: Path | None = None,
) -> tuple[Any, Callable[[], None] | None]:
    """按实际加载模型修正回放配置（快照模型需物化为临时 profile）。

    只要 ``sources["model"]`` 存在（选中了模型快照）就**无条件物化**：
    即使快照 profile 里的相对 ``model_path`` 恰好解析回快照内的模型文件
    （``snapshot/files/src/robot/...``），快照目录也没有复制 mesh/texture
    资产，直接加载会因资产缺失而失败。只有外部回退（``sources`` 无
    ``model``）才保持原配置。

    Returns:
        ``(config, cleanup)``；未物化时 ``cleanup`` 为 None。
    """
    if model_path is None or "model" not in sources:
        return config, None
    temp_profile, cleanup = _materialize_snapshot_model(
        payload, sources, Path(model_path), output_dir=output_dir
    )
    from dataclasses import replace

    return replace(config, robot_profile_path=temp_profile), cleanup
def _build_replay_bundle_with_snapshot(
    artifacts: EpisodeArtifacts,
    sources: Mapping[str, Path],
    config: Any,
    model_path: Path | None,
) -> V12RobotBundle:
    """物化（必要时）并构建回放 bundle，临时目录生命周期绑定到 bundle。"""
    replay_config, cleanup = _prepare_replay_config(
        artifacts.payload,
        sources,
        config,
        model_path,
        output_dir=artifacts.directory,
    )
    from src.sim.v12_robot_adapter import build_v12_robot  # 延迟导入

    try:
        # F01：回放与真实执行消费同一份已保存 runtime_overrides（limits/
        # 碰撞/来球开关），而不是重新按默认值构建 bundle。
        from src.sim.episode_runtime import (  # 延迟导入避免循环依赖
            apply_runtime_limits,
            resolve_runtime_options,
        )

        replay_config = resolve_runtime_options(replay_config, {})
        bundle = build_v12_robot(replay_config)
        apply_runtime_limits(replay_config, bundle)
    except BaseException:
        if cleanup is not None:
            cleanup()
        raise
    if cleanup is not None:
        # planning_env_factory 在首次规划时才按 profile.model_path 建规划
        # 环境，临时目录必须活到 bundle 生命周期结束（弱引用终结器兜底）。
        weakref.finalize(bundle, cleanup)
    return bundle
