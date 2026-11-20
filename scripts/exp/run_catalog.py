"""运行目录 catalog（RunCatalog）的加载、路径基准与一致性校验。

R1D×V12 论文链路的每个 output 根都由生产端（``run_r1d_paper_smoke``）写一个
``catalog.json``，声明 run_root / data_root / runs.csv / manifest 与数据来源集合。
消费端（提取、绘图、统计）通过本模块解析路径，不再逐层猜测父目录：

- catalog 自身声明的 JSON 路径**相对 catalog 文件所在目录**解析，也接受绝对路径；
- ``runs.csv`` 中的 ``artifact_dir`` / ``log_path`` 相对路径以 ``run_root`` 为基准
  （``path_base=run_root``），绝对路径原样使用（兼容换 output 后仍指向旧产物的记录）；
- 显式 catalog 不存在、声明文件互相冲突、身份不一致时抛 :class:`CatalogError`，
  CLI 以非零码退出——绝不回退去猜父目录。

本模块只用标准库，便于提取/绘图脚本独立调用。
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

CATALOG_FILENAME = "catalog.json"
CATALOG_SCHEMA_VERSION = 1
PATH_BASE_CATALOG = "catalog"
PATH_BASE_RUN_ROOT = "run_root"
VALID_PATH_BASES = (PATH_BASE_CATALOG, PATH_BASE_RUN_ROOT)
REQUIRED_STRING_KEYS = (
    "robot_id", "run_root", "data_root", "runs_csv", "manifest_path")
REQUIRED_KEYS = ("schema_version", *REQUIRED_STRING_KEYS, "source_sets")


class CatalogError(RuntimeError):
    """catalog 缺失、格式非法、声明冲突或身份不一致。"""


@dataclass(frozen=True)
class RunCatalog:
    """解析后的运行目录 catalog（所有路径已按基准解析为绝对路径）。

    Attributes:
        schema_version: catalog 结构版本（当前固定为 1）。
        robot_id: 机器人/场景身份（如 ``unitree_r1d``、``rm65``）。
        run_root: output 根目录（catalog 所在目录）。
        data_root: 批量结果根目录（可位于 run_root 内外）。
        runs_csv: 富化后的 runs.csv 路径（必须位于 run_root 内）。
        manifest_path: 解析后的 manifest 路径（必须位于 run_root 内）。
        source_sets: 数据来源集合名 → 实验名（至少含 ``nominal_mechanism``）。
        catalog_path: catalog 文件自身的绝对路径（加载后补入）。
        path_base: ``runs.csv`` 相对路径的基准（恒为 ``run_root``）。
        synthetic: 是否为合成/示例 catalog（R0 冻结样例为 True）。
        declared: 原始 JSON 声明（只读副本，供诊断/审计）。
    """

    schema_version: int
    robot_id: str
    run_root: Path
    data_root: Path
    runs_csv: Path
    manifest_path: Path
    source_sets: Mapping[str, str]
    catalog_path: Path
    path_base: str = PATH_BASE_RUN_ROOT
    synthetic: bool = False
    declared: Mapping[str, Any] = field(default_factory=dict)


def _as_nonempty_str(value: Any, key: str) -> str:
    """读取必填非空字符串字段。"""
    if not isinstance(value, str) or not value.strip():
        raise CatalogError(f"catalog 字段 {key!r} 必须是非空字符串: {value!r}")
    return value.strip()


def _resolve_json_path(base_dir: Path, value: str) -> Path:
    """解析 catalog 自身声明的 JSON 路径（相对 base_dir，绝对原样）。"""
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _ensure_under_run_root(run_root: Path, path: Path, key: str) -> None:
    """校验声明文件位于 run_root 内（runs.csv/manifest 属于 output 根）。"""
    try:
        path.relative_to(run_root)
    except ValueError as exc:
        raise CatalogError(
            f"catalog 声明冲突: {key} 不在 run_root 内: {path} (run_root={run_root})"
        ) from exc


def validate_catalog_payload(payload: Any) -> dict[str, Any]:
    """校验 catalog JSON 的结构（不解析路径、不读其他文件）。

    Args:
        payload: ``json.loads`` 得到的对象。

    Returns:
        原始声明字典（浅拷贝）。

    Raises:
        CatalogError: 结构、版本或必填字段不合法。
    """
    if not isinstance(payload, Mapping):
        raise CatalogError(f"catalog 顶层必须是 JSON 对象: {type(payload).__name__}")
    schema_version = payload.get("schema_version")
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogError(
            f"catalog schema_version 必须为 {CATALOG_SCHEMA_VERSION}: {schema_version!r}")
    for key in REQUIRED_STRING_KEYS:
        _as_nonempty_str(payload.get(key), key)
    source_sets = payload.get("source_sets")
    if not isinstance(source_sets, Mapping) or not source_sets:
        raise CatalogError("catalog 字段 'source_sets' 必须是非空对象")
    for name, experiment in source_sets.items():
        _as_nonempty_str(experiment, f"source_sets.{name}")
    path_base = payload.get("path_base", PATH_BASE_RUN_ROOT)
    if path_base not in VALID_PATH_BASES:
        raise CatalogError(f"catalog path_base 非法: {path_base!r}")
    return dict(payload)


def _check_declared_files(
    *,
    catalog_path: Path,
    run_root: Path,
    data_root: Path,
    runs_csv: Path,
    manifest_path: Path,
    robot_id: str,
) -> None:
    """校验声明文件之间的一致性（冲突即 :class:`CatalogError`）。"""
    if runs_csv == manifest_path:
        raise CatalogError(
            f"catalog 声明冲突: runs_csv 与 manifest_path 指向同一文件: {runs_csv}")
    _ensure_under_run_root(run_root, runs_csv, "runs_csv")
    _ensure_under_run_root(run_root, manifest_path, "manifest_path")
    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CatalogError(f"声明文件无法解析: {manifest_path}: {exc}") from exc
        if not isinstance(manifest, Mapping):
            raise CatalogError(f"声明文件冲突: manifest 不是 JSON 对象: {manifest_path}")
        declared_root = manifest.get("run_root")
        if isinstance(declared_root, str) and declared_root.strip():
            candidate = _resolve_json_path(catalog_path.parent, declared_root)
            if candidate != run_root:
                raise CatalogError(
                    "catalog 声明冲突: manifest.run_root 与 catalog.run_root 不一致: "
                    f"{candidate} != {run_root}")
        declared_data = manifest.get("data_root")
        if isinstance(declared_data, str) and declared_data.strip():
            candidate = _resolve_json_path(catalog_path.parent, declared_data)
            if candidate != data_root:
                raise CatalogError(
                    "catalog 声明冲突: manifest.data_root 与 catalog.data_root 不一致: "
                    f"{candidate} != {data_root}")
        identity = manifest.get("identity")
        if isinstance(identity, Mapping):
            manifest_robot = identity.get("robot_id")
            if (isinstance(manifest_robot, str) and manifest_robot.strip()
                    and manifest_robot != robot_id):
                raise CatalogError(
                    "catalog 身份不一致: manifest.identity.robot_id="
                    f"{manifest_robot!r} != catalog.robot_id={robot_id!r}")
    if runs_csv.is_file():
        try:
            with runs_csv.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                if "robot_id" in (reader.fieldnames or []):
                    for line_no, row in enumerate(reader, start=2):
                        value = (row.get("robot_id") or "").strip()
                        if value and value != robot_id:
                            raise CatalogError(
                                "catalog 身份不一致: "
                                f"runs.csv:{line_no} robot_id={value!r} "
                                f"!= catalog.robot_id={robot_id!r}")
        except OSError as exc:
            raise CatalogError(f"runs.csv 无法读取: {runs_csv}: {exc}") from exc


def load_catalog(path: Path) -> RunCatalog:
    """加载并校验 catalog JSON，解析所有声明路径。

    Args:
        path: catalog 文件路径（显式给出，不回退搜索父目录）。

    Returns:
        :class:`RunCatalog`，``catalog_path`` 为绝对路径。

    Raises:
        CatalogError: 文件不存在、JSON/结构非法、声明冲突或身份不一致。
    """
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise CatalogError(f"catalog 不存在: {catalog_path}")
    try:
        raw = catalog_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise CatalogError(f"catalog 无法读取: {catalog_path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise CatalogError(f"catalog JSON 解析失败: {catalog_path}: {exc}") from exc
    declared = validate_catalog_payload(payload)

    catalog_path = catalog_path.resolve()
    base_dir = catalog_path.parent
    robot_id = _as_nonempty_str(declared["robot_id"], "robot_id")
    run_root = _resolve_json_path(base_dir, declared["run_root"])
    data_root = _resolve_json_path(base_dir, declared["data_root"])
    runs_csv = _resolve_json_path(base_dir, declared["runs_csv"])
    manifest_path = _resolve_json_path(base_dir, declared["manifest_path"])
    source_sets = {
        str(name): _as_nonempty_str(value, f"source_sets.{name}")
        for name, value in dict(declared["source_sets"]).items()
    }

    _check_declared_files(
        catalog_path=catalog_path,
        run_root=run_root,
        data_root=data_root,
        runs_csv=runs_csv,
        manifest_path=manifest_path,
        robot_id=robot_id,
    )
    return RunCatalog(
        schema_version=int(declared["schema_version"]),
        robot_id=robot_id,
        run_root=run_root,
        data_root=data_root,
        runs_csv=runs_csv,
        manifest_path=manifest_path,
        source_sets=source_sets,
        catalog_path=catalog_path,
        path_base=str(declared.get("path_base", PATH_BASE_RUN_ROOT)),
        synthetic=bool(declared.get("synthetic", False)),
        declared=declared,
    )


def resolve_catalog_path(
    catalog: RunCatalog,
    value: str | Path,
    *,
    base: str = PATH_BASE_CATALOG,
) -> Path:
    """解析 catalog 声明的 JSON 路径。

    Args:
        catalog: 已加载的 catalog。
        value: 声明值（相对或绝对）。
        base: ``"catalog"``（相对 catalog 文件目录，默认）或 ``"run_root"``。

    Returns:
        绝对路径；绝对输入原样返回（不要求存在）。

    Raises:
        CatalogError: ``base`` 非法。
    """
    if base not in VALID_PATH_BASES:
        raise CatalogError(f"base 必须属于 {VALID_PATH_BASES}: {base!r}")
    path = Path(value)
    if path.is_absolute():
        return path
    root = catalog.catalog_path.parent if base == PATH_BASE_CATALOG else catalog.run_root
    return (root / path).resolve()


def resolve_run_path(catalog: RunCatalog, value: str | Path) -> Path:
    """按 ``path_base=run_root`` 解析 runs.csv 内的产物/日志路径。

    Args:
        catalog: 已加载的 catalog。
        value: runs.csv 中的 ``artifact_dir`` / ``log_path``（相对或绝对）。

    Returns:
        绝对路径；绝对输入原样返回（换 output 后仍指向旧产物）。
    """
    return resolve_catalog_path(catalog, value, base=PATH_BASE_RUN_ROOT)


def _summary_payload(catalog: RunCatalog) -> dict[str, Any]:
    """把 catalog 整理为可打印/可序列化的摘要。"""
    return {
        "catalog_path": str(catalog.catalog_path),
        "schema_version": catalog.schema_version,
        "robot_id": catalog.robot_id,
        "run_root": str(catalog.run_root),
        "data_root": str(catalog.data_root),
        "runs_csv": str(catalog.runs_csv),
        "manifest_path": str(catalog.manifest_path),
        "path_base": catalog.path_base,
        "synthetic": catalog.synthetic,
        "source_sets": dict(catalog.source_sets),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI：校验并打印一个显式 catalog。

    Args:
        argv: 命令行参数（默认 ``sys.argv[1:]``）。

    Returns:
        0 表示通过；2 表示 catalog 缺失/冲突/身份不一致。
    """
    parser = argparse.ArgumentParser(
        description="校验并打印运行目录 catalog（RunCatalog v1）。")
    parser.add_argument("catalog", type=Path, help="catalog JSON 路径（显式路径）")
    parser.add_argument("--json", action="store_true", help="以 JSON 输出解析结果")
    args = parser.parse_args(argv)
    try:
        catalog = load_catalog(args.catalog)
    except CatalogError as exc:
        print(f"[run_catalog] 错误: {exc}", file=sys.stderr)
        return 2
    payload = _summary_payload(catalog)
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print(f"catalog: {payload['catalog_path']}")
        print(f"run_root: {payload['run_root']}")
        print(f"data_root: {payload['data_root']}")
        print(f"runs_csv: {payload['runs_csv']}")
        print(f"manifest: {payload['manifest_path']}")
        print(f"robot_id: {payload['robot_id']}  path_base={payload['path_base']}")
        print(f"source_sets: {payload['source_sets']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
