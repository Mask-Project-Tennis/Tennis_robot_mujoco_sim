#!/usr/bin/env python3
"""batch_runner —— 通用并行批量实验执行器（T8a：身份/数据根/超时/续跑）。

功能:
- ProcessPoolExecutor 并行执行参数网格（默认 16 workers；workers=1 时改为串行，
  R1D 论文 smoke 按矩阵要求固定 workers=1）
- --resume: 按“身份命名空间”跳过已完成任务
  (experiment, robot_id, mode, racket_id, config_hash, metric_schema, config_id, seed)
- 身份列入 CSV: robot_id / mode / racket_id / config_hash / metric_schema 每行落盘，
  不同机器人、执行模式、冻结配置、指标 schema 禁止混写同一 results.csv（发现即报错）
- --data-root/--output: 数据根与输出目录贯穿，不再写死 experiment_data；
  data_root 默认 PROJECT/experiment_data，output 覆盖单个实验目录
- 产物隔离: RM65 V12 入口（``scripts/rm65_mpc_v12.py``）默认把 JSON/NPZ 与
  ``snapshot/`` 写到仓库级 ``results/``，并行任务会互相覆盖。批量执行时凡
  未显式给 ``--output`` 的 V12 任务，一律注入独立目录
  ``<output>/artifacts/<config slug>__<digest>/seed<seed>``（dry-run 打印的
  命令与真实执行、CSV ``artifact_dir`` 三者一致；resume 身份不含该路径）
- --timeout: 显式单 run 超时（秒）；任务级 ``_timeout`` meta 可覆盖；
  超时行落盘（error=timeout、timed_out=1）且视为终态，resume 不重跑覆盖
- --dry-run: 只构造并打印完整任务清单（命令/身份/config_id/seed），不执行、不落盘
- 兼容旧 RM65 实验：默认身份 LEGACY_IDENTITY，旧 21 列 CSV 在显式
  --legacy-resume 下可续跑；新 CSV 自动带身份列与状态列
- 单 run 失败记 error 行，不阻塞整体
- OMP_NUM_THREADS=1 防止 numpy/MuJoCo 内部线程与 worker 数超订

实验矩阵定义见 overnight_experiments.py（本文件只负责执行）。

用法:
    python scripts/exp/batch_runner.py exp15_speed --workers 16
    python scripts/exp/batch_runner.py exp17a_noise --limit 3      # smoke test
    python scripts/exp/batch_runner.py exp15_speed --dry-run       # 只打印命令
    python scripts/exp/batch_runner.py exp28_always_collision \\
        --data-root experiment_data --timeout 900
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping

# 防 numpy/MuJoCo 内部多线程与 worker 数超订（必须在 import numpy 前设置）
os.environ.setdefault("OMP_NUM_THREADS", "1")

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

RESULT_RE = re.compile(r"__RESULT__: (.+)")
R1D_RESULT_RE = re.compile(r"__R1D_RESULT__: (.+)")
DEFAULT_TIMEOUT_S = 60
# 旧常量名保留（外部脚本/文档引用兼容）
RUN_TIMEOUT_S = DEFAULT_TIMEOUT_S

#: RM65 V12 主脚本名：其 ``--output`` 缺省值是仓库级 ``results/``，
#: 并行批量若都不传该参数就会写同一套 JSON/NPZ 并重建同一 ``snapshot/``。
RM65_V12_SCRIPT_NAME = "rm65_mpc_v12.py"
#: 批量产物根目录名：位于实验输出目录下，与 ``results.csv``/``logs/`` 平级。
ARTIFACT_ROOT_NAME = "artifacts"
#: 产物目录名中 config_id slug 的截断长度（与 ``_log_stem`` 同口径）；
#: 实测 exp28 的 config_id 可达 226 字符，必须截断 + 摘要才能安全落盘。
ARTIFACT_SLUG_MAX_CHARS = 48

# 旧 CSV 列（T8a 之前格式，顺序不可变，保证旧文件可读）
LEGACY_CSV_FIELDS = [
    "experiment", "config_id", "config_json", "seed", "hit", "hit_type",
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_time_error_ms", "hit_pos_error",
    "v_racket_at_hit", "wall_time", "first_plan_ms", "ball_out_speed",
    "ball_out_vy", "error",
]
IDENTITY_FIELDS = ["robot_id", "mode", "racket_id", "config_hash", "metric_schema"]
STATUS_FIELDS = [
    "timeout_s", "timed_out", "exit_code", "failure_reason", "total_steps",
    "safe_steps", "hit_step", "artifact_dir", "log_path",
]
CSV_FIELDS = LEGACY_CSV_FIELDS + IDENTITY_FIELDS + STATUS_FIELDS
# 单 run 日志文件名的总长度上限（含 ".log"）；Windows 全路径限制 260，
# 再留出 log_dir/<experiment>/ 前缀的空间。
LOG_FILENAME_MAX_CHARS = 96

NUMERIC_FIELDS = {
    "pos_error", "vel_error", "min_dist", "ball_near_ms", "tube_ready_ms",
    "max_tcp", "max_qdot", "max_face", "hit_time_error_ms", "hit_pos_error",
    "v_racket_at_hit", "wall_time", "first_plan_ms", "ball_out_speed",
    "ball_out_vy",
}
INT_FIELDS = {"total_steps", "safe_steps", "hit_step"}
# R1D 结果行内 "None" 字面量需归一为空串的字段
R1D_NONE_FIELDS = {"failure_reason"}


@dataclass(frozen=True)
class RunIdentity:
    """单次 run 的身份命名空间（resume 隔离与 CSV 混写保护）。

    Attributes:
        robot_id: 机器人标识（``rm65`` / ``unitree_r1d``）。
        mode: 执行模式（``torque`` / ``position``）；空串表示按任务参数推导。
        racket_id: 球拍标识（旧 RM65 数据为 ``rm65_legacy``）。
        config_hash: 冻结配置哈希（旧数据为 ``legacy``；新数据运行时计算）。
        metric_schema: 指标 schema 版本（RM65 旧 ``v12_result_v1``）。
    """

    robot_id: str = "rm65"
    mode: str = ""
    racket_id: str = "rm65_legacy"
    config_hash: str = "legacy"
    metric_schema: str = "v12_result_v1"

    def as_dict(self) -> dict[str, str]:
        """转为可 JSON 序列化的字典（mode 可能为空串，表示待按任务推导）。"""
        return {
            "robot_id": self.robot_id,
            "mode": self.mode,
            "racket_id": self.racket_id,
            "config_hash": self.config_hash,
            "metric_schema": self.metric_schema,
        }


LEGACY_IDENTITY = RunIdentity()


@dataclass(frozen=True, order=True)
class ResumeKey:
    """resume 键：身份命名空间 + config_id + seed。"""

    experiment: str
    robot_id: str
    mode: str
    racket_id: str
    config_hash: str
    metric_schema: str
    config_id: str
    seed: int

    def as_dict(self) -> dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "experiment": self.experiment,
            "robot_id": self.robot_id,
            "mode": self.mode,
            "racket_id": self.racket_id,
            "config_hash": self.config_hash,
            "metric_schema": self.metric_schema,
            "config_id": self.config_id,
            "seed": self.seed,
        }


@dataclass
class ExperimentSpec:
    """单个批量实验的完整定义。

    Attributes:
        name: 实验名（CLI 参数 & CSV experiment 列）。
        script: 相对项目根的仿真脚本路径。
        grid: 参数网格，每个元素是一次 run 的 CLI 参数字典
              （{"--flag": value, "--开关": None}，seed 由调用方写入 grid）。
        report_ref: 旧报告对照摘要（写进 config.yaml 供 sanity check）。
        identity: 身份命名空间（默认旧 RM65 身份，mode 留空按任务推导）。
        grid_factory: 延迟网格工厂（如 exp28 需读取当前 data-root 下的
              exp24 CSV）；非 None 时优先于 grid，且 import 阶段不读数据。
              工厂接收运行期解析好的 data-root（Path），确保 ``--data-root``
              真正贯穿到延迟构造阶段。
    """

    name: str
    script: str
    grid: list[dict[str, Any]] = field(default_factory=list)
    report_ref: str = ""
    identity: RunIdentity = field(default_factory=lambda: LEGACY_IDENTITY)
    grid_factory: Callable[[Path | None], list[dict[str, Any]]] | None = None

    def materialize_grid(
        self,
        data_root: str | Path | None = None,
    ) -> list[dict[str, Any]]:
        """返回本次执行使用的参数网格（延迟工厂优先，不修改原对象）。

        Args:
            data_root: 运行期解析的数据根；传给延迟工厂，避免构造期固化
                旧默认根。None 表示沿用工厂自身的默认行为。
        """
        if self.grid_factory is not None:
            root = Path(data_root) if data_root is not None else None
            return self.grid_factory(root)
        return self.grid


def resolve_identity(identity: RunIdentity, params: Mapping[str, Any]) -> RunIdentity:
    """把身份中留空的 mode 按任务参数推导为 torque/position。

    Args:
        identity: 实验级身份（mode 可为空串）。
        params: 单次 run 的 CLI 参数字典。

    Returns:
        完整身份（mode 已确定）。
    """
    if identity.mode:
        return identity
    return replace(identity, mode="position" if "--position-mode" in params else "torque")


def identity_key(identity: RunIdentity) -> tuple[str, str, str, str, str]:
    """返回身份的值元组（跨模块同名 dataclass 也能正确比较）。

    注意：``scripts/exp/batch_runner.py`` 作为 __main__ 运行时与
    ``scripts.exp.batch_runner`` 是两份模块，``RunIdentity`` 类对象不同；
    续跑身份比较必须按字段值而非类/实例相等（T8a 评审 B3）。
    """
    return (identity.robot_id, identity.mode, identity.racket_id,
            identity.config_hash, identity.metric_schema)


def _task_identity(
    identity: RunIdentity,
    params: Mapping[str, Any],
) -> RunIdentity:
    """推导单任务身份：按参数定 mode，并允许 ``_config_hash`` 覆盖 hash。

    ``_config_hash`` 由调用方（R1D smoke runner）按 per-task 有效配置
    （含 runtime/task overrides）计算，保证同一实验内不同 runtime 变体
    不会跨身份混续跑。
    """
    resolved = resolve_identity(identity, params)
    override = params.get("_config_hash")
    if override:
        resolved = replace(resolved, config_hash=str(override))
    return resolved


def _sha256_file(path: Path) -> str:
    """计算文件 sha256（分块读取，避免大模型文件占内存）。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_config_hash(
    role_paths: Mapping[str, str | Path],
    extra: Mapping[str, Any] | None = None,
) -> str:
    """按冻结配置文件计算 config_hash（16 hex）。

    禁止伪造：任一角色文件缺失都会抛 FileNotFoundError；不提供“默认哈希”。

    Args:
        role_paths: 角色名 → 文件路径（相对路径按项目根解析）。
        extra: 额外参与哈希的元数据（如执行模式、overrides 规范化字典）。

    Returns:
        sha256(canonical JSON(role→文件 sha256, extra))[:16]。
    """
    payload: dict[str, Any] = {}
    for role in sorted(role_paths):
        path = Path(role_paths[role])
        if not path.is_absolute():
            path = PROJECT / path
        if not path.is_file():
            raise FileNotFoundError(
                f"config_hash 无法计算：角色 {role!r} 的文件不存在: {path}")
        payload[role] = _sha256_file(path)
    if extra:
        payload["extra"] = dict(sorted(extra.items()))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def build_command(script: str, params: Mapping[str, Any]) -> list[str]:
    """集中构造单次 run 的子进程命令（禁止各处手工拼接）。

    Args:
        script: 相对项目根的脚本路径（调用方可先做 ``_script`` 覆盖）。
        params: CLI 参数；``_`` 开头键为 meta 不传给子进程；
            None 值表示布尔开关。

    Returns:
        完整 argv 列表（首元素为当前 Python 解释器）。
    """
    cmd = [sys.executable, str(PROJECT / script)]
    for key, val in params.items():
        if str(key).startswith("_"):
            continue
        if isinstance(val, (list, tuple)):
            # 列表值展开为重复 flag（如 --task-overrides k=v 可重复多次）
            for item in val:
                cmd.append(str(key))
                cmd.append(str(item))
            continue
        cmd.append(str(key))
        if val is not None:
            cmd.append(str(val))
    return cmd


def make_config_id(params: Mapping[str, Any]) -> str:
    """从非 seed 的 CLI 参数生成可读且稳定的 config_id。

    排序保证同一参数集合恒定映射到同一 id（resume 依赖此性质）。
    ``--seed``/``--output`` 与 ``_`` 开头的 meta 键不参与 id（输出目录与
    seed 属于任务位置而非配置身份，避免同一配置因输出路径不同碎片化）。

    Args:
        params: 单次 run 的 CLI 参数字典。

    Returns:
        形如 "ballspeed9__servebox" 的可读字符串；无参数时为 "default"。
    """
    parts: list[str] = []
    for key in sorted(params):
        if key in ("--seed", "--output") or str(key).startswith("_"):
            continue
        val = params[key]
        flag = key.lstrip("-").replace("-", "")
        parts.append(f"{flag}{'' if val is None else val}")
    # 多值参数（如 --kp "500 500 ..."）会含空格，压成 -
    joined = "__".join(parts).replace(" ", "-").replace("/", "_")
    return joined or "default"


def _effective_config_id(params: Mapping[str, Any]) -> str:
    """返回单个任务的配置身份（优先 manifest 冻结的 ``_config_id``）。

    生产端（``run_r1d_paper_smoke``）在解析 manifest 时已算出稳定
    ``config_id``，并通过保留元数据 ``_config_id`` 传给 batch。它排除了
    ``--output``/runtime 快照等会随输出根变化的绝对路径，因此换 output
    续跑时身份不变。旧调用没有该元数据时，退回按参数重算的
    :func:`make_config_id`，保持既有行为。

    Args:
        params: 单次 run 的 CLI 参数字典（可含 ``_`` 前缀保留元数据）。

    Returns:
        非空配置身份字符串。
    """
    value = params.get("_config_id")
    if value is not None and str(value).strip():
        return str(value)
    return make_config_id(params)


def _is_rm65_v12_entry(script: str | Path) -> bool:
    """判断脚本是否为 RM65 V12 入口（只看文件名）。

    生产端既可能传相对项目根的路径（``scripts/rm65_mpc_v12.py``），也可能
    通过 ``_script`` 元数据传绝对路径或 ``Path``；只有 basename 是配置入口名
    时才需要注入独立产物目录，避免误伤 R1D 与其他实验脚本。

    Args:
        script: 脚本路径（``str`` 或 ``Path``）。

    Returns:
        文件名等于 :data:`RM65_V12_SCRIPT_NAME` 时为 True。
    """
    return Path(str(script)).name == RM65_V12_SCRIPT_NAME


def _artifact_dir_name(config_id: str) -> str:
    """把 config_id 压成 Windows 合法、长度受控的产物子目录名。

    与 :func:`_log_stem` 同思路：slug 清洗截断 + sha256 摘要。摘要只取
    ``config_id`` 本身，因此同一配置的所有 seed 共用一个配置目录，不会被
    截断成同一 slug 的不同配置误合并。

    Args:
        config_id: 配置身份字符串（可能含空格、方括号、数百字符）。

    Returns:
        形如 ``ballspeed9__1a2b3c4d5e`` 的安全目录名（不含 seed 层级）。
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(config_id))[:ARTIFACT_SLUG_MAX_CHARS]
    slug = slug.strip("-.")
    digest = hashlib.sha256(str(config_id).encode("utf-8")).hexdigest()[:10]
    return f"{slug or 'case'}__{digest}"


def _seed_dir_name(seed: Any) -> str:
    """生成 seed 子目录名（非法字符清洗、长度受控）。

    Args:
        seed: seed 值（int/str/None 均可）。

    Returns:
        形如 ``seed42`` 的安全目录名；seed 缺失或清洗后为空时记为 ``seedNA``。
    """
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", "NA" if seed is None else str(seed))
    safe = safe[:16].strip("-.")
    return f"seed{safe or 'NA'}"


def rm65_artifact_dir(out_dir: str | Path, config_id: str, seed: Any) -> Path:
    """单个 RM65 V12 任务的独立产物目录。

    布局为 ``<out_dir>/artifacts/<config slug>__<digest>/seed<seed>``：与
    R1D 论文 smoke 的 ``<run_root>/artifacts/<unit>/<case>/seed<N>`` 同构，
    下游按 ``<case>/seed<N>`` 尾部识别任务时不需特殊处理。

    Args:
        out_dir: 本实验的输出目录（``run_batch`` 的 ``output`` 解析结果）。
        config_id: 配置身份（``_effective_config_id``，不含 seed/--output）。
        seed: seed 值。

    Returns:
        计划产物目录（不创建目录；真实写盘由 ``write_episode_artifacts``
        自行 ``mkdir(parents=True)``）。
    """
    return (Path(out_dir) / ARTIFACT_ROOT_NAME
            / _artifact_dir_name(config_id) / _seed_dir_name(seed))


def _prepare_task_params(
    script: str,
    params: Mapping[str, Any],
    out_dir: str | Path,
) -> dict[str, Any]:
    """为一个任务准备实际执行的参数（必要时注入独立 ``--output``）。

    规则（优先方案 A：批量入口注入，不改仿真脚本语义）：

    1. 已显式带 ``--output`` 的任务原样返回——R1D 论文 smoke 与其他实验脚本
       的产物路径由 manifest 决定，不能被二次注入或改写；
    2. 只有 RM65 V12 入口（含 ``_script`` 覆盖）且缺 ``--output`` 时才注入，
       路径由 :func:`rm65_artifact_dir` 按 ``(config_id, seed)`` 决定，因此
       不同 seed/config 不共享目录，同一任务重复 prepare/resume 结果相同；
    3. 返回浅拷贝，绝不修改 ``ExperimentSpec.grid`` 里的原始字典。

    Args:
        script: 实验级脚本路径（``ExperimentSpec.script``）。
        params: 网格里单个任务的 CLI 参数（不可变视图亦可）。
        out_dir: 本实验输出目录。

    Returns:
        实际传给 :func:`build_command` 的新参数字典。
    """
    prepared = dict(params)
    if prepared.get("--output") not in (None, ""):
        return prepared
    effective_script = str(prepared.get("_script", script))
    if not _is_rm65_v12_entry(effective_script):
        return prepared
    prepared["--output"] = str(rm65_artifact_dir(
        out_dir, _effective_config_id(prepared), prepared.get("--seed")))
    return prepared


def parse_result_line(stdout: str) -> dict[str, Any]:
    """从仿真脚本输出中解析结果行（RM65 ``__RESULT__`` / R1D ``__R1D_RESULT__``）。

    Args:
        stdout: 子进程完整输出（stdout + stderr）。

    Returns:
        指标字典（数值字段已转 float/int；R1D 的 ``failure_reason=None``
        归一为空串、``output=`` 转为 ``artifact_dir``）；
        两行都不存在时返回 ``{"error": "no_result"}``。
    """
    match = R1D_RESULT_RE.search(stdout)
    kind = "r1d"
    if match is None:
        match = RESULT_RE.search(stdout)
        kind = "rm65"
    if match is None:
        return {"error": "no_result"}
    fields: dict[str, Any] = dict(re.findall(r"(\S+)=(\S+)", match.group(1)))
    fields["result_line_kind"] = kind
    for key in NUMERIC_FIELDS:
        if key in fields:
            try:
                fields[key] = float(fields[key])
            except ValueError:
                pass
    for key in INT_FIELDS:
        if key in fields:
            try:
                fields[key] = int(float(fields[key]))
            except ValueError:
                pass
    if kind == "r1d":
        for key in R1D_NONE_FIELDS:
            if fields.get(key, "").strip().lower() in ("none", "null"):
                fields[key] = ""
        if "output" in fields:
            fields["artifact_dir"] = fields.pop("output")
    return fields


def load_done_keys(csv_path: Path) -> set[tuple[str, int]]:
    """读取旧格式 results.csv，返回已完成 (config_id, seed) 集合（兼容接口）。

    Args:
        csv_path: results.csv 路径（可不存在，返回空集）。

    Returns:
        已完成 run 的旧键集合；error 行（timeout 除外）不计入完成。
    """
    done: set[tuple[str, int]] = set()
    if not csv_path.exists():
        return done
    for row in _read_csv(csv_path)[1]:
        if not _row_is_done(row):
            continue
        try:
            done.add((row["config_id"], int(row["seed"])))
        except (KeyError, ValueError):
            continue
    return done


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取 CSV，返回 (表头, 行字典列表)。"""
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _row_is_done(row: Mapping[str, Any]) -> bool:
    """判断一行是否终态：显式 timed_out，或 error 为空/timeout。"""
    if str(row.get("timed_out", "")).strip() in ("1", "true", "True"):
        return True
    error = str(row.get("error") or "").strip()
    return not error or error == "timeout"


def _row_identity(row: Mapping[str, Any]) -> RunIdentity:
    """从一行读取身份（缺失列按空串，旧 CSV 不应调用）。"""
    return RunIdentity(
        robot_id=str(row.get("robot_id") or ""),
        mode=str(row.get("mode") or ""),
        racket_id=str(row.get("racket_id") or ""),
        config_hash=str(row.get("config_hash") or ""),
        metric_schema=str(row.get("metric_schema") or ""),
    )


def make_resume_key(experiment: str, params: Mapping[str, Any],
                    identity: RunIdentity) -> ResumeKey:
    """构造单个任务的 resume 键（身份 + config_id + seed）。"""
    seed_val = params.get("--seed")
    try:
        seed = int(seed_val) if seed_val is not None else -1
    except (TypeError, ValueError):
        seed = -1
    return ResumeKey(
        experiment=experiment,
        robot_id=identity.robot_id,
        mode=identity.mode,
        racket_id=identity.racket_id,
        config_hash=identity.config_hash,
        metric_schema=identity.metric_schema,
        config_id=_effective_config_id(params),
        seed=seed,
    )


def _record(
    experiment: str,
    params: Mapping[str, Any],
    identity: RunIdentity,
    fields: Mapping[str, Any],
    *,
    timeout_s: float,
    exit_code: int | str = "",
    log_path: str = "",
) -> dict[str, Any]:
    """把指标字典补全为一行完整 CSV 记录（含身份与状态列）。"""
    record: dict[str, Any] = {key: "" for key in CSV_FIELDS}
    record.update({
        "experiment": experiment,
        "config_id": _effective_config_id(params),
        # 只记录可见 CLI 参数：``_`` 元数据（含 runtime 快照绝对路径）既不是
        # 配置内容，也不能被下游（如 scripts/video/pick_seeds.py）重放成命令行。
        "config_json": json.dumps(
            {k: v for k, v in params.items()
             if k != "--seed" and not str(k).startswith("_")},
            ensure_ascii=False, sort_keys=True),
        "seed": params.get("--seed"),
        "error": fields.get("error", ""),
        "timeout_s": timeout_s,
        "timed_out": 1 if fields.get("error") == "timeout" else 0,
        "exit_code": exit_code,
        "log_path": log_path,
        "artifact_dir": fields.get("artifact_dir", params.get("--output", "")),
    })
    record.update(identity.as_dict())
    for key in CSV_FIELDS:
        if key in fields:
            record[key] = fields[key]
    record["hit_type"] = fields.get("hit_type", record.get("hit_type", ""))
    record["hit"] = fields.get("hit", record.get("hit", ""))
    return record


def _canonical_config_json(params: Mapping[str, Any], seed: Any) -> str:
    """把任务配置身份与 seed 规范化为稳定 JSON（键排序、紧凑分隔符）。

    只包含 ``_effective_config_id`` 与 seed：``--output`` 和 ``_`` 前缀
    元数据（runtime 快照绝对路径等）不参与摘要，保证同一任务换 output
    续跑时日志摘要稳定。签名保留 ``params`` 以兼容既有调用。

    Args:
        params: 单次 run 的 CLI 参数字典（读取其 ``_config_id``）。
        seed: seed 值（int/str/None 均可）。

    Returns:
        同一 ``(config_id, seed)`` 恒定映射到同一字符串，供日志名摘要使用。
    """
    return json.dumps(
        {"config_id": _effective_config_id(params), "seed": seed},
        ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def _log_stem(config_id: str, seed: Any) -> str:
    """按配置身份与 seed 生成 Windows 合法、长度受控的日志文件名主干。

    规则与 :func:`_safe_log_stem` 完全一致（slug 48 + seed 16 + 摘要 10，
    加 ``.log`` 后 ≤96），只是身份直接取 ``config_id`` 字符串，使 manifest
    计划路径与实际写日志共用同一实现（N01）。

    Args:
        config_id: 配置身份字符串（manifest 稳定 id 或 make_config_id 结果）。
        seed: seed 值（int/str/None 均可）。

    Returns:
        形如 ``ballspeed9__s100__1a2b3c4d5e`` 的安全文件名主干（不含 ``.log``）。
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(config_id))[:48].strip("-.")
    safe_seed = re.sub(r"[^A-Za-z0-9._-]+", "-", "NA" if seed is None else str(seed))
    safe_seed = safe_seed[:16].strip("-.")
    digest = hashlib.sha256(
        json.dumps({"config_id": str(config_id), "seed": seed},
                   ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()[:10]
    return f"{slug or 'case'}__s{safe_seed or 'NA'}__{digest}"


def task_log_path(log_dir: Path, experiment: str, config_id: str,
                  seed: int | str | None = None) -> Path:
    """构造单个任务的日志路径（manifest 计划与实际落盘共用，N01）。

    日志目录布局为 ``<log_dir>/<experiment>/<stem>.log``，其中 ``stem`` 由
    :func:`_log_stem` 生成：``config_id`` 清洗截断到 48 字符 + seed + 10 位
    摘要，Windows 合法且文件名总长 ≤96 字符。

    Args:
        log_dir: 日志根目录（通常是 ``<run_root>/logs``）。
        experiment: 实验/单元名（作为日志子目录）。
        config_id: 配置身份字符串。
        seed: seed 值（int/str/None 均可，None 记为 ``NA``）。

    Returns:
        计划日志文件路径（不要求文件或目录已存在）。
    """
    return Path(log_dir) / experiment / f"{_log_stem(config_id, seed)}.log"


def _safe_log_stem(params: Mapping[str, Any], seed: Any) -> str:
    """生成 Windows 合法且长度受控的日志文件名主干（R02，不含 ``.log``）。

    ``make_config_id`` 会带上完整配置路径与 ``--task-overrides`` 列表，
    真实 R1D 用例可达 300+ 字符且含 ``:``/``\\``/``?`` 等 Windows 非法字符，
    直接当文件名会触发 ``OSError: [Errno 22]`` / ``WinError 123``。因此：

    1. slug：非 ``[A-Za-z0-9._-]`` 的连续字符压成单个 ``-``，截断到 48 字符，
       再去掉首尾的 ``-``/``.``（Windows 不允许文件名以点结尾）；
    2. seed：单独清洗并截断（非法 seed 如 ``"a/b"`` 也能安全落盘）；
    3. digest：对 ``(params, seed)`` 的规范化 JSON 取 sha256 前 10 位，
       保证 slug 截断相同、但配置不同的两个 run 不会碰撞。

    主干最长 ``48 + 2 + 16 + 2 + 10 = 78`` 字符，加 ``.log`` 后
    ``≤ LOG_FILENAME_MAX_CHARS``（96）。

    Args:
        params: 单次 run 的 CLI 参数字典。
        seed: seed 值（int/str/None 均可）。

    Returns:
        形如 ``ballspeed9__noplot__s100__1a2b3c4d5e`` 的安全文件名主干。
    """
    return _log_stem(_effective_config_id(params), seed)


def _write_run_log(log_dir: str | Path | None, experiment: str,
                   params: Mapping[str, Any], cmd: list[str], text: str) -> str:
    """写单 run 日志（失败/超时/成功都保留完整输出），返回路径字符串。

    文件名用 ``_safe_log_stem``（Windows 合法字符、总长 ≤96）；完整
    ``config_id``、参数 JSON 与命令行写在正文首部，文件名被截断后仍可追溯。

    Args:
        log_dir: 日志根目录；None 表示不写日志。
        experiment: 实验名（作为日志子目录）。
        params: 单次 run 的 CLI 参数字典。
        cmd: 实际执行的命令行（列表）。
        text: 子进程输出（stdout + stderr）。

    Returns:
        日志文件路径字符串；``log_dir`` 为 None 时返回空串。
    """
    if log_dir is None:
        return ""
    seed = params.get("--seed")
    path = task_log_path(Path(log_dir), experiment, _effective_config_id(params), seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "# config_id: " + _effective_config_id(params) + "\n"
        + "# params: "
        + json.dumps(dict(params), ensure_ascii=False, sort_keys=True) + "\n"
        + "# command: " + " ".join(cmd) + "\n\n")
    path.write_text(header + (text or ""), encoding="utf-8", errors="replace")
    return str(path)


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    """提取超时异常的已捕获输出（text=True 时可能是 str，也可能是 bytes）。"""
    parts: list[str] = []
    for chunk in (exc.stdout, exc.stderr):
        if chunk is None:
            continue
        parts.append(chunk if isinstance(chunk, str)
                     else chunk.decode("utf-8", "replace"))
    return "".join(parts)


def run_one(
    experiment: str,
    script: str,
    params: dict[str, Any],
    identity: RunIdentity = LEGACY_IDENTITY,
    default_timeout_s: float = DEFAULT_TIMEOUT_S,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """执行单次仿真 run（在 worker 进程内或 workers=1 时在主进程内运行）。

    Args:
        experiment: 实验名（透传进 CSV）。
        script: 仿真脚本路径（``_script`` meta 可覆盖）。
        params: CLI 参数字典，"--开关": None 表示布尔开关；
            ``_timeout`` 覆盖默认超时。
        identity: 实验级身份（mode 留空时按任务参数推导）。
        default_timeout_s: 默认单 run 超时（秒）。
        log_dir: 非 None 时写 per-run 日志目录。

    Returns:
        一行完整的 CSV 记录（dict），失败时含 error 键。
    """
    from src.sim.hit_detection import determine_hit_from_type

    resolved = _task_identity(identity, params)
    timeout_s = float(params.get("_timeout", default_timeout_s))
    script_path = str(params.get("_script", script))
    cmd = build_command(script_path, params)

    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
            cwd=str(PROJECT), encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        text = _timeout_output(exc)
        log_path = _write_run_log(log_dir, experiment, params, cmd, text)
        return _record(
            experiment, params, resolved, {"error": "timeout"},
            timeout_s=timeout_s, exit_code=-1, log_path=log_path)
    wall = round(time.perf_counter() - t0, 2)

    text = result.stdout + result.stderr
    log_path = _write_run_log(log_dir, experiment, params, cmd, text)
    fields = parse_result_line(text)
    if "error" in fields:
        fields["wall_time"] = wall
        return _record(experiment, params, resolved, fields,
                       timeout_s=timeout_s, exit_code=result.returncode,
                       log_path=log_path)

    hit_type = str(fields.get("hit_type", "miss"))
    fields["hit"] = determine_hit_from_type(hit_type)
    fields["wall_time"] = wall
    return _record(experiment, params, resolved, fields,
                   timeout_s=timeout_s, exit_code=result.returncode,
                   log_path=log_path)


def _display_path(path: Path) -> str:
    """尽量相对项目根显示路径，便于日志与 JSON；越界时返回绝对路径。"""
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def _inspect_existing_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """读取已存在的 results.csv，返回 (表头, 行列表)（R09）。

    与 ``resume`` 解耦：只要文件存在就先读出来做 schema/身份校验，
    避免 ``--no-resume`` 绕过校验、把不同身份的行追加进同一 CSV。

    Args:
        csv_path: 已存在的 results.csv 路径。

    Returns:
        (header, rows)。

    Raises:
        ValueError: 文件存在但没有表头（空文件无法判定身份，禁止盲目追加）。
    """
    header, rows = _read_csv(csv_path)
    if not header:
        raise ValueError(
            f"{csv_path} 已存在但表头为空，无法校验身份/schema；"
            f"请先删除该文件，或改用新的 --data-root/--output。")
    return header, rows


def _validate_csv_identity(
    header: list[str],
    rows: list[dict[str, str]],
    spec: ExperimentSpec,
    *,
    identities: set[str],
    identity_dicts: dict[str, dict[str, Any]],
    csv_path: Path,
    allow_legacy_resume: bool = False,
) -> tuple[bool, set[ResumeKey], set[tuple[str, int]]]:
    """校验已有 CSV 的 schema/身份，并收集已完成任务（R09）。

    ``--no-resume`` 只关闭"跳过已完成"，不关闭校验：CSV 一旦存在，
    其 schema 与身份必须与当前实验一致，否则追加会污染既有数据。

    Args:
        header: 已有 CSV 表头。
        rows: 已有 CSV 数据行。
        spec: 当前实验定义（身份来源）。
        identities: 当前网格全部任务的身份键集合。
        identity_dicts: 身份键 → 身份字典（报错信息用）。
        csv_path: CSV 路径（报错定位用）。
        allow_legacy_resume: 是否允许续跑旧 21 列 CSV。

    Returns:
        ``(legacy_csv, done_full, done_legacy)``：已有文件是否旧格式，
        以及新/旧两种口径下的已完成任务集合。

    Raises:
        ValueError: schema 不兼容（缺核心列/缺身份列）、旧格式身份不匹配、
            或新格式混入不同身份的行。
    """
    missing_core = [k for k in ("experiment", "config_id", "seed", "error")
                    if k not in header]
    if missing_core:
        raise ValueError(
            f"{csv_path} 表头缺少核心列 {missing_core}（现有列 {header}）；"
            f"该文件不是本 runner 写出的 results.csv，禁止追加。"
            f"请换 --data-root/--output。")
    legacy_csv = "robot_id" not in header
    done_full: set[ResumeKey] = set()
    done_legacy: set[tuple[str, int]] = set()
    if legacy_csv:
        # 按字段值比较身份：CLI 直接运行脚本时 __main__ 与
        # scripts.exp.batch_runner 是两个模块（两个 RunIdentity 类），
        # 类相等/实例相等都会恒 False（T8a 评审 B3）。
        legacy_identity = identity_key(LEGACY_IDENTITY)
        if not (allow_legacy_resume
                or identity_key(spec.identity) == legacy_identity):
            raise ValueError(
                f"{csv_path} 是旧格式 CSV（无身份列），当前实验身份为 "
                f"{spec.identity.as_dict()}；不同身份禁止混写。请改用新的 "
                f"--data-root/--output 目录，或显式传 --legacy-resume 延续旧数据。")
        for row in rows:
            if _row_is_done(row):
                try:
                    done_legacy.add((row["config_id"], int(row["seed"])))
                except (KeyError, ValueError):
                    continue
        print(f"[{spec.name}] 警告: 续跑旧格式 CSV（无身份列），"
              f"新行仍按旧列写入: {csv_path}", flush=True)
        return legacy_csv, done_full, done_legacy

    missing_identity = [k for k in IDENTITY_FIELDS if k not in header]
    if missing_identity:
        raise ValueError(
            f"{csv_path} 表头缺少身份列 {missing_identity}，无法做身份隔离；"
            f"禁止追加。请换 --data-root/--output。")
    for row in rows:
        row_key = identity_key(_row_identity(row))
        if row_key not in identities:
            raise ValueError(
                f"{csv_path} 混入不同身份的行（{_row_identity(row).as_dict()}），"
                f"当前实验身份集合为 "
                f"{[identity_dicts[k] for k in sorted(identities)]}；"
                f"不同身份禁止混写同一 CSV，请换 --data-root/--output。")
        if _row_is_done(row):
            done_full.add(_row_identity_key(spec.name, row))
    return legacy_csv, done_full, done_legacy


def run_batch(
    spec: ExperimentSpec,
    workers: int = 16,
    resume: bool = True,
    limit: int | None = None,
    *,
    data_root: str | Path | None = None,
    output: str | Path | None = None,
    timeout_s: float | None = None,
    dry_run: bool = False,
    allow_legacy_resume: bool = False,
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """并行执行一个实验的完整网格并增量写 CSV。

    Args:
        spec: 实验定义（含网格或延迟网格工厂）。
        workers: 并行进程数；``<=1`` 时串行执行（R1D 论文 smoke 要求 1）。
        resume: True 时按身份命名空间跳过已完成任务。
        limit: 只跑前 N 个待完成任务（smoke test 用）。
        data_root: 数据根目录（默认 ``PROJECT/experiment_data``）。
        output: 直接指定本实验输出目录（优先于 data_root/spec.name）。
        timeout_s: 默认单 run 超时（秒，默认 60）；任务 ``_timeout`` 可覆盖。
        dry_run: True 时只构造任务清单，不执行、不建目录、不写 CSV。
        allow_legacy_resume: 允许续跑旧 21 列 CSV（仅旧身份实验）。
        log_dir: 非 None 时写 per-run 日志（建议 <output>/logs/<exp>）。

    Returns:
        汇总字典: total / done_before / launched / hits_this_run / errors /
        timeouts / elapsed_s / csv / data_root / dry_run；
        dry_run 时额外含 ``tasks``（每条命令与身份）。

    Note:
        RM65 V12 任务的 ``--output`` 缺省会被注入
        ``<out_dir>/artifacts/<config slug>__<digest>/seed<seed>``（见
        :func:`_prepare_task_params`）：dry-run 打印的命令、真实执行的命令与
        CSV ``artifact_dir`` 因此完全一致。``--output`` 不参与身份与 resume
        键，同一 CSV 内同一 config/seed 仍被认作同一任务，注入不会让已完成
        任务被当成新任务重跑。

    Raises:
        ValueError: 旧 CSV 需要 --legacy-resume、或 CSV 混入不同身份的行。
        FileNotFoundError: 延迟网格工厂依赖的数据文件缺失（如 exp28 的
            ``<data-root>/exp24_shared_full/results.csv``）；绝不退回旧根。
    """
    base_data_root = (Path(data_root) if data_root is not None
                      else PROJECT / "experiment_data")
    # 先解析 data-root 再物化延迟网格：--data-root 必须贯穿到构造阶段
    # （exp28 seed 池读取当前数据根，而不是 import 期捕获的旧默认根）。
    grid = spec.materialize_grid(base_data_root)
    out_dir = Path(output) if output is not None else base_data_root / spec.name
    csv_path = out_dir / "results.csv"
    default_timeout = float(timeout_s if timeout_s is not None else DEFAULT_TIMEOUT_S)

    # 构造任务 + 身份（同一配置/seed 只保留一个，防止重复行）
    # 注入的 --output 在算身份之前完成：dry-run 命令、真实命令与 CSV
    # artifact_dir 都取自同一份 prepared 参数，不会出现“打印路径与实际不一致”。
    prepared: list[tuple[dict[str, Any], RunIdentity, ResumeKey]] = []
    seen: set[ResumeKey] = set()
    for raw_params in grid:
        params = _prepare_task_params(spec.script, raw_params, out_dir)
        identity = _task_identity(spec.identity, params)
        key = make_resume_key(spec.name, params, identity)
        if key in seen:
            continue
        seen.add(key)
        prepared.append((params, identity, key))
    identity_dicts = {identity_key(identity): identity.as_dict()
                      for _params, identity, _key in prepared}
    identities = set(identity_dicts)

    # 已有 CSV 一律先校验 schema/身份（R09）；resume 只决定是否跳过已完成任务。
    legacy_csv = False
    done_full: set[ResumeKey] = set()
    done_legacy: set[tuple[str, int]] = set()
    existing_header: list[str] | None = None
    if csv_path.exists():
        existing_header, rows = _inspect_existing_csv(csv_path)
        legacy_csv, done_full, done_legacy = _validate_csv_identity(
            existing_header, rows, spec,
            identities=identities, identity_dicts=identity_dicts,
            csv_path=csv_path, allow_legacy_resume=allow_legacy_resume)

    pending: list[tuple[dict[str, Any], RunIdentity, ResumeKey]] = []
    for params, identity, key in prepared:
        if resume:
            if legacy_csv:
                if (key.config_id, key.seed) in done_legacy:
                    continue
            elif key in done_full:
                continue
        pending.append((params, identity, key))
    done_before = len(prepared) - len(pending)
    if limit is not None:
        pending = pending[:limit]

    print(f"[{spec.name}] 网格 {len(grid)} | 去重后 {len(prepared)} | "
          f"已完成 {done_before} | 待跑 {len(pending)} | workers={workers} | "
          f"timeout={default_timeout:g}s | "
          f"身份={[identity_dicts[k] for k in sorted(identities)]}", flush=True)

    if dry_run:
        planned = []
        for params, identity, key in pending:
            planned.append({
                "experiment": spec.name,
                "config_id": key.config_id,
                "seed": key.seed,
                "identity": identity.as_dict(),
                "timeout_s": float(params.get("_timeout", default_timeout)),
                "command": build_command(str(params.get("_script", spec.script)),
                                         params),
            })
        for item in planned:
            print("  DRY-RUN: " + " ".join(item["command"]), flush=True)
        return {
            "experiment": spec.name,
            "total": len(grid),
            "done_before": done_before,
            "launched": len(pending),
            "hits_this_run": 0,
            "errors": 0,
            "timeouts": 0,
            "elapsed_s": 0.0,
            "csv": _display_path(csv_path),
            "data_root": _display_path(out_dir),
            "dry_run": True,
            "tasks": planned,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    if existing_header is None:
        with open(csv_path, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_FIELDS).writeheader()
        writer_fields = list(CSV_FIELDS)
    else:
        # 沿用已有表头（R09）：旧 CSV 实测列数为 20/21，与 LEGACY_CSV_FIELDS
        # 不一定等长，按固定列宽写行会错列；restval="" 补齐缺失列。
        writer_fields = list(existing_header)
    hits = errors = timeouts = 0
    t0 = time.perf_counter()

    def _consume(record: dict[str, Any], writer: csv.DictWriter) -> None:
        writer.writerow(record)
        nonlocal hits, errors, timeouts
        if record.get("error") == "timeout":
            timeouts += 1
        elif record.get("error"):
            errors += 1
        elif record.get("hit"):
            hits += 1

    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=writer_fields,
                                extrasaction="ignore", restval="")
        if workers <= 1:
            total = len(pending)
            for i, (params, identity, _key) in enumerate(pending, 1):
                record = run_one(spec.name, spec.script, params, identity,
                                 default_timeout, log_dir)
                _consume(record, writer)
                fh.flush()
                if i % 50 == 0 or i == total:
                    rate = i / max(time.perf_counter() - t0, 1e-6)
                    print(f"[{spec.name}] {i}/{total} "
                          f"({rate:.2f} runs/s, err={errors})", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(run_one, spec.name, spec.script, params,
                                identity, default_timeout, log_dir): params
                    for params, identity, _key in pending}
                for i, fut in enumerate(as_completed(futures), 1):
                    _consume(fut.result(), writer)
                    fh.flush()
                    if i % 50 == 0 or i == len(futures):
                        rate = i / max(time.perf_counter() - t0, 1e-6)
                        print(f"[{spec.name}] {i}/{len(futures)} "
                              f"({rate:.2f} runs/s, err={errors})", flush=True)

    summary = {
        "experiment": spec.name,
        "total": len(grid),
        "done_before": done_before,
        "launched": len(pending),
        "hits_this_run": hits,
        "errors": errors,
        "timeouts": timeouts,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "csv": _display_path(csv_path),
        "data_root": _display_path(out_dir),
        "dry_run": False,
    }
    # 实验配置快照（覆盖写入，便于归档实验定义与冻结身份）
    (out_dir / "config.yaml").write_text(
        f"# 由 batch_runner 自动生成\n# experiment: {spec.name}\n"
        f"# grid_size: {len(grid)}\n# report_ref: {spec.report_ref}\n"
        f"# identity: {json.dumps(spec.identity.as_dict(), ensure_ascii=False)}\n"
        f"# timeout_s: {default_timeout:g}\n"
        f"# config_ids: {sorted({_effective_config_id(p) for p in grid})}\n",
        encoding="utf-8")
    print(f"[{spec.name}] 完成: {json.dumps(summary, ensure_ascii=False)}",
          flush=True)
    return summary


def _row_identity_key(experiment: str, row: Mapping[str, Any]) -> ResumeKey:
    """从新格式 CSV 行构造 resume 键。"""
    identity = _row_identity(row)
    return ResumeKey(
        experiment=experiment,
        robot_id=identity.robot_id,
        mode=identity.mode,
        racket_id=identity.racket_id,
        config_hash=identity.config_hash,
        metric_schema=identity.metric_schema,
        config_id=str(row.get("config_id") or ""),
        seed=int(row["seed"]) if str(row.get("seed", "")).strip() else -1,
    )


def main() -> None:
    """CLI 入口: 按实验名执行（矩阵定义见 overnight_experiments.py）。"""
    from scripts.exp.overnight_experiments import EXPERIMENTS

    parser = argparse.ArgumentParser(description="通用并行批量实验执行器")
    parser.add_argument("experiment", choices=sorted(EXPERIMENTS),
                        help="实验名（见 overnight_experiments.py）")
    parser.add_argument("--workers", type=int, default=16, help="并行进程数")
    parser.add_argument("--no-resume", action="store_true",
                        help="忽略已有 results.csv 重跑全部")
    parser.add_argument("--legacy-resume", action="store_true",
                        help="允许续跑旧格式（无身份列）CSV")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个待完成任务（smoke test 用）")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="数据根目录（默认 <repo>/experiment_data）")
    parser.add_argument("--output", type=Path, default=None,
                        help="本实验输出目录（默认 <data-root>/<experiment>）")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S,
                        help=f"单 run 超时秒数（默认 {DEFAULT_TIMEOUT_S}）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印任务命令与身份，不执行/不落盘")
    args = parser.parse_args()

    try:
        run_batch(
            EXPERIMENTS[args.experiment],
            workers=args.workers,
            resume=not args.no_resume,
            limit=args.limit,
            data_root=args.data_root,
            output=args.output,
            timeout_s=args.timeout,
            dry_run=args.dry_run,
            allow_legacy_resume=args.legacy_resume,
        )
    except (FileNotFoundError, ValueError) as exc:
        # 延迟网格依赖缺失/身份混写等属于可诊断的输入错误：打印清楚并
        # 以非 0 退出，绝不静默退回旧 data-root（T8a 评审 B1）。
        print(f"[batch_runner] 错误: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
