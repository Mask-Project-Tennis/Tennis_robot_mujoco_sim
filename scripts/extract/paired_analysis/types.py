"""统计分析的共享类型、常量与纯解析原语（本包叶模块）。

本模块只依赖标准库：其它子模块（identity/sources/dataset/metrics/comparisons/
report/cli）均自这里取常量、dataclass 与异常，保证依赖方向单一。
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

SCHEMA = "paper_stats_v2"


STATS_FILENAMES = ("06-stats-active.json", "paper_stats_active.json")



MODES = ("full", "tube_only", "softmin_only", "none")


REFERENCE = "none"  # 点目标基线（消融档名 none）


NOFEEDFORWARD_SUFFIX = "_nofeedforward"  # 前馈关闭变体档后缀（M02：不进 nominal 四档）



# 执行模式（F04）：同一次统计只允许一种模式进入同一个样本池。
VALID_MODES = ("torque", "position")


MODE_CHOICES = (*VALID_MODES, "both")


MULTI_MODE_SCHEMA = "paired_stats_multi_mode_v1"


MULTI_MODE_FILENAME = "modes.json"


SEED_DIR_RE = re.compile(r"^seed(?P<seed>-?\d+)$")



# paper/figures/table_data/ 现状表格定义（只读对照；文件名与内容错位的说明见 tables 内注释）
TABLE3_CELLS = ((0.0, 0.1), (0.0, 0.2), (0.0, 0.3), (25.0, 0.2), (50.0, 0.2))


TIER_LABELS = {
    "full": "full",
    "tube_only": "tube-only",
    "softmin_only": "candidate-set-only",
    "none": "point-target",
}


HIT_SEMANTICS = {
    "legacy": {
        "rule": '旧口径 hit_type == "active"（passive 单独计数，不计入 active）',
        "source": "results.csv 的 hit_type 列（RM65 历史 CSV / R1D batch CSV）",
    },
    "physical": {
        "rule": "v12_episode_v2 hit.physical_hit_type == \"active\""
                "（physical_hit_type=active 口径：首次物理拍面接触且拍速 > 0.3 m/s；"
                "passive/miss 一律不计命中）",
        "source": "v12_episode.json 的 hit.physical_hit_type 优先、"
                  "hit.physical_hit 仅作缺失兜底（经 runs.csv 汇总）",
    },
}


def _repo_root() -> Path:
    """仓库根：``paired_analysis`` 位于 ``scripts/extract/`` 下，上溯 3 层。"""
    return Path(__file__).resolve().parents[3]


DATA = _repo_root() / "experiment_data"
# 兼容旧常量名：默认实验数据根与 DATA 同义（cli.build_parser 的 help 文本继续用它）。
DEFAULT_DATA_ROOT = DATA




def _to_bool_or_none(raw: Any) -> bool | None:
    """宽松布尔解析；无法判定返回 None（禁止把缺失当 False）。"""
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower() if raw is not None else ""
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no"):
        return False
    return None




def cfg_of(row: dict[str, str]) -> dict[str, Any]:
    """解析 config_json。"""
    return json.loads(row["config_json"])




def is_error(row: dict[str, str]) -> bool:
    """该行是否被排除（error 非空，实测均为 no_result）。"""
    return bool((row.get("error") or "").strip())




# ----------------------------------------------------------------------------
# 运行上下文：数据根 / run_root / catalog / 执行模式（F03 / F04）
# ----------------------------------------------------------------------------

class StatsError(RuntimeError):
    """统计链路的可预期错误（CLI 以退出码 2 报告，绝不静默回退）。"""




class ModeSelectionError(StatsError):
    """执行模式无法唯一确定，或混合模式未显式选择。"""




class DuplicateRecordError(StatsError):
    """单一来源内出现同身份重复记录（禁止静默去重）。"""




class IdentityError(StatsError):
    """robot/racket/来源身份不一致，禁止把不同身份的记录配对。"""




class DatasetConflictError(StatsError):
    """同一记录身份命中多行富化记录且归一化后不一致（禁止静默择一）。"""




@dataclass(frozen=True)
class StatsContext:
    """不可变运行上下文：逐层显式传递，不用进程全局变量或 monkey-patch。"""

    data_root: Path
    run_root: Path
    catalog: Any = None
    catalog_path: Path | None = None
    mode: str | None = None
    source_sets: Mapping[str, str] = field(default_factory=dict)
    synthetic: bool = False
    manifest_path: Path | None = None

    def with_mode(self, mode: str | None) -> StatsContext:
        """派生一个只改执行模式的上下文（其余字段保持不可变）。"""
        return replace(self, mode=mode)




@dataclass(frozen=True)
class RecordIdentity:
    """记录身份：唯一键与跨视图连接键的唯一定义（F04）。

    ``unique_key`` 额外包含分层（tier）、设计指纹与来源文件：旧 RM65 数据里同一个
    seed 会按 ``--ablation``/扰动档位重复出现，若只按连接键查重会把合法的不同设计点
    误判为重复；而完全相同的记录（同层同配置同 seed 同来源）仍会被判为重复并报错。
    """

    experiment: str
    case: str
    config_id: str
    seed: int
    mode: str
    robot_id: str
    racket_id: str
    source: str
    tier: str = ""
    design: str = ""

    def join_key(self) -> tuple[str, str, str, int, str, str, str]:
        """跨视图连接键：不含来源文件，只认同一条科学记录。"""
        return (self.experiment, self.case, self.config_id, self.seed,
                self.mode, self.robot_id, self.racket_id)

    def unique_key(self) -> tuple[Any, ...]:
        """单一来源内的唯一键：连接键 + 分层 + 设计指纹 + 来源身份。"""
        return (*self.join_key(), self.tier, self.design, self.source)




@dataclass(frozen=True)
class PhysicalIndex:
    """runs.csv 富化行索引（M03）。

    Attributes:
        by_identity: ``RecordIdentity.join_key()`` → 同身份 runs 行列表。
        by_artifact: 归一化 artifact 路径 → runs 行列表（只服务 legacy 数据）。
        path: 实际读取的 runs.csv 路径（缺失为 None）。
    """

    by_identity: dict[tuple[Any, ...], list[dict[str, str]]] = field(default_factory=dict)
    by_artifact: dict[str, list[dict[str, str]]] = field(default_factory=dict)
    path: Path | None = None




class HitPolicy:
    """命中口径：默认只计 active，可选把 passive 一并计入。"""

    def __init__(self, include_passive: bool = False) -> None:
        self.include_passive = include_passive

    def __call__(self, row: dict[str, str]) -> int:
        if is_error(row):
            return -1
        ht = (row.get("hit_type") or "").strip()
        if ht == "active":
            return 1
        if ht == "passive":
            return 1 if self.include_passive else 0
        return 0


__all__ = [
    "DATA",
    "DEFAULT_DATA_ROOT",
    "HIT_SEMANTICS",
    "MODES",
    "MODE_CHOICES",
    "MULTI_MODE_FILENAME",
    "MULTI_MODE_SCHEMA",
    "NOFEEDFORWARD_SUFFIX",
    "REFERENCE",
    "SCHEMA",
    "SEED_DIR_RE",
    "STATS_FILENAMES",
    "TABLE3_CELLS",
    "TIER_LABELS",
    "VALID_MODES",
    "DatasetConflictError",
    "DuplicateRecordError",
    "HitPolicy",
    "IdentityError",
    "ModeSelectionError",
    "PhysicalIndex",
    "RecordIdentity",
    "StatsContext",
    "StatsError",
    "_repo_root",
    "_to_bool_or_none",
    "cfg_of",
    "is_error",
]
