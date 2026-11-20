#!/usr/bin/env python3
"""论文统计口径统一提取：active-hit 主指标 + same-seed 配对检验（RM65 / R1D 同源）。

背景（2026-09-12 审稿意见核实后确立的口径）：
1. 主指标改为 **active-hit rate**：`hit_type == "active"`（击球瞬间球拍面速度 > 0.3 m/s）；
   被动接触（0 < v_r <= 0.3 m/s）单独计数、不进入主指标；
2. 全部实验为 same-seed 配对设计（同 seed 同发球同扰动交给各档），
   因此显著性检验改用 **配对 McNemar 精确检验** + **配对 bootstrap 95% CI**，
   取代此前提取脚本中的两独立样本比例 z 检验（忽略配对，口径偏保守）；
3. 排除行（`error` 非空，均为 `no_result`）逐档计数并报告，验证排除与方法无关；
4. 限速层 × 机制档的交互效应用 **配对 DiD**（每 seed 的增益差）报告。

命中语义（T8b 起显式区分，二者都报告、互不替代）：
- **legacy**：旧口径 `hit_type == "active"`（RM65 历史 CSV，亦为 R1D batch CSV 的
  `hit_type`；`passive` 不计入 active；可 `--include-passive` 复现旧合并口径）；
- **physical**：R1D v2 产物口径 `v12_episode.json` 的 `hit.physical_hit_type ==
  "active"`（首次物理拍面接触且拍速 > 0.3 m/s；`passive`/`miss` 不计命中），
  由 T8a 汇总进 `<run_root>/runs.csv` 的 `physical_hit` / `physical_hit_type` 列；
  `physical_hit` 布尔仅在类型缺失/未知时兜底。旧 RM65 根没有该列 → physical
  一律 `rate=null` + `unavailable_reason`，**不伪造 0.0**，也不拿 legacy 冒充
  physical；有判定但全 miss 时 `rate=0.0` 且标 `all_miss`（0/N 是测得的 0%）。

数据根与输出（T8b / F03 / F04）：
- `--data-root` 指定实验数据根（默认 `experiment_data/`，旧行为不变）；
- `--catalog` 指定 `run_catalog` 产出的 `catalog.json`：据此解析真实 artifact 位置
  （output 与 data-root 分离、跨目录时不靠猜数据根）；显式坏 catalog 直接退出码 2，
  **不回退**旧的数据根推断；无 catalog 的旧 RM65 根保持旧行为；
- `--mode torque|position|both` 指定执行模式：同一次统计只允许一种模式进入同一个
  样本池（torque/position 合成会掩盖两模式各自的命中率）；无 `--mode` 时仅在数据
  恰好只有一种模式时自动推断，混合模式直接非零退出并要求显式选择；
  `--mode both --output <dir>` 分别写 `<dir>/torque/` 与 `<dir>/position/` 下的
  `06-stats-active.json` / `paper_stats_active.json`，并写 `<dir>/modes.json`
  （schema `paired_stats_multi_mode_v1`，只作索引与验证状态，不重复统计内容）；
- `--output` 指定输出目录，默认 `<data_root>/../stats/`（不再硬写 `paper/planning/`）；
  单模式时写出 `06-stats-active.json` 与 `paper_stats_active.json` 两份内容相同的产物；
- `--dry-run` 只打印不写盘（写盘前必先打印目标路径）；
- `E1`–`E8` 顶层键与字段保持旧定义不变（旧绘图键兼容），新增内容一律追加在
  `hit_semantics` / `physical` / `E9_shared_input` / `tables` / `data_root` /
  `unavailable` 等新顶层键下，避免旧消费者把新键误当数据键。

用法:
    python scripts/extract/paired_stats.py [--include-passive]
    python scripts/extract/paired_stats.py --data-root <new_root>/batch \\
        --output results/.../stats
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Hashable, Mapping

PROJECT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = PROJECT / "experiment_data"
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


# ----------------------------------------------------------------------------
# 基础统计工具
# ----------------------------------------------------------------------------

def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson 95% 置信区间（返回百分比 pp）。"""
    if n == 0:
        return 0.0, 0.0
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return 100 * (c - h) / d, 100 * (c + h) / d


def mcnemar_exact(b: int, c: int) -> float:
    """配对 McNemar 精确检验（双侧，二项分布）。

    Args:
        b: A 命中且 B 未命中的配对数。
        c: A 未命中且 B 命中的配对数。

    Returns:
        双侧 p 值。
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(2 * tail, 1.0)


def paired_bootstrap(
    hits_a: list[int], hits_b: list[int], n_boot: int = 20000, seed: int = 0
) -> tuple[float, float]:
    """配对 bootstrap：对 seed 重采样，返回命中率差的 95% 分位区间（pp）。"""
    rng = random.Random(seed)
    n = len(hits_a)
    diffs = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        pa = sum(hits_a[i] for i in idx)
        pb = sum(hits_b[i] for i in idx)
        diffs.append((pa - pb) / n * 100)
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]


def sign_test(dids: list[int]) -> float:
    """非零配对差上的符号检验（DiD 用）。"""
    nz = [d for d in dids if d != 0]
    if not nz:
        return 1.0
    pos = sum(1 for d in nz if d > 0)
    k = min(pos, len(nz) - pos)
    tail = sum(math.comb(len(nz), i) for i in range(k + 1)) / 2 ** len(nz)
    return min(2 * tail, 1.0)


# ----------------------------------------------------------------------------
# 配置读取：RM65 旧 flag 与 R1D `--task-overrides` 的统一访问器
# ----------------------------------------------------------------------------

def _parse_override_value(raw: str) -> Any:
    """把 R1D `--task-overrides` 的字符串值解析成 bool/int/float/str。"""
    text = raw.strip()
    low = text.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def overrides_of(cfg: dict[str, Any]) -> dict[str, Any]:
    """解析 R1D `--task-overrides` 列表为 {dotted_key: value}。"""
    raw = cfg.get("--task-overrides") or []
    out: dict[str, Any] = {}
    if isinstance(raw, str):
        raw = [raw]
    for item in raw:
        key, sep, value = str(item).partition("=")
        if sep:
            out[key.strip()] = _parse_override_value(value)
    return out


def _ov(ov: dict[str, Any], key: str, default: Any) -> Any:
    """取 override 值，缺省给默认值。"""
    value = ov.get(key, default)
    return default if value is None else value


def _is_r1d_row(cfg: dict[str, Any]) -> bool:
    """判断一行是否属于 R1D 实验（R04：默认档识别的前置条件）。

    R1D 行至少带 ``--robot-config``/``--runtime-config`` 指向 ``r1d_*`` 配置，
    或显式给出 ``--sim-task-config``；RM65 行两者都没有。

    Args:
        cfg: 由 ``config_json`` 解析出的参数字典。

    Returns:
        True 表示 R1D 行；False 表示 RM65 行或无法判定。
    """
    if cfg.get("--sim-task-config"):
        return True
    for key in ("--robot-config", "--runtime-config"):
        text = str(cfg.get(key) or "").replace("\\", "/").lower()
        if "r1d" in text:
            return True
    return False


def _feedforward_of_cfg(cfg: dict[str, Any]) -> bool | None:
    """从参数字典判断位置模式前馈是否被显式关闭。

    优先级必须与真实 V12 配置解析一致（``src/sim/v12_config.py``：任务覆盖先写入
    草稿，随后 ``_apply_cli`` 里的 ``--no-feedforward`` 再覆盖，故 CLI 旗标优先）：

    1. CLI 开关 ``--no-feedforward``（``store_true``，出现即表示关闭；显式
       ``false``/``0`` 视为未关闭）——它覆盖 ``--task-overrides``（R07）；
    2. CLI 开关 ``--feedforward`` / ``--enable-feedforward``（显式正/负值）；
    3. ``--task-overrides`` 里的 ``actuator.feedforward``（实际 argv 为
       ``["actuator.feedforward=false"]``，也兼容 YAML 风格 ``False``/``0``）。

    Args:
        cfg: 由 ``config_json`` 解析出的参数字典。

    Returns:
        False 表示显式关闭前馈；True 表示显式开启；None 表示未声明。
    """
    if "--no-feedforward" in cfg:
        flag = _to_bool_or_none(cfg.get("--no-feedforward"))
        # 开关缺值（argparse store_true）视为启用；显式 false/0 表示未关闭前馈
        return False if flag is None or flag else True
    for key in ("--feedforward", "--enable-feedforward"):
        if key in cfg:
            value = _to_bool_or_none(cfg.get(key))
            if value is not None:
                return value
    ov = overrides_of(cfg)
    if "actuator.feedforward" in ov:
        value = _to_bool_or_none(ov["actuator.feedforward"])
        if value is not None:
            return value
    return None


def _feedforward_of(row: dict[str, str]) -> bool | None:
    """判断一行是否为“前馈关闭”运行（M02）。

    Args:
        row: results/runs 视图的一行。

    Returns:
        False 表示显式关闭前馈；True 表示显式开启；None 表示未声明
        （未声明的行沿用普通四档语义）。
    """
    return _feedforward_of_cfg(cfg_of(row))


def _base_tier_of_cfg(cfg: dict[str, Any]) -> str:
    """按 tube/softmin 开关推导基础消融档（不含前馈变体后缀）。"""
    tier = cfg.get("--ablation")
    if tier:
        return str(tier)
    ov = overrides_of(cfg)
    if not ov:
        return "full" if _is_r1d_row(cfg) else ""
    q_n = float(_ov(ov, "cost.tube.Q_n_tube", 1.0) or 0.0)
    q_p = float(_ov(ov, "cost.tube.Q_p_tube", 1.0) or 0.0)
    q_v = float(_ov(ov, "cost.tube.Q_v_tube", 1.0) or 0.0)
    tube_off = q_n == 0.0 and q_p == 0.0 and q_v == 0.0
    softmin_on = bool(_ov(ov, "cost.tube.use_softmin_terminal", True))
    if tube_off and softmin_on:
        return "softmin_only"
    if tube_off and not softmin_on:
        return "none"
    if not tube_off and not softmin_on:
        return "tube_only"
    return "full"


def tier_of(row: dict[str, str]) -> str:
    """消融档：RM65 读 `--ablation`；R1D 从 `--task-overrides` 的 tube 开关推导。

    R1D 行未给 `--task-overrides` 时等价于四个开关全开（默认 ``full`` 档）；
    旧实现此时返回空串，导致默认 full 档在表格/配对里整体缺失（R04）。
    RM65 行无 overrides 仍返回空串（不臆造档位），显式 `--ablation` 优先。

    M02：基础档属于四档且显式关闭前馈时返回 ``f"{base}_nofeedforward"``
    （例如 ``full_position_nofeedforward`` 行为 ``full_nofeedforward``），
    普通四档的取值与语义完全不变。
    """
    cfg = cfg_of(row)
    base = _base_tier_of_cfg(cfg)
    if base in MODES and _feedforward_of_cfg(cfg) is False:
        return f"{base}{NOFEEDFORWARD_SUFFIX}"
    return base


def speed_of(row: dict[str, str]) -> int:
    """球速（m/s）；缺失按 0 处理（与旧实现分组行为一致）。"""
    value = cfg_of(row).get("--ball-speed")
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def time_perturb_ms_of(row: dict[str, str]) -> int:
    """时间扰动（ms）：RM65 `--time-perturb-ms`，R1D `mpc.time_perturb_s` 换算。"""
    cfg = cfg_of(row)
    if "--time-perturb-ms" in cfg:
        return int(cfg.get("--time-perturb-ms") or 0)
    ov = overrides_of(cfg)
    if "mpc.time_perturb_s" in ov:
        return int(round(float(ov["mpc.time_perturb_s"]) * 1000.0))
    return 0


def space_perturb_of(row: dict[str, str]) -> float:
    """空间扰动（m）：RM65 `--space-perturb-m`，R1D `mpc.space_perturb_m`。"""
    cfg = cfg_of(row)
    if "--space-perturb-m" in cfg:
        return float(cfg.get("--space-perturb-m") or 0.0)
    ov = overrides_of(cfg)
    if "mpc.space_perturb_m" in ov:
        return float(ov["mpc.space_perturb_m"])
    return 0.0


def _tcp_label(value: float | int | str) -> str:
    """统一 TCP 限速标签：固定 1 位小数（R05：1.0 → ``TCP1.0`` 而非 ``TCP1``）。"""
    return f"TCP{float(value):.1f}"


def tcp_label_of(row: dict[str, str]) -> str:
    """限速层标签：TCP1.0 / TCP1.4 / TCP1.8（RM65 limits-config / max-tcp；R1D runtime 文件名）。

    三种来源统一走 ``_tcp_label``，避免 ``:g`` 把 1.0 格式化成 ``TCP1``。
    """
    cfg = cfg_of(row)
    limits = cfg.get("--limits-config")
    if limits:
        return _tcp_label(1.0)
    max_tcp = cfg.get("--max-tcp")
    if max_tcp is not None:
        return _tcp_label(max_tcp)
    runtime = str(cfg.get("--runtime-config") or "")
    match = re.search(r"tcp(\d+)", runtime)
    if match:
        return _tcp_label(int(match.group(1)) / 10)
    return "TCP1.8"


def _obs_freq_of(row: dict[str, str]) -> str:
    """观测频率标签（E5 obsfreq）。"""
    return str(cfg_of(row).get("--obs-freq", "?"))


def shared_input_of(row: dict[str, str]) -> bool:
    """是否启用 shared-input 协议（RM65 旧 flag / R1D override）。"""
    cfg = cfg_of(row)
    if "--shared-input-full" in cfg:
        return bool(cfg.get("--shared-input-full"))
    return bool(overrides_of(cfg).get("mpc.shared_input_full", False))


# ----------------------------------------------------------------------------
# 数据加载（含 R1D runs.csv 的 physical 列富化）
# ----------------------------------------------------------------------------

def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    """读 CSV 为 dict 列表（缺失返回空）。"""
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


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


def _catalog_module() -> Any:
    """延迟导入 ``scripts.exp.run_catalog``（catalog 契约的唯一事实来源）。"""
    if str(PROJECT) not in sys.path:
        sys.path.insert(0, str(PROJECT))
    from scripts.exp import run_catalog

    return run_catalog


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

    def with_mode(self, mode: str | None) -> "StatsContext":
        """派生一个只改执行模式的上下文（其余字段保持不可变）。"""
        return replace(self, mode=mode)


def make_context(
    data_root: Path | None = None,
    *,
    catalog_path: Path | None = None,
    mode: str | None = None,
) -> StatsContext:
    """构造统计上下文。

    Args:
        data_root: 实验数据根；为 None 时用 catalog 声明的 data_root，再退回默认根。
        catalog_path: 可选 ``catalog.json``；显式给出且加载失败时直接抛 StatsError。
        mode: ``torque`` / ``position`` / None（自动推断）。

    Returns:
        不可变 StatsContext。

    Raises:
        StatsError: catalog 非法或 mode 取值未知。
    """
    if mode is not None and mode not in VALID_MODES:
        raise StatsError(f"未知执行模式 {mode!r}：只接受 {' / '.join(VALID_MODES)}")
    catalog = None
    if catalog_path is not None:
        api = _catalog_module()
        try:
            catalog = api.load_catalog(Path(catalog_path))
        except api.CatalogError as exc:
            raise StatsError(f"catalog 加载失败（{catalog_path}）：{exc}") from exc
    if catalog is not None:
        root = Path(data_root) if data_root is not None else catalog.data_root
        return StatsContext(
            data_root=Path(root),
            run_root=Path(catalog.run_root),
            catalog=catalog,
            catalog_path=Path(catalog.catalog_path),
            mode=mode,
            source_sets=dict(catalog.source_sets),
            synthetic=bool(catalog.synthetic),
            manifest_path=Path(catalog.manifest_path),
        )
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    return StatsContext(data_root=root, run_root=root.parent, mode=mode)


def _row_artifact_value(row: dict[str, str]) -> str:
    """行的产物目录字符串：artifact_dir 列优先，否则 config_json 的 --output。"""
    explicit = (row.get("artifact_dir") or "").strip()
    if explicit:
        return explicit
    try:
        output = cfg_of(row).get("--output")
    except (json.JSONDecodeError, KeyError, TypeError):
        return ""
    return str(output or "")


def _normalize_path(value: str, base: Path | None) -> str:
    """把产物路径归一化为可比较键（相对 base 解析、绝对原样、大小写归一）。"""
    if not value:
        return ""
    path = Path(value)
    if base is not None and not path.is_absolute():
        path = base / path
    try:
        resolved = path.resolve()
    except OSError:  # pragma: no cover - 极端路径错误
        resolved = path
    return os.path.normcase(str(resolved))


def _artifact_key(row: dict[str, str], ctx: StatsContext | None = None) -> str:
    """artifact 身份键：跨 output / data-root 两个根都能对上同一条记录。"""
    base = None if ctx is None else ctx.run_root
    return _normalize_path(_row_artifact_value(row), base)


def case_of(row: dict[str, str]) -> str:
    """记录所属 case：从产物路径 ``<experiment>/<case>/seed<N>`` 推断，未知返回空串。"""
    value = _row_artifact_value(row)
    if not value:
        return ""
    parts = Path(value).parts
    for index in range(len(parts) - 1, 1, -1):
        if SEED_DIR_RE.match(parts[index]):
            return parts[index - 1]
    return ""


def mode_of(row: dict[str, str]) -> str | None:
    """记录的执行模式：显式 mode 列 > config_json 证据 > 旧 RM65 默认 torque。

    R1D 行缺 ``mode`` 时返回 None（不把未知默认成 torque）；旧 RM65 行没有位置模式
    证据时按项目默认口径（力矩模式）计为 ``torque``，从而保持旧数据根可统计。
    """
    explicit = (row.get("mode") or "").strip().lower()
    if explicit in VALID_MODES:
        return explicit
    try:
        cfg = cfg_of(row)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    for key in ("--actuator-mode", "actuator.mode", "actuator_mode"):
        value = str(cfg.get(key) or "").strip().lower()
        if value in VALID_MODES:
            return value
    if cfg.get("--position-mode") not in (None, False, "", "false", "False"):
        return "position"
    if _is_r1d_row(cfg):
        return None
    return "torque"


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


def _seed_value(row: Mapping[str, Any]) -> int:
    """宽松解析 seed；缺失/非法返回 -1（不臆造 seed）。"""
    try:
        return int(str(row.get("seed") or "").strip())
    except (TypeError, ValueError):
        return -1


_RUNTIME_CONFIG_KEYS = frozenset({"--output", "--seed", "_config_hash", "_timeout", "--log-level"})

#: 仅表示"从哪个文件读配置"的路径键：整树搬移后绝对/相对写法不同，
#: 不能据此判为设计冲突；归一化为文件名保留配置身份（config_id/哈希另证内容）。
_PATH_CONFIG_KEYS = frozenset({
    "--robot-config",
    "--runtime-config",
    "--racket-config",
    "--limits-config",
    "--sim-task-config",
    "--scenario-config",
})


def _normalize_config_path(value: Any) -> Any:
    """把路径型配置值归一化为文件名（非字符串原样返回）。"""
    if not isinstance(value, str) or not value.strip():
        return value
    return Path(value.replace("\\", "/")).name


def _design_fingerprint(row: Mapping[str, Any]) -> str:
    """配置设计指纹：去掉运行期键并归一化路径键后的规范化 config_json。"""
    try:
        cfg = json.loads(row.get("config_json") or "{}")
    except (json.JSONDecodeError, TypeError):
        return ""
    if not isinstance(cfg, dict):
        return ""
    cleaned = {
        key: (_normalize_config_path(value) if key in _PATH_CONFIG_KEYS else value)
        for key, value in cfg.items()
        if key not in _RUNTIME_CONFIG_KEYS
    }
    return json.dumps(cleaned, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def record_identity(row: dict[str, str], *, source: str = "") -> RecordIdentity:
    """从一行记录构造不可变身份对象（缺失字段留空，缺失 seed 记 -1）。"""
    return RecordIdentity(
        experiment=(row.get("experiment") or "").strip(),
        case=case_of(row),
        config_id=(row.get("config_id") or "").strip(),
        seed=_seed_value(row),
        mode=mode_of(row) or "",
        robot_id=(row.get("robot_id") or "").strip(),
        racket_id=(row.get("racket_id") or "").strip(),
        source=source,
        tier=tier_of(row),
        design=_design_fingerprint(row),
    )


def _is_completed_row(row: Mapping[str, Any]) -> bool:
    """该行是否代表一次已完成的执行尝试。

    续跑语义（P2）：batch_runner 失败后允许 ``--resume``，原失败行保留、
    成功行追加，两行共享同一 ``RecordIdentity.unique_key()``。因此同一身份
    允许保留多条失败尝试（``error`` 非空或 ``timed_out``），但完成尝试至多
    一条。``timed_out`` 为真时即使 ``error`` 为空也不算完成（生产超时终止）。

    Args:
        row: results.csv / runs.csv 的一行。

    Returns:
        True 表示该行是一次完成尝试。
    """
    if str(row.get("error") or "").strip():
        return False
    return _to_bool_or_none(row.get("timed_out")) is not True


def _group_attempts(
    rows: list[dict[str, str]], path: Path
) -> dict[tuple[Any, ...], list[tuple[int, dict[str, str]]]]:
    """按单一来源内的完整身份归并执行尝试。

    Args:
        rows: 同一 CSV 的全部数据行。
        path: 行所属文件（参与身份构造，与既有查重口径一致）。

    Returns:
        身份键 → ``[(行号, 行)]``；行号从 2 起，与 CSV 表头占第 1 行对齐。
    """
    groups: dict[tuple[Any, ...], list[tuple[int, dict[str, str]]]] = {}
    for line, row in enumerate(rows, start=2):
        key = record_identity(row, source=str(path)).unique_key()
        groups.setdefault(key, []).append((line, row))
    return groups


def _check_attempts(rows: list[dict[str, str]], path: Path) -> None:
    """校验执行尝试契约：同一身份至多一条完成尝试，失败尝试可多条。

    Args:
        rows: 同一 CSV 的全部数据行。
        path: 行所属文件（错误信息用）。

    Raises:
        DuplicateRecordError: 同一身份出现两条及以上完成尝试。续跑恢复只允许
            一条完成行；两条完成行仍是真正的重复记录，按既有契约拒绝。
    """
    for key, attempts in _group_attempts(rows, path).items():
        completed = [line for line, row in attempts if _is_completed_row(row)]
        if len(completed) > 1:
            raise DuplicateRecordError(
                f"{path.name} 内出现同身份重复完成记录（第 {completed[0]} 行与"
                f"第 {completed[1]} 行）：{key}；同一身份允许多条失败尝试但至多"
                "一条完成尝试，按契约禁止静默去重，请先修正数据或拆分数据根"
            )


def _final_attempt_rows(
    rows: list[dict[str, str]], path: Path
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """为每个身份选择最终执行尝试行，作为统计样本。

    选择规则：有完成尝试则取该完成行；没有完成尝试则保留**最后一条失败行**
    （供 exclusion 统计与“无对照”报告使用）。其余早期尝试行只从统计视图剔除，
    仍保留在 CSV 原文件中，并在此处以 ``ignored`` 摘要返回，禁止静默丢弃文件行。

    Args:
        rows: 同一 CSV 的全部数据行。
        path: 行所属文件（身份构造与错误信息用）。

    Returns:
        ``(selected, ignored)``：``selected`` 保持文件行序；``ignored`` 每项含
        ``line`` / ``experiment`` / ``case`` / ``config_id`` / ``seed`` /
        ``mode`` / ``reason``，按行号升序。

    Raises:
        DuplicateRecordError: 同一身份出现两条及以上完成尝试（经
            :func:`_check_attempts` 判定）。
    """
    _check_attempts(rows, path)
    selected_lines: set[int] = set()
    ignored: list[dict[str, Any]] = []
    for attempts in _group_attempts(rows, path).values():
        completed = [line for line, row in attempts if _is_completed_row(row)]
        chosen_line = completed[0] if completed else attempts[-1][0]
        selected_lines.add(chosen_line)
        for line, row in attempts:
            if line == chosen_line:
                continue
            ignored.append({
                "line": line,
                "experiment": (row.get("experiment") or "").strip(),
                "case": case_of(row),
                "config_id": (row.get("config_id") or "").strip(),
                "seed": _seed_value(row),
                "mode": mode_of(row) or "",
                "reason": (
                    "superseded_completed_attempt"
                    if _is_completed_row(row)
                    else "superseded_failed_attempt"
                ),
            })
    selected = [row for line, row in enumerate(rows, start=2) if line in selected_lines]
    ignored.sort(key=lambda item: int(item["line"]))
    return selected, ignored


def _runs_csv_paths(ctx: StatsContext) -> list[Path]:
    """候选 runs.csv：catalog 声明优先，否则数据根下 / 数据根同级。"""
    if ctx.catalog is not None:
        return [Path(ctx.catalog.runs_csv)]
    return [ctx.data_root / "runs.csv", ctx.data_root.parent / "runs.csv"]


def _runs_csv_path(ctx: StatsContext) -> Path | None:
    """实际存在的 runs.csv（缺失返回 None）。"""
    for path in _runs_csv_paths(ctx):
        if path.is_file():
            return path
    return None


# 富化列：results 视图缺失时由 runs 视图补齐（顺序即写回顺序）
_PHYSICAL_COLUMNS = (
    "physical_hit",
    "physical_hit_type",
    "physical_hit_step",
    "contact_event_count",
    "failure_reason",
    "backend",
)

# 共有结果列（R05）：producer 会把 results.csv 的 legacy 结果、完成状态与错误字段
# 原样复制进 runs.csv，因此同一运行的两份视图里这些同名列必须一致。它们只参与
# 一致性比较（富化签名/冲突报告），**不参与富化填空**——填空仍只认
# ``_PHYSICAL_COLUMNS``。这里只比较“同名对同名”，不跨越 legacy 与 physical 口径。
_SHARED_RESULT_COLUMNS = (
    "hit",
    "hit_type",
    "physical_hit",
    "physical_hit_type",
    "physical_hit_step",
    "contact_event_count",
    "failure_reason",
    "backend",
    "status",
    "error",
    "timed_out",
    "exit_code",
)

# 完成状态列（R05）：这些列描述“这次运行是否成功”，空值本身也是信息。
# 例如 results.csv 的 error 为空、runs.csv 却记为非空错误，说明同一身份的两份
# 视图对完成状态说法相反，必须判冲突，不能按“空值=未声明”放过。
_STATE_RESULT_COLUMNS = ("status", "error", "timed_out", "exit_code")


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


def _normalized_column(row: Mapping[str, Any], column: str) -> str:
    """富化列的可比取值：去空白，并把布尔字面量统一成小写。"""
    raw = row.get(column)
    if raw is None:
        return ""
    text = str(raw).strip()
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered
    return text


def _identity_text(row: dict[str, str], column: str) -> str:
    """身份列的可比取值（``mode`` 走 ``mode_of`` 推断，其余直接读列）。"""
    if column == "mode":
        return mode_of(row) or ""
    return str(row.get(column) or "").strip()


def _enrichment_signature(row: dict[str, str]) -> tuple[str, ...]:
    """富化行归一化签名：设计指纹 + 身份列 + 共有结果列。

    ``_SHARED_RESULT_COLUMNS`` 已包含 ``_PHYSICAL_COLUMNS``，故签名同时覆盖
    physical 丰富列与 legacy 共有列；两份视图同名列不一致即视为不同签名。
    """
    return (
        _design_fingerprint(row),
        _identity_text(row, "mode"),
        _identity_text(row, "robot_id"),
        _identity_text(row, "racket_id"),
        _identity_text(row, "config_hash"),
        *(_normalized_column(row, column) for column in _SHARED_RESULT_COLUMNS),
    )


def _enrichment_field_names() -> tuple[str, ...]:
    """``_enrichment_signature`` 各分量的字段名（错误信息用）。"""
    return (
        "design",
        "mode",
        "robot_id",
        "racket_id",
        "config_hash",
        *_SHARED_RESULT_COLUMNS,
    )


def _enrichment_conflicts(a: dict[str, str], b: dict[str, str]) -> list[str]:
    """比较两行富化相关字段；结果列空值视为未声明，完成状态列不适用该规则。

    比较范围 = 设计指纹 + 身份列 + ``_SHARED_RESULT_COLUMNS``（同名对同名）。
    ``hit``/``hit_type`` 等结果列必须与 physical 列分开看：这里只要求同一
    运行的两份视图对同名列说法一致，不要求 legacy 命中与 physical 命中相等。

    ``_STATE_RESULT_COLUMNS`` 是例外：``results.csv`` 的 error/exit_code 为空、
    ``runs.csv`` 却记为失败时，两份视图对“是否完成”说法相反，空值也必须判冲突
    （R05：单独改完成状态或错误不得被静默放过）。
    """
    fields: list[str] = []
    if _design_fingerprint(a) != _design_fingerprint(b):
        fields.append("design")
    for column in ("mode", "robot_id", "racket_id", "config_hash"):
        left, right = _identity_text(a, column), _identity_text(b, column)
        if left and right and left != right:
            fields.append(column)
    for column in _SHARED_RESULT_COLUMNS:
        left = _normalized_column(a, column)
        right = _normalized_column(b, column)
        if column in _STATE_RESULT_COLUMNS:
            if left != right:
                fields.append(column)
        elif left and right and left != right:
            fields.append(column)
    return fields


def _dedupe_sources(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """按归一化签名去重（保留首次出现），只消除完全相同的富化行。"""
    seen: set[tuple[str, ...]] = set()
    unique: list[dict[str, str]] = []
    for row in candidates:
        signature = _enrichment_signature(row)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(row)
    return unique


def _candidate_conflict_fields(candidates: list[dict[str, str]]) -> list[str]:
    """多个候选富化行之间不一致的字段名（供 ``DatasetConflictError`` 报告）。"""
    signatures = [_enrichment_signature(row) for row in candidates]
    fields: list[str] = []
    for index, name in enumerate(_enrichment_field_names()):
        if len({signature[index] for signature in signatures}) > 1:
            fields.append(name)
    return fields


def _physical_index(ctx: StatsContext) -> PhysicalIndex:
    """建立 runs.csv 富化行索引（M03：按完整身份，不按 artifact 路径串接）。

    续跑恢复后同一身份可能有多条执行尝试；索引只收最终尝试行（完成行优先，
    否则最后一条失败行），避免把早期失败行的空 physical 列当成第二条富化记录。

    Args:
        ctx: 统计上下文。

    Returns:
        ``PhysicalIndex``；没有带 physical 列的 runs.csv 时返回空索引。
    """
    for path in _runs_csv_paths(ctx):
        rows = _read_csv_rows(path)
        if not rows:
            continue
        if "physical_hit" not in rows[0] and "physical_hit_type" not in rows[0]:
            continue
        rows, _ = _final_attempt_rows(rows, path)
        by_identity: dict[tuple[Any, ...], list[dict[str, str]]] = {}
        by_artifact: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            by_identity.setdefault(record_identity(row).join_key(), []).append(row)
            key = _artifact_key(row, ctx)
            if key:
                by_artifact.setdefault(key, []).append(row)
        return PhysicalIndex(by_identity=by_identity, by_artifact=by_artifact, path=path)
    return PhysicalIndex()


def _resolve_physical_source(
    results_row: dict[str, str],
    index: PhysicalIndex,
    ctx: StatsContext,
    *,
    results_path: Path | None = None,
) -> tuple[dict[str, str] | None, str, list[str]]:
    """为一行 results 记录选择可安全回填的 runs 富化行（M03 唯一连接入口）。

    连接顺序：先按 ``RecordIdentity.join_key()``（完整身份）命中；只有该行缺少身份
    组件（legacy 数据）时才退回 artifact 路径通道，且要求路径唯一、两侧可用字段一致。

    Args:
        results_row: results.csv 的一行。
        index: runs.csv 富化行索引。
        ctx: 统计上下文（artifact 相对路径归一化用）。
        results_path: 该行所属 results 文件（错误信息用）。

    Returns:
        ``(source, status, conflict_fields)``；``source`` 为 None 表示不回填，
        ``status`` ∈ ``matched`` / ``conflict`` / ``missing``。

    Raises:
        DatasetConflictError: 同一身份键命中多行且归一化后不一致。
    """
    identity = record_identity(results_row)
    has_identity = bool(
        identity.experiment and identity.case and identity.config_id and identity.seed >= 0
    )
    candidates = index.by_identity.get(identity.join_key(), []) if has_identity else []
    if candidates:
        unique = _dedupe_sources(candidates)
        if len(unique) > 1:
            raise DatasetConflictError(
                "runs.csv 富化行冲突："
                f"join_key={identity.join_key()}，results 文件={results_path or '?'}，"
                f"runs 文件={index.path or '?'}，"
                f"冲突字段={_candidate_conflict_fields(unique)}；"
                "同一身份下禁止静默择一，请先修正数据"
            )
        conflicts = _enrichment_conflicts(unique[0], results_row)
        if conflicts:
            return None, "conflict", conflicts
        return unique[0], "matched", []
    if not has_identity:
        path_key = _artifact_key(results_row, ctx)
        path_candidates = index.by_artifact.get(path_key, []) if path_key else []
        unique = _dedupe_sources(path_candidates)
        if len(unique) > 1:
            raise DatasetConflictError(
                "runs.csv 富化行冲突（legacy 路径通道）："
                f"artifact={path_key}，results 文件={results_path or '?'}，"
                f"runs 文件={index.path or '?'}，"
                f"冲突字段={_candidate_conflict_fields(unique)}；"
                "路径下存在多条不一致富化记录，请先修正数据"
            )
        if unique:
            conflicts = _enrichment_conflicts(unique[0], results_row)
            if conflicts:
                return None, "conflict", conflicts
            return unique[0], "matched", []
    return None, "missing", []


def _load_rows(ctx: StatsContext, name: str) -> list[dict[str, str]]:
    """读取实验 results.csv，按执行模式过滤，并按完整身份富化 physical 列。

    M03：富化只走 ``RecordIdentity.join_key()``（legacy 缺身份组件时退回唯一 artifact
    路径），不再以 artifact 路径作为跨 mode/身份串接的通道。

    续跑恢复（P2）：同一身份的多条执行尝试先按 :func:`_final_attempt_rows` 收敛为
    一条统计样本（完成行优先，否则最后一条失败行），失败尝试仍保留在 CSV 原文件。
    """
    path = ctx.data_root / name / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 {path}")
    rows = _read_csv_rows(path)
    rows, _ = _final_attempt_rows(rows, path)
    index = _physical_index(ctx)
    kept: list[dict[str, str]] = []
    for row in rows:
        if ctx.mode is not None and mode_of(row) not in (None, ctx.mode):
            continue
        source, status, fields = _resolve_physical_source(
            row, index, ctx, results_path=path
        )
        if source is not None:
            for column in _PHYSICAL_COLUMNS:
                if not (row.get(column) or "").strip() and (source.get(column) or "").strip():
                    row[column] = source[column]
            row["_physical_source"] = str(index.path)
        row["_physical_status"] = status
        if fields:
            row["_physical_conflicts"] = ",".join(fields)
        kept.append(row)
    return kept


def load(data_root: Path, name: str) -> list[dict[str, str]]:
    """旧签名包装（无 catalog、无模式过滤），保持既有调用方可用。"""
    return _load_rows(make_context(data_root), name)


def _manifest_tasks(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """兼容 manifest 的 ``tasks`` 列表与 ``experiments/cases/seeds`` 结构。"""
    tasks = payload.get("tasks")
    if isinstance(tasks, list):
        return [task for task in tasks if isinstance(task, dict)]
    out: list[dict[str, Any]] = []
    for experiment in payload.get("experiments") or []:
        if not isinstance(experiment, dict):
            continue
        exp_id = str(experiment.get("experiment") or experiment.get("experiment_id") or "")
        for case in experiment.get("cases") or []:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get("case_id") or case.get("case") or "")
            mode = str(case.get("mode") or "")
            for seed in case.get("seeds") or []:
                out.append({"experiment": exp_id, "case_id": case_id,
                            "seed": seed, "mode": mode})
    return out


def _sample_keys(keys: set[tuple[Any, ...]]) -> list[dict[str, Any]]:
    """把身份键集合渲染成最多 5 条可读样例（报告 missing 用）。"""
    samples: list[dict[str, Any]] = []
    for key in sorted(keys)[:5]:
        samples.append({
            "experiment": key[0] if len(key) > 0 else "",
            "case": key[1] if len(key) > 1 else "",
            "config_id": key[2] if len(key) > 2 else "",
            "seed": key[3] if len(key) > 3 else -1,
            "mode": key[4] if len(key) > 4 else "",
        })
    return samples


def _validate_dataset(ctx: StatsContext) -> dict[str, Any]:
    """校验唯一性/身份/两视图一对一连接，返回 validation 摘要。

    两视图（results.csv 基础字段、runs.csv 富化字段）**不纵向拼接**：各自先按
    执行尝试收敛（同一身份多条失败尝试 + 至多一条完成尝试，见
    :func:`_final_attempt_rows`），再按完整身份一对一连接；缺富化行只报告，
    不伪造 physical 判定。

    计数口径：``results_records`` / ``runs_records`` 是文件行数（含被忽略的早期
    尝试），``mode_records`` 是最终尝试样本数；被忽略的文件行数与样例见
    ``attempts_ignored`` / ``attempts_ignored_samples``，不静默丢弃。
    """
    results_paths = sorted(ctx.data_root.glob("*/results.csv"))
    results_records = 0
    results_rows: list[dict[str, str]] = []
    ignored_attempts: list[dict[str, Any]] = []
    for path in results_paths:
        rows = _read_csv_rows(path)
        selected, ignored = _final_attempt_rows(rows, path)
        results_records += len(rows)
        results_rows.extend(selected)
        ignored_attempts.extend({**item, "view": "results"} for item in ignored)

    runs_path = _runs_csv_path(ctx)
    runs_records = 0
    runs_rows: list[dict[str, str]] = []
    if runs_path is not None:
        rows = _read_csv_rows(runs_path)
        runs_records = len(rows)
        runs_rows, ignored = _final_attempt_rows(rows, runs_path)
        ignored_attempts.extend({**item, "view": "runs"} for item in ignored)

    robot_ids = Counter(
        (row.get("robot_id") or "").strip()
        for row in results_rows
        if (row.get("robot_id") or "").strip()
    )
    if ctx.catalog is not None and robot_ids and set(robot_ids) != {ctx.catalog.robot_id}:
        raise IdentityError(
            f"robot_id 与 catalog 声明不一致：数据 {sorted(robot_ids)} vs "
            f"catalog {ctx.catalog.robot_id!r}；禁止跨机器人配对"
        )
    rackets: dict[str, set[str]] = defaultdict(set)
    for row in results_rows:
        racket = (row.get("racket_id") or "").strip()
        if racket:
            rackets[mode_of(row) or "unknown"].add(racket)
    for mode, values in rackets.items():
        if len(values) > 1:
            raise IdentityError(
                f"mode={mode} 内出现多个 racket_id {sorted(values)}："
                "不同球拍/来源不得配对，请拆分数据根或显式指定身份"
            )

    # 校验与 CLI 拒绝都以“当前执行模式”为作用域：position 的冲突不能让
    # torque 统计失败（反之亦然）；--mode both 时每个模式各自看自己的记录。
    def in_scope(row: dict[str, str]) -> bool:
        return ctx.mode is None or mode_of(row) == ctx.mode

    scoped_results = [row for row in results_rows if in_scope(row)]
    scoped_runs = [row for row in runs_rows if in_scope(row)]

    results_keys = {record_identity(row).join_key() for row in scoped_results}
    runs_keys = {record_identity(row).join_key() for row in scoped_runs}
    missing_enrichment = results_keys - runs_keys
    runs_only = runs_keys - results_keys

    # M03：富化必须按完整身份命中；同路径异身份/异设计只报告，绝不静默回填
    physical_index = _physical_index(ctx)
    enrichment_status: Counter[str] = Counter()
    physical_conflicts: list[dict[str, Any]] = []
    for row in scoped_results:
        try:
            _, status, fields = _resolve_physical_source(row, physical_index, ctx)
        except DatasetConflictError as exc:
            status, fields = "conflict", [str(exc)]
        enrichment_status[status] += 1
        if status == "conflict":
            physical_conflicts.append({
                "experiment": (row.get("experiment") or "").strip(),
                "case": case_of(row),
                "config_id": (row.get("config_id") or "").strip(),
                "seed": _seed_value(row),
                "mode": mode_of(row) or "",
                "fields": fields,
            })

    manifest_path = ctx.manifest_path or (ctx.run_root / "manifest_resolved.json")
    expected: set[tuple[str, str, int, str]] = set()
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        for task in _manifest_tasks(payload):
            task_mode = str(task.get("mode") or "")
            if ctx.mode is not None and task_mode != ctx.mode:
                continue
            expected.add((
                str(task.get("experiment") or ""),
                str(task.get("case_id") or ""),
                _seed_value({"seed": task.get("seed")}),
                task_mode,
            ))
    actual = {(key[0], key[1], key[3], key[4]) for key in runs_keys}
    manifest_missing = expected - actual

    mode_counts: Counter[str] = Counter()
    for row in results_rows:
        mode_counts[mode_of(row) or "unknown"] += 1
    mode_records = (
        sum(1 for row in results_rows if mode_of(row) == ctx.mode)
        if ctx.mode is not None else len(results_rows)
    )
    return {
        "mode": ctx.mode,
        "modes_found": dict(mode_counts),
        "mode_records": mode_records,
        "results_files": [str(path) for path in results_paths],
        "runs_csv": str(runs_path) if runs_path is not None else None,
        "results_records": results_records,
        "runs_records": runs_records,
        "attempts_ignored": len(ignored_attempts),
        "attempts_ignored_samples": ignored_attempts[:5],
        "manifest_path": str(manifest_path) if manifest_path.is_file() else None,
        "manifest_expected": len(expected),
        "manifest_missing": len(manifest_missing),
        "manifest_missing_samples": _sample_keys(manifest_missing),
        "missing_enrichment": len(missing_enrichment),
        "missing_enrichment_samples": _sample_keys(missing_enrichment),
        "runs_only": len(runs_only),
        "runs_only_samples": _sample_keys(runs_only),
        "enrichment_status": dict(enrichment_status),
        "physical_conflicts": len(physical_conflicts),
        "physical_conflict_samples": physical_conflicts[:5],
        "robot_ids": sorted(robot_ids),
        "mode_rackets": {mode: sorted(values) for mode, values in rackets.items()},
        "synthetic": bool(ctx.synthetic),
    }


def _discover_modes(ctx: StatsContext) -> dict[str, int]:
    """扫描数据根内的执行模式分布（results.csv 优先，空时退回 runs.csv）。"""
    counts: Counter[str] = Counter()
    for path in sorted(ctx.data_root.glob("*/results.csv")):
        for row in _read_csv_rows(path):
            counts[mode_of(row) or "unknown"] += 1
    if not counts:
        runs_path = _runs_csv_path(ctx)
        if runs_path is not None:
            for row in _read_csv_rows(runs_path):
                counts[mode_of(row) or "unknown"] += 1
    return dict(counts)


def resolve_mode(ctx: StatsContext) -> str | None:
    """确定本次统计的执行模式；混合模式必须显式选择（F04）。

    Args:
        ctx: 统计上下文（``ctx.mode`` 为显式选择或 None）。

    Returns:
        解析后的模式（``torque`` / ``position`` / None）。

    Raises:
        ModeSelectionError: 数据含多种模式或存在无法判定的模式。
    """
    if ctx.mode is not None:
        return ctx.mode
    counts = _discover_modes(ctx)
    known = {mode: counts[mode] for mode in VALID_MODES if counts.get(mode)}
    if len(known) > 1:
        detail = "、".join(f"{mode}={count}" for mode, count in sorted(known.items()))
        raise ModeSelectionError(
            f"检测到混合执行模式（{detail}）：必须显式选择 --mode torque / "
            "--mode position / --mode both，禁止把两种模式合成一个样本池"
        )
    if len(known) == 1:
        return next(iter(known))
    if counts.get("unknown"):
        raise ModeSelectionError(
            f"有 {counts['unknown']} 条记录无法确定执行模式（缺 mode 列，且无 "
            "--position-mode/--actuator-mode 证据）：请显式 --mode torque|position"
        )
    return None


def cfg_of(row: dict[str, str]) -> dict[str, Any]:
    """解析 config_json。"""
    return json.loads(row["config_json"])


def is_error(row: dict[str, str]) -> bool:
    """该行是否被排除（error 非空，实测均为 no_result）。"""
    return bool((row.get("error") or "").strip())


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


def physical_of(row: dict[str, str]) -> int | None:
    """physical 口径命中：1 / 0；无判定（缺失或不可用）返回 None。

    R03：以 ``physical_hit_type`` 为准 —— 只有 ``active`` 计命中，
    ``passive``/``miss`` 一律不计；``physical_hit`` 布尔仅在类型缺失或未知时兜底，
    避免"被动接触被布尔列误计为主动命中"。
    """
    ptype = (row.get("physical_hit_type") or "").strip().lower()
    if ptype == "active":
        return 1
    if ptype in ("passive", "miss"):
        return 0
    hit = _to_bool_or_none(row.get("physical_hit"))
    if hit is not None:
        return 1 if hit else 0
    return None


def _fmt_optional(value: Any, fmt: str = "{:.2f}") -> str:
    """格式化可缺省数值：None/NaN 渲染为 ``--``（不伪造 0）。"""
    if value is None:
        return "--"
    if isinstance(value, float) and math.isnan(value):
        return "--"
    return fmt.format(value)


def _fmt_ci(ci: Any) -> str:
    """格式化配对 CI：None/不可解析 → ``不可估计``（R06：绝不写零宽区间）。"""
    if ci is None:
        return "不可估计"
    try:
        lo, hi = float(ci[0]), float(ci[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return "不可估计"
    return f"[{lo:+.2f},{hi:+.2f}]"


def _fmt_p(p_value: Any, fmt: str = "{:.4f}") -> str:
    """格式化配对 p 值：None/NaN → ``不可估计``（R06：绝不格式化成 0）。"""
    if p_value is None:
        return "不可估计"
    try:
        value = float(p_value)
    except (TypeError, ValueError):
        return "不可估计"
    if math.isnan(value):
        return "不可估计"
    return fmt.format(value)


def _fmt_pp(value: Any) -> str:
    """格式化 pp 差值/增益：None/NaN → ``不可估计``。"""
    if value is None:
        return "不可估计"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "不可估计"
    if math.isnan(v):
        return "不可估计"
    return f"{v:+.2f}"


def _reason_text(stats: dict[str, Any]) -> str:
    """从 cell 统计里取不可估计原因（缺失时给通用说明）。"""
    return str(stats.get("unavailable_reason") or "无有效判定（全 miss / 无 verdict / 缺数据）")


def cell_stats(rows: list[dict[str, str]], hitfn: HitPolicy) -> dict[str, Any]:
    """统计一格的样本构成与命中率（含 Wilson CI、passive 计数）。"""
    errs: Counter[str] = Counter()
    types: Counter[str] = Counter()
    for r in rows:
        if is_error(r):
            errs[(r.get("error") or "").strip()] += 1
        else:
            types[(r.get("hit_type") or "").strip()] += 1
    n_valid = sum(types.values())
    hits = sum(hitfn(r) for r in rows if not is_error(r))
    lo, hi = wilson(hits, n_valid)
    return {
        "rows": len(rows),
        "n_valid": n_valid,
        "hits": hits,
        "rate": 100 * hits / n_valid if n_valid else float("nan"),
        "ci95": [lo, hi],
        "active": types.get("active", 0),
        "passive": types.get("passive", 0),
        "miss": types.get("miss", 0),
        "passive_share": 100 * types.get("passive", 0) / n_valid if n_valid else 0.0,
        "errors": dict(errs),
    }


def physical_stats(rows: list[dict[str, str]]) -> dict[str, Any]:
    """physical 口径统计：有判定即给真实 rate（0/N 记 0.0），无判定才 rate=null。

    R03：``physical_hit=True + type=passive`` 不计主动命中，该 cell 得到
    ``n_active=0, n_passive=1, rate=0.0``，并保留 ``unavailable_reason="all_miss"``
    供表格标注。0/N 是测得的 0%，不是"缺数据"，故不再返回 null；只有完全
    没有 physical 判定（如旧 RM65 无该列）时 rate 才为 null，不伪造 0.0。
    CI 仅在有命中时报告（全 miss 与 R06 配对口径一致，不给零命中区间）。
    """
    verdicts = [physical_of(r) for r in rows if not is_error(r)]
    seen = [v for v in verdicts if v is not None]
    n_valid = len(verdicts)
    n_active = sum(seen)
    n_passive = sum(
        1 for r in rows
        if not is_error(r) and physical_of(r) == 0
        and (r.get("physical_hit_type") or "").strip().lower() == "passive"
    )
    reason = None
    rate = None
    if not seen:
        reason = "no_physical_verdict"
    else:
        rate = 100 * n_active / len(seen)
        if n_active == 0:
            reason = "all_miss"
    return {
        "rate": rate,
        "denominator": len(seen),
        "n_valid": n_valid,
        "n_missing_verdict": n_valid - len(seen),
        "n_active": n_active,
        "n_passive": n_passive,
        "n_miss": len(seen) - n_active - n_passive,
        "ci95": list(wilson(n_active, len(seen))) if n_active else None,
        "unavailable_reason": reason,
        "hit_semantics": "physical",
    }


def _unique_index(
    rows: list[dict[str, str]],
    keyfn: Callable[[dict[str, str]], Hashable],
    valuefn: Callable[[dict[str, str]], int | None],
) -> tuple[dict[Hashable, int], list[Hashable]]:
    """按配对键建立去重取值索引（M02：行序无关，禁止后写覆盖）。

    同一个 key 出现多次时：取值全相同则去重（保留一份）；取值不同则把该 key
    整体排除（无法判定哪一行才是真值），并在返回值里报告。

    Args:
        rows: 已过滤的候选行。
        keyfn: 配对键（``paired_test`` 用 seed 字符串，DiD 用 seed 整数；只要求可哈希）。
        valuefn: 取值函数；返回 None 的行直接跳过（例如缺 physical 判定）。

    Returns:
        ``(index, conflicts)``：index 为 key → 取值；conflicts 为因取值冲突被
        排除的 key 列表（升序）。
    """
    groups: dict[Hashable, set[int]] = {}
    for row in rows:
        value = valuefn(row)
        if value is None:
            continue
        groups.setdefault(keyfn(row), set()).add(value)
    index: dict[Hashable, int] = {}
    conflicts: list[Hashable] = []
    for key in sorted(groups):
        values = groups[key]
        if len(values) == 1:
            index[key] = next(iter(values))
        else:
            conflicts.append(key)
    return index, conflicts


def paired_test(
    rows: list[dict[str, str]],
    hitfn: HitPolicy,
    tier_a: str,
    tier_b: str,
    keyfn: Callable[[dict[str, str]], str] = lambda r: r["seed"],
    filterfn: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, Any]:
    """same-seed 配对比较：McNemar 精确检验 + 配对 bootstrap 95% CI。

    M02：同一 seed 在同一档重复出现时不再后写覆盖——同值去重、异值整键排除，
    因此结果与 CSV 行序无关；被排除的键在 ``excluded_duplicate_keys`` 中报告。

    Returns:
        含 n_paired / diff_pp / ci95 / p_mcnemar / 各档率的字典；无共同 seed 时
        ``unavailable_reason`` 为 ``no_common_seed`` 或 ``duplicate_seed_conflict``。
    """
    sel = [r for r in rows if (filterfn(r) if filterfn else True) and not is_error(r)]
    rows_a = [r for r in sel if tier_of(r) == tier_a]
    rows_b = [r for r in sel if tier_of(r) == tier_b]
    da, conflicts_a = _unique_index(rows_a, keyfn, hitfn)
    db, conflicts_b = _unique_index(rows_b, keyfn, hitfn)
    excluded = sorted(set(conflicts_a) | set(conflicts_b))
    keys = sorted(set(da) & set(db))
    if not keys:
        out: dict[str, Any] = {
            "n_paired": 0,
            "unavailable_reason": (
                "duplicate_seed_conflict" if excluded else "no_common_seed"
            ),
        }
        if excluded:
            out["excluded_duplicate_keys"] = excluded
        return out
    ha = [da[k] for k in keys]
    hb = [db[k] for k in keys]
    b = sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1)
    lo, hi = paired_bootstrap(ha, hb)
    result: dict[str, Any] = {
        "n_paired": len(keys),
        "rate_a": 100 * sum(ha) / len(keys),
        "rate_b": 100 * sum(hb) / len(keys),
        "diff_pp": 100 * (sum(ha) - sum(hb)) / len(keys),
        "ci95": [lo, hi],
        "p_mcnemar": mcnemar_exact(b, c),
        "discordant": {"a_hit_b_miss": b, "a_miss_b_hit": c},
    }
    if excluded:
        result["excluded_duplicate_keys"] = excluded
    return result


def _physical_pair_summary(ha: list[int], hb: list[int]) -> dict[str, Any]:
    """physical 口径配对结果汇总（R06：两档共同 seed 全 miss 时显式不可估计）。

    Args:
        ha: 档 A 每个共同 seed 的 physical 命中（0/1）。
        hb: 档 B 每个共同 seed 的 physical 命中（0/1）。

    Returns:
        n_paired / rate_a / rate_b / diff_pp / discordant / hit_semantics 恒有；
        ci95 / p_mcnemar 在可估计时为数值，两档全 miss 时为 None 并给
        ``unavailable_reason="all_miss"``（不输出 [0,0] 零宽区间、不伪造 p=1）。
    """
    n = len(ha)
    total_a = sum(ha)
    total_b = sum(hb)
    b = sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1)
    out: dict[str, Any] = {
        "n_paired": n,
        "rate_a": 100 * total_a / n,
        "rate_b": 100 * total_b / n,
        "diff_pp": 100 * (total_a - total_b) / n,
        "discordant": {"a_hit_b_miss": b, "a_miss_b_hit": c},
        "hit_semantics": "physical",
    }
    if total_a == 0 and total_b == 0:
        out["ci95"] = None
        out["p_mcnemar"] = None
        out["unavailable_reason"] = "all_miss"
        return out
    out["ci95"] = list(paired_bootstrap(ha, hb))
    out["p_mcnemar"] = mcnemar_exact(b, c)
    return out


def paired_test_physical(
    rows: list[dict[str, str]],
    tier_a: str,
    tier_b: str,
    keyfn: Callable[[dict[str, str]], str] = lambda r: r["seed"],
    filterfn: Callable[[dict[str, str]], bool] | None = None,
) -> dict[str, Any]:
    """physical 口径的同 seed 配对比较（缺判定/无共同 seed/全 miss 显式不可估计）。

    M02：同一 seed 重复出现时同值去重、异值整键排除，结果与行序无关；
    冲突键在 ``excluded_duplicate_keys`` 中报告。
    """
    sel = [r for r in rows if (filterfn(r) if filterfn else True) and not is_error(r)]
    rows_a = [r for r in sel if tier_of(r) == tier_a]
    rows_b = [r for r in sel if tier_of(r) == tier_b]
    da, conflicts_a = _unique_index(rows_a, keyfn, physical_of)
    db, conflicts_b = _unique_index(rows_b, keyfn, physical_of)
    excluded = sorted(set(conflicts_a) | set(conflicts_b))
    keys = sorted(set(da) & set(db))
    if not keys:
        if excluded:
            return {
                "n_paired": 0,
                "unavailable_reason": "duplicate_seed_conflict",
                "excluded_duplicate_keys": excluded,
            }
        return {
            "n_paired": 0,
            "unavailable_reason": "no_common_seed" if (da or db) else "no_physical_verdict",
        }
    summary = _physical_pair_summary([da[k] for k in keys], [db[k] for k in keys])
    if excluded:
        summary["excluded_duplicate_keys"] = excluded
    return summary


def baseline_cell(row: dict[str, str]) -> bool:
    """exp17a 的「无噪声、无 KF」基线条件（空配置）。"""
    c = cfg_of(row)
    return (
        not c.get("--obs-noise-pos")
        and not c.get("--obs-noise-vel")
        and "--obs-use-kf" not in c
    )


# ----------------------------------------------------------------------------
# 各实验统计（每个函数返回 (legacy 段落, physical 段落)）
# ----------------------------------------------------------------------------

def e1_speed_sweep(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E1: 球速扫描（exp15，9 档 × 200 seeds，TCP 1.8 默认限位）。"""
    rows = _load_rows(ctx, "exp15_speed_v2")
    by_speed: dict[int, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_speed[speed_of(r)].append(r)
    legacy: dict[str, Any] = {}
    physical: dict[str, Any] = {}
    for sp in sorted(by_speed):
        legacy[str(sp)] = cell_stats(by_speed[sp], hitfn)
        physical[str(sp)] = physical_stats(by_speed[sp])
    return legacy, physical


def e2_limit_cost(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E2: 真机限位代价（exp16，7 m/s × 200 seeds，TCP 1.8 vs 1.0）。"""
    rows = _load_rows(ctx, "exp16_limits_v2")
    by_lim: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows:
        by_lim[tcp_label_of(r)].append(r)
    legacy: dict[str, Any] = {}
    physical: dict[str, Any] = {}
    for key, sub in by_lim.items():
        st = cell_stats(sub, hitfn)
        speeds = [
            float(r["v_racket_at_hit"])
            for r in sub
            if not is_error(r) and hitfn(r) == 1 and (r.get("v_racket_at_hit") or "")
        ]
        st["mean_racket_speed_on_hit"] = mean(speeds) if speeds else float("nan")
        legacy[key] = st
        physical[key] = physical_stats(sub)
    # 配对比较（同 seed 跨限速层）
    da, db = {}, {}
    pa, pb = {}, {}
    for r in rows:
        if is_error(r):
            continue
        is_tcp10 = tcp_label_of(r) == "TCP1.0"
        (da if is_tcp10 else db)[r["seed"]] = hitfn(r)
        verdict = physical_of(r)
        if verdict is not None:
            (pa if is_tcp10 else pb)[r["seed"]] = verdict
    keys = sorted(set(da) & set(db))
    if keys:
        ha = [da[k] for k in keys]
        hb = [db[k] for k in keys]
        legacy["paired_TCP10_vs_TCP18"] = {
            "n_paired": len(keys),
            "diff_pp": 100 * (sum(ha) - sum(hb)) / len(keys),
            "ci95": list(paired_bootstrap(ha, hb)),
            "p_mcnemar": mcnemar_exact(
                sum(1 for x, y in zip(ha, hb) if x == 1 and y == 0),
                sum(1 for x, y in zip(ha, hb) if x == 0 and y == 1),
            ),
        }
    pkeys = sorted(set(pa) & set(pb))
    if pkeys:
        physical["paired_TCP10_vs_TCP18"] = _physical_pair_summary(
            [pa[k] for k in pkeys], [pb[k] for k in pkeys])
    else:
        physical["paired_TCP10_vs_TCP18"] = {
            "n_paired": 0,
            "unavailable_reason": "no_physical_verdict",
        }
    return legacy, physical


def e3_nominal(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E3: 标称四档（来源由 catalog 声明决定；缺 exp17a_noise 不阻断）。

    F06 根因：旧实现无条件先读 ``exp17a_noise``，R1D manifest 只声明
    ``source_sets.nominal_mechanism = exp17d_mechanism`` 时整节被记 unavailable。
    现在 catalog 声明了 mechanism 就只读它（四档都取自该来源）；无 catalog 的旧
    RM65 根仍走 exp17a（full/none）+ exp17d（tube_only/softmin_only）两段式来源。
    """
    declared = str(ctx.source_sets.get("nominal_mechanism") or "").strip()
    out: dict[str, Any] = {"cells": {}, "paired": {}, "data_sources": []}
    physical: dict[str, Any] = {"cells": {}, "paired": {}, "data_sources": []}
    if declared:
        rows = _load_rows(ctx, declared)
        tier_rows: dict[str, list[dict[str, str]]] = {tier: rows for tier in MODES}
        combined = rows
        out["data_sources"] = [declared]
    else:
        rows_a = _load_rows(ctx, "exp17a_noise")
        rows_d = _load_rows(ctx, "exp17d_mechanism")
        tier_rows = {"full": rows_a, "none": rows_a,
                     "tube_only": rows_d, "softmin_only": rows_d}
        combined = rows_a + rows_d
        out["data_sources"] = ["exp17a_noise", "exp17d_mechanism"]
    physical["data_sources"] = list(out["data_sources"])

    for sp in (7, 9, 12):
        for tier in MODES:
            sub = [r for r in tier_rows[tier]
                   if baseline_cell(r) and speed_of(r) == sp and tier_of(r) == tier]
            key = f"{sp}|{tier}"
            out["cells"][key] = cell_stats(sub, hitfn)
            physical["cells"][key] = physical_stats(sub)

    def at_speed(sp: int):
        def f(r: dict[str, str]) -> bool:
            if not baseline_cell(r):
                return False
            return speed_of(r) == sp
        return f

    for sp in (7, 9, 12):
        for a, b in (("full", "none"), ("softmin_only", "none"),
                     ("tube_only", "none"), ("full", "softmin_only"),
                     ("tube_only", "softmin_only")):
            key = f"{sp}|{a}_vs_{b}"
            out["paired"][key] = paired_test(combined, hitfn, a, b, filterfn=at_speed(sp))
            physical["paired"][key] = paired_test_physical(
                combined, a, b, filterfn=at_speed(sp))

    # M02：前馈关闭变体独立成节（不混入 nominal 四档，也不参与 nominal 配对）
    seen_variants: set[str] = set()
    for row in combined:
        tier = tier_of(row)
        if tier.endswith(NOFEEDFORWARD_SUFFIX):
            seen_variants.add(tier)
    variant_tiers = sorted(seen_variants)
    variants_legacy: dict[str, Any] = {
        "tiers": variant_tiers, "cells": {}, "paired": {},
    }
    variants_physical: dict[str, Any] = {
        "tiers": list(variant_tiers), "cells": {}, "paired": {},
    }
    for variant in variant_tiers:
        for sp in (7, 9, 12):
            sub = [r for r in combined
                   if baseline_cell(r) and speed_of(r) == sp and tier_of(r) == variant]
            cell_key = f"{sp}|{variant}"
            variants_legacy["cells"][cell_key] = cell_stats(sub, hitfn)
            variants_physical["cells"][cell_key] = physical_stats(sub)
            pair_key = f"{sp}|none_vs_{variant}"
            variants_legacy["paired"][pair_key] = paired_test(
                combined, hitfn, "none", variant, filterfn=at_speed(sp))
            variants_physical["paired"][pair_key] = paired_test_physical(
                combined, "none", variant, filterfn=at_speed(sp))
    out["variants"] = variants_legacy
    physical["variants"] = variants_physical
    return out, physical


def e4_grid(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E4: 扰动网格（exp17b+17f，9 m/s，20 格合并，配对键 = (格, seed)）。"""
    rows = _load_rows(ctx, "exp17b_perturb") + _load_rows(ctx, "exp17f_mechanism_perturb")
    TS = {0, 10, 25, 50, 100}
    SS = {0.0, 0.05, 0.1, 0.2}

    def in_grid(r: dict[str, str]) -> bool:
        return (
            speed_of(r) == 9
            and time_perturb_ms_of(r) in TS
            and space_perturb_of(r) in SS
        )

    sub = [r for r in rows if in_grid(r)]
    out: dict[str, Any] = {"merged_20cell": {}, "paired_20cell": {}}
    physical: dict[str, Any] = {"merged_20cell": {}, "paired_20cell": {}}
    for tier in MODES:
        tier_rows = [r for r in sub if tier_of(r) == tier]
        out["merged_20cell"][tier] = cell_stats(tier_rows, hitfn)
        physical["merged_20cell"][tier] = physical_stats(tier_rows)

    def grid_key(r: dict[str, str]) -> str:
        return (f"{time_perturb_ms_of(r)}|"
                f"{space_perturb_of(r)}|{r['seed']}")

    for a, b in (("full", "none"), ("softmin_only", "none"), ("tube_only", "none"),
                 ("full", "softmin_only"), ("full", "tube_only"),
                 ("tube_only", "softmin_only")):
        key = f"{a}_vs_{b}"
        out["paired_20cell"][key] = paired_test(sub, hitfn, a, b, keyfn=grid_key)
        physical["paired_20cell"][key] = paired_test_physical(
            sub, a, b, keyfn=grid_key)
    # 每格明细（供热图显著性标注）+ 每格配对检验
    per_cell: dict[str, Any] = {}
    per_cell_paired: dict[str, Any] = {}
    per_cell_physical: dict[str, Any] = {}
    per_cell_paired_physical: dict[str, Any] = {}
    for t in sorted(TS):
        for s in sorted(SS):
            for tier in MODES:
                cellrows = [r for r in sub
                            if time_perturb_ms_of(r) == t
                            and space_perturb_of(r) == s
                            and tier_of(r) == tier]
                key = f"t{t}|s{s}|{tier}"
                per_cell[key] = cell_stats(cellrows, hitfn)
                per_cell_physical[key] = physical_stats(cellrows)

            def cell_filter(r: dict[str, str], t: int = t, s: float = s) -> bool:
                return time_perturb_ms_of(r) == t and space_perturb_of(r) == s

            for a, b in (("full", "none"), ("tube_only", "none"),
                         ("softmin_only", "none"), ("full", "softmin_only")):
                key = f"t{t}|s{s}|{a}_vs_{b}"
                per_cell_paired[key] = paired_test(
                    sub, hitfn, a, b, filterfn=cell_filter)
                per_cell_paired_physical[key] = paired_test_physical(
                    sub, a, b, filterfn=cell_filter)
    out["per_cell"] = per_cell
    out["per_cell_paired"] = per_cell_paired
    physical["per_cell"] = per_cell_physical
    physical["per_cell_paired"] = per_cell_paired_physical
    return out, physical


def e5_perception(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E5: 噪声 × KF（exp17a）与观测频率（exp17c）。"""
    rows = _load_rows(ctx, "exp17a_noise")
    out: dict[str, Any] = {"noise_x_kf": {}}
    physical: dict[str, Any] = {"noise_x_kf": {}}
    for sp in (7, 9, 12):
        for pos, vel in sorted({(cfg_of(r).get("--obs-noise-pos"),
                                 cfg_of(r).get("--obs-noise-vel"))
                                for r in rows}, key=str):
            for kf in (False, True):
                sub = [r for r in rows
                       if speed_of(r) == sp
                       and cfg_of(r).get("--obs-noise-pos") == pos
                       and cfg_of(r).get("--obs-noise-vel") == vel
                       and ((("--obs-use-kf" in cfg_of(r)) == kf))]
                for tier in ("full", "none"):
                    cellrows = [r for r in sub if tier_of(r) == tier]
                    if cellrows:
                        key = f"{sp}|pos{pos}|vel{vel}|kf{int(kf)}|{tier}"
                        out["noise_x_kf"][key] = cell_stats(cellrows, hitfn)
                        physical["noise_x_kf"][key] = physical_stats(cellrows)
    rows_c = _load_rows(ctx, "exp17c_obsfreq")
    by_rate: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in rows_c:
        by_rate[_obs_freq_of(r)].append(r)
    out["obsfreq"] = {k: cell_stats(v, hitfn) for k, v in sorted(by_rate.items())}
    physical["obsfreq"] = {k: physical_stats(v) for k, v in sorted(by_rate.items())}
    return out, physical


def e7_corners(hitfn: HitPolicy, ctx: StatsContext) -> tuple[dict[str, Any], dict[str, Any]]:
    """E7: 角点高功率格（exp17g / exp17h，预先设定的 4 个对比）。"""
    rows_g, rows_h = _load_rows(ctx, "exp17g_spatial_power"), _load_rows(ctx, "exp17h_extreme")
    out: dict[str, Any] = {"cells": {}, "paired": {}}
    physical: dict[str, Any] = {"cells": {}, "paired": {}}
    specs = [("exp17g", rows_g, 9, 0.2, 0), ("exp17g", rows_g, 9, 0.2, 50),
             ("exp17h", rows_h, 12, 0.2, 0), ("exp17h", rows_h, 9, 0.3, 0),
             ("exp17h", rows_h, 9, 0.4, 0)]

    def filt(sp: int, s: float, t: int):
        def f(r: dict[str, str]) -> bool:
            return (speed_of(r) == sp
                    and space_perturb_of(r) == s
                    and time_perturb_ms_of(r) == t)
        return f

    for src, rows, sp, s, t in specs:
        tag = f"{sp}|s{s}|t{t}"
        for tier in MODES:
            cellrows = [r for r in rows if filt(sp, s, t)(r) and tier_of(r) == tier]
            key = f"{tag}|{tier}"
            out["cells"][key] = cell_stats(cellrows, hitfn)
            physical["cells"][key] = physical_stats(cellrows)
        key = f"{tag}|tube_only_vs_none"
        out["paired"][key] = paired_test(rows, hitfn, "tube_only", "none",
                                         filterfn=filt(sp, s, t))
        physical["paired"][key] = paired_test_physical(
            rows, "tube_only", "none", filterfn=filt(sp, s, t))
        key = f"{tag}|full_vs_softmin_only"
        out["paired"][key] = paired_test(rows, hitfn, "full", "softmin_only",
                                         filterfn=filt(sp, s, t))
        physical["paired"][key] = paired_test_physical(
            rows, "full", "softmin_only", filterfn=filt(sp, s, t))
    return out, physical


def _did_reason(
    relevant: list[dict[str, str]],
    errors: list[dict[str, str]],
    verdicts: int,
    semantics: str,
) -> str:
    """DiD 不可估计原因（区分缺行、全 error、无判定、无公共 seed）。"""
    if not relevant:
        return "no_rows"
    if errors and len(errors) == len(relevant):
        return "all_error"
    if verdicts == 0:
        return "no_physical_verdict" if semantics == "physical" else "no_verdict"
    return "no_common_seed"


def _did(
    rows: list[dict[str, str]],
    value_fn: Callable[[dict[str, str]], int | None],
    baseline: str = "none",
    *,
    semantics: str = "legacy",
) -> dict[str, Any]:
    """配对 DiD：每 seed 的 (档 - 基线) 增益在 TCP1.0/TCP1.8 两层之差。

    R06/M04：DiD 的四个条件（档 × TCP 层 × 是否基线）各自建立唯一索引，
    同一 seed 在同一条件内取值相同则去重、取值不同则整键排除；只有四个索引
    都存在的 seed 才进入 DiD。因此结果与 CSV 行序无关，也不会出现“普通配对
    已判 duplicate_seed_conflict、DiD 却仍给出显著差异”的自相矛盾。

    Args:
        rows: 同一实验的记录行。
        value_fn: 命中判定函数（legacy ``HitPolicy`` 或 ``physical_of``）。
        baseline: 基线档名（默认 ``none``）。
        semantics: ``legacy`` / ``physical``；physical 全 miss 时保留样本数与观察差
            （可为 0.0），但 CI/p 推断字段置 None 并标 ``unavailable_reason="all_miss"``。

    Returns:
        档名 → DiD 结果（``n_pairs`` 恒在；有冲突键时附
        ``excluded_duplicate_keys``；无法估计时给 ``unavailable_reason``，
        若因重复 seed 冲突导致无样本则为 ``duplicate_seed_conflict``）。
    """
    out: dict[str, Any] = {}
    for a in ("softmin_only", "full", "tube_only"):
        relevant = [r for r in rows if tier_of(r) in (a, baseline)]
        errors = [r for r in relevant if is_error(r)]
        usable = [r for r in relevant if not is_error(r)]
        verdicts = sum(1 for r in usable if value_fn(r) is not None)
        # 四个条件各自去重取值：同值去重、异值整键排除（行序无关）
        indexes: dict[tuple[str, str], dict[Hashable, int]] = {}
        raw_keys: list[set[Hashable]] = []
        excluded: set[Hashable] = set()
        for tcp_label in ("TCP1.0", "TCP1.8"):
            for tier in (a, baseline):
                subset = [
                    r for r in usable
                    if tcp_label_of(r) == tcp_label and tier_of(r) == tier
                ]
                index, conflicts = _unique_index(subset, _seed_value, value_fn)
                indexes[(tcp_label, tier)] = index
                raw_keys.append(set(index) | set(conflicts))
                excluded.update(conflicts)
        common = set(indexes[("TCP1.0", a)])
        for index in indexes.values():
            common &= set(index)
        dids = [
            (indexes[("TCP1.0", a)][seed] - indexes[("TCP1.0", baseline)][seed])
            - (indexes[("TCP1.8", a)][seed] - indexes[("TCP1.8", baseline)][seed])
            for seed in sorted(common)
        ]
        excluded_keys = sorted(excluded)
        entry: dict[str, Any] = {"n_pairs": len(dids), "tier": a, "baseline": baseline}
        if semantics == "physical":
            entry["hit_semantics"] = "physical"
        if excluded_keys:
            entry["excluded_duplicate_keys"] = excluded_keys
        if not dids:
            entry["mean_did_pp"] = None
            entry["ci95"] = None
            entry["p_sign"] = None
            if excluded_keys and set.intersection(*raw_keys):
                # 四个条件本来有共同 seed，但冲突把它们全部排除：
                # 这与普通配对的 duplicate_seed_conflict 保持一致，不输出显著性
                reason = "duplicate_seed_conflict"
            else:
                reason = _did_reason(relevant, errors, verdicts, semantics)
            entry["unavailable_reason"] = reason
            out[a] = entry
            continue
        entry["mean_did_pp"] = 100 * mean(dids)
        total_hits = sum(
            1 for index in indexes.values() for value in index.values() if value == 1
        )
        if semantics == "physical" and total_hits == 0:
            # 全 miss 是测得的 0%，但不能据此声称"无差异"：不估计区间与符号检验。
            entry["ci95"] = None
            entry["p_sign"] = None
            entry["unavailable_reason"] = "all_miss"
            out[a] = entry
            continue
        rng = random.Random(42)
        n = len(dids)
        boot = sorted(
            sum(dids[i] for i in [rng.randrange(n) for _ in range(n)]) / n
            for _ in range(20000))
        entry["ci95"] = [100 * boot[int(0.025 * 20000)],
                         100 * boot[int(0.975 * 20000)]]
        entry["p_sign"] = sign_test(dids)
        out[a] = entry
    return out


def e8_limit_mechanism(
    hitfn: HitPolicy, ctx: StatsContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """E8: 限速 × 机制配对（exp17i）+ 配对 DiD。"""
    rows = _load_rows(ctx, "exp17i_limits_ablation")
    out: dict[str, Any] = {"cells": {}, "paired": {}, "did": {}, "diagnostics": {}}
    physical: dict[str, Any] = {"cells": {}, "paired": {}, "did": {}}
    for lim in ("TCP1.8", "TCP1.0"):
        for tier in MODES:
            sub = [r for r in rows
                   if tcp_label_of(r) == lim and tier_of(r) == tier]
            key = f"{lim}|{tier}"
            out["cells"][key] = cell_stats(sub, hitfn)
            physical["cells"][key] = physical_stats(sub)
            # 命中行的诊断量：时间误差、击球瞬间拍速
            hitrows = [r for r in sub if not is_error(r) and hitfn(r) == 1]
            errs = [float(r["hit_time_error_ms"]) for r in hitrows
                    if (r.get("hit_time_error_ms") or "")]
            vels = [float(r["v_racket_at_hit"]) for r in hitrows
                    if (r.get("v_racket_at_hit") or "")]
            out["diagnostics"][key] = {
                "n_hit_rows": len(hitrows),
                "mean_hit_time_error_ms": mean(errs) if errs else float("nan"),
                "mean_v_racket": mean(vels) if vels else float("nan"),
            }
        for a, b in (("softmin_only", "none"), ("full", "none"),
                     ("tube_only", "none"), ("full", "softmin_only")):
            key = f"{lim}|{a}_vs_{b}"
            filterfn = lambda r, lim=lim: tcp_label_of(r) == lim  # noqa: E731
            out["paired"][key] = paired_test(rows, hitfn, a, b, filterfn=filterfn)
            physical["paired"][key] = paired_test_physical(
                rows, a, b, filterfn=filterfn)
    # 配对 DiD：每 seed 的 (档 - 基线) 增益在两层之差
    legacy_did = _did(rows, hitfn, semantics="legacy")
    phys_did = _did(rows, physical_of, semantics="physical")
    for a, d in legacy_did.items():
        g1 = out["paired"].get(f"TCP1.0|{a}_vs_none", {})
        g0 = out["paired"].get(f"TCP1.8|{a}_vs_none", {})
        d["gain_TCP18_pp"] = g0.get("diff_pp", float("nan"))
        d["gain_TCP10_pp"] = g1.get("diff_pp", float("nan"))
    out["did"] = legacy_did
    for a, d in phys_did.items():
        g1 = physical["paired"].get(f"TCP1.0|{a}_vs_none", {})
        g0 = physical["paired"].get(f"TCP1.8|{a}_vs_none", {})
        d["gain_TCP18_pp"] = g0.get("diff_pp")
        d["gain_TCP10_pp"] = g1.get("diff_pp")
        d["hit_semantics"] = "physical"
    physical["did"] = phys_did
    return out, physical


def e9_shared_input(
    hitfn: HitPolicy, ctx: StatsContext
) -> tuple[dict[str, Any], dict[str, Any]]:
    """E9: shared-input 协议（exp24_shared_full；table3 数据源，旧 JSON 中无此节）。"""
    out: dict[str, Any] = {"cells": {}, "paired": {}, "data_sources": []}
    physical: dict[str, Any] = {"cells": {}, "paired": {}}
    rows: list[dict[str, str]] = []
    for name in ("exp24_shared_full", "exp22_shared_spatial"):
        try:
            loaded = _load_rows(ctx, name)
        except FileNotFoundError:
            continue
        out["data_sources"].append(name)
        rows.extend(loaded)
    if not rows:
        out["unavailable_reason"] = "missing exp24_shared_full / exp22_shared_spatial"
        physical["unavailable_reason"] = out["unavailable_reason"]
        return out, physical

    def cell_of(r: dict[str, str]) -> tuple[int, float]:
        return time_perturb_ms_of(r), space_perturb_of(r)

    def grid_key(r: dict[str, str]) -> str:
        t, s = cell_of(r)
        return f"{t}|{s}|{r['seed']}"

    for t, s in sorted({cell_of(r) for r in rows}):
        sub = [r for r in rows if cell_of(r) == (t, s)]
        for tier in MODES:
            tier_rows = [r for r in sub if tier_of(r) == tier]
            if not tier_rows:
                continue
            key = f"t{t}|s{s}|{tier}"
            out["cells"][key] = cell_stats(tier_rows, hitfn)
            physical["cells"][key] = physical_stats(tier_rows)
        for a, b in (("full", "none"), ("tube_only", "none"),
                     ("softmin_only", "none"), ("full", "softmin_only")):
            key = f"t{t}|s{s}|{a}_vs_{b}"
            out["paired"][key] = paired_test(sub, hitfn, a, b, keyfn=grid_key)
            physical["paired"][key] = paired_test_physical(
                sub, a, b, keyfn=grid_key)
    return out, physical


def exclusion_report(ctx: StatsContext) -> dict[str, Any]:
    """排除行逐档计数（验证排除与方法无关）。"""
    out: dict[str, Any] = {}
    for name in ("exp15_speed_v2", "exp16_limits_v2", "exp17a_noise",
                 "exp17b_perturb", "exp17c_obsfreq", "exp17d_mechanism",
                 "exp17e_perturb7", "exp17f_mechanism_perturb",
                 "exp17g_spatial_power", "exp17h_extreme",
                 "exp17i_limits_ablation"):
        try:
            rows = _load_rows(ctx, name)
        except FileNotFoundError:
            continue
        by_tier: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # [valid, error]
        causes: Counter[str] = Counter()
        for r in rows:
            tier = tier_of(r) or "-"
            if is_error(r):
                by_tier[tier][1] += 1
                causes[(r.get("error") or "").strip()] += 1
            else:
                by_tier[tier][0] += 1
        total_err = sum(causes.values())
        out[name] = {
            "rows": len(rows),
            "errors": total_err,
            "error_rate_pct": 100 * total_err / len(rows) if rows else 0.0,
            "causes": dict(causes),
            "per_tier": {k: {"valid": v[0], "error": v[1]} for k, v in by_tier.items()},
        }
    return out


# ----------------------------------------------------------------------------
# 表格数据（T8c 的 table1/2/3 数据源；只产数据不画图）
# ----------------------------------------------------------------------------

def _tier_cell(cells: dict[str, Any], key: str, tier: str) -> dict[str, Any]:
    """从 legacy cells 与 physical cells 组合出表格单元（含 missing/n 标记）。"""
    legacy_cell = cells.get(f"{key}|{tier}")
    if legacy_cell is None:
        return {"present": False, "tier": tier, "tier_label": TIER_LABELS[tier],
                "unavailable_reason": "missing_cell"}
    return legacy_cell


def build_tables(
    legacy: dict[str, Any], physical: dict[str, Any]
) -> dict[str, Any]:
    """构造 table1/2/3 的稳定数据键（列映射见 column_map；不画图）。"""
    t1_rows: list[dict[str, Any]] = []
    table1_specs = [
        ("nominal 7 m/s", "E3_nominal", "7"),
        ("nominal 9 m/s", "E3_nominal", "9"),
        ("nominal 12 m/s", "E3_nominal", "12"),
        ("perturbed grid (20 cells, 9 m/s)", "E4_grid", "merged"),
        ('corner 12 m/s t=0 s=0.2 m', "E7_corners", "12|s0.2|t0"),
    ]
    for label, section, prefix in table1_specs:
        cells_legacy = legacy.get(section) or {}
        cells_physical = physical.get(section) or {}
        row: dict[str, Any] = {"condition": label, "source": f"{section}:{prefix}", "cells": {}}
        n_values: list[int] = []
        for tier in MODES:
            if section == "E4_grid":
                leg = cells_legacy.get("merged_20cell", {}).get(tier, {})
                phy = cells_physical.get("merged_20cell", {}).get(tier, {})
            else:
                leg = cells_legacy.get("cells", {}).get(f"{prefix}|{tier}", {})
                phy = cells_physical.get("cells", {}).get(f"{prefix}|{tier}", {})
            present = bool(leg)
            if present:
                n_values.append(int(leg.get("n_valid") or 0))
            row["cells"][tier] = {
                "present": present,
                "tier_label": TIER_LABELS[tier],
                "legacy_rate": leg.get("rate"),
                "legacy_n": leg.get("n_valid"),
                "legacy_hits": leg.get("hits"),
                "physical_rate": phy.get("rate"),
                "physical_denominator": phy.get("denominator"),
                "physical_n_active": phy.get("n_active"),
                "unavailable_reason": phy.get("unavailable_reason"),
            }
        if n_values:
            row["n_range"] = [min(n_values), max(n_values)] if min(n_values) != max(n_values) \
                else [n_values[0], n_values[0]]
        else:
            row["n_range"] = None
            row["unavailable_reason"] = "missing_cell"
        t1_rows.append(row)

    t2_rows: list[dict[str, Any]] = []
    e8_legacy = legacy.get("E8_limit_mechanism") or {}
    e8_physical = physical.get("E8_limit_mechanism") or {}
    for tier in ("full", "tube_only", "softmin_only"):
        entry: dict[str, Any] = {"tier": tier, "tier_label": TIER_LABELS[tier]}
        for label in ("TCP1.8", "TCP1.0"):
            entry[f"gain_{label}_legacy"] = e8_legacy.get("paired", {}).get(
                f"{label}|{tier}_vs_none", {})
            entry[f"gain_{label}_physical"] = e8_physical.get("paired", {}).get(
                f"{label}|{tier}_vs_none", {})
        entry["did_legacy"] = e8_legacy.get("did", {}).get(tier, {})
        entry["did_physical"] = e8_physical.get("did", {}).get(tier, {})
        t2_rows.append(entry)

    t3_rows: list[dict[str, Any]] = []
    shared_legacy = legacy.get("E9_shared_input", {})
    shared_physical = physical.get("E9_shared_input", {})
    for t, s in TABLE3_CELLS:
        row = {"cell": f"t{t:g}|s{s:g}", "t": t, "s": s, "cells": {}, "missing": []}
        for tier in MODES:
            key = f"t{t:g}|s{s:g}|{tier}"
            leg = shared_legacy.get("cells", {}).get(key)
            phy = shared_physical.get("cells", {}).get(key)
            if leg is None:
                row["missing"].append(tier)
                row["cells"][tier] = {"present": False, "tier_label": TIER_LABELS[tier],
                                      "unavailable_reason": "missing_cell"}
                continue
            row["cells"][tier] = {
                "present": True,
                "tier_label": TIER_LABELS[tier],
                "legacy_rate": leg.get("rate"),
                "legacy_n": leg.get("n_valid"),
                "physical_rate": phy.get("rate") if phy else None,
                "physical_denominator": phy.get("denominator") if phy else 0,
                "unavailable_reason": phy.get("unavailable_reason") if phy else "no_physical_verdict",
            }
        p = shared_legacy.get("paired", {}).get(f"t{t:g}|s{s:g}|full_vs_none", {})
        row["p_raw_legacy"] = p.get("p_mcnemar")
        row["n_paired_legacy"] = p.get("n_paired")
        pp = shared_physical.get("paired", {}).get(f"t{t:g}|s{s:g}|full_vs_none", {})
        row["p_raw_physical"] = pp.get("p_mcnemar")
        row["n_paired_physical"] = pp.get("n_paired")
        t3_rows.append(row)

    return {
        "table1_comparison": {
            "description": "消融四档命中率（对应 paper/figures/table_data/table1_comparison.tex，"
                           "文件名与内容错位：该文件实为表 II ablation 表）",
            "columns": ["condition", "full", "tube-only", "candidate-set-only",
                        "point-target", "n", "design"],
            "rows": t1_rows,
        },
        "table2_limits": {
            "description": "限速层配对增益与 DiD（对应 table2_ablation.tex，实为表 I limits 表）",
            "columns": ["tier", "gain_TCP1.8", "gain_TCP1.0", "did"],
            "rows": t2_rows,
        },
        "table3_shared_input": {
            "description": "shared-input 协议（对应 table3_shared_input.tex，数据源 exp24_shared_full）",
            "columns": ["cell", "full", "tube-only", "candidate-set-only",
                        "point-target", "p_raw"],
            "rows": t3_rows,
        },
        "column_map": {
            "table1_comparison.tex": {
                "columns": {
                    "full": "tables.table1_comparison.rows[].cells.full.legacy_rate",
                    "tube-only": "tables.table1_comparison.rows[].cells.tube_only.legacy_rate",
                    "candidate-set-only": "tables.table1_comparison.rows[].cells.softmin_only.legacy_rate",
                    "point-target": "tables.table1_comparison.rows[].cells.none.legacy_rate",
                    "n": "tables.table1_comparison.rows[].n_range",
                },
                "note": "physical_rate 为 R1D 新口径（旧 RM65 根为 null + unavailable_reason）",
            },
            "table2_ablation.tex": {
                "columns": {
                    "Gain over point-target (pp)": "tables.table2_limits.rows[].gain_TCP1.8_legacy.diff_pp",
                    "p": "tables.table2_limits.rows[].gain_TCP1.8_legacy.p_mcnemar",
                    "DiD": "tables.table2_limits.rows[].did_legacy",
                },
                "note": "上块为 E8 paired legacy；下块为 E8 did legacy；physical 为并列新增",
            },
            "table3_shared_input.tex": {
                "columns": {
                    "$(t,s)$ cell": "tables.table3_shared_input.rows[].cell",
                    "full": "tables.table3_shared_input.rows[].cells.full.legacy_rate",
                    "tube-only": "tables.table3_shared_input.rows[].cells.tube_only.legacy_rate",
                    "candidate-set-only": "tables.table3_shared_input.rows[].cells.softmin_only.legacy_rate",
                    "point-target": "tables.table3_shared_input.rows[].cells.none.legacy_rate",
                    "$p_{raw}$": "tables.table3_shared_input.rows[].p_raw_legacy",
                },
                "note": "缺失 (t,s)/tier 在 rows[].missing / cells[*].present=False 中显式标注",
            },
        },
    }


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def default_output_dir(data_root: Path) -> Path:
    """默认输出目录：数据根的兄弟目录 ``stats/``（绝不写回 data_root 或 paper/）。"""
    return data_root.parent / "stats"


def _catalog_metadata(ctx: StatsContext) -> dict[str, Any] | None:
    """summary 中的 catalog 元数据（无 catalog 时为 None，旧调用结构不变）。"""
    if ctx.catalog is None:
        return None
    return {
        "path": str(ctx.catalog_path) if ctx.catalog_path is not None else None,
        "schema_version": int(getattr(ctx.catalog, "schema_version", 0)),
        "robot_id": str(getattr(ctx.catalog, "robot_id", "")),
        "run_root": str(ctx.catalog.run_root),
        "data_root": str(ctx.catalog.data_root),
        "path_base": str(getattr(ctx.catalog, "path_base", "")),
        "source_sets": dict(ctx.source_sets),
        "synthetic": bool(ctx.synthetic),
    }


def compute_summary(
    data_root: Path | None = None,
    include_passive: bool = False,
    *,
    catalog_path: Path | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    """计算完整统计 summary（不写盘；供主流程与测试复用）。

    Args:
        data_root: 实验数据根；None 时用 catalog 声明的 data_root，再退回默认根。
        include_passive: 是否把 passive 计入 legacy 命中（复现旧口径）。
        catalog_path: 可选 ``catalog.json``；显式坏 catalog 抛 StatsError（不回退）。
        mode: ``torque`` / ``position`` / None（自动推断；混合模式必须显式指定）。

    Returns:
        统计 summary。既有键结构不变，只新增 ``mode`` / ``catalog`` / ``synthetic`` /
        ``validation`` 元数据，以及 ``E3_nominal.data_sources``。

    Raises:
        StatsError: catalog 非法、模式混合未指定、记录重复或身份冲突。
    """
    ctx = make_context(data_root, catalog_path=catalog_path, mode=mode)
    resolved = resolve_mode(ctx)
    ctx = ctx.with_mode(resolved)
    validation = _validate_dataset(ctx)
    hitfn = HitPolicy(include_passive=include_passive)
    metric = "hit(active+passive)" if include_passive else "active-hit"

    summary: dict[str, Any] = {"metric": metric, "include_passive": include_passive}
    physical: dict[str, Any] = {}
    unavailable: list[str] = []
    for key, builder in (
        ("E1_speed_sweep", e1_speed_sweep),
        ("E2_limit_cost", e2_limit_cost),
        ("E3_nominal", e3_nominal),
        ("E4_grid", e4_grid),
        ("E5_perception", e5_perception),
        ("E7_corners", e7_corners),
        ("E8_limit_mechanism", e8_limit_mechanism),
    ):
        try:
            legacy_section, physical_section = builder(hitfn, ctx)
        except FileNotFoundError as exc:
            unavailable.append(f"{key}: {exc}")
            legacy_section, physical_section = {}, {}
        summary[key] = legacy_section
        physical[key] = physical_section
    summary["exclusion"] = exclusion_report(ctx)
    legacy_e9, physical_e9 = e9_shared_input(hitfn, ctx)
    summary["E9_shared_input"] = legacy_e9
    physical["E9_shared_input"] = physical_e9
    if legacy_e9.get("unavailable_reason"):
        unavailable.append(f"E9_shared_input: {legacy_e9['unavailable_reason']}")

    summary["mode"] = resolved
    summary["catalog"] = _catalog_metadata(ctx)
    summary["synthetic"] = bool(ctx.synthetic)
    summary["validation"] = validation
    summary["hit_semantics"] = HIT_SEMANTICS
    summary["physical"] = physical
    summary["tables"] = build_tables(summary, physical)
    summary["schema"] = SCHEMA
    summary["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    summary["data_root"] = str(ctx.data_root.resolve())
    summary["sources"] = _existing_sources(ctx)
    summary["unavailable"] = unavailable
    return summary


def _existing_sources(ctx: StatsContext) -> list[str]:
    """列出本次实际读取过的 CSV 路径（供 T8c 追溯数据来源）。"""
    paths: list[Path] = []
    runs_csv = _runs_csv_path(ctx)
    if runs_csv is not None and runs_csv.is_file():
        paths.append(runs_csv)
    for csv_path in sorted(ctx.data_root.glob("*/results.csv")):
        paths.append(csv_path)
    return [str(path) for path in paths]


def build_parser() -> argparse.ArgumentParser:
    """构造命令行解析器（--catalog / --mode / --data-root / --output / --dry-run）。"""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--include-passive", action="store_true",
                        help="把被动接触也计入命中（复现旧口径，用于对照）")
    parser.add_argument("--data-root", type=Path, default=None,
                        help=f"实验数据根（默认 {DEFAULT_DATA_ROOT}）")
    parser.add_argument("--catalog", type=Path, default=None,
                        help="catalog.json 路径；显式坏 catalog 退出码 2（不回退旧逻辑）")
    parser.add_argument("--mode", choices=MODE_CHOICES, default=None,
                        help="执行模式 torque/position/both；混合数据必须显式选择")
    parser.add_argument("--output", type=Path, default=None,
                        help="输出目录（给出时写 <output>/06-stats-active.json 与 "
                             "paper_stats_active.json；--mode both 时写 <output>/<mode>/）")
    parser.add_argument("--dry-run", action="store_true",
                        help="只计算并打印，不写任何文件")
    return parser


def _print_report(summary: dict[str, Any]) -> None:
    """打印可读报告（保持旧输出小节；分模式时由 main 加标题）。"""
    print(f"命中口径: {summary['metric']}")
    print(f"数据根: {summary['data_root']}")
    if summary.get("mode"):
        print(f"执行模式: {summary['mode']}")
    catalog = summary.get("catalog") or {}
    if catalog:
        print(f"catalog: {catalog.get('path')}")
    if summary["unavailable"]:
        print(f"不可用数据: {summary['unavailable']}")
    print()

    # ---- 打印可读报告（保持旧输出小节） ----
    print("=== E1 球速扫描（active-hit %，含 95% CI 与 n）===")
    for sp, st in summary["E1_speed_sweep"].items():
        if st.get("rate") is None:
            print(f"  {sp:>2} m/s: 不可估计（{_reason_text(st)}）")
            continue
        print(f"  {sp:>2} m/s: {st['rate']:.1f}% [{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}] "
              f"n={st['n_valid']} passive={st['passive']} ({st['passive_share']:.1f}%)")

    print("\n=== E2 限位代价（exp16, 7 m/s）===")
    for key in ("TCP1.8", "TCP1.0"):
        st = summary["E2_limit_cost"].get(key, {})
        if st:
            if st.get("rate") is None:
                print(f"  {key}: 不可估计（{_reason_text(st)}）")
                continue
            print(f"  {key}: {st['rate']:.1f}% n={st['n_valid']} "
                  f"v_racket={_fmt_optional(st.get('mean_racket_speed_on_hit'), '{:.2f}')} m/s")
    p = summary["E2_limit_cost"].get("paired_TCP10_vs_TCP18", {})
    if p.get("n_paired"):
        print(f"  配对差: {_fmt_optional(p.get('diff_pp'), '{:+.1f}')}pp "
              f"CI={_fmt_ci(p.get('ci95'))} p={_fmt_p(p.get('p_mcnemar'), '{:.2e}')}")

    print("\n=== E3 标称四档（active-hit %）===")
    e3_cells = summary["E3_nominal"].get("cells", {})
    for sp in (7, 9, 12):
        vals = []
        for tier in MODES:
            st = e3_cells.get(f"{sp}|{tier}", {})
            vals.append(f"{tier}={_fmt_optional(st.get('rate'), '{:.1f}')}%"
                        f"(n={st.get('n_valid', 0)})")
        print(f"  {sp} m/s: " + " ".join(vals))
        for a, b in (("full", "none"), ("softmin_only", "none"),
                     ("tube_only", "none"), ("full", "softmin_only")):
            t = summary["E3_nominal"].get("paired", {}).get(f"{sp}|{a}_vs_{b}", {})
            if t.get("n_paired"):
                print(f"    {a} vs {b}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                      f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))}")

    print("\n=== E4 扰动网格（20 格合并, 9 m/s）===")
    for tier in MODES:
        st = summary["E4_grid"].get("merged_20cell", {}).get(tier)
        if st:
            if st.get("rate") is None:
                print(f"  {tier}: 不可估计（{_reason_text(st)}）")
                continue
            print(f"  {tier}: {st['rate']:.1f}% [{st['ci95'][0]:.1f},{st['ci95'][1]:.1f}] "
                  f"n={st['n_valid']}")
    for comp, t in summary["E4_grid"].get("paired_20cell", {}).items():
        if t.get("n_paired"):
            print(f"  {comp}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                  f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))}")

    print("\n=== E7 角点（预先设定对比）===")
    for tag, t in summary["E7_corners"].get("paired", {}).items():
        if t.get("n_paired"):
            print(f"  {tag}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                  f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))} "
                  f"n={t.get('n_paired')}")

    print("\n=== E8 限速 × 机制（exp17i）===")
    e8_cells = summary["E8_limit_mechanism"].get("cells", {})
    for lim in ("TCP1.8", "TCP1.0"):
        vals = []
        for tier in MODES:
            st = e8_cells.get(f"{lim}|{tier}", {})
            vals.append(f"{tier}={_fmt_optional(st.get('rate'), '{:.1f}')}%")
        print(f"  {lim}: " + " ".join(vals))
    for comp, t in summary["E8_limit_mechanism"].get("paired", {}).items():
        if t.get("n_paired"):
            print(f"    {comp}: {_fmt_optional(t.get('diff_pp'), '{:+.2f}')}pp "
                  f"CI={_fmt_ci(t.get('ci95'))} p={_fmt_p(t.get('p_mcnemar'))}")
    for a, d in summary["E8_limit_mechanism"].get("did", {}).items():
        print(f"    DiD {a}: gain {_fmt_pp(d.get('gain_TCP18_pp'))} → "
              f"{_fmt_pp(d.get('gain_TCP10_pp'))} pp, "
              f"mean DiD={_fmt_pp(d.get('mean_did_pp'))}pp CI={_fmt_ci(d.get('ci95'))} "
              f"sign p={_fmt_p(d.get('p_sign'), '{:.2e}')}")
    print("  诊断（命中行均值）:")
    for key, d in summary["E8_limit_mechanism"].get("diagnostics", {}).items():
        print(f"    {key}: n_hit={d['n_hit_rows']} "
              f"t_err={_fmt_optional(d.get('mean_hit_time_error_ms'), '{:.1f}')}ms "
              f"v={_fmt_optional(d.get('mean_v_racket'), '{:.2f}')}m/s")

    print("\n=== 排除行统计（error 均为 no_result）===")
    for name, st in summary["exclusion"].items():
        tiers = ", ".join(f"{k}:{v['error']}/{v['valid']+v['error']}"
                          for k, v in st["per_tier"].items())
        print(f"  {name}: 排除 {st['errors']}/{st['rows']} "
              f"({st['error_rate_pct']:.2f}%) [{tiers}]")

    e9_cells = summary["E9_shared_input"].get("cells") or {}
    if e9_cells:
        print("\n=== E9 shared-input（table3 数据源）===")
        for key, st in sorted(e9_cells.items()):
            print(f"  {key}: {_fmt_optional(st.get('rate'), '{:.1f}')}% "
                  f"(n={st.get('n_valid', 0)})")

def _write_stats(out_dir: Path, summary: dict[str, Any]) -> list[Path]:
    """写出一份 summary 的两份同内容 JSON（06-stats-active / paper_stats_active）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    written: list[Path] = []
    for filename in STATS_FILENAMES:
        target = out_dir / filename
        target.write_text(payload, encoding="utf-8")
        written.append(target)
    return written


def _multi_mode_index(
    out_dir: Path, summaries: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """构造 ``stats/modes.json`` 索引（只索引，不复制统计内容）。"""
    modes: dict[str, Any] = {}
    for mode, summary in summaries.items():
        validation = summary.get("validation") or {}
        unavailable = list(summary.get("unavailable") or [])
        records = int(validation.get("mode_records") or 0)
        modes[mode] = {
            "directory": mode,
            "files": [f"{mode}/{filename}" for filename in STATS_FILENAMES],
            # ok 只代表该模式的数据定位/唯一性通过，不要求所有实验节都有数据。
            "validation": {
                "ok": (
                    records > 0
                    and not validation.get("duplicates")
                    and not validation.get("physical_conflicts")
                ),
                "records": records,
                "results_records": int(validation.get("results_records") or 0),
                "runs_records": int(validation.get("runs_records") or 0),
                "manifest_missing": int(validation.get("manifest_missing") or 0),
                "duplicates": 0,
                "unavailable_sections": len(unavailable),
                "unavailable": unavailable,
            },
            "statistics": {
                "E3_cells": len((summary.get("E3_nominal") or {}).get("cells") or {}),
                "E8_did_tiers": sorted(
                    ((summary.get("E8_limit_mechanism") or {}).get("did") or {})),
            },
        }
    first = next(iter(summaries.values())) if summaries else {}
    return {
        "schema": MULTI_MODE_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "data_root": first.get("data_root"),
        "catalog": first.get("catalog"),
        "modes": modes,
        "note": "本文件只是分模式产物的索引；统计内容以各模式目录下的 "
                f"{' / '.join(STATS_FILENAMES)} 为准",
    }


def _physical_conflict_summary(summaries: dict[str, dict[str, Any]]) -> tuple[int, list[str]]:
    """汇总各模式的富化冲突数与样例，供 CLI 以非零码明确拒绝。"""
    total = 0
    samples: list[str] = []
    for mode, summary in summaries.items():
        validation = summary.get("validation") or {}
        count = int(validation.get("physical_conflicts") or 0)
        if count <= 0:
            continue
        total += count
        for item in (validation.get("physical_conflict_samples") or [])[:3]:
            samples.append(
                f"{mode}:{item.get('experiment')}/{item.get('case')}/"
                f"{item.get('config_id')} seed={item.get('seed')} "
                f"fields={','.join(item.get('fields') or [])}"
            )
    return total, samples


def _reject_physical_conflicts(summaries: dict[str, dict[str, Any]]) -> int:
    """有富化冲突时打印明确错误并返回 2，避免冲突数据被当作正常结果。"""
    total, samples = _physical_conflict_summary(summaries)
    if total <= 0:
        return 0
    print(
        f"[paired_stats] 错误：runs.csv 与 results.csv 有 {total} 条身份匹配但字段冲突，"
        "已拒绝用 runs.csv 的 physical 判定富化统计。",
        file=sys.stderr,
    )
    for sample in samples:
        print(f"  冲突样例：{sample}", file=sys.stderr)
    print(
        "  请修复 runs.csv/config_json 中不一致的字段，或删除对应 runs 行后重跑。",
        file=sys.stderr,
    )
    return 2


def main(argv: list[str] | None = None) -> int:
    """跑完全部统计并打印报告；仅在显式 --output 时写出 JSON（--dry-run 不写盘）。"""
    args = build_parser().parse_args(argv)
    data_root: Path = args.data_root if args.data_root is not None else DEFAULT_DATA_ROOT
    out_dir: Path = args.output if args.output is not None else default_output_dir(data_root)
    try:
        summaries: dict[str, dict[str, Any]] = {}
        if args.mode == "both":
            for mode in VALID_MODES:
                summaries[mode] = compute_summary(
                    args.data_root,
                    include_passive=args.include_passive,
                    catalog_path=args.catalog,
                    mode=mode,
                )
        else:
            single = compute_summary(
                args.data_root,
                include_passive=args.include_passive,
                catalog_path=args.catalog,
                mode=args.mode,
            )
            summaries[single.get("mode") or "unspecified"] = single
    except StatsError as exc:
        print(f"[paired_stats] 错误：{exc}", file=sys.stderr)
        return 2

    for mode, summary in summaries.items():
        if args.mode == "both":
            print(f"===== mode={mode} =====")
        _print_report(summary)
    if args.dry_run:
        print(f"\n[dry-run] 不写文件（目标目录 {out_dir}）")
        return _reject_physical_conflicts(summaries)
    if args.output is None:
        # 未显式给 --output：只打印，绝不写默认根/paper（旧 paper/planning 不被覆盖）
        print(f"\n[paired_stats] 未给 --output：只打印，不写文件（默认目标 {out_dir}）")
        return _reject_physical_conflicts(summaries)
    if args.mode == "both":
        print(f"\n写出目标目录: {out_dir}")
        for mode, summary in summaries.items():
            for target in _write_stats(out_dir / mode, summary):
                print(f"  已写出 {target}")
        index_path = out_dir / MULTI_MODE_FILENAME
        index_path.write_text(
            json.dumps(_multi_mode_index(out_dir, summaries), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"  已写出 {index_path}")
        return _reject_physical_conflicts(summaries)
    summary = next(iter(summaries.values()))
    print(f"\n写出目标目录: {out_dir}")
    for target in _write_stats(out_dir, summary):
        print(f"  已写出 {target}")
    return _reject_physical_conflicts(summaries)


if __name__ == "__main__":
    raise SystemExit(main())
