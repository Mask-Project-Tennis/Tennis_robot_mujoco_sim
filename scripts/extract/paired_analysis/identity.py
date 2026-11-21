"""记录身份解析：配置 JSON → tier/case/mode/seed/config_id/design 指纹。

`RecordIdentity` 的唯一键与连接键定义见 :mod:`paired_analysis.types`；本模块只负责
从 CSV 行（含 config_json 与 --task-overrides）还原科学身份，不做数据校验。
"""
from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .types import (
    MODES,
    NOFEEDFORWARD_SUFFIX,
    SEED_DIR_RE,
    VALID_MODES,
    RecordIdentity,
    _to_bool_or_none,
    cfg_of,
)


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
        return not (flag is None or flag)
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
        return round(float(ov["mpc.time_perturb_s"]) * 1000.0)
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




def _tcp_label(value: float | int | str) -> str:  # noqa: PYI041  # 保持旧签名：--max-tcp 的 JSON 值可能为 int
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


__all__ = [
    "_PATH_CONFIG_KEYS",
    "_RUNTIME_CONFIG_KEYS",
    "_base_tier_of_cfg",
    "_design_fingerprint",
    "_feedforward_of",
    "_feedforward_of_cfg",
    "_is_r1d_row",
    "_normalize_config_path",
    "_obs_freq_of",
    "_ov",
    "_parse_override_value",
    "_row_artifact_value",
    "_seed_value",
    "_tcp_label",
    "case_of",
    "mode_of",
    "overrides_of",
    "record_identity",
    "shared_input_of",
    "space_perturb_of",
    "speed_of",
    "tcp_label_of",
    "tier_of",
    "time_perturb_ms_of",
]
