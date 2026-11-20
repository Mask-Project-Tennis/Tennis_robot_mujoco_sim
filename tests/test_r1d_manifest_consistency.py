"""R1-A7-D 协议 manifest ↔ 代码常量/硬件配置自动对照测试（阶段 4 评审修复）。

A-M5 / B-M2（评审 2026-11-17）：`configs/robots/r1d_protocol_manifest.yaml`
（Task 4.1 冻结的协议事实来源）与代码常量（传输层话题/域、类型层槽位数、
命令策略槽位组）及双侧硬件配置 YAML（`r1d_hardware_{left,right}.yaml`）
之间此前只有人工核对，无自动对照——任何一侧漂移（话题改名、槽位重排、
domain 变更）都不会被测试捕获。本文件把三层两两钉死：

    manifest  ↔  代码常量（transport/types/command_policy）
    manifest  ↔  硬件配置 YAML（communication 节 + layout.joint_slots）

对照读数全部取自文件原文（yaml.safe_load），不经加载器转换——本测试
校验的是"文档级冻结值一致"，加载器行为由各自单测覆盖。

说明：`r1d_command_policy` 的 `_LEFT_ARM_SLOTS` / `_RIGHT_ARM_SLOTS` /
`_HEAD_SLOTS` 是私有常量，但其模块注释已显式声明"仅供 manifest 一致性
对照测试（tests/test_r1d_manifest_consistency.py）与文档参考，不参与
运行时寻址"——本文件即该声明所指向的消费者，此导入是契约的一部分。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from src.real.r1d_command_policy import (
    _HEAD_SLOTS,
    _LEFT_ARM_SLOTS,
    _RIGHT_ARM_SLOTS,
)
from src.real.r1d_types import R1D_STATE_SLOT_COUNT
from src.real.r1d_transport import (
    R1D_DDS_DOMAIN_DEFAULT,
    R1D_LOWCMD_TOPIC,
    R1D_LOWSTATE_TOPIC,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
_MANIFEST_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_protocol_manifest.yaml"
_HARDWARE_PATHS = {
    "left": _REPO_ROOT / "configs" / "robots" / "r1d_hardware_left.yaml",
    "right": _REPO_ROOT / "configs" / "robots" / "r1d_hardware_right.yaml",
}

# 冻结序列化长度（manifest lowcmd_layout/lowstate_layout.serialized_length_bytes；
# 与 tests/test_r1d_sdk_serialization.py 的冻结对照字节数一致。注意 I-04 勘误：
# 该值是 CRC 打包结构像长度，本机 cyclonedds 11.1.0 原生 serialize 总长
# 与之不同——序列化证据边界见该测试；此处只钉 manifest 数值不漂移）
_FROZEN_LOWCMD_LENGTH_BYTES = 1004
_FROZEN_LOWSTATE_LENGTH_BYTES = 2092


@pytest.fixture(scope="module")
def manifest() -> dict[str, Any]:
    """加载协议 manifest（冻结事实来源）。"""
    with _MANIFEST_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope="module")
def hardware_configs() -> dict[str, dict[str, Any]]:
    """加载双侧硬件配置 YAML 原文。"""
    loaded: dict[str, dict[str, Any]] = {}
    for side, path in _HARDWARE_PATHS.items():
        with path.open(encoding="utf-8") as handle:
            loaded[side] = yaml.safe_load(handle)
    return loaded


# ---------------------------------------------------------------------------
# manifest ↔ 代码常量（A-M5）
# ---------------------------------------------------------------------------


def test_dds_domain_matches_transport_constant(manifest: dict[str, Any]) -> None:
    """manifest dds.domain ↔ 传输层默认域常量。"""
    assert manifest["dds"]["domain"] == R1D_DDS_DOMAIN_DEFAULT


@pytest.mark.parametrize(
    ("topic_key", "code_constant"),
    [("lowcmd", R1D_LOWCMD_TOPIC), ("lowstate", R1D_LOWSTATE_TOPIC)],
)
def test_topics_match_transport_constants(
    manifest: dict[str, Any], topic_key: str, code_constant: str
) -> None:
    """manifest dds.topics.{lowcmd,lowstate}.name ↔ 传输层话题常量。"""
    assert manifest["dds"]["topics"][topic_key]["name"] == code_constant


def test_slot_count_matches_types_constant(manifest: dict[str, Any]) -> None:
    """manifest whole_body.slot_count ↔ 类型层状态槽位数常量。"""
    assert manifest["whole_body"]["slot_count"] == R1D_STATE_SLOT_COUNT


@pytest.mark.parametrize(
    ("manifest_path", "code_slots"),
    [
        (("whole_body", "selected_arm", "joint_indices_left"), _LEFT_ARM_SLOTS),
        (("whole_body", "selected_arm", "joint_indices_right"), _RIGHT_ARM_SLOTS),
        (("whole_body", "other_arm", "joint_indices_left"), _LEFT_ARM_SLOTS),
        (("whole_body", "other_arm", "joint_indices_right"), _RIGHT_ARM_SLOTS),
    ],
    ids=["selected-left", "selected-right", "other-left", "other-right"],
)
def test_arm_slot_groups_match_policy_constants(
    manifest: dict[str, Any], manifest_path: tuple[str, ...], code_slots: tuple[int, ...]
) -> None:
    """manifest 双侧臂槽位组 ↔ 命令策略对照常量（顺序也须一致）。"""
    node: Any = manifest
    for key in manifest_path:
        node = node[key]
    assert tuple(node) == code_slots


def test_head_slots_match_policy_constant(manifest: dict[str, Any]) -> None:
    """manifest whole_body.head.joint_indices（按名）↔ 命令策略头槽常量。"""
    head = manifest["whole_body"]["head"]["joint_indices"]
    assert sorted(head.values()) == sorted(_HEAD_SLOTS)
    assert head["pitch"] == 29 and head["yaw"] == 30


@pytest.mark.parametrize(
    ("layout_key", "frozen_bytes"),
    [
        ("lowcmd_layout", _FROZEN_LOWCMD_LENGTH_BYTES),
        ("lowstate_layout", _FROZEN_LOWSTATE_LENGTH_BYTES),
    ],
)
def test_serialized_lengths_frozen(
    manifest: dict[str, Any], layout_key: str, frozen_bytes: int
) -> None:
    """manifest 序列化冻结长度不得漂移（I-04 勘误口径，见模块 docstring）。"""
    assert manifest["dds"][layout_key]["serialized_length_bytes"] == frozen_bytes


# ---------------------------------------------------------------------------
# manifest ↔ 硬件配置 YAML（B-M2）
# ---------------------------------------------------------------------------


def test_hardware_communication_matches_manifest_dds(
    manifest: dict[str, Any], hardware_configs: dict[str, dict[str, Any]]
) -> None:
    """双侧硬件 YAML communication 节 ↔ manifest dds 冻结值（domain/双话题）。"""
    dds = manifest["dds"]
    for side, config in hardware_configs.items():
        comm = config["communication"]
        assert comm["dds_domain"] == dds["domain"], f"{side} 侧 domain 漂移"
        assert comm["lowcmd_topic"] == dds["topics"]["lowcmd"]["name"], (
            f"{side} 侧 lowcmd 话题漂移"
        )
        assert comm["lowstate_topic"] == dds["topics"]["lowstate"]["name"], (
            f"{side} 侧 lowstate 话题漂移"
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_hardware_joint_slots_match_manifest_arm_indices(
    manifest: dict[str, Any], hardware_configs: dict[str, dict[str, Any]], side: str
) -> None:
    """硬件 YAML layout.joint_slots 值组 ↔ manifest 目标臂槽位（同侧、同序）。

    硬件配置按关节名寻址，槽位值的顺序由文件书写顺序给出；对照要求
    逐名一致且整体与 manifest 同侧 joint_indices 同序——防止槽位重排
    （如左右臂槽位互换）静默通过。
    """
    expected = manifest["whole_body"]["selected_arm"][f"joint_indices_{side}"]
    joint_slots = hardware_configs[side]["layout"]["joint_slots"]
    assert tuple(joint_slots.values()) == tuple(expected), (
        f"{side} 侧 joint_slots 与 manifest joint_indices_{side} 不一致"
    )


@pytest.mark.parametrize("side", ["left", "right"])
def test_hardware_joint_slots_unique_in_range(
    hardware_configs: dict[str, dict[str, Any]], side: str
) -> None:
    """硬件 YAML 槽位合法性独立校验：唯一、0..slot_count-1 域内。"""
    joint_slots = hardware_configs[side]["layout"]["joint_slots"]
    values = list(joint_slots.values())
    assert len(values) == len(set(values)) == 7, f"{side} 侧槽位须 7 个且不重复"
    assert all(0 <= v < R1D_STATE_SLOT_COUNT for v in values), (
        f"{side} 侧槽位越界 [0, {R1D_STATE_SLOT_COUNT})"
    )
