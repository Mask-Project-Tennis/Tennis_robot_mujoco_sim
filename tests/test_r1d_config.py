"""R1-A7-D 硬件配置加载与 purpose 校验测试（阶段 4 Task 4.2）。

本文件所有 sign/offset/限值/周期/网卡名均为**测试专用有限参数，非实机参数**：
仅用于验证加载与校验逻辑，不代表任何实物标定或实机承诺。
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.real.r1d_config import (
    R1DHardwareConfig,
    load_r1d_hardware_config,
    validate_r1d_hardware_config,
)
from src.robot.robot_profile import RobotProfile, load_robot_profile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEFT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"
_RIGHT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"
_LEFT_TEMPLATE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_hardware_left.yaml"
_RIGHT_TEMPLATE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_hardware_right.yaml"

# 左臂 7 关节名（与 r1d_left.yaml / 协议 manifest 槽位 15-21 同序）
_LEFT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
_LEFT_SLOTS = {name: 15 + i for i, name in enumerate(_LEFT_JOINT_NAMES)}


# ---------------------------------------------------------------------------
# 辅助构造（测试专用，非实机参数）
# ---------------------------------------------------------------------------


def _base_config_dict() -> dict[str, Any]:
    """返回结构完整的模板级配置字典（校准/限值全 null，非实机参数）。"""
    return {
        "schema_version": 1,
        "robot_id": "unitree_r1d",
        "side": "left",
        "communication": {
            "dds_domain": 0,
            "network_interface": None,
            "lowcmd_topic": "rt/lowcmd",
            "lowstate_topic": "rt/lowstate",
        },
        "layout": {"layout_verified": False, "joint_slots": dict(_LEFT_SLOTS)},
        "calibration": {
            "verified": False,
            "joint_signs": None,
            "joint_offsets_rad": None,
        },
        "limits": {
            "verified": False,
            "q_lower_rad": None,
            "q_upper_rad": None,
            "dq_limit_rad_s": None,
            "tau_limit_nm": None,
            "kp_nm_per_rad": None,
            "kd_nms_per_rad": None,
        },
        "timing": {
            "control_period_s": None,
            "state_timeout_s": None,
            "diagnostic_duration_s": None,
        },
    }


def _motion_ready_dict() -> dict[str, Any]:
    """返回全部硬件门槛满足的配置字典（测试专用有限参数，非实机参数）。"""
    cfg = _base_config_dict()
    cfg["communication"]["network_interface"] = "eth-test"  # 测试专用，非实机参数
    cfg["layout"]["layout_verified"] = True
    cfg["calibration"] = {
        "verified": True,
        "joint_signs": {n: 1 for n in _LEFT_JOINT_NAMES},  # 测试值，非实机标定
        "joint_offsets_rad": {n: 0.0 for n in _LEFT_JOINT_NAMES},  # 测试值
    }
    cfg["limits"] = {
        "verified": True,
        "q_lower_rad": {n: -1.5 for n in _LEFT_JOINT_NAMES},  # 测试值，非实机限位
        "q_upper_rad": {n: 1.5 for n in _LEFT_JOINT_NAMES},
        "dq_limit_rad_s": {n: 2.0 for n in _LEFT_JOINT_NAMES},
        "tau_limit_nm": {n: 30.0 for n in _LEFT_JOINT_NAMES},
        "kp_nm_per_rad": {n: 100.0 for n in _LEFT_JOINT_NAMES},
        "kd_nms_per_rad": {n: 2.0 for n in _LEFT_JOINT_NAMES},
    }
    cfg["timing"] = {
        "control_period_s": 0.002,  # 数值同官方示例，仅测试用，非实机承诺
        "state_timeout_s": 0.05,
        "diagnostic_duration_s": 5.0,
    }
    return cfg


def _write_config(tmp_path: Path, data: dict[str, Any], name: str = "hw.yaml") -> Path:
    """把配置字典写成 UTF-8 YAML 文件并返回路径。"""
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


def _load(tmp_path: Path, data: dict[str, Any]) -> R1DHardwareConfig:
    """写 YAML 并加载（便捷组合）。"""
    return load_r1d_hardware_config(_write_config(tmp_path, data))


def _left_profile() -> RobotProfile:
    """加载仓库左臂描述。"""
    return load_robot_profile(_LEFT_PROFILE_PATH)


# ---------------------------------------------------------------------------
# 模块纯净性
# ---------------------------------------------------------------------------


def test_module_import_is_pure() -> None:
    """import 配置模块不得加载 SDK/MuJoCo、不得启动线程（子进程独立验证）。

    I-09 修复（评审 2026-11-17）：原进程内断言是空操作——被测模块在文件
    顶层已 import，测试体再 import 命中 sys.modules 缓存，不执行模块体，
    断言不可能失败。参照 test_r1d_transport.py 先例改为子进程断言。
    """
    code = (
        "import sys\n"
        "import threading\n"
        "import src.real.r1d_config\n"
        "leaked = [m for m in sys.modules if m == 'unitree_sdk2py'\n"
        "          or m.startswith('unitree_sdk2py.') or m == 'cyclonedds'\n"
        "          or m.startswith('cyclonedds.') or m == 'mujoco'\n"
        "          or m.startswith('mujoco.')]\n"
        "assert not leaked, leaked\n"
        "assert threading.active_count() == 1, threading.enumerate()\n"
        "print('PURE_OK')\n"
    )
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(_REPO_ROOT),
        timeout=120,
    )
    assert result.returncode == 0, f"stderr={result.stderr}"
    assert "PURE_OK" in result.stdout


# ---------------------------------------------------------------------------
# 模板文件
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "template_path,profile_path,side",
    [
        (_LEFT_TEMPLATE_PATH, _LEFT_PROFILE_PATH, "left"),
        (_RIGHT_TEMPLATE_PATH, _RIGHT_PROFILE_PATH, "right"),
    ],
)
def test_templates_load_offline_valid(
    template_path: Path, profile_path: Path, side: str
) -> None:
    """模板初始状态：可离线解析，校准/限值/网卡/周期全 null 或未核验。"""
    config = load_r1d_hardware_config(template_path)
    assert config.side == side
    assert config.robot_id == "unitree_r1d"
    # 实物校准与实机限值初始 null（未知值不得代填）
    assert config.calibration.joint_signs is None
    assert config.calibration.joint_offsets_rad is None
    assert config.limits.q_lower_rad is None
    assert config.limits.kp_nm_per_rad is None
    # 布局与通信未核验/未确认
    assert config.layout.layout_verified is False
    assert config.communication.network_interface is None
    assert config.timing.control_period_s is None

    profile = load_robot_profile(profile_path)
    # offline：模板可解析即通过
    validate_r1d_hardware_config(config, profile, purpose="offline")
    # read_only / motion：模板门槛不足必须拒绝
    with pytest.raises(ValueError, match="network_interface"):
        validate_r1d_hardware_config(config, profile, purpose="read_only")
    with pytest.raises(ValueError, match="network_interface"):
        validate_r1d_hardware_config(config, profile, purpose="motion")


# ---------------------------------------------------------------------------
# 未知字段 / 缺失节
# ---------------------------------------------------------------------------


def test_unknown_top_level_field_rejected(tmp_path: Path) -> None:
    """顶层未知字段拒绝（防拼写错误静默生效）。"""
    data = _base_config_dict()
    data["bogus_top_key"] = 1
    with pytest.raises(ValueError, match="bogus_top_key"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    "section", ["communication", "layout", "calibration", "limits", "timing"]
)
def test_unknown_nested_field_rejected(section: str, tmp_path: Path) -> None:
    """各节内未知字段拒绝。"""
    data = _base_config_dict()
    data[section]["bogus_key"] = 1
    with pytest.raises(ValueError, match="bogus_key"):
        _load(tmp_path, data)


@pytest.mark.parametrize(
    "section", ["communication", "layout", "calibration", "limits", "timing"]
)
def test_missing_section_rejected(section: str, tmp_path: Path) -> None:
    """缺少必需节拒绝。"""
    data = _base_config_dict()
    del data[section]
    with pytest.raises(ValueError, match=section):
        _load(tmp_path, data)


def test_missing_side_field_rejected(tmp_path: Path) -> None:
    """缺少 side 顶层字段拒绝。"""
    data = _base_config_dict()
    del data["side"]
    with pytest.raises(ValueError, match="side"):
        _load(tmp_path, data)


def test_invalid_side_value_rejected(tmp_path: Path) -> None:
    """side 只能是 left/right。"""
    data = _base_config_dict()
    data["side"] = "middle"
    with pytest.raises(ValueError, match="side"):
        _load(tmp_path, data)


def test_unsupported_schema_version_rejected(tmp_path: Path) -> None:
    """不支持的 schema_version 拒绝。"""
    data = _base_config_dict()
    data["schema_version"] = 2
    with pytest.raises(ValueError, match="schema_version"):
        _load(tmp_path, data)


# ---------------------------------------------------------------------------
# 布局负例：重复 / 越界 / 漏第七关节
# ---------------------------------------------------------------------------


def test_duplicate_slot_index_rejected(tmp_path: Path) -> None:
    """两个关节映射到同一 SDK 槽位拒绝。"""
    data = _base_config_dict()
    data["layout"]["joint_slots"]["left_shoulder_roll_joint"] = 15  # 与 pitch 重复
    with pytest.raises(ValueError, match="重复"):
        _load(tmp_path, data)


@pytest.mark.parametrize("bad_slot", [35, -1])
def test_out_of_range_slot_index_rejected(bad_slot: int, tmp_path: Path) -> None:
    """槽位越界（35 槽整机布局之外）拒绝。"""
    data = _base_config_dict()
    data["layout"]["joint_slots"]["left_elbow_joint"] = bad_slot
    with pytest.raises(ValueError, match="槽位"):
        _load(tmp_path, data)


def test_missing_seventh_joint_rejected(tmp_path: Path) -> None:
    """joint_slots 漏第七个关节（仅 6 项）拒绝。"""
    data = _base_config_dict()
    slots = data["layout"]["joint_slots"]
    slots.pop("left_wrist_yaw_joint")
    assert len(slots) == 6
    with pytest.raises(ValueError, match="7"):
        _load(tmp_path, data)


def test_slot_index_must_be_integer(tmp_path: Path) -> None:
    """槽位必须是整数（字符串/浮点拒绝）。"""
    data = _base_config_dict()
    data["layout"]["joint_slots"]["left_elbow_joint"] = "18"
    with pytest.raises(ValueError, match="整数"):
        _load(tmp_path, data)


# ---------------------------------------------------------------------------
# 校准负例：长度 / sign 取值 / 键集一致 / NaN / Inf
# ---------------------------------------------------------------------------


def _with_calibration(
    base: dict[str, Any], signs: dict[str, int], offsets: dict[str, float]
) -> dict[str, Any]:
    """在基础配置上替换校准节（测试专用值，非实机标定）。"""
    data = _base_config_dict()
    data["calibration"] = {
        "verified": base.get("verified", True),
        "joint_signs": signs,
        "joint_offsets_rad": offsets,
    }
    return data


def test_wrong_length_signs_rejected(tmp_path: Path) -> None:
    """joint_signs 只给 3 项（错误长度）拒绝。"""
    signs = dict(list({n: 1 for n in _LEFT_JOINT_NAMES}.items())[:3])
    data = _with_calibration({}, signs, {n: 0.0 for n in signs})
    with pytest.raises(ValueError, match="7"):
        _load(tmp_path, data)


def test_wrong_length_offsets_rejected(tmp_path: Path) -> None:
    """joint_offsets_rad 只给 5 项拒绝。"""
    offsets = dict(list({n: 0.0 for n in _LEFT_JOINT_NAMES}.items())[:5])
    data = _with_calibration({}, {n: 1 for n in offsets}, offsets)
    with pytest.raises(ValueError, match="7"):
        _load(tmp_path, data)


@pytest.mark.parametrize("bad_sign", [2, 0, -2])
def test_sign_not_pm1_rejected(bad_sign: int, tmp_path: Path) -> None:
    """sign 只能取 ±1。"""
    signs = {n: 1 for n in _LEFT_JOINT_NAMES}
    signs["left_elbow_joint"] = bad_sign
    data = _with_calibration({}, signs, {n: 0.0 for n in _LEFT_JOINT_NAMES})
    with pytest.raises(ValueError, match="sign"):
        _load(tmp_path, data)


def test_calibration_key_set_mismatch_rejected(tmp_path: Path) -> None:
    """校准键集合与 joint_slots 不一致（含拼写错误键）拒绝。"""
    signs = {n: 1 for n in _LEFT_JOINT_NAMES}
    signs.pop("left_elbow_joint")
    signs["left_elbow_typo_joint"] = 1  # 数量仍 7，但键不一致
    data = _with_calibration({}, signs, {n: 0.0 for n in signs})
    with pytest.raises(ValueError, match="joint_slots"):
        _load(tmp_path, data)


def test_calibration_partial_pair_rejected(tmp_path: Path) -> None:
    """signs 与 offsets 必须同时提供或同时为 null。"""
    data = _base_config_dict()
    data["calibration"]["joint_signs"] = {n: 1 for n in _LEFT_JOINT_NAMES}
    data["calibration"]["joint_offsets_rad"] = None
    with pytest.raises(ValueError, match="同时"):
        _load(tmp_path, data)


def test_nan_inf_rejected(tmp_path: Path) -> None:
    """offset 含 NaN / 周期含 Inf / 下限含 NaN 均拒绝。"""
    nan_offsets = {n: 0.0 for n in _LEFT_JOINT_NAMES}
    nan_offsets["left_elbow_joint"] = float("nan")
    data_nan = _with_calibration({}, {n: 1 for n in _LEFT_JOINT_NAMES}, nan_offsets)
    with pytest.raises(ValueError, match="NaN"):
        _load(tmp_path, data_nan)

    data_inf = _base_config_dict()
    data_inf["timing"]["control_period_s"] = float("inf")
    with pytest.raises(ValueError, match="有限"):
        _load(tmp_path, data_inf)

    bad_lower = {n: -1.0 for n in _LEFT_JOINT_NAMES}
    bad_lower["left_elbow_joint"] = float("nan")
    data_lower = _base_config_dict()
    data_lower["limits"].update(
        {
            "q_lower_rad": bad_lower,
            "q_upper_rad": {n: 1.0 for n in _LEFT_JOINT_NAMES},
        }
    )
    with pytest.raises(ValueError, match="NaN"):
        _load(tmp_path, data_lower)


# ---------------------------------------------------------------------------
# 限值负例
# ---------------------------------------------------------------------------


def test_wrong_length_limits_rejected(tmp_path: Path) -> None:
    """限值映射只给 6 项拒绝。"""
    upper = {n: 1.0 for n in _LEFT_JOINT_NAMES}
    upper.pop("left_wrist_yaw_joint")
    data = _base_config_dict()
    data["limits"].update(
        {
            "q_lower_rad": {n: -1.0 for n in _LEFT_JOINT_NAMES},
            "q_upper_rad": upper,
        }
    )
    with pytest.raises(ValueError, match="7"):
        _load(tmp_path, data)


def test_q_lower_above_upper_rejected(tmp_path: Path) -> None:
    """同名关节 q_lower > q_upper 拒绝。"""
    lower = {n: -1.0 for n in _LEFT_JOINT_NAMES}
    lower["left_elbow_joint"] = 2.0
    data = _base_config_dict()
    data["limits"].update(
        {
            "q_lower_rad": lower,
            "q_upper_rad": {n: 1.0 for n in _LEFT_JOINT_NAMES},
        }
    )
    with pytest.raises(ValueError, match="q_lower"):
        _load(tmp_path, data)


@pytest.mark.parametrize("field", ["dq_limit_rad_s", "tau_limit_nm", "kp_nm_per_rad", "kd_nms_per_rad"])
def test_negative_physical_limit_rejected(field: str, tmp_path: Path) -> None:
    """物理限值（速度/力矩/增益）不得为负。"""
    values = {n: 1.0 for n in _LEFT_JOINT_NAMES}
    values["left_elbow_joint"] = -5.0
    data = _base_config_dict()
    data["limits"][field] = values
    with pytest.raises(ValueError, match="非负"):
        _load(tmp_path, data)


def test_verified_flag_must_be_bool(tmp_path: Path) -> None:
    """verified 标志必须是布尔（整数 1 拒绝）。"""
    data = _base_config_dict()
    data["calibration"]["verified"] = 1
    with pytest.raises(ValueError, match="verified"):
        _load(tmp_path, data)


# ---------------------------------------------------------------------------
# timing 负例
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key,value",
    [
        ("control_period_s", 0.0),
        ("control_period_s", -0.002),
        ("state_timeout_s", 0.0),
        ("diagnostic_duration_s", -1.0),
    ],
)
def test_nonpositive_timing_rejected(key: str, value: float, tmp_path: Path) -> None:
    """控制周期与诊断时长只接受正有限值（0/负数拒绝）。"""
    data = _base_config_dict()
    data["timing"][key] = value
    with pytest.raises(ValueError, match="正"):
        _load(tmp_path, data)


# ---------------------------------------------------------------------------
# validate：profile 一致性与 purpose 梯度
# ---------------------------------------------------------------------------


def test_side_profile_conflict_rejected(tmp_path: Path) -> None:
    """side=left 配置配 right profile 拒绝。"""
    config = _load(tmp_path, _base_config_dict())
    right_profile = load_robot_profile(_RIGHT_PROFILE_PATH)
    with pytest.raises(ValueError, match="side"):
        validate_r1d_hardware_config(config, right_profile, purpose="offline")


def test_robot_id_mismatch_rejected(tmp_path: Path) -> None:
    """配置 robot_id 与 profile 不一致拒绝。"""
    data = _base_config_dict()
    data["robot_id"] = "some_other_robot"
    config = _load(tmp_path, data)
    with pytest.raises(ValueError, match="robot_id"):
        validate_r1d_hardware_config(config, _left_profile(), purpose="offline")


def test_non_seven_dof_profile_rejected(tmp_path: Path) -> None:
    """6 关节 profile 与 R1D 硬件配置不兼容拒绝。"""
    config = _load(tmp_path, _base_config_dict())
    fake_profile = RobotProfile(
        schema_version=1,
        robot_id="unitree_r1d",
        model_path=Path("fake_model.xml"),
        active_joint_names=("j1", "j2", "j3", "j4", "j5", "j6"),
        actuator_names=("u1", "u2", "u3", "u4", "u5", "u6"),
        held_joint_names=(),
        tcp_site_name="tcp",
    )
    with pytest.raises(ValueError, match="7"):
        validate_r1d_hardware_config(config, fake_profile, purpose="offline")


def test_joint_names_mismatch_with_profile_rejected(tmp_path: Path) -> None:
    """joint_slots 与 profile 关节名集合不一致在 validate 拒绝。"""
    data = _base_config_dict()
    data["layout"]["joint_slots"].pop("left_elbow_joint")
    data["layout"]["joint_slots"]["left_elbow_typo_joint"] = 18
    config = _load(tmp_path, data)
    with pytest.raises(ValueError, match="joint_slots"):
        validate_r1d_hardware_config(config, _left_profile(), purpose="offline")


def test_invalid_purpose_rejected(tmp_path: Path) -> None:
    """未知 purpose 拒绝。"""
    config = _load(tmp_path, _base_config_dict())
    with pytest.raises(ValueError, match="purpose"):
        validate_r1d_hardware_config(config, _left_profile(), purpose="full_send")  # type: ignore[arg-type]


def test_read_only_requires_communication_and_timeout_not_layout(
    tmp_path: Path,
) -> None:
    """read_only 门槛：有效通信 + 状态超时；**布局未核验不拒绝**（§3.1 勘误）。

    I-07（评审 2026-11-17）：state_timeout_s 为 read_only 必需项——
    connect() 在状态缓存禁用（无超时）时拒绝连接，预检即显式报错，
    避免按规程准备的只读配置到 connect 才隐式失败。
    §3.1（评审 2026-11-18）：布局核验强制档位改为 motion——只读采集
    是首轮逐关节核对的数据来源，read_only 强制 verified 会让"首轮原始
    采集"无可执行路径（逼操作员虚标）；motion 仍强制（见下方断言）。
    """
    profile = _left_profile()
    # 模板：网卡 null 且布局未核验 → 拒绝（通信参数）
    with pytest.raises(ValueError, match="network_interface"):
        validate_r1d_hardware_config(
            _load(tmp_path, _base_config_dict()), profile, purpose="read_only"
        )
    # 只补网卡：布局仍未核验但 read_only 不再拒绝——下一门槛是状态超时
    data = _base_config_dict()
    data["communication"]["network_interface"] = "eth-test"  # 测试专用，非实机参数
    with pytest.raises(ValueError, match="state_timeout_s"):
        validate_r1d_hardware_config(_load(tmp_path, data), profile, purpose="read_only")
    # 补状态超时：布局仍 false → read_only 通过（首轮原始采集路径可执行）
    data["timing"]["state_timeout_s"] = 0.05  # 测试专用，非实机参数
    validate_r1d_hardware_config(_load(tmp_path, data), profile, purpose="read_only")
    # 状态超时非正有限 → 拒绝（I-07 数值复查）
    data["timing"]["state_timeout_s"] = 0.0
    with pytest.raises(ValueError, match="state_timeout_s"):
        validate_r1d_hardware_config(_load(tmp_path, data), profile, purpose="read_only")


def test_motion_requires_verified_layout(tmp_path: Path) -> None:
    """motion 门槛：布局未核验（layout_verified=false）拒绝接管/运动。"""
    data = _motion_ready_dict()
    data["layout"]["layout_verified"] = False
    with pytest.raises(ValueError, match="layout_verified"):
        validate_r1d_hardware_config(
            _load(tmp_path, data), _left_profile(), purpose="motion"
        )


def test_motion_rejects_provisional_or_null(tmp_path: Path) -> None:
    """motion 门槛：null 校准拒绝；示例值未核验（provisional）拒绝；限值 null 拒绝。"""
    profile = _left_profile()
    # 1) 模板（网卡 null/校准 null）→ 拒绝：模板/仿真数值不能启用运动
    with pytest.raises(ValueError, match="purpose=motion"):
        validate_r1d_hardware_config(
            _load(tmp_path, _base_config_dict()), profile, purpose="motion"
        )
    # 2) provisional：填了示例数值但 calibration.verified=false → 拒绝
    data = _motion_ready_dict()
    data["calibration"]["verified"] = False
    with pytest.raises(ValueError, match="verified"):
        validate_r1d_hardware_config(_load(tmp_path, data), profile, purpose="motion")
    # 3) 校准齐备但限值节回退为模板（null/未核验）→ 拒绝
    data2 = _motion_ready_dict()
    data2["limits"] = _base_config_dict()["limits"]
    with pytest.raises(ValueError, match="limits"):
        validate_r1d_hardware_config(_load(tmp_path, data2), profile, purpose="motion")


def test_motion_requires_timing(tmp_path: Path) -> None:
    """motion 门槛：周期/超时/诊断时长缺一拒绝。"""
    profile = _left_profile()
    for missing in ("control_period_s", "state_timeout_s", "diagnostic_duration_s"):
        data = _motion_ready_dict()
        data["timing"][missing] = None
        with pytest.raises(ValueError, match=missing):
            validate_r1d_hardware_config(_load(tmp_path, data), profile, purpose="motion")


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("control_period_s", 0.0),
        ("control_period_s", -0.002),
        ("control_period_s", float("nan")),
        ("state_timeout_s", float("inf")),
    ],
)
def test_motion_revalidates_positive_timing_bypassing_load(
    field: str, bad_value: float, tmp_path: Path
) -> None:
    """绕过 load 直接构造的 config（dataclasses.replace）在 validate 层复查正有限（M2 双层校验）。"""
    config = _load(tmp_path, _motion_ready_dict())
    bad_timing = dataclasses.replace(config.timing, **{field: bad_value})
    bad_config = dataclasses.replace(config, timing=bad_timing)
    with pytest.raises(ValueError, match="正有限"):
        validate_r1d_hardware_config(bad_config, _left_profile(), purpose="motion")


def test_motion_passes_with_fully_verified_config(tmp_path: Path) -> None:
    """全部硬件门槛满足时 motion 校验通过（测试专用有限参数，非实机参数）。"""
    config = _load(tmp_path, _motion_ready_dict())
    validate_r1d_hardware_config(config, _left_profile(), purpose="motion")
