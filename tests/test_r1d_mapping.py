"""R1-A7-D 帧数据类型与七关节映射测试（阶段 4 Task 4.2）。

所有 sign/offset/哨兵值均为**测试专用有限参数，非实机参数**：
仅用于验证帧校验与映射换算逻辑，不代表任何实物标定。
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from src.real.r1d_config import R1DHardwareConfig, load_r1d_hardware_config
from src.real.r1d_mapping import R1DJointMapping
from src.real.r1d_types import (
    R1D_COMMAND_SLOT_COUNT,
    R1D_STATE_SLOT_COUNT,
    R1DCommandFrame,
    R1DStateFrame,
    R1DStatus,
    R1DTransport,
)
from src.robot.robot_profile import RobotProfile, load_robot_profile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEFT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"
_RIGHT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"

_LEFT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
_RIGHT_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# 测试专用校准（非实机标定）：sign 与 offset 逐关节交错，避免全 1/全 0 掩盖换算错误
_TEST_SIGNS = (1, -1, 1, -1, 1, -1, 1)
_TEST_OFFSETS = (0.0, 0.05, -0.1, 0.2, -0.3, 0.4, 0.0)


# ---------------------------------------------------------------------------
# 辅助构造（测试专用，非实机参数）
# ---------------------------------------------------------------------------


def _make_state(**overrides: Any) -> R1DStateFrame:
    """构造整机状态帧（35 槽，测试专用值）。"""
    values: dict[str, Any] = {
        "q": np.arange(35, dtype=np.float64),
        "dq": np.zeros(35, dtype=np.float64),
        "sequence": None,
        "source_time": None,
        "received_at": 100.0,
        "mode_machine": 7,
        "session_id": 1,
    }
    values.update(overrides)
    return R1DStateFrame(**values)


def _make_command(**overrides: Any) -> R1DCommandFrame:
    """构造整机命令帧（35 槽，测试专用值）。"""
    values: dict[str, Any] = {
        "q": np.zeros(35, dtype=np.float64),
        "dq": np.zeros(35, dtype=np.float64),
        "kp": np.zeros(35, dtype=np.float64),
        "kd": np.zeros(35, dtype=np.float64),
        "tau": np.zeros(35, dtype=np.float64),
        "motor_mode": np.zeros(35, dtype=np.float64),
        "mode_machine": 7,
        "mode_pr": 0,
    }
    values.update(overrides)
    return R1DCommandFrame(**values)


def _hardware_config_dict(
    joint_names: tuple[str, ...], base_slot: int
) -> dict[str, Any]:
    """返回映射可用的硬件配置字典（测试专用有限参数，非实机参数）。"""
    slots = {name: base_slot + i for i, name in enumerate(joint_names)}
    signs = {name: s for name, s in zip(joint_names, _TEST_SIGNS)}
    offsets = {name: o for name, o in zip(joint_names, _TEST_OFFSETS)}
    side = "left" if joint_names[0].startswith("left_") else "right"
    return {
        "schema_version": 1,
        "robot_id": "unitree_r1d",
        "side": side,
        "communication": {
            "dds_domain": 0,
            "network_interface": "eth-test",  # 测试专用，非实机参数
            "lowcmd_topic": "rt/lowcmd",
            "lowstate_topic": "rt/lowstate",
        },
        "layout": {"layout_verified": True, "joint_slots": slots},
        "calibration": {"verified": True, "joint_signs": signs, "joint_offsets_rad": offsets},
        "limits": {
            "verified": True,
            "q_lower_rad": {n: -1.5 for n in joint_names},  # 测试值，非实机限位
            "q_upper_rad": {n: 1.5 for n in joint_names},
            "dq_limit_rad_s": {n: 2.0 for n in joint_names},
            "tau_limit_nm": {n: 30.0 for n in joint_names},
            "kp_nm_per_rad": {n: 100.0 for n in joint_names},
            "kd_nms_per_rad": {n: 2.0 for n in joint_names},
        },
        "timing": {
            "control_period_s": 0.002,  # 数值同官方示例，仅测试用
            "state_timeout_s": 0.05,
            "diagnostic_duration_s": 5.0,
        },
    }


def _load_config(tmp_path: Path, data: dict[str, Any]) -> R1DHardwareConfig:
    """写 YAML 并加载。"""
    path = tmp_path / "hw.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return load_r1d_hardware_config(path)


@pytest.fixture
def left_mapping(tmp_path: Path) -> R1DJointMapping:
    """左臂映射（槽位 15-21，测试校准）。"""
    config = _load_config(tmp_path, _hardware_config_dict(_LEFT_JOINT_NAMES, 15))
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    return R1DJointMapping.from_profile(profile, config)


@pytest.fixture
def right_mapping(tmp_path: Path) -> R1DJointMapping:
    """右臂映射（槽位 22-28，测试校准）。"""
    config = _load_config(tmp_path, _hardware_config_dict(_RIGHT_JOINT_NAMES, 22))
    profile = load_robot_profile(_RIGHT_PROFILE_PATH)
    return R1DJointMapping.from_profile(profile, config)


# ---------------------------------------------------------------------------
# 模块纯净性
# ---------------------------------------------------------------------------


def test_module_import_is_pure() -> None:
    """import 类型/映射模块不得加载 SDK/MuJoCo、不得启动线程（子进程验证）。

    I-09 修复（评审 2026-11-17）：原进程内断言命中 sys.modules 缓存，
    不执行模块体，属空操作；参照 test_r1d_transport.py 先例改子进程。
    """
    code = (
        "import sys\n"
        "import threading\n"
        "import src.real.r1d_mapping\n"
        "import src.real.r1d_types\n"
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
# R1DStateFrame 校验与不可变性
# ---------------------------------------------------------------------------


def test_state_frame_rejects_wrong_length() -> None:
    """q/dq 长度不是 35 拒绝。"""
    with pytest.raises(ValueError, match="35"):
        _make_state(q=np.zeros(34))
    with pytest.raises(ValueError, match="35"):
        _make_state(dq=np.zeros(36))


def test_state_frame_rejects_nan_inf() -> None:
    """q/dq 含 NaN/Inf 拒绝。"""
    q = np.zeros(35)
    q[15] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _make_state(q=q)
    dq = np.zeros(35)
    dq[22] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        _make_state(dq=dq)


def test_state_frame_copies_input_arrays() -> None:
    """边界复制：构造后修改原输入数组不污染快照。"""
    q = np.zeros(35)
    dq = np.zeros(35)
    frame = _make_state(q=q, dq=dq)
    q[3] = 42.0
    dq[3] = 7.0
    assert frame.q[3] == 0.0
    assert frame.dq[3] == 0.0


def test_state_frame_arrays_are_readonly() -> None:
    """存储数组只读：调用方无法修改快照内容。"""
    frame = _make_state()
    with pytest.raises(ValueError):
        frame.q[0] = 99.0
    with pytest.raises(ValueError):
        frame.dq[0] = 1.0


def test_state_frame_is_frozen() -> None:
    """frozen dataclass：字段不可重绑定。"""
    frame = _make_state()
    with pytest.raises(dataclasses.FrozenInstanceError):
        frame.session_id = 2  # type: ignore[misc]


def test_state_frame_mode_machine_uint8_range() -> None:
    """mode_machine 为 uint8 语义：256/-1 拒绝。"""
    with pytest.raises(ValueError, match="uint8"):
        _make_state(mode_machine=256)
    with pytest.raises(ValueError, match="uint8"):
        _make_state(mode_machine=-1)


def test_state_frame_accepts_none_optional_fields() -> None:
    """sequence/source_time 无支持时用 None（不编造）；received_at 必填有限。"""
    frame = _make_state(sequence=None, source_time=None)
    assert frame.sequence is None
    assert frame.source_time is None
    with pytest.raises(ValueError):
        _make_state(received_at=float("inf"))
    # 有支持时可填整数/浮点
    frame2 = _make_state(sequence=12345, source_time=1.5)
    assert frame2.sequence == 12345
    assert frame2.source_time == 1.5


# ---------------------------------------------------------------------------
# R1DCommandFrame 校验与不可变性
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["q", "dq", "kp", "kd", "tau", "motor_mode"])
def test_command_frame_rejects_wrong_length(field: str) -> None:
    """全部命令数组长度不是 35 拒绝。"""
    with pytest.raises(ValueError, match="35"):
        _make_command(**{field: np.zeros(20)})


def test_command_frame_rejects_nan_inf() -> None:
    """命令数组含 NaN 拒绝。"""
    tau = np.zeros(35)
    tau[16] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        _make_command(tau=tau)


def test_command_frame_copies_and_is_readonly() -> None:
    """边界复制 + 存储只读。"""
    q = np.zeros(35)
    command = _make_command(q=q)
    q[15] = 5.0
    assert command.q[15] == 0.0
    with pytest.raises(ValueError):
        command.q[15] = 1.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        command.mode_pr = 100  # type: ignore[misc]


def test_command_frame_mode_fields_uint8_range() -> None:
    """mode_machine/mode_pr 均为 uint8 语义：越界拒绝。"""
    with pytest.raises(ValueError, match="uint8"):
        _make_command(mode_pr=256)
    with pytest.raises(ValueError, match="uint8"):
        _make_command(mode_machine=-1)


@pytest.mark.parametrize(
    "bad_value",
    [0.5, 256.0, -1.0],
)
def test_command_frame_motor_mode_integer_uint8_only(bad_value: float) -> None:
    """motor_mode 为协议 uint8 语义：非整数与越界值拒绝（M1）。"""
    motor_mode = np.zeros(35)
    motor_mode[16] = bad_value
    with pytest.raises(ValueError, match="uint8"):
        _make_command(motor_mode=motor_mode)


def test_command_frame_motor_mode_accepts_protocol_values() -> None:
    """motor_mode 接受协议整数值 0/1（Enable/Disable，语义解释归 4.3）。"""
    motor_mode = np.zeros(35)
    motor_mode[15:22] = 1.0
    command = _make_command(motor_mode=motor_mode)
    assert bool(np.all(command.motor_mode[15:22] == 1.0))
    assert command.motor_mode[0] == 0.0


def test_command_frame_mode_pr_passthrough() -> None:
    """mode_pr 只透传不解释：0/100 等协议值原样保存。"""
    assert _make_command(mode_pr=0).mode_pr == 0
    assert _make_command(mode_pr=100).mode_pr == 100


# ---------------------------------------------------------------------------
# R1DStatus 校验
# ---------------------------------------------------------------------------


def test_status_valid_construction() -> None:
    """合法状态快照可构造。"""
    status = R1DStatus(
        phase="READ_ONLY",
        session_id=3,
        fault_reason=None,
        stop_confirmed=False,
        last_publish_at=None,
    )
    assert status.phase == "READ_ONLY"
    assert status.fault_reason is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"phase": "RUNNING"},  # 非法 phase
        {"fault_reason": ""},  # 空故障原因
        {"stop_confirmed": 1},  # 非布尔
        {"last_publish_at": float("nan")},  # 非有限时间
    ],
)
def test_status_rejects_invalid_fields(overrides: dict[str, Any]) -> None:
    """非法 phase/空 fault_reason/非布尔 stop_confirmed/非有限时间拒绝。"""
    values: dict[str, Any] = {
        "phase": "ACTIVE",
        "session_id": 1,
        "fault_reason": None,
        "stop_confirmed": False,
        "last_publish_at": None,
    }
    values.update(overrides)
    with pytest.raises(ValueError):
        R1DStatus(**values)


# ---------------------------------------------------------------------------
# R1DTransport Protocol
# ---------------------------------------------------------------------------


class _FakeTransport:
    """测试用最小传输实现（非实机）。"""

    def open(self, on_state: Callable[[R1DStateFrame], None]) -> None:
        """只订阅回调。"""

    def acquire_control(self) -> None:
        """获取控制权。"""

    def publish(self, command: R1DCommandFrame) -> None:
        """发布命令。"""

    def release_control(self) -> None:
        """释放控制权。"""

    def close(self) -> None:
        """关闭（幂等）。"""


class _IncompleteTransport:
    """缺少方法的实现（负例）。"""

    def open(self, on_state: Callable[[R1DStateFrame], None]) -> None:
        """只订阅回调。"""


def test_transport_protocol_runtime_checkable() -> None:
    """R1DTransport 为 runtime_checkable Protocol：完整实现可识别。"""
    assert isinstance(_FakeTransport(), R1DTransport)
    assert not isinstance(_IncompleteTransport(), R1DTransport)
    # 槽位常量一致性：状态与命令同用整机 35 槽布局
    assert R1D_STATE_SLOT_COUNT == 35
    assert R1D_COMMAND_SLOT_COUNT == 35


# ---------------------------------------------------------------------------
# 映射：按名称（非前缀/顺序猜测）
# ---------------------------------------------------------------------------


# 独立手算期望表（哨兵 q_sdk[slot] = slot*0.5 + 2.0；q_model = sign*(q_sdk - offset)，
# sign=(+1,-1,+1,-1,+1,-1,+1)、offset=(0,0.05,-0.1,0.2,-0.3,0.4,0) 逐关节手算，
# 不经被测代码生成）
_LEFT_EXPECTED = (9.5, -9.95, 10.6, -10.8, 11.8, -11.6, 12.5)
_RIGHT_EXPECTED = (13.0, -13.45, 14.1, -14.3, 15.3, -15.1, 16.0)


@pytest.mark.parametrize(
    "mapping_fixture,expected",
    [("left_mapping", _LEFT_EXPECTED), ("right_mapping", _RIGHT_EXPECTED)],
)
def test_mapping_uses_names_not_prefix(
    mapping_fixture: str, expected: tuple[float, ...], request: pytest.FixtureRequest
) -> None:
    """全机 35 槽唯一哨兵值 + 独立期望表验证两侧七项（非实机参数）。"""
    mapping: R1DJointMapping = request.getfixturevalue(mapping_fixture)
    # 每个槽位填唯一哨兵：slot -> slot*0.5 + 2.0
    q_sdk = np.array([slot * 0.5 + 2.0 for slot in range(35)])
    frame = _make_state(q=q_sdk)
    got = mapping.extract(frame)
    assert got.shape == (7,)
    assert_allclose(got, np.array(expected), atol=1e-12)


def test_mapping_preserves_profile_order(left_mapping: R1DJointMapping) -> None:
    """映射顺序与 profile.active_joint_names 一致（模型序）。"""
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    assert left_mapping.joint_names == profile.active_joint_names
    assert left_mapping.sdk_indices == (15, 16, 17, 18, 19, 20, 21)


def test_mapping_ignores_other_arm_slots(left_mapping: R1DJointMapping) -> None:
    """左臂映射不受右臂槽位（22-28）数值影响。"""
    q_sdk = np.array([slot * 0.5 + 2.0 for slot in range(35)])
    frame_a = _make_state(q=q_sdk.copy())
    q_sdk[22:29] = -777.0  # 篡改右臂槽位
    frame_b = _make_state(q=q_sdk)
    assert_allclose(left_mapping.extract(frame_a), left_mapping.extract(frame_b), atol=1e-12)


def test_mapping_addresses_slots_by_name_not_order(tmp_path: Path) -> None:
    """B-M3（评审 2026-11-17）：乱序槽位分配下按名寻址仍正确。

    joint_slots 把 7 个关节映射到乱序槽位（非连续、非升序、避开对侧
    臂 22-28），映射须按名字取各关节自身槽位——顺序切片实现（如
    q[15:22]）在此配置下会取错槽。既有序列测试均为升序连续分配，
    无法判别按名寻址。
    """
    shuffled = {
        "left_shoulder_pitch_joint": 18,
        "left_shoulder_roll_joint": 15,
        "left_shoulder_yaw_joint": 21,
        "left_elbow_joint": 16,
        "left_wrist_roll_joint": 20,
        "left_wrist_pitch_joint": 17,
        "left_wrist_yaw_joint": 19,
    }
    data = _hardware_config_dict(_LEFT_JOINT_NAMES, 15)
    data["layout"]["joint_slots"] = shuffled
    config = _load_config(tmp_path, data)
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    mapping = R1DJointMapping.from_profile(profile, config)
    # 模型序（profile.active_joint_names 顺序）对应的 SDK 槽位
    assert mapping.sdk_indices == (18, 15, 21, 16, 20, 17, 19)
    # 每槽唯一哨兵 q_sdk[slot] = 0.5*slot + 2.0；q_model = sign*(q_sdk - offset)
    # sign=(+1,-1,+1,-1,+1,-1,+1)、offset=(0,0.05,-0.1,0.2,-0.3,0.4,0)
    # 期望逐关节独立手算（乱序槽位取值）：
    #   pitch(18): 11.0-0=11.0；roll(15): -(9.5-0.05)=-9.45；
    #   yaw(21): 12.5+0.1=12.6；elbow(16): -(10.0-0.2)=-9.8；
    #   wrist_roll(20): 12.0+0.3=12.3；wrist_pitch(17): -(10.5-0.4)=-10.1；
    #   wrist_yaw(19): 11.5-0=11.5
    q_sdk = np.array([slot * 0.5 + 2.0 for slot in range(35)])
    frame = _make_state(q=q_sdk)
    expected = (11.0, -9.45, 12.6, -9.8, 12.3, -10.1, 11.5)
    assert_allclose(mapping.extract(frame), np.array(expected), atol=1e-12)
    # 逆方向：to_sdk_positions 输出模型序 7 项（q_sdk = sign*q + offset 手算）
    q_model = np.array(expected)
    expected_sdk = np.array(
        [11.0, -(q_model[1]) + 0.05, 12.6 - 0.1, -(q_model[3]) + 0.2,
         12.3 - 0.3, -(q_model[5]) + 0.4, 11.5]
    )
    assert_allclose(mapping.to_sdk_positions(q_model), expected_sdk, atol=1e-12)


# ---------------------------------------------------------------------------
# 映射：sign/offset 往返
# ---------------------------------------------------------------------------


def test_sign_offset_round_trip(left_mapping: R1DJointMapping) -> None:
    """非零偏置与正负方向往返 atol=1e-12；速度只变符号不加偏置。"""
    q_model = np.array([0.10, -0.20, 0.30, -0.40, 0.50, -0.60, 0.70])
    q_sdk7 = left_mapping.to_sdk_positions(q_model)
    # 独立手算：q_sdk = sign*q_model + offset（左臂槽 15-21 顺序）
    expected_sdk = np.array([0.10, 0.25, 0.20, 0.60, 0.20, 1.00, 0.70])
    assert_allclose(q_sdk7, expected_sdk, atol=1e-12)

    # 放回整机帧（非目标槽位填区分哨兵），extract 应回到 q_model
    q_full = np.full(35, 7.77)  # 测试专用哨兵，非实机参数
    q_full[15:22] = q_sdk7
    dq_full = np.array([slot * 0.11 + 0.02 for slot in range(35)])  # 每槽唯一
    frame = _make_state(q=q_full, dq=dq_full)
    assert_allclose(left_mapping.extract(frame), q_model, atol=1e-12)

    # 速度只变符号不加偏置：dq_model = sign*dq_sdk（手算 slots 15-21）
    expected_dq = np.array([1.67, -1.78, 1.89, -2.00, 2.11, -2.22, 2.33])
    assert_allclose(left_mapping.extract_velocities(frame), expected_dq, atol=1e-12)

    # 逆往返：模型序 → SDK → 模型序 应逐位还原
    assert_allclose(
        left_mapping.to_sdk_positions(left_mapping.extract(frame)), q_sdk7, atol=1e-12
    )


def test_to_sdk_positions_validates_input(left_mapping: R1DJointMapping) -> None:
    """to_sdk_positions 输入长度错误/NaN 拒绝。"""
    with pytest.raises(ValueError, match="7"):
        left_mapping.to_sdk_positions(np.zeros(6))
    bad = np.zeros(7)
    bad[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        left_mapping.to_sdk_positions(bad)


def test_extract_returns_independent_copy(left_mapping: R1DJointMapping) -> None:
    """extract 返回新数组：修改返回值不影响后续读取。"""
    q_sdk = np.array([slot * 0.5 + 2.0 for slot in range(35)])
    frame = _make_state(q=q_sdk)
    first = left_mapping.extract(frame)
    first[0] = 123.0
    second = left_mapping.extract(frame)
    assert second[0] != 123.0


# ---------------------------------------------------------------------------
# from_profile 负例
# ---------------------------------------------------------------------------


def test_from_profile_requires_calibration(tmp_path: Path) -> None:
    """校准为 null（未知校准）时拒绝构造映射——未知校准不能启用运动。"""
    data = _hardware_config_dict(_LEFT_JOINT_NAMES, 15)
    data["calibration"] = {"verified": False, "joint_signs": None, "joint_offsets_rad": None}
    config = _load_config(tmp_path, data)
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    with pytest.raises(ValueError, match="校准"):
        R1DJointMapping.from_profile(profile, config)


def test_from_profile_rejects_unknown_joint_names(tmp_path: Path) -> None:
    """joint_slots 含 profile 未知关节名/缺关节名拒绝。

    三处按名表同步替换同一 typo 键（load 层内部一致性通过），
    专测 from_profile 对 profile 关节名的精确匹配。
    """
    data = _hardware_config_dict(_LEFT_JOINT_NAMES, 15)
    replacements = (
        (data["layout"]["joint_slots"], 18),
        (data["calibration"]["joint_signs"], 1),
        (data["calibration"]["joint_offsets_rad"], 0.0),
        (data["limits"]["q_lower_rad"], -1.5),
        (data["limits"]["q_upper_rad"], 1.5),
        (data["limits"]["dq_limit_rad_s"], 2.0),
        (data["limits"]["tau_limit_nm"], 30.0),
        (data["limits"]["kp_nm_per_rad"], 100.0),
        (data["limits"]["kd_nms_per_rad"], 2.0),
    )
    for table, typo_value in replacements:
        table.pop("left_elbow_joint")
        table["left_elbow_typo_joint"] = typo_value
    config = _load_config(tmp_path, data)
    profile = load_robot_profile(_LEFT_PROFILE_PATH)
    with pytest.raises(ValueError, match="joint_slots"):
        R1DJointMapping.from_profile(profile, config)


def test_from_profile_rejects_non_seven_profile(tmp_path: Path) -> None:
    """6 关节 profile 拒绝。"""
    config = _load_config(tmp_path, _hardware_config_dict(_LEFT_JOINT_NAMES, 15))
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
        R1DJointMapping.from_profile(fake_profile, config)
