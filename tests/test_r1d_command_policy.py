"""R1-A7-D 全机命令构造与非目标部件策略测试（阶段 4 Task 4.3）。

所有 sign/offset/增益/限值/显式槽位策略/时钟均为**测试专用有限参数，非实机参数**：
仅用于验证全机 35 槽命令构造与非目标槽位策略逻辑，不代表任何实物标定或
已核验保持策略。对侧臂/头保持数值断言与 configs/robots/r1d_protocol_manifest.yaml
whole_body.other_arm / whole_body.head 的冻结示例语义一致（[公开示例] 证据等级）。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pytest
import yaml
from numpy.testing import assert_allclose

from src.real.r1d_command_policy import (
    R1DCommandPolicy,
    R1DCommandPolicyError,
    R1DSlotHold,
)
from src.real.r1d_types import R1DStateFrame
from src.robot.robot_profile import load_robot_profile

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

# —— 测试专用参数（非实机参数）——
_TEST_SIGNS = (1, -1, 1, -1, 1, -1, 1)
_TEST_OFFSETS = (0.0, 0.05, -0.1, 0.2, -0.3, 0.4, 0.0)
_TEST_KP = (60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0)  # N·m/rad
_TEST_KD = (1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0)  # N·m·s/rad
_TEST_DQ_LIMIT = (2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0)  # rad/s
_TEST_Q_BOUND = 1.5  # rad，位置上下限对称
_TEST_TAU_LIMIT = 30.0  # N·m
_PERIOD = 0.002  # s（数值同官方示例，仅测试用）
_TIMEOUT = 0.05  # s（状态新鲜度超时，测试专用）
_NOW = 1000.0  # 注入的 fake monotonic 时钟（测试专用）

# 逐关节单周期步长上限 = dq_limit * period（手算）
_MAX_STEP = tuple(d * _PERIOD for d in _TEST_DQ_LIMIT)  # (0.004, ..., 0.010)

# manifest 示例数值哨兵（whole_body.other_arm / whole_body.head，
# [公开示例]）——R5 修订（评审 I-05）后 manifest 示例不作运行时回退，
# 测试把它们作为**显式策略哨兵**注入（数值断言保留判别力）
_OTHER_ARM_HOLD_KP = (100.0, 100.0, 100.0, 100.0, 50.0, 50.0, 50.0)
_OTHER_ARM_HOLD_KD = 2.0
_HEAD_HOLD_KP = (50.0, 10.0)
_HEAD_HOLD_KD = (2.0, 0.1)

# 哨兵策略槽位（manifest null 组，逐槽唯一值）：
# chassis(0,1) + 未编址(2-11,14) + lift_column(12) + waist(13) + 未编址(31-34)
_EXPLICIT_SLOTS = tuple(range(0, 15)) + (31, 32, 33, 34)
# 左侧策略的对侧臂=右臂 22-28；右侧策略的对侧臂=左臂 15-21；头恒 29/30
_OTHER_ARM_SLOTS_FOR_LEFT = (22, 23, 24, 25, 26, 27, 28)
_OTHER_ARM_SLOTS_FOR_RIGHT = (15, 16, 17, 18, 19, 20, 21)
_HEAD_SLOTS = (29, 30)
# 全部 28 个非目标槽（left 侧）：显式策略现在须全覆盖（R5 修订）
_ALL_NON_TARGET_SLOTS_LEFT = (
    _EXPLICIT_SLOTS + _OTHER_ARM_SLOTS_FOR_LEFT + _HEAD_SLOTS
)
_EXPECTED_MISSING_WHEN_NO_EXPLICIT = (
    "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, "
    "22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34]"
)

# 实测姿态哨兵：q[slot] = 0.1*slot - 2.0（35 槽唯一，测试专用）
# 左臂槽 15-21 实测 SDK = (-0.5,-0.4,-0.3,-0.2,-0.1,0.0,0.1)
# 右臂槽 22-28 实测 SDK = (0.2,0.3,0.4,0.5,0.6,0.7,0.8)
# 锚定首帧的模型序 q_desired 手算：q_model = sign*(q_sdk - offset)
_LEFT_Q_DESIRED_ANCHORED = (-0.5, 0.45, -0.2, 0.4, 0.2, 0.4, 0.1)
_RIGHT_Q_DESIRED_ANCHORED = (0.2, -0.25, 0.5, -0.3, 0.9, -0.3, 0.8)
_LEFT_MEASURED_SDK = (-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1)
_RIGHT_MEASURED_SDK = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


# ---------------------------------------------------------------------------
# 辅助构造（测试专用，非实机参数）
# ---------------------------------------------------------------------------


def _measured_q() -> np.ndarray:
    """返回 35 槽唯一哨兵实测姿态（q[slot] = 0.1*slot - 2.0）。"""
    return np.array([0.1 * slot - 2.0 for slot in range(35)])


def _modified_measured(*pairs: tuple[int, float]) -> np.ndarray:
    """返回替换了若干槽位值的实测姿态副本。"""
    q = _measured_q()
    for slot, value in pairs:
        q[slot] = value
    return q


def _fresh_state(**overrides: Any) -> R1DStateFrame:
    """构造新鲜整机状态帧（received_at == _NOW，测试专用值）。"""
    values: dict[str, Any] = {
        "q": _measured_q(),
        "dq": np.zeros(35, dtype=np.float64),
        "sequence": None,
        "source_time": None,
        "received_at": _NOW,
        "mode_machine": 7,
        "session_id": 1,
    }
    values.update(overrides)
    return R1DStateFrame(**values)


def _explicit_hold(slot: int) -> R1DSlotHold:
    """测试专用显式槽位保持策略（逐槽唯一哨兵，非实机参数）。"""
    return R1DSlotHold(
        mode=1,
        q=0.007 - 0.031 * slot,
        dq=0.0,
        kp=11.0 + slot,
        kd=0.4 + 0.13 * slot,
        tau=0.21 + 0.07 * slot,
    )


def _manifest_example_hold(slot: int) -> R1DSlotHold:
    """对侧臂/头槽的显式哨兵策略（数值=manifest [公开示例]，q=0 保持）。

    R5 修订后这些数值不再是运行时默认——测试显式注入以保留对示例
    语义数值填充的判别力（期望表手算对照同一数值）。
    """
    if slot in _HEAD_SLOTS:
        offset = _HEAD_SLOTS.index(slot)
        kp, kd = _HEAD_HOLD_KP[offset], _HEAD_HOLD_KD[offset]
    elif 15 <= slot <= 21:
        kp, kd = _OTHER_ARM_HOLD_KP[slot - 15], _OTHER_ARM_HOLD_KD
    else:
        kp, kd = _OTHER_ARM_HOLD_KP[slot - 22], _OTHER_ARM_HOLD_KD
    return R1DSlotHold(mode=1, q=0.0, dq=0.0, kp=kp, kd=kd, tau=0.0)


def _full_explicit_policies(side: str = "left") -> dict[int, R1DSlotHold]:
    """覆盖该侧全部 28 个非目标槽位的显式策略（R5 修订：须全覆盖）。

    manifest-null 19 槽用逐槽唯一哨兵；对侧臂 7 槽 + 头 2 槽用
    manifest 示例数值哨兵（显式来源，非运行时回退）。
    """
    other_arm = _OTHER_ARM_SLOTS_FOR_LEFT if side == "left" else _OTHER_ARM_SLOTS_FOR_RIGHT
    policies = {slot: _explicit_hold(slot) for slot in _EXPLICIT_SLOTS}
    for slot in other_arm:
        policies[slot] = _manifest_example_hold(slot)
    for slot in _HEAD_SLOTS:
        policies[slot] = _manifest_example_hold(slot)
    return policies


def _hardware_config_dict(
    joint_names: tuple[str, ...], base_slot: int
) -> dict[str, Any]:
    """返回 motion 级硬件配置字典（测试专用有限参数，非实机参数）。"""
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
        "calibration": {
            "verified": True,
            "joint_signs": signs,
            "joint_offsets_rad": offsets,
        },
        "limits": {
            "verified": True,
            "q_lower_rad": {n: -_TEST_Q_BOUND for n in joint_names},
            "q_upper_rad": {n: _TEST_Q_BOUND for n in joint_names},
            "dq_limit_rad_s": {n: d for n, d in zip(joint_names, _TEST_DQ_LIMIT)},
            "tau_limit_nm": {n: _TEST_TAU_LIMIT for n in joint_names},
            "kp_nm_per_rad": {n: k for n, k in zip(joint_names, _TEST_KP)},
            "kd_nms_per_rad": {n: k for n, k in zip(joint_names, _TEST_KD)},
        },
        "timing": {
            "control_period_s": _PERIOD,
            "state_timeout_s": _TIMEOUT,
            "diagnostic_duration_s": 5.0,
        },
    }


# slot_policies 参数的"默认给全量显式策略"哨兵（与 None=不给显式策略区分）
_UNSET = object()


def _make_policy(
    tmp_path: Path,
    side: str = "left",
    *,
    filename: str | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
    slot_policies: Any = _UNSET,
    expected_mode_machine: int | None = None,
    mode_pr: int | None = None,
) -> R1DCommandPolicy:
    """构造被测策略（clock 注入 fake 单调时钟，测试专用）。"""
    if side == "left":
        names, base, profile_path = _LEFT_JOINT_NAMES, 15, _LEFT_PROFILE_PATH
    else:
        names, base, profile_path = _RIGHT_JOINT_NAMES, 22, _RIGHT_PROFILE_PATH
    data = _hardware_config_dict(names, base)
    if mutate is not None:
        mutate(data)
    path = tmp_path / (filename or f"hw_{side}.yaml")
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    from src.real.r1d_config import load_r1d_hardware_config

    config = load_r1d_hardware_config(path)
    profile = load_robot_profile(profile_path)
    if slot_policies is _UNSET:
        slot_policies = _full_explicit_policies(side)
    return R1DCommandPolicy(
        config,
        profile,
        slot_policies=slot_policies,
        expected_mode_machine=expected_mode_machine,
        mode_pr=mode_pr,
        clock=lambda: _NOW,
    )


def _left_q_desired_with(joint: int, value: float) -> np.ndarray:
    """返回仅替换一个关节值的左臂锚定 q_desired（模型序）。"""
    vals = list(_LEFT_Q_DESIRED_ANCHORED)
    vals[joint] = value
    return np.array(vals)


@pytest.fixture
def left_policy(tmp_path: Path) -> R1DCommandPolicy:
    """左臂命令策略（全量显式策略 + fake 时钟，测试专用）。"""
    return _make_policy(tmp_path, "left")


@pytest.fixture
def right_policy(tmp_path: Path) -> R1DCommandPolicy:
    """右臂命令策略（全量显式策略 + fake 时钟，测试专用）。"""
    return _make_policy(tmp_path, "right")


# ---------------------------------------------------------------------------
# 模块纯净性
# ---------------------------------------------------------------------------


def test_module_import_is_pure() -> None:
    """import 策略模块不得加载 SDK/MuJoCo、不得启动线程。

    断言在 finally 恢复 saved 模块**之前**执行（同 4.2 测试约定）：
    同进程预载过 mujoco 时，恢复后再断言会读到已恢复的真实模块造成假失败。
    """
    banned = ("unitree_sdk2py", "cyclonedds", "mujoco")
    saved = {name: sys.modules.get(name) for name in banned}
    for name in banned:
        sys.modules.pop(name, None)
    before_threads = threading.active_count()
    try:
        import src.real.r1d_command_policy  # noqa: F401

        for name in banned:
            assert name not in sys.modules, f"import 不应加载 {name}"
        assert threading.active_count() == before_threads, "import 不应启动线程"
    finally:
        for name, module in saved.items():
            if module is not None:
                sys.modules[name] = module


# ---------------------------------------------------------------------------
# 全机 35 槽逐槽位冻结策略
# ---------------------------------------------------------------------------


def test_all_non_target_slots_follow_frozen_policy(
    left_policy: R1DCommandPolicy,
) -> None:
    """逐槽位检查 q/dq/kp/kd/tau/mode 与头字段（不只检查目标七项）。

    期望表独立手算：目标 15-21（实测锚定 + 测试配置增益）、对侧臂 22-28 与
    头 29/30（显式策略哨兵=manifest 冻结示例数值 [公开示例]——R5 修订后
    这些槽位同样须显式给出，缺任一即拒绝）、其余 19 槽（逐槽唯一哨兵策略）。
    """
    state = _fresh_state()
    q_before = state.q.copy()
    dq_before = state.dq.copy()

    frame = left_policy.build_active(state, np.array(_LEFT_Q_DESIRED_ANCHORED))

    # 独立期望表（不经被测代码生成）
    exp_q = np.zeros(35)
    exp_dq = np.zeros(35)
    exp_kp = np.zeros(35)
    exp_kd = np.zeros(35)
    exp_tau = np.zeros(35)
    exp_mode = np.zeros(35)
    # 显式策略 19 槽（manifest-null 组，测试哨兵）
    for slot in _EXPLICIT_SLOTS:
        hold = _explicit_hold(slot)
        exp_q[slot] = hold.q
        exp_dq[slot] = hold.dq
        exp_kp[slot] = hold.kp
        exp_kd[slot] = hold.kd
        exp_tau[slot] = hold.tau
        exp_mode[slot] = hold.mode
    # 对侧臂 22-28：manifest whole_body.other_arm.hold_semantics_example [公开示例]
    exp_kp[22:29] = _OTHER_ARM_HOLD_KP
    exp_kd[22:29] = _OTHER_ARM_HOLD_KD
    exp_mode[22:29] = 1.0
    # 头 29(pitch)/30(yaw)：manifest whole_body.head.hold_semantics_example [公开示例]
    exp_kp[29], exp_kp[30] = _HEAD_HOLD_KP
    exp_kd[29], exp_kd[30] = _HEAD_HOLD_KD
    exp_mode[29] = exp_mode[30] = 1.0
    # 目标 15-21：首帧锚定实测 + 测试配置增益，dq/tau 恒 0
    exp_q[15:22] = _LEFT_MEASURED_SDK
    exp_kp[15:22] = _TEST_KP
    exp_kd[15:22] = _TEST_KD
    exp_mode[15:22] = 1.0

    assert_allclose(frame.q, exp_q, atol=1e-12)
    assert_allclose(frame.dq, exp_dq, atol=1e-12)
    assert_allclose(frame.kp, exp_kp, atol=1e-12)
    assert_allclose(frame.kd, exp_kd, atol=1e-12)
    assert_allclose(frame.tau, exp_tau, atol=1e-12)
    assert_allclose(frame.motor_mode, exp_mode, atol=1e-12)

    # 头字段：mode_machine 回填自状态；mode_pr 默认 0（Ruling R6）
    assert frame.mode_machine == state.mode_machine
    assert frame.mode_pr == 0

    # 输入 state 保持原样（逐位相等）
    assert np.array_equal(state.q, q_before)
    assert np.array_equal(state.dq, dq_before)


def test_left_and_right_policies_do_not_cross_overwrite(tmp_path: Path) -> None:
    """左右侧策略各只写自己的目标七槽，对侧臂一律按保持策略，互不覆盖。"""
    left = _make_policy(tmp_path, "left")
    right = _make_policy(tmp_path, "right")

    left_frame = left.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))
    right_frame = right.build_active(
        _fresh_state(), np.array(_RIGHT_Q_DESIRED_ANCHORED)
    )

    # 左策略：目标 15-21 写命令值；22-28 为对侧保持（显式哨兵 q=0 +
    # manifest 示例增益——R5 修订后同样来自显式策略）
    assert_allclose(left_frame.q[15:22], _LEFT_MEASURED_SDK, atol=1e-12)
    assert_allclose(left_frame.kp[15:22], _TEST_KP, atol=1e-12)
    assert_allclose(left_frame.q[22:29], np.zeros(7), atol=1e-12)
    assert_allclose(left_frame.kp[22:29], _OTHER_ARM_HOLD_KP, atol=1e-12)
    # 右策略：目标 22-28 写命令值；15-21 为对侧保持（显式哨兵）
    assert_allclose(right_frame.q[22:29], _RIGHT_MEASURED_SDK, atol=1e-12)
    assert_allclose(right_frame.kp[22:29], _TEST_KP, atol=1e-12)
    assert_allclose(right_frame.q[15:22], np.zeros(7), atol=1e-12)
    assert_allclose(right_frame.kp[15:22], _OTHER_ARM_HOLD_KP, atol=1e-12)
    # 显式哨兵槽两侧一致（同一显式策略来源，不被对侧目标覆盖）
    assert left_frame.q[5] == pytest.approx(0.007 - 0.031 * 5, abs=1e-12)
    assert right_frame.q[5] == pytest.approx(0.007 - 0.031 * 5, abs=1e-12)


# ---------------------------------------------------------------------------
# 首帧锚定与新鲜度
# ---------------------------------------------------------------------------


def test_first_frame_anchors_to_fresh_measured_posture(
    left_policy: R1DCommandPolicy,
) -> None:
    """首帧目标必须锚定新鲜实测姿态：零步长通过、远跳拒绝、过期/未来状态拒绝。"""
    state = _fresh_state()
    # 首帧远跳拒绝（策略尚无在途命令，以新鲜实测为基准；joint0 上限 0.004 rad）
    with pytest.raises(R1DCommandPolicyError, match="首帧"):
        left_policy.build_active(state, _left_q_desired_with(0, -0.5 + 0.05))

    # 锚定通过：q_desired = 实测换算 → 命令目标 == 实测（零步长）
    frame = left_policy.build_active(state, np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert_allclose(frame.q[15:22], state.q[15:22], atol=1e-12)

    # 过期状态拒绝（fresh 判定标准来自硬件配置 timing.state_timeout_s）
    stale = _fresh_state(received_at=_NOW - _TIMEOUT - 1e-6)
    with pytest.raises(R1DCommandPolicyError, match="新鲜"):
        left_policy.build_active(stale, np.array(_LEFT_Q_DESIRED_ANCHORED))
    # 未来时间戳拒绝（received_at 晚于当前时钟，无法判定新鲜）
    future = _fresh_state(received_at=_NOW + 0.1)
    with pytest.raises(R1DCommandPolicyError, match="新鲜"):
        left_policy.build_active(future, np.array(_LEFT_Q_DESIRED_ANCHORED))

    # 后续帧以在途上一命令为基准：joint6 上限 0.010 rad
    left_policy.build_active(state, _left_q_desired_with(6, 0.1 + 0.008))  # 0.008 通过
    with pytest.raises(R1DCommandPolicyError, match="在途"):
        left_policy.build_active(state, _left_q_desired_with(6, 0.108 + 0.011))


def test_rejects_missing_state(left_policy: R1DCommandPolicy) -> None:
    """缺状态（None/类型不符）拒绝构造命令。"""
    with pytest.raises(R1DCommandPolicyError, match="缺状态"):
        left_policy.build_active(None, np.array(_LEFT_Q_DESIRED_ANCHORED))


# ---------------------------------------------------------------------------
# 拒绝构造：未标定 / 机型不匹配 / 缺全机策略 / 缺 motion 级参数
# ---------------------------------------------------------------------------


def test_rejects_uncalibrated_config(tmp_path: Path) -> None:
    """校准为 null（未标定槽位）时拒绝构造命令；stop 返回 None。"""

    def null_calibration(data: dict[str, Any]) -> None:
        data["calibration"] = {
            "verified": False,
            "joint_signs": None,
            "joint_offsets_rad": None,
        }

    policy = _make_policy(tmp_path, "left", mutate=null_calibration)
    with pytest.raises(R1DCommandPolicyError, match="标定"):
        policy.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert policy.build_stop(_fresh_state(), emergency=False) is None


def test_rejects_mode_machine_mismatch(tmp_path: Path) -> None:
    """机型不匹配拒绝：仅在冻结了期望值时断言；无期望时只回填透传。"""
    policy = _make_policy(tmp_path, "left", expected_mode_machine=7)
    # 一致：通过
    frame = policy.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert frame.mode_machine == 7
    # 不一致：build_active 拒绝、build_stop 无法安全构造
    mismatch = _fresh_state(mode_machine=9)
    with pytest.raises(R1DCommandPolicyError, match="机型"):
        policy.build_active(mismatch, np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert policy.build_stop(mismatch, emergency=False) is None
    # 无冻结期望（默认 None）：不做相等断言，mode_machine 原样回填
    policy_open = _make_policy(tmp_path, "left")
    frame_open = policy_open.build_active(
        _fresh_state(mode_machine=9), np.array(_LEFT_Q_DESIRED_ANCHORED)
    )
    assert frame_open.mode_machine == 9


def test_rejects_missing_whole_body_policy(tmp_path: Path) -> None:
    """缺全机策略拒绝构造：任一非目标槽位无显式策略即列出全部 28 个缺失槽位。

    R5 修订（评审 I-05）：对侧臂/头不再回退 manifest 示例——不注入
    slot_policies 时全部 28 个非目标槽位（含 22-30）都被列为缺失。
    """
    policy_none = _make_policy(tmp_path, "left", slot_policies=None)
    with pytest.raises(R1DCommandPolicyError, match="全机") as excinfo:
        policy_none.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert _EXPECTED_MISSING_WHEN_NO_EXPLICIT in str(excinfo.value)
    assert "22" in str(excinfo.value)  # 对侧臂槽亦缺（无 manifest 回退）
    # stop 同样无法安全构造完整 35 槽报文 → None
    assert policy_none.build_stop(_fresh_state(), emergency=False) is None

    # 部分覆盖仍拒绝：已覆盖槽位不再列为缺失
    policy_partial = _make_policy(
        tmp_path, "left", slot_policies={13: _explicit_hold(13)}
    )
    with pytest.raises(R1DCommandPolicyError, match="全机") as excinfo2:
        policy_partial.build_active(
            _fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED)
        )
    message = str(excinfo2.value)
    assert "12" in message  # waist 之外的 lift_column 仍缺
    assert "13" not in message  # 已显式覆盖的 waist 不再列缺

    # 显式策略不能覆盖目标臂槽位（目标七槽由映射+配置计算）
    bad_policies: Mapping[int, R1DSlotHold] = {
        **_full_explicit_policies(),
        15: _explicit_hold(15),
    }
    policy_bad = _make_policy(tmp_path, "left", slot_policies=dict(bad_policies))
    with pytest.raises(R1DCommandPolicyError, match="目标"):
        policy_bad.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))


def test_rejects_missing_motion_numbers(tmp_path: Path) -> None:
    """硬件配置缺 motion 级参数（null 待确认）时拒绝构造；stop 返回 None。"""

    def null_dq_limit(data: dict[str, Any]) -> None:
        data["limits"]["dq_limit_rad_s"] = None

    policy = _make_policy(tmp_path, "left", mutate=null_dq_limit)
    with pytest.raises(R1DCommandPolicyError, match="dq_limit_rad_s"):
        policy.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert policy.build_stop(_fresh_state(), emergency=False) is None

    def null_period(data: dict[str, Any]) -> None:
        data["timing"]["control_period_s"] = None

    policy2 = _make_policy(
        tmp_path, "left", filename="hw_left_period.yaml", mutate=null_period
    )
    with pytest.raises(R1DCommandPolicyError, match="control_period_s"):
        policy2.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))


# ---------------------------------------------------------------------------
# 目标位置/速度变化约束
# ---------------------------------------------------------------------------


def test_target_change_constraints_enforced(
    left_policy: R1DCommandPolicy, tmp_path: Path
) -> None:
    """目标位置限位与单周期隐含速度约束按硬件配置生效，过界显式拒绝不静默截断。"""
    state = _fresh_state()

    # —— 位置限位：实测已在限位外邻域时，边界内小步目标仍被限位拒绝 ——
    edge_state = _fresh_state(
        q=_modified_measured((15, 1.55), (16, -1.55))
    )
    # joint0（sign=+1, offset=0）：实测 1.55 → 目标 1.552（步长 0.002 合规，越上界 1.5）
    with pytest.raises(R1DCommandPolicyError, match="限位"):
        left_policy.build_active(
            edge_state, np.array([1.552, 1.60, -0.2, 0.4, 0.2, 0.4, 0.1])
        )
    # joint1（sign=-1, offset=0.05）：实测 -1.55 → 目标 SDK -1.558 越下界 -1.5
    with pytest.raises(R1DCommandPolicyError, match="限位"):
        left_policy.build_active(
            edge_state, np.array([1.55, 1.608, -0.2, 0.4, 0.2, 0.4, 0.1])
        )

    # —— 隐含速度约束：逐关节独立限值，仅 joint3（上限 0.007 rad/周期）过界 ——
    left_policy.build_active(state, np.array(_LEFT_Q_DESIRED_ANCHORED))  # 锚定
    with pytest.raises(R1DCommandPolicyError, match="隐含速度"):
        left_policy.build_active(state, _left_q_desired_with(3, 0.408))  # 步长 0.008
    left_policy.build_active(state, _left_q_desired_with(3, 0.406))  # 步长 0.006 通过

    # —— 不静默截断：边界内请求原样出现在命令帧 ——
    frame = left_policy.build_active(state, _left_q_desired_with(3, 0.412))
    # SDK 侧 = -1*0.412 + 0.2 = -0.212（非截断值）
    assert_allclose(frame.q[18], -0.212, atol=1e-12)

    # —— 步长恰等于上限（0.004）时通过且目标原样保留 ——
    policy_fresh = _make_policy(tmp_path, "left", filename="hw_left_edge.yaml")
    near_state = _fresh_state(q=_modified_measured((15, 1.396)))
    frame_edge = policy_fresh.build_active(
        near_state, np.array([1.4, 0.45, -0.2, 0.4, 0.2, 0.4, 0.1])
    )
    assert_allclose(frame_edge.q[15], 1.4, atol=1e-12)


# ---------------------------------------------------------------------------
# stop 语义
# ---------------------------------------------------------------------------


def test_stop_without_fresh_state_returns_none(tmp_path: Path) -> None:
    """stop 无新鲜状态分支返回 None：不伪造保持角度、不抛异常。"""
    policy = _make_policy(tmp_path, "left")
    assert policy.build_stop(None, emergency=False) is None
    assert policy.build_stop(None, emergency=True) is None
    stale = _fresh_state(received_at=_NOW - _TIMEOUT - 1e-6)
    assert policy.build_stop(stale, emergency=False) is None
    assert policy.build_stop(stale, emergency=True) is None


def test_stop_with_fresh_state_holds_measured_posture(tmp_path: Path) -> None:
    """正常 stop 用新鲜 state 构造有限停止目标（锚定实测），并复位首帧基准。"""
    policy = _make_policy(tmp_path, "left")
    state = _fresh_state()

    frame = policy.build_stop(state, emergency=False)
    assert frame is not None
    # 目标七槽：锚定新鲜实测姿态的有限保持目标
    assert_allclose(frame.q[15:22], state.q[15:22], atol=1e-12)
    assert_allclose(frame.dq[15:22], np.zeros(7), atol=1e-12)
    assert_allclose(frame.tau[15:22], np.zeros(7), atol=1e-12)
    assert_allclose(frame.kp[15:22], _TEST_KP, atol=1e-12)
    assert_allclose(frame.kd[15:22], _TEST_KD, atol=1e-12)
    assert_allclose(frame.motor_mode[15:22], np.ones(7), atol=1e-12)
    # 头字段回填
    assert frame.mode_machine == state.mode_machine
    assert frame.mode_pr == 0
    # 非目标槽位与 active 同策略（对侧臂/头/显式哨兵）
    assert_allclose(frame.q[22:29], np.zeros(7), atol=1e-12)
    assert frame.kp[29] == pytest.approx(_HEAD_HOLD_KP[0], abs=1e-12)
    assert frame.q[13] == pytest.approx(_explicit_hold(13).q, abs=1e-12)

    # emergency 分支：同为有限保持（无已核验更强急停语义，差异化锁存归 4.6）
    emergency_frame = policy.build_stop(state, emergency=True)
    assert emergency_frame is not None
    assert_allclose(emergency_frame.q, frame.q, atol=1e-12)
    assert_allclose(emergency_frame.kp, frame.kp, atol=1e-12)
    assert_allclose(emergency_frame.kd, frame.kd, atol=1e-12)
    assert emergency_frame.mode_machine == frame.mode_machine

    # stop 复位首帧锚定基准：停止后实测移动 0.5 rad，新 active 以新实测为基准通过
    state_moved = _fresh_state(q=_modified_measured((21, 0.6)))
    frame_resume = policy.build_active(
        state_moved, np.array([-0.5, 0.45, -0.2, 0.4, 0.2, 0.4, 0.6])
    )
    assert_allclose(frame_resume.q[21], 0.6, atol=1e-12)


# ---------------------------------------------------------------------------
# 头字段与可记录性
# ---------------------------------------------------------------------------


def test_mode_pr_override_and_uint8(tmp_path: Path) -> None:
    """mode_pr 默认 0（Ruling R6），可显式覆盖；越界 uint8 拒绝。"""
    policy = _make_policy(tmp_path, "left", mode_pr=5)
    frame = policy.build_active(_fresh_state(), np.array(_LEFT_Q_DESIRED_ANCHORED))
    assert frame.mode_pr == 5
    with pytest.raises(ValueError, match="mode_pr"):
        _make_policy(tmp_path, "left", mode_pr=256)
    with pytest.raises(ValueError, match="mode_pr"):
        _make_policy(tmp_path, "left", mode_pr=-1)


def test_last_target_is_recordable(left_policy: R1DCommandPolicy) -> None:
    """实际构造的目标可记录：属性返回副本，拒绝构造不更新记录。"""
    state = _fresh_state()
    frame = left_policy.build_active(state, np.array(_LEFT_Q_DESIRED_ANCHORED))
    last = left_policy.last_target_q_sdk
    assert last is not None
    assert last.shape == (7,)
    assert_allclose(last, frame.q[15:22], atol=1e-12)
    # 返回副本：外部修改不污染内部记录
    last[0] = 99.0
    assert left_policy.last_target_q_sdk is not None
    assert left_policy.last_target_q_sdk[0] != 99.0
    # 拒绝构造不更新记录
    with pytest.raises(R1DCommandPolicyError):
        left_policy.build_active(state, _left_q_desired_with(0, -0.5 + 0.05))
    assert_allclose(
        left_policy.last_target_q_sdk, frame.q[15:22], atol=1e-12
    )


# ---------------------------------------------------------------------------
# 构造参数与 R1DSlotHold 校验
# ---------------------------------------------------------------------------


def test_constructor_validates_slot_policies(tmp_path: Path) -> None:
    """显式槽位策略的键/值类型在构造时校验。"""
    with pytest.raises(ValueError, match="槽位"):
        _make_policy(tmp_path, "left", slot_policies={35: _explicit_hold(35)})
    with pytest.raises(ValueError, match="槽位"):
        _make_policy(tmp_path, "left", slot_policies={-1: _explicit_hold(-1)})
    with pytest.raises(ValueError, match="R1DSlotHold"):
        _make_policy(tmp_path, "left", slot_policies={13: "not-a-hold"})  # type: ignore[dict-item]


def test_slot_hold_validates_fields() -> None:
    """R1DSlotHold 字段校验：mode 整数 uint8、数值有限、增益非负。"""
    with pytest.raises(ValueError, match="mode"):
        R1DSlotHold(mode=0.5, q=0.0, dq=0.0, kp=1.0, kd=1.0, tau=0.0)
    with pytest.raises(ValueError, match="mode"):
        R1DSlotHold(mode=256, q=0.0, dq=0.0, kp=1.0, kd=1.0, tau=0.0)
    with pytest.raises(ValueError, match="有限"):
        R1DSlotHold(mode=1, q=float("nan"), dq=0.0, kp=1.0, kd=1.0, tau=0.0)
    with pytest.raises(ValueError, match="kp"):
        R1DSlotHold(mode=1, q=0.0, dq=0.0, kp=-1.0, kd=1.0, tau=0.0)
    with pytest.raises(ValueError, match="kd"):
        R1DSlotHold(mode=1, q=0.0, dq=0.0, kp=1.0, kd=-0.1, tau=0.0)
    # 合法值（tau 允许为负：前馈力矩方向任意）
    hold = R1DSlotHold(mode=1, q=-0.1, dq=0.0, kp=10.0, kd=0.5, tau=-2.0)
    assert hold.mode == 1
    assert hold.tau == -2.0
