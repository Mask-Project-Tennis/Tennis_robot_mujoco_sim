"""R1-A7-D 真实 SDK 序列化/CRC 对照测试（阶段 4 Task 4.5）。

在本机（已安装 unitree_sdk2py rev-pin 克隆 + cyclonedds）做无网络 IDL
encode/decode，与按 manifest 冻结布局独立实现的参考 packer/参考 CRC 互验；
不是 mock 自身 encode 函数的自证。缺依赖机器按 Ruling R8 明确 skip，
本机（SDK 已装）必须真跑并全过，不得 skip。

事实源：configs/robots/r1d_protocol_manifest.yaml
  - dds.lowcmd_layout.serialized_length_bytes = 1004
  - dds.lowstate_layout.serialized_length_bytes = 2092
  - dds.crc 节：非反射 MSB-first，多项式 0x04C11DB7，初值 0xFFFFFFFF，
    无最终异或，4 字节小端字对齐覆盖除末尾 crc 外全部
  - whole_body.selected_arm.joint_indices_left = [15..21]（槽 21 = 左臂第七关节）
环境事实（manifest sdk.local_dev_install_note，Ruling R4/R8）：cyclonedds 11.1.0
运行时下生成 IDL 类要求显式全字段构造（无默认值），适配层使用显式工厂。

**wire 布局证据边界（I-04 勘误，评审 2026-11-17）**：1004/2092 是官方
crc.py 格式串的 **CRC 结构像长度**（"#size" 注释语义），不是 cyclonedds
serialize() 总长的承诺值。本机 cyclonedds 11.1.0 实测 serialize() 总长
恰为 1004/2092 属"4 B CDR 头 + 缺 4 B 的 body"的巧合：其 wire body 把
motor_cmd[0].mode（LowCmd）/ motor_state[0].mode（LowState）连同 3 B
padding 折叠，与冻结布局**不互操作**（用官方布局字节反序列化得 m0.q=0.0
垃圾值）。因此本文件只声称：官方 CRC 与独立参考打包互验（③）、本机
自洽往返保真（④）、wire body 与"本机已知偏差布局"逐字节一致（②b）；
**wire 兼容证据待部署机 cyclonedds 0.10.2 复跑**（合规布局 body 应与
参考体逐字节一致，届时总长为 1008/2096）。
"""

from __future__ import annotations

import struct

import pytest

pytest.importorskip(
    "unitree_sdk2py",
    reason="缺少 unitree_sdk2py：真实序列化对照只能在装有 SDK 的主机执行（Ruling R8 允许缺依赖机器明确 skip）",
)
pytest.importorskip("cyclonedds", reason="缺少 cyclonedds Python 绑定")

import numpy as np
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (
    IMUState_,
    LowCmd_,
    LowState_,
    MotorCmd_,
    MotorState_,
)
from unitree_sdk2py.utils.crc import CRC

from src.real.r1d_transport import (
    _build_lowcmd,
    _load_sdk_bundle,
    _state_frame_from_lowstate,
)
from src.real.r1d_types import R1D_SLOT_COUNT, R1DCommandFrame

# manifest 冻结的序列化长度
MANIFEST_LOWCMD_BYTES = 1004
MANIFEST_LOWSTATE_BYTES = 2092
# 左臂第七关节槽位（manifest whole_body.selected_arm.joint_indices_left[6]）
LEFT_SEVENTH_JOINT_SLOT = 21

# 官方 crc.py 打包格式串（evidence 引用：unitree_sdk2py/utils/crc.py L23/L25，
# 注释 "#size 1004"/"#size 2092"）——长度推算对照用，非本项目实现
OFFICIAL_LOWCMD_FMT = "<2B2x" + "B3x5fI" * 35 + "5I"
OFFICIAL_LOWSTATE_FMT = "<2I2B2xI" + "13fh2x" + "B3x4f2hf7I" * 35 + "40B5I"


# ---------------------------------------------------------------------------
# 独立参考实现（按 manifest 冻结布局/算法编写，与官方 crc.py 代码互不复制）
# ---------------------------------------------------------------------------


def reference_crc32(data: bytes) -> int:
    """独立参考 CRC32（manifest dds.crc 节冻结算法）。

    非反射 MSB-first：多项式 0x04C11DB7，初值 0xFFFFFFFF，无最终异或；
    输入按 4 字节小端字读取，覆盖除末尾 4 字节（crc 字段）外的全部内容。

    实现形式与官方 crc.py 的逐位移位写法独立：本实现用标准反馈式——
    每输入位与寄存器最高位异或决定是否回乘多项式。

    Args:
        data: 含末尾 crc 字段占位在内的完整打包字节。

    Returns:
        32 位 CRC 值。
    """
    crc = 0xFFFFFFFF
    for i in range((len(data) >> 2) - 1):  # 覆盖除最后一个字（crc）外全部
        word = int.from_bytes(data[4 * i : 4 * i + 4], "little")
        for shift in range(31, -1, -1):
            feedback = ((crc >> 31) & 1) ^ ((word >> shift) & 1)
            crc = (crc << 1) & 0xFFFFFFFF
            if feedback:
                crc ^= 0x04C11DB7
    return crc


def reference_pack_lowcmd(cmd: LowCmd_) -> bytes:
    """独立参考打包：hg LowCmd_ 按 manifest lowcmd_layout 逐字段序列化。

    与官方 crc.py 的复合格式串独立：本函数逐字段 struct.pack 拼
    （uint8 + 显式对齐填充），布局语义同 manifest 冻结字段表。
    """
    out = bytearray()
    out += struct.pack("<2B2x", int(cmd.mode_pr), int(cmd.mode_machine))
    for m in cmd.motor_cmd:
        out += struct.pack(
            "<B3x5fI",
            int(m.mode),
            float(m.q),
            float(m.dq),
            float(m.tau),
            float(m.kp),
            float(m.kd),
            int(m.reserve),
        )
    out += struct.pack(
        "<5I",
        int(cmd.reserve[0]),
        int(cmd.reserve[1]),
        int(cmd.reserve[2]),
        int(cmd.reserve[3]),
        int(cmd.crc),
    )
    return bytes(out)


def reference_pack_lowstate(state: LowState_) -> bytes:
    """独立参考打包：hg LowState_ 按 manifest lowstate_layout 逐字段序列化。"""
    out = bytearray()
    out += struct.pack(
        "<2I2B2xI",
        int(state.version[0]),
        int(state.version[1]),
        int(state.mode_pr),
        int(state.mode_machine),
        int(state.tick),
    )
    imu = state.imu_state
    out += struct.pack(
        "<13fh2x",
        *[float(v) for v in imu.quaternion],
        *[float(v) for v in imu.gyroscope],
        *[float(v) for v in imu.accelerometer],
        *[float(v) for v in imu.rpy],
        int(imu.temperature),
    )
    for m in state.motor_state:
        out += struct.pack(
            "<B3x4f2hf7I",
            int(m.mode),
            float(m.q),
            float(m.dq),
            float(m.ddq),
            float(m.tau_est),
            int(m.temperature[0]),
            int(m.temperature[1]),
            float(m.vol),
            int(m.sensor[0]),
            int(m.sensor[1]),
            int(m.motorstate),
            *[int(r) for r in m.reserve],
        )
    out += struct.pack(
        "<40B5I",
        *[int(b) for b in state.wireless_remote],
        *[int(r) for r in state.reserve],
        int(state.crc),
    )
    return bytes(out)


# ---------------------------------------------------------------------------
# 哨兵构造（float32 可精确表示值；测试专用，非实机参数）
# ---------------------------------------------------------------------------


def sentinel_command_frame() -> R1DCommandFrame:
    """哨兵命令帧：35 槽每槽每字段取唯一可分辨值（float32 精确）。"""
    n = R1D_SLOT_COUNT
    return R1DCommandFrame(
        q=np.array([0.25 * (i + 1) for i in range(n)]),
        dq=np.array([-0.125 * (i + 1) for i in range(n)]),
        kp=np.array([float(i + 1) for i in range(n)]),
        kd=np.array([float(i % 7) for i in range(n)]),
        tau=np.array([0.5 * (i + 1) for i in range(n)]),
        motor_mode=np.array([float(i % 2) for i in range(n)]),
        mode_machine=7,
        mode_pr=0,
    )


def sentinel_lowstate() -> LowState_:
    """哨兵状态帧：显式全字段构造（cyclonedds 11.1.0 无默认值）。"""
    imu = IMUState_(
        quaternion=[1.0, 0.0, 0.0, 0.0],
        gyroscope=[0.25, -0.25, 0.5],
        accelerometer=[0.125, 0.25, 9.8125],
        rpy=[0.0, 0.0, 0.0625],
        temperature=25,
    )
    motors = [
        MotorState_(
            mode=i % 2,
            q=0.5 * (i + 1),
            dq=-0.25 * (i + 1),
            ddq=0.125 * (i + 1),
            tau_est=0.0625 * (i + 1),
            temperature=[30, 31],
            vol=24.0 + 0.5 * i,
            sensor=[0, 0],
            motorstate=0,
            reserve=[0, 0, 0, 0],
        )
        for i in range(R1D_SLOT_COUNT)
    ]
    return LowState_(
        version=[1, 2],
        mode_pr=0,
        mode_machine=7,
        tick=123456,
        imu_state=imu,
        motor_state=motors,
        wireless_remote=[0] * 40,
        reserve=[0, 0, 0, 0],
        crc=0,
    )


@pytest.fixture(scope="module")
def real_bundle():
    """真实 SDK bundle（module 级加载一次）。"""
    return _load_sdk_bundle()


# ---------------------------------------------------------------------------
# ① 显式全字段工厂（R8：cyclonedds 11.1.0 无默认值）
# ---------------------------------------------------------------------------


def test_explicit_full_field_construction_works() -> None:
    """cyclonedds 11.1.0 下 IDL 类显式全字段构造可用（适配层显式工厂依据）。"""
    motor = MotorCmd_(mode=1, q=0.25, dq=-0.5, tau=1.5, kp=60.0, kd=2.0, reserve=7)
    assert motor.mode == 1 and motor.reserve == 7
    motors = [
        MotorCmd_(mode=1, q=0.0, dq=0.0, tau=0.0, kp=0.0, kd=0.0, reserve=0)
        for _ in range(R1D_SLOT_COUNT)
    ]
    cmd = LowCmd_(mode_pr=0, mode_machine=7, motor_cmd=motors, reserve=[0, 0, 0, 0], crc=0)
    assert len(cmd.motor_cmd) == R1D_SLOT_COUNT


# ---------------------------------------------------------------------------
# ② 序列化字节长度对照 manifest 冻结值（官方布局推算 + cyclonedds 实测两种方法）
# ---------------------------------------------------------------------------


def test_serialized_lengths_match_manifest() -> None:
    """长度对照（布局证据边界版，I-04 勘误）：官方 crc.py 布局推算 =
    独立参考打包 = manifest 冻结值（LowCmd 1004 / LowState 2092）。

    cyclonedds 原生 serialize() 总长**不在此断言**——1004/2092 是 CRC
    结构像长度；本机 11.1.0 的原生总长恰等于该值属"4 B CDR 头 + 缺 4 B
    body"的巧合（见 test_native_wire_body_layout_documented）。"""
    # 方法 A：官方 crc.py 布局推算（格式串引用 crc.py L23/L25）
    assert struct.calcsize(OFFICIAL_LOWCMD_FMT) == MANIFEST_LOWCMD_BYTES
    assert struct.calcsize(OFFICIAL_LOWSTATE_FMT) == MANIFEST_LOWSTATE_BYTES
    # 方法 B：独立参考打包实测（按 manifest 冻结布局逐字段打包）
    cmd = _build_lowcmd(sentinel_command_frame(), _load_sdk_bundle())
    assert len(reference_pack_lowcmd(cmd)) == MANIFEST_LOWCMD_BYTES
    state = sentinel_lowstate()
    assert len(reference_pack_lowstate(state)) == MANIFEST_LOWSTATE_BYTES


def test_native_wire_body_layout_documented(real_bundle) -> None:
    """cyclonedds 原生 wire body 布局双分支对照（I-04，评审 2026-11-17）。

    - 合规布局（部署机 cyclonedds 0.10.2 期望）：body 与参考体**逐字节
      一致**（body 长度 = 1004/2092，serialize 总长 = 1008/2096）；
    - 本机 R4 偏差布局（cyclonedds 11.1.0）：body 缺首槽 mode+3 B padding
      （LowCmd: body == ref[:4]+ref[8:]；LowState: body == ref[:75]+ref[79:]），
      与冻结布局不互操作——wire 兼容证据待部署机复跑后升级。

    本测试把两种布局都显式断言（部署机复跑时自动落入合规分支），
    不用"总长 == 1004"的巧合冒充布局一致。
    """
    cmd = _build_lowcmd(sentinel_command_frame(), real_bundle)
    native_cmd = bytes(LowCmd_.serialize(cmd))
    assert native_cmd[:4] == b"\x00\x01\x00\x00"  # CDR-LE 封装头
    body_cmd = native_cmd[4:]
    ref_cmd = reference_pack_lowcmd(cmd)
    if len(body_cmd) == MANIFEST_LOWCMD_BYTES:
        # 合规布局（部署机期望分支）
        assert body_cmd == ref_cmd
    else:
        # 本机 11.1.0 已知偏差：折叠 motor_cmd[0].mode + 3B padding
        assert len(body_cmd) == MANIFEST_LOWCMD_BYTES - 4
        assert body_cmd == ref_cmd[:4] + ref_cmd[8:]

    state = sentinel_lowstate()
    state.crc = 0
    native_state = bytes(LowState_.serialize(state))
    assert native_state[:4] == b"\x00\x01\x00\x00"
    body_state = native_state[4:]
    ref_state = reference_pack_lowstate(state)
    if len(body_state) == MANIFEST_LOWSTATE_BYTES:
        assert body_state == ref_state
    else:
        # 本机 11.1.0 已知偏差：折叠 motor_state[0].mode + 3B padding
        # （首个差异字节位于 IMU 节后第一槽 mode 位置，idx=75）
        assert len(body_state) == MANIFEST_LOWSTATE_BYTES - 4
        assert body_state == ref_state[:75] + ref_state[79:]


# ---------------------------------------------------------------------------
# ③ 官方 CRC 与独立参考 CRC 向量互验（命令 + 状态）
# ---------------------------------------------------------------------------


def test_official_crc_equals_reference_lowcmd(real_bundle) -> None:
    """哨兵 LowCmd_：官方 CRC().Crc == 独立参考 CRC（且非零防平凡相等）。"""
    cmd = _build_lowcmd(sentinel_command_frame(), real_bundle)
    official = int(CRC().Crc(cmd))
    reference = reference_crc32(reference_pack_lowcmd(cmd))
    assert official == reference
    assert official != 0


def test_official_crc_equals_reference_lowstate() -> None:
    """哨兵 LowState_：官方 CRC().Crc == 独立参考 CRC（且非零）。"""
    state = sentinel_lowstate()
    official = int(CRC().Crc(state))
    reference = reference_crc32(reference_pack_lowstate(state))
    assert official == reference
    assert official != 0


# ---------------------------------------------------------------------------
# ④ cyclonedds 原生往返保真（全部 35 槽 + 左臂第七关节槽 21）
# ---------------------------------------------------------------------------


def test_native_roundtrip_preserves_all_slots_and_seventh_joint(real_bundle) -> None:
    """原生 serialize→deserialize 往返：35 槽全字段 + 槽 21（左臂第七关节）保真。"""
    cmd = _build_lowcmd(sentinel_command_frame(), real_bundle)
    cmd.crc = int(CRC().Crc(cmd))
    restored = LowCmd_.deserialize(LowCmd_.serialize(cmd))
    assert restored.mode_pr == cmd.mode_pr
    assert restored.mode_machine == cmd.mode_machine
    assert list(restored.reserve) == [0, 0, 0, 0]
    assert restored.crc == cmd.crc
    for i in range(R1D_SLOT_COUNT):
        src, dst = cmd.motor_cmd[i], restored.motor_cmd[i]
        assert dst.mode == src.mode
        assert dst.q == src.q
        assert dst.dq == src.dq
        assert dst.tau == src.tau
        assert dst.kp == src.kp
        assert dst.kd == src.kd
        assert dst.reserve == src.reserve
    # 左臂第七关节槽位（21）哨兵特查
    seventh = restored.motor_cmd[LEFT_SEVENTH_JOINT_SLOT]
    assert seventh.q == 0.25 * (LEFT_SEVENTH_JOINT_SLOT + 1)
    assert seventh.kd == float(LEFT_SEVENTH_JOINT_SLOT % 7)

    state = sentinel_lowstate()
    state.crc = int(CRC().Crc(state))
    restored_state = LowState_.deserialize(LowState_.serialize(state))
    assert list(restored_state.version) == [1, 2]
    assert restored_state.tick == 123456
    assert restored_state.mode_machine == 7
    assert restored_state.crc == state.crc
    for i in range(R1D_SLOT_COUNT):
        src, dst = state.motor_state[i], restored_state.motor_state[i]
        assert dst.q == src.q and dst.dq == src.dq
        assert dst.tau_est == src.tau_est and dst.vol == src.vol
        assert list(dst.temperature) == list(src.temperature)
        assert dst.motorstate == src.motorstate
    assert restored_state.motor_state[LEFT_SEVENTH_JOINT_SLOT].q == 0.5 * (
        LEFT_SEVENTH_JOINT_SLOT + 1
    )


def test_adapter_lowcmd_fidelity_all_slots(real_bundle) -> None:
    """适配器保真：R1DCommandFrame 哨兵 → 显式 LowCmd_ → 原生 wire 往返后仍逐字段一致。"""
    frame = sentinel_command_frame()
    cmd = _build_lowcmd(frame, real_bundle)
    assert cmd.mode_pr == frame.mode_pr
    assert cmd.mode_machine == frame.mode_machine
    for i in range(R1D_SLOT_COUNT):
        motor = cmd.motor_cmd[i]
        assert motor.mode == int(frame.motor_mode[i])
        assert motor.q == frame.q[i]
        assert motor.dq == frame.dq[i]
        assert motor.tau == frame.tau[i]
        assert motor.kp == frame.kp[i]
        assert motor.kd == frame.kd[i]
        assert motor.reserve == 0  # manifest lowcmd_layout：保留字段示例写 0
    # wire 往返后保持（float32 精确哨兵无舍入）
    restored = LowCmd_.deserialize(LowCmd_.serialize(cmd))
    for i in range(R1D_SLOT_COUNT):
        assert restored.motor_cmd[i].q == frame.q[i]
        assert restored.motor_cmd[i].tau == frame.tau[i]
    assert restored.motor_cmd[LEFT_SEVENTH_JOINT_SLOT].q == frame.q[
        LEFT_SEVENTH_JOINT_SLOT
    ]


def test_adapter_state_frame_fidelity() -> None:
    """适配器保真：显式 LowState_ 哨兵 → R1DStateFrame（tick→sequence、
    source_time=None、槽 21 含 seventh joint 值、数组只读）。"""
    state = sentinel_lowstate()
    frame = _state_frame_from_lowstate(
        state, session_id=9, received_at=1234.5
    )
    for i in range(R1D_SLOT_COUNT):
        assert frame.q[i] == state.motor_state[i].q
        assert frame.dq[i] == state.motor_state[i].dq
    assert frame.q[LEFT_SEVENTH_JOINT_SLOT] == 0.5 * (LEFT_SEVENTH_JOINT_SLOT + 1)
    assert frame.sequence == 123456  # sequence 承接 LowState_.tick
    assert frame.source_time is None  # 协议无源时间戳，不得编造
    assert frame.mode_machine == 7
    assert frame.received_at == 1234.5
    assert frame.session_id == 9
    assert frame.q.flags.writeable is False
    assert frame.dq.flags.writeable is False


# ---------------------------------------------------------------------------
# ⑤ CRC 增量敏感性（单槽单字段改动 → CRC 变化；多向量继续互验）
# ---------------------------------------------------------------------------


def test_crc_incremental_sensitivity(real_bundle) -> None:
    """CRC 增量敏感性：改槽 21 单 q、改槽 0 单 tau、改 mode_machine 均使 CRC 变化，
    且每个向量上官方与参考实现仍相等。"""
    base = _build_lowcmd(sentinel_command_frame(), real_bundle)
    crc_base_official = int(CRC().Crc(base))
    crc_base_ref = reference_crc32(reference_pack_lowcmd(base))
    assert crc_base_official == crc_base_ref

    # 改动 1：左臂第七关节（槽 21）单字段 q
    m1 = _build_lowcmd(sentinel_command_frame(), real_bundle)
    m1.motor_cmd[LEFT_SEVENTH_JOINT_SLOT].q += 0.5
    crc1_official = int(CRC().Crc(m1))
    crc1_ref = reference_crc32(reference_pack_lowcmd(m1))
    assert crc1_official != crc_base_official
    assert crc1_official == crc1_ref

    # 改动 2：槽 0 单字段 tau
    m2 = _build_lowcmd(sentinel_command_frame(), real_bundle)
    m2.motor_cmd[0].tau += 1.0
    crc2_official = int(CRC().Crc(m2))
    crc2_ref = reference_crc32(reference_pack_lowcmd(m2))
    assert crc2_official not in (crc_base_official, crc1_official)
    assert crc2_official == crc2_ref

    # 改动 3：头字段 mode_machine
    m3 = _build_lowcmd(sentinel_command_frame(), real_bundle)
    m3.mode_machine = 9
    crc3_official = int(CRC().Crc(m3))
    crc3_ref = reference_crc32(reference_pack_lowcmd(m3))
    assert crc3_official != crc_base_official
    assert crc3_official == crc3_ref
