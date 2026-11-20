"""R1-A7-D 独立构造工厂与 RM65 兼容边界测试（阶段 4 Task 4.7）。

验证 `src/real/r1d_factory.py::build_r1d_arm`：
- 构造零传输副作用（open/acquire/publish/release/close 计数全 0）；
- 错误 profile（robot_id 不符）/ 错误 side / 硬件配置与 profile 不一致
  在**传输创建前**失败（monkeypatch 传输工厂 spy 断言零调用）；
- 真实路径显式透传 acquire_timeout_s（必填、无默认，锁死不静默依赖
  传输层 10.0 工程默认）与共享 clock（传输与驱动拿到同一 callable）；
- 注入 fake transport 时工厂构造路径零 SDK import（子进程锁死）、
  不把 FakeRobot 六维行为用于 R1D；
- RM65 `build_robot_arm` 默认语义不受影响（R1D profile 仍被其拒绝）。

所有 sign/offset/增益/限值/超时/阈值均为**测试专用有限参数，非实机参数**。
fake clock / fake transport 拷贝自 tests/test_unitree_r1d_interface.py 的
4.6 测试模式（保持本文件自包含，不跨测试文件 import）。
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterator

import numpy as np
import pytest
import yaml

from src.real.config import RealRobotConfig
from src.real.r1d_config import R1DHardwareConfig, load_r1d_hardware_config
from src.real.r1d_factory import build_r1d_arm
from src.real.r1d_transport import SdkR1DTransport
from src.real.runner_factory import build_robot_arm
from src.real.unitree_r1d_interface import UnitreeR1DInterface
from src.robot.robot_profile import load_robot_profile

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEFT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_left.yaml"
_RIGHT_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_right.yaml"
_RM65_PROFILE_PATH = _REPO_ROOT / "configs" / "robots" / "rm65.yaml"
# 4.2 冻结的真实硬件模板（offline 档可解析；本测试只读不写）
_LEFT_HW_TEMPLATE_PATH = _REPO_ROOT / "configs" / "robots" / "r1d_hardware_left.yaml"

_LEFT_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)

# —— 测试专用参数（非实机参数）——
# 刻意不等于传输层工程默认 10.0：锁死"工厂显式透传"而非静默依赖默认
_TEST_ACQUIRE_TIMEOUT_S = 3.5  # s（测试专用，真实数值冻结时点在 G4 现场配置）
_TEST_PERIOD = 0.002  # s（数值同官方示例，仅测试用）
_COMMAND_TIMEOUT = 0.2  # s（≥ 2×周期，测试专用）


# ---------------------------------------------------------------------------
# 测试辅助（fake clock / fake transport / 配置构造 / spy）
# ---------------------------------------------------------------------------


class FakeClock:
    """可注入的可变单调时钟（测试专用）。"""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


class FakeR1DTransport:
    """测试专用 fake 传输层（记录全部调用；拷贝自 4.6 测试模式）。"""

    def __init__(self) -> None:
        self.open_count = 0
        self.acquire_count = 0
        self.publish_count = 0
        self.release_count = 0
        self.close_count = 0

    def open(self, on_state: Callable[[Any], None]) -> None:
        self.open_count += 1

    def acquire_control(self) -> None:
        self.acquire_count += 1

    def publish(self, command: Any) -> None:
        self.publish_count += 1

    def release_control(self) -> None:
        self.release_count += 1

    def close(self) -> None:
        self.close_count += 1


def _write_motion_hw_config(tmp_path: Path) -> Path:
    """写 motion 级测试专用硬件配置 YAML（有限参数，非实机参数）。

    与 4.6 测试的 _base_config_dict 同源：offline/read_only/motion 三档
    均可通过结构校验，工厂构造期（offline 档）不拒绝。
    """
    slots = {name: 15 + i for i, name in enumerate(_LEFT_JOINT_NAMES)}
    data = {
        "schema_version": 1,
        "robot_id": "unitree_r1d",
        "side": "left",
        "communication": {
            "dds_domain": 0,
            "network_interface": "eth-test",  # 测试专用，非实机参数
            "lowcmd_topic": "rt/lowcmd",
            "lowstate_topic": "rt/lowstate",
        },
        "layout": {"layout_verified": True, "joint_slots": slots},
        "calibration": {
            "verified": True,
            "joint_signs": {n: 1 for n in _LEFT_JOINT_NAMES},
            "joint_offsets_rad": {n: 0.0 for n in _LEFT_JOINT_NAMES},
        },
        "limits": {
            "verified": True,
            "q_lower_rad": {n: -1.5 for n in _LEFT_JOINT_NAMES},
            "q_upper_rad": {n: 1.5 for n in _LEFT_JOINT_NAMES},
            "dq_limit_rad_s": {n: 2.0 for n in _LEFT_JOINT_NAMES},
            "tau_limit_nm": {n: 30.0 for n in _LEFT_JOINT_NAMES},
            "kp_nm_per_rad": {n: 60.0 for n in _LEFT_JOINT_NAMES},
            "kd_nms_per_rad": {n: 1.0 for n in _LEFT_JOINT_NAMES},
        },
        "timing": {
            "control_period_s": _TEST_PERIOD,
            "state_timeout_s": 0.05,  # s（测试专用）
            "diagnostic_duration_s": 5.0,  # s（测试专用）
        },
    }
    path = tmp_path / "hw_left_test.yaml"
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


@pytest.fixture
def left_profile() -> Iterator[Any]:
    """仓库真实 R1D 左臂描述（路径存在性是测试前提）。"""
    assert _LEFT_PROFILE_PATH.is_file()
    yield load_robot_profile(_LEFT_PROFILE_PATH)


def _spy_transport_factory(
    monkeypatch: pytest.MonkeyPatch, transport: FakeR1DTransport
) -> list[dict[str, Any]]:
    """把 SdkR1DTransport.from_hardware_config 替换为记录调用的 spy。

    spy 返回注入的 fake transport：断言"真实路径"参数透传时不触发
    真实 SDK 加载/DDS（本任务红线：测试不得触发真实 SDK 加载）。
    """
    calls: list[dict[str, Any]] = []

    def _spy(cls: type, config: R1DHardwareConfig, **kwargs: Any) -> FakeR1DTransport:
        calls.append({"config": config, **kwargs})
        return transport

    monkeypatch.setattr(SdkR1DTransport, "from_hardware_config", classmethod(_spy))
    return calls


# ---------------------------------------------------------------------------
# 构造零副作用与必填参数
# ---------------------------------------------------------------------------


class TestFactoryConstruction:
    """构造不 connect / acquire / publish；acquire_timeout_s 必填。"""

    def test_construction_zero_transport_calls(
        self, tmp_path: Path, left_profile: Any
    ) -> None:
        """工厂构造后 fake transport 五类调用计数全 0，phase=CLOSED。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        fake = FakeR1DTransport()
        arm = build_r1d_arm(
            left_profile,
            config,
            acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,  # 测试专用，非实机参数
            transport=fake,
            clock=FakeClock(),
        )
        assert isinstance(arm, UnitreeR1DInterface)
        assert arm.get_status().phase == "CLOSED"
        assert fake.open_count == 0
        assert fake.acquire_count == 0
        assert fake.publish_count == 0
        assert fake.release_count == 0
        assert fake.close_count == 0

    def test_acquire_timeout_s_is_required(
        self, tmp_path: Path, left_profile: Any
    ) -> None:
        """acquire_timeout_s 无默认值：缺省调用直接 TypeError（裁定 1）。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        with pytest.raises(TypeError, match="acquire_timeout_s"):
            build_r1d_arm(
                left_profile,
                config,
                transport=FakeR1DTransport(),
                clock=FakeClock(),
            )


# ---------------------------------------------------------------------------
# 失败路径在传输创建前（brief 复选框 1 / 裁定 4）
# ---------------------------------------------------------------------------


class TestFailuresBeforeTransportCreation:
    """错误 profile/side/硬件配置在传输创建前失败。"""

    def test_wrong_robot_id_fails_before_transport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RM65 profile（robot_id 不符）+ R1D 硬件配置 → 传输工厂零调用。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        assert _RM65_PROFILE_PATH.is_file()
        rm65_profile = load_robot_profile(_RM65_PROFILE_PATH)
        calls = _spy_transport_factory(monkeypatch, FakeR1DTransport())
        with pytest.raises(ValueError, match="robot_id"):
            build_r1d_arm(
                rm65_profile,
                config,
                acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
                clock=FakeClock(),
            )
        assert calls == []  # 失败路径未触碰传输工厂

    def test_wrong_side_fails_before_transport(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """右臂 profile + 左臂硬件配置（side 冲突）→ 传输工厂零调用。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        assert _RIGHT_PROFILE_PATH.is_file()
        right_profile = load_robot_profile(_RIGHT_PROFILE_PATH)
        calls = _spy_transport_factory(monkeypatch, FakeR1DTransport())
        with pytest.raises(ValueError, match="side"):
            build_r1d_arm(
                right_profile,
                config,
                acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
                clock=FakeClock(),
            )
        assert calls == []

    def test_invalid_hardware_config_fails_before_transport(
        self, tmp_path: Path, left_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """joint_slots 与 profile 关节名不一致（结构绕过 load 的非法配置）。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        # 直接构造绕过 load 的结构校验：替换一个槽位键为不存在的关节名
        broken_slots = dict(config.layout.joint_slots)
        broken_slots.pop(_LEFT_JOINT_NAMES[0])
        broken_slots["left_nonexistent_joint"] = 99
        broken = replace(
            config,
            layout=replace(
                config.layout, joint_slots=MappingProxyType(broken_slots)
            ),
        )
        assert isinstance(broken, R1DHardwareConfig)
        calls = _spy_transport_factory(monkeypatch, FakeR1DTransport())
        with pytest.raises(ValueError, match="joint_slots"):
            build_r1d_arm(
                left_profile,
                broken,
                acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
                clock=FakeClock(),
            )
        assert calls == []


# ---------------------------------------------------------------------------
# 真实路径：acquire_timeout_s 显式透传 + 共享 clock（裁定 1 / 2）
# ---------------------------------------------------------------------------


class TestRealTransportPath:
    """真实路径（未注入 transport）经 from_hardware_config 构造。"""

    def test_forwards_acquire_timeout_and_shared_clock(
        self, tmp_path: Path, left_profile: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """透传 acquire_timeout_s（非 10.0 默认）；transport 与驱动同一 clock。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        fake = FakeR1DTransport()
        calls = _spy_transport_factory(monkeypatch, fake)
        shared_clock = FakeClock()
        arm = build_r1d_arm(
            left_profile,
            config,
            acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
            clock=shared_clock,
        )
        # 真实路径恰好一次传输构造，参数显式透传
        assert len(calls) == 1
        assert calls[0]["config"] is config
        assert calls[0]["acquire_timeout_s"] == _TEST_ACQUIRE_TIMEOUT_S
        assert calls[0]["acquire_timeout_s"] != 10.0  # 不静默依赖工程默认
        # 共享时钟：传输工厂与驱动拿到同一 callable（停止确认判据同一时基）
        assert calls[0]["clock"] is shared_clock
        assert arm._clock is shared_clock  # 测试观测驱动内部时基（无公开访问器）
        # spy 返回的 fake 即驱动持有的传输
        assert arm._transport is fake  # 测试观测（无公开访问器）

    def test_command_timeout_bound_delegated_to_driver(
        self, tmp_path: Path, left_profile: Any
    ) -> None:
        """command_timeout_s ≥ 2×control_period 校验由 4.6 驱动承担（透传）。

        工厂不重复实现该下界（裁定 3）：违规值经透传后在驱动构造器被拒。
        """
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        with pytest.raises(ValueError, match="2 倍控制周期"):
            build_r1d_arm(
                left_profile,
                config,
                acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
                transport=FakeR1DTransport(),
                clock=FakeClock(),
                command_timeout_s=0.001,  # < 2×0.002（测试专用非法值）
            )

    def test_valid_command_timeout_accepted(
        self, tmp_path: Path, left_profile: Any
    ) -> None:
        """合法 command_timeout_s（≥ 2×周期）透传成功（对照负例）。"""
        config = load_r1d_hardware_config(_write_motion_hw_config(tmp_path))
        arm = build_r1d_arm(
            left_profile,
            config,
            acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
            transport=FakeR1DTransport(),
            clock=FakeClock(),
            command_timeout_s=_COMMAND_TIMEOUT,
        )
        assert arm.get_status().phase == "CLOSED"


# ---------------------------------------------------------------------------
# SDK import 边界（子进程锁死）
# ---------------------------------------------------------------------------


class TestSdkImportBoundary:
    """注入 fake transport 时工厂构造路径零 SDK import、不触 FakeRobot。"""

    def test_fake_transport_path_never_imports_sdk(self) -> None:
        """子进程验证：注入 fake 构造全程不加载 unitree_sdk2py/cyclonedds。

        同时断言 R1D 工厂路径不 import src.real.fake_robot（裁定 7：
        FakeRobot 六维行为不用于 R1D）。用独立子进程避免同进程其他测试
        模块先行 import 造成假阳性/假阴性（同 4.5 的隔离方式）。
        """
        code = (
            "import sys\n"
            "from pathlib import Path\n"
            "from src.real.r1d_config import load_r1d_hardware_config\n"
            "from src.real.r1d_factory import build_r1d_arm\n"
            "from src.robot.robot_profile import load_robot_profile\n"
            "\n"
            "class _RefusingTransport:\n"
            "    def open(self, on_state):\n"
            "        raise AssertionError('构造期不得触碰传输')\n"
            "    def acquire_control(self):\n"
            "        raise AssertionError('构造期不得触碰传输')\n"
            "    def publish(self, command):\n"
            "        raise AssertionError('构造期不得触碰传输')\n"
            "    def release_control(self):\n"
            "        raise AssertionError('构造期不得触碰传输')\n"
            "    def close(self):\n"
            "        raise AssertionError('构造期不得触碰传输')\n"
            "\n"
            "profile = load_robot_profile(Path('configs/robots/r1d_left.yaml'))\n"
            "config = load_r1d_hardware_config(\n"
            "    Path('configs/robots/r1d_hardware_left.yaml'))\n"
            "def fake_clock():\n"
            "    return 1000.0\n"
            "arm = build_r1d_arm(profile, config,\n"
            "                     acquire_timeout_s=3.5,\n"
            "                     transport=_RefusingTransport(),\n"
            "                     clock=fake_clock)\n"
            "assert arm.get_status().phase == 'CLOSED'\n"
            "leaked_sdk = [m for m in sys.modules if m == 'unitree_sdk2py'\n"
            "              or m.startswith('unitree_sdk2py.')\n"
            "              or m == 'cyclonedds'\n"
            "              or m.startswith('cyclonedds.')]\n"
            "assert not leaked_sdk, leaked_sdk\n"
            "assert 'src.real.fake_robot' not in sys.modules\n"
            "print('FACTORY_PURE_OK')\n"
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
        assert "FACTORY_PURE_OK" in result.stdout

    def test_real_template_and_profile_pass_offline_gate(
        self, tmp_path: Path
    ) -> None:
        """仓库真实模板（offline 档）+ 真实 profile 可注入 fake 构造。

        4.2 冻结模板（network_interface=null/layout_verified=false）通过
        工厂构造期 offline 校验——read_only/motion 门槛归 connect/activate。
        """
        assert _LEFT_HW_TEMPLATE_PATH.is_file()
        config = load_r1d_hardware_config(_LEFT_HW_TEMPLATE_PATH)
        profile = load_robot_profile(_LEFT_PROFILE_PATH)
        arm = build_r1d_arm(
            profile,
            config,
            acquire_timeout_s=_TEST_ACQUIRE_TIMEOUT_S,
            transport=FakeR1DTransport(),
            clock=FakeClock(),
        )
        assert isinstance(arm, UnitreeR1DInterface)
        assert arm.get_status().phase == "CLOSED"


# ---------------------------------------------------------------------------
# RM65 兼容边界（Ruling 7：RM65 默认语义不动）
# ---------------------------------------------------------------------------


class TestRm65Boundary:
    """RM65 build_robot_arm 不接受 R1D；R1D 工厂不导入 FakeRobot。"""

    def test_rm65_build_robot_arm_rejects_r1d_profile(
        self, left_profile: Any
    ) -> None:
        """R1D profile 传入 RM65 工厂 → ValueError（仅支持 RM65）。

        守护 RM65 工厂默认语义不被本切片放宽：R1D 只走独立工厂。
        """
        config = RealRobotConfig(robot_ip="fake-ip")
        with pytest.raises(ValueError, match="RM65"):
            build_robot_arm(
                config,
                profile=left_profile,
                mock=True,
                init_q=np.zeros(left_profile.ndof),
            )

    def test_r1d_factory_module_import_is_pure(self) -> None:
        """r1d_factory 模块 import 无 SDK/FakeRobot 依赖（进程内补测）。"""
        import src.real.r1d_factory as factory_mod

        # 模块顶层只 import 类型与工厂依赖（fake_robot 不在其依赖闭包）
        assert not hasattr(factory_mod, "FakeRobot")
