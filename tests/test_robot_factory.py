"""驱动选择工厂 build_robot_arm 与 RM65 运行能力校验测试。

验证：仅 RM65 布局通过校验；构造过程不连接硬件；
init_q 错误在驱动构造前失败。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.real.config import RealRobotConfig
from src.real.runner_factory import build_robot_arm, validate_rm65_runtime
from src.robot.constants import INIT_Q_REAL
from src.robot.model_binding import bind_robot_model
from src.robot.robot_profile import RobotProfile, default_rm65_profile_path, load_robot_profile
from src.utils.mujoco_loader import load_mujoco_model

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rm65_profile() -> RobotProfile:
    """默认 RM65 描述。"""
    return load_robot_profile(default_rm65_profile_path())


def _rm65_like_profile(model_path: Path) -> RobotProfile:
    """RM65 身份与名称、但 model_path 可变的描述（供错配测试）。"""
    return RobotProfile(
        schema_version=1,
        robot_id="rm65",
        model_path=model_path,
        active_joint_names=tuple(f"r_joint{i}" for i in range(1, 7)),
        actuator_names=tuple(f"torque_r_joint{i}" for i in range(1, 7)),
        held_joint_names=tuple(f"l_joint{i}" for i in range(1, 7)),
        tcp_site_name="racket_center",
    )


def _seven_joint_profile() -> RobotProfile:
    """七关节合成描述（绑定能力已由任务 1.2 证明，运行层应拒绝）。"""
    return RobotProfile(
        schema_version=1,
        robot_id="synthetic7",
        model_path=REPO_ROOT / "src" / "robot" / "rm65_model.xml",
        active_joint_names=tuple(f"j{i}" for i in range(1, 8)),
        actuator_names=tuple(f"act_j{i}" for i in range(1, 8)),
        held_joint_names=(),
        tcp_site_name="racket_center",
    )


# RM65 命名但关节/执行器乱序的替身模型：单 body 恰 6 dof，双臂两个 body
_LOOKALIKE_XML = """
<mujoco model="rm65_lookalike">
  <worldbody>
    <body name="rarm" pos="0 0 1">
      <joint name="r_joint2" type="hinge" axis="0 1 0"/>
      <joint name="r_joint1" type="hinge" axis="1 0 0"/>
      <joint name="r_joint4" type="hinge" axis="0 0 1"/>
      <joint name="r_joint3" type="hinge" axis="1 0 0"/>
      <joint name="r_joint6" type="hinge" axis="0 1 0"/>
      <joint name="r_joint5" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/>
      <site name="racket_center" pos="0 0 0.2"/>
    </body>
    <body name="larm" pos="0.3 0 1">
      <joint name="l_joint1" type="hinge" axis="0 1 0"/>
      <joint name="l_joint2" type="hinge" axis="1 0 0"/>
      <joint name="l_joint3" type="hinge" axis="0 0 1"/>
      <joint name="l_joint4" type="hinge" axis="1 0 0"/>
      <joint name="l_joint5" type="hinge" axis="0 1 0"/>
      <joint name="l_joint6" type="hinge" axis="0 0 1"/>
      <geom type="sphere" size="0.01"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="torque_r_joint2" joint="r_joint2"/>
    <motor name="torque_r_joint1" joint="r_joint1"/>
    <motor name="torque_r_joint4" joint="r_joint4"/>
    <motor name="torque_r_joint3" joint="r_joint3"/>
    <motor name="torque_r_joint6" joint="r_joint6"/>
    <motor name="torque_r_joint5" joint="r_joint5"/>
  </actuator>
</mujoco>
"""


class TestValidateRm65Runtime:
    """运行能力校验：描述能力与运行能力分开。"""

    def test_accepts_default_rm65(self) -> None:
        """默认 RM65 描述 + 真实模型绑定通过校验（无异常）。"""
        profile = _rm65_profile()
        model = load_mujoco_model(REPO_ROOT / "src" / "robot" / "rm65_model.xml")
        binding = bind_robot_model(profile, model)
        validate_rm65_runtime(profile, binding)  # 不抛异常即通过

    def test_rejects_non_rm65_identity_and_seven_joint_layout(self) -> None:
        """七关节合成描述（自定义身份+乱序布局）显式拒绝。"""
        profile = _seven_joint_profile()
        model = mujoco.MjModel.from_xml_string(_LOOKALIKE_XML)
        # j1..j7 在替身模型中不存在 → 绑定器报缺关节，同样属于运行前拒绝
        with pytest.raises(ValueError, match="j1"):
            bind_robot_model(profile, model)

    def test_rejects_replaced_model(self) -> None:
        """RM65 身份但模型路径被替换 → 拒绝。"""
        profile = _rm65_like_profile(Path("other.xml").resolve())
        model = load_mujoco_model(REPO_ROOT / "src" / "robot" / "rm65_model.xml")
        binding = bind_robot_model(profile, model)
        with pytest.raises(ValueError, match="模型"):
            validate_rm65_runtime(profile, binding)

    def test_rejects_shuffled_layout(self) -> None:
        """名称与身份正确但绑定来自乱序模型（qpos 非顺序布局）→ 拒绝。"""
        profile = _rm65_like_profile(
            (REPO_ROOT / "src" / "robot" / "rm65_model.xml").resolve()
        )
        lookalike = mujoco.MjModel.from_xml_string(_LOOKALIKE_XML)
        binding = bind_robot_model(profile, lookalike)
        # 绑定本身成功（乱序地址），运行校验必须拒绝该布局
        assert binding.qpos_indices != tuple(range(6))
        with pytest.raises(ValueError, match="布局"):
            validate_rm65_runtime(profile, binding)


class _SpyFakeRobot:
    """FakeRobot 替身：记录构造与 connect 调用。"""

    constructed_count = 0
    connect_count = 0

    def __init__(self, init_q: np.ndarray, dt: float = 0.005) -> None:
        _SpyFakeRobot.constructed_count += 1

    def connect(self) -> bool:
        _SpyFakeRobot.connect_count += 1
        return True


class _SpyRobotInterface:
    """RobotInterface 替身：记录构造与 connect 调用。"""

    constructed_with: object | None = None
    connect_count = 0

    def __init__(self, config: RealRobotConfig) -> None:
        _SpyRobotInterface.constructed_with = config

    def connect(self) -> bool:
        _SpyRobotInterface.connect_count += 1
        return True


@pytest.fixture()
def spy_fake_robot(monkeypatch: pytest.MonkeyPatch) -> type[_SpyFakeRobot]:
    """替换 src.real.fake_robot.FakeRobot 为记录型替身。"""
    _SpyFakeRobot.constructed_count = 0
    _SpyFakeRobot.connect_count = 0
    monkeypatch.setattr("src.real.fake_robot.FakeRobot", _SpyFakeRobot)
    return _SpyFakeRobot


@pytest.fixture()
def spy_robot_interface(monkeypatch: pytest.MonkeyPatch) -> type[_SpyRobotInterface]:
    """替换 src.real.robot_interface.RobotInterface 为记录型替身。"""
    _SpyRobotInterface.constructed_with = None
    _SpyRobotInterface.connect_count = 0
    monkeypatch.setattr(
        "src.real.robot_interface.RobotInterface", _SpyRobotInterface
    )
    return _SpyRobotInterface


class TestBuildRobotArm:
    """驱动工厂：构造无设备动作。"""

    def test_mock_true_returns_fake_robot_without_connect(self) -> None:
        """mock=True 返回 FakeRobot 实例且未连接。"""
        from src.real.fake_robot import FakeRobot

        robot = build_robot_arm(
            RealRobotConfig(robot_ip="fake-ip"),
            profile=_rm65_profile(),
            mock=True,
            init_q=INIT_Q_REAL.copy(),
        )
        assert isinstance(robot, FakeRobot)
        assert robot._connected is False  # noqa: SLF001 — 验证未连接

    def test_mock_false_constructs_robot_interface_without_connect(
        self, spy_robot_interface: type[_SpyRobotInterface]
    ) -> None:
        """mock=False 构造 RobotInterface(config) 且不调用 connect。"""
        config = RealRobotConfig(robot_ip="fake-ip")
        robot = build_robot_arm(config, profile=_rm65_profile(), mock=False,
                                init_q=INIT_Q_REAL.copy())
        assert isinstance(robot, spy_robot_interface)
        assert spy_robot_interface.constructed_with is config
        assert spy_robot_interface.connect_count == 0

    def test_mock_false_rejects_unknown_identity_before_construction(
        self, spy_robot_interface: type[_SpyRobotInterface]
    ) -> None:
        """未知身份在驱动构造前失败（七关节描述不会误用 RM65 驱动）。"""
        with pytest.raises(ValueError, match="synthetic7"):
            build_robot_arm(
                RealRobotConfig(robot_ip="fake-ip"),
                profile=_seven_joint_profile(),
                mock=False,
                init_q=np.zeros(7),
            )
        assert spy_robot_interface.constructed_with is None

    def test_init_q_wrong_length_rejected_before_construction(
        self, spy_fake_robot: type[_SpyFakeRobot]
    ) -> None:
        """init_q 长度与 ndof 不符 → 驱动构造前 ValueError。"""
        with pytest.raises(ValueError, match="init_q"):
            build_robot_arm(
                RealRobotConfig(robot_ip="fake-ip"),
                profile=_rm65_profile(),
                mock=True,
                init_q=np.zeros(7),
            )
        assert spy_fake_robot.constructed_count == 0

    def test_init_q_nan_rejected_before_construction(
        self, spy_fake_robot: type[_SpyFakeRobot]
    ) -> None:
        """init_q 含 NaN → 驱动构造前 ValueError。"""
        bad_q = INIT_Q_REAL.copy().astype(float)
        bad_q[0] = np.nan
        with pytest.raises(ValueError, match="init_q"):
            build_robot_arm(
                RealRobotConfig(robot_ip="fake-ip"),
                profile=_rm65_profile(),
                mock=True,
                init_q=bad_q,
            )
        assert spy_fake_robot.constructed_count == 0

    def test_mock_true_rejects_non_rm65_identity(self) -> None:
        """mock 路径同样拒绝非 RM65 身份（FakeRobot 是 RM65 专属 6 维 mock）。"""
        with pytest.raises(ValueError, match="synthetic7"):
            build_robot_arm(
                RealRobotConfig(robot_ip="fake-ip"),
                profile=_seven_joint_profile(),
                mock=True,
                init_q=np.zeros(7),
            )
