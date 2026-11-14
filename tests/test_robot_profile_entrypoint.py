"""真机入口 run_real_robot 的 --robot-config 接入测试。

验证：缺省与显式 rm65.yaml 走同一组装流程且行为一致；
坏 profile 在构造驱动/启动传感器前报错；
旧 YAML 覆盖行为（kp/dt/限位）保持不变。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 沿用 tests/test_create_runner.py 的导入方式（scripts/ 加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import numpy as np
import pytest

import run_real_robot
from run_real_robot import create_runner
from src.real.config import RealRobotConfig
from src.robot.constants import INIT_Q_REAL
from src.robot.robot_profile import (
    RobotProfile,
    default_rm65_profile_path,
    load_robot_profile,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _rm65_like_profile(robot_id: str) -> RobotProfile:
    """RM65 名称但身份可变的描述（供不支持机器人测试）。"""
    return RobotProfile(
        schema_version=1,
        robot_id=robot_id,
        model_path=(REPO_ROOT / "src" / "robot" / "rm65_model.xml").resolve(),
        active_joint_names=tuple(f"r_joint{i}" for i in range(1, 7)),
        actuator_names=tuple(f"torque_r_joint{i}" for i in range(1, 7)),
        held_joint_names=tuple(f"l_joint{i}" for i in range(1, 7)),
        tcp_site_name="racket_center",
    )


class TestDefaultVsExplicitProfile:
    """缺省与显式 profile 等价。"""

    def test_default_and_explicit_rm65_equivalent(self) -> None:
        """缺省与显式 rm65.yaml 构造出相同的 FakeRobot 与规划配置。"""
        config = RealRobotConfig(robot_ip="fake-ip")
        runner_default = create_runner(config, mock=True)
        runner_explicit = create_runner(
            config,
            mock=True,
            profile=load_robot_profile(default_rm65_profile_path()),
        )

        # FakeRobot 初始角一致，且为真机 INIT_Q_REAL（第二关节 1.40，非仿真 1.57）
        assert np.array_equal(runner_default._robot._q, runner_explicit._robot._q)
        assert np.array_equal(runner_default._robot._q, INIT_Q_REAL)

        # 规划配置一致
        assert runner_default._config.dt == runner_explicit._config.dt

        # 环境执行器配置一致
        assert runner_default._env.actuator_mode == runner_explicit._env.actuator_mode

        # 异步规划器引用同一已校验模型路径
        assert (
            runner_default._replanner._model_path
            == runner_explicit._replanner._model_path
            == (REPO_ROOT / "src" / "robot" / "rm65_model.xml").resolve()
        )


class TestBadProfileRejected:
    """坏 profile 在任何驱动构造/传感器启动前报错。"""

    def test_unsupported_profile_fails_before_driver_and_sensor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """非 RM65 身份：构造 FakeRobot / 启动传感器之前抛 ValueError。"""
        fake_calls = {"constructed": 0, "sensor_started": 0}

        import src.real.fake_robot as fake_robot_mod
        import src.real.ball_sensor as ball_sensor_mod

        class _SpyFake:
            def __init__(self, *args: object, **kwargs: object) -> None:
                fake_calls["constructed"] += 1

        real_start = ball_sensor_mod.SimulatedBallSensor.start

        def _spy_start(self: object) -> None:
            fake_calls["sensor_started"] += 1

        monkeypatch.setattr(fake_robot_mod, "FakeRobot", _SpyFake)
        monkeypatch.setattr(
            ball_sensor_mod.SimulatedBallSensor, "start", _spy_start
        )

        with pytest.raises(ValueError, match="testbot"):
            create_runner(
                RealRobotConfig(robot_ip="fake-ip"),
                mock=True,
                profile=_rm65_like_profile("testbot"),
            )
        assert fake_calls["constructed"] == 0
        assert fake_calls["sensor_started"] == 0
        assert real_start is not None  # noqa: F841 — 保留原方法引用防误删


class TestYamlOverridePreserved:
    """旧 YAML 覆盖行为保持不变（新配置不遮蔽旧值）。"""

    def test_custom_kp_dt_limits_flow_through(self) -> None:
        """自定义 kp/dt/TCP 限速从 RealRobotConfig 流入 env/驱动/限位。"""
        config = RealRobotConfig(
            robot_ip="fake-ip",
            dt=0.002,
            kp=[300.0] * 6,
            kd=[30.0] * 6,
            max_tcp_speed=0.5,
        )
        runner = create_runner(
            config, mock=True, profile=load_robot_profile(default_rm65_profile_path())
        )

        # dt 流入环境与 FakeRobot
        assert runner._env.dt == pytest.approx(0.002)
        assert runner._robot._dt == pytest.approx(0.002)
        # kp 流入位置模式执行器
        assert runner._env.kp is not None
        assert np.allclose(runner._env.kp, [300.0] * 6)
        # TCP 限速流入 robot_limits
        assert runner._robot_limits.max_tcp_speed == pytest.approx(0.5)


class TestCliArg:
    """CLI 参数解析。"""

    def test_parse_args_accepts_robot_config(self) -> None:
        """--robot-config 可解析，缺省为 None（走默认 RM65 描述）。"""
        args = run_real_robot._parse_args(["--mock", "--max-steps", "3"])
        assert args.robot_config is None

        args_explicit = run_real_robot._parse_args(
            ["--mock", "--robot-config", str(default_rm65_profile_path())]
        )
        assert args_explicit.robot_config == str(default_rm65_profile_path())
