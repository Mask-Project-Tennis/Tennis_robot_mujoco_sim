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
    """坏 profile 在任何驱动构造/传感器启动/后台线程前报错。"""

    def test_unsupported_profile_fails_before_driver_sensor_and_thread(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """非 RM65 身份：构造 FakeRobot / 启动传感器 / 启动重规划线程之前抛 ValueError。"""
        fake_calls = {"constructed": 0, "sensor_started": 0, "thread_started": 0}

        import src.ilqt.async_replanner as replanner_mod
        import src.real.ball_sensor as ball_sensor_mod
        import src.real.fake_robot as fake_robot_mod

        class _SpyFake:
            def __init__(self, *args: object, **kwargs: object) -> None:
                fake_calls["constructed"] += 1

        def _spy_sensor_start(self: object) -> None:
            fake_calls["sensor_started"] += 1

        def _spy_thread_start(self: object, *args: object, **kwargs: object) -> None:
            fake_calls["thread_started"] += 1

        monkeypatch.setattr(fake_robot_mod, "FakeRobot", _SpyFake)
        monkeypatch.setattr(
            ball_sensor_mod.SimulatedBallSensor, "start", _spy_sensor_start
        )
        monkeypatch.setattr(replanner_mod.AsyncReplanner, "start", _spy_thread_start)

        with pytest.raises(ValueError, match="testbot"):
            create_runner(
                RealRobotConfig(robot_ip="fake-ip"),
                mock=True,
                profile=_rm65_like_profile("testbot"),
            )
        assert fake_calls["constructed"] == 0
        assert fake_calls["sensor_started"] == 0
        assert fake_calls["thread_started"] == 0


class TestR1dProfileRejected:
    """阶段 2 交付的真实 R1D profile 仍被真机入口拒绝（阶段 3 前不支持）。

    与 TestBadProfileRejected（虚构 robot_id）、TestMainCli（手写 YAML）不同，
    本组测试使用仓库中真实存在的 configs/robots/r1d_{left,right}.yaml 走
    main() 完整解析路径：证明的是"真实交付 profile 无法经现有入口启动
    运行组件"，而非仅校验逻辑分支。
    """

    @pytest.mark.parametrize("profile_name", ["r1d_left", "r1d_right"])
    def test_r1d_profile_still_rejected_by_real_entrypoint(
        self, monkeypatch: pytest.MonkeyPatch, profile_name: str
    ) -> None:
        """真实 r1d profile 调 main() → 退出码 2，运行组件调用次数为零。

        断言三重隔离：create_runner（组装层）、RobotInterface.connect
        （SDK 连接层）、AsyncReplanner.start（后台线程层）均未被调用。
        """
        import src.ilqt.async_replanner as replanner_mod
        import src.real.robot_interface as robot_interface_mod

        calls = {"create_runner": 0, "sdk_connect": 0, "replanner_start": 0}

        def _fake_create_runner(*args: object, **kwargs: object) -> None:
            calls["create_runner"] += 1

        def _spy_sdk_connect(self: object, *args: object, **kwargs: object) -> None:
            calls["sdk_connect"] += 1

        def _spy_replanner_start(self: object, *args: object, **kwargs: object) -> None:
            calls["replanner_start"] += 1

        monkeypatch.setattr(run_real_robot, "create_runner", _fake_create_runner)
        monkeypatch.setattr(
            robot_interface_mod.RobotInterface, "connect", _spy_sdk_connect
        )
        monkeypatch.setattr(
            replanner_mod.AsyncReplanner, "start", _spy_replanner_start
        )

        # 真实仓库 profile（非临时构造）：路径存在性是测试前提，不是被测行为
        profile_path = REPO_ROOT / "configs" / "robots" / f"{profile_name}.yaml"
        assert profile_path.is_file(), f"缺少真实 profile: {profile_path}"

        monkeypatch.setattr(
            sys,
            "argv",
            ["run_real_robot.py", "--mock", "--robot-config", str(profile_path)],
        )
        with pytest.raises(SystemExit) as exc_info:
            run_real_robot.main()

        # 配置类错误以退出码 2 呈现（与阶段 1 CLI 契约一致）
        assert int(exc_info.value.code) == 2
        # 任何运行组件（组装/SDK 连接/后台线程）都未被触碰
        assert calls["create_runner"] == 0
        assert calls["sdk_connect"] == 0
        assert calls["replanner_start"] == 0


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


class TestMainCli:
    """main() 的 --robot-config 加载路径：配置错误以退出码 2 呈现。"""

    def _run_main_with_fake_runner(
        self,
        monkeypatch: pytest.MonkeyPatch,
        argv: list[str],
    ) -> tuple[int, dict[str, int]]:
        """以替身 create_runner 运行 main，返回 (SystemExit 退出码, 调用计数)。"""
        calls = {"create_runner": 0}

        def _fake_create_runner(*args: object, **kwargs: object) -> None:
            calls["create_runner"] += 1

        monkeypatch.setattr(run_real_robot, "create_runner", _fake_create_runner)
        monkeypatch.setattr(sys, "argv", ["run_real_robot.py", *argv])
        with pytest.raises(SystemExit) as exc_info:
            run_real_robot.main()
        return int(exc_info.value.code), calls

    def test_missing_robot_config_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """--robot-config 指向不存在的文件 → 退出码 2，create_runner 未被调用。"""
        code, calls = self._run_main_with_fake_runner(
            monkeypatch,
            ["--mock", "--robot-config", str(tmp_path / "no_such.yaml")],
        )
        assert code == 2
        assert calls["create_runner"] == 0

    def test_unsupported_robot_config_exits_2(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """合法 YAML 但 robot_id 非 rm65 → 退出码 2（描述校验失败）。"""
        import yaml

        data = {
            "schema_version": 1,
            "robot_id": "d2",
            "model_path": "model.xml",
            "active_joint_names": ["j1"],
            "actuator_names": ["a1"],
            "held_joint_names": [],
            "tcp_site_name": "tcp",
        }
        # model_path 指向已存在文件（加载器存在性校验），身份校验在其后失败
        (tmp_path / "model.xml").touch()
        yaml_path = tmp_path / "d2.yaml"
        yaml_path.write_text(
            yaml.safe_dump(data, allow_unicode=True), encoding="utf-8"
        )

        code, calls = self._run_main_with_fake_runner(
            monkeypatch, ["--mock", "--robot-config", str(yaml_path)]
        )
        assert code == 2
        assert calls["create_runner"] == 0
