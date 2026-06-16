"""球感知模块测试。

测试 BallSensor ABC + SimulatedBallSensor + BallPerceiver。
"""

import numpy as np
import pytest

from src.real.ball_sensor import BallSensor, SimulatedBallSensor
from src.real.ball_perceiver import BallPerceiver


# ── S1-S3: BallSensor ──


class TestSimulatedBallSensor:
    """SimulatedBallSensor 测试。"""

    def test_push_and_get_latest(self):
        """S1: push 后 get_latest 返回正确位置。"""
        sensor = SimulatedBallSensor()
        sensor.start()

        pos = np.array([0.1, 0.2, 1.5])
        sensor.push(pos, timestamp=1.0)

        got_pos, got_ts = sensor.get_latest()
        np.testing.assert_allclose(got_pos, pos)
        assert got_ts == pytest.approx(1.0)

    def test_is_running_after_start_stop(self):
        """S2: is_running 在 start/stop 后正确切换。"""
        sensor = SimulatedBallSensor()
        assert not sensor.is_running
        sensor.start()
        assert sensor.is_running
        sensor.stop()
        assert not sensor.is_running

    def test_returns_none_when_no_data(self):
        """S3: 无数据或未启动时 get_latest 返回 (None, None)。"""
        sensor = SimulatedBallSensor()
        sensor.start()
        pos, ts = sensor.get_latest()
        assert pos is None
        assert ts is None


# ── S4-S7: BallPerceiver ──


class TestBallPerceiver:
    """BallPerceiver 测试。"""

    def test_returns_none_before_first_update(self):
        """S4: 首次 update 前返回 None。"""
        sensor = SimulatedBallSensor()
        sensor.start()
        perceiver = BallPerceiver(sensor, dt=0.005)
        assert perceiver.get_latest_filtered() is None

    def test_update_returns_filtered_pos_vel(self):
        """S5: update 后返回滤波后的 (pos, vel)。"""
        sensor = SimulatedBallSensor()
        sensor.start()
        perceiver = BallPerceiver(sensor, dt=0.05)

        # 第一次 push + update（初始化）
        sensor.push(np.array([0.0, 0.0, 1.0]), timestamp=0.0)
        result = perceiver.update()
        assert result is not None
        pos, _ = result
        np.testing.assert_allclose(pos, [0.0, 0.0, 1.0], atol=1e-6)

        # 第二次 push + update（有速度）
        sensor.push(np.array([0.01, 0.0, 1.0]), timestamp=0.05)
        result = perceiver.update()
        assert result is not None
        pos, vel = result
        np.testing.assert_allclose(pos, [0.01, 0.0, 1.0], atol=1e-6)
        assert vel[0] > 0.05  # 正 X 方向速度

    def test_reset_clears_estimator(self):
        """S6: reset 清空 KF 状态。"""
        sensor = SimulatedBallSensor()
        sensor.start()
        perceiver = BallPerceiver(sensor, dt=0.1)

        sensor.push(np.array([0.0, 0.0, 1.0]), timestamp=0.0)
        perceiver.update()
        sensor.push(np.array([0.1, 0.0, 1.0]), timestamp=0.1)
        perceiver.update()
        assert perceiver.get_latest_filtered() is not None

        perceiver.reset()
        assert perceiver.get_latest_filtered() is None

    def test_finite_difference_velocity(self):
        """S7: 有限差分速度计算正确（零噪声直通模式）。"""
        sensor = SimulatedBallSensor()
        sensor.start()
        perceiver = BallPerceiver(
            sensor, dt=0.1,
            estimator_config={"pos_noise_std": 0.0, "vel_noise_std": 0.0},
        )

        # 匀速运动: 0.1s 移动 0.1m → vel = [1.0, 0, 0]
        sensor.push(np.array([0.0, 0.0, 1.0]), timestamp=0.0)
        perceiver.update()

        sensor.push(np.array([0.1, 0.0, 1.0]), timestamp=0.1)
        perceiver.update()

        pos, vel = perceiver.get_latest_filtered()
        np.testing.assert_allclose(pos, [0.1, 0.0, 1.0], atol=1e-6)
        np.testing.assert_allclose(vel, [1.0, 0.0, 0.0], atol=1e-6)

    def test_stale_data_does_not_inject_zero_velocity(self):
        """S8: 传感器无新数据时重复 update 不注入零速度。"""
        sensor = SimulatedBallSensor()
        sensor.start()
        perceiver = BallPerceiver(
            sensor, dt=0.1,
            estimator_config={"pos_noise_std": 0.0, "vel_noise_std": 0.0},
        )

        # 两次 push 建立速度
        sensor.push(np.array([0.0, 0.0, 1.0]), timestamp=0.0)
        perceiver.update()
        sensor.push(np.array([0.1, 0.0, 1.0]), timestamp=0.1)
        result1 = perceiver.update()
        _, vel1 = result1
        np.testing.assert_allclose(vel1, [1.0, 0.0, 0.0], atol=1e-6)

        # 不 push 新数据，重复 update → 返回缓存，速度不变
        result2 = perceiver.update()
        assert result2 is not None
        _, vel2 = result2
        np.testing.assert_allclose(vel2, vel1, atol=1e-6)

    def test_update_returns_none_when_sensor_not_running(self):
        """S9: 传感器未启动时 update 返回 None。"""
        sensor = SimulatedBallSensor()
        # 故意不调用 start()
        perceiver = BallPerceiver(sensor, dt=0.005)
        assert perceiver.update() is None
