"""BallPredictor 解析球轨迹预测器测试。"""

import numpy as np

from src.ilqt.ball_predictor import BallPredictor


class TestBallPredictorState:
    """测试 BallPredictor 状态管理。"""

    def test_set_get_state_roundtrip(self) -> None:
        """set_state 设置后 get_state 应返回相同的位置和速度。"""
        predictor = BallPredictor(dt=0.005)
        pos = np.array([1.0, 2.0, 3.0])
        vel = np.array([0.5, -0.3, 1.2])

        predictor.set_state(pos, vel)
        got_pos, got_vel = predictor.get_state()

        np.testing.assert_allclose(got_pos, pos)
        np.testing.assert_allclose(got_vel, vel)


class TestBallPredictorParabola:
    """测试 BallPredictor 抛物线运动（无弹跳）。"""

    def test_predict_parabolic_free_fall(self) -> None:
        """前 3 步位置应吻合解析抛物线公式 p0 + v0*t + 0.5*g*t^2。

        给定 p0=[0,0,10], v0=[1,0,0], dt=0.01, g=9.81（向下）。
        """
        dt = 0.01
        g = 9.81
        p0 = np.array([0.0, 0.0, 10.0])
        v0 = np.array([1.0, 0.0, 0.0])

        predictor = BallPredictor(dt=dt, g=g)
        predictor.set_state(p0, v0)
        positions, _ = predictor.predict(n_steps=3)

        # 步 k 的期望位置：t = k*dt
        for k in range(1, 4):
            t = k * dt
            expected = p0 + v0 * t + 0.5 * np.array([0.0, 0.0, -g]) * (t ** 2)
            # positions[k-1] 是第 k 步的位置
            np.testing.assert_allclose(positions[k - 1], expected, atol=1e-6)


class TestBallPredictorBounce:
    """测试 BallPredictor 地面弹跳。"""

    def test_predict_bounce(self) -> None:
        """球下落穿越 BALL_RADIUS 时应反弹，且 vz_new ≈ -vz * BOUNCE_RESTITUTION。

        构造必然弹跳场景：p0=[0,0,0.04], v0=[0,0,-1.0], dt=0.005。
        预测足够多步，在弹跳步处验证 vz 由负转正且满足恢复系数关系。
        """
        dt = 0.005
        g = 9.81
        p0 = np.array([0.0, 0.0, 0.04])
        v0 = np.array([0.0, 0.0, -1.0])

        predictor = BallPredictor(dt=dt, g=g)
        predictor.set_state(p0, v0)
        positions, velocities = predictor.predict(n_steps=20)

        # 找到首个弹跳步：vz 由负转正
        vz_seq = velocities[:, 2]
        bounce_idx = None
        for k in range(1, len(vz_seq)):
            if vz_seq[k - 1] < 0 < vz_seq[k]:
                bounce_idx = k
                break
        assert bounce_idx is not None, "未检测到弹跳"

        pre_vz = vz_seq[bounce_idx - 1]   # 弹跳前一步存储速度（负）
        post_vz = vz_seq[bounce_idx]      # 弹跳后存储速度（正）

        # 反弹后 vz 为正
        assert post_vz > 0

        # 弹跳瞬间的下落速度 = 上一步存储速度 + 本步重力更新（更负）
        v_at_bounce = pre_vz - g * dt
        expected_post_vz = -v_at_bounce * predictor.BOUNCE_RESTITUTION
        np.testing.assert_allclose(post_vz, expected_post_vz, atol=1e-9)

        # 弹跳步位置应被钳位到 BALL_RADIUS
        np.testing.assert_allclose(
            positions[bounce_idx, 2], predictor.BALL_RADIUS, atol=1e-9
        )


class TestBallPredictorPredictFrom:
    """测试 BallPredictor.predict_from（无状态接口）。"""

    def test_predict_from_stateless(self) -> None:
        """predict_from(p0, v0, n) 与 set_state(p0,v0); predict(n) 结果完全相同。

        且 predict_from 不依赖先前的 set_state（纯函数语义）。
        """
        dt = 0.005
        g = 9.81
        p0 = np.array([-0.5, 0.2, 1.2])
        v0 = np.array([3.0, -0.5, -2.0])
        n_steps = 50

        # 路径 A：predict_from 直接调用（不 set_state）
        predictor_a = BallPredictor(dt=dt, g=g)
        pos_a, vel_a = predictor_a.predict_from(p0, v0, n_steps)

        # 路径 B：set_state + predict
        predictor_b = BallPredictor(dt=dt, g=g)
        predictor_b.set_state(p0, v0)
        pos_b, vel_b = predictor_b.predict(n_steps)

        np.testing.assert_allclose(pos_a, pos_b, atol=1e-12)
        np.testing.assert_allclose(vel_a, vel_b, atol=1e-12)



