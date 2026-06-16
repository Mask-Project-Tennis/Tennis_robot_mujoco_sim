"""RealEnv 单元测试。

测试 RealEnv 实现 RobotEnv Protocol 的全部接口。
使用现有 rm65_model.xml（双臂+球），RealEnv 只操作右臂 6 DOF。
"""

import numpy as np
import pytest
from pathlib import Path

from src.real.real_env import RealEnv
from src.ilqt.robot_env_protocol import RobotEnv


MODEL_PATH = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"


@pytest.fixture
def env() -> RealEnv:
    """提供 RealEnv 实例。"""
    return RealEnv(MODEL_PATH, dt=0.005)


class TestRealEnvBasic:
    """RealEnv 基础接口测试。"""

    def test_model_loads_right_arm_6_dof(self, env: RealEnv):
        """S1: 模型加载成功，右臂 6 DOF。"""
        assert env.NQ == 6
        assert env.NX == 12
        assert env.NU == 6
        assert env.dt == pytest.approx(0.005)

    def test_satisfies_robot_env_protocol(self, env: RealEnv):
        """S2: isinstance(RealEnv, RobotEnv) 通过。"""
        assert isinstance(env, RobotEnv)

    def test_get_arm_state_zero_returns_12(self, env: RealEnv):
        """S3: get_arm_state() 零位返回 (12,)。"""
        env.reset(np.zeros(6))
        state = env.get_arm_state()
        assert state.shape == (12,)
        np.testing.assert_allclose(state, np.zeros(12))

    def test_get_ee_pos_reasonable(self, env: RealEnv):
        """S4: get_ee_pos() 返回合理位置（~1m 高度）。"""
        env.reset(np.zeros(6))
        pos = env.get_ee_pos()
        assert pos.shape == (3,)
        assert pos[2] > 0.5  # 球拍应在 0.5m 以上
        assert pos[2] < 2.0  # 但不超 2m

    def test_set_arm_state_changes_ee_pos(self, env: RealEnv):
        """S5: set_arm_state + get_ee_pos 随关节角变化。"""
        env.reset(np.zeros(6))
        pos_zero = env.get_ee_pos()

        # 转动关节 2（肩俯仰）0.5 rad，使手臂倾斜
        x = np.zeros(12)
        x[1] = 0.5
        env.set_arm_state(x)
        pos_rotated = env.get_ee_pos()

        # 位置应有明显变化
        diff = np.linalg.norm(pos_rotated - pos_zero)
        assert diff > 0.05  # 至少 5cm 变化


class TestRealEnvPhysics:
    """RealEnv 物理仿真测试。"""

    def test_step_from_state_torque_mode(self, env: RealEnv):
        """S6: step_from_state 力矩模式物理前进一步。"""
        env.reset(np.zeros(6))
        x0 = env.get_arm_state()

        # 施加关节 1 力矩
        u = np.zeros(6)
        u[0] = 10.0
        x1 = env.step_from_state(x0, u)

        # 状态应有变化（关节开始运动）
        assert not np.allclose(x1, x0)
        # 关节 1 应有正加速度（力矩为正）
        assert abs(x1[6]) > 1e-6  # qdot[0] 非零

    def test_get_ee_jacp_shape(self, env: RealEnv):
        """S7: get_ee_jacp 返回 (3, 6)。"""
        env.reset(np.zeros(6))
        jacp = env.get_ee_jacp()
        assert jacp.shape == (3, 6)

    def test_get_ee_normal_is_unit_vector(self, env: RealEnv):
        """S8: get_ee_normal 返回单位向量。"""
        env.reset(np.zeros(6))
        normal = env.get_ee_normal()
        assert normal.shape == (3,)
        assert np.linalg.norm(normal) == pytest.approx(1.0, abs=1e-6)


class TestRealEnvPositionMode:
    """RealEnv 位置模式测试。"""

    def test_configure_actuator_mode_position(self, env: RealEnv):
        """S9: configure_actuator_mode("position") 成功切换。"""
        kp = np.array([200, 200, 100, 50, 50, 20])
        kd = np.array([20, 20, 10, 5, 5, 2])
        env.configure_actuator_mode("position", kp, kd)

        assert env.actuator_mode == 1
        np.testing.assert_allclose(env.kp, kp)
        np.testing.assert_allclose(env.kd, kd)
        assert env.use_feedforward is True

    def test_position_mode_step_with_feedforward(self, env: RealEnv):
        """S10: 位置模式 step + feedforward 正常工作。"""
        kp = np.array([200, 200, 100, 50, 50, 20])
        kd = np.array([20, 20, 10, 5, 5, 2])
        env.configure_actuator_mode("position", kp, kd)
        env.configure_feedforward(True)

        env.reset(np.zeros(6))
        x0 = env.get_arm_state()

        # 发送非零目标角度
        u = np.zeros(6)
        u[0] = 0.1  # 期望关节 1 转 0.1 rad
        x1 = env.step(u)

        # 状态应有变化（PD 控制器驱动关节向目标移动）
        assert not np.allclose(x1, x0)
        # 关节 1 应朝目标方向移动
        assert x1[0] > x0[0]
