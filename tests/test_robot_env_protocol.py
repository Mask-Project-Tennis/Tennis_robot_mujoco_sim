"""RobotEnv Protocol 测试。

验证 RM65Env 满足 RobotEnv Protocol（结构化子类型），
以及 Protocol 声明了核心模块实际访问的属性。
"""

from pathlib import Path

from src.ilqt.robot_env_protocol import RobotEnv
from src.sim.rm65_env import RM65Env


def test_rm65env_satisfies_protocol():
    """RM65Env 满足 RobotEnv Protocol（isinstance 成立）。"""
    env = RM65Env(Path("src/robot/rm65_model.xml"))
    assert isinstance(env, RobotEnv)


def test_robot_env_protocol_has_model():
    """RobotEnv Protocol 声明 model 属性。"""
    assert "model" in RobotEnv.__annotations__


def test_robot_env_protocol_has_data():
    """RobotEnv Protocol 声明 data 属性。"""
    assert "data" in RobotEnv.__annotations__


def test_robot_env_protocol_has_init_q_left():
    """RobotEnv Protocol 声明 init_q_left 属性。"""
    assert "init_q_left" in RobotEnv.__annotations__


def test_robot_env_protocol_has_left_arm_nq():
    """RobotEnv Protocol 声明 LEFT_ARM_NQ 属性。"""
    assert "LEFT_ARM_NQ" in RobotEnv.__annotations__
