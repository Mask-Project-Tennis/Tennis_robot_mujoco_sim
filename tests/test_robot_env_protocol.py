"""RobotEnv Protocol 测试。

验证 RM65Env 满足 RobotEnv Protocol（结构化子类型）。
"""

from pathlib import Path

from src.ilqt.robot_env_protocol import RobotEnv
from src.sim.rm65_env import RM65Env


def test_rm65env_satisfies_protocol():
    """RM65Env 满足 RobotEnv Protocol（isinstance 成立）。"""
    env = RM65Env(Path("src/robot/rm65_model.xml"))
    assert isinstance(env, RobotEnv)
