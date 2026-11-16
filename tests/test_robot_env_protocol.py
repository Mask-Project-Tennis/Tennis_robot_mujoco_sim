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


def test_robot_env_protocol_has_control_range():
    """RobotEnv Protocol 声明 control_range 属性（Task 3.3 新增）。"""
    assert "control_range" in RobotEnv.__annotations__


def test_robot_env_protocol_left_arm_fields_are_legacy():
    """LEFT_ARM_NQ/init_q_left 已移出协议（RM65 旧实现专属，非通用必需字段）。

    Task 3.3 把协议从 RM65 专属假设中解放：通用规划消费者不再要求
    左臂字段；RM65Env/PlanningEnv 继续提供它们供旧 C++/回放路径消费。
    """
    assert "LEFT_ARM_NQ" not in RobotEnv.__annotations__
    assert "init_q_left" not in RobotEnv.__annotations__
