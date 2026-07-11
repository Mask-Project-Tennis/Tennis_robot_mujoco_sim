"""共享常量模块测试 — 验证 constants.py 中的物理常量定义。"""

from src.robot.constants import BOUNCE_RESTITUTION


def test_bounce_restitution_in_constants() -> None:
    """BOUNCE_RESTITUTION 定义在 constants.py 中，值为 0.75。"""
    assert BOUNCE_RESTITUTION == 0.75


def test_ball_predictor_uses_shared_constant() -> None:
    """BallPredictor.BOUNCE_RESTITUTION 来自 constants.py。"""
    from src.ilqt.ball_predictor import BallPredictor
    from src.robot.constants import BOUNCE_RESTITUTION as CONST

    assert BallPredictor.BOUNCE_RESTITUTION == CONST
