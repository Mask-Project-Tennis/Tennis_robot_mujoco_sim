"""组件 Protocol 接口验证测试。"""

import numpy as np
from src.ilqt.components.protocols import (
    PerceptionComponent,
    ExecutorComponent,
    SafetyComponent,
)


class FakePerception:
    def get_ball_state(self):
        return np.zeros(3), np.zeros(3)


class FakeExecutor:
    def get_arm_state(self):
        return np.zeros(12)
    def execute(self, u_cmd):
        pass
    def get_metrics(self):
        return {}


class FakeSafety:
    def filter(self, u_cmd, arm_state):
        return u_cmd, True


def test_perception_protocol():
    assert isinstance(FakePerception(), PerceptionComponent)

def test_executor_protocol():
    assert isinstance(FakeExecutor(), ExecutorComponent)

def test_safety_protocol():
    assert isinstance(FakeSafety(), SafetyComponent)
