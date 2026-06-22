"""安全契约回归测试。

验证所有安全组件的关键行为，防止重构导致安全约束绕过。
"""
import numpy as np
import pytest
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "src" / "robot" / "rm65_model.xml"


@pytest.fixture
def make_env():
    """创建 RM65Env 实例。"""
    from src.sim.rm65_env import RM65Env
    return RM65Env(MODEL_PATH)


# ============================================================================
# X-wall 共享常量一致性
# ============================================================================


class TestXWallBodyNamesConsistency:
    """X-wall body-name 列表必须在整个代码库中一致。"""

    def test_predictive_safety_exports_public_constant(self):
        """PredictiveSafetyFilter 模块导出公共常量 X_WALL_BODY_NAMES。"""
        from src.ilqt.components.predictive_safety import X_WALL_BODY_NAMES
        assert isinstance(X_WALL_BODY_NAMES, tuple)
        assert len(X_WALL_BODY_NAMES) > 0
        assert all(isinstance(n, str) for n in X_WALL_BODY_NAMES)

    def test_v12_uses_shared_constant(self):
        """V12 的 X-wall body-name 解析应引用共享常量，而非硬编码重复。"""
        from src.ilqt.components.predictive_safety import X_WALL_BODY_NAMES
        # 读取 V12 源码，验证它 import 了共享常量
        v12_path = Path(__file__).resolve().parent.parent / "scripts" / "rm65_mpc_v12.py"
        v12_source = v12_path.read_text(encoding="utf-8")
        assert "X_WALL_BODY_NAMES" in v12_source, (
            "V12 应导入并使用 X_WALL_BODY_NAMES 共享常量"
        )


# ============================================================================
# PredictiveSafetyFilter 安全契约（特征化测试 — 回归守卫）
# ============================================================================


class TestPredictiveSafetyContract:
    """PredictiveSafetyFilter 必须拒绝已知不安全输入。"""

    def test_rejects_excessive_torque(self, make_env):
        """远超 ctrlrange 的力矩 → is_safe=False 或 u 被削减。"""
        env = make_env
        env.reset(np.zeros(6))
        from src.ilqt.robot_limits import RobotLimits
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
        limits = RobotLimits.from_config(
            {}, dt=0.005, ctrlrange=env.model.actuator_ctrlrange[:env.NU],
        )
        safety = PredictiveSafetyFilter(
            env, limits, is_position_mode=False,
        )
        u_huge = np.full(6, 100.0)  # 远超 ctrlange
        arm_state = env.get_arm_state()
        safe_u, is_safe = safety.filter(u_huge, arm_state)
        # 要么拒绝，要么削减 — 不能原样放行
        assert (not is_safe) or (np.linalg.norm(safe_u) < np.linalg.norm(u_huge))

    def test_zero_torque_is_safe(self, make_env):
        """零力矩在零位 → is_safe=True（回归基线）。"""
        env = make_env
        env.reset(np.zeros(6))
        from src.ilqt.robot_limits import RobotLimits
        from src.ilqt.components.predictive_safety import PredictiveSafetyFilter
        limits = RobotLimits.from_config(
            {}, dt=0.005, ctrlrange=env.model.actuator_ctrlrange[:env.NU],
        )
        safety = PredictiveSafetyFilter(
            env, limits, is_position_mode=False,
        )
        u_zero = np.zeros(6)
        arm_state = env.get_arm_state()
        _, is_safe = safety.filter(u_zero, arm_state)
        assert is_safe
