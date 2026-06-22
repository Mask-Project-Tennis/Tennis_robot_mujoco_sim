"""测试 src/sim/hit_detection.py — 击球结果判定模块。

TDD Cycle 5: classify_hit_result
TDD Cycle 6: determine_hit_from_type
"""
import pytest

from src.sim.hit_detection import classify_hit_result, determine_hit_from_type


# ============================================================================
# Cycle 5: classify_hit_result — 控制台消息分类
# ============================================================================


class TestClassifyHitResult:
    """测试击球结果分类（替代 pos_error 阈值）。"""

    def test_active_contact_returns_success(self):
        """主动接触（球拍速度 > 0.3 m/s）→ 击打成功。"""
        result = classify_hit_result(active_contact=True, passive_contact=False, pos_error=0.2)
        assert "成功" in result

    def test_passive_contact_returns_passive(self):
        """被动接触（球拍速度 ≤ 0.3 m/s）→ 被动命中。"""
        result = classify_hit_result(active_contact=False, passive_contact=True, pos_error=0.1)
        assert "被动" in result

    def test_no_contact_returns_miss_even_with_small_error(self):
        """无接触 + 小 pos_error → 仍判定为未命中（核心修复）。"""
        result = classify_hit_result(active_contact=False, passive_contact=False, pos_error=0.02)
        assert "未命中" in result


# ============================================================================
# Cycle 6: determine_hit_from_type — 提取脚本命中判定
# ============================================================================


class TestDetermineHitFromType:
    """测试基于 hit_type 的命中判定（统一提取脚本逻辑）。"""

    def test_active_returns_true(self):
        assert determine_hit_from_type("active") is True

    def test_passive_returns_true(self):
        assert determine_hit_from_type("passive") is True

    def test_miss_returns_false(self):
        assert determine_hit_from_type("miss") is False
