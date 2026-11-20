"""球场白线纯几何模块测试（V01，E 工作线）。

覆盖 ITF 2026 标准尺寸、legacy 逐字兼容、profile 拒绝、无球网横线、
白线不参与碰撞以及模块不依赖 MuJoCo。
"""

from __future__ import annotations

import dataclasses
import math
import sys

import pytest

from src.robot import court_geometry as cg

TOL = 1e-9


def _find(lines: tuple[cg.CourtLine, ...], name: str) -> cg.CourtLine:
    """按名称取一条白线。"""
    for line in lines:
        if line.name == name:
            return line
    raise AssertionError(f"缺少白线 {name}")


def _extent(line: cg.CourtLine, axis: int) -> tuple[float, float]:
    """返回一条线在指定轴上的 (min, max)。"""
    center = line.center[axis]
    half = line.half_size[axis]
    return center - half, center + half


class TestLegacyProfile:
    def test_ten_lines_in_original_order(self) -> None:
        lines = cg.build_court_lines("legacy_v1")
        assert len(lines) == 10
        # 前两条是整场十字中线，尺寸/位置逐字等于旧 rm65_model.xml
        # （旧线宽 0.10 m，即半宽 0.05；不要误用 ITF 的 0.05 m 全宽）。
        assert lines[0].half_size == (12.0, 0.05, 0.001)
        assert lines[0].center == (0.0, 0.0, 0.001)
        assert lines[1].half_size == (0.05, 12.0, 0.001)
        assert lines[1].center == (0.0, 0.0, 0.001)
        # 旧底线 ±6、发球线 ±3、边线 ±4.5/±5 保持原值。
        assert lines[2].center[0] == pytest.approx(-6.0)
        assert lines[3].center[0] == pytest.approx(6.0)
        assert lines[4].center[0] == pytest.approx(-3.0)
        assert lines[5].center[0] == pytest.approx(3.0)
        assert lines[6].center[0] == pytest.approx(-4.5)
        assert lines[7].center[0] == pytest.approx(4.5)
        assert lines[8].center[1] == pytest.approx(-5.0)
        assert lines[9].center[1] == pytest.approx(5.0)

    def test_legacy_all_flat_and_white(self) -> None:
        for line in cg.build_court_lines("legacy_v1"):
            assert line.center[2] == pytest.approx(cg.COURT_LINE_Z_M)
            assert line.half_size[2] == pytest.approx(cg.COURT_LINE_THICKNESS_M)
            assert line.rgba == cg.COURT_LINE_RGBA

    def test_legacy_matches_rm65_geoms_exactly(self) -> None:
        """legacy geom 属性必须与旧 ``_COURT_LINE_GEOMS`` 逐字一致。"""
        old = (
            ("12 0.05 0.001", "0 0 0.001"),
            ("0.05 12 0.001", "0 0 0.001"),
            ("0.05 6 0.001", "-6 0 0.001"),
            ("0.05 6 0.001", "6 0 0.001"),
            ("0.05 5 0.001", "-3 0 0.001"),
            ("0.05 5 0.001", "3 0 0.001"),
            ("3 0.05 0.001", "-4.5 0 0.001"),
            ("3 0.05 0.001", "4.5 0 0.001"),
            ("0.05 12 0.001", "0 -5 0.001"),
            ("0.05 12 0.001", "0 5 0.001"),
        )
        lines = cg.build_court_lines("legacy_v1")
        assert len(lines) == len(old)
        for line, (size, pos) in zip(lines, old):
            attrs = line.geom_attributes()
            assert attrs["size"] == size
            assert attrs["pos"] == pos


class TestItfProfile:
    def test_twelve_named_lines(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        assert len(lines) == 12
        names = [line.name for line in lines]
        assert len(set(names)) == len(names)
        assert set(names) == {
            "baseline_near",
            "baseline_far",
            "doubles_sideline_left",
            "doubles_sideline_right",
            "singles_sideline_left",
            "singles_sideline_right",
            "service_line_near",
            "service_line_far",
            "centre_service_line_near",
            "centre_service_line_far",
            "centre_mark_near",
            "centre_mark_far",
        }

    def test_doubles_outer_rectangle(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        x_min = min(_extent(line, 0)[0] for line in lines)
        x_max = max(_extent(line, 0)[1] for line in lines)
        y_min = min(_extent(line, 1)[0] for line in lines)
        y_max = max(_extent(line, 1)[1] for line in lines)
        assert x_min == pytest.approx(-11.885, abs=TOL)
        assert x_max == pytest.approx(11.885, abs=TOL)
        assert y_min == pytest.approx(-5.485, abs=TOL)
        assert y_max == pytest.approx(5.485, abs=TOL)

    def test_baselines_centres_and_width(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        for name, sign in (("baseline_near", -1.0), ("baseline_far", 1.0)):
            line = _find(lines, name)
            assert line.center[0] == pytest.approx(sign * 11.860, abs=TOL)
            assert line.half_size[0] == pytest.approx(0.025, abs=TOL)
            # 底线横跨双打全场：半长 = 10.97 / 2。
            assert line.half_size[1] == pytest.approx(5.485, abs=TOL)
            lo, hi = _extent(line, 0)
            if sign < 0:
                assert (lo, hi) == pytest.approx((-11.885, -11.835))
            else:
                assert (lo, hi) == pytest.approx((11.835, 11.885))

    def test_sidelines_geometry(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        doubles = [
            _find(lines, "doubles_sideline_left"),
            _find(lines, "doubles_sideline_right"),
        ]
        singles = [
            _find(lines, "singles_sideline_left"),
            _find(lines, "singles_sideline_right"),
        ]
        for line in doubles:
            assert line.half_size[0] == pytest.approx(11.885, abs=TOL)
            assert line.half_size[1] == pytest.approx(0.025, abs=TOL)
            assert any(
                math.isclose(line.center[1], sign * 5.460, abs_tol=TOL)
                for sign in (-1.0, 1.0)
            )
        for line in singles:
            assert line.half_size[0] == pytest.approx(11.885, abs=TOL)
            assert any(
                math.isclose(line.center[1], sign * 4.090, abs_tol=TOL)
                for sign in (-1.0, 1.0)
            )
        # 单打边线外缘 = ±4.115（8.23 / 2）。
        assert min(_extent(line, 1)[0] for line in singles) == pytest.approx(
            -4.115, abs=TOL
        )
        assert max(_extent(line, 1)[1] for line in singles) == pytest.approx(
            4.115, abs=TOL
        )

    def test_service_lines_stop_at_singles_sideline(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        for name, sign in (("service_line_near", -1.0), ("service_line_far", 1.0)):
            line = _find(lines, name)
            assert line.center[0] == pytest.approx(sign * 6.375, abs=TOL)
            lo, hi = _extent(line, 0)
            outer = lo if sign < 0 else hi
            assert outer == pytest.approx(sign * 6.400, abs=TOL)
            # 发球线只到单打边线外缘，不伸入双打巷道。
            assert line.half_size[1] == pytest.approx(4.115, abs=TOL)

    def test_centre_service_line_only_between_service_lines(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        near = _find(lines, "centre_service_line_near")
        far = _find(lines, "centre_service_line_far")
        assert near.half_size[0] == pytest.approx(6.375 / 2.0, abs=TOL)
        assert far.half_size[0] == pytest.approx(6.375 / 2.0, abs=TOL)
        assert _extent(near, 0) == pytest.approx((-6.375, 0.0))
        assert _extent(far, 0) == pytest.approx((0.0, 6.375))
        assert near.half_size[1] == pytest.approx(0.025, abs=TOL)

    def test_centre_marks_length_and_position(self) -> None:
        lines = cg.build_court_lines("itf_2026_v1")
        for name, sign in (("centre_mark_near", -1.0), ("centre_mark_far", 1.0)):
            line = _find(lines, name)
            lo, hi = _extent(line, 0)
            # 从底线内缘（±11.835）向场内 0.10 m。
            if sign < 0:
                assert (lo, hi) == pytest.approx((-11.835, -11.735))
            else:
                assert (lo, hi) == pytest.approx((11.735, 11.835))
            assert line.half_size[1] == pytest.approx(0.025, abs=TOL)

    def test_no_cross_court_line_at_net(self) -> None:
        """ITF profile 不得再画横贯 X=0 的“球网线”。"""
        for line in cg.build_court_lines("itf_2026_v1"):
            if abs(line.center[0]) < 0.1:
                assert line.half_size[1] <= 0.1

    def test_width_flat_and_colour_uniform(self) -> None:
        for line in cg.build_court_lines("itf_2026_v1"):
            assert line.center[2] == pytest.approx(cg.COURT_LINE_Z_M)
            assert line.half_size[2] == pytest.approx(cg.COURT_LINE_THICKNESS_M)
            assert line.rgba == cg.COURT_LINE_RGBA
            # 宽 0.05 的线：至少一条轴半尺寸为 0.025。
            assert min(line.half_size[:2]) == pytest.approx(0.025, abs=TOL)


class TestProfileContract:
    def test_unknown_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="未知球场 profile"):
            cg.build_court_lines("wimbledon_2026")

    def test_profile_whitelist(self) -> None:
        assert cg.COURT_PROFILES == ("legacy_v1", "itf_2026_v1")

    def test_metadata(self) -> None:
        meta = cg.court_profile_metadata("itf_2026_v1")
        assert meta == {
            "profile": "itf_2026_v1",
            "geometry_version": cg.COURT_GEOMETRY_VERSION,
            "line_count": 12,
            "line_width_m": 0.05,
        }
        # P3：legacy 白线实际全宽 0.10 m（半宽 0.05），元数据不得再报 0.05。
        legacy = cg.court_profile_metadata("legacy_v1")
        assert legacy["line_count"] == 10
        assert legacy["line_width_m"] == pytest.approx(0.10, abs=TOL)
        assert legacy["line_width_m"] == pytest.approx(
            2.0 * cg.build_court_lines("legacy_v1")[0].half_size[1], abs=TOL)

    def test_metadata_rejects_unknown(self) -> None:
        with pytest.raises(ValueError):
            cg.court_profile_metadata("unknown")


class TestCourtLineValueObject:
    def test_frozen(self) -> None:
        line = cg.build_court_lines("itf_2026_v1")[0]
        with pytest.raises(dataclasses.FrozenInstanceError):
            line.name = "changed"  # type: ignore[misc]

    def test_geom_attributes_are_non_colliding(self) -> None:
        line = _find(cg.build_court_lines("itf_2026_v1"), "baseline_near")
        attrs = line.geom_attributes()
        assert attrs["type"] == "plane"
        assert attrs["contype"] == "0"
        assert attrs["conaffinity"] == "0"
        assert attrs["name"] == "baseline_near"
        assert attrs["size"] == "0.025 5.485 0.001"
        assert attrs["pos"] == "-11.86 0 0.001"
        assert attrs["rgba"] == "1 1 1 1"

    def test_module_does_not_import_mujoco(self) -> None:
        assert "mujoco" not in sys.modules or "mujoco" not in vars(
            sys.modules["src.robot.court_geometry"]
        )
        assert not hasattr(cg, "mujoco")
