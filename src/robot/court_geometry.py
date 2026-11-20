"""标准网球场白线几何（纯几何模块，不依赖 MuJoCo）。

坐标约定（与 R1D 场景一致）：

- ``X`` 为球场长度方向，网平面位于 ``X = 0``；
- ``Y`` 为球场宽度方向；
- ``Z`` 向上，白线贴地绘制（默认中心 ``z = 0.001``，半厚 ``0.001``）。

两类 profile：

- ``legacy_v1``：保留旧 RM65 场地的 10 条白线（含两条整场十字中线、
  ±6 m 底线、±3 m 发球线、±4.5/±5 m 边线），仅用于旧产物与旧测试兼容；
- ``itf_2026_v1``：按 ITF Rules of Tennis (2026) 第 1 条的标准双打/单打场地，
  共 12 条线段，且不绘制横贯网平面的“球网线”。

白线为纯视觉几何，调用方必须设置 ``contype=conaffinity=0``；本模块不含
碰撞、材质或机器人位置。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "COURT_GEOMETRY_VERSION",
    "COURT_LINE_RGBA",
    "COURT_LINE_THICKNESS_M",
    "COURT_LINE_WIDTH_M",
    "COURT_LINE_Z_M",
    "COURT_PROFILES",
    "CourtLine",
    "build_court_lines",
    "court_profile_metadata",
]

# 几何版本：写入快照/产物，便于回放判断白线布局口径。
COURT_GEOMETRY_VERSION = 1

# ITF 标准线宽 5 cm；白线半厚与旧 RM65 模型一致（2 mm 薄片贴地）。
COURT_LINE_WIDTH_M = 0.05
COURT_LINE_THICKNESS_M = 0.001
COURT_LINE_Z_M = 0.001
COURT_LINE_RGBA: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

# ITF 标准场地尺寸（米）。
COURT_LENGTH_M = 23.77
COURT_DOUBLES_WIDTH_M = 10.97
COURT_SINGLES_WIDTH_M = 8.23
SERVICE_LINE_FROM_NET_M = 6.40
CENTRE_MARK_LENGTH_M = 0.10

# profile 白名单：未知值必须报错，不做静默回退。
COURT_PROFILES: tuple[str, ...] = ("legacy_v1", "itf_2026_v1")


@dataclass(frozen=True)
class CourtLine:
    """一条球场白线的不可变几何描述。

    Attributes:
        name: 稳定逻辑名称（用于测试、快照和渲染诊断）。
        center: 白线中心 ``(x, y, z)``，单位米。
        half_size: MuJoCo plane geom 的半尺寸 ``(sx, sy, sz)``，单位米。
        rgba: 白线颜色，四元组。
    """

    name: str
    center: tuple[float, float, float]
    half_size: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = COURT_LINE_RGBA

    def geom_attributes(self) -> dict[str, Any]:
        """返回可直接写给 MuJoCo ``<geom>`` 的属性字典。

        Returns:
            含 ``name/type/size/pos/rgba/contype/conaffinity`` 的字典；
            白线不参与碰撞。
        """
        sx, sy, sz = self.half_size
        cx, cy, cz = self.center
        r, g, b, a = self.rgba
        return {
            "name": self.name,
            "type": "plane",
            "size": f"{sx:g} {sy:g} {sz:g}",
            "pos": f"{cx:g} {cy:g} {cz:g}",
            "rgba": f"{r:g} {g:g} {b:g} {a:g}",
            "contype": "0",
            "conaffinity": "0",
        }


def _line(
    name: str,
    center_x: float,
    center_y: float,
    half_x: float,
    half_y: float,
) -> CourtLine:
    """构造一条贴地白线（统一 z 与半厚）。"""
    return CourtLine(
        name=name,
        center=(center_x, center_y, COURT_LINE_Z_M),
        half_size=(half_x, half_y, COURT_LINE_THICKNESS_M),
    )


def _legacy_lines() -> tuple[CourtLine, ...]:
    """旧 RM65 场地的 10 条白线（与 ``rm65_model.xml`` 逐字一致）。

    旧模型的线宽是 0.10 m（``size`` 半宽 0.05），与 ITF 的 0.05 m 不同；
    这里刻意保留旧几何，避免 legacy 渲染/像素对比发生变化。
    """
    legacy_half_w = 0.05  # 旧 RM65 白线半宽（0.10 m 全宽），不是 ITF 线宽
    return (
        _line("legacy_center_x", 0.0, 0.0, 12.0, legacy_half_w),
        _line("legacy_center_y", 0.0, 0.0, legacy_half_w, 12.0),
        _line("legacy_baseline_neg", -6.0, 0.0, legacy_half_w, 6.0),
        _line("legacy_baseline_pos", 6.0, 0.0, legacy_half_w, 6.0),
        _line("legacy_service_neg", -3.0, 0.0, legacy_half_w, 5.0),
        _line("legacy_service_pos", 3.0, 0.0, legacy_half_w, 5.0),
        _line("legacy_side_neg", -4.5, 0.0, 3.0, legacy_half_w),
        _line("legacy_side_pos", 4.5, 0.0, 3.0, legacy_half_w),
        _line("legacy_edge_neg", 0.0, -5.0, legacy_half_w, 12.0),
        _line("legacy_edge_pos", 0.0, 5.0, legacy_half_w, 12.0),
    )


def _itf_2026_lines() -> tuple[CourtLine, ...]:
    """按 ITF 2026 第 1 条生成标准场地 12 条白线。

    外缘尺寸：双打 23.77 × 10.97 m，单打宽 8.23 m；发球线距网 6.40 m；
    线宽 0.05 m。所有线段按“外缘/中心线”换算成中心与半尺寸。

    Returns:
        12 条白线：2 底线、2 双打边线、2 单打边线、2 发球线、
        2 段中心发球线（网两侧各一段）、2 个底线中心标记。
    """
    half_w = COURT_LINE_WIDTH_M / 2.0
    half_length = COURT_LENGTH_M / 2.0  # 11.885：双打外缘
    half_doubles = COURT_DOUBLES_WIDTH_M / 2.0  # 5.485
    half_singles = COURT_SINGLES_WIDTH_M / 2.0  # 4.115
    baseline_center = half_length - half_w  # 11.860
    doubles_side_center = half_doubles - half_w  # 5.460
    singles_side_center = half_singles - half_w  # 4.090
    service_center = SERVICE_LINE_FROM_NET_M - half_w  # 6.375
    centre_mark_center = (
        half_length - COURT_LINE_WIDTH_M - CENTRE_MARK_LENGTH_M / 2.0
    )  # 11.785：从底线内缘向场内 0.10 m

    # 底线：沿 Y 横跨双打全场，中心在双打外缘内侧半线宽处。
    baseline_y_half = half_doubles
    # 边线：沿 X 贯通两条底线外缘。
    side_x_half = half_length
    # 发球线：只到单打边线外缘，不伸入双打巷道。
    service_y_half = half_singles
    # 中心发球线：只在两条发球线之间（网两侧各一段）。
    centre_service_x_half = service_center

    return (
        _line("baseline_near", -baseline_center, 0.0, half_w, baseline_y_half),
        _line("baseline_far", baseline_center, 0.0, half_w, baseline_y_half),
        _line(
            "doubles_sideline_left",
            0.0,
            -doubles_side_center,
            side_x_half,
            half_w,
        ),
        _line(
            "doubles_sideline_right",
            0.0,
            doubles_side_center,
            side_x_half,
            half_w,
        ),
        _line(
            "singles_sideline_left",
            0.0,
            -singles_side_center,
            side_x_half,
            half_w,
        ),
        _line(
            "singles_sideline_right",
            0.0,
            singles_side_center,
            side_x_half,
            half_w,
        ),
        _line(
            "service_line_near",
            -service_center,
            0.0,
            half_w,
            service_y_half,
        ),
        _line(
            "service_line_far",
            service_center,
            0.0,
            half_w,
            service_y_half,
        ),
        _line(
            "centre_service_line_near",
            -service_center / 2.0,
            0.0,
            centre_service_x_half / 2.0,
            half_w,
        ),
        _line(
            "centre_service_line_far",
            service_center / 2.0,
            0.0,
            centre_service_x_half / 2.0,
            half_w,
        ),
        _line(
            "centre_mark_near",
            -centre_mark_center,
            0.0,
            CENTRE_MARK_LENGTH_M / 2.0,
            half_w,
        ),
        _line(
            "centre_mark_far",
            centre_mark_center,
            0.0,
            CENTRE_MARK_LENGTH_M / 2.0,
            half_w,
        ),
    )


def build_court_lines(profile: str = "legacy_v1") -> tuple[CourtLine, ...]:
    """按 profile 返回球场白线几何。

    Args:
        profile: ``legacy_v1``（旧 RM65 10 线）或 ``itf_2026_v1``
            （ITF 标准 12 线）。

    Returns:
        不可变的白线元组，顺序稳定（legacy 保持旧输出顺序）。

    Raises:
        ValueError: profile 不在白名单内。
    """
    if profile == "legacy_v1":
        return _legacy_lines()
    if profile == "itf_2026_v1":
        return _itf_2026_lines()
    raise ValueError(
        f"未知球场 profile {profile!r}；允许值：{COURT_PROFILES}"
    )


def court_profile_metadata(profile: str) -> dict[str, Any]:
    """返回可写入快照的球场视觉元数据。

    ``line_width_m`` 按 profile 给出真实白线全宽：``legacy_v1`` 沿用旧
    RM65 模型的 0.10 m（半宽 0.05，由 :func:`_legacy_lines` 推导），
    ``itf_2026_v1`` 为 ITF 标准 0.05 m。该字段只作文本元数据，不参与
    几何/碰撞。

    Args:
        profile: 球场 profile 名称。

    Returns:
        含 ``profile/geometry_version/line_count/line_width_m`` 的字典。
        未知 profile 由 :func:`build_court_lines` 抛错。

    Raises:
        ValueError: profile 不在白名单内。
    """
    lines = build_court_lines(profile)
    if profile == "legacy_v1":
        # 旧白线半宽 0.05（见 _legacy_lines 的 legacy_half_w）→ 全宽 0.10；
        # 从几何反推，避免元数据与真实白线两处常量漂移。
        line_width_m = float(lines[0].half_size[1]) * 2.0
    else:
        line_width_m = COURT_LINE_WIDTH_M
    return {
        "profile": profile,
        "geometry_version": COURT_GEOMETRY_VERSION,
        "line_count": len(lines),
        "line_width_m": line_width_m,
    }
