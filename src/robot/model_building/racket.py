"""R1D 球拍装配。

负责把 ``RacketProfile``（质量/惯量/几何/视觉）写入 MJCF，并保留旧版
``_apply_legacy_racket`` 路径（模板生成与未提供 profile 时使用）。

依赖 :mod:`src.robot.model_building.template` 的常量与 body/数值 helper；
不 import ``scene``/``loading``，保持分层单向。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np

from src.robot.model_building.template import (
    _RACKET_BODY,
    _RACKET_CONNECTOR_GEOM,
    _RACKET_CONNECTOR_RGBA,
    _RACKET_FACE_HALF_LENGTH,
    _RACKET_FACE_HALF_THICKNESS,
    _RACKET_FACE_HALF_WIDTH,
    _RACKET_FACE_VISUAL_GEOM,
    _RACKET_FACE_VISUAL_RGBA,
    _RACKET_FRAME_GEOM_FORMAT,
    _RACKET_FRAME_GEOMS,
    _RACKET_FRAME_RGBA,
    _RACKET_GEOM,
    _RACKET_HANDLE_GEOM,
    _RACKET_HANDLE_LENGTH,
    _RACKET_HANDLE_RADIUS,
    _RACKET_HANDLE_RGBA,
    _RACKET_THROAT_GEOM_FORMAT,
    _RACKET_THROAT_GEOMS,
    _RACKET_THROAT_RGBA,
    _TCP_SITE,
    _find_body,
    _find_body_with_parent,
    _find_geom,
    _fmt_float,
    _fmt_vec,
)
from src.robot.model_building.types import RacketAssembly
from src.robot.racket_profile import RacketProfile
from src.robot.runtime_parameters import RobotRuntimeParameters


def _has_legacy_racket(scene: Any) -> bool:
    """判断 scene 是否提供了旧 API 球拍字段（5 项全有才算提供）。"""
    fields = (
        scene.racket_position,
        scene.racket_quaternion,
        scene.racket_radius,
        scene.racket_half_thickness,
        scene.racket_mass,
    )
    return all(value is not None for value in fields)


def _require_no_legacy_racket(scene: Any) -> None:
    """RacketProfile 模式下禁止 scene 再给球拍几何/质量（单一物理来源）。

    Raises:
        ValueError: scene 仍带任一 racket_* 字段。
    """
    leftovers = [
        name
        for name in (
            "racket_position",
            "racket_quaternion",
            "racket_radius",
            "racket_half_thickness",
            "racket_mass",
        )
        if getattr(scene, name) is not None
    ]
    if leftovers:
        raise ValueError(
            f"提供 racket_profile 时 scene.racket_* 必须全部为 None"
            f"（球拍几何/质量/安装的唯一物理来源是 RacketProfile；"
            f"冲突字段: {leftovers}）"
        )


def _racket_inertial(
    profile: RacketProfile,
) -> tuple[float, np.ndarray, np.ndarray]:
    """合并球拍与连接件质量/COM/惯量（相对合并 COM，球拍局部系）。

    连接件只在安装时计入一次；连接件惯量按其自身质心给出，合并时用
    平行轴定理搬到合并质心。
    """
    mass = float(profile.mass)
    com = np.asarray(profile.com, dtype=np.float64)
    inertia = np.asarray(profile.inertia.as_matrix(), dtype=np.float64)
    connector = profile.connector
    if connector is None:
        return mass, com, inertia
    connector_mass = float(connector.mass)
    connector_com = np.asarray(connector.com, dtype=np.float64)
    connector_inertia = np.asarray(connector.inertia.as_matrix(), dtype=np.float64)
    total = mass + connector_mass
    merged_com = (mass * com + connector_mass * connector_com) / total

    def _shift(
        local_inertia: np.ndarray,
        local_mass: float,
        local_com: np.ndarray,
        target_com: np.ndarray,
    ) -> np.ndarray:
        offset = local_com - target_com
        return local_inertia + local_mass * (
            float(np.dot(offset, offset)) * np.eye(3) - np.outer(offset, offset)
        )

    merged_inertia = _shift(
        inertia, mass, com, merged_com
    ) + _shift(connector_inertia, connector_mass, connector_com, merged_com)
    return total, merged_com, merged_inertia


def _racket_assembly(profile: RacketProfile) -> RacketAssembly:
    """把 RacketProfile 的装配事实固化为 bundle 暴露的只读记录。"""
    total_mass, com, inertia = _racket_inertial(profile)
    connector_mass = 0.0 if profile.connector is None else float(
        profile.connector.mass
    )
    geometry = profile.geometry
    return RacketAssembly(
        racket_id=str(profile.racket_id),
        source_path=Path(profile.source_path),
        total_mass=total_mass,
        com=np.array(com, dtype=np.float64, copy=True),
        inertia_diagonal=np.array(np.diag(inertia), dtype=np.float64, copy=True),
        face_center=np.array(geometry.face.center, dtype=np.float64, copy=True),
        face_semi_axes=np.array(geometry.face.semi_axes, dtype=np.float64, copy=True),
        mount_position=np.array(
            profile.mount.position, dtype=np.float64, copy=True
        ),
        mount_wxyz=np.array(profile.mount.wxyz, dtype=np.float64, copy=True),
        connector_mass=connector_mass,
    )


def _quat_aligning_z(axis: Any) -> np.ndarray:
    """求把局部 +Z 轴对齐到给定方向的单位四元数（wxyz）。

    Raises:
        ValueError: 轴向为零向量。
    """
    direction = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        raise ValueError("连接件安装轴为零向量（RacketProfile 非法）")
    direction = direction / norm
    dot = float(direction[2])
    if dot > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    if dot < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
    cross = np.cross(np.array([0.0, 0.0, 1.0]), direction)
    quat = np.array([1.0 + dot, cross[0], cross[1], cross[2]], dtype=np.float64)
    return quat / float(np.linalg.norm(quat))


def _append_visual_geom(
    body: ET.Element,
    *,
    name: str,
    geom_type: str,
    size: str,
    rgba: str,
    pos: str | None = None,
    fromto: str | None = None,
    quat: str | None = None,
) -> ET.Element:
    """追加一个不参与碰撞（contype=0/conaffinity=0）的视觉 geom。"""
    attrs: dict[str, str] = {
        "name": name,
        "type": geom_type,
        "size": size,
        "rgba": rgba,
        "contype": "0",
        "conaffinity": "0",
    }
    if pos is not None:
        attrs["pos"] = pos
    if fromto is not None:
        attrs["fromto"] = fromto
    if quat is not None:
        attrs["quat"] = quat
    return ET.SubElement(body, "geom", attrs)


def _apply_racket_profile(
    root: ET.Element, profile: RacketProfile, flange_pos: np.ndarray
) -> None:
    """按 RacketProfile 重建球拍 body 子树（几何/质量/惯量/安装单一来源）。

    视觉/碰撞/TCP 同源：拍面碰撞椭球与视觉椭球都以 face.center 为中心、
    semi_axes 为半轴，细轴（半厚）沿球拍局部 +X，即 ``get_ee_normal`` 的
    拍面法向契约；TCP site 位于 face.center。

    Raises:
        ValueError: 模板缺少球拍 body 或 profile 质心/惯量非法。
    """
    _parent, body = _find_body_with_parent(root, _RACKET_BODY)
    for child in list(body):
        body.remove(child)
    mount = profile.mount
    body.set("pos", _fmt_vec(np.asarray(flange_pos) + np.asarray(mount.position)))
    body.set("quat", _fmt_vec(mount.wxyz))

    total_mass, com, inertia = _racket_inertial(profile)
    diagonal = np.diag(inertia)
    off_diagonal = inertia - np.diag(diagonal)
    inertial_attrs: dict[str, str] = {
        "pos": _fmt_vec(com),
        "mass": _fmt_float(total_mass),
    }
    if float(np.max(np.abs(off_diagonal))) < 1e-12:
        inertial_attrs["diaginertia"] = _fmt_vec(diagonal)
    else:
        inertial_attrs["fullinertia"] = _fmt_vec(
            (
                inertia[0, 0],
                inertia[1, 1],
                inertia[2, 2],
                inertia[0, 1],
                inertia[0, 2],
                inertia[1, 2],
            )
        )
    ET.SubElement(body, "inertial", inertial_attrs)

    geometry = profile.geometry
    face = geometry.face
    # 拍面碰撞几何：椭球，细轴（半厚）沿局部 +X ⇒ 法向 = 局部 +X
    ET.SubElement(
        body,
        "geom",
        {
            "name": _RACKET_GEOM,
            "type": "ellipsoid",
            "pos": _fmt_vec(face.center),
            "size": _fmt_vec(face.semi_axes),
            "rgba": "0.9 0.9 0.25 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    _append_visual_geom(
        body,
        name=_RACKET_FACE_VISUAL_GEOM,
        geom_type="ellipsoid",
        pos=_fmt_vec(face.center),
        size=_fmt_vec(face.semi_axes),
        rgba=_RACKET_FACE_VISUAL_RGBA,
    )
    handle = geometry.handle
    _append_visual_geom(
        body,
        name=_RACKET_HANDLE_GEOM,
        geom_type="capsule",
        fromto=_fmt_vec((*handle.start, *handle.end)),
        size=_fmt_float(handle.radius),
        rgba=_RACKET_HANDLE_RGBA,
    )
    for index, capsule in enumerate(geometry.throat):
        _append_visual_geom(
            body,
            name=_RACKET_THROAT_GEOM_FORMAT.format(index=index),
            geom_type="capsule",
            fromto=_fmt_vec((*capsule.start, *capsule.end)),
            size=_fmt_float(capsule.radius),
            rgba=_RACKET_THROAT_RGBA,
        )
    for index, capsule in enumerate(geometry.frame):
        _append_visual_geom(
            body,
            name=_RACKET_FRAME_GEOM_FORMAT.format(index=index),
            geom_type="capsule",
            fromto=_fmt_vec((*capsule.start, *capsule.end)),
            size=_fmt_float(capsule.radius),
            rgba=_RACKET_FRAME_RGBA,
        )
    connector = profile.connector
    if connector is not None:
        _append_visual_geom(
            body,
            name=_RACKET_CONNECTOR_GEOM,
            geom_type="cylinder",
            pos=_fmt_vec(connector.center),
            quat=_fmt_vec(_quat_aligning_z(connector.axis)),
            size=_fmt_vec((connector.radius, connector.length / 2.0)),
            rgba=_RACKET_CONNECTOR_RGBA,
        )
    ET.SubElement(
        body, "site", name=_TCP_SITE, pos=_fmt_vec(face.center), size="0.005"
    )


def _apply_legacy_racket(
    root: ET.Element, parameters: RobotRuntimeParameters, flange_pos: np.ndarray
) -> None:
    """旧 API 球拍参数化（无 RacketProfile 时的 T0 语义，保持向后兼容）。

    圆盘碰撞面 + 0.30 kg 缩放惯量；拍面细轴（局部 Z）经 quat 对齐到
    body 局部 +X（``get_ee_normal`` 契约）。

    Raises:
        ValueError: scene 未提供旧球拍字段或模板缺少期望元素。
    """
    scene = parameters.scene
    if scene is None or not _has_legacy_racket(scene):
        raise ValueError("旧 API 球拍参数缺失：需要 scene 提供完整 racket_* 字段")
    racket = _find_body(root, _RACKET_BODY)
    racket.set("pos", _fmt_vec(flange_pos + np.asarray(scene.racket_position)))
    racket.set("quat", _fmt_vec(scene.racket_quaternion))
    inertial = racket.find("inertial")
    racket_geom = racket.find("geom")
    if inertial is None or racket_geom is None:
        raise ValueError("模板球拍 body 缺少 inertial/geom（模板漂移）")
    inertial.set("mass", _fmt_float(float(scene.racket_mass)))
    # 复合体惯量（拍头+拍柄）：轴向 X 约 0.0018、横向 Y/Z 约 0.004
    # （0.30 kg 成人拍的迷你化近似；避免腕部动力学被放大 10 倍）。
    scale = float(scene.racket_mass) / 0.30
    inertial.set(
        "diaginertia",
        _fmt_vec((0.0018 * scale, 0.0040 * scale, 0.0040 * scale)),
    )
    racket_geom.set(
        "size",
        _fmt_vec((float(scene.racket_radius), float(scene.racket_half_thickness))),
    )

    face_visual = next(
        (
            geom
            for geom in root.iter("geom")
            if geom.get("name") == _RACKET_FACE_VISUAL_GEOM
        ),
        None,
    )
    if face_visual is not None:
        face_visual.set(
            "size",
            _fmt_vec(
                (
                    _RACKET_FACE_HALF_THICKNESS,
                    _RACKET_FACE_HALF_WIDTH,
                    _RACKET_FACE_HALF_LENGTH,
                )
            ),
        )
        face_visual.set("rgba", "0.92 0.92 0.84 0.30")

    handle = next(
        (
            geom
            for geom in root.iter("geom")
            if geom.get("name") == _RACKET_HANDLE_GEOM
        ),
        None,
    )
    if handle is not None:
        handle_start_z = -float(scene.racket_position[2]) + 0.02
        handle_end_z = handle_start_z + _RACKET_HANDLE_LENGTH
        handle.set(
            "fromto",
            _fmt_vec((0.0, 0.0, handle_start_z, 0.0, 0.0, handle_end_z)),
        )
        handle.set("size", _fmt_float(_RACKET_HANDLE_RADIUS))

        throat_starts = (
            (0.0, 0.035, handle_end_z, 0.0, 0.085, -_RACKET_FACE_HALF_LENGTH),
            (0.0, -0.035, handle_end_z, 0.0, -0.085, -_RACKET_FACE_HALF_LENGTH),
        )
        for name, values in zip(_RACKET_THROAT_GEOMS, throat_starts):
            _find_geom(root, name).set("fromto", _fmt_vec(values))

        frame_values = (
            (0.0, -_RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH,
             0.0, _RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH),
            (0.0, -_RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH,
             0.0, _RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH),
            (0.0, _RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH,
             0.0, _RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH),
            (0.0, -_RACKET_FACE_HALF_WIDTH, -_RACKET_FACE_HALF_LENGTH,
             0.0, -_RACKET_FACE_HALF_WIDTH, _RACKET_FACE_HALF_LENGTH),
        )
        for name, values in zip(_RACKET_FRAME_GEOMS, frame_values):
            _find_geom(root, name).set("fromto", _fmt_vec(values))
