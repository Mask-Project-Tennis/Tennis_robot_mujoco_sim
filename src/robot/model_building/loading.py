"""R1D 模型加载、装配管线与依赖清单。

``load_robot_model`` 从 runtime YAML 参数构建 MuJoCo 模型；``_assemble_bound_model``
是模板加载后的内存装配管线（场景数值、球拍、限位、校验）；``collect_model_dependencies``
输出模型文件依赖清单（A1 的 ``model_assets`` 为唯一扫描实现，本模块不重复扫描）。

分层：本模块位于依赖链末端（types ← template ← racket ← scene ← loading）。
"""

from __future__ import annotations

import logging
import sys
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import mujoco

from src.robot.model_assets import (
    DEPENDENCY_ROLE_ARM_FRAGMENT,
    collect_model_file_dependencies,
)
from src.robot.model_assets.dependencies import _dependency_entry
from src.robot.model_binding import bind_robot_model
from src.robot.model_building.scene import (
    _apply_court_profile,
    _apply_scene_parameters,
    _check_initial_state,
    _populate_joint_limits,
    _validate_model_shape,
)
from src.robot.model_building.template import (
    SUPPORTED_ROBOT_ID,
    _find_body_with_parent,
    _fragment_worldbody,
    _model_side,
    _parse_fragment,
    _passive_side,
    _require_scene,
    _static_arm_copy,
)
from src.robot.model_building.types import RobotModelBundle
from src.robot.racket_profile import RacketProfile
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import (
    RobotRuntimeParameters,
    validate_runtime_parameters,
)
from src.utils.mujoco_loader import load_mujoco_model

logger = logging.getLogger(__name__)

# 本模块位于 ``src/robot/model_building/``；R1D 臂片段的回退目录是 ``src/robot``
# （``r1d_arm_*.xml`` 的真实位置）。用 ``__file__`` 显式推导，禁止依赖进程 CWD。
_ROBOT_DIR = Path(__file__).resolve().parents[1]

# 导入期绑定的 A1 文件级扫描实现（仅用于识别“本模块绑定是否被 test patch 替换”，
# 不是第二份扫描实现）。
_IMPORTED_FILE_SCAN = collect_model_file_dependencies


def load_robot_model(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    racket_profile: RacketProfile | None = None,
) -> RobotModelBundle:
    """组装并加载 R1D 单侧主动臂 + 对侧静态渲染臂模型。

    流程：校验身份/维数/初态（复用 2.2 的 kinematics 级校验）→ 读取磁盘
    模板 → 仅在模板副本（xml_text）上应用本次参数 → 经 load_mujoco_model
    加载 → bind_robot_model 绑定 → 校验模型维数与初态在限位内 → 从
    model.jnt_range 填充 q_lower/q_upper（不重复抄 URDF 数字）。

    Args:
        profile: 机器人描述（当前仅支持 robot_id="unitree_r1d"；
            model_path 指向对应侧模板，侧别由此显式表达）。
        parameters: 运行参数（须含 scene；q_lower/q_upper 通常为 None，
            由模型 jnt_range 回填；若显式提供则必须与模型一致，不一致
            即拒绝，不接受"收紧/放宽"覆盖）。
        racket_profile: 可选球拍档案（``RacketProfile``）。提供时球拍几何/
            质量/惯量/安装变换全部来自档案，且 ``parameters.scene`` 不得
            再给 legacy ``racket_*`` 字段；省略（None）时保持旧 API 语义：
            球拍几何由 scene 的 legacy 字段或模板默认值决定。

    Returns:
        同源组装结果；parameters 为填充关节限位后的新对象（输入不被改写）。

    Raises:
        ValueError: robot_id 不支持、scene 缺失、校验失败、模板结构漂移、
            模型维数与描述不符、初态超出模型限位，或显式 q_lower/q_upper
            与模型 jnt_range 不一致；档案模式与 scene legacy 球拍字段同时
            出现也在此拒绝（球拍物理唯一来源是档案）。
    """
    if profile.robot_id != SUPPORTED_ROBOT_ID:
        raise ValueError(
            f"load_robot_model 当前仅支持 {SUPPORTED_ROBOT_ID!r} 的场景组装，"
            f"收到 robot_id={profile.robot_id!r}；RM65 等继续使用既有模型入口"
            f"与兼容适配器"
        )
    _require_scene(parameters, "load_robot_model")
    # 身份/关节序/维数/初态（kinematics 级）统一走 2.2 校验
    validate_runtime_parameters(parameters, profile, purpose="kinematics")

    bundle = _assemble_bound_model(
        profile,
        parameters,
        validate_shape=_validate_model_shape,
        racket_profile=racket_profile,
    )
    logger.info(
        "R1D 模型组装完成: nq=%d nv=%d nu=%d TCP site=%s 球拍=%s（参数来源: %s）",
        bundle.model.nq,
        bundle.model.nv,
        bundle.model.nu,
        profile.tcp_site_name,
        bundle.racket.racket_id if bundle.racket is not None else "legacy",
        bundle.parameters.source,
    )
    return bundle


def _assemble_bound_model(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    validate_shape: Callable[[mujoco.MjModel, RobotProfile], None],
    mutate: Callable[[ET.Element], None] | None = None,
    racket_profile: RacketProfile | None = None,
) -> RobotModelBundle:
    """共享组装管线（load_robot_model 与 3.2 带球场景 build_tennis_scene 共用）。

    流程：读取磁盘模板 → 仅在模板副本（xml_text）上应用本次参数 →
    可选 ``mutate`` 追加组装（如 3.2 的球/掩码）→ 经 load_mujoco_model
    编译 → ``validate_shape`` 校验模型形状（裸模型与带球模型维数契约
    不同，由调用方注入）→ bind_robot_model 按名绑定 → 校验初态 → 从
    model.jnt_range 填充 q_lower/q_upper 并复核。磁盘模板字节不变。

    Args:
        profile: 机器人描述（model_path 指向模板）。
        parameters: 运行参数（须含 scene；仅做 kinematics 级校验，
            完整仿真预检由 3.1 validate_simulation_inputs 负责）。
        validate_shape: 模型形状校验回调（裸/带球维数契约不同）。
        mutate: 模板副本的追加组装回调（元素级修改，须保持可编译）。
        racket_profile: 可选球拍档案；None 保持旧 API 的模板/legacy 语义。

    Returns:
        同源组装结果；parameters 为填充关节限位后的新对象（输入不改写）。

    Raises:
        ValueError: scene 缺失、模板结构漂移、形状校验失败、名称绑定
            失败、初态超限或显式限位与模型不一致。
    """
    _require_scene(parameters, "_assemble_bound_model")

    # 仅在模板副本上应用参数：磁盘模板只读，数值经 xml_text 编译
    template_root = ET.fromstring(profile.model_path.read_bytes())
    # 白线口径来自 scene.court_profile（V01/I2）：磁盘模板固定 legacy，
    # 运行时按需在内存副本替换为 ITF 标准 12 条（纯视觉，不参与碰撞）。
    _apply_court_profile(template_root, parameters)
    # 侧别只在需要重烘对侧固定姿态时推导：合成夹具/旧调用不传 fixed poses，
    # 仍须能完成装配（不因关节命名不是 left_/right_ 而拒绝）
    scene = parameters.scene
    side = (
        _model_side(profile)
        if scene is not None and scene.fixed_joint_poses
        else None
    )
    # F08：装配过程中真实读取的兄弟片段（仅 fixed_joint_poses 非空的分支
    # 会读取对侧臂片段），由 _apply_fixed_joint_poses 在读到文件时追加。
    arm_fragments: list[Path] = []
    racket = _apply_scene_parameters(
        template_root,
        parameters,
        side=side,
        profile=profile,
        racket_profile=racket_profile,
        template_mode=False,
        fragments=arm_fragments,
    )
    if mutate is not None:
        mutate(template_root)
    xml_text = ET.tostring(template_root, encoding="unicode")
    model = load_mujoco_model(profile.model_path, xml_text=xml_text)

    validate_shape(model, profile)
    binding = bind_robot_model(profile, model)
    _check_initial_state(model, binding, parameters)

    filled = _populate_joint_limits(parameters, model, binding)
    # 填充限位后复核：init_q 必须落在模型 jnt_range 界内
    validate_runtime_parameters(filled, profile, purpose="kinematics")
    # F08：记录本次装配实际读取的文件闭包（含按真实分支读到的对侧臂片段），
    # 供产物快照完整保存装配依赖；不改变模型/绑定/参数任何既有语义。
    dependency_manifest = collect_model_dependencies(
        profile, filled, arm_fragments=arm_fragments
    )
    return RobotModelBundle(
        profile=profile,
        parameters=filled,
        model=model,
        binding=binding,
        racket=racket,
        dependency_manifest=dependency_manifest,
    )


def _apply_fixed_joint_poses(
    root: ET.Element,
    parameters: RobotRuntimeParameters,
    side: str,
    profile: RobotProfile | None = None,
    *,
    fragments: list[Path] | None = None,
) -> None:
    """按配置重新烘焙对侧固定臂姿态（替换模板中的静态臂 body）。

    片段从模型模板同目录的 ``r1d_arm_{passive}.xml`` 读取；只重建被动臂
    子树，主动臂/环境链/头部不受影响，磁盘模板与片段均不被改写。

    Args:
        root: 模板 XML 根元素（原地修改）。
        parameters: 运行参数（``scene.fixed_joint_poses`` 非空才生效）。
        side: 主动臂侧别（``left`` / ``right``）。
        profile: 机器人描述（决定片段的首选目录）。
        fragments: 可选输出列表；本函数实际读到片段文件时把路径追加进去，
            供 :func:`collect_model_dependencies` 记录装配闭包（F08）。

    Raises:
        ValueError: 覆盖关节不在片段中或角度超出该关节限位。
        FileNotFoundError: 对侧臂片段缺失。
    """
    scene = parameters.scene
    if scene is None or not scene.fixed_joint_poses:
        return
    passive_side = _passive_side(side)
    fragment_path = _arm_fragment_path(profile, side)
    # F08：只有这里会读取兄弟片段——就地登记，避免调用方猜测
    if fragments is not None:
        fragments.append(Path(fragment_path).resolve())
    passive_root = _parse_fragment(fragment_path, "mujocoinclude")
    passive_worldbody = _fragment_worldbody(passive_root, fragment_path)
    passive_root_body = next(
        (el for el in passive_worldbody if el.tag == "body"), None
    )
    if passive_root_body is None:
        raise ValueError(f"对侧裸臂片段缺少根 body（片段: {fragment_path}）")
    new_body = _static_arm_copy(
        passive_root_body, passive_side, dict(scene.fixed_joint_poses)
    )
    body_name = new_body.attrib.get("name")
    parent, old_body = _find_body_with_parent(root, str(body_name))
    index = list(parent).index(old_body)
    parent.remove(old_body)
    parent.insert(index, new_body)


def _arm_fragment_path(profile: RobotProfile | None, side: str) -> Path:
    """定位对侧臂 MJCF 片段路径（优先模型模板同目录，回退仓库 src/robot）。

    Raises:
        FileNotFoundError: 两种路径都找不到片段。
    """
    passive_side = _passive_side(side)
    name = f"r1d_arm_{passive_side}.xml"
    candidates: list[Path] = []
    if profile is not None:
        candidates.append(Path(profile.model_path).parent / name)
    # 回退到仓库内的 src/robot（本文件位于 src/robot/model_building/）；
    # 不使用 CWD：调用方可能从任意目录加载模型。
    candidates.append(_ROBOT_DIR / name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"对侧臂片段缺失，无法重烘固定关节姿态（查找: "
        f"{[str(c) for c in candidates]}）"
    )


def collect_model_dependencies(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    *,
    arm_fragments: Sequence[Path] = (),
) -> tuple[dict[str, Any], ...]:
    """汇总一次装配实际读取的完整依赖闭包。

    文件级闭包来自 :func:`collect_model_file_dependencies`（模型 +
    ``<include>`` + 网格纹理）；对侧臂片段由调用方按**真实装配分支**传入
    ``arm_fragments``（只有 ``scene.fixed_joint_poses`` 非空时工厂才会读取
    ``r1d_arm_{passive}.xml``，F08 根因即该片段此前未进快照）。

    Args:
        profile: 机器人描述（``model_path`` 为实际加载的模板）。
        parameters: 运行参数（当前不参与闭包判定，保留以表达"同一次装配"）。
        arm_fragments: 装配过程中真实读取的对侧臂片段路径。

    Returns:
        依赖条目元组（按 ``role/relative_path`` 排序，按绝对路径去重）。
    """
    # 契约 §6：旧 facade 上的 ``collect_model_file_dependencies`` monkeypatch
    # 必须继续生效（既有测试的 patch 点）。优先使用本模块当前绑定（新 patch
    # 点）；本模块绑定未被替换时，回退旧 facade 的运行时绑定。这里只查询
    # ``sys.modules``，不主动 import 旧 facade，因此不构成反向依赖或循环 import。
    scan = collect_model_file_dependencies
    facade = sys.modules.get("src.robot.model_factory")
    if facade is not None and scan is _IMPORTED_FILE_SCAN:
        scan = getattr(facade, "collect_model_file_dependencies", scan)
    entries, _ = scan(profile.model_path)
    merged: list[dict[str, Any]] = list(entries)
    for fragment in arm_fragments:
        merged.append(
            _dependency_entry(DEPENDENCY_ROLE_ARM_FRAGMENT, Path(fragment).resolve())
        )
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for item in merged:
        deduped.setdefault((str(item["role"]), str(item["source"])), item)
    return tuple(
        sorted(deduped.values(), key=lambda item: (item["role"], item["relative_path"]))
    )
