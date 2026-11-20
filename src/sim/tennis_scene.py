"""R1D 带球场景组装（TennisSceneBundle）— 阶段 3 Task 3.2。

职责：
- ``build_tennis_scene(profile, parameters, task)``：在 R1D 模板的**内存
  XML 副本**上组装网球与地面/碰撞掩码，经 ``load_mujoco_model`` 的
  xml_text 通道编译，编译后**重新按名称绑定**（球/拍/保护对 id 均在
  本模型内解析，不复用另一模型的 id），返回 ``TennisSceneBundle``。
- ``TennisSceneBundle``：带球 RobotModelBundle + 球关节 qpos/qvel 起点 +
  球 geom id + 拍 geom id 元组 + 保护 geom id 对元组（施工计划
  §共享接口契约逐字段实现）。

球物理参数来源（rm65_model.xml 逐字抄录并记录，不随意新造）：
- 球段命名与几何/质量：L304-L308（body "ball"、freejoint
  "ball_freejoint"、geom "ball"，type=sphere size=0.033 mass=0.058，
  rgba 黄绿色）；关节/geom 实名来自 task 配置（沿项目命名先例）。
- 球接触掩码：L38 ``<default class="ball">``（contype=16，
  conaffinity=10 = 机身|球拍）。
- 球—拍弹性接触：L75 ``<pair r_racket_face ball condim="1"
  solref="-5000 0" solimp="0.9 0.99 0.001"/>``（零阻尼完美弹性反弹；
  参数逐字迁移，geom 名按 task 配置解析）。

碰撞掩码设计（沿 rm65_model.xml 文件头位方案；位分配 1/2/4/8/16 =
地面/机身/臂/球拍/球）：
- ``ground`` 1/31：地面万能碰撞（rm65 先例）；
- ``torso_box`` 2/29：机身与地面、臂、拍、球碰撞（简化机身保护域）；
- 主动臂链 geom 4/2：仅与机身碰撞 —— 保护对可被观测，臂—臂/臂—球/
  臂—拍不开启（臂段间非相邻保护由限位与 3.7 保护对负责）；
- ``racket_face`` 8/16：球拍与球（及机身，经机身 conaffinity 位 8）；
- 球 16/10：机身|球拍（rm65 class ball 逐字），球—地面经地面万能位；
- 其余 geom（环境链 mesh 等）保持模板视觉掩码 0/0 —— **非相邻碰撞只对
  已有可信的简化几何（torso_box/racket_face/球）负责，不宣称整机碰撞
  完备**（施工计划 Task 3.2 约束）。
- 球—拍接触**单独控制**：显式 ``<pair>`` 绕过 contype/conaffinity 掩码
  且不与动态碰撞重复（MuJoCo 3.x 实测，见 task-3.2 报告），运行时开关
  通道为 ``model.pair_margin[i]``（置大负值仅关球—拍、恢复原值即开）；
  3.4 的 ``set_arm_collision`` 按此通道仅切换球—拍接触，不关闭臂/拍—
  机身等其他保护碰撞。

球的 free joint 插入 worldbody 首位（位于全部主动关节之前）：下游消费
方必须按名称绑定地址，不得假设主动关节占据 qpos 前缀（施工计划
Review Focus 1 的布局压力场景即本模块的真实布局）。
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from dataclasses import dataclass

import mujoco

from src.robot.model_factory import (
    SUPPORTED_ROBOT_ID,
    RobotModelBundle,
    _GROUND_GEOM,
    _TORSO_GEOM,
    _assemble_bound_model,
)
from src.robot.racket_profile import RacketProfile
from src.robot.robot_profile import RobotProfile
from src.robot.runtime_parameters import RobotRuntimeParameters
from src.sim.sim_config import SimTaskConfig

logger = logging.getLogger(__name__)

# ---------------- 球物理参数（rm65_model.xml 逐字来源，勿随意改） ----------------

# 球 body 名（rm65_model.xml L304，沿命名先例；球关节/geom 实名来自 task）
_BALL_BODY_NAME = "ball"
# 球半径 m（L308 size="0.033"）
_BALL_RADIUS = 0.033
# 球质量 kg（L308 mass="0.058"）
_BALL_MASS = 0.058
# 球颜色（L308 rgba，黄绿色网球）
_BALL_RGBA = "0.95 0.95 0.1 1"
# 球接触掩码（L38 class="ball"：contype=16，conaffinity=10=机身|球拍）
_BALL_CONTYPE = 16
_BALL_CONAFFINITY = 10

# 球—拍弹性接触参数（L75 逐字）：condim=1 无摩擦法向 + 直接刚度/阻尼
# (-5000, 0) 零阻尼 → 完美弹性反弹
_PAIR_CONDIM = "1"
_PAIR_SOLREF = "-5000 0"
_PAIR_SOLIMP = "0.9 0.99 0.001"

# ---------------- 碰撞掩码位分配（沿 rm65 文件头方案） ----------------

# 地面：万能碰撞（rm65 L10 "contype=1 conaffinity=31"）
_MASK_GROUND = (1, 31)
# 机身：与地面(1)|臂(4)|拍(8)|球(16) 碰撞
_MASK_TORSO = (2, 29)
# 臂：仅机身（保护对可观测；臂—臂/臂—球/臂—拍不开启）
_MASK_ARM = (4, 2)
# 球拍：球（拍—机身经机身 conaffinity 位 8 生效）
_MASK_RACKET = (8, 16)


@dataclass(frozen=True)
class TennisSceneBundle:
    """带球场景同源组装结果 — 机器人 bundle + 球/拍/保护对的按名解析 id。

    Attributes:
        robot: 同源机器人组装结果（model 为带球模型，binding 只含主动
            关节/执行器/TCP，不含球 free joint）。
        ball_joint_id: 球 free joint id（在本 model 内按名解析）。
        ball_qpos_start: 球 free joint 的 qpos 起点（位置 xyz + 四元数
            wxyz 共 7 项）。
        ball_qvel_start: 球 free joint 的速度（dof）起点（线速度 + 角
            速度共 6 项）。
        ball_geom_id: 球碰撞 geom id。
        racket_geom_ids: 拍碰撞 geom id 元组（按 task.racket_geom_names 顺序）。
        protected_geom_pairs: 禁止接触 geom id 对元组（按
            task.protected_geom_pairs 顺序；球拍击球接触不在此列）。
    """

    robot: RobotModelBundle
    ball_joint_id: int
    ball_qpos_start: int
    ball_qvel_start: int
    ball_geom_id: int
    racket_geom_ids: tuple[int, ...]
    protected_geom_pairs: tuple[tuple[int, int], ...]


def build_tennis_scene(
    profile: RobotProfile,
    parameters: RobotRuntimeParameters,
    task: SimTaskConfig,
    *,
    racket_profile: RacketProfile | None = None,
) -> TennisSceneBundle:
    """组装 R1D 带球场景并按名称绑定球/拍/保护对地址。

    流程：task 与 profile 身份防御校验 → 复用 ``_assemble_bound_model``
    共享管线（模板副本应用参数 + 追加球/掩码/pair 组装 → 编译 → 带球
    形状校验 → 绑定 → 初态校验 → 限位回填）→ 编译后在本模型内按名解析
    球关节（校验 free 类型）、球 geom、拍 geom、保护对（校验两端互异且
    掩码可观测）→ 返回 TennisSceneBundle。

    Args:
        profile: 机器人描述（robot_id 须为 unitree_r1d，model_path 指向
            对应侧模板；侧别由配置显式表达）。
        parameters: 运行参数（须含 scene；本函数做 kinematics 级校验，
            完整仿真预检由 validate_simulation_inputs 负责）。
        task: 仿真任务配置（球/拍/保护对名称与球初位姿来源）。
        racket_profile: 可选球拍档案（T2 新增）；None 保持旧 API 语义
            （模板默认拍 / scene.racket_* legacy 参数），提供时球拍几何/
            质量/惯量/安装全部来自 RacketProfile。

    Returns:
        带球场景组装结果（robot.parameters 已从模型 jnt_range 回填限位）。

    Raises:
        ValueError: task 与 profile 身份/关节序不一致、robot_id 不支持、
            球关节/geom 名与模板既有名称冲突（含同名非 free 关节的类型
            错误场景）、模板缺少地面/机身 geom、拍/保护对名称不存在、
            球关节编译后类型非 free、保护对两端同 geom、保护对掩码
            不相容（不可观测）或带球模型形状与描述不符。
    """
    _check_task_identity(profile, task)
    context = f"robot_id: {profile.robot_id}，模型: {profile.model_path}"
    if profile.robot_id != SUPPORTED_ROBOT_ID:
        raise ValueError(
            f"build_tennis_scene 当前仅支持 {SUPPORTED_ROBOT_ID!r} 的带球场景"
            f"组装，收到 robot_id={profile.robot_id!r}（{context}）"
        )

    def _mutate(root: ET.Element) -> None:
        """在模板副本上组装球、翻转碰撞掩码、追加球—拍弹性 pair。"""
        _assemble_ball_and_masks(root, profile, task)

    robot = _assemble_bound_model(
        profile,
        parameters,
        validate_shape=_validate_tennis_model_shape,
        mutate=_mutate,
        racket_profile=racket_profile,
    )
    model = robot.model

    # —— 编译后按名称重新绑定：全部 id 在本模型内解析，不复用其他模型 ——
    ball_joint_id = _resolve_id(
        model, mujoco.mjtObj.mjOBJ_JOINT, task.ball_joint_name, context
    )
    if int(model.jnt_type[ball_joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError(
            f"球关节 '{task.ball_joint_name}' 类型非 free joint"
            f"（jnt_type={int(model.jnt_type[ball_joint_id])}，{context}）"
        )
    ball_geom_id = _resolve_id(
        model, mujoco.mjtObj.mjOBJ_GEOM, task.ball_geom_name, context
    )
    racket_geom_ids = tuple(
        _resolve_id(model, mujoco.mjtObj.mjOBJ_GEOM, name, context)
        for name in task.racket_geom_names
    )

    protected: list[tuple[int, int]] = []
    for name1, name2 in task.protected_geom_pairs:
        g1 = _resolve_id(model, mujoco.mjtObj.mjOBJ_GEOM, name1, context)
        g2 = _resolve_id(model, mujoco.mjtObj.mjOBJ_GEOM, name2, context)
        if g1 == g2:
            raise ValueError(
                f"保护对 ({name1}, {name2}) 两端为同一 geom，无法构成"
                f"禁止接触对（{context}）"
            )
        if not _masks_allow_contact(model, g1, g2):
            raise ValueError(
                f"保护对 ({name1}, {name2}) 碰撞掩码不相容，禁止接触不可"
                f"被观测（掩码必须允许该对产生接触才能供 3.7 检测，{context}）"
            )
        protected.append((g1, g2))

    scene = TennisSceneBundle(
        robot=robot,
        ball_joint_id=ball_joint_id,
        ball_qpos_start=int(model.jnt_qposadr[ball_joint_id]),
        ball_qvel_start=int(model.jnt_dofadr[ball_joint_id]),
        ball_geom_id=ball_geom_id,
        racket_geom_ids=racket_geom_ids,
        protected_geom_pairs=tuple(protected),
    )
    logger.info(
        "R1D 带球场景组装完成: nq=%d nv=%d nu=%d 球关节=%s(qpos@%d) "
        "拍 geom=%d 个 保护对=%d 对（参数来源: %s）",
        model.nq,
        model.nv,
        model.nu,
        task.ball_joint_name,
        scene.ball_qpos_start,
        len(racket_geom_ids),
        len(protected),
        robot.parameters.source,
    )
    return scene


# ---------------------------------------------------------------- 组装辅助


def _check_task_identity(profile: RobotProfile, task: SimTaskConfig) -> None:
    """校验 task 与 profile 的身份/关节序一致（防御手构对象，同 3.1 规则）。

    Raises:
        ValueError: robot_id 或 joint_names 与 profile 不一致。
    """
    context = f"robot_id: {profile.robot_id}"
    if task.robot_id != profile.robot_id:
        raise ValueError(
            f"字段 robot_id 任务 '{task.robot_id}' 与 profile "
            f"'{profile.robot_id}' 不一致（{context}）"
        )
    if tuple(task.joint_names) != tuple(profile.active_joint_names):
        raise ValueError(
            f"字段 joint_names {list(task.joint_names)} 与 "
            f"profile.active_joint_names {list(profile.active_joint_names)} "
            f"不一致（须逐项同序，{context}）"
        )


def _assemble_ball_and_masks(
    root: ET.Element, profile: RobotProfile, task: SimTaskConfig
) -> None:
    """在模板副本上组装球 body、翻转碰撞掩码、追加球—拍弹性 pair。

    步骤：
    1. 名称冲突预检：模板不得已存在与球关节/球 geom/球 body 同名的元素
       （同名非 free 关节即"球 free joint 类型错误"负例，提前给出清晰
       错误而不是让编译器报重复名称）；
    2. 球 body 插入 worldbody 首位（free joint 位于主动关节之前），
       pos = task.initial_ball_position（进入 qpos0），物理参数取
       rm65_model.xml 逐字常量；
    3. 按 §碰撞掩码设计翻转 ground/torso_box/臂链 geom/拍 geom 掩码
       （环境链 mesh 等其余 geom 保持视觉掩码）；
    4. 为每个拍 geom 追加球—拍弹性 <pair>（rm65 L75 参数逐字）。

    Raises:
        ValueError: 模板漂移（缺 worldbody/地面/机身 geom）、球名称与模板
            既有元素冲突、拍 geom 名称不存在或承载主动关节的 body 数
            与关节数不符。
    """
    context = f"robot_id: {profile.robot_id}，模型: {profile.model_path}"
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"模板缺少 <worldbody>（模板漂移，{context}）")
    _ensure_ball_names_free(root, task, context)

    # 球 body：置于 worldbody 首位 → 球 free joint id/qpos 地址在主动关节
    # 之前，消费方必须按名称绑定（主动绑定不得含球 quaternion）
    ball_body = ET.Element(
        "body", name=_BALL_BODY_NAME, pos=_fmt_vec(task.initial_ball_position)
    )
    ET.SubElement(ball_body, "freejoint", name=task.ball_joint_name)
    ET.SubElement(
        ball_body,
        "geom",
        name=task.ball_geom_name,
        type="sphere",
        size=_fmt_scalar(_BALL_RADIUS),
        rgba=_BALL_RGBA,
        mass=_fmt_scalar(_BALL_MASS),
        contype=str(_BALL_CONTYPE),
        conaffinity=str(_BALL_CONAFFINITY),
    )
    worldbody.insert(0, ball_body)

    # 碰撞掩码：地面/机身/拍按名称定位，臂 geom 按主动关节承载 body 的
    # 子树定位（含末端拍 body，拍 geom 已按拍掩码配置故跳过）
    _apply_masks(_find_geom(root, _GROUND_GEOM, context), _MASK_GROUND)
    _apply_masks(_find_geom(root, _TORSO_GEOM, context), _MASK_TORSO)
    racket_names = set(task.racket_geom_names)
    for name in task.racket_geom_names:
        _apply_masks(_find_geom(root, name, context), _MASK_RACKET)
    for geom in _arm_subtree_geoms(root, profile, context):
        if geom.attrib.get("name") in racket_names:
            continue
        _apply_masks(geom, _MASK_ARM)

    # 球—拍弹性 pair（显式 pair 绕过位掩码且优先于动态碰撞参数）；
    # <contact> 不存在时创建（合成模板允许无排除表）
    contact = root.find("contact")
    if contact is None:
        contact = ET.SubElement(root, "contact")
    for name in task.racket_geom_names:
        ET.SubElement(
            contact,
            "pair",
            geom1=name,
            geom2=task.ball_geom_name,
            condim=_PAIR_CONDIM,
            solref=_PAIR_SOLREF,
            solimp=_PAIR_SOLIMP,
        )


def _ensure_ball_names_free(
    root: ET.Element, task: SimTaskConfig, context: str
) -> None:
    """预检球关节/球 geom/球 body 名称不与模板既有元素冲突。

    Raises:
        ValueError: 模板中已存在同名关节（含 freejoint；同名 hinge 即
            "球 free joint 类型错误"场景）、同名 geom 或同名 body。
    """
    for tag in ("joint", "freejoint"):
        for joint in root.iter(tag):
            if joint.attrib.get("name") == task.ball_joint_name:
                raise ValueError(
                    f"模板中已存在名为 '{task.ball_joint_name}' 的 <{tag}>"
                    f"关节（球 free joint 名称冲突/类型错误，{context}）"
                )
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == task.ball_geom_name:
            raise ValueError(
                f"模板中已存在名为 '{task.ball_geom_name}' 的 geom"
                f"（球 geom 名称冲突，{context}）"
            )
    for body in root.iter("body"):
        if body.attrib.get("name") == _BALL_BODY_NAME:
            raise ValueError(
                f"模板中已存在名为 '{_BALL_BODY_NAME}' 的 body"
                f"（球 body 名称冲突，{context}）"
            )


def _arm_subtree_geoms(
    root: ET.Element, profile: RobotProfile, context: str
) -> list[ET.Element]:
    """收集主动关节承载 body 及其全部后代 body 的 geom（臂链）。

    环境链等无主动关节的 body 不在其中（保持视觉掩码）。拍 body 的 geom
    由 ``task.racket_geom_names`` 单独配置，这里整体跳过——否则球拍的
    视觉拍面/拍柄会被误设为碰撞臂，产生非预期接触。承载 body 数必须
    等于主动关节数（一个 body 承载多个主动关节视为模板漂移）。

    Raises:
        ValueError: 承载主动关节的 body 数与 active_joint_names 数不符。
    """
    active = set(profile.active_joint_names)
    hosting = [
        body
        for body in root.iter("body")
        if any(joint.attrib.get("name") in active for joint in body.findall("joint"))
    ]
    if len(hosting) != len(profile.active_joint_names):
        raise ValueError(
            f"模板中承载主动关节的 body 数 {len(hosting)} 与关节数 "
            f"{len(profile.active_joint_names)} 不符（模板漂移，{context}）"
        )
    # 拍 body 的 geom 已按 racket_geom_names 配掩码，整棵子树在此跳过；
    # 否则新增的视觉拍面/拍柄会被当作碰撞臂，产生非预期接触。
    racket_body = next(
        (body for body in root.iter("body") if body.get("name") == "r1d_racket"),
        None,
    )
    racket_geom_ids = (
        {id(geom) for geom in racket_body.iter("geom")}
        if racket_body is not None
        else set()
    )
    geoms: list[ET.Element] = []
    seen: set[int] = set()
    for body in hosting:
        for geom in body.iter("geom"):
            if id(geom) in racket_geom_ids:
                continue
            if id(geom) not in seen:  # 臂链嵌套时去重（外层已含后代）
                seen.add(id(geom))
                geoms.append(geom)
    return geoms


def _validate_tennis_model_shape(
    model: mujoco.MjModel, profile: RobotProfile
) -> None:
    """校验带球模型形状：n 个铰链主动关节 + 恰 1 个球 free joint。

    期望 nq = n+7（球 free joint 7 项 qpos）、nv = n+6、nu = n、
    njnt = n+1；除球外全部关节必须为铰链（free/ball/slide 均拒绝）。

    Raises:
        ValueError: 维数不符、free joint 数不为 1 或存在其他非铰链关节。
    """
    n = profile.ndof
    free_ids = [
        jid
        for jid in range(model.njnt)
        if int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE)
    ]
    if model.njnt != n + 1 or len(free_ids) != 1:
        raise ValueError(
            f"带球模型应恰有 {n} 个主动关节 + 1 个球 free joint，实际 "
            f"njnt={model.njnt}, free 关节数={len(free_ids)}"
            f"（robot_id: {profile.robot_id}，模型: {profile.model_path}）"
        )
    for jid in range(model.njnt):
        if jid in free_ids:
            continue
        if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
            raise ValueError(
                f"带球模型存在非铰链主动关节 '{name}'（jnt_type="
                f"{int(model.jnt_type[jid])}，R1D 主动关节应为 hinge）"
                f"（robot_id: {profile.robot_id}，模型: {profile.model_path}）"
            )
    if model.nq != n + 7 or model.nv != n + 6 or model.nu != n:
        raise ValueError(
            f"带球模型维数与描述不符: nq={model.nq}, nv={model.nv}, "
            f"nu={model.nu}，期望 nq={n + 7}, nv={n + 6}, nu={n}"
            f"（robot_id: {profile.robot_id}，模型: {profile.model_path}）"
        )


def _masks_allow_contact(
    model: mujoco.MjModel, geom1: int, geom2: int
) -> bool:
    """按 MuJoCo 位掩码规则判断两 geom 是否允许产生动态接触。"""
    ct1, ca1 = int(model.geom_contype[geom1]), int(model.geom_conaffinity[geom1])
    ct2, ca2 = int(model.geom_contype[geom2]), int(model.geom_conaffinity[geom2])
    return (ct1 & ca2) != 0 or (ct2 & ca1) != 0


def _resolve_id(
    model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str, context: str
) -> int:
    """按名称解析 id，缺失即报错（名称缺失负例的统一出口）。

    Raises:
        ValueError: 模型中不存在该名称。
    """
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"模型中不存在名称 '{name}'（{context}）")
    return int(obj_id)


def _find_geom(root: ET.Element, name: str, context: str) -> ET.Element:
    """在元素树中按名称查找 geom，找不到即报模板漂移/名称缺失错误。

    Raises:
        ValueError: 树中不存在该名称的 geom。
    """
    for geom in root.iter("geom"):
        if geom.attrib.get("name") == name:
            return geom
    raise ValueError(f"模板中不存在 geom '{name}'（{context}）")


def _apply_masks(geom: ET.Element, masks: tuple[int, int]) -> None:
    """把 (contype, conaffinity) 掩码写入 geom 元素（内存副本）。"""
    geom.set("contype", str(masks[0]))
    geom.set("conaffinity", str(masks[1]))


def _fmt_scalar(value: float) -> str:
    """把标量格式化为 12 位有效数字字符串（与 model_factory 往返精度一致）。"""
    return f"{float(value):.12g}"


def _fmt_vec(values: Iterable[float]) -> str:
    """把数值序列格式化为空格分隔字符串（逐项 12 位有效数字）。"""
    return " ".join(_fmt_scalar(item) for item in values)
