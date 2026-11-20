#!/usr/bin/env python3
"""渲染挥拍 hero 实景图（论文 Fig.3）：整臂多帧残影 + 球轨迹残影 + 走廊管 + 窗口配色。

与 `render_ball_trail.py` 的分工：那个脚本出「一帧看全任务」的全景图（artifact），
本脚本出方法节 Fig.3 的近景——把「球沿预测轨迹飞来、手臂扫过去、恰好击中」拍成一张图，
同时按比例画出走廊（半径 0.12 m 的三维垂直容差）与候选窗口（±50 ms），
顶替旧的 `fig3_alt_2d.pdf` 二维示意图。

三个机制（前两条是本轮新增，第三条从 /tmp/ghost_poc3.py 移植）:
1) 整臂多帧残影：同一机位渲染多次（无机械臂背景一次 + 各帧位姿各一次），
   用「与背景的像素差」取该帧的机械臂掩膜，再按时间顺序做带羽化的 alpha 合成。
   合成只在掩膜内进行，所以背景/球场/走廊管（各帧完全一致）不会被反复混合。
   残影帧单独关阴影，避免地面出现多条互相打架的影子；主帧保留阴影。
2) 走廊管：wrapper 里加一个 capsule geom（半径 0.12 m），运行时按窗口弦的
   center/quat/half-length 摆放。用管而不是带，是因为代价罚的是三维垂直距离
   ‖P⊥(p_ee − p_ref)‖ —— 几何上是绕球线的圆柱，不是平面带。
3) 窗口配色：候选窗口（±50 ms，与 tube_builder 的 window_half_steps 同源）内
   的球残影用橙色、窗口外保持黄色（沿用旧 2D 图的「橙=窗内」约定）。

⚠ 旧的 `fig3_alt_2d` 把窗口画成「结束于 best candidate 的 50 ms」（只有后半个窗口），
   而求解器里是 ±50 ms 对称窗（21 槽，见 tube_builder/tube_types）——本脚本按对称窗
   绘制，题注据此改写。

用法（mujoco_tennis 环境，离屏渲染需 MUJOCO_GL=egl）:
    # 单张定稿（论文 Fig.3：3.387×3.0 in @600 dpi；默认机位/残影即为定稿值）
    MUJOCO_GL=egl python scripts/tools/render_swing_hero.py \
        --width 2032 --height 1800 --azimuth -42 --elevation -24 --distance 1.60 \
        --lookat -0.68 -0.40 0.98 --fovy 45 --corridor-alpha 0.14 \
        --out paper/figures/fig3_swing_hero_render.png

    # 机位调研：五机位 × (完整/素图) contact sheet
    MUJOCO_GL=egl python scripts/tools/render_swing_hero.py --sheet

输出：PNG + 同名 `.json` 附加文件（关键点像素坐标，供 matplotlib 端叠加矢量标记）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_ball_trail as rbt  # noqa: E402  (sys.path 先注入脚本目录，见上)

from src.utils.mujoco_loader import load_mujoco_model  # noqa: E402

MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"
R1D_MODEL_RIGHT = REPO_ROOT / "src" / "robot" / "r1d_model_right.xml"
R1D_MODEL_LEFT = REPO_ROOT / "src" / "robot" / "r1d_model_left.xml"

# 机器人身份 → 资源映射（T8c：hero 渲染不再写死 RM65）
ROBOT_ID_RM65 = "rm65"
ROBOT_ID_R1D = "unitree_r1d"
ROBOT_N_JOINTS: dict[str, int] = {ROBOT_ID_RM65: 6, ROBOT_ID_R1D: 7}
ROBOT_DEFAULT_MODEL: dict[str, Path] = {ROBOT_ID_RM65: MODEL_XML,
                                        ROBOT_ID_R1D: R1D_MODEL_RIGHT}
# 自动取景用的「肩部」body 名：RM65 右臂基座 / R1D 右肩 pitch link
ROBOT_SHOULDER_BODY: dict[str, str] = {ROBOT_ID_RM65: "r_base_link1",
                                       ROBOT_ID_R1D: "right_shoulder_pitch_Link"}
# 参与相机掩膜的「运动链」body 前缀：R1D 侧静态左臂/机身不参与残影
ROBOT_MOVING_BODY_PREFIXES: dict[str, tuple[str, ...]] = {
    ROBOT_ID_RM65: ("r_", "l_"),
    ROBOT_ID_R1D: ("right_", "r1d_racket"),
}

BALL_R = 0.033                       # 网球半径（与 rm65_model.xml 一致）
BALL_RGB = (0.98, 0.95, 0.25)        # 窗口外：黄（与全景图一致的「球」色）
WIN_RGB = (0.72, 0.27, 0.00)         # 窗口内：橙（#B84500；比 #D55E00 深两档，管内透出仍不发褐）
CORRIDOR_R = 0.12                    # 走廊半宽 = 拍半径（method.tex 式(2)）
CORRIDOR_RGBA = (0.0, 0.40, 0.64, 0.16)   # 走廊管色 = 论文调色板 #0072B2 加深一档（#0066A3）；
                                      # 叠在绿色场地上要偏蓝才读得出「管」，故比 Fig.1/4/5 的走廊色深一点
N_BALL = 320                         # 球残影池槽位（覆盖整条 episode 的 307 步）
HIDDEN = np.array([0.0, 0.0, -50.0])  # 隐藏槽位位置（地板下方、画面外）

# 机位预设：azimuth 为绕 Z 轴方位角（0 = 相机在 +x 侧），elevation 负值 = 俯视。
# lookat / distance / fovy 可由 CLI 覆盖，缺省时用 --auto-frame 自动取景。
VIEWS: dict[str, dict[str, float]] = {
    "x+": {"azimuth": 0.0, "elevation": -10.0},
    "x-": {"azimuth": 180.0, "elevation": -10.0},
    "y+": {"azimuth": 90.0, "elevation": -10.0},
    "y-": {"azimuth": 270.0, "elevation": -10.0},
    "iso": {"azimuth": 130.0, "elevation": -34.0},     # 用户 poc3_D 参考机位
    "top": {"azimuth": 90.0, "elevation": -70.0},
}


# =============================================================================
# 模型选择 / 回放归一化（T8c：默认行为 = 原 RM65 路径，R1D 不再 7→6 硬塞）
# =============================================================================

@dataclass(frozen=True)
class ModelSelection:
    """一次渲染选定的机器人模型（robot_id 决定 qpos 布局与掩膜 body 前缀）。"""

    robot_id: str
    model_xml: Path
    n_joints: int
    joint_names: tuple[str, ...] = ()

    @property
    def is_r1d(self) -> bool:
        """是否为单臂 7 关节 R1D 场景（qpos[0:7]，无 free joint 球）。"""
        return self.robot_id == ROBOT_ID_R1D


@dataclass
class HeroEpisode:
    """归一化后的回放 episode（RM65 exp18 格式与 R1D v12 格式统一入口）。"""

    source: Path
    robot_id: str
    n_joints: int
    q_actual: np.ndarray
    tcp_pos: np.ndarray
    ball_pos: np.ndarray
    dt: float
    hit_step: int
    metadata: dict
    model_selection: ModelSelection
    init_q_left: np.ndarray
    hit_step_source: str
    # v12_episode 记录的是场景世界坐标：模型根在原点时需减去这一平移才能对齐
    world_offset: np.ndarray | None = None

    def as_record(self) -> dict:
        """转成 `render_variant` 使用的 record 字典（保持旧字段名）。"""
        return {
            "ball_pos": self.ball_pos,
            "tcp_pos": self.tcp_pos,
            "q_actual": self.q_actual,
            "init_q_left": self.init_q_left,
            "hit_step": self.hit_step,
            "dt": self.dt,
            "metadata": self.metadata,
            "robot_id": self.robot_id,
            "n_joints": self.n_joints,
            "model_xml": self.model_selection.model_xml,
            "source": self.source,
            "world_offset": self.world_offset,
        }


def _load_robot_config(robot_config: Path) -> dict:
    """读取机器人描述 YAML（robot_id / model_path / active_joint_names）。"""
    cfg_path = Path(robot_config)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"机器人配置不存在: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"机器人配置不是 YAML 映射: {cfg_path}")
    return raw


def _resolve_model_file(model_path: str | Path, base: Path) -> Path:
    """解析 model_path：先相对配置文件目录，再退回 `src/robot/<basename>`。"""
    p = Path(model_path)
    if p.is_absolute():
        return p.resolve()
    candidates = [base / p, REPO_ROOT / "src" / "robot" / p.name]
    for cand in candidates:
        if cand.is_file():
            return cand.resolve()
    return candidates[0].resolve()


def resolve_model_selection(*, model_xml: Path | None = None,
                            robot_config: Path | None = None,
                            metadata: dict | None = None) -> ModelSelection:
    """按「显式 --model-xml > --robot-config > NPZ metadata > 默认 RM65」选模型。

    Args:
        model_xml: 显式模型 XML（最高优先级；不存在则报错）。
        robot_config: `configs/robots/*.yaml` 机器人描述（读 robot_id/model_path）。
        metadata: NPZ 内嵌或同名 JSON sidecar 的元数据（可含 robot_id/joint_names）。

    Returns:
        ModelSelection（含 robot_id / model_xml / n_joints）。

    Raises:
        FileNotFoundError: 显式路径或配置里的模型文件不存在。
        ValueError: robot_id 不在支持列表（拒绝把未知 7 关节模型按 6 关节渲染）。
    """
    md = dict(metadata or {})
    meta_robot_id = md.get("robot_id")
    meta_joint_names = tuple(md.get("joint_names") or ())
    if meta_robot_id is not None and meta_robot_id not in ROBOT_N_JOINTS:
        raise ValueError(
            f"不支持的 robot_id: {meta_robot_id}（仅支持 {sorted(ROBOT_N_JOINTS)}）")

    # 1) 显式 --model-xml 优先
    if model_xml is not None:
        path = Path(model_xml).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"模型 XML 不存在: {path}")
        rid = meta_robot_id or (ROBOT_ID_R1D if "r1d" in path.stem.lower()
                                else ROBOT_ID_RM65)
        return ModelSelection(rid, path, ROBOT_N_JOINTS[rid], meta_joint_names)

    # 2) --robot-config
    if robot_config is not None:
        cfg = _load_robot_config(Path(robot_config))
        rid = cfg.get("robot_id") or meta_robot_id
        if rid is not None and rid not in ROBOT_N_JOINTS:
            raise ValueError(
                f"不支持的 robot_id: {rid}（仅支持 {sorted(ROBOT_N_JOINTS)}）")
        names = tuple(cfg.get("active_joint_names") or meta_joint_names)
        raw_model = cfg.get("model_path")
        if raw_model is None:
            rid = rid or ROBOT_ID_RM65
            return ModelSelection(rid, ROBOT_DEFAULT_MODEL[rid],
                                  ROBOT_N_JOINTS[rid], names)
        path = _resolve_model_file(
            raw_model, Path(robot_config).resolve().parent)
        if not path.is_file():
            raise FileNotFoundError(f"模型 XML 不存在: {path}")
        rid = rid or (ROBOT_ID_R1D if "r1d" in path.stem.lower()
                      else ROBOT_ID_RM65)
        return ModelSelection(rid, path, ROBOT_N_JOINTS[rid], names)

    # 3) 仅 metadata / 缺省
    rid = meta_robot_id or ROBOT_ID_RM65
    return ModelSelection(rid, ROBOT_DEFAULT_MODEL[rid],
                          ROBOT_N_JOINTS[rid], meta_joint_names)


def resolve_output_path(path: str | Path) -> Path:
    """解析渲染输出路径：相对路径相对仓库根；拒绝写入 `paper/`。"""
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    p = p.resolve()
    paper_dir = (REPO_ROOT / "paper").resolve()
    if p == paper_dir or paper_dir in p.parents:
        raise ValueError(
            f"hero 渲染输出不得写入 paper/（论文图由 paper_figs.py 统一产出）：{p}")
    return p


def _read_sidecar_metadata(npz_path: Path) -> dict:
    """读取与 NPZ 同名的 JSON sidecar（v12_episode.json / r1d_episode.json）。

    T7 的 `v12_episode.npz` 本身没有 metadata，但同目录 `v12_episode.json`
    带 robot_id / joint_names / dt / hit.physical_hit_step，足以选模型与 k_ref。
    """
    side = npz_path.with_suffix(".json")
    if not side.is_file():
        return {}
    try:
        raw = json.loads(side.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    md: dict = {}
    for key in ("robot_id", "joint_names", "dt", "seed", "mode"):
        if raw.get(key) is not None:
            md[key] = raw[key]
    hit = raw.get("hit") if isinstance(raw.get("hit"), dict) else {}
    step = hit.get("physical_hit_step")
    if step is None:
        step = hit.get("legacy_hit_step")
    if step is not None and int(step) >= 0:
        md["hit_step"] = int(step)
    elif hit:
        # v12 产物里 miss 记为 hit_step=-1 / physical_hit=false；没有可渲染的接触帧
        md["_no_hit"] = not (hit.get("physical_hit") or hit.get("legacy_hit"))
    md["_sidecar"] = side.name
    return md


def _dt_from_time_arrays(npz, keys: set[str]) -> float:
    """从 `time` / `timestamps` 数组反推采样间隔（R1D v12 NPZ 无 dt 标量）。"""
    for name in ("time", "timestamps"):
        if name in keys:
            t = np.asarray(npz[name], dtype=float).ravel()
            if t.size >= 2:
                return float(np.median(np.diff(t)))
    raise ValueError("回放 NPZ 缺少 dt / time：无法确定采样间隔")


def _infer_hit_step(tcp_pos: np.ndarray, ball_pos: np.ndarray) -> int:
    """无记录 hit_step 时，用「拍心—球心最近距离」反推接触步。"""
    d = np.linalg.norm(np.asarray(tcp_pos, float) - np.asarray(ball_pos, float),
                       axis=1)
    return int(np.argmin(d))


def _compute_world_offset(selection: ModelSelection, q_hit: np.ndarray,
                          tcp_hit: np.ndarray) -> np.ndarray | None:
    """计算回放世界系与模型根系的平移（R1D v12 回放记录的是场景世界坐标）。

    模型 XML 只有机器人本体、根在原点；而 r1d_runner 产物里的关节/拍心轨迹都
    在场景世界系（含底座平移）。用同一 hit 步的 FK 拍心与记录拍心之差作为常量
    平移，渲染前从球/拍心轨迹里减掉，才能让球轨迹与机械臂对齐。
    """
    if not selection.is_r1d:
        return None
    try:
        model = load_mujoco_model(selection.model_xml)
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        data.qpos[:min(len(q_hit), model.nq)] = np.asarray(q_hit, float)[:model.nq]
        mujoco.mj_forward(model, data)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
        if site_id < 0:
            return None
        offset = np.asarray(tcp_hit, float) - np.asarray(data.site_xpos[site_id], float)
        print(f"[hero] 世界系对齐：模型根→场景平移 {np.round(offset, 3).tolist()} m")
        return offset
    except Exception as exc:  # 对齐失败不阻断渲染，只是画面可能错位
        print(f"[hero] 世界系对齐失败（按模型根系数值渲染）: {exc}")
        return None


def load_episode(path: str | Path, *, robot_config: Path | None = None,
                 model_xml: Path | None = None) -> HeroEpisode:
    """加载并归一化回放 NPZ（RM65 exp18 格式或 R1D v12 格式）。

    两种格式差异：
    - RM65（exp18）：q_actual(N,6)/tcp_pos(N,3)/ball_pos(N,3)/dt/hit_step/metadata；
      模型自带 free-joint 球，`rbt.set_pose` 直接写 qpos[0:19]。
    - R1D（v12）：X(N,14)=q(7)+qdot(7)/U(N-1,7)/ball_pos/ball_vel/time/racket_pos；
      无 free joint 球，7 关节写 qpos[0:7]，球用 wrapper 的 hero_ball geom 表示。

    Args:
        path: 回放 NPZ 路径。
        robot_config: 显式机器人描述（NPZ 无 metadata 时用于选模型）。
        model_xml: 显式模型 XML（最高优先级）。

    Returns:
        HeroEpisode。

    Raises:
        FileNotFoundError: NPZ 不存在。
        ValueError: 缺关键字段（R1D 缺 racket_pos / X）、关节数与模型不匹配。
    """
    npz_path = Path(path).resolve()
    if not npz_path.is_file():
        raise FileNotFoundError(f"回放 NPZ 不存在: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as npz:
        keys = set(npz.files)
        if "metadata" in keys:
            metadata = json.loads(str(npz["metadata"]))
            if not isinstance(metadata, dict):
                raise ValueError(f"回放 NPZ metadata 不是 JSON 对象: {npz_path}")
        else:
            metadata = _read_sidecar_metadata(npz_path)
        if metadata.get("_no_hit"):
            raise ValueError(
                f"回放无命中（miss）：{npz_path} 的 sidecar 记录 physical/legacy hit "
                "均为 false，没有可渲染的接触帧；hero 渲染请换命中 episode")
        selection = resolve_model_selection(model_xml=model_xml,
                                            robot_config=robot_config,
                                            metadata=metadata or None)
        if "ball_pos" not in keys:
            raise ValueError(f"回放 NPZ 缺少 ball_pos: {npz_path}")
        ball_pos = np.asarray(npz["ball_pos"], dtype=float)
        if ball_pos.ndim != 2 or ball_pos.shape[1] != 3:
            raise ValueError(f"回放 NPZ ball_pos 形状应为 (N,3): {ball_pos.shape}")

        if selection.is_r1d:
            if "racket_pos" not in keys:
                raise ValueError(
                    f"R1D 回放 NPZ 缺少 racket_pos（拍心轨迹）：{npz_path}；"
                    "请改用 v12_episode.npz（含 racket_pos），或在 r1d_runner "
                    "产物中补拍心轨迹")
            if "X" not in keys:
                raise ValueError(f"R1D 回放 NPZ 缺少 X（关节轨迹）: {npz_path}")
            q_actual = np.asarray(npz["X"], dtype=float)[:, :selection.n_joints]
            tcp_pos = np.asarray(npz["racket_pos"], dtype=float)
            init_q_left = np.zeros(0, dtype=float)
            dt = float(npz["dt"]) if "dt" in keys else _dt_from_time_arrays(npz, keys)
        else:
            for req in ("q_actual", "tcp_pos"):
                if req not in keys:
                    raise ValueError(f"RM65 回放 NPZ 缺少 {req}: {npz_path}")
            q_actual = np.asarray(npz["q_actual"], dtype=float)
            if q_actual.ndim != 2 or q_actual.shape[1] != selection.n_joints:
                raise ValueError(
                    f"关节数与模型不匹配：NPZ q_actual 形状 {q_actual.shape}，"
                    f"而 robot_id={selection.robot_id} 期望 {selection.n_joints} 关节；"
                    "R1D 回放请用 --robot-config/--model-xml 指定 7 关节模型")
            tcp_pos = np.asarray(npz["tcp_pos"], dtype=float)
            init_q_left = (np.asarray(npz["init_q_left"], dtype=float)
                           if "init_q_left" in keys else np.zeros(6))
            dt = float(npz["dt"]) if "dt" in keys else _dt_from_time_arrays(npz, keys)

        # hit_step 标量只在 RM65（exp18）格式里出现；R1D 由 metadata/最近距离反推
        hit_step_npz = (int(npz["hit_step"])
                        if (not selection.is_r1d and "hit_step" in keys) else None)

    if len(q_actual) != len(ball_pos) or len(tcp_pos) != len(ball_pos):
        raise ValueError(
            f"回放 NPZ 各轨迹长度不一致: q={len(q_actual)} tcp={len(tcp_pos)} "
            f"ball={len(ball_pos)}")

    # 记录步：metadata（含 sidecar）优先 → NPZ hit_step 标量 → 最近距离反推
    hit_step: int | None = None
    hit_source = ""
    if metadata.get("hit_step") is not None:
        hit_step, hit_source = int(metadata["hit_step"]), "metadata.hit_step"
    elif hit_step_npz is not None:
        hit_step, hit_source = hit_step_npz, "npz.hit_step"
    if hit_step is None:
        hit_step, hit_source = _infer_hit_step(tcp_pos, ball_pos), "argmin|tcp-ball|"
    hit_step = int(np.clip(hit_step, 0, len(ball_pos) - 1))

    world_offset = _compute_world_offset(selection, q_actual[hit_step], tcp_pos[hit_step])

    return HeroEpisode(
        source=npz_path, robot_id=selection.robot_id, n_joints=selection.n_joints,
        q_actual=q_actual, tcp_pos=tcp_pos, ball_pos=ball_pos, dt=float(dt),
        hit_step=hit_step, metadata=dict(metadata or {}),
        model_selection=selection, init_q_left=init_q_left,
        hit_step_source=hit_source, world_offset=world_offset)


def resolve_k_ref(ep: HeroEpisode) -> tuple[int, str]:
    """解析参考步 k_ref（候选窗口中心）：metadata 优先，其次预测，最后 fallback。"""
    md = ep.metadata or {}
    if md.get("k_hit") is not None:
        return int(md["k_hit"]), "metadata.k_hit"
    if md.get("hit_step") is not None:
        return int(md["hit_step"]), "metadata.hit_step"
    if (not ep.model_selection.is_r1d and md.get("p0") is not None
            and md.get("v0") is not None):
        try:
            from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
            from src.sim.rm65_env import RM65Env
            from src.tennis.hitting import find_hitting_point_physics
            env = RM65Env(MODEL_XML)
            hi = find_hitting_point_physics(
                env, np.asarray(md["p0"], dtype=float),
                np.asarray(md["v0"], dtype=float), SHOULDER_POS,
                WORKSPACE_RADIUS, len(ep.ball_pos))
            if hi is not None:
                return int(hi["k_hit"]), "predict.find_hitting_point_physics"
        except Exception as exc:  # 预测失败不阻断渲染，明确回落并打印
            print(f"[hero] k_ref 预测失败（{type(exc).__name__}: {exc}），回退记录 hit_step")
    return ep.hit_step, (
        f"fallback: 记录 hit_step={ep.hit_step}（来源 {ep.hit_step_source}，"
        "metadata 无 k_hit）")


def make_pose_fn(data: mujoco.MjData, model: mujoco.MjModel, *,
                 robot_id: str, q: np.ndarray, q_left: np.ndarray,
                 ball_pos: np.ndarray, ball_geom_id: int):
    """构造「应用某帧位姿与球位」的闭包（RM65 / R1D 两套 qpos 布局）。"""
    q_arr = np.asarray(q, dtype=float)
    if robot_id == ROBOT_ID_RM65:
        q_left_arr = np.asarray(q_left, dtype=float)

        def _apply_rm65() -> None:
            rbt.set_pose(data, q_arr, q_left_arr, np.asarray(ball_pos, dtype=float))
        return _apply_rm65

    def _apply_r1d() -> None:
        data.qpos[:] = 0.0
        data.qpos[:q_arr.shape[0]] = q_arr
        if ball_geom_id >= 0:
            model.geom_pos[ball_geom_id] = np.asarray(ball_pos, dtype=float)
        mujoco.mj_forward(model, data)
    return _apply_r1d


def build_model(ball_scale: float, fovy: float, w: int, h: int,
                corridor: tuple[np.ndarray, np.ndarray],
                model_xml: Path | None = None) -> mujoco.MjModel:
    """构建含球残影池 + 走廊管 + hero_ball 的包装模型（include 指定机器人 XML）。

    池内球默认隐藏（位置在地板下方 + alpha=0），运行时改位姿与透明度。
    ⚠ 槽位数必须一次建好：同一个 Renderer 换 ngeom 不同的模型会静默丢 geom。
    ⚠ 走廊管必须用 `fromto` 在编译期取向：实测本机 MuJoCo 忽略运行时对
    `model.geom_quat` 的修改（mj_forward 后 data.geom_xmat 仍是编译期值），
    而 geom_pos/geom_rgba 的运行时修改有效——故弦端点直接在 XML 里烘死。
    ⚠ 用 `xml_text=` 在同目录编译：include 只写文件名（MuJoCo C 层打不开
    含中文的绝对路径），非 ASCII 项目由 load_mujoco_model 自动落到 ASCII 副本。
    """
    xml = Path(model_xml) if model_xml is not None else MODEL_XML
    hide = f'pos="{HIDDEN[0]} {HIDDEN[1]} {HIDDEN[2]}" rgba="1 1 1 0"'
    ghosts = "".join(
        f'    <geom name="ghost_{k}" type="sphere" size="{BALL_R * ball_scale:.4f}" '
        f'{hide} contype="0" conaffinity="0"/>\n' for k in range(N_BALL))
    c0, c1 = corridor
    corridor_xml = (
        f'    <geom name="corridor_tube" type="capsule" size="{CORRIDOR_R}" '
        f'fromto="{c0[0]:.6f} {c0[1]:.6f} {c0[2]:.6f} {c1[0]:.6f} {c1[1]:.6f} '
        f'{c1[2]:.6f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>\n')
    # R1D 场景无 free-joint 球，接触时刻的实心球由这个 wrapper geom 承担；
    # RM65 场景保持由模型自带球渲染，hero_ball 仅占位（alpha=0）。
    hero_ball_xml = (
        f'    <geom name="hero_ball" type="sphere" size="{BALL_R * ball_scale:.4f}" '
        f'{hide} contype="0" conaffinity="0"/>\n')
    wrapper = (
        "<mujoco>\n"
        f'  <include file="{xml.name}"/>\n'
        "  <visual>\n"
        f'    <global offwidth="{w}" offheight="{h}" fovy="{fovy}"/>\n'
        '    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7" '
        'specular="0.15 0.15 0.15"/>\n'
        "  </visual>\n"
        "  <worldbody>\n"
        '    <light name="key_top" pos="0.3 -1.2 3.2" dir="-0.08 0.32 -1" '
        'directional="true" diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2" '
        'castshadow="false"/>\n'
        f"{ghosts}{corridor_xml}{hero_ball_xml}"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    return load_mujoco_model(xml, xml_text=wrapper)


def moving_body_geoms(model: mujoco.MjModel,
                      robot_id: str = ROBOT_ID_RM65) -> list[int]:
    """机械臂运动链 geom——取掩膜时要把它们藏起来（按 robot_id 分前缀）。"""
    prefixes = ROBOT_MOVING_BODY_PREFIXES.get(robot_id,
                                              ROBOT_MOVING_BODY_PREFIXES[ROBOT_ID_RM65])
    ids = []
    for g in range(model.ngeom):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[g]) or ""
        if bn.startswith(prefixes):
            ids.append(g)
    return ids


def shoulder_body_id(model: mujoco.MjModel, robot_id: str) -> int:
    """自动取景用的肩部 body id（RM65 r_base_link1 / R1D right_shoulder_pitch_Link）。"""
    name = ROBOT_SHOULDER_BODY.get(robot_id, ROBOT_SHOULDER_BODY[ROBOT_ID_RM65])
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise ValueError(f"模型缺少肩部 body: {name}（robot_id={robot_id}）")
    return bid


def fill_ball_pool(model: mujoco.MjModel, ids: list[int], ball: np.ndarray,
                   steps: np.ndarray, hit_step: int, *, strobe_steps: int,
                   strobe_alpha: float, ribbon_alpha: float,
                   win: tuple[int, int],
                   win_rgb: tuple[float, float, float]) -> tuple[int, int]:
    """把画面内的球位写进残影池（跳过主帧的实心球位）；窗口内橙、窗口外黄。"""
    slot, n_strobe = 0, 0
    for i in steps:
        if slot >= N_BALL:
            break
        if i == hit_step:            # 该步有实心真球，跳过以免 z-fighting
            continue
        if (i - hit_step) % strobe_steps == 0:
            alpha, n_strobe = strobe_alpha, n_strobe + 1
        else:
            alpha = ribbon_alpha
        if alpha <= 0:
            continue
        rgb = win_rgb if win[0] <= i <= win[1] else BALL_RGB
        if win[0] <= i <= win[1]:
            alpha = max(alpha, 0.55)     # 窗内球提亮：走廊管会压掉一部分对比度
        model.geom_pos[ids[slot]] = ball[i]
        model.geom_rgba[ids[slot]] = [*rgb, alpha]
        slot += 1
    return slot, n_strobe


def predicted_line(ball: np.ndarray, hit_step: int, dt: float, steps: np.ndarray) -> np.ndarray:
    """按「接触点 + 有限差分速度 + 重力」外推自由飞行线（球线 / 走廊轴）。

    记录轨迹在接触前就是这条抛物线（模型无空气阻力），所以击球前的实际球位与
    外推线逐点重合；击球后真球被拍面打飞，但规划器看到的仍是这条线的延拓——
    候选窗口（±50 ms）的后半段正是落在这条延拓线上，故窗口段按此线绘制。
    """
    i = int(np.clip(hit_step - 4, 0, len(ball) - 1))
    if hit_step - i <= 0:      # 接触帧贴着回放起点：无前段轨迹可差分，退化为静止
        v = np.zeros(3)
    else:
        v = (ball[hit_step] - ball[i]) / ((hit_step - i) * dt)
    g = np.array([0.0, 0.0, -9.81])
    tau = (np.asarray(steps, dtype=float) - hit_step) * dt
    return ball[hit_step][None] + v[None] * tau[:, None] + 0.5 * g[None] * tau[:, None] ** 2


def trail_positions(ball: np.ndarray, line: np.ndarray, hit_step: int,
                    steps: np.ndarray) -> np.ndarray:
    """残影球位：击球前用记录位（含真实弹跳），击球后用预测线延拓。"""
    return np.where((np.asarray(steps) <= hit_step)[:, None], ball[np.asarray(steps)],
                    line)


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData,
                 cam: mujoco.MjvCamera, opt: mujoco.MjvOption,
                 apply_pose, *, shadows: bool) -> np.ndarray:
    """渲染一帧（apply_pose 负责写位姿/球位，兼容 RM65 与 R1D 两套 qpos 布局）。"""
    apply_pose()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
    return renderer.render()


def base_frame(renderer: mujoco.Renderer, data: mujoco.MjData,
               cam: mujoco.MjvCamera, opt: mujoco.MjvOption,
               model: mujoco.MjModel, hide_ids: list[int],
               apply_pose) -> np.ndarray:
    """无机械臂背景帧（用于取掩膜）：临时把机械臂 geom 的 alpha 置零。"""
    saved = model.geom_rgba[hide_ids].copy()
    model.geom_rgba[hide_ids, 3] = 0.0
    img = render_frame(renderer, data, cam, opt, apply_pose, shadows=False)
    model.geom_rgba[hide_ids] = saved
    return img


def mask_from_diff(fg: np.ndarray, bg: np.ndarray, thresh: int = 10,
                   feather: float = 1.2, grow: int = 1) -> np.ndarray:
    """由「与背景的像素差」取掩膜（膨胀 + 高斯羽化，返回 0..1 float 图）。"""
    diff = np.abs(fg.astype(np.int16) - bg.astype(np.int16)).max(axis=2)
    m = Image.fromarray(((diff > thresh) * 255).astype(np.uint8))
    if grow:
        m = m.filter(ImageFilter.MaxFilter(2 * grow + 1))
    if feather:
        m = m.filter(ImageFilter.GaussianBlur(feather))
    return np.asarray(m, dtype=float) / 255.0


def composite_ghosts(base: np.ndarray,
                     frames: list[tuple[np.ndarray, np.ndarray, float]]) -> np.ndarray:
    """按时间先后把 (帧, 掩膜, alpha) 合成到 base 上（只在掩膜内混合）。"""
    out = base.astype(float).copy()
    for img, mask, alpha in frames:
        a = (mask * alpha)[..., None]
        out = out * (1.0 - a) + img.astype(float) * a
    return np.clip(out, 0, 255).astype(np.uint8)


def auto_lookat_n_distance(ball: np.ndarray, tcp: np.ndarray, k_ref: int,
                           win: tuple[int, int], shoulder: np.ndarray,
                           hit_step: int, fovy: float, aspect: float) -> tuple[np.ndarray, float]:
    """自动取景：注视点取肩部与接触点的加权中点，距离按「关键点到注视点的最大跨度」定。"""
    contact = ball[k_ref]
    lookat = 0.45 * shoulder + 0.55 * contact
    key_pts = np.vstack([
        ball[win[0]:win[1] + 1], tcp[max(0, hit_step - 40):hit_step + 20], shoulder[None],
        contact[None],
    ])
    d = np.linalg.norm(key_pts - lookat, axis=1)
    radius = float(np.percentile(d, 95))          # 用 95 分位，避免个别远点把画面拉太远
    tan_v = np.tan(np.deg2rad(fovy) / 2)
    dist = max(radius / tan_v * 1.25, 0.8)
    return lookat, dist


def render_variant(*, args: argparse.Namespace, rec: dict, k_ref: int, win: tuple[int, int],
                   view: str, out_png: Path | None,
                   arm_ghosts: list[tuple[int, float]], with_corridor: bool,
                   with_line: bool, detail: bool = False) -> tuple[np.ndarray, dict]:
    """渲染一个（机位 × 配置）组合，返回 (图像, 关键点像素信息)。"""
    ball = np.asarray(rec["ball_pos"])
    tcp = np.asarray(rec["tcp_pos"])
    q = np.asarray(rec["q_actual"])
    q_left = np.asarray(rec["init_q_left"]) if rec.get("init_q_left") is not None \
        else np.zeros(0)
    hit_step = int(rec["hit_step"])
    n, dt = len(ball), float(rec["dt"])
    robot_id = str(rec.get("robot_id") or ROBOT_ID_RM65)
    model_xml = rec.get("model_xml") or ROBOT_DEFAULT_MODEL.get(robot_id, MODEL_XML)
    preset = VIEWS[view]

    # R1D v12 回放记录的是场景世界坐标（含底座平移）；模型根在原点，先对齐到根系
    world_offset = rec.get("world_offset")
    if world_offset is not None:
        offset = np.asarray(world_offset, dtype=float)
        if np.abs(offset).max() > 1e-9:
            ball = ball - offset
            tcp = tcp - offset

    # 走廊轴（窗口弦的直线近似）与残影球位都要先算出来：管端点需在建模时烘进 XML
    line = predicted_line(ball, hit_step, dt, np.arange(n))
    trail = trail_positions(ball, line, hit_step, np.arange(n))
    chord = (line[win[0]], line[win[1]])

    model = build_model(args.ball_scale, args.fovy or preset.get("fovy", 40.0),
                        args.width, args.height, chord, model_xml=model_xml)
    palette = None if args.palette in ("original", "none") else args.palette
    for _line in rbt.apply_palette(model, palette):
        print("      " + _line)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    ball_ids = [g for g, nm in enumerate(names) if nm and nm.startswith("ghost_")]
    tube_id = next(g for g, nm in enumerate(names) if nm == "corridor_tube")
    hero_ball_id = next((g for g, nm in enumerate(names) if nm == "hero_ball"), -1)
    hide_ids = moving_body_geoms(model, robot_id)
    # R1D 场景没有 free-joint 球，实心球用 hero_ball；RM65 仍由模型自带球渲染
    model.geom_rgba[hero_ball_id] = [*BALL_RGB, 0.0 if robot_id == ROBOT_ID_RM65 else 1.0]

    # 自动取景（肩部位置需要先摆好姿态再读 xpos）
    shoulder_id = shoulder_body_id(model, robot_id)
    make_pose_fn(data, model, robot_id=robot_id, q=q[hit_step], q_left=q_left,
                 ball_pos=ball[hit_step], ball_geom_id=hero_ball_id)()
    shoulder = np.array(data.xpos[shoulder_id], dtype=float)
    fovy = args.fovy or preset.get("fovy", 40.0)
    lookat, auto_dist = auto_lookat_n_distance(ball, tcp, k_ref, win, shoulder,
                                               hit_step, fovy, args.width / args.height)
    if args.lookat is not None:
        lookat = np.array(args.lookat, dtype=float)
    distance = args.distance if args.distance is not None else auto_dist
    cam = mujoco.MjvCamera()
    rbt.configure_camera(cam, lookat, distance,
                         args.azimuth if args.azimuth is not None else preset["azimuth"],
                         args.elevation if args.elevation is not None else preset["elevation"])
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0
    opt.sitegroup[0] = 0

    # ── 球残影池 + 画面内步 ──
    renderer.update_scene(data, camera=cam, scene_option=opt)
    proj = rbt.Projector(model, renderer.scene.camera[0], w=args.width, h=args.height)
    px, ok = proj(trail)
    inside = (ok & (px[:, 0] > -60) & (px[:, 0] < args.width + 60)
              & (px[:, 1] > -60) & (px[:, 1] < args.height + 60))
    steps = np.flatnonzero(inside)
    strobe_steps = max(1, round(args.strobe_ms / (dt * 1000.0)))
    slot, n_strobe = fill_ball_pool(
        model, ball_ids, trail, steps, hit_step, strobe_steps=strobe_steps,
        strobe_alpha=args.strobe_alpha, ribbon_alpha=args.ribbon_alpha, win=win,
        win_rgb=tuple(args.win_rgb))

    # 走廊管只切颜色/透明度（取向已在建模时烘死，见 build_model 注释）
    model.geom_rgba[tube_id] = [*args.corridor_rgb,
                                args.corridor_alpha if with_corridor else 0.0]

    def _apply(frame_q: np.ndarray, frame_ball: np.ndarray):
        """固定接触帧球位（残影帧的球不参与掩膜），只换手臂位姿。"""
        return make_pose_fn(data, model, robot_id=robot_id, q=frame_q, q_left=q_left,
                            ball_pos=frame_ball, ball_geom_id=hero_ball_id)

    # ── 渲染：背景（无机械臂）→ 残影帧（无阴影）→ 主帧（接触位姿，带阴影） ──
    bg = base_frame(renderer, data, cam, opt, model, hide_ids,
                    _apply(q[hit_step], ball[hit_step]))
    # 接触帧的机械臂轮廓：残影一律不得覆盖它。否则主帧手臂被逐帧 alpha 反复稀释、
    # 看起来半透明（3 帧时已可见，加到 5 帧更明显）。grow=2 把边缘描边也一起盖住。
    main_ns = render_frame(renderer, data, cam, opt,
                           _apply(q[hit_step], ball[hit_step]), shadows=False)
    silhouette = mask_from_diff(main_ns, bg, thresh=args.mask_thresh,
                                feather=args.mask_feather, grow=2)
    ghost_frames = []
    for off, alpha in arm_ghosts:
        s = int(np.clip(hit_step + off, 0, n - 1))
        # 残影帧的球位固定为主帧球位：真球不参与掩膜，否则会被当成残影糊上
        f = render_frame(renderer, data, cam, opt,
                         _apply(q[s], ball[hit_step]), shadows=False)
        m = mask_from_diff(f, bg, thresh=args.mask_thresh,
                           feather=args.mask_feather) * (1.0 - silhouette)
        ghost_frames.append((f, m, alpha))
    img = render_frame(renderer, data, cam, opt,
                       _apply(q[hit_step], ball[hit_step]),
                       shadows=not args.no_shadows)
    img = composite_ghosts(img, ghost_frames)

    if with_line and args.path_line:
        img = rbt.draw_path_line(img, px, inside, width=args.path_line_width,
                                 alpha=args.path_line_alpha)
    if args.brighten != 1.0:
        from PIL import ImageEnhance
        img = np.asarray(ImageEnhance.Brightness(Image.fromarray(img)).enhance(args.brighten))
    if args.crop_top or args.crop_bottom:
        img = img[args.crop_top: args.height - args.crop_bottom]

    # ── 关键点投影（供 matplotlib 端叠加矢量标记） ──
    make_pose_fn(data, model, robot_id=robot_id, q=q[hit_step], q_left=q_left,
                 ball_pos=ball[hit_step], ball_geom_id=hero_ball_id)()
    renderer.update_scene(data, camera=cam, scene_option=opt)
    proj2 = rbt.Projector(model, renderer.scene.camera[0], w=img.shape[1], h=img.shape[0])
    keys = {
        "reference_ball": line[k_ref],       # 预测线上 best candidate 步的位置
        "contact_ball": ball[hit_step],      # 实际接触点（记录值）
        "contact_tcp": tcp[hit_step],        # 接触时刻拍面中心
        "window_start": line[win[0]],
        "window_end": line[win[1]],
        "shoulder": shoulder,
    }
    pxk, okk = proj2(np.array(list(keys.values())))
    md = rec.get("metadata") if isinstance(rec.get("metadata"), dict) else {}
    info = {
        "view": view, "azimuth": float(cam.azimuth), "elevation": float(cam.elevation),
        "distance": float(distance), "fovy": float(fovy), "lookat": lookat.tolist(),
        "width": int(img.shape[1]), "height": int(img.shape[0]),
        "robot_id": robot_id, "n_joints": int(q.shape[1]),
        "model_xml": str(model_xml),
        "k_ref": int(k_ref), "hit_step": hit_step, "window": [int(win[0]), int(win[1])],
        "ball_speed": md.get("ball_speed"),
        "n_ghost_slots": slot, "n_strobe": n_strobe,
        "arm_ghosts": [[int(o), float(a)] for o, a in arm_ghosts],
        "with_corridor": bool(with_corridor), "with_line": bool(with_line),
        "win_rgb": [float(c) for c in args.win_rgb],
        "corridor_rgb": [float(c) for c in args.corridor_rgb],
        "px": {k: (np.round(v, 2).tolist() if o else None)
               for (k, _), v, o in zip(keys.items(), pxk, okk)},
    }
    if detail:
        print(f"  [{view}{'' if with_corridor else ' 素图'}] lookat="
              f"{np.round(lookat, 3).tolist()} d={distance:.2f} fovy={fovy} "
              f"az={cam.azimuth} el={cam.elevation}  残影槽={slot}（频闪 {n_strobe}）")
        for k, v in info["px"].items():
            print(f"      {k:14s} px={v}")
    if out_png is not None:
        out_png.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(out_png, dpi=(300, 300))
        out_png.with_suffix(".json").write_text(
            json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"  saved {out_png} ({img.shape[1]}x{img.shape[0]})")
    return img, info


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染挥拍 hero 实景图（论文 Fig.3）")
    ap.add_argument("--episode", type=Path,
                    default=REPO_ROOT / "experiment_data" / "exp18_fig_assets" / "raw"
                    / "a_hit_clean.npz")
    ap.add_argument("--out", type=str, default="results/scene_angles/chat27/hero.png")
    ap.add_argument("--robot-config", type=Path, default=None,
                    help="机器人描述 YAML（如 configs/robots/r1d_right.yaml）；"
                         "缺省且 NPZ 无 metadata 时按原 RM65 行为")
    ap.add_argument("--model-xml", type=Path, default=None,
                    help="显式 MuJoCo 模型 XML（最高优先级；缺省由 robot-config/"
                         "metadata 决定，最终回退 RM65）")
    ap.add_argument("--view", type=str, default="x+", choices=sorted(VIEWS))
    ap.add_argument("--sheet", action="store_true", help="多机位 contact sheet 模式")
    ap.add_argument("--window-ms", type=float, default=50.0, help="候选窗口半宽（ms）")
    # 相机（缺省 = 预设角 + 自动取景）
    ap.add_argument("--azimuth", type=float, default=None)
    ap.add_argument("--elevation", type=float, default=None)
    ap.add_argument("--distance", type=float, default=None)
    ap.add_argument("--lookat", type=float, nargs=3, default=None)
    ap.add_argument("--fovy", type=float, default=None)
    ap.add_argument("--width", type=int, default=3048)
    ap.add_argument("--height", type=int, default=1980)
    # 残影
    ap.add_argument("--strobe-ms", type=float, default=40.0)
    ap.add_argument("--strobe-alpha", type=float, default=0.65)
    ap.add_argument("--ribbon-alpha", type=float, default=0.12)
    ap.add_argument("--ball-scale", type=float, default=1.0)
    ap.add_argument("--palette", type=str, default="industrial",
                    choices=["industrial", "original", "none"],
                    help="运行期配色（industrial = 与 Fig.1 实景同一套工业灰；"
                         "original/none = 模型原色，见 render_ball_trail.PALETTES）")
    ap.add_argument("--corridor-alpha", type=float, default=CORRIDOR_RGBA[3],
                    help="走廊管透明度（印刷下 0.10~0.16 之间）")
    ap.add_argument("--corridor-rgb", type=float, nargs=3,
                    default=list(CORRIDOR_RGBA[:3]), metavar=("R", "G", "B"),
                    help="走廊管色（默认 #0072B2；调深用 0.00 0.40 0.64）")
    ap.add_argument("--win-rgb", type=float, nargs=3, default=list(WIN_RGB),
                    metavar=("R", "G", "B"),
                    help="候选窗口内球色（默认 #B84500 = #D55E00 加深两档；再深一档即发褐）")
    ap.add_argument("--arm-ghosts", type=str, default="-200,-120,-60,-30,-10",
                    help="整臂残影帧（相对击球步的步数，逗号分隔，按时间先后）；"
                         "默认 = 定稿值（接触前 1.0/0.6/0.3/0.15/0.05 s，"
                         "按运动取帧而非等时间：该 episode 拍心在 −120..−30 步几乎停在底部）")
    ap.add_argument("--arm-ghost-alpha", type=str, default="0.22,0.30,0.38,0.48,0.58",
                    help="各帧 alpha（越晚越实，与不透明的接触帧衔接）")
    ap.add_argument("--mask-thresh", type=int, default=10)
    ap.add_argument("--mask-feather", type=float, default=1.2)
    ap.add_argument("--path-line", action="store_true", default=True)
    ap.add_argument("--no-path-line", dest="path_line", action="store_false")
    ap.add_argument("--path-line-width", type=int, default=5)
    ap.add_argument("--path-line-alpha", type=int, default=120)
    ap.add_argument("--no-shadows", action="store_true")
    ap.add_argument("--brighten", type=float, default=1.0)
    ap.add_argument("--crop-top", type=int, default=0)
    ap.add_argument("--crop-bottom", type=int, default=0)
    ap.add_argument("--detail", action="store_true")
    args = ap.parse_args()

    # 回放归一化：RM65 exp18 / R1D v12 两套格式；模型选择 = 显式 > 配置 > metadata > RM65
    ep = load_episode(args.episode, robot_config=args.robot_config,
                      model_xml=args.model_xml)
    rec = ep.as_record()
    ball = ep.ball_pos
    hit_step, dt = ep.hit_step, ep.dt
    win_steps = round(args.window_ms / 1000.0 / dt)

    # 参考步 = 起始时刻预测的最优候选步（与 fig3_alt_2d、replan_core 同源函数）；
    # metadata 有 k_hit/hit_step 时直接用，否则预测，仍不可得则明确 fallback。
    k_ref, k_ref_source = resolve_k_ref(ep)
    win = (max(0, k_ref - win_steps), min(len(ball) - 1, k_ref + win_steps))
    print(f"episode={ep.source.name}  robot_id={ep.robot_id}  "
          f"模型={ep.model_selection.model_xml.name}  hit_step={hit_step}  "
          f"k_ref={k_ref} ({k_ref_source})  速度={ep.metadata.get('ball_speed')} m/s  "
          f"窗口=±{args.window_ms:g} ms (步 {win[0]}..{win[1]}, 弦长 "
          f"{np.linalg.norm(ball[win[1]] - ball[win[0]]):.3f} m)")
    if "fallback" in k_ref_source:
        print("[hero] k_ref 无 metadata 依据，已明确回退到记录 hit_step（不伪造预测值）")

    offs = [int(x) for x in args.arm_ghosts.split(",") if x.strip()]
    alphas = [float(x) for x in args.arm_ghost_alpha.split(",") if x.strip()]
    if len(alphas) < len(offs):
        alphas = (alphas + [alphas[-1]] * len(offs))[:len(offs)]
    ghosts = list(zip(offs, alphas))

    t0 = time.time()
    if args.sheet:
        out_dir = REPO_ROOT / "results" / "scene_angles" / "chat27"
        out_dir.mkdir(parents=True, exist_ok=True)
        variants = [("full", {"with_corridor": True, "with_line": True}),
                    ("plain", {"with_corridor": False, "with_line": False})]
        views = ("x+", "x-", "y+", "y-", "iso")
        tiles: list[tuple[str, np.ndarray]] = []
        for view in views:
            for tag, kw in variants:
                sub = argparse.Namespace(**vars(args))
                sub.width, sub.height = 1524, 990
                img, _ = render_variant(args=sub, rec=rec, k_ref=k_ref, win=win, view=view,
                                        out_png=None, arm_ghosts=ghosts, detail=True, **kw)
                Image.fromarray(img).save(out_dir / f"sheet_{view}_{tag}.png")
                tiles.append((f"{view} / {tag}", img))
        ncol = len(variants)
        th, tw = tiles[0][1].shape[:2]
        pad, lab = 8, 20
        nrow = len(tiles) // ncol
        sheet = Image.new("RGB", (ncol * tw + (ncol + 1) * pad,
                                  nrow * (th + lab) + (nrow + 1) * pad), (245, 245, 245))
        dr = ImageDraw.Draw(sheet)
        for i, (name, im) in enumerate(tiles):
            r, c = divmod(i, ncol)
            x = pad + c * (tw + pad)
            y = pad + r * (th + lab + pad)
            sheet.paste(Image.fromarray(im), (x, y + lab))
            dr.text((x + 4, y + 4), name, fill=(20, 20, 20))
        sheet_path = out_dir / "contact_sheet.png"
        sheet.save(sheet_path)
        print(f"saved {sheet_path} ({sheet.width}x{sheet.height})")
    else:
        render_variant(args=args, rec=rec, k_ref=k_ref, win=win, view=args.view,
                       out_png=resolve_output_path(args.out), arm_ghosts=ghosts,
                       with_corridor=True, with_line=True, detail=True)
    print(f"总计 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
