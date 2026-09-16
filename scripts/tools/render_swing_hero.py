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
    # 单张定稿尺寸（3.387×2.2 in 面板 @ ~900 dpi）
    MUJOCO_GL=egl python scripts/tools/render_swing_hero.py \
        --view x+ --width 3048 --height 1980 --out results/scene_angles/chat27/hero.png

    # 机位调研：五机位 × (完整/素图) contact sheet
    MUJOCO_GL=egl python scripts/tools/render_swing_hero.py --sheet

输出：PNG + 同名 `.json` 附加文件（关键点像素坐标，供 matplotlib 端叠加矢量标记）。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_ball_trail as rbt

from src.utils.mujoco_loader import load_mujoco_model

MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"

BALL_R = 0.033                       # 网球半径（与 rm65_model.xml 一致）
BALL_RGB = (0.98, 0.95, 0.25)        # 窗口外：黄（与全景图一致的「球」色）
WIN_RGB = (0.84, 0.37, 0.00)         # 窗口内：橙（= paper 配色 C["ball"] #D55E00）
CORRIDOR_R = 0.12                    # 走廊半宽 = 拍半径（method.tex 式(2)）
CORRIDOR_RGBA = (0.0, 0.45, 0.72, 0.16)
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


def build_model(ball_scale: float, fovy: float, w: int, h: int,
                corridor: tuple[np.ndarray, np.ndarray]) -> mujoco.MjModel:
    """构建含球残影池 + 走廊管的包装模型（原模型 include + 提亮光照）。

    池内球默认隐藏（位置在地板下方 + alpha=0），运行时改位姿与透明度。
    ⚠ 槽位数必须一次建好：同一个 Renderer 换 ngeom 不同的模型会静默丢 geom。
    ⚠ 走廊管必须用 `fromto` 在编译期取向：实测本机 MuJoCo 忽略运行时对
    `model.geom_quat` 的修改（mj_forward 后 data.geom_xmat 仍是编译期值），
    而 geom_pos/geom_rgba 的运行时修改有效——故弦端点直接在 XML 里烘死。
    """
    hide = f'pos="{HIDDEN[0]} {HIDDEN[1]} {HIDDEN[2]}" rgba="1 1 1 0"'
    ghosts = "".join(
        f'    <geom name="ghost_{k}" type="sphere" size="{BALL_R * ball_scale:.4f}" '
        f'{hide} contype="0" conaffinity="0"/>\n' for k in range(N_BALL))
    c0, c1 = corridor
    corridor_xml = (
        f'    <geom name="corridor_tube" type="capsule" size="{CORRIDOR_R}" '
        f'fromto="{c0[0]:.6f} {c0[1]:.6f} {c0[2]:.6f} {c1[0]:.6f} {c1[1]:.6f} '
        f'{c1[2]:.6f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>\n')
    wrapper = (
        "<mujoco>\n"
        f'  <include file="{MODEL_XML.as_posix()}"/>\n'
        "  <visual>\n"
        f'    <global offwidth="{w}" offheight="{h}" fovy="{fovy}"/>\n'
        '    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7" '
        'specular="0.15 0.15 0.15"/>\n'
        "  </visual>\n"
        "  <worldbody>\n"
        '    <light name="key_top" pos="0.3 -1.2 3.2" dir="-0.08 0.32 -1" '
        'directional="true" diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2" '
        'castshadow="false"/>\n'
        f"{ghosts}{corridor_xml}"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    tmp = Path(tempfile.mkdtemp()) / "wrap_swing_hero.xml"
    tmp.write_text(wrapper, encoding="utf-8")
    return load_mujoco_model(str(tmp))


def moving_body_geoms(model: mujoco.MjModel) -> list[int]:
    """机械臂 geom（body 名以 r_/l_ 开头）——取掩膜时要把它们藏起来。"""
    ids = []
    for g in range(model.ngeom):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g]) or ""
        if bn.startswith(("r_", "l_")):
            ids.append(g)
    return ids


def fill_ball_pool(model: mujoco.MjModel, ids: list[int], ball: np.ndarray,
                   steps: np.ndarray, hit_step: int, *, strobe_steps: int,
                   strobe_alpha: float, ribbon_alpha: float,
                   win: tuple[int, int]) -> tuple[int, int]:
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
        rgb = WIN_RGB if win[0] <= i <= win[1] else BALL_RGB
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
    v = (ball[hit_step] - ball[i]) / ((hit_step - i) * dt)
    g = np.array([0.0, 0.0, -9.81])
    tau = (np.asarray(steps, dtype=float) - hit_step) * dt
    return ball[hit_step][None] + v[None] * tau[:, None] + 0.5 * g[None] * tau[:, None] ** 2


def trail_positions(ball: np.ndarray, line: np.ndarray, hit_step: int,
                    steps: np.ndarray) -> np.ndarray:
    """残影球位：击球前用记录位（含真实弹跳），击球后用预测线延拓。"""
    return np.where((np.asarray(steps) <= hit_step)[:, None], ball[np.asarray(steps)],
                    line)


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData, cam: mujoco.MjvCamera,
                 opt: mujoco.MjvOption, pose: np.ndarray, q_left: np.ndarray,
                 ball_pos: np.ndarray, *, shadows: bool) -> np.ndarray:
    """渲染一帧（指定位姿 + 球位），返回 RGB 数组。"""
    rbt.set_pose(data, pose, q_left, ball_pos)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
    return renderer.render()


def base_frame(renderer: mujoco.Renderer, data: mujoco.MjData, cam: mujoco.MjvCamera,
               opt: mujoco.MjvOption, model: mujoco.MjModel, hide_ids: list[int],
               pose: np.ndarray, q_left: np.ndarray, ball_pos: np.ndarray) -> np.ndarray:
    """无机械臂背景帧（用于取掩膜）：临时把机械臂 geom 的 alpha 置零。"""
    saved = model.geom_rgba[hide_ids].copy()
    model.geom_rgba[hide_ids, 3] = 0.0
    img = render_frame(renderer, data, cam, opt, pose, q_left, ball_pos, shadows=False)
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
    q_left = np.asarray(rec["init_q_left"])
    hit_step = int(rec["hit_step"])
    n, dt = len(ball), float(rec["dt"])
    preset = VIEWS[view]

    # 走廊轴（窗口弦的直线近似）与残影球位都要先算出来：管端点需在建模时烘进 XML
    line = predicted_line(ball, hit_step, dt, np.arange(n))
    trail = trail_positions(ball, line, hit_step, np.arange(n))
    chord = (line[win[0]], line[win[1]])

    model = build_model(args.ball_scale, args.fovy or preset.get("fovy", 40.0),
                        args.width, args.height, chord)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in range(model.ngeom)]
    ball_ids = [g for g, nm in enumerate(names) if nm and nm.startswith("ghost_")]
    tube_id = next(g for g, nm in enumerate(names) if nm == "corridor_tube")
    hide_ids = moving_body_geoms(model)

    # 自动取景（肩部位置需要先摆好姿态再读 xpos）
    shoulder_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "r_base_link1")
    rbt.set_pose(data, q[hit_step], q_left, ball[hit_step])
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
        strobe_alpha=args.strobe_alpha, ribbon_alpha=args.ribbon_alpha, win=win)

    # 走廊管只切透明度（取向已在建模时烘死，见 build_model 注释）
    model.geom_rgba[tube_id] = [*CORRIDOR_RGBA[:3],
                                args.corridor_alpha if with_corridor else 0.0]

    # ── 渲染：背景（无机械臂）→ 残影帧（无阴影）→ 主帧（接触位姿，带阴影） ──
    bg = base_frame(renderer, data, cam, opt, model, hide_ids,
                    q[hit_step], q_left, ball[hit_step])
    ghost_frames = []
    for off, alpha in arm_ghosts:
        s = int(np.clip(hit_step + off, 0, n - 1))
        # 残影帧的球位固定为主帧球位：真球不参与掩膜，否则会被当成残影糊上
        f = render_frame(renderer, data, cam, opt, q[s], q_left, ball[hit_step], shadows=False)
        ghost_frames.append((f, mask_from_diff(f, bg, thresh=args.mask_thresh,
                                               feather=args.mask_feather), alpha))
    img = render_frame(renderer, data, cam, opt, q[hit_step], q_left, ball[hit_step],
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
    rbt.set_pose(data, q[hit_step], q_left, ball[hit_step])
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
    info = {
        "view": view, "azimuth": float(cam.azimuth), "elevation": float(cam.elevation),
        "distance": float(distance), "fovy": float(fovy), "lookat": lookat.tolist(),
        "width": int(img.shape[1]), "height": int(img.shape[0]),
        "k_ref": int(k_ref), "hit_step": hit_step, "window": [int(win[0]), int(win[1])],
        "ball_speed": json.loads(str(rec["metadata"])).get("ball_speed"),
        "n_ghost_slots": slot, "n_strobe": n_strobe,
        "arm_ghosts": [[int(o), float(a)] for o, a in arm_ghosts],
        "with_corridor": bool(with_corridor), "with_line": bool(with_line),
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
    ap.add_argument("--corridor-alpha", type=float, default=CORRIDOR_RGBA[3],
                    help="走廊管透明度（印刷下 0.10~0.16 之间）")
    ap.add_argument("--arm-ghosts", type=str, default="-45,-30,-15",
                    help="整臂残影帧（相对击球步的步数，逗号分隔，按时间先后）")
    ap.add_argument("--arm-ghost-alpha", type=str, default="0.22,0.34,0.48")
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

    rec = np.load(args.episode, allow_pickle=True)
    md = json.loads(str(rec["metadata"]))
    hit_step = int(rec["hit_step"])
    dt = float(rec["dt"])
    ball = np.asarray(rec["ball_pos"])
    win_steps = round(args.window_ms / 1000.0 / dt)

    # 参考步 = 起始时刻预测的最优候选步（与 fig3_alt_2d、replan_core 同源函数）
    from src.robot.constants import SHOULDER_POS, WORKSPACE_RADIUS
    from src.sim.rm65_env import RM65Env
    from src.tennis.hitting import find_hitting_point_physics
    _env = RM65Env(MODEL_XML)
    hi = find_hitting_point_physics(_env, np.array(md["p0"]), np.array(md["v0"]),
                                    SHOULDER_POS, WORKSPACE_RADIUS, len(ball))
    k_ref = int(hi["k_hit"]) if hi is not None else hit_step
    win = (max(0, k_ref - win_steps), min(len(ball) - 1, k_ref + win_steps))
    print(f"episode={args.episode.name}  hit_step={hit_step}  k_ref={k_ref}  "
          f"速度={md.get('ball_speed')} m/s  窗口=±{args.window_ms:g} ms "
          f"(步 {win[0]}..{win[1]}, 弦长 "
          f"{np.linalg.norm(ball[win[1]] - ball[win[0]]):.3f} m)")

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
                       out_png=REPO_ROOT / args.out, arm_ghosts=ghosts,
                       with_corridor=True, with_line=True, detail=True)
    print(f"总计 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
