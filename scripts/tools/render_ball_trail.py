"""渲染球轨迹残影单图（论文 Fig.2 场景图）。

用法（mujoco_tennis 环境，需 MUJOCO_GL=egl 走离屏渲染）:
    # 定稿方案：单图全景 + 球轨迹残影（相机 = 已认可的 az130°/el−34°/d3.0）
    MUJOCO_GL=egl python scripts/tools/render_ball_trail.py

    # 调参示例：频闪 25 ms + 更亮的带 + 细轨迹线 + 机器人取击球前 60 ms 位姿
    MUJOCO_GL=egl python scripts/tools/render_ball_trail.py \
        --strobe-ms 25 --strobe-alpha 0.7 --ribbon-alpha 0.16 \
        --path-line --pose-offset -12 --out results/fig2_trail.png

残影机制（2026-09-14 调研定稿，前三条是实测踩过的坑）:
1) 做法：把 episode 每个历史时刻的球位姿「烘焙」成 wrapper XML 里的静态半透明 geom，
   再用 model.geom_pos / geom_rgba 运行时改位姿与透明度；透明几何走 MuJoCo 原生
   透明通道做深度排序，遮挡与透视天然正确，不需要逐帧渲染再合成。
2) 池化：一次建好 N_BALL 个槽位的 ghost 池，各方案只改数值。原因：同一个
   mujoco.Renderer 换成 ngeom 不同的模型会静默丢弃多出来的 geom——残影会整体
   不出现且不报错。
3) MJCF 属性归属：emission 属于 material、castshadow 属于 light，写在 geom 上
   直接 schema violation 加载失败。
4) 速度编码：等时距频闪的「间距」才是速度的物理读数（full_seed040：命中前
   6.68 m/s → 命中后 7.91 m/s，40 ms 间距 0.267 → 0.316 m = 4.0 → 4.8 个球直径）；
   用透明度编码速度在同一 episode 上不可辨（与近击球点加亮方案像素差仅 0.27%）。
5) 可读性：球残影直径约 38~56 px（2160 px 宽画面；单栏 3.4 in 印刷 ≈ 1.8~2.2 mm），
   稀疏频闪在印刷尺寸下只出点、不出「路径」→ 默认叠 ribbon（每步一个低透明度球），
   必要时再叠细轨迹线（--path-line）。
6) 残影会在地面投出真实阴影（--no-shadows 可关，但机器人接触阴影会一起消失）；
   球拍/整臂残影实测在 3 m 取景下会糊住击球点，故本脚本不提供。

输出: 单张 PNG（2160×1440 @300dpi = 7.2 in 宽）。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.utils.mujoco_loader import load_mujoco_model

MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"

W, H = 2160, 1440                  # 画面分辨率（7.2 in @300dpi）
BALL_R = 0.033                     # 网球半径（与 rm65_model.xml 一致）
BALL_RGB = (0.98, 0.95, 0.25)      # 比真球略亮，保证叠在绿色场地上的对比度
N_BALL = 320                       # 残影池槽位数（覆盖整条 episode 的 307 步）
HIDDEN = np.array([0.0, 0.0, -50.0])   # 隐藏槽位位置（地板下方、画面外）


def build_pool_model() -> mujoco.MjModel:
    """构建含球残影池的包装模型（原模型 include + 提亮光照 + N_BALL 个透明球槽）。

    池内 geom 默认隐藏（位置在地板下方 + alpha=0），由残影逻辑按需放置。
    """
    ghosts = "".join(
        f'    <geom name="ghost_{k}" type="sphere" size="{BALL_R}" '
        f'pos="{HIDDEN[0]} {HIDDEN[1]} {HIDDEN[2]}" rgba="1 1 1 0" '
        f'contype="0" conaffinity="0"/>\n' for k in range(N_BALL))
    wrapper = (
        "<mujoco>\n"
        f'  <include file="{MODEL_XML.as_posix()}"/>\n'
        "  <visual>\n"
        f'    <global offwidth="{W}" offheight="{H}"/>\n'
        # 提亮：头灯（跟随相机）+ 顶部定向光，避免俯视时机器人背面全黑
        '    <headlight ambient="0.35 0.35 0.35" diffuse="0.7 0.7 0.7" '
        'specular="0.15 0.15 0.15"/>\n'
        "  </visual>\n"
        "  <worldbody>\n"
        '    <light name="key_top" pos="0.3 -1.2 3.2" dir="-0.08 0.32 -1" '
        'directional="true" diffuse="0.55 0.55 0.55" specular="0.2 0.2 0.2" '
        'castshadow="false"/>\n'
        f"{ghosts}"
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    tmp = Path(tempfile.mkdtemp()) / "wrap_ball_trail.xml"
    tmp.write_text(wrapper, encoding="utf-8")
    return load_mujoco_model(str(tmp))


def set_pose(data: mujoco.MjData, q_right: np.ndarray, q_left: np.ndarray,
             ball_pos: np.ndarray) -> None:
    """按 episode 记录设置双臂关节角与球位姿（qpos 布局: 右臂6 + 左臂6 + 球7）。"""
    data.qpos[0:6] = q_right
    data.qpos[6:12] = q_left
    data.qpos[12:15] = ball_pos
    data.qpos[15:19] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(data.model, data)


def configure_camera(cam: mujoco.MjvCamera, lookat: np.ndarray,
                     distance: float, azimuth: float, elevation: float) -> None:
    """设置自由相机。azimuth 相对 +X 轴（度），elevation 相对水平面（度，负值=俯视）。"""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation


class Projector:
    """解析投影：世界点 → 像素坐标（用渲染场景相机的 pos/forward/up + fovy）。"""

    def __init__(self, model: mujoco.MjModel, scene_cam, w: int = W, h: int = H):
        self.pos = np.array(scene_cam.pos, dtype=float)
        self.fwd = np.array(scene_cam.forward, dtype=float)
        self.up = np.array(scene_cam.up, dtype=float)
        self.right = np.cross(self.fwd, self.up)
        self.tan_half = np.tan(np.deg2rad(model.vis.global_.fovy) / 2)
        self.w, self.h = w, h

    def __call__(self, pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """返回 (像素坐标 (N,2), 是否在相机前方 (N,))。"""
        d = np.asarray(pts, dtype=float) - self.pos
        z = d @ self.fwd
        ok = z > 0.05
        zc = np.where(ok, z, 1.0)
        u = (d @ self.right) / (zc * self.tan_half * self.w / self.h)
        v = (d @ self.up) / (zc * self.tan_half)
        px = np.stack([(u + 1) / 2 * self.w, (1 - v) / 2 * self.h], axis=1)
        return px, ok


def draw_path_line(img: np.ndarray, px: np.ndarray, ok: np.ndarray, *,
                   width: int, alpha: int, color: tuple[int, int, int] = (255, 232, 90),
                   glow: bool = True) -> np.ndarray:
    """在图像上叠加轨迹线（PIL，含外发光），跳过画面外的点。"""
    pts = [tuple(p) for p, k in zip(px, ok) if k]
    if len(pts) < 2:
        return img
    base = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    ImageDraw.Draw(layer).line(pts, fill=(*color, alpha), width=width, joint="curve")
    if glow:
        base = Image.alpha_composite(base, layer.filter(ImageFilter.GaussianBlur(6)))
    return np.array(Image.alpha_composite(base, layer).convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染球轨迹残影单图（论文 Fig.2）")
    parser.add_argument(
        "--episode", type=Path,
        default=REPO_ROOT / "experiment_data" / "exp18_tcp_exempt" / "raw"
        / "full_seed040.npz",
        help="轨迹 NPZ（含 q_actual/ball_pos/hit_step/init_q_left/dt）")
    parser.add_argument("--out", type=str,
                        default="paper/figures/fig2_trajectory.png",
                        help="输出 PNG 路径（相对仓库根）")
    # 相机（默认 = 已认可的 Fig.2 角度：11 点方向俯视 34°）
    parser.add_argument("--azimuth", type=float, default=130.0,
                        help="方位角（度，相对 +X 轴）")
    parser.add_argument("--elevation", type=float, default=-34.0,
                        help="俯仰角（度，负值 = 相机在上方俯视机器人）")
    parser.add_argument("--distance", type=float, default=3.0, help="相机距离（m）")
    parser.add_argument("--lookat", type=float, nargs=3,
                        default=[-0.35, -0.45, 1.00],
                        metavar=("X", "Y", "Z"), help="相机注视点（世界系）")
    parser.add_argument("--pose-offset", type=int, default=0,
                        help="机器人位姿相对击球步的偏移（步）；0 = 接触瞬间")
    # 残影参数
    parser.add_argument("--strobe-ms", type=float, default=40.0,
                        help="频闪间隔（ms）；间距即速度的物理读数")
    parser.add_argument("--strobe-alpha", type=float, default=0.65,
                        help="频闪球透明度")
    parser.add_argument("--ribbon-alpha", type=float, default=0.12,
                        help="密带（每步一个）透明度；0 = 只用频闪")
    parser.add_argument("--path-line", action="store_true", help="叠加细轨迹线")
    parser.add_argument("--path-line-width", type=int, default=5)
    parser.add_argument("--path-line-alpha", type=int, default=130)
    parser.add_argument("--no-shadows", action="store_true",
                        help="关闭地面投影（机器人接触阴影会一起消失）")
    args = parser.parse_args()

    rec = np.load(args.episode, allow_pickle=True)
    md = json.loads(str(rec["metadata"]))
    hit_step = int(rec["hit_step"])
    ball = np.asarray(rec["ball_pos"])
    n, dt = len(ball), float(rec["dt"])
    pose_step = int(np.clip(hit_step + args.pose_offset, 0, n - 1))
    print(f"episode={args.episode.name}  hit_step={hit_step}  "
          f"hit_type={md.get('hit_type')}  ball_speed={md.get('ball_speed')}  "
          f"pose_step={pose_step}")

    model = build_pool_model()
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, H, W)
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_{k}")
           for k in range(N_BALL)]
    cam = mujoco.MjvCamera()
    configure_camera(cam, np.array(args.lookat, dtype=float), args.distance,
                     args.azimuth, args.elevation)
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0           # 隐藏碰撞体可视化（col_* 红蓝胶囊）
    opt.sitegroup[0] = 0           # 隐藏 hit_target 等标记点

    # ── 1. 解析投影求「画面内」的球位（画面外放残影没有意义） ──
    set_pose(data, rec["q_actual"][pose_step], rec["init_q_left"], ball[pose_step])
    renderer.update_scene(data, camera=cam, scene_option=opt)
    proj = Projector(model, renderer.scene.camera[0])
    px, ok = proj(ball)
    inside = (ok & (px[:, 0] > -60) & (px[:, 0] < W + 60)
              & (px[:, 1] > -60) & (px[:, 1] < H + 60))
    steps = np.flatnonzero(inside)
    strobe_steps = max(1, round(args.strobe_ms / (dt * 1000.0)))
    print(f"画面内球位: 步 {steps[0]}..{steps[-1]} "
          f"(t = {(steps[0] - hit_step) * dt:+.3f}..{(steps[-1] - hit_step) * dt:+.3f} s)  "
          f"像素端点 {np.round(px[steps[0]]).astype(int).tolist()} → "
          f"{np.round(px[steps[-1]]).astype(int).tolist()}  "
          f"频闪间隔 {strobe_steps} 步")

    # ── 2. 填残影池：等时距频闪（亮）+ 密带（淡） ──
    slot, n_strobe = 0, 0
    for i in steps:
        if i == pose_step:          # 该步有实心真球，跳过以免 z-fighting
            continue
        if (i - hit_step) % strobe_steps == 0:
            alpha, n_strobe = args.strobe_alpha, n_strobe + 1
        else:
            alpha = args.ribbon_alpha
        if alpha <= 0 or slot >= N_BALL:
            continue
        model.geom_pos[ids[slot]] = ball[i]
        model.geom_rgba[ids[slot]] = [*BALL_RGB, alpha]
        slot += 1
    print(f"残影槽位={slot}（其中频闪 {n_strobe} 个）")

    # ── 3. 渲染 + 线条叠加 + 保存 ──
    set_pose(data, rec["q_actual"][pose_step], rec["init_q_left"], ball[pose_step])
    renderer.update_scene(data, camera=cam, scene_option=opt)
    renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0 if args.no_shadows else 1
    img = renderer.render()
    if args.path_line:
        img = draw_path_line(img, px, inside, width=args.path_line_width,
                             alpha=args.path_line_alpha)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out, dpi=(300, 300))
    print(f"saved: {out} ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
