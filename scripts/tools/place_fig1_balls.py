"""计算 Fig.1 面板 (a)/(b) 里那串「预测球」精灵在幻灯片坐标下的落位与直径。

背景（chat29 图轮，用户换过背景之后）：
  Fig.1 的实景背景换成了 `render_fig1_scene.py` 的新机位渲染
  （az −100 / el 8 / d 2.0 / lookat −0.7 −0.6 0.8 / fovy 36，2160×1440，3:2），
  但 PPTX 里压在图片上的 6 个球精灵仍是旧机位坐标（旧链朝右下斜，新机位入射段朝右上），
  两者对不上。用户口径：**只把球挪到合适位置，其余标注（红线/管带/圆点/箭头）他自己在 PPTX 里调**。

本脚本用与背景渲染**同一台相机**做解析投影，算出：

  * 入射段 4 个球精灵（默认步 160/176/192/208，间隔 80 ms）的中心与直径；
  * 触球后 2 个球精灵——原图里是「没打中、球越过拍面继续飞」的续飞段，
    按纯抛物线合成（取击球前速度，只受重力），默认 +20 ms / +60 ms；
  * 参考点：拍面中心（预测命中点）、背景那帧实拍的球（位姿步 213）。

坐标换算（PPTX 图片框与渲染图严格 3:2 同比例）：
  面板 (a) 图片左上角幻灯片坐标 (164.374, 61.527) pt，尺寸 340.157×226.771 pt；
  面板 (b) 同尺寸、x 平移 +340.157 pt；1 px = 340.157/2160 = 0.157 480 pt。

⚠ 相机取法（2026-09-16 标定）：**不要用 `mjvScene.camera[0]`**。MuJoCo 报告的
自由相机 `pos` 与它自己的 `fwd` 自相矛盾（方位角差 1°、俯仰符号相反），
按它投影会整体偏 ~38 px（≈6 pt）；按 `pos = lookat − d·fwd` 解析重建后，
用 20 个已知位置的球做栅格标定，x 残差 ≤1 px（y 残差 ~4 px，来自球底部阴影
被阈值切掉导致的质心偏置）。标定口径：az/el 给的是**视线方向**，相机坐在
lookat 沿该方向后退 d 处。

用法（mujoco_tennis 环境；纯 CPU，不需要 GL/EGL）：
    python scripts/tools/place_fig1_balls.py                      # 只打印
    python scripts/tools/place_fig1_balls.py --check-image        # 顺便与背景 PNG 对表
    python scripts/tools/place_fig1_balls.py --apply              # 写回 ICRA.pptx

`--apply` 就地改写 `paper/figures/ICRA.pptx` 里 12 张球精灵图片的 `<a:off>`/`<a:ext>`
（其余部件字节原样搬回），改完必须重跑生成链：`python paper/tools/pptx2fig1.py`。
"""

from __future__ import annotations

import argparse
import io
import re
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_ball_trail as rbt  # noqa: E402
import render_fig1_scene as rfs  # noqa: E402

DEFAULT_EPISODE = (REPO_ROOT / "experiment_data" / "exp18_tcp_exempt" / "raw"
                   / "full_seed040.npz")
DEFAULT_PPTX = REPO_ROOT / "paper" / "figures" / "ICRA.pptx"
DEFAULT_BG = REPO_ROOT / "paper" / "figures" / "fig1_scene_render.png"

# 幻灯片坐标换算（PPTX 面板图片框实测值）
PANEL_A_XY = (164.374, 61.527)
PANEL_DX = 340.157
PT_PER_PX = PANEL_DX / 2160.0
EMU_PER_PT = 12700.0

# 球精灵：面板 (a) 名 → 面板 (b) 名，顺序为「入射 4 + 续飞 2」
BALL_SHAPES = ("图片 33", "图片 34", "图片 32", "图片 30", "图片 35", "图片 36")
BALL_SHAPES_B = ("图片 101", "图片 102", "图片 100", "图片 99", "图片 103", "图片 104")

BALL_RADIUS = 0.0335  # 网球半径（m），用于算投影直径


class _SceneCam:
    """承载 pos/forward/up 的最小相机载体（接口对齐 mjvScene.camera[0]）。"""

    def __init__(self, pos: np.ndarray, fwd: np.ndarray, up: np.ndarray) -> None:
        self.pos, self.forward, self.up = pos, fwd, up


class _FovyModel:
    """`Projector` 只读 `model.vis.global_.fovy`，用轻量替身避免依赖完整 MjModel。"""

    def __init__(self, fovy: float) -> None:
        self.vis = type("_V", (), {"global_": type("_G", (), {"fovy": fovy})()})()


def analytic_camera(lookat: np.ndarray, distance: float, azimuth: float,
                    elevation: float) -> _SceneCam:
    """解析重建自由相机（与离屏栅格实测一致，见模块 docstring 的标定说明）。

    az/el 给的是视线方向 n = (cos el·cos az, cos el·sin az, sin el)：
    相机位置 = lookat − d·n，fwd = n，up 由世界 +z 与 fwd 正交化。
    """
    az, el = np.deg2rad(azimuth), np.deg2rad(elevation)
    n = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    pos = lookat - distance * n
    right = np.cross(n, np.array([0.0, 0.0, 1.0]))
    right = right / np.linalg.norm(right)
    up = np.cross(right, n)
    return _SceneCam(pos, n, up)


def ball_size_pt(proj, pts: np.ndarray) -> np.ndarray:
    """每个球心处的投影直径（幻灯片 pt）：沿相机右向量各偏移 ±r 后取像素距离。"""
    right = proj.right / np.linalg.norm(proj.right)
    px_hi, _ = proj(pts + BALL_RADIUS * right[None, :])
    px_lo, _ = proj(pts - BALL_RADIUS * right[None, :])
    return np.linalg.norm(px_hi - px_lo, axis=1) * PT_PER_PX


def ballistic_after_hit(ball: np.ndarray, hit_step: int, dt: float,
                        steps: list[int]) -> np.ndarray:
    """击球后按「未击中」续飞的位置：取击球前速度，只受重力（g = −9.81 ẑ）。"""
    v = (ball[hit_step] - ball[hit_step - 2]) / (2.0 * dt)
    t = (np.asarray(steps, dtype=float) - hit_step) * dt
    g = np.array([0.0, 0.0, -9.81])
    return ball[hit_step][None, :] + v[None, :] * t[:, None] + 0.5 * g[None, :] * t[:, None] ** 2


def check_against_bg(bg: Path, proj, ball: np.ndarray, pose_step: int) -> None:
    """自检：背景 PNG 里那颗实拍球的黄块外接盒中心 vs 解析投影（阈值 8 px）。"""
    from PIL import Image
    img = np.asarray(Image.open(bg).convert("RGB")).astype(int)
    m = (img[..., 0] > 150) & (img[..., 1] > 180) & (img[..., 2] < 120)
    ys, xs = np.where(m)
    if len(xs) < 50:
        print("⚠ 背景 PNG 里没找到黄色球，跳过自检")
        return
    c = np.array([(xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0])
    p, _ = proj(ball[pose_step][None, :])
    d = p[0] - c
    ok = abs(d[0]) < 8 and abs(d[1]) < 8
    print(f"[自检] 背景 PNG 实拍球（步{pose_step}）：解析投影 "
          f"({p[0,0]:.1f}, {p[0,1]:.1f})  黄块盒心 ({c[0]:.1f}, {c[1]:.1f})  "
          f"差 ({d[0]:+.1f}, {d[1]:+.1f}) px  {'✅' if ok else '⚠ 相机口径不一致'}")


def patch_pptx(pptx: Path, rows: list[dict]) -> None:
    """就地改写球精灵图片的 off/ext（其余部件字节原样保留）。"""
    xml = zipfile.ZipFile(pptx).read("ppt/slides/slide1.xml").decode("utf-8")
    n_done = 0
    for row in rows:
        name = row["shape"]
        pat = re.compile(
            r'(<p:cNvPr[^>]*name="%s"[^>]*>)(.*?)(<a:off x=")(-?\d+)(" y=")(-?\d+)'
            r'("\/>)(<a:ext cx=")(\d+)(" cy=")(\d+)("\/>)' % re.escape(name), re.S)
        m = pat.search(xml)
        if m is None:
            raise SystemExit(f"⚠ 未找到形状 {name} 的 xfrm，PPTX 结构可能已变")
        cx = round(row["size_pt"] * EMU_PER_PT)
        ox = round((row["center_pt"][0] - row["size_pt"] / 2.0) * EMU_PER_PT)
        oy = round((row["center_pt"][1] - row["size_pt"] / 2.0) * EMU_PER_PT)
        new = (m.group(1) + m.group(2) + m.group(3) + str(ox) + m.group(5) + str(oy)
               + m.group(7) + m.group(8) + str(cx) + m.group(10) + str(cx) + m.group(12))
        xml = xml[:m.start()] + new + xml[m.end():]
        n_done += 1
    tmp = pptx.with_suffix(".pptx.tmp")
    zin = zipfile.ZipFile(pptx)
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "ppt/slides/slide1.xml":
                data = xml.encode("utf-8")
            zout.writestr(item, data)
    zin.close()
    shutil.move(str(tmp), str(pptx))
    print(f"✅ 已写回 {pptx.relative_to(REPO_ROOT)}：{n_done} 个球精灵")


def main() -> None:
    ap = argparse.ArgumentParser(description="算 Fig.1 球精灵在幻灯片坐标下的落位与直径")
    ap.add_argument("--episode", type=Path, default=DEFAULT_EPISODE)
    ap.add_argument("--pptx", type=Path, default=DEFAULT_PPTX)
    ap.add_argument("--bg", type=Path, default=DEFAULT_BG, help="自检用的背景渲染 PNG")
    ap.add_argument("--in-steps", type=int, nargs="+", default=[160, 176, 192, 208],
                    help="入射段球精灵的仿真步（hit_step 之前）")
    ap.add_argument("--post-steps", type=int, nargs="+", default=[229, 237],
                    help="续飞段球精灵的仿真步（hit_step 之后，按未击中抛物线）")
    # 相机必须与背景渲染一致（render_fig1_scene.py 的当前默认值）
    ap.add_argument("--azimuth", type=float, default=-100.0)
    ap.add_argument("--elevation", type=float, default=8.0)
    ap.add_argument("--distance", type=float, default=2.0)
    ap.add_argument("--lookat", type=float, nargs=3, default=[-0.7, -0.6, 0.8])
    ap.add_argument("--fovy", type=float, default=36.0)
    ap.add_argument("--width", type=int, default=2160)
    ap.add_argument("--height", type=int, default=1440)
    ap.add_argument("--pose-offset", type=int, default=-12)
    ap.add_argument("--check-image", action="store_true", help="与背景 PNG 的实拍球对表自检")
    ap.add_argument("--apply", action="store_true", help="改动 ICRA.pptx（就地）")
    args = ap.parse_args()

    rec = np.load(args.episode, allow_pickle=True)
    ball = np.asarray(rec["ball_pos"], dtype=float)
    q = np.asarray(rec["q_actual"], dtype=float)
    q_left = np.asarray(rec["init_q_left"], dtype=float)
    hit_step, dt = int(rec["hit_step"]), float(rec["dt"])
    pose_step = int(np.clip(hit_step + args.pose_offset, 0, len(ball) - 1))

    # 只为读拍面中心（FK），不需要 GL
    model = rfs.build_model(args.fovy, args.width, args.height)
    data = mujoco.MjData(model)
    rbt.set_pose(data, q[pose_step], q_left, ball[pose_step])
    cam = analytic_camera(np.array(args.lookat, dtype=float), args.distance,
                          args.azimuth, args.elevation)
    proj = rbt.Projector(_FovyModel(args.fovy), cam, w=args.width, h=args.height)

    in_steps, post_steps = list(args.in_steps), list(args.post_steps)
    pts = np.vstack([ball[in_steps], ballistic_after_hit(ball, hit_step, dt, post_steps)])
    px, ok = proj(pts)
    size = ball_size_pt(proj, pts)
    cam_dist = np.linalg.norm(pts - proj.pos[None, :], axis=1)

    racket_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
    px_rk, _ = proj(np.asarray(data.site_xpos[racket_site], dtype=float)[None, :])
    px_bg, _ = proj(ball[pose_step][None, :])

    def to_slide(p: np.ndarray, panel: int = 0) -> tuple[float, float]:
        return (round(PANEL_A_XY[0] + panel * PANEL_DX + p[0] * PT_PER_PX, 3),
                round(PANEL_A_XY[1] + p[1] * PT_PER_PX, 3))

    if args.check_image:
        check_against_bg(args.bg, proj, ball, pose_step)

    print(f"episode={args.episode.name}  hit_step={hit_step}  dt={dt}  pose_step={pose_step}")
    print(f"相机 az={args.azimuth} el={args.elevation} d={args.distance} fovy={args.fovy} "
          f"lookat={args.lookat}  1 px = {PT_PER_PX:.6f} pt")
    print(f"参考：拍面中心 px=({px_rk[0,0]:.1f}, {px_rk[0,1]:.1f})   "
          f"实拍球(步{pose_step}) px=({px_bg[0,0]:.1f}, {px_bg[0,1]:.1f})")
    print()
    print(f"{'#':>2} {'步':>4} {'距击球':>7} {'px':>17} {'相机距':>7} {'直径pt':>7} "
          f"{'(a) 中心':>17} {'(b) 中心':>17}")
    rows = []
    for i, (st, p, k, s, d) in enumerate(zip(in_steps + post_steps, px, ok, size, cam_dist)):
        tag = "入射" if i < len(in_steps) else "续飞"
        ca, cb = to_slide(p, 0), to_slide(p, 1)
        flag = "" if k else "  ⚠ 画面外/相机后"
        print(f"{tag} {st:>4} {int(st - hit_step):>+7d} ({p[0]:>7.1f},{p[1]:>7.1f}) "
              f"{d:>7.3f} {s:>7.2f} ({ca[0]:>7.1f},{ca[1]:>6.1f}) ({cb[0]:>7.1f},{cb[1]:>6.1f}){flag}")
        if i >= len(BALL_SHAPES):
            continue
        rows.append({"shape": BALL_SHAPES[i], "step": int(st), "kind": tag,
                     "center_pt": list(ca), "size_pt": round(s, 2)})
        rows.append({"shape": BALL_SHAPES_B[i], "step": int(st), "kind": tag,
                     "center_pt": list(cb), "size_pt": round(s, 2)})

    if len(in_steps + post_steps) != len(BALL_SHAPES):
        print(f"⚠ 步数 {len(in_steps + post_steps)} ≠ 球精灵数 {len(BALL_SHAPES)}，"
              f"只有前 {len(BALL_SHAPES)} 个会被写回")

    if args.apply:
        patch_pptx(args.pptx, rows)
        print("下一步：python paper/tools/pptx2fig1.py 重出 fig1_handdrawn.pdf")


if __name__ == "__main__":
    main()
