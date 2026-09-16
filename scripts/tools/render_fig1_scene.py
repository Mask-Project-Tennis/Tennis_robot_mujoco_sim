"""渲染 Fig.1（系统概览图）的实景背景：一次挥拍的接触帧，单帧、无残影、浅底。

背景（chat29 图轮）：
  Fig.1 原背景实景来自旧版 `render_robot_scene.py` 的 `--pan-only` 分支
  （2160×1440、hit_step−12、az −100 / el 14 / d 2.0 / lookat −0.7 −0.6 0.8、fovy 45），
  该分支已在 0f71adb 移除，HEAD 无法复现这条配方。本脚本把配方固定下来，并补三件事：
  ① 运行期调色板覆盖（`rbt.apply_palette`：右臂亮蓝→工业浅灰、拍面亮绿→深炭灰、
     立柱近黑→中灰），**不改 `src/robot/rm65_model.xml`**，避免波及仿真视频与回放；
  ② 黑天空 → 灰渐变（复刻 chat16 的 PIL 后处理：仅替换近黑像素，保持浅底风格）；
  ③ 输出关键点像素 JSON（球的频闪位、接触位、拍面中心、0.12 m 的像素尺度），
     供标注层（`scripts/plot/paper_figs.py::fig1_strip`）用同一相机投影叠加矢量标注。

用法（mujoco_tennis 环境）：
    MUJOCO_GL=egl python scripts/tools/render_fig1_scene.py --palette industrial \
        --out paper/figures/fig1_scene_render.png
    # 复现门（应与 results/scene_angles/front_H_bright.png 高度一致）：
    MUJOCO_GL=egl python scripts/tools/render_fig1_scene.py --palette original \
        --out /tmp/fig1_repro.png

输出：PNG（3:2）+ 同名 `.json`。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import render_ball_trail as rbt

from src.utils.mujoco_loader import load_mujoco_model  # noqa: E402

MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"
DEFAULT_EPISODE = (REPO_ROOT / "experiment_data" / "exp18_tcp_exempt" / "raw"
                   / "full_seed040.npz")

# chat16 复刻参数：近黑（max 通道 < 15）替换为 252→236 的竖直灰渐变
SKY_DARK_MAX = 15
SKY_TOP, SKY_BOTTOM, SKY_HORIZON = 252, 236, 0.76


def build_model(fovy: float, w: int, h: int) -> mujoco.MjModel:
    """构建仅含提亮灯光的包装模型（无残影池——Fig.1 背景是单帧）。"""
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
        "  </worldbody>\n"
        "</mujoco>\n"
    )
    tmp = Path(tempfile.mkdtemp()) / "wrap_fig1.xml"
    tmp.write_text(wrapper, encoding="utf-8")
    return load_mujoco_model(str(tmp))


def sky_gradient(img: np.ndarray) -> np.ndarray:
    """把近黑的天空像素替换为 252→236 的竖直灰渐变（chat16 复刻，其它像素不动）。"""
    h, w = img.shape[:2]
    ys = np.arange(h, dtype=float)
    horizon = int(SKY_HORIZON * h)
    rows = np.where(ys < horizon,
                    SKY_TOP - (SKY_TOP - SKY_BOTTOM) * np.clip(ys / max(horizon, 1), 0, 1),
                    float(SKY_BOTTOM))
    gray = np.repeat(rows[:, None, None], w, axis=1).repeat(3, axis=2).astype(np.uint8)
    mask = img.max(axis=2) < SKY_DARK_MAX
    return np.where(mask[..., None], gray, img)


def main() -> None:
    ap = argparse.ArgumentParser(description="渲染 Fig.1 背景实景（接触帧、单帧、浅底）")
    ap.add_argument("--episode", type=Path, default=DEFAULT_EPISODE,
                    help="轨迹 NPZ（含 q_actual/ball_pos/hit_step/init_q_left/dt）")
    ap.add_argument("--pose-offset", type=int, default=-12,
                    help="机器人位姿相对击球步的偏移（步）；chat16 原图用 −12 = 击球前 60 ms")
    ap.add_argument("--azimuth", type=float, default=-100.0, help="视线方位角（度）")
    ap.add_argument("--elevation", type=float, default=14.0, help="视线俯仰（度）")
    ap.add_argument("--distance", type=float, default=2.0, help="相机距离（m）")
    ap.add_argument("--lookat", type=float, nargs=3, default=[-0.7, -0.6, 0.8],
                    metavar=("X", "Y", "Z"), help="相机注视点（世界系）")
    ap.add_argument("--fovy", type=float, default=45.0, help="竖向视场角（度）")
    ap.add_argument("--width", type=int, default=2160, help="画面宽度（px）")
    ap.add_argument("--height", type=int, default=1440, help="画面高度（px）")
    ap.add_argument("--palette", type=str, default="industrial",
                    choices=["industrial", "original", "none"],
                    help="运行期配色（original/none = 保持模型原色）")
    ap.add_argument("--no-sky-gradient", action="store_true",
                    help="关闭黑天空→灰渐变后处理（对照用）")
    ap.add_argument("--strobe-ms", type=float, default=40.0,
                    help="频闪间隔（ms）——JSON 里按此给出球的候选标注位")
    ap.add_argument("--out", type=str,
                    default="paper/figures/fig1_scene_render.png",
                    help="输出 PNG（相对仓库根；同名 .json 附带关键点）")
    args = ap.parse_args()

    rec = np.load(args.episode, allow_pickle=True)
    md = json.loads(str(rec["metadata"]))
    ball = np.asarray(rec["ball_pos"], dtype=float)
    q = np.asarray(rec["q_actual"], dtype=float)
    q_left = np.asarray(rec["init_q_left"], dtype=float)
    hit_step, dt = int(rec["hit_step"]), float(rec["dt"])
    n = len(ball)
    pose_step = int(np.clip(hit_step + args.pose_offset, 0, n - 1))
    print(f"episode={args.episode.name}  hit_step={hit_step}  dt={dt}  n={n}  "
          f"ball_speed={md.get('ball_speed')}  pose_step={pose_step}")

    model = build_model(args.fovy, args.width, args.height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, args.height, args.width)
    for line in rbt.apply_palette(model, None if args.palette in ("original", "none")
                                 else args.palette):
        print("      " + line)

    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0            # 隐藏碰撞体可视化（col_* 胶囊）
    opt.sitegroup[0] = 0            # 隐藏 site 标记

    rbt.set_pose(data, q[pose_step], q_left, ball[pose_step])
    cam = mujoco.MjvCamera()
    rbt.configure_camera(cam, np.array(args.lookat, dtype=float), args.distance,
                         args.azimuth, args.elevation)
    renderer.update_scene(data, camera=cam, scene_option=opt)
    proj = rbt.Projector(model, renderer.scene.camera[0], w=args.width, h=args.height)

    # 关键点：球的频闪位（画面内）、击球点、拍面中心、0.12 m 的像素尺度
    strobe_steps = max(1, round(args.strobe_ms / (dt * 1000.0)))
    steps = np.unique(np.concatenate([np.arange(0, n, strobe_steps), [hit_step]]))
    px_ball, ok_ball = proj(ball[steps])
    racket_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "racket_center")
    contact_pt = np.asarray(data.site_xpos[racket_site], dtype=float)
    px_contact, ok_contact = proj(contact_pt[None, :])
    # 0.12 m（拍半径）在该深度处的像素尺度：沿相机右向量偏移一点点再投影
    right = proj.right / np.linalg.norm(proj.right)
    px_scale, _ = proj((ball[hit_step] + 0.12 * right)[None, :])

    img = renderer.render()
    if not args.no_sky_gradient:
        img = sky_gradient(img)
    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(out, dpi=(300, 300))

    meta = {
        "episode": args.episode.name, "hit_step": hit_step, "dt": dt,
        "pose_step": pose_step, "ball_speed": md.get("ball_speed"),
        "camera": {"azimuth": args.azimuth, "elevation": args.elevation,
                   "distance": args.distance, "lookat": list(args.lookat),
                   "fovy": args.fovy},
        "palette": args.palette, "sky_gradient": not args.no_sky_gradient,
        "width": args.width, "height": args.height,
        "strobe_steps": int(strobe_steps),
        "px": {
            "ball_steps": steps.tolist(),
            "ball": px_ball.tolist(),
            "ball_ok": ok_ball.tolist(),
            "contact": px_contact[0].tolist(),
            "contact_ok": bool(ok_contact[0]),
            "px_per_0p12m": float(np.abs(px_scale[0, 0] - px_ball[steps == hit_step][0, 0])),
        },
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"saved: {out} ({img.shape[1]}x{img.shape[0]}) + {out.with_suffix('.json').name}")
    print(f"      频闪 {len(steps)} 点；画面内 "
          f"{int(np.sum(ok_ball & (px_ball[:, 0] > 0) & (px_ball[:, 0] < args.width)))} 点；"
          f"0.12 m ≈ {meta['px']['px_per_0p12m']:.1f} px")


if __name__ == "__main__":
    main()
