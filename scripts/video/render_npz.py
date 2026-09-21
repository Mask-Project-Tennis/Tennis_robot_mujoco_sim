#!/usr/bin/env python
"""随稿视频 NPZ 轨迹离线重渲器（无漂移管线）。

v12 主管线（replan_core/MPCController）与 render_hits.py 复制的 run_20hits 内联
循环存在微小行为差异，边缘种子（如 seed 25 @12 m/s none）会被翻转——已被
--expect 核验警报证实。本脚本改为**运动学回放 v12 自己 dump 的轨迹 npz**
（q_actual / ball_pos 逐帧摆姿态，不跑物理），保证画面就是实验真实记录：
hit_type 与 CSV 逐位一致，双盲审查下最可辩护。

每侧输出 4 个 MP4 + 拼贴：
  {label}_clean.mp4          全程 0.5× 广角（无叠加，组装用；两栏并排素材）
  {label}_preview.mp4        同上 + Simulation 角标 + 0.5× playback 标注
  {label}_contact_clean.mp4  接触/擦过窗口 0.25× 推近（无叠加）
  {label}_contact_preview.mp4 同上 + 角标 + 0.25× playback 标注
miss 轨迹的「接触窗口」取球-拍最近时刻（擦过瞬间），对比叙事同样成立。

用法（mujoco_tennis 环境，仓库根目录）：
    python scripts/video/render_npz.py --npz paper/video/assets/paircheck_s25_none.npz --label s5_pt
    python scripts/video/render_npz.py --npz paper/video/assets/seed47_full.npz --label s9 \
        --cam-distance 2.6 --cam-azimuth 118
"""
from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault(
    "__EGL_VENDOR_LIBRARY_FILENAMES",
    "/usr/share/glvnd/egl_vendor.d/10_nvidia.json",
)

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import mujoco
import render_hits as rh  # 复用尾迹常量 / 包装模型 / 叠加绘制 / 输出目录
from PIL import Image, ImageDraw

from src.utils.mujoco_loader import load_mujoco_model

OUT_DIR = rh.OUT_DIR

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("render_npz")


def set_pose(data: mujoco.MjData, q_right: np.ndarray, q_left: np.ndarray,
             ball_pos: np.ndarray) -> None:
    """按记录帧设置双臂关节角与球位姿（qpos 布局: 右臂6 + 左臂6 + 球7）。"""
    data.qpos[0:6] = q_right
    data.qpos[6:12] = q_left
    data.qpos[12:15] = ball_pos
    data.qpos[15:19] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(data.model, data)


def main() -> None:
    """入口：读 npz → 逐帧摆姿态双机位渲染 → 合成 4 版 MP4 + 拼贴。"""
    parser = argparse.ArgumentParser(description="随稿视频 NPZ 轨迹离线重渲器")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    # 广角相机（S5 配对默认 = S1 同款；S9 收尾可推近换机位）
    parser.add_argument("--cam-distance", type=float, default=3.5)
    parser.add_argument("--cam-elevation", type=float, default=-15)
    parser.add_argument("--cam-azimuth", type=float, default=135)
    # 推近窗口
    parser.add_argument("--close-pre-ms", type=int, default=350)
    parser.add_argument("--close-post-ms", type=int, default=550)
    parser.add_argument("--close-cam-distance", type=float, default=2.2)
    parser.add_argument("--close-cam-azimuth", type=float, default=112.0)
    # 播放速度
    parser.add_argument("--speed-wide", type=float, default=0.5,
                        help="全程段播放速度（0.5× = 半速）")
    parser.add_argument("--speed-close", type=float, default=0.25,
                        help="接触窗口段播放速度")
    args = parser.parse_args()

    rec = np.load(args.npz)
    md = json.loads(str(rec["metadata"]))
    q_actual = np.asarray(rec["q_actual"])
    ball = np.asarray(rec["ball_pos"])
    q_left = np.asarray(rec["init_q_left"])
    tcp = np.asarray(rec["tcp_pos"])
    dt = float(rec["dt"])
    hit_step = int(rec["hit_step"])
    n = len(ball)
    # 接触窗口中心：命中轨迹用 hit_step；miss 轨迹用球-拍最近时刻（擦过瞬间）
    center = hit_step if hit_step >= 0 else int(
        np.argmin(np.linalg.norm(ball - tcp, axis=1)))
    logger.info("[npz] %s: n=%d hit_step=%d center=%d hit_type=%s",
                args.npz.name, n, hit_step, center, md.get("hit_type"))

    model = load_mujoco_model(str(rh.build_trail_model(args.width, args.height)))
    data = mujoco.MjData(model)
    ghost_ids = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"ghost_{k}")
         for k in range(rh.GHOST_N)], dtype=int)
    sys.path.insert(0, str(rh.REPO / "scripts" / "tools"))
    from render_ball_trail import apply_palette
    apply_palette(model, "industrial")
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0
    opt.sitegroup[0] = 0

    renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = (
        args.cam_distance, args.cam_elevation, args.cam_azimuth)
    cam.lookat[:] = [0.0, 0.0, 1.0]
    cam2 = mujoco.MjvCamera()
    cam2.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam2.distance = args.close_cam_distance
    cam2.azimuth = args.close_cam_azimuth
    cam2.elevation = -10
    cam2.lookat[:] = ball[center] + np.array([0.0, -0.20, 0.02])

    close_start = max(0, center - int(args.close_pre_ms / 1000.0 / dt) - 6)
    close_end = min(n - 1, center + int(args.close_post_ms / 1000.0 / dt) + 6)

    def pose_at(x: float) -> tuple[int, np.ndarray, np.ndarray]:
        """亚步插值：x 为小数步索引，返回 (整数历史步, 插值后关节角, 插值后球位)。

        5 ms 内关节/球位移极小，线性内插视觉精确；修 0.1×/0.25× 慢放下
        「源帧整数驻留」造成的步进感。
        """
        x = float(np.clip(x, 0.0, n - 1.0))
        i0 = min(int(x), n - 2)
        a = x - i0
        q = (1.0 - a) * q_actual[i0] + a * q_actual[i0 + 1]
        bp = (1.0 - a) * ball[i0] + a * ball[i0 + 1]
        return i0, q, bp

    def render_frame(x: float, camera) -> np.ndarray:
        """更新尾迹 ghost 池（按整数历史步）→ 摆插值姿态 → 渲染。"""
        i0, q, bp = pose_at(x)
        lo = max(0, i0 - rh.TRAIL_WINDOW_STEPS)
        slot = 0
        for k in range(lo, i0, rh.TRAIL_STRIDE):
            alpha = (rh.TRAIL_STROBE_ALPHA if k % rh.TRAIL_STROBE_STEPS == 0
                     else rh.TRAIL_RIBBON_ALPHA)
            model.geom_pos[ghost_ids[slot]] = ball[k]
            model.geom_rgba[ghost_ids[slot]] = [*rh.BALL_RGB, alpha]
            slot += 1
            if slot >= rh.GHOST_N:
                break
        for k in range(slot, rh.GHOST_N):
            model.geom_pos[ghost_ids[k]] = rh.GHOST_HIDDEN
            model.geom_rgba[ghost_ids[k]] = [1, 1, 1, 0]
        set_pose(data, q, q_left, bp)
        renderer.update_scene(data, camera=camera, scene_option=opt)
        return renderer.render()

    # ── 合成：逐输出帧按需渲染（亚步插值），全程广角 + 接触窗口推近各出 clean/preview ──
    import imageio

    def write_clip(path: Path, x0: float, x1: float, speed: float,
                   tag: str | None, camera) -> float:
        """渲染 [x0, x1) 步区间并按 speed 合成；返回时长（秒）。"""
        clean = OUT_DIR / path
        w = imageio.get_writer(str(clean), fps=args.fps,
                               output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
        prev_path = clean.with_name(clean.stem.replace("_clean", "_preview") + ".mp4")
        wp = imageio.get_writer(str(prev_path), fps=args.fps,
                                output_params=["-crf", "18", "-pix_fmt", "yuv420p"])
        n_out = int((x1 - x0) * dt / speed * args.fps)
        fade = int(0.4 * args.fps)
        for j in range(n_out):
            x = x0 + j * speed / (args.fps * dt)   # 亚步推进，无整数驻留
            frame = render_frame(x, camera)
            w.append_data(frame)
            img = Image.fromarray(frame).convert("RGBA")
            rh.draw_badge(img)
            if tag:
                rh.draw_tag(img, tag)
            rem = n_out - j - 1
            if rem < fade:
                img = Image.blend(img, Image.new("RGBA", img.size, (0, 0, 0, 255)),
                                  1.0 - (rem + 1) / fade)
            wp.append_data(np.array(img.convert("RGB")))
        w.close()
        wp.close()
        return n_out / args.fps

    dur_wide = write_clip(Path(f"{args.label}_clean.mp4"), 0.0, float(n),
                          args.speed_wide, f"{args.speed_wide:g}× playback", cam)
    dur_close = write_clip(Path(f"{args.label}_contact_clean.mp4"),
                           float(close_start), float(close_end),
                           args.speed_close, f"{args.speed_close:g}× playback", cam2)
    logger.info("[clip] wide %.2fs, contact %.2fs", dur_wide, dur_close)

    # 拼贴：全程 4 帧（广角）+ 接触窗口 4 帧（推近）
    tile_w, tile_h = 640, 360
    sheet = Image.new("RGB", (tile_w * 4, tile_h * 2), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    font = rh._font(18, bold=True)
    wide_xs = [n * r for r in (0.08, 0.4, 0.7, 0.97)]
    close_xs = [close_start, float(center),
                min(center + 8, close_end), float(close_end)]
    tiles = [(render_frame(x, cam), x, False) for x in wide_xs] + \
            [(render_frame(x, cam2), x, True) for x in close_xs]
    for k, (frame, x, is_close) in enumerate(tiles):
        tile = Image.fromarray(frame).resize((tile_w, tile_h))
        px, py = (k % 4) * tile_w, (k // 4) * tile_h
        sheet.paste(tile, (px, py))
        tagtxt = f"t={x * dt:.2f}s" + (
            "  CONTACT" if hit_step >= 0 and abs(x - hit_step) <= 1
            else ("  PASS" if is_close and abs(x - center) <= 1 else ""))
        d.rectangle((px + 8, py + 8, px + 8 + d.textlength(tagtxt, font=font) + 14,
                     py + 34), fill=(0, 0, 0, 160))
        d.text((px + 15, py + 12), tagtxt, font=font, fill=(255, 255, 255))
    sheet.save(OUT_DIR / f"{args.label}_sheet.png")
    Image.fromarray(render_frame(float(min(center, n - 1)), cam)).save(
        OUT_DIR / f"{args.label}_contact.png")

    meta = {"npz": str(args.npz), "hit_step": hit_step, "center": center,
            "hit_type": md.get("hit_type"), "n_steps": n,
            "label": args.label, "speed_wide": args.speed_wide,
            "speed_close": args.speed_close}
    (OUT_DIR / f"{args.label}_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("[done] %s: 4 mp4 + sheet + contact.png", args.label)


if __name__ == "__main__":
    main()
