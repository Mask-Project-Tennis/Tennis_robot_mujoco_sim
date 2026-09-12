#!/usr/bin/env python3
"""Fig.1: 系统场景图 —— RM-65B 双臂 + 球拍 + 网球（击球瞬间离屏渲染）。

数据源: exp18 raw/a_hit_clean.npz 的击球步关节构型 + 球位置（真实命中瞬间）。
模型无命名相机，用 MjvCamera 手动指定机位。

用法:
    MUJOCO_GL=egl python scripts/plot/fig1_render.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT))

import mujoco  # noqa: E402

from src.utils.mujoco_loader import load_mujoco_model  # noqa: E402

MODEL_PATH = PROJECT / "src" / "robot" / "rm65_model.xml"
NPZ_PATH = PROJECT / "experiment_data" / "exp18_fig_assets" / "raw" / "a_hit_clean.npz"
OUT_PNG = PROJECT / "paper" / "figures" / "fig1_system_overview.png"


def main() -> None:
    """加载模型 → 置击球瞬间构型 → 渲染 → 存 PNG。"""
    d = np.load(NPZ_PATH)
    hit = int(d["hit_step"])
    # 用击球前 12 步（60 ms）的逼近瞬间：球与拍面分离可见，读作「击球瞬间」更清楚
    k = max(0, hit - 12)
    q_pose = d["q_actual"][k]
    ball_pos = d["ball_pos"][k]

    # 渲染需要 1400px 离屏缓冲（模型默认 640）：注入 visual/global 子句到临时副本，
    # 不改动 src/robot/rm65_model.xml（唯一事实来源）
    xml_text = MODEL_PATH.read_text(encoding="utf-8")
    clause = '<global offwidth="1400" offheight="900"/>'
    if "<visual>" in xml_text:
        xml_text = xml_text.replace("<visual>", "<visual>\n    " + clause, 1)
    else:
        xml_text = xml_text.replace(
            "<worldbody>", f"<visual>\n  {clause}\n</visual>\n<worldbody>", 1)
    # 临时副本须与原模型同目录（XML 内 mesh 引用为相对路径），渲染后删除
    tmp_path = MODEL_PATH.with_name("rm65_model_fig1_tmp.xml")
    tmp_path.write_text(xml_text, encoding="utf-8")

    try:
        model = load_mujoco_model(tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    data = mujoco.MjData(model)
    # 右臂 6 关节 + 左臂保持零位（qpos[6:12] 默认即 0）
    data.qpos[0:6] = q_pose
    # 球自由关节 qpos 布局为「位置在前、四元数在后」：
    # qpos[12:15] = xyz 位置，qpos[15:19] = (w, x, y, z) 四元数
    data.qpos[12:15] = ball_pos
    data.qpos[15:19] = [1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    # 隐藏 group 1 的碰撞几何（col_body/col_arm 默认红色球体）：
    # 它们不是视觉网格，画出来会让机械臂像一串红球，图读起来像渲染错误。
    # 可见性开关属于 mjvOption（渲染选项），不属于 model.vis
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0

    renderer = mujoco.Renderer(model, height=900, width=1400)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # 取景：以球/拍交互区为中心；俯角加大抬高地平线，压缩黑色天空占比
    # （旧版 distance=2.9/elevation=-6 时黑天空约占画面一半）
    cam.lookat[:] = np.array([-0.60, -0.62, 0.86])
    cam.distance = 2.25
    cam.azimuth = 166
    cam.elevation = -20
    renderer.update_scene(data, camera=cam, scene_option=opt)
    img = renderer.render()

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image, ImageEnhance
    pil = Image.fromarray(img)
    # 场景偏暗：适度提亮 + 增对比（仅展示用，不改物理）
    pil = ImageEnhance.Brightness(pil).enhance(1.30)
    pil = ImageEnhance.Contrast(pil).enhance(1.14)
    pil = _annotate(pil)
    pil.save(OUT_PNG)
    print(f"已保存 {OUT_PNG} ({pil.size[0]}x{pil.size[1]})")


def _annotate(pil):
    """裁剪 + 叠加图注（racket / ball / incoming serve）；纯展示标注，不改物理。

    像素坐标以 1400×900 渲染分辨率为基准（先裁掉顶部纯黑天空带，再标注）。
    """
    from PIL import ImageDraw, ImageFont
    pil = pil.crop((0, 95, 1400, 900))          # 去掉顶部纯黑天空带
    draw = ImageDraw.Draw(pil, "RGBA")
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 32)
    except OSError:
        font = ImageFont.load_default()

    def label(text: str, xy: tuple[int, int], target: tuple[int, int]) -> None:
        draw.line([xy, target], fill=(255, 255, 255, 200), width=3)
        bbox = draw.textbbox(xy, text, font=font)
        pad = 7
        draw.rectangle([bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
                       fill=(0, 0, 0, 160))
        draw.text(xy, text, fill=(255, 255, 255, 255), font=font)

    label("racket", (600, 455), (838, 335))
    label("ball", (300, 190), (556, 280))
    # 来球方向：虚线箭头指示发球方来向（serve box 为随机采样区域，不在画面内）
    draw.line([(360, 330), (505, 288)], fill=(255, 255, 255, 200), width=3)
    draw.line([(505 - 14, 288 - 6), (505, 288)], fill=(255, 255, 255, 230), width=4)
    draw.line([(505 - 8, 288 + 10), (505, 288)], fill=(255, 255, 255, 230), width=4)
    label("incoming serve", (120, 348), (368, 328))
    return pil



if __name__ == "__main__":
    main()
