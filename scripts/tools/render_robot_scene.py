"""渲染机器人场景图：全景 + 关键时刻动作序列（论文 Fig. 机器人场景）。

用法（mujoco_tennis 环境）:
    MUJOCO_GL=egl python scripts/tools/render_robot_scene.py \
        [--episode experiment_data/exp18_tcp_exempt/raw/full_seed001.npz]

输出: paper/figures/fig_robot_scene.png（全景 + 4 帧动作序列竖排合成）。

设计说明:
- 数据源为真实仿真 episode 的 q_actual/ball_pos/hit_step 记录，不伪造位姿;
- 全景用接触时刻位姿（球拍触球瞬间，最能展示任务配置）;
- 序列取 hit-55 / hit-25 / hit / hit+25 四帧（后摆→挥拍→触球→随挥）;
- 左臂按 episode 记录的 init_q_left 保持零位（视觉完整性，不参与任务）。
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
from src.utils.mujoco_loader import load_mujoco_model
MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"

# 输出分辨率（px）—— 合成图 7.2in 宽 @300dpi ≈ 2160px，帧按此比例分配
PAN_W, PAN_H = 2160, 1440          # 全景帧
SEQ_W, SEQ_H = 540, 360            # 序列单帧（4 帧横排 = 2160px）
SEQ_OFFSETS = (-55, -25, 0, 25)    # 相对 hit_step 的帧偏移
SEQ_LABELS = ("backswing", "approach", "contact", "follow-through")


def build_model_offscreen() -> tuple[mujoco.MjModel, str]:
    """构建带大离屏帧缓冲的模型包装 XML（不改动模型源文件）。"""
    wrapper = (
        "<mujoco>\n"
        f'  <include file="{MODEL_XML.as_posix()}"/>\n'
        f'  <visual><global offwidth="{PAN_W}" offheight="{PAN_H}"/></visual>\n'
        "</mujoco>\n"
    )
    tmp = Path(tempfile.mkdtemp()) / "wrap.xml"
    tmp.write_text(wrapper, encoding="utf-8")
    model = load_mujoco_model(str(tmp))
    return model, str(tmp)


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
    """设置自由相机。azimuth 相对 +X 轴（度），elevation 相对水平面（度）。"""
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation


def render_frame(renderer: mujoco.Renderer, data: mujoco.MjData,
                 cam: mujoco.MjvCamera) -> np.ndarray:
    """渲染一帧：隐藏碰撞胶囊（geom group 1）与标记点（site group 0）。"""
    opt = mujoco.MjvOption()
    opt.geomgroup[1] = 0  # 隐藏碰撞体可视化（col_body/col_arm 红蓝胶囊）
    opt.sitegroup[0] = 0  # 隐藏 hit_target 等标记点
    renderer.update_scene(data, camera=cam, scene_option=opt)
    return renderer.render()


def main() -> None:
    parser = argparse.ArgumentParser(description="渲染机器人场景图（全景+动作序列）")
    parser.add_argument(
        "--episode",
        type=Path,
        default=REPO_ROOT / "experiment_data" / "exp18_tcp_exempt" / "raw"
        / "full_seed040.npz",
        help="轨迹 NPZ（含 q_actual/ball_pos/hit_step/init_q_left）",
    )
    args = parser.parse_args()

    rec = np.load(args.episode, allow_pickle=True)
    md = json.loads(str(rec["metadata"]))
    hit_step = int(rec["hit_step"])
    print(f"episode: {args.episode.name}  hit_step={hit_step}  "
          f"hit_type={md.get('hit_type')}  ball_speed={md.get('ball_speed')}")

    model, wrap_path = build_model_offscreen()
    data = mujoco.MjData(model)
    pan_renderer = mujoco.Renderer(model, PAN_H, PAN_W)  # MuJoCo 参数序为 (height, width)
    seq_renderer = mujoco.Renderer(model, SEQ_H, SEQ_W)

    # 接触时刻的球/拍参考点（相机取景用）
    ball_hit = rec["ball_pos"][hit_step]
    tcp_hit = rec["tcp_pos"][hit_step]
    print(f"ball@hit={np.round(ball_hit, 3)}  tcp@hit={np.round(tcp_hit, 3)}")

    # 全景相机：框住机器人（立柱+双臂+球拍）与接触点，保留黑色 MuJoCo 背景
    cam = mujoco.MjvCamera()
    lookat_pan = np.array([0.18, -0.28, 1.0])
    configure_camera(cam, lookat_pan, distance=3.2, azimuth=-115.0, elevation=-12.0)

    # ── 全景：击球前 60 ms（球在拍前方可见，展示来球与机器人任务配置） ──
    pan_step = int(np.clip(hit_step - 12, 0, len(rec["q_actual"]) - 1))
    set_pose(data, rec["q_actual"][pan_step], rec["init_q_left"],
             rec["ball_pos"][pan_step])
    pan = render_frame(pan_renderer, data, cam)
    print("pan shape:", pan.shape)

    # ── 动作序列：更近的相机聚焦拍-球交互区 ──
    cam_seq = mujoco.MjvCamera()
    lookat_seq = np.array([-0.35, -0.45, 1.0])
    configure_camera(cam_seq, lookat_seq, distance=2.2, azimuth=-115.0,
                     elevation=-12.0)
    seq = []
    for off in SEQ_OFFSETS:
        i = int(np.clip(hit_step + off, 0, len(rec["q_actual"]) - 1))
        set_pose(data, rec["q_actual"][i], rec["init_q_left"], rec["ball_pos"][i])
        seq.append(render_frame(seq_renderer, data, cam_seq))
    print("seq shape:", seq[0].shape)

    # ── 合成：全景 + 4 帧横排，白底标签 ──
    margin = 8
    label_h = 30
    gap = 0  # 4 帧无缝横排恰好铺满全景宽度（4×SEQ_W = PAN_W）
    total_w = PAN_W
    total_h = PAN_H + margin + label_h + SEQ_H + label_h
    canvas = np.full((total_h, total_w, 3), 255, dtype=np.uint8)

    canvas[:PAN_H, :] = pan
    y = PAN_H + margin
    for k, (off, label) in enumerate(zip(SEQ_OFFSETS, SEQ_LABELS)):
        x0 = k * (SEQ_W + gap)
        x1 = x0 + SEQ_W
        canvas[y:y + SEQ_H, x0:x1] = seq[k][:, :(x1 - x0)]
        canvas[y + SEQ_H:y + SEQ_H + label_h, x0:x1] = (245, 245, 245)
    # 用 PIL 写阶段文字标签（白底黑字）
    from PIL import ImageDraw, ImageFont
    img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # 旧版 PIL 不支持 size 参数
        font = ImageFont.load_default()
    y_lab = PAN_H + margin + SEQ_H
    for k, label in enumerate(SEQ_LABELS):
        x0 = k * (SEQ_W + gap)
        draw.text((x0 + 12, y_lab + 4), f"{label}  ({SEQ_OFFSETS[k]:+d} ms)",
                  fill=(20, 20, 20), font=font)
    out = REPO_ROOT / "paper" / "figures" / "fig_robot_scene.png"
    img.save(out, dpi=(300, 300))
    print(f"saved: {out} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
