"""渲染 Fig.2 单帧全轨迹评估场景图（论文 Experiments 场景图）。

用法（mujoco_tennis 环境）:
    MUJOCO_GL=egl python scripts/tools/render_robot_scene.py \
        [--episode experiment_data/exp18_tcp_exempt/raw/full_seed040.npz]

输出: paper/figures/fig2_ball_scene.png（单帧：完整发球-弹跳-击球-回球轨迹）。

设计说明:
- 数据源为真实仿真 episode 的 q_actual/ball_pos/hit_step 记录，不伪造位姿;
- 单帧相机拉远，框住完整球轨迹（发球区 y≈-8 m → 弹跳 → 击球 → 回球）;
- 轨迹尾迹: 沿 ball_pos 每 3 步放置真实球径的小半透明黄球（回球段更淡）;
- 击球点: 模型球置于触球位置 + 半透明红色高亮球包裹; 弹跳点: 白色小球;
- 机器人/球拍位姿取击球帧（拍面与球在接触点相遇）;
- 左臂按 episode 记录的 init_q_left 保持零位（视觉完整性，不参与任务）;
- 灯光提亮 + PIL 亮化暗像素，保证印刷可读。
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageEnhance

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.utils.mujoco_loader import load_mujoco_model
MODEL_XML = REPO_ROOT / "src" / "robot" / "rm65_model.xml"

# 输出分辨率（px）—— 通栏图 7.16in 宽 @300dpi ≈ 2150px；
# 3:1 画幅 + 长焦（fovy 14°）使完整轨迹占据画面宽度约 78%
PAN_W, PAN_H = 2160, 728
TRAIL_STEP = 3  # 轨迹尾迹取样间隔（步；5ms×3=15ms，7m/s 下间距约 10.5cm）


def build_model_offscreen(rec: np.ndarray, fovy: float) -> tuple[mujoco.MjModel, str]:
    """构建带大离屏帧缓冲 + 轨迹尾迹 geoms 的模型包装 XML（不改动模型源文件）。"""
    bp = rec["ball_pos"]
    hs = int(rec["hit_step"])
    geoms: list[str] = []
    # 轨迹：每 TRAIL_STEP 步一个真实球径小黄球；回球段（hit 之后）更淡
    for i in range(0, len(bp), TRAIL_STEP):
        p = bp[i]
        rgba = "1 0.82 0.25 0.55" if i <= hs else "1 0.95 0.6 0.4"
        geoms.append(
            f'    <geom type="sphere" size="0.033" pos="{p[0]:.4f} {p[1]:.4f} '
            f'{p[2]:.4f}" rgba="{rgba}" contype="0" conaffinity="0"/>'
        )
    # 弹跳点白球（击球前 z 最小处）
    ib = int(np.argmin(bp[:hs, 2]))
    pb = bp[ib]
    geoms.append(
        f'    <geom type="sphere" size="0.045" pos="{pb[0]:.4f} {pb[1]:.4f} '
        f'{pb[2]:.4f}" rgba="1 1 1 0.9" contype="0" conaffinity="0"/>'
    )
    # 击球点红色高亮（包裹触球球体）
    ph = bp[hs]
    geoms.append(
        f'    <geom type="sphere" size="0.055" pos="{ph[0]:.4f} {ph[1]:.4f} '
        f'{ph[2]:.4f}" rgba="0.95 0.2 0.15 0.5" contype="0" conaffinity="0"/>'
    )
    trail = "\n".join(geoms)
    wrapper = (
        "<mujoco>\n"
        f'  <include file="{MODEL_XML.as_posix()}"/>\n'
        "  <visual>\n"
        f'    <global offwidth="{PAN_W}" offheight="{PAN_H}" fovy="{fovy}"/>\n'
        '    <headlight ambient="0.45 0.45 0.45" diffuse="0.75 0.75 0.75" '
        'specular="0.15 0.15 0.15"/>\n'
        "  </visual>\n"
        "  <worldbody>\n"
        '    <light name="key_top" pos="0.3 -4.0 6.0" dir="0 -0.2 -1" '
        'directional="true" diffuse="0.6 0.6 0.6" specular="0.2 0.2 0.2" '
        'castshadow="false"/>\n'
        '    <light name="fill_side" pos="-8 2.0 3.0" dir="0.6 -0.15 -0.35" '
        'directional="true" diffuse="0.35 0.35 0.35" specular="0.05 0.05 0.05" '
        'castshadow="false"/>\n'
        f"{trail}\n"
        "  </worldbody>\n"
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
    """设置自由相机。

    MuJoCo 约定：azimuth/elevation 描述**视线方向**（相机位于 lookat 反向
    distance 处）——视线方向 = (cos az·cos el, sin az·cos el, sin el)。
    因此 elevation 为负表示俯视（相机在场景上方），为正表示仰视。
    """
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
    parser = argparse.ArgumentParser(description="渲染 Fig.2 单帧全轨迹场景图")
    parser.add_argument(
        "--episode",
        type=Path,
        default=REPO_ROOT / "experiment_data" / "exp18_tcp_exempt" / "raw"
        / "full_seed040.npz",
        help="轨迹 NPZ（含 q_actual/ball_pos/hit_step/init_q_left）",
    )
    # 相机参数（改角度只需 CLI，不改代码）
    parser.add_argument("--azimuth", type=float, default=10.0,
                        help="视线方位角（度，相对 +X 轴；0=沿 +x 看）")
    parser.add_argument("--elevation", type=float, default=-6.0,
                        help="视线俯仰（度；负值=俯视，相机在上方）")
    parser.add_argument("--distance", type=float, default=13.0,
                        help="相机距离（m）")
    parser.add_argument("--fovy", type=float, default=14.0,
                        help="竖向视场角（度）——长焦压缩透视，使轨迹铺满画幅")
    parser.add_argument("--lookat", type=float, nargs=3,
                        default=[0.5, -4.0, 1.05],
                        metavar=("X", "Y", "Z"), help="相机注视点（世界系）")
    parser.add_argument("--out", type=str,
                        default="paper/figures/fig2_ball_scene.png",
                        help="输出 PNG 路径（相对仓库根）")
    args = parser.parse_args()

    rec = np.load(args.episode, allow_pickle=True)
    md = json.loads(str(rec["metadata"]))
    hit_step = int(rec["hit_step"])
    print(f"episode: {args.episode.name}  hit_step={hit_step}  "
          f"hit_type={md.get('hit_type')}  ball_speed={md.get('ball_speed')}")
    ball_hit = rec["ball_pos"][hit_step]
    tcp_hit = rec["tcp_pos"][hit_step]
    print(f"ball@hit={np.round(ball_hit, 3)}  tcp@hit={np.round(tcp_hit, 3)}")

    model, wrap_path = build_model_offscreen(rec, args.fovy)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, PAN_H, PAN_W)  # MuJoCo 参数序为 (height, width)

    # 击球帧位姿：球与拍在接触点相遇
    set_pose(data, rec["q_actual"][hit_step], rec["init_q_left"],
             rec["ball_pos"][hit_step])

    # 单帧相机：拉远框住完整轨迹（默认从 -x/+y 侧看，轨迹横向铺满）
    cam = mujoco.MjvCamera()
    lookat = np.array(args.lookat, dtype=float)
    configure_camera(cam, lookat, distance=args.distance,
                     azimuth=args.azimuth, elevation=args.elevation)

    pan = render_frame(renderer, data, cam)
    print("pan shape:", pan.shape)

    # 暗像素亮化（温和：亮度+对比度各一提，保证印刷可读）
    img = Image.fromarray(pan)
    img = ImageEnhance.Brightness(img).enhance(1.18)
    img = ImageEnhance.Contrast(img).enhance(1.05)

    out = REPO_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, dpi=(300, 300))
    print(f"saved: {out} ({img.width}x{img.height})")


if __name__ == "__main__":
    main()
