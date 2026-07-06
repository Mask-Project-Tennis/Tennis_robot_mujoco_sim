"""查看 tennis_rbt 自制机器人模型（交互式 MuJoCo 查看器）。

用途：打开 GUI 窗口交互查看修改后的双臂机器人，可旋转/缩放/平移视角。

用法：
    # 默认：V12 初始姿态（后摆待击）
    python scripts/tools/view_tennis_rbt.py

    # 零位姿态
    python scripts/tools/view_tennis_rbt.py --no-init-q

    # 指定其它模型（如对比原版 rm65）
    python scripts/tools/view_tennis_rbt.py --model src/robot/rm65_model.xml

Linux 需先设 MuJoCo 库路径：
    export LD_LIBRARY_PATH="$(python -c 'import mujoco,os;print(os.path.dirname(mujoco.__file__))'):$LD_LIBRARY_PATH"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import mujoco.viewer

# 将项目根目录加入 sys.path（脚本在 scripts/tools/ 下，需上溯三层）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.utils.mujoco_loader import load_mujoco_model  # noqa: E402

# V12 初始关节角（rad），见 scripts/rm65_mpc_v12.py:287-288
INIT_Q_RIGHT = [-1.5, 1.57, -0.236, 0.404, 0.446, 2.45]
INIT_Q_LEFT = [-0.373, -1.57, 0.236, -0.404, -0.446, -2.45]


def main() -> None:
    """加载模型并启动交互查看器。"""
    parser = argparse.ArgumentParser(description="查看 tennis_rbt 机器人模型")
    parser.add_argument(
        "--model",
        type=str,
        default="src/robot/tennis_rbt.xml",
        help="MuJoCo XML 模型路径（默认 tennis_rbt.xml）",
    )
    parser.add_argument(
        "--init-q",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="应用 V12 初始姿态（默认开启；--no-init-q 显示零位）",
    )
    parser.add_argument(
        "--step",
        action="store_true",
        help="启用物理步进（球会下落），默认静态便于观察",
    )
    args = parser.parse_args()

    model_path = Path(args.model).resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    model = load_mujoco_model(model_path)
    data = mujoco.MjData(model)

    # 设置关节姿态
    if args.init_q:
        data.qpos[0:6] = INIT_Q_RIGHT
        data.qpos[6:12] = INIT_Q_LEFT
        print("已应用 V12 初始姿态 init_q")
    else:
        print("零位姿态（q=0）")
    mujoco.mj_forward(model, data)

    # 打印关键信息
    racket = data.site("racket_center").xpos
    r_base = data.body("r_base_link1").xpos
    l_base = data.body("l_base_link1").xpos
    print(f"模型: {model_path.name}  nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"右臂底座={r_base.round(3)}  左臂底座={l_base.round(3)}")
    print(f"racket_center={racket.round(3)}")
    print("提示: 鼠标拖动旋转, 滚轮缩放, 右键平移; 关闭窗口退出。")

    # 启动交互查看器（阻塞至窗口关闭）
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            if args.step:
                mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
