"""关节调节查看器（位置控制版，RM65 / R1D 通用）。

默认（不带 ``--robot-config``）保持旧行为：把 RM65 XML 的 ``<actuator>``
片段替换为 ``position`` 执行器，双臂 12 个滑条 + 旧初始位姿。指定 R1D 描述
（``--robot-config configs/robots/r1d_right.yaml``）时经
``src.robot.model_factory.load_robot_model`` 按描述 + runtime + 球拍档案装配
**实际**模型（球拍几何/质量/惯量/安装来自 ``RacketProfile``，不再校验后丢弃），
再原地把已绑定执行器改为位置伺服（kp/kd 来自 runtime，滑条范围 = 模型
``jnt_range``，即 runtime 显式限位校验后的真实限位），得到 7 个滑条。

操作说明：
  - MuJoCo 右侧 Control 面板直接拖动滑条控制关节角度（弧度）
  - 按键 R 重置到初始位姿
  - 按键 P 打印当前关节角度（Python 数组格式）与 TCP 位置
  - 按键 F 切换坐标系显示（World / Racket Center / 关闭）
  - 红=X轴，绿=Y轴，蓝=Z轴

用法：
  python scripts/tools/rm65_joint_viewer.py
  python scripts/tools/rm65_joint_viewer.py \
      --robot-config configs/robots/r1d_right.yaml \
      --runtime-config configs/robots/r1d_v12_runtime_right.yaml \
      --racket-config configs/rackets/blade100_v9_reference.yaml
"""

import argparse
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import mujoco
import mujoco.viewer

from src.robot.robot_profile import (
    default_rm65_profile_path,
    load_robot_profile,
)
from src.robot.runtime_parameters import load_runtime_parameters
from src.utils.mujoco_loader import load_mujoco_model

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent

INIT_Q_RIGHT = np.array([0.373, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q_LEFT = np.array([0.0, -2.0, 0.7, 0.7, 1.0, 0.0], dtype=np.float64)

#: RM65 旧版 12 执行器片段（默认路径逐字保持，避免改变旧 viewer 行为）
POSITION_ACTUATORS = """  <actuator>
    <!-- 右臂：位置执行器，ctrl 直接控制目标关节角（弧度） -->
    <position name="pos_r_joint1" joint="r_joint1" ctrlrange="-3.11 3.11" kp="200" damping="20"/>
    <position name="pos_r_joint2" joint="r_joint2" ctrlrange="-2.27 2.27" kp="200" damping="20"/>
    <position name="pos_r_joint3" joint="r_joint3" ctrlrange="-2.36 2.36" kp="100" damping="10"/>
    <position name="pos_r_joint4" joint="r_joint4" ctrlrange="-3.11 3.11" kp="50" damping="5"/>
    <position name="pos_r_joint5" joint="r_joint5" ctrlrange="-2.23 2.23" kp="50" damping="5"/>
    <position name="pos_r_joint6" joint="r_joint6" ctrlrange="-6.28 6.28" kp="20" damping="2"/>
    <!-- 左臂 -->
    <position name="pos_l_joint1" joint="l_joint1" ctrlrange="-3.11 3.11" kp="200" damping="20"/>
    <position name="pos_l_joint2" joint="l_joint2" ctrlrange="-2.27 2.27" kp="200" damping="20"/>
    <position name="pos_l_joint3" joint="l_joint3" ctrlrange="-2.36 2.36" kp="100" damping="10"/>
    <position name="pos_l_joint4" joint="l_joint4" ctrlrange="-3.11 3.11" kp="50" damping="5"/>
    <position name="pos_l_joint5" joint="l_joint5" ctrlrange="-2.23 2.23" kp="50" damping="5"/>
    <position name="pos_l_joint6" joint="l_joint6" ctrlrange="-6.28 6.28" kp="20" damping="2"/>
  </actuator>"""


@dataclass(frozen=True)
class JointViewerSpec:
    """关节查看器构建结果（模型 + 关节/执行器/球地址元数据）。"""

    robot_id: str
    joint_names: tuple[str, ...]
    init_q: np.ndarray
    init_q_left: np.ndarray | None
    actuator_count: int
    ball_qpos_start: int
    ball_qvel_start: int
    tcp_site_name: str
    model: mujoco.MjModel


def _joint_qpos_address(model: mujoco.MjModel, joint_name: str) -> int:
    """按关节名解析 qpos 地址（缺失抛 ValueError）。"""
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"模型中不存在关节 '{joint_name}'")
    return int(model.jnt_qposadr[joint_id])


def _ball_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    """解析球 free joint 的 qpos/qvel 起始地址（找不到返回 (-1, -1)）。"""
    for name in ("ball_freejoint", "ball", "tennis_ball"):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0 and int(model.jnt_type[joint_id]) == int(
            mujoco.mjtJoint.mjJNT_FREE
        ):
            return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    return -1, -1


def _apply_position_servo(
    model: mujoco.MjModel,
    actuator_ids: tuple[int, ...] | list[int],
    kp: np.ndarray,
    kd: np.ndarray,
) -> None:
    """把已绑定执行器原地改为位置伺服（不重建模型/不丢球拍装配）。

    MuJoCo ``general`` 执行器的位置伺服力为
    ``kp*(ctrl-q) - kd*qdot``：``gainprm[0]=kp``、``biasprm=[0,-kp,-kd]``、
    ``biastype=mjBIAS_AFFINE``；``ctrlrange`` 取对应关节的 ``jnt_range``，
    使查看器滑条范围与真实限位一致。

    Args:
        model: 已编译模型（由 ``load_robot_model`` 按描述/球拍档案装配）。
        actuator_ids: 与 ``kp``/``kd`` 同序的执行器地址（``ModelBinding``）。
        kp: 位置增益，形状 (n,)，来自 runtime 参数。
        kd: 阻尼增益，形状 (n,)，来自 runtime 参数。

    Raises:
        ValueError: 执行器未绑定到单一关节或增益长度与执行器数不一致。
    """
    if len(actuator_ids) != len(kp) or len(actuator_ids) != len(kd):
        raise ValueError(
            "位置伺服增益维数与执行器数不一致: "
            f"actuators={len(actuator_ids)} kp={len(kp)} kd={len(kd)}"
        )
    for index, actuator_id in enumerate(actuator_ids):
        aid = int(actuator_id)
        joint_id = int(model.actuator_trnid[aid, 0])
        if joint_id < 0:
            raise ValueError(f"执行器 {aid} 未绑定到单一关节，无法配置位置伺服")
        gain = float(kp[index])
        model.actuator_gainprm[aid, :] = 0.0
        model.actuator_gainprm[aid, 0] = gain
        model.actuator_biasprm[aid, :] = 0.0
        model.actuator_biasprm[aid, 1] = -gain
        model.actuator_biasprm[aid, 2] = -float(kd[index])
        model.actuator_biastype[aid] = int(mujoco.mjtBias.mjBIAS_AFFINE)
        model.actuator_ctrlrange[aid, :] = model.jnt_range[joint_id]
        model.actuator_ctrllimited[aid] = 1


def _replace_actuator_section(xml_text: str, actuator_xml: str) -> str:
    """把 XML 中的 ``<actuator>`` 段替换为给定片段（不存在则追加）。"""
    if re.search(r"<actuator>.*?</actuator>", xml_text, flags=re.DOTALL):
        return re.sub(
            r"<actuator>.*?</actuator>", actuator_xml, xml_text, flags=re.DOTALL
        )
    return xml_text.replace("</mujoco>", f"{actuator_xml}\n</mujoco>")


def build_position_actuator_model(
    robot_config: Path | None = None,
    runtime_config: Path | None = None,
    racket_config: Path | None = None,
) -> JointViewerSpec:
    """构建 position 执行器模型并返回查看器元数据。

    RM65 分支保持旧行为：复制 XML 并把 12 个执行器替换为 position 片段。
    R1D 分支经 ``src.robot.model_factory.load_robot_model`` 按机器人描述 +
    runtime + 球拍档案装配**实际**模型（球拍几何/质量/惯量/安装全部来自
    ``RacketProfile``，不再校验后丢弃），再原地把已绑定的 torque motor 改为
    位置伺服（kp/kd/ctrlrange 来自 runtime 与真实关节限位）。

    Args:
        robot_config: 机器人描述 YAML；缺省 ``configs/robots/rm65.yaml``
            （与旧脚本一致：RM65 12 执行器 + 旧初始位姿/球位）。
        runtime_config: 运行参数 YAML（提供 init_q/kp/kd 与场景几何；
            R1D 分支必需）。
        racket_config: 球拍描述 YAML（仅 R1D 分支支持，作为球拍物理来源；
            RM65 分支没有对应的装配入口，显式给出会明确拒绝而不是静默忽略）。

    Returns:
        :class:`JointViewerSpec`（含编译后的 model）。

    Raises:
        ValueError: 模型/关节解析失败、R1D 缺 runtime、或机器人类型不支持
            球拍装配入口。
        FileNotFoundError: 配置、模型或球拍档案文件不存在。
    """
    profile_path = (
        Path(robot_config).resolve()
        if robot_config is not None
        else Path(default_rm65_profile_path()).resolve()
    )
    profile = load_robot_profile(profile_path)
    model_path = Path(profile.model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"机器人模型文件不存在: {model_path}")
    if profile.robot_id not in ("rm65", "unitree_r1d"):
        raise ValueError(
            f"关节查看器不支持 robot_id={profile.robot_id!r} 的模型装配"
            "（仅支持 rm65 / unitree_r1d）"
        )

    racket = None
    if racket_config is not None:
        from src.robot.racket_profile import load_racket_profile

        racket = load_racket_profile(Path(racket_config).resolve())

    runtime = (
        load_runtime_parameters(Path(runtime_config).resolve(), profile)
        if runtime_config is not None
        else None
    )

    names = tuple(profile.active_joint_names)

    if profile.robot_id == "rm65":
        if racket is not None:
            raise ValueError(
                "RM65 关节查看器不支持 --racket-config：该入口使用固定旧拍 XML，"
                "没有按 RacketProfile 装配模型的路径；请改用 R1D 配置，"
                "或不传 --racket-config"
            )
        xml_text = model_path.read_text(encoding="utf-8")
        new_xml = _replace_actuator_section(xml_text, POSITION_ACTUATORS)
        model = load_mujoco_model(model_path, xml_text=new_xml)
        ball_qpos_start, ball_qvel_start = _ball_addresses(model)
        return JointViewerSpec(
            robot_id=profile.robot_id,
            joint_names=names,
            init_q=INIT_Q_RIGHT.copy(),
            init_q_left=INIT_Q_LEFT.copy(),
            actuator_count=12,
            ball_qpos_start=ball_qpos_start,
            ball_qvel_start=ball_qvel_start,
            tcp_site_name=str(profile.tcp_site_name),
            model=model,
        )

    # R1D：实际装配模型（含 RacketProfile 注入）后原地改位置伺服
    if runtime is None:
        raise ValueError(
            "R1D 关节查看器需要 --runtime-config（scene/init_q/kp/kd 来源），"
            "不接受隐式缺省"
        )
    from src.robot.model_factory import load_robot_model

    bundle = load_robot_model(profile, runtime, racket_profile=racket)
    model = bundle.model
    kp = (
        np.asarray(runtime.kp, dtype=float)
        if runtime.kp is not None
        else np.full(len(names), 200.0)
    )
    kd = (
        np.asarray(runtime.kd, dtype=float)
        if runtime.kd is not None
        else np.full(len(names), 20.0)
    )
    _apply_position_servo(model, bundle.binding.actuator_ids, kp, kd)
    ball_qpos_start, ball_qvel_start = _ball_addresses(model)
    return JointViewerSpec(
        robot_id=profile.robot_id,
        joint_names=names,
        init_q=np.asarray(bundle.parameters.init_q, dtype=float).copy(),
        init_q_left=None,
        actuator_count=len(names),
        ball_qpos_start=ball_qpos_start,
        ball_qvel_start=ball_qvel_start,
        tcp_site_name=str(profile.tcp_site_name),
        model=model,
    )


def _build_parser() -> argparse.ArgumentParser:
    """构造 CLI 参数解析器。"""
    parser = argparse.ArgumentParser(
        description="关节调节查看器（位置执行器；默认 RM65，可选 R1D）"
    )
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=None,
        help="机器人描述 YAML（默认 configs/robots/rm65.yaml 旧行为）",
    )
    parser.add_argument(
        "--runtime-config",
        type=Path,
        default=None,
        help="可选运行参数 YAML（init_q/kp/kd/真实关节限位）",
    )
    parser.add_argument(
        "--racket-config",
        type=Path,
        default=None,
        help=(
            "球拍描述 YAML；R1D 分支作为球拍物理（几何/质量/惯量/安装）"
            "唯一来源装配模型；RM65 分支没有该装配入口，显式给出会报错"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """运行关节查看器。

    Args:
        argv: 可选命令行参数（默认取 ``sys.argv[1:]``）。

    Returns:
        进程退出码（0 正常）。
    """
    args = _build_parser().parse_args(argv)
    spec = build_position_actuator_model(
        args.robot_config, args.runtime_config, args.racket_config
    )
    model = spec.model
    data = mujoco.MjData(model)

    joint_qpos = [_joint_qpos_address(model, name) for name in spec.joint_names]
    if spec.init_q_left is not None:
        joint_qpos += [_joint_qpos_address(model, f"l_joint{i}") for i in range(1, 7)]

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[joint_qpos[: len(spec.joint_names)]] = spec.init_q
    if spec.init_q_left is not None:
        data.qpos[joint_qpos[len(spec.joint_names):]] = spec.init_q_left
    data.ctrl[:] = np.concatenate(
        [spec.init_q, spec.init_q_left]
        if spec.init_q_left is not None
        else [spec.init_q]
    )
    if spec.ball_qpos_start >= 0:
        data.qpos[spec.ball_qpos_start:spec.ball_qpos_start + 3] = [5.0, 0.0, 2.0]
        data.qpos[spec.ball_qpos_start + 3:spec.ball_qpos_start + 7] = [1, 0, 0, 0]
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, spec.tcp_site_name
    )
    frame_mode = [1]

    print("=" * 60)
    print(f"  关节调节查看器（位置控制版）robot={spec.robot_id}")
    print(f"  主动关节: {list(spec.joint_names)}")
    print(f"  Control 滑条数: {spec.actuator_count}")
    print(f"  初始位姿: {np.round(spec.init_q, 4).tolist()}")
    if spec.init_q_left is not None:
        print(f"  左臂初始位姿: {np.round(spec.init_q_left, 4).tolist()}")
    print("-" * 60)
    print("  操作:")
    print("    右侧 Control 面板拖动滑条 → 直接控制关节角（弧度）")
    print("    R  重置到初始位姿")
    print("    P  打印当前关节角度")
    print("    F  切换坐标系 (World → Racket → 关闭)")
    print("    红=X轴, 绿=Y轴, 蓝=Z轴")
    print("=" * 60)

    def reset_pose() -> None:
        """把关节/控制量重置到初始位姿。"""
        data.qpos[joint_qpos[: len(spec.joint_names)]] = spec.init_q
        data.qvel[joint_qpos[: len(spec.joint_names)]] = 0.0
        if spec.init_q_left is not None:
            data.qpos[joint_qpos[len(spec.joint_names):]] = spec.init_q_left
            data.qvel[joint_qpos[len(spec.joint_names):]] = 0.0
        data.ctrl[:] = np.concatenate(
            [spec.init_q, spec.init_q_left]
            if spec.init_q_left is not None
            else [spec.init_q]
        )
        mujoco.mj_forward(model, data)

    def key_callback(keycode: int) -> None:
        """查看器按键回调（R 重置 / P 打印 / F 坐标系）。"""
        if keycode == ord("R"):
            reset_pose()
            print("  [重置] 所有关节已回到初始位姿")
        elif keycode == ord("P"):
            q = np.array([data.qpos[index] for index in joint_qpos], dtype=float)
            p_ee = (
                data.site_xpos[site_id].copy()
                if site_id >= 0 else np.zeros(3)
            )
            print()
            print(f"  robot_id = {spec.robot_id!r}")
            print(f"  joint_names = {list(spec.joint_names)}")
            print(f"  init_q = np.array("
                  f"{np.round(q, 4).tolist()}, dtype=np.float64)")
            print(f"  球拍中心位置: {np.round(p_ee, 4).tolist()}")
            print()
        elif keycode == ord("F"):
            frame_mode[0] = (frame_mode[0] + 1) % 3
            labels = ["关闭", "世界坐标系 (World)", "球拍中心坐标系 (Racket)"]
            print(f"  [坐标系] {labels[frame_mode[0]]}")

    try:
        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer.cam.distance = 3.5
            viewer.cam.elevation = -15
            viewer.cam.azimuth = 135
            viewer.cam.lookat[:] = [0.0, 0.0, 1.0]

            while viewer.is_running():
                if frame_mode[0] == 0:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_NONE
                elif frame_mode[0] == 1:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD
                else:
                    viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

                mujoco.mj_step(model, data)
                viewer.sync()
                time.sleep(1.0 / 240.0)
    except Exception as exc:  # pragma: no cover - 无显示环境时提示
        print(f"  [viewer] 无法打开 MuJoCo 窗口（无显示环境？）：{exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
