#!/usr/bin/env python3
"""子步接触敏感性检查（chat14 审稿 P4 补充验证，一次性工具）。

问题：5 ms 步长下 15 m/s 的球单步移动 7.5 cm，超过拍椭球半厚 2.5 cm + 球半径
3.3 cm，理论上存在隧穿风险（球在一个积分步内跳过拍面，步末穿透被 MuJoCo 记作
接触；或反之，连续路径不穿过拍面而离散步末落点落入拍内）。

方法（纯几何、无动力学回放）：直接用实跑记录的轨迹 (q_actual[k], ball_pos[k])，
把拍面按 q_actual[k] 摆位（MuJoCo 真实 geom），球在两帧之间以 0.5 ms 细采样
（15 m/s × 0.5 ms = 7.5 mm < 拍+球联合厚度 ~6 cm），逐点调用 mj_collision
检查是否产生 ball×racket 接触。对照实跑裁决：
- 实跑命中：细采样下接触应存在且帧号与实跑 hit_step 接近（±2 帧内）；
- 实跑未命中：细采样下不应出现接触（若出现 → 5 ms 积分漏检真实接触）。

用法（mujoco_tennis 环境）:
    python scripts/test/check_substep_sensitivity.py [npz ...]

默认扫描 exp18 图资产（9 m/s 命中/未命中）、exp18_tcp_exempt（7 m/s）与
exp28_substep/raw（15 m/s）下的全部 npz。只读，不写文件。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.sim.rm65_env import RM65Env
from src.utils.mujoco_loader import load_mujoco_model

MODEL_XML = REPO / "src" / "robot" / "rm65_model.xml"
NSUB = 10          # 每 5 ms 帧内 10 个几何采样点（0.5 ms 细粒度）


def _ball_racket_contact(env: RM65Env) -> bool:
    """当前 ncon 中是否存在 ball×racket 接触对。"""
    for ci in range(env.data.ncon):
        c = env.data.contact[ci]
        g1 = env.model.geom(c.geom1).name
        g2 = env.model.geom(c.geom2).name
        if ("ball" in g1 or "ball" in g2) and ("racket" in g1 or "racket" in g2):
            return True
    return False


def check_one(npz_path: Path) -> dict:
    """几何细采样接触核对（不依赖动力学回放）。"""
    rec = np.load(npz_path, allow_pickle=True)
    md = json.loads(str(rec["metadata"]))
    q_actual, ball_pos = rec["q_actual"], rec["ball_pos"]
    n = len(q_actual)
    live_hit = int(rec["hit_step"])
    live_type = md.get("hit_type", "?")

    env = RM65Env(str(MODEL_XML), dt=0.005)
    env.reset(q_actual[0])
    env.data.qpos[env.NQ:env.NQ + env.LEFT_ARM_NQ] = rec["init_q_left"]
    env.set_arm_collision(False)         # 先禁用以初始化 _arm_contype_save
    env.set_arm_collision(True)          # 几何核对需要臂-拍 geom 碰撞参与
    bq = env.NQ + env.LEFT_ARM_NQ        # 球自由关节 qpos 起点（右臂 6 + 左臂 6）

    contacts: list[tuple[int, int]] = []  # (帧, 子采样序号)
    for k in range(n - 1):
        env.data.qpos[:env.NQ] = q_actual[k]
        for j in range(1, NSUB + 1):
            alpha = j / NSUB
            pos = ball_pos[k] + alpha * (ball_pos[k + 1] - ball_pos[k])
            env.data.qpos[bq:bq + 3] = pos
            env.data.qpos[bq + 3:bq + 7] = [1.0, 0.0, 0.0, 0.0]
            # 球位姿更新后必须先刷新运动学（geom 位置），再算碰撞
            mujoco.mj_kinematics(env.model, env.data)
            mujoco.mj_collision(env.model, env.data)
            if _ball_racket_contact(env):
                contacts.append((k, j))
                break                      # 该帧已确认接触，进入下一帧

    first_k = contacts[0][0] if contacts else -1
    if live_hit >= 0:
        ok = first_k >= 0 and abs(first_k - live_hit) <= 2
        verdict = ("一致" if ok else
                   ("未检出接触（细采样）" if first_k < 0 else
                    f"接触偏移 {first_k - live_hit:+d} 帧"))
    else:
        ok = first_k < 0
        verdict = "一致(无接触)" if ok else f"细采样检出接触@{first_k - (-1):+d}"
    return {
        "file": npz_path.name,
        "speed": md.get("ball_speed"),
        "live": live_type,
        "live_hit_step": live_hit,
        "geom_first_k": first_k,
        "n_contact_frames": len(contacts),
        "ok": ok,
        "verdict": verdict,
    }


def main() -> None:
    """扫描默认目录 + 命令行给出的 npz，打印核对表。"""
    paths: list[Path] = []
    for pat in sys.argv[1:]:
        paths.extend(sorted(Path(pat).glob("*.npz")) if "*" in pat else [Path(pat)])
    if not paths:
        for d in ("experiment_data/exp18_fig_assets/raw",
                  "experiment_data/exp18_tcp_exempt/raw",
                  "experiment_data/exp28_substep/raw"):
            dd = REPO / d
            if dd.exists():
                paths.extend(sorted(dd.glob("*.npz")))
    print(f"{'file':28s} {'m/s':>4s} {'live':>8s} {'hit_step':>8s} {'几何接触':>8s}  verdict")
    bad = 0
    for p in paths:
        r = check_one(p)
        mark = "OK" if r["ok"] else "!!"
        if not r["ok"]:
            bad += 1
        print(f"{r['file']:28s} {r['speed']!s:>4s} {r['live']:>8s} "
              f"{r['live_hit_step']:>8d} {r['geom_first_k']:>8d}  {r['verdict']} [{mark}]")
    print(f"\n{len(paths)} 条轨迹，异常 {bad} 条")


if __name__ == "__main__":
    main()
