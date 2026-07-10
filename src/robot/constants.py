"""RM-65B 机器人共享常量 — 单一事实来源。

所有物理常量（初始关节角度、肩部位置、工作空间半径、PD 增益）
集中在此模块，供 src/ 和 scripts/ 共同导入，消除跨模块字面量重复。

依赖方向：本模块仅依赖 numpy，不依赖任何 ilqt/real/sim 模块，
确保 scripts/ 可以直接导入而无需拉入真机或仿真层依赖。
"""

from __future__ import annotations

import numpy as np

# ── 仿真时间步长 ──
DT: float = 0.005

# ── 初始关节角度（弧度）──
# 仿真姿势：MuJoCo XML 关节范围宽（J2 ±130°），1.57 rad 无限制
# 设为只读，防止通过局部别名意外修改共享常量
INIT_Q = np.array([-1.5, 1.57, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q.flags.writeable = False

# 真机姿势：出厂限位 J2 ≤ 90°（configs/real_robot.yaml q_upper），
# 1.40 rad ≈ 80.2° 保留 8.8° 安全 margin（commit fe9c0c8）
INIT_Q_REAL = np.array([-1.5, 1.40, -0.236, 0.404, 0.446, 2.45], dtype=np.float64)
INIT_Q_REAL.flags.writeable = False

# 左臂初始姿势（对称镜像，不驱动保持零位）
INIT_Q_LEFT = np.array([-0.373, -1.57, 0.236, -0.404, -0.446, -2.45], dtype=np.float64)
INIT_Q_LEFT.flags.writeable = False

# ── 几何常量 ──
SHOULDER_POS = np.array([-0.1, -0.22693, 1.302645], dtype=np.float64)
SHOULDER_POS.flags.writeable = False
WORKSPACE_RADIUS: float = 0.90

# ── 位置模式 PD 增益（真机角度控制）──
KP = np.array([200.0, 200.0, 100.0, 50.0, 50.0, 20.0], dtype=np.float64)
KP.flags.writeable = False
KD = np.array([20.0, 20.0, 10.0, 5.0, 5.0, 2.0], dtype=np.float64)
KD.flags.writeable = False

# NOTE: 以下为共享只读数组；不要重新绑定（如 init_q = init_q + offset），
# 重新赋值会断开别名共享，使 .flags.writeable 保护失效。
__all__ = [
    "DT",
    "INIT_Q",
    "INIT_Q_REAL",
    "INIT_Q_LEFT",
    "SHOULDER_POS",
    "WORKSPACE_RADIUS",
    "KP",
    "KD",
]
