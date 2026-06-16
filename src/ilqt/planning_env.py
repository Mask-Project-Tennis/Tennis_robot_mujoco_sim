"""MPC 规划计算环境 — 基于 MuJoCo 的正运动学/雅可比/前向仿真。

供 iLQR 规划使用：给定关节状态和候选控制序列，模拟未来 N 步的
末端位置/速度/雅可比，用于优化最优轨迹。

本模块不接触真机硬件。真机关节状态的读取和指令发送由
RobotInterface 负责，球感知由 BallPerceiver 负责。

与 RM65Env 的核心区别：
  - 无球物理（无 _handle_ball_bounce / BALL_* 常量）
  - 无左臂 PD 维持（左臂固定零位，不影响右臂动力学）
  - 无估计器/预处理（感知由 BallPerceiver 外部处理）
  - 初始化时禁用所有 MuJoCo 碰撞（碰撞由控制器固件负责）
  - 支持力矩模式和位置模式（真机默认位置模式）
"""

import logging
from pathlib import Path

import numpy as np
import mujoco
from numpy.typing import NDArray

from src.utils.mujoco_loader import load_mujoco_model

logger = logging.getLogger(__name__)


class PlanningEnv:
    """MPC 规划计算环境 — MuJoCo 纯计算（单臂 + 球拍）。

    实现 RobotEnv Protocol 接口，加载 MuJoCo 模型做正运动学/雅可比/
    前向仿真，供 iLQR 规划使用。不接触真机硬件。
    真机模型就绪后换 model_path 即可。

    Attributes:
        NQ: 右臂关节数 = 6。
        NX: 右臂状态维度 = 12。
        NU: 右臂控制维度 = 6。
    """

    NQ: int = 6
    NX: int = 12
    NU: int = 6

    def __init__(
        self,
        model_path: Path | str | None = None,
        dt: float = 0.005,
    ) -> None:
        """初始化真机环境。

        Args:
            model_path: MuJoCo 模型路径。None 时使用默认 rm65_model.xml。
            dt: 仿真时间步长（秒）。
        """
        if model_path is None:
            model_path = Path(__file__).resolve().parent.parent / "robot" / "rm65_model.xml"

        self.model = load_mujoco_model(model_path)
        if dt is not None:
            self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        # 缓存 site/body ID
        self.racket_center_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
        )
        self.racket_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "r_racket_body"
        )

        # 禁用所有 MuJoCo 碰撞（碰撞由控制器固件负责）
        self.model.geom_contype[:] = 0
        self.model.geom_conaffinity[:] = 0
        logger.info("PlanningEnv 已禁用所有 MuJoCo 碰撞（碰撞由控制器固件负责）")

        # 保存原始力矩模式参数（configure_actuator_mode 切换时恢复用）
        self._torque_ctrlrange = self.model.actuator_ctrlrange[: self.NU].copy()
        self._torque_gainprm = self.model.actuator_gainprm[: self.NU].copy()
        self._torque_biasprm = self.model.actuator_biasprm[: self.NU].copy()
        self._torque_biastype = self.model.actuator_biastype[: self.NU].copy()
        self._torque_forcerange = self.model.actuator_forcerange[: self.NU].copy()
        self._actuator_mode: int = 0  # 0=力矩, 1=位置
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._use_feedforward: bool = False

        # 雅可比缓存
        self._jacp_cache: np.ndarray | None = None
        self._jacr_cache: np.ndarray | None = None

        mujoco.mj_forward(self.model, self.data)

    # ── 状态读写 ──

    def reset(self, q0: np.ndarray | None = None) -> np.ndarray:
        """重置仿真状态。

        Args:
            q0: 右臂初始关节角度 (6,)。None 时用零位。

        Returns:
            初始右臂状态 (12,)。
        """
        mujoco.mj_resetData(self.model, self.data)
        if q0 is not None:
            self.data.qpos[: self.NQ] = q0
        self.data.qvel[: self.NQ] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()
        return self.get_arm_state()

    def get_arm_state(self) -> NDArray[np.floating]:
        """获取右臂状态 [q(6), qdot(6)]，形状 (12,)。"""
        return np.concatenate(
            [self.data.qpos[: self.NQ].copy(), self.data.qvel[: self.NQ].copy()]
        )

    def set_arm_state(self, x: NDArray[np.floating]) -> None:
        """设置右臂状态 [q, qdot] 并刷新运动学。

        Args:
            x: 右臂状态，形状 (12,)。
        """
        self.data.qpos[: self.NQ] = x[: self.NQ]
        self.data.qvel[: self.NQ] = x[self.NQ:]
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()

    def update_kinematics(self) -> None:
        """轻量运动学刷新（不计算雅可比/惯性）。"""
        mujoco.mj_kinematics(self.model, self.data)

    # ── 前向仿真 ──

    def step(self, u: NDArray[np.floating]) -> NDArray[np.floating]:
        """施加右臂控制并前进一步。

        力矩模式: u = 力矩。
        位置模式: u = 期望角度，裁剪位置误差后等效于 forcerange 力矩限制。

        Args:
            u: 右臂控制输入 (6,)。力矩模式为力矩，位置模式为期望角度。

        Returns:
            新的右臂状态 (12,)。
        """
        ctrl_range = self.model.actuator_ctrlrange[: self.NU]
        u_clipped = np.clip(u, ctrl_range[:, 0], ctrl_range[:, 1])

        # 位置模式：裁剪位置误差，等效于 forcerange 力矩限制
        if self._actuator_mode == 1 and self._kp is not None:
            q_now = self.data.qpos[: self.NQ].copy()
            max_err = np.abs(self._torque_ctrlrange[:, 1]) / self._kp
            err = u_clipped - q_now
            err_clipped = np.clip(err, -max_err, max_err)
            u_clipped = q_now + err_clipped

        self.data.ctrl[: self.NU] = u_clipped

        # 左臂固定零位（防止漂移干扰右臂动力学）
        self.data.qpos[self.NQ : self.NQ + 6] = 0.0
        self.data.qvel[self.NQ : self.NQ + 6] = 0.0

        # 前馈补偿：位置模式下计算偏置力 h(q,qdot) = C*qdot + g(q)
        if self._actuator_mode == 1 and self._use_feedforward:
            self.data.qacc[:] = 0.0
            tau_bias = np.zeros(self.model.nv)
            mujoco.mj_rne(self.model, self.data, 0, tau_bias)
            self.data.qfrc_applied[: self.NQ] = tau_bias[: self.NQ]
        else:
            self.data.qfrc_applied[: self.NQ] = 0.0

        mujoco.mj_step(self.model, self.data)
        self._invalidate_jacobian_cache()
        return self.get_arm_state()

    def step_from_state(
        self, x: NDArray[np.floating], u: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """从指定状态出发，施加控制并前进一步。

        Args:
            x: 右臂状态 [q, qdot] (12,)。
            u: 控制输入 (6,)。
        """
        self.set_arm_state(x)
        return self.step(u)

    # ── 末端执行器 ──

    def _ensure_jacobians(self) -> None:
        """惰性计算并缓存球拍中心雅可比矩阵。"""
        if self._jacp_cache is not None:
            return
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.racket_center_id)
        self._jacp_cache = jacp[:, : self.NQ].copy()
        self._jacr_cache = jacr[:, : self.NQ].copy()

    def _invalidate_jacobian_cache(self) -> None:
        """清除雅可比缓存。"""
        self._jacp_cache = None
        self._jacr_cache = None

    def get_ee_pos(self) -> NDArray[np.floating]:
        """获取球拍中心世界坐标位置 (3,)。"""
        return self.data.site_xpos[self.racket_center_id].copy()

    def get_ee_vel(self) -> NDArray[np.floating]:
        """获取球拍中心线速度 (3,)。"""
        self._ensure_jacobians()
        assert self._jacp_cache is not None
        return (self._jacp_cache @ self.data.qvel[: self.NQ]).copy()

    def get_ee_angular_vel(self) -> NDArray[np.floating]:
        """获取球拍中心角速度 (3,)。"""
        self._ensure_jacobians()
        assert self._jacr_cache is not None
        return (self._jacr_cache @ self.data.qvel[: self.NQ]).copy()

    def get_ee_jacp(self) -> NDArray[np.floating]:
        """获取球拍中心位置雅可比 (3, 6)。"""
        self._ensure_jacobians()
        assert self._jacp_cache is not None
        return self._jacp_cache.copy()

    def get_ee_jacr(self) -> NDArray[np.floating]:
        """获取球拍中心旋转雅可比 (3, 6)。"""
        self._ensure_jacobians()
        assert self._jacr_cache is not None
        return self._jacr_cache.copy()

    def get_ee_normal(self) -> NDArray[np.floating]:
        """获取球拍面法向量（球拍局部 X 轴在世界坐标系中的方向）。

        Returns:
            法向量 (3,)，单位向量。
        """
        R = self.data.xmat[self.racket_body_id].reshape(3, 3)
        normal = R[:, 0].copy()
        normal /= np.linalg.norm(normal) + 1e-12
        return normal

    # ── 逆运动学 ──

    def solve_ik(
        self,
        target_pos: NDArray[np.floating],
        q_init: NDArray[np.floating] | None = None,
        max_iter: int = 200,
        eps: float = 1e-3,
        damp: float = 1e-6,
        step_size: float = 0.1,
    ) -> NDArray[np.floating]:
        """阻尼最小二乘逆运动学求解器。

        Args:
            target_pos: 目标末端位置 (3,)。
            q_init: 初始关节角度 (6,)。None 时用零位。
            max_iter: 最大迭代次数。
            eps: 收敛阈值（位置误差，米）。
            damp: 阻尼因子。
            step_size: 步长缩放。

        Returns:
            求解的关节角度 (6,)。
        """
        if q_init is None:
            q = np.zeros(self.NQ)
        else:
            q = q_init.copy()

        for _ in range(max_iter):
            x = np.zeros(self.NX)
            x[: self.NQ] = q
            self.set_arm_state(x)

            p_ee = self.get_ee_pos()
            err = target_pos - p_ee
            if np.linalg.norm(err) < eps:
                break

            J = self.get_ee_jacp()
            JJT = J @ J.T + damp * np.eye(3)
            dq = J.T @ np.linalg.solve(JJT, err)
            q = q + step_size * dq

        return q

    # ── 执行器模式 ──

    @property
    def dt(self) -> float:
        """仿真时间步长。"""
        return self.model.opt.timestep

    @property
    def actuator_mode(self) -> int:
        """执行器模式：0=力矩, 1=位置。"""
        return self._actuator_mode

    @property
    def kp(self) -> np.ndarray | None:
        """位置模式比例增益，力矩模式下为 None。"""
        return self._kp

    @property
    def kd(self) -> np.ndarray | None:
        """位置模式速度增益，力矩模式下为 None。"""
        return self._kd

    @property
    def use_feedforward(self) -> bool:
        """位置模式下是否启用前馈补偿。"""
        return self._use_feedforward

    def configure_feedforward(self, enabled: bool) -> None:
        """启用或禁用前馈补偿。

        Args:
            enabled: True=启用, False=禁用。
        """
        self._use_feedforward = enabled
        if not enabled:
            self.data.qfrc_applied[: self.NQ] = 0.0

    def configure_actuator_mode(
        self,
        mode: str,
        kp: NDArray[np.floating] | None = None,
        kd: NDArray[np.floating] | None = None,
    ) -> None:
        """配置执行器模式。

        Args:
            mode: "torque" 或 "position"。
            kp: (6,) 位置增益。position 模式必须提供。
            kd: (6,) 速度增益。position 模式必须提供。
        """
        if mode == "torque":
            self._actuator_mode = 0
            self.model.actuator_ctrlrange[: self.NU] = self._torque_ctrlrange
            self.model.actuator_gainprm[: self.NU] = self._torque_gainprm
            self.model.actuator_biasprm[: self.NU] = self._torque_biasprm
            self.model.actuator_biastype[: self.NU] = self._torque_biastype
            self.model.actuator_forcerange[: self.NU] = self._torque_forcerange
            self._kp = None
            self._kd = None
            self._use_feedforward = False
            self.data.qfrc_applied[: self.NQ] = 0.0

        elif mode == "position":
            if kp is None or kd is None:
                raise ValueError("位置模式必须提供 kp 和 kd")
            kp = np.asarray(kp, dtype=np.float64).reshape(self.NU)
            kd = np.asarray(kd, dtype=np.float64).reshape(self.NU)
            self._actuator_mode = 1
            self._kp = kp.copy()
            self._kd = kd.copy()
            self._use_feedforward = True

            for i in range(self.NU):
                self.model.actuator_gainprm[i, 0] = kp[i]
                self.model.actuator_biasprm[i, 0] = 0.0
                self.model.actuator_biasprm[i, 1] = -kp[i]
                self.model.actuator_biasprm[i, 2] = -kd[i]
                self.model.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
                jnt_id = self.model.actuator_trnid[i, 0]
                self.model.actuator_ctrlrange[i] = self.model.jnt_range[jnt_id]
        else:
            raise ValueError(f"未知执行器模式: {mode}")

    def clone_actuator_config(self, target_env: "PlanningEnv") -> None:
        """将当前 env 的执行器配置复制到目标 env。

        Args:
            target_env: 目标 PlanningEnv 实例。
        """
        if self._actuator_mode == 0:
            target_env.configure_actuator_mode("torque")
        else:
            target_env.configure_actuator_mode("position", self._kp, self._kd)
            target_env.configure_feedforward(self._use_feedforward)
