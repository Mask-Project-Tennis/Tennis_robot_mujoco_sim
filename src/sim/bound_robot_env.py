"""BoundRobotEnv — 绑定驱动的维度无关 RobotEnv 实现（阶段 3 · Task 3.3）。

职责边界（施工计划 §共享接口契约）：
- 完全按 ``ModelBinding`` 的名称地址访问状态/执行器/TCP：主动关节的
  qpos/qvel/ctrl 地址来自绑定（支持 free joint 位于主动关节之前、臂链
  树序/执行器序乱序的模型），n=6/7 通吃，不依赖 ModelKinematics 私有缓存；
- 实现现有 ``RobotEnv`` 协议的状态/推进/FK/IK 方法；``NQ/NX/NU`` 是
  实例属性（构造时按绑定维数确定）；
- 新增 ``control_range``：按 ``binding.actuator_ids`` 返回 (n,2) 副本；
- ``configure_actuator_mode(mode, kp, kd)`` / ``configure_feedforward(enabled)``：
  执行器**仅按 actuator_ids 配置**，实际应用限界（不止配置对象里有值）：
  - 力矩模式：ctrlrange/forcerange = ±torque_max（ctrllimited/forcelimited
    置位）+ 步进软件裁剪，力矩上限来自 runtime ``torque_max``；模型显式
    声明的执行器范围可作回退来源，两者皆无则拒绝构造（不静默无限制）；
  - 位置模式：MuJoCo ``general`` 执行器内部 PD（tau = Kp*(u-q) - Kd*qdot，
    gainprm/biasprm/biastype 落位）+ ctrlrange = 所驱动关节 jnt_range +
    forcerange = ±torque_max + 软件位置误差裁剪 |u-q| ≤ torque_max/kp；
  - mode 往返恢复力矩模式构造态配置；异常输入在步进前拒绝。

``step_from_state`` 副作用约定（隔离契约，消费方不得依赖额外副作用）：
- 使用**私有 scratch MjData**，live data/time/ctrl/qfrc_applied/qacc_warmstart
  全部不变；输入数组不被修改；重复输入输出相同；
- scratch 每次从规定的环境基态重置（力矩模式配方，与测试对照基准一致）：
  1. ``mj_resetData``（time=0、ctrl=0、qfrc_applied=0、warmstart=0、
     旁观关节回 qpos0）；
  2. 保持关节 ← held_q / 零速；
  3. 主动关节 qpos/qvel ← x；
  4. ``ctrl[actuator_ids]`` ← 按当前模式裁剪后的 u（力矩模式 =
     clip(u, ±torque_max)；位置模式 = 关节限位裁剪 + 误差裁剪）；位置
     模式启用前馈时：``mj_forward`` → ``mj_rne`` 偏置力 → 按剩余
     力矩余量限幅后写入 ``qfrc_applied[qvel_indices]``（PD+前馈
     总驱动力矩 ≤ ±torque_max）；
  5. ``mj_step``；
  6. 保持关节再钉回 held_q/零速（步内漂移不累积）。
- 返回后**不承诺** live FK 等于预测终态；消费者若需要应显式
  ``set_arm_state``。

本类不线程安全：异步重规划须为工作线程构建独立环境实例（3.4）。
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.robot.model_binding import ModelBinding
from src.robot.model_factory import RobotModelBundle

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]


class BoundRobotEnv:
    """按 ModelBinding 名称地址访问的维度无关机器人环境。

    用法::

        env = BoundRobotEnv(bundle)          # bundle: RobotModelBundle
        env.set_arm_state(x)                 # x = [q(n), qdot(n)]
        x_next = env.step_from_state(x, u)   # 预测一步（live 不变）
        x_live = env.step(u)                 # live 前进一步
        jacp = env.get_ee_jacp()             # (3, n)，列序 = binding.qvel_indices

    Attributes:
        NQ: 主动关节数 n（实例属性）。
        NX: 状态维数 2n（实例属性）。
        NU: 控制维数 n（实例属性）。
        model: MuJoCo 模型（来自 bundle，构造后执行器范围按模式改写）。
        data: live MuJoCo 数据（step/set_arm_state 操作的对象）。
    """

    # 维度注解（实例属性，构造时按绑定维数赋值）
    NQ: int
    NX: int
    NU: int

    def __init__(self, bundle: RobotModelBundle) -> None:
        """初始化绑定环境并应用力矩模式限界。

        Args:
            bundle: 同源组装结果（profile/parameters/model/binding）；
                力矩上限优先取 ``parameters.torque_max``，其次模型显式
                声明的执行器范围（forcelimited/ctrllimited），两者皆无
                则拒绝构造（不静默无限制）。

        Raises:
            ValueError: held_q 维数不符、执行器无有限力矩限界、
                torque_max 非有限/非正、或 init_q 维数不符。
        """
        self._bundle = bundle
        self._binding = bundle.binding
        self._parameters = bundle.parameters
        self.model = bundle.model
        self.data = mujoco.MjData(self.model)
        # 私有 scratch：step_from_state 专用，绝不触碰 live data
        self._scratch = mujoco.MjData(self.model)

        self._qpos_indices = np.asarray(self._binding.qpos_indices, dtype=int)
        self._qvel_indices = np.asarray(self._binding.qvel_indices, dtype=int)
        self._actuator_ids = np.asarray(self._binding.actuator_ids, dtype=int)
        self._held_qpos_indices = np.asarray(
            self._binding.held_qpos_indices, dtype=int
        )
        self._held_qvel_indices = np.asarray(
            self._binding.held_qvel_indices, dtype=int
        )
        n = int(self._qpos_indices.size)
        self._n = n
        self.NQ = n
        self.NX = 2 * n
        self.NU = n
        # T4：SimulationEnv 契约需要 NV（MuJoCo 速度维数；带球执行模型为
        # n+6，裸规划模型为 n）。只读实例属性，不改变既有状态布局语义。
        self.NV = int(self.model.nv)

        # 保持关节目标角（held_joint_names 为空时为零长度数组）
        held_q = np.asarray(
            self._parameters.held_q if self._parameters.held_q is not None else [],
            dtype=np.float64,
        )
        if held_q.size != self._held_qpos_indices.size:
            raise ValueError(
                f"held_q 长度 {held_q.size} 与保持关节数 "
                f"{self._held_qpos_indices.size} 不符"
                f"（robot_id: {self._parameters.robot_id}，"
                f"来源: {self._parameters.source}）"
            )
        self._held_q = np.array(held_q, dtype=np.float64)  # 可写副本

        # 力矩上限解析（runtime torque_max 优先，模型显式范围回退）
        self._torque_max = self._resolve_torque_limits()

        # 编译期执行器增益快照（torque 模式恢复基准；位置模式覆写）
        self._orig_gainprm = np.array(self.model.actuator_gainprm[self._actuator_ids])
        self._orig_biasprm = np.array(self.model.actuator_biasprm[self._actuator_ids])
        self._orig_biastype = np.array(
            self.model.actuator_biastype[self._actuator_ids]
        )

        # 执行器模式状态（0=力矩, 1=位置）
        self._actuator_mode: int = 0
        self._kp: np.ndarray | None = None
        self._kd: np.ndarray | None = None
        self._use_feedforward: bool = False

        # 雅可比缓存（set_arm_state/step/reset/update_kinematics 失效）
        self._jacp_cache: np.ndarray | None = None
        self._jacr_cache: np.ndarray | None = None

        self._apply_torque_mode()
        self.reset()
        logger.debug(
            "BoundRobotEnv 初始化: n=%d NQ/NX/NU=%d/%d/%d TCP site id=%d "
            "执行器=%s（robot_id: %s）",
            n,
            self.NQ,
            self.NX,
            self.NU,
            self._binding.tcp_site_id,
            self._actuator_ids.tolist(),
            self._parameters.robot_id,
        )

    # ------------------------------------------------------------ 只读属性

    @property
    def dt(self) -> float:
        """仿真时间步长（s）。"""
        return float(self.model.opt.timestep)

    @property
    def binding(self) -> ModelBinding:
        """名称 → 地址绑定（只读复用，供下游按名访问）。"""
        return self._binding

    @property
    def torque_max(self) -> np.ndarray:
        """(n,) 力矩上限副本（构造时解析的显式有限限界）。"""
        return np.array(self._torque_max, dtype=np.float64)

    @property
    def control_range(self) -> np.ndarray:
        """(n,2) 每执行器控制范围副本（行序 = binding.actuator_ids）。

        力矩模式 = ±torque_max；位置模式 = 所驱动关节的 jnt_range。
        """
        return np.array(
            self.model.actuator_ctrlrange[self._actuator_ids], dtype=np.float64
        )

    @property
    def actuator_mode(self) -> int:
        """执行器模式：0=力矩, 1=位置。"""
        return self._actuator_mode

    @property
    def kp(self) -> np.ndarray | None:
        """位置模式比例增益 (n,)；力矩模式下为 None。"""
        return self._kp

    @property
    def kd(self) -> np.ndarray | None:
        """位置模式速度增益 (n,)；力矩模式下为 None。"""
        return self._kd

    @property
    def use_feedforward(self) -> bool:
        """位置模式下是否启用前馈补偿（重力+科氏力偏置）。"""
        return self._use_feedforward

    # ------------------------------------------------------------ 状态读写

    def get_arm_state(self) -> np.ndarray:
        """读取主动臂状态 x = [q(n), qdot(n)]（按绑定地址收集）。

        Returns:
            (2n,) 状态向量（新数组）。
        """
        x = np.empty(2 * self._n, dtype=np.float64)
        x[: self._n] = self.data.qpos[self._qpos_indices]
        x[self._n :] = self.data.qvel[self._qvel_indices]
        return x

    def set_arm_state(self, x: Array) -> None:
        """写入主动臂状态并刷新运动学（校验通过后才写入）。

        保持关节同步钉回 held_q/零速；旁观关节（如球 free joint）不被触碰。

        Args:
            x: (2n,) 状态向量 [q, qdot]。

        Raises:
            ValueError: shape 不符或含非有限值。
        """
        x_arr = self._validate_state(x)
        self._pin_held_joints(self.data)
        self.data.qpos[self._qpos_indices] = x_arr[: self._n]
        self.data.qvel[self._qvel_indices] = x_arr[self._n :]
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()

    def reset(self, q0: Array | None = None) -> np.ndarray:
        """重置 live 数据到初始状态（不重置执行器模式配置）。

        Args:
            q0: (n,) 初始关节角；None 时用 parameters.init_q。

        Returns:
            初始主动臂状态 (2n,)。

        Raises:
            ValueError: q0 shape 不符或含非有限值。
        """
        mujoco.mj_resetData(self.model, self.data)
        if q0 is None:
            q = np.asarray(self._parameters.init_q, dtype=np.float64)
        else:
            q = np.asarray(q0, dtype=np.float64)
        if q.shape != (self._n,):
            raise ValueError(
                f"初始关节角 shape 应为 ({self._n},)，实际 {q.shape}"
            )
        if not bool(np.all(np.isfinite(q))):
            raise ValueError("初始关节角含非有限值")
        self._pin_held_joints(self.data)
        self.data.qpos[self._qpos_indices] = q
        self.data.qvel[self._qvel_indices] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()
        return self.get_arm_state()

    def update_kinematics(self) -> None:
        """轻量运动学刷新（不计算雅可比/惯性），并使雅可比缓存失效。"""
        mujoco.mj_kinematics(self.model, self.data)
        self._invalidate_jacobian_cache()

    # ------------------------------------------------------------ 前向仿真

    def step(self, u: Array) -> np.ndarray:
        """live 前进一步：施加控制（按模式裁剪）→ mj_step → 钉回保持关节。

        沿旧环境约定：``mj_step`` 后 MuJoCo 运动学数组停留在步前位形，
        精确 FK/雅可比查询前应先 ``set_arm_state``/``update_kinematics``
        （本方法已使雅可比缓存失效，不会返回步前状态的旧缓存）。

        Args:
            u: (n,) 控制输入。力矩模式为期望力矩（裁剪到 ±torque_max），
                位置模式为期望角度（裁剪到关节限位 + 误差限幅）。

        Returns:
            新的主动臂状态 (2n,)。

        Raises:
            ValueError: u shape 不符或含非有限值（写入前拒绝）。
        """
        u_arr = self._validate_control(u)
        self._pin_held_joints(self.data)
        q_now = np.array(self.data.qpos[self._qpos_indices])
        ctrl = self._compute_ctrl(u_arr, q_now)
        self.data.ctrl[self._actuator_ids] = ctrl
        self._apply_step_forces(self.data)
        mujoco.mj_step(self.model, self.data)
        self._pin_held_joints(self.data)
        self._invalidate_jacobian_cache()
        return self.get_arm_state()

    def step_from_state(self, x: Array, u: Array) -> np.ndarray:
        """从指定状态预测前进一步（私有 scratch，live 完全不变）。

        副作用约定见模块 docstring 的基态配方；重复输入输出相同；
        输入数组不被修改；异常输入在任何写入前拒绝。

        Args:
            x: (2n,) 状态向量 [q, qdot]。
            u: (n,) 控制输入（语义同 step）。

        Returns:
            预测的下一状态 (2n,)（新数组；live FK 不保证等于该终态）。

        Raises:
            ValueError: x/u shape 不符或含非有限值。
        """
        x_arr = self._validate_state(x)
        u_arr = self._validate_control(u)
        scratch = self._scratch
        mujoco.mj_resetData(self.model, scratch)
        self._pin_held_joints(scratch)
        scratch.qpos[self._qpos_indices] = x_arr[: self._n]
        scratch.qvel[self._qvel_indices] = x_arr[self._n :]
        q_now = np.array(scratch.qpos[self._qpos_indices])
        scratch.ctrl[self._actuator_ids] = self._compute_ctrl(u_arr, q_now)
        self._apply_step_forces(scratch)
        mujoco.mj_step(self.model, scratch)
        self._pin_held_joints(scratch)
        x_next = np.empty(2 * self._n, dtype=np.float64)
        x_next[: self._n] = scratch.qpos[self._qpos_indices]
        x_next[self._n :] = scratch.qvel[self._qvel_indices]
        return x_next

    # ------------------------------------------------------------ 末端运动学

    def get_ee_pos(self) -> np.ndarray:
        """TCP（绑定 site）世界坐标位置 (3,)。"""
        return np.array(self.data.site_xpos[self._binding.tcp_site_id])

    def get_ee_vel(self) -> np.ndarray:
        """TCP 世界线速度 (3,) = J_p @ qdot（雅可比列序 = qvel_indices）。"""
        self._ensure_jacobians()
        assert self._jacp_cache is not None
        return self._jacp_cache @ self.data.qvel[self._qvel_indices]

    def get_ee_jacp(self) -> np.ndarray:
        """TCP 平移雅可比 (3, n)，列序 = binding.qvel_indices。"""
        self._ensure_jacobians()
        assert self._jacp_cache is not None
        return self._jacp_cache.copy()

    def get_ee_jacr(self) -> np.ndarray:
        """TCP 角速度雅可比 (3, n)，列序 = binding.qvel_indices。"""
        self._ensure_jacobians()
        assert self._jacr_cache is not None
        return self._jacr_cache.copy()

    def get_ee_normal(self) -> np.ndarray:
        """拍面法向 (3,)：TCP site 局部 +X 轴在世界系中的方向（单位化）。

        R1D 球拍采用 RM65 拓扑：拍柄沿局部 +Z，拍面法向为局部 +X
        （碰撞圆柱已用 geom quat 转到 X 轴）。通用模型以 TCP site 帧的
        X 轴为法向约定。
        """
        rot = self.data.site_xmat[self._binding.tcp_site_id].reshape(3, 3)
        normal = np.array(rot[:, 0])
        normal /= np.linalg.norm(normal) + 1e-12
        return normal

    def _ensure_jacobians(self) -> None:
        """惰性计算并缓存 TCP 雅可比（按绑定列收取）。"""
        if self._jacp_cache is not None:
            return
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(
            self.model, self.data, jacp, jacr, self._binding.tcp_site_id
        )
        self._jacp_cache = np.array(jacp[:, self._qvel_indices])
        self._jacr_cache = np.array(jacr[:, self._qvel_indices])

    def _invalidate_jacobian_cache(self) -> None:
        """清除雅可比缓存（状态/运动学变化后调用）。"""
        self._jacp_cache = None
        self._jacr_cache = None

    # ------------------------------------------------------------ 逆运动学

    def solve_ik(
        self,
        target_pos: Array,
        q_init: Array | None = None,
        max_iter: int = 200,
        eps: float = 1e-3,
        damp: float = 1e-6,
        step_size: float = 0.1,
    ) -> np.ndarray:
        """阻尼最小二乘逆运动学（维数无关，基于绑定雅可比）。

        Args:
            target_pos: 目标末端位置 (3,)。
            q_init: 初始关节角 (n,)；None 时用零位。
            max_iter: 最大迭代次数。
            eps: 收敛阈值（位置误差，m）。
            damp: 阻尼因子。
            step_size: 步长缩放。

        Returns:
            求解的关节角 (n,)。

        Raises:
            ValueError: 目标/初值 shape 不符。
        """
        target = np.asarray(target_pos, dtype=np.float64)
        if target.shape != (3,):
            raise ValueError(f"目标位置 shape 应为 (3,)，实际 {target.shape}")
        if q_init is None:
            q = np.zeros(self._n)
        else:
            q = np.array(q_init, dtype=np.float64)
            if q.shape != (self._n,):
                raise ValueError(
                    f"初始关节角 shape 应为 ({self._n},)，实际 {q.shape}"
                )
        x = np.zeros(2 * self._n)
        for _ in range(max_iter):
            x[: self._n] = q
            self.set_arm_state(x)
            err = target - self.get_ee_pos()
            if float(np.linalg.norm(err)) < eps:
                break
            jac = self.get_ee_jacp()
            jjt = jac @ jac.T + damp * np.eye(3)
            dq = jac.T @ np.linalg.solve(jjt, err)
            q = q + step_size * dq
        return q

    # ------------------------------------------------------------ 执行器模式

    def configure_actuator_mode(
        self,
        mode: str,
        kp: Array | None = None,
        kd: Array | None = None,
    ) -> None:
        """配置执行器模式（仅改写 actuator_ids 行；校验通过后才写入模型）。

        Args:
            mode: "torque" 或 "position"。
            kp: (n,) 位置增益（各轴可非均匀），position 模式必填。
            kd: (n,) 速度增益（各轴可非均匀），position 模式必填。

        Raises:
            ValueError: 未知模式、position 缺 kp/kd、增益 shape 不符/
                非有限/非正（kp）/负（kd）、或驱动关节未声明限位
                （ctrlrange 无法确定）；失败时模型不被部分写入。
        """
        if mode == "torque":
            self._apply_torque_mode()
            self._actuator_mode = 0
            self._kp = None
            self._kd = None
            self._use_feedforward = False
            self.data.qfrc_applied[self._qvel_indices] = 0.0
            return

        if mode != "position":
            raise ValueError(
                f"未知执行器模式: {mode!r}（支持 'torque' / 'position'）"
            )
        if kp is None:
            raise ValueError("位置模式必须提供 kp 和 kd")
        if kd is None:
            raise ValueError("位置模式必须提供 kp 和 kd")
        kp_arr = np.asarray(kp, dtype=np.float64)
        kd_arr = np.asarray(kd, dtype=np.float64)
        if kp_arr.shape != (self._n,):
            raise ValueError(
                f"kp shape 应为 ({self._n},)，实际 {kp_arr.shape}"
            )
        if kd_arr.shape != (self._n,):
            raise ValueError(
                f"kd shape 应为 ({self._n},)，实际 {kd_arr.shape}"
            )
        if not bool(np.all(np.isfinite(kp_arr))):
            raise ValueError("kp 含非有限值")
        if not bool(np.all(np.isfinite(kd_arr))):
            raise ValueError("kd 含非有限值")
        if bool(np.any(kp_arr <= 0.0)):
            raise ValueError("kp 含非正值（位置增益必须为正）")
        if bool(np.any(kd_arr < 0.0)):
            raise ValueError("kd 含负值（阻尼增益必须非负）")
        # 驱动关节必须声明限位（位置指令 ctrlrange 来源）；先校验后写入
        for aid in self._actuator_ids:
            jid = int(self.model.actuator_trnid[aid, 0])
            if int(self.model.jnt_limited[jid]) != 1:
                name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_JOINT, jid
                )
                raise ValueError(
                    f"驱动关节 '{name}' 未声明限位，位置模式 ctrlrange"
                    " 无法确定（关节限位须显式声明，不静默无限制）"
                )

        self._apply_position_mode(kp_arr, kd_arr)
        self._actuator_mode = 1
        self._kp = kp_arr.copy()
        self._kd = kd_arr.copy()
        self._use_feedforward = True
        logger.debug(
            "BoundRobotEnv 切换位置模式: kp=%s kd=%s", kp_arr.tolist(), kd_arr.tolist()
        )

    def configure_feedforward(self, enabled: bool) -> None:
        """启用/禁用位置模式前馈补偿（重力+科氏力偏置）。

        禁用时清除主动关节 dof 上的 qfrc_applied。

        Args:
            enabled: True=启用, False=禁用。
        """
        self._use_feedforward = enabled
        if not enabled:
            self.data.qfrc_applied[self._qvel_indices] = 0.0

    # ------------------------------------------------------------ 内部实现

    def _resolve_torque_limits(self) -> np.ndarray:
        """解析每执行器力矩上限（runtime torque_max 优先，模型显式范围回退）。

        Returns:
            (n,) 正有限力矩上限（可写副本）。

        Raises:
            ValueError: 两个来源均无有限限界，或数值非有限/非正。
        """
        n = self._n
        params = self._parameters
        source_desc = f"robot_id: {params.robot_id}，来源: {params.source}"
        if params.torque_max is not None:
            limits = np.array(params.torque_max, dtype=np.float64)
            origin = "runtime torque_max"
        elif bool(np.all(self.model.actuator_forcelimited[self._actuator_ids] == 1)):
            fr = np.array(self.model.actuator_forcerange[self._actuator_ids])
            limits = np.maximum(np.abs(fr[:, 0]), np.abs(fr[:, 1]))
            origin = "模型 actuator_forcerange"
        elif bool(np.all(self.model.actuator_ctrllimited[self._actuator_ids] == 1)):
            cr = np.array(self.model.actuator_ctrlrange[self._actuator_ids])
            limits = np.maximum(np.abs(cr[:, 0]), np.abs(cr[:, 1]))
            origin = "模型 actuator_ctrlrange"
        else:
            raise ValueError(
                "执行器无有限力矩限界：runtime torque_max 为 None 且模型"
                f"未声明 forcerange/ctrlrange，拒绝静默无限制（{source_desc}）"
            )
        if limits.shape != (n,):
            raise ValueError(
                f"力矩上限 shape 应为 ({n},)，实际 {limits.shape}（{source_desc}）"
            )
        if not bool(np.all(np.isfinite(limits))) or bool(np.any(limits <= 0.0)):
            raise ValueError(
                f"力矩上限必须全部为正有限值（{origin}，{source_desc}）"
            )
        logger.debug("BoundRobotEnv 力矩上限来源: %s", origin)
        return limits

    def _apply_torque_mode(self) -> None:
        """恢复纯力矩执行器并写入显式限界（gain/bias 回编译期快照）。"""
        ids = self._actuator_ids
        self.model.actuator_gainprm[ids] = self._orig_gainprm
        self.model.actuator_biasprm[ids] = self._orig_biasprm
        self.model.actuator_biastype[ids] = self._orig_biastype
        for i, aid in enumerate(ids):
            self.model.actuator_ctrllimited[aid] = 1
            self.model.actuator_forcelimited[aid] = 1
            self.model.actuator_ctrlrange[aid] = (
                -self._torque_max[i],
                self._torque_max[i],
            )
            self.model.actuator_forcerange[aid] = (
                -self._torque_max[i],
                self._torque_max[i],
            )

    def _apply_position_mode(self, kp: np.ndarray, kd: np.ndarray) -> None:
        """按绑定序写入位置模式 PD/ctrlrange/forcerange（先校验后调用）。"""
        for i, aid in enumerate(self._actuator_ids):
            jid = int(self.model.actuator_trnid[aid, 0])
            self.model.actuator_gainprm[aid, 0] = kp[i]
            self.model.actuator_biasprm[aid, 0] = 0.0
            self.model.actuator_biasprm[aid, 1] = -kp[i]
            self.model.actuator_biasprm[aid, 2] = -kd[i]
            self.model.actuator_biastype[aid] = mujoco.mjtBias.mjBIAS_AFFINE
            self.model.actuator_ctrllimited[aid] = 1
            self.model.actuator_ctrlrange[aid] = self.model.jnt_range[jid]
            self.model.actuator_forcelimited[aid] = 1
            self.model.actuator_forcerange[aid] = (
                -self._torque_max[i],
                self._torque_max[i],
            )

    def _compute_ctrl(self, u: np.ndarray, q_now: np.ndarray) -> np.ndarray:
        """按当前模式计算写入 ctrl 的控制量（软件层裁剪）。

        力矩模式：clip(u, ±torque_max)。
        位置模式：先裁剪到关节限位（ctrlrange），再限幅位置误差
        |u - q| ≤ torque_max/kp（等效力矩上限；模型 forcerange 双保险）。
        """
        if self._actuator_mode == 1 and self._kp is not None:
            lo = self.model.actuator_ctrlrange[self._actuator_ids, 0]
            hi = self.model.actuator_ctrlrange[self._actuator_ids, 1]
            u_clipped = np.clip(u, lo, hi)
            max_err = self._torque_max / self._kp
            err = u_clipped - q_now
            return q_now + np.clip(err, -max_err, max_err)
        return np.clip(u, -self._torque_max, self._torque_max)

    def _apply_step_forces(self, data: mujoco.MjData) -> None:
        """步进前的主动 dof 外施力处理（前馈偏置或清零）。

        位置模式 + 前馈：mj_forward → mj_rne 偏置力（h = C*qdot + g）
        写入主动 dof 的 qfrc_applied，使 PD 无需对抗重力/科氏力；
        前馈与 PD 执行器分量之和按 ±torque_max 限幅——模型 forcerange
        只裁剪执行器分量，qfrc_applied 不受其约束，总驱动力矩须在此
        软件层统一限界（PD 分量由已写入的 ctrl 与步前 q/qdot 预测，
        与 MuJoCo affine 执行器 tau = kp*(ctrl-q) - kd*qdot 逐值一致）。
        否则清零主动 dof 的 qfrc_applied（旁观 dof 不触碰）。
        """
        if self._actuator_mode == 1 and self._use_feedforward:
            mujoco.mj_forward(self.model, data)
            data.qacc[:] = 0.0
            tau_bias = np.zeros(self.model.nv)
            mujoco.mj_rne(self.model, data, 0, tau_bias)
            tau_ff = tau_bias[self._qvel_indices]
            kp = self._kp
            kd = self._kd
            assert kp is not None and kd is not None
            # PD 执行器分量预测（ctrl 由 step/step_from_state 在本方法
            # 前写入；含 forcerange ±torque_max 裁剪，与 mj_step 一致）
            tau_pd = kp * (
                data.ctrl[self._actuator_ids] - data.qpos[self._qpos_indices]
            ) - kd * data.qvel[self._qvel_indices]
            tau_pd = np.clip(tau_pd, -self._torque_max, self._torque_max)
            # 总驱动力矩 |tau_pd + tau_ff| ≤ torque_max（带符号余量限幅）
            data.qfrc_applied[self._qvel_indices] = np.clip(
                tau_ff, -self._torque_max - tau_pd, self._torque_max - tau_pd
            )
        else:
            data.qfrc_applied[self._qvel_indices] = 0.0

    def _pin_held_joints(self, data: mujoco.MjData) -> None:
        """把保持关节钉回 held_q/零速（无保持关节时为空操作）。"""
        if self._held_qpos_indices.size:
            data.qpos[self._held_qpos_indices] = self._held_q
            data.qvel[self._held_qvel_indices] = 0.0

    def _validate_state(self, x: Array) -> np.ndarray:
        """校验状态向量（shape/有限性），失败在写入前抛出。

        Raises:
            ValueError: shape 不符或含非有限值。
        """
        try:
            x_arr = np.asarray(x, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"状态向量无法解析为 float 数组: {exc}") from exc
        if x_arr.shape != (2 * self._n,):
            raise ValueError(
                f"状态向量 shape 应为 ({2 * self._n},)，实际 {x_arr.shape}"
            )
        if not bool(np.all(np.isfinite(x_arr))):
            bad = np.flatnonzero(~np.isfinite(x_arr)).tolist()
            raise ValueError(f"状态向量含非有限值（索引 {bad}）")
        return x_arr

    def _validate_control(self, u: Array) -> np.ndarray:
        """校验控制向量（shape/有限性），失败在写入前抛出。

        Raises:
            ValueError: shape 不符或含非有限值。
        """
        try:
            u_arr = np.asarray(u, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"控制输入无法解析为 float 数组: {exc}") from exc
        if u_arr.shape != (self._n,):
            raise ValueError(
                f"控制输入 shape 应为 ({self._n},)，实际 {u_arr.shape}"
            )
        if not bool(np.all(np.isfinite(u_arr))):
            bad = np.flatnonzero(~np.isfinite(u_arr)).tolist()
            raise ValueError(f"控制输入含非有限值（索引 {bad}）")
        return u_arr
