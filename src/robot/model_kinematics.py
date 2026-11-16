"""ModelKinematics — 名称映射驱动的运动学探针（阶段 2 · Task 2.6）。

职责边界（共享约束 §2.1）：
- 按 ``ModelBinding`` 的 qpos/qvel 地址读写主动关节状态，
  提供 TCP（``racket_center`` site）位姿与解析雅可比；
- **每实例自有 MjData**（构造时创建，不共享/不复用全局 data）；
- 只读运动学：无动力学 step、控制器、后台线程或硬件生命周期，
  不实现 RobotEnv 完整协议，不引入 IK/规划。

数值契约：
- 状态 x = [q(n), qdot(n)]；set_state 在写入 MjData **之前**完成全部
  校验（shape/有限性/关节位置在模型 jnt_range 界内），非法输入不产生
  部分写入；
- 关节位置界限以 ``model.jnt_range``（按 binding.joint_ids 行序）为
  准——工厂构建的 bundle 中 ``parameters.q_lower/q_upper`` 即同一数值
  的回填副本；未声明限位（jnt_limited=0）的关节跳过界限检查；
- ``tcp_pose``/``tcp_jacobian`` 返回世界系结果，旋转用 3×3 矩阵，
  雅可比两块（平移/角速度）均为 3×n、列序 = binding.qvel_indices
  （即 profile.active_joint_names 顺序）。
- **保持关节**（binding.held_qpos_indices）在构造时按
  ``parameters.held_q`` 写入、速度置零；``set_state`` 只写主动关节，
  不触碰保持关节，也不改非主动自由关节（如 free joint）。
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np

from src.robot.model_factory import RobotModelBundle

logger = logging.getLogger(__name__)


class ModelKinematics:
    """按名称绑定读写的机器人运动学探针（每实例自有 MjData）。

    用法::

        kin = ModelKinematics(bundle)
        kin.set_state(np.concatenate([q, qdot]))
        pos, rot = kin.tcp_pose()          # (3,) 世界位置 + 3×3 世界旋转
        jacp, jacr = kin.tcp_jacobian()    # 3×n 平移 / 3×n 角速度（世界系）

    Attributes:
        n_dof: 主动关节自由度数（= len(binding.qpos_indices)）。
    """

    def __init__(self, bundle: RobotModelBundle) -> None:
        """初始化探针。

        Args:
            bundle: 2.5b 工厂产出的同源组装结果（model/binding/parameters）。
        """
        self._model = bundle.model
        self._binding = bundle.binding
        self._qpos_indices = np.asarray(bundle.binding.qpos_indices, dtype=int)
        self._qvel_indices = np.asarray(bundle.binding.qvel_indices, dtype=int)
        self._n = len(self._qpos_indices)

        # 保持关节：构造时按 parameters.held_q 写入初态（速度为零），
        # set_state 只碰主动关节，不再改动保持关节
        self._held_qpos_indices = np.asarray(
            bundle.binding.held_qpos_indices, dtype=int
        )
        self._held_qvel_indices = np.asarray(
            bundle.binding.held_qvel_indices, dtype=int
        )
        self._held_q = np.asarray(bundle.parameters.held_q, dtype=np.float64)
        if self._held_q.size != self._held_qpos_indices.size:
            raise ValueError(
                f"held_q 长度 {self._held_q.size} 与保持关节数 "
                f"{self._held_qpos_indices.size} 不符"
            )

        # 关节位置界限：模型 jnt_range 按绑定行序读取；未声明限位的
        # 关节不参与界限检查（jnt_limited=0 的行 range 为 [0,0] 占位）
        joint_ids = np.asarray(bundle.binding.joint_ids, dtype=int)
        self._limited = np.asarray(self._model.jnt_limited[joint_ids], dtype=bool)
        self._q_lower = np.asarray(
            self._model.jnt_range[joint_ids, 0], dtype=np.float64
        )
        self._q_upper = np.asarray(
            self._model.jnt_range[joint_ids, 1], dtype=np.float64
        )

        # 每实例自有 MjData（禁止共享/复用全局 data）；构造后立即使
        # 位姿/雅可比查询可用（零初态前向一次）
        self._data = mujoco.MjData(self._model)
        # 保持关节按 held_q 初始化、速度置零（任务卡：初始化保持关节为
        # held_q、保持速度为零）。R1D 双侧 held_joint_names 为空，此分支
        # 不执行；任何带保持关节的模型在此拿到正确初态。
        if self._held_q.size:
            self._data.qpos[self._held_qpos_indices] = self._held_q
            self._data.qvel[self._held_qvel_indices] = 0.0
        mujoco.mj_forward(self._model, self._data)
        logger.debug(
            "ModelKinematics 初始化: n_dof=%d, TCP site id=%d",
            self._n,
            self._binding.tcp_site_id,
        )

    @property
    def n_dof(self) -> int:
        """主动关节自由度数。"""
        return self._n

    def set_state(self, x: np.ndarray) -> None:
        """写入状态 x = [q(n), qdot(n)] 并前向更新运动学。

        校验（全部通过后才写入，非法输入不产生部分写入）：
        1. shape 必须为 (2n,)；
        2. 全部元素有限（拒绝 NaN/Inf）；
        3. q 在模型 jnt_range 界内（闭区间；未声明限位关节跳过）。

        Args:
            x: (2n,) 状态向量 [q, qdot]（rad, rad/s）。

        Raises:
            ValueError: shape 不符、含非有限值或关节位置超界。
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

        q = x_arr[: self._n]
        if bool(np.any(self._limited & (q < self._q_lower))) or bool(
            np.any(self._limited & (q > self._q_upper))
        ):
            below = np.flatnonzero(self._limited & (q < self._q_lower)).tolist()
            above = np.flatnonzero(self._limited & (q > self._q_upper)).tolist()
            names = self._binding_names()
            detail = []
            if below:
                detail.append(f"低于下界 {[(names[i], q[i]) for i in below]}")
            if above:
                detail.append(f"高于上界 {[(names[i], q[i]) for i in above]}")
            raise ValueError("关节位置超出模型 jnt_range：" + "；".join(detail))

        # 校验全部通过，一次性写入（先 qpos 后 qvel 均为覆盖写，不残留）
        self._data.qpos[self._qpos_indices] = q
        self._data.qvel[self._qvel_indices] = x_arr[self._n :]
        mujoco.mj_forward(self._model, self._data)

    def get_state(self) -> np.ndarray:
        """读取状态 x = [q(n), qdot(n)]（按 binding 地址收集）。

        Returns:
            (2n,) 状态向量副本。
        """
        x = np.empty(2 * self._n, dtype=np.float64)
        x[: self._n] = self._data.qpos[self._qpos_indices]
        x[self._n :] = self._data.qvel[self._qvel_indices]
        return x

    def tcp_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP（racket_center site）的世界位姿。

        Returns:
            (pos, rot)：(3,) 世界位置与 3×3 世界旋转矩阵（副本）。
            对应最近一次 set_state 的状态。
        """
        sid = self._binding.tcp_site_id
        pos = self._data.site_xpos[sid].copy()
        rot = self._data.site_xmat[sid].reshape(3, 3).copy()
        return pos, rot

    def tcp_jacobian(self) -> tuple[np.ndarray, np.ndarray]:
        """TCP 的解析雅可比（mj_jacSite，世界系）。

        Returns:
            (jacp, jacr)：3×n 平移雅可比与 3×n 角速度雅可比，
            列序 = binding.qvel_indices（profile.active_joint_names 顺序）。
        """
        jacp = np.zeros((3, self._model.nv))
        jacr = np.zeros((3, self._model.nv))
        mujoco.mj_jacSite(self._model, self._data, jacp, jacr, self._binding.tcp_site_id)
        return (
            jacp[:, self._qvel_indices].copy(),
            jacr[:, self._qvel_indices].copy(),
        )

    def _binding_names(self) -> list[str]:
        """主动关节名列表（错误消息用，按绑定顺序）。"""
        return [
            mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, int(jid))
            or f"joint_id={jid}"
            for jid in self._binding.joint_ids
        ]
