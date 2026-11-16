"""ProfileSimEnv — 带物理球的执行环境（阶段 3 · Task 3.4）。

职责（施工计划 §共享接口契约 ProfileSimEnv 条目）：
- 继承 ``BoundRobotEnv``（3.3 冻结接口），``__init__(scene: TennisSceneBundle)``；
- 物理球执行路径：``step_full(u)`` 返回 (arm_state, ball_pos, ball_vel)，
  球由 MuJoCo 物理推进（球—地面/球—拍接触均为真实碰撞，执行路径保留
  球物理）；
- 球状态读写：``get_ball_state`` / ``set_ball_state`` / ``set_ball_vel`` /
  ``get_ball_pos`` / ``get_ball_vel``（按 scene 的球 qpos/qvel 名称地址访问，
  与主动臂地址天然不串）；
- 拍面速度：``get_racket_face_speed()``（拍面中心线速度模，float 标量）与
  ``get_ee_angular_vel()``（拍面角速度）；
- ``set_arm_collision(enabled)``：兼容旧命名，在新环境中**仅切换球—拍
  接触**，不关闭其他保护碰撞 —— 通道为 3.2 的显式 <pair> +
  ``model.pair_margin``（置 -1.0 关闭、恢复编译默认值开启；按
  pair_geom1/2 定位球—拍 pair），不改 contype/conaffinity；
- ``predict_ball_trajectory``：委托 ``BallPredictor`` 解析抛物线预测
  （**调用球预测不得改物理球**，纯数学单元，无 MuJoCo 副作用）。

副作用约定（继承 + 新增）：
- ``step_from_state`` 试探步仍用父类私有 scratch data，live/物理球不变；
  execution 不得作异步线程工作环境（异步规划须用工厂的 planning 副本，
  见 ``src/sim/env_factory.py``）；
- ``reset(q0)`` 覆写：臂恢复 init_q/零速（父类语义）+ 物理球恢复到
  reset 基准（构造时 = model.qpos0 的球位置 + 零速度；
  ``set_initial_ball_state`` 可改写基准，工厂用它把 task 球初态编入）；
  ``set_ball_state`` 写球位置 + 单位四元数 [1,0,0,0] + 线速度 + 零角速度。

R1D 与 RM65 的拍面语义差异：RM65 拍面中心相对法兰有 0.25 m 连杆偏移
（RM65Env.get_racket_face_speed 用 v_tcp + ω×offset）；R1D 绑定 TCP
site（racket_center）即拍面中心（无额外偏移），拍面线速度 = get_ee_vel。
"""

from __future__ import annotations

import logging

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.ilqt.ball_predictor import BallPredictor
from src.sim.bound_robot_env import BoundRobotEnv
from src.sim.tennis_scene import TennisSceneBundle

logger = logging.getLogger(__name__)

Array = NDArray[np.float64]

# 球—拍 pair 关闭用的 margin 值：gap 须穿透 1 m 才产生接触，
# 物理上等效关闭该显式 pair（3.2 交付的单独控制通道约定值）
_BALL_RACKET_PAIR_MARGIN_DISABLED = -1.0

# set_ball_state 写入的单位四元数（wxyz，MuJoCo free joint 布局）
_BALL_QUAT_UNIT = (1.0, 0.0, 0.0, 0.0)


class ProfileSimEnv(BoundRobotEnv):
    """带物理球的执行环境 — BoundRobotEnv + 网球 free joint 读写。

    用法::

        scene = build_tennis_scene(profile, parameters, task)
        env = ProfileSimEnv(scene)
        env.set_initial_ball_state(task.initial_ball_position,
                                   task.initial_ball_velocity)
        arm, ball_pos, ball_vel = env.step_full(u)
        pos_pred, vel_pred = env.predict_ball_trajectory(p0, v0, n)

    Attributes:
        NQ/NX/NU: 主动臂维数（实例属性，n=7 时 NX=14）。
        model: 带球 MuJoCo 模型（nq = n+7, nv = n+6）。
        data: live MuJoCo 数据（球物理在 data 上推进）。
        scene: 构造时的带球场景（只读，含球/拍/保护对 id）。
    """

    def __init__(self, scene: TennisSceneBundle) -> None:
        """初始化带球执行环境（力矩模式 + 球 reset 基准）。

        Args:
            scene: 3.2 的带球场景组装结果（robot 为同源 bundle）。

        Raises:
            ValueError: 带球模型中找不到球—拍显式 pair（set_arm_collision
                的单独控制通道缺失，拒绝静默改 contype/conaffinity）。
        """
        # —— 子类状态先于 super().__init__ 就绪（其末尾调用覆写版 reset）——
        self._scene = scene
        self._ball_qpos_start = int(scene.ball_qpos_start)
        self._ball_qvel_start = int(scene.ball_qvel_start)
        self._ball_geom_id = int(scene.ball_geom_id)
        self._racket_geom_ids = tuple(int(g) for g in scene.racket_geom_ids)

        # 球—拍显式 pair 定位（按 pair_geom1/2，双向匹配）+ margin 快照
        self._ball_racket_pair_ids = self._locate_ball_racket_pairs(
            scene.robot.model
        )
        self._pair_margin_ref = np.array(
            scene.robot.model.pair_margin[self._ball_racket_pair_ids],
            dtype=np.float64,
        )

        # reset 基准：球位置取编译默认 qpos0（3.2 已把 task 初位置写入
        # body pos），速度零；工厂用 set_initial_ball_state 编入 task 初速
        self._initial_ball_pos = np.array(
            scene.robot.model.qpos0[
                self._ball_qpos_start : self._ball_qpos_start + 3
            ],
            dtype=np.float64,
        )
        self._initial_ball_vel = np.zeros(3, dtype=np.float64)

        # 球轨迹预测器（解析抛物线，纯数学；与物理球存储分离）
        self._ball_predictor = BallPredictor(
            dt=float(scene.robot.model.opt.timestep)
        )

        super().__init__(scene.robot)
        logger.debug(
            "ProfileSimEnv 初始化: 球 qpos@%d qvel@%d 球—拍 pair=%s "
            "（robot_id: %s）",
            self._ball_qpos_start,
            self._ball_qvel_start,
            self._ball_racket_pair_ids.tolist(),
            scene.robot.parameters.robot_id,
        )

    # ------------------------------------------------------------ 只读属性

    @property
    def scene(self) -> TennisSceneBundle:
        """构造时的带球场景（球/拍/保护对 id 与同源 bundle）。"""
        return self._scene

    @property
    def ball_predictor(self) -> BallPredictor:
        """解析球轨迹预测器（只读复用；预测不触碰物理球）。"""
        return self._ball_predictor

    # ------------------------------------------------------------ reset 基准

    def set_initial_ball_state(self, pos: Array, vel: Array) -> None:
        """改写 reset 球基准并同步写入当前球状态（工厂消费）。

        Args:
            pos: (3,) reset 基准球位置（世界系，m）。
            vel: (3,) reset 基准球速度（世界系，m/s）。

        Raises:
            ValueError: shape 不符或含非有限值（写入前拒绝）。
        """
        pos_arr, vel_arr = self._validate_ball_state(pos, vel)
        self._initial_ball_pos = pos_arr.copy()
        self._initial_ball_vel = vel_arr.copy()
        self._write_ball_state(pos_arr, vel_arr)
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()

    def reset(self, q0: Array | None = None) -> np.ndarray:
        """重置 live 数据：臂恢复 init_q/零速 + 球恢复 reset 基准。

        Args:
            q0: (n,) 初始关节角；None 时用 parameters.init_q。

        Returns:
            初始主动臂状态 (2n,)。
        """
        x = super().reset(q0)
        self._write_ball_state(self._initial_ball_pos, self._initial_ball_vel)
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()
        return x

    # ------------------------------------------------------------ 球状态读写

    def get_ball_state(self) -> tuple[np.ndarray, np.ndarray]:
        """读取物理球 (pos, vel)（live data，新数组副本）。"""
        return self.get_ball_pos(), self.get_ball_vel()

    def get_ball_pos(self) -> np.ndarray:
        """读取物理球位置 (3,)（世界系）。"""
        return np.array(
            self.data.qpos[self._ball_qpos_start : self._ball_qpos_start + 3]
        )

    def get_ball_vel(self) -> np.ndarray:
        """读取物理球线速度 (3,)（世界系）。"""
        return np.array(
            self.data.qvel[self._ball_qvel_start : self._ball_qvel_start + 3]
        )

    def set_ball_state(self, pos: Array, vel: Array) -> None:
        """写入物理球位置 + 单位四元数 + 线速度 + 零角速度，刷新 FK。

        主动臂 qpos/qvel 不被触碰（球/臂地址互不串）。

        Args:
            pos: (3,) 球位置（世界系，m）。
            vel: (3,) 球线速度（世界系，m/s）。

        Raises:
            ValueError: shape 不符或含非有限值（写入前拒绝）。
        """
        pos_arr, vel_arr = self._validate_ball_state(pos, vel)
        self._write_ball_state(pos_arr, vel_arr)
        mujoco.mj_forward(self.model, self.data)
        self._invalidate_jacobian_cache()

    def set_ball_vel(self, vel: Array) -> None:
        """仅写入物理球速度（不改位置；沿 RM65Env 语义）。

        Args:
            vel: 球速度 (3,) 仅线速度，或 (6,) [线速度, 角速度]。

        Raises:
            ValueError: shape 不符或含非有限值（写入前拒绝）。
        """
        vel_arr = np.asarray(vel, dtype=np.float64)
        if vel_arr.shape not in ((3,), (6,)):
            raise ValueError(
                f"球速度 shape 应为 (3,) 或 (6,)，实际 {vel_arr.shape}"
            )
        if not bool(np.all(np.isfinite(vel_arr))):
            raise ValueError("球速度含非有限值")
        self.data.qvel[self._ball_qvel_start : self._ball_qvel_start + 3] = (
            vel_arr[:3]
        )
        if vel_arr.shape == (6,):
            self.data.qvel[
                self._ball_qvel_start + 3 : self._ball_qvel_start + 6
            ] = vel_arr[3:6]

    # ------------------------------------------------------------ 复合步进

    def step_full(self, u: Array) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """施加控制并前进一步，返回 (臂状态, 球位置, 球速度)。

        球由 MuJoCo 物理推进（重力 + 球—地面/球—拍碰撞），仅推进
        一个 dt；共享管线（SimComponent 等）按本接口消费。

        Args:
            u: (n,) 控制输入（语义同 step，力矩/位置按当前模式）。

        Returns:
            (arm_state (2n,), ball_pos (3,), ball_vel (3,))。

        Raises:
            ValueError: u shape 不符或含非有限值（写入前拒绝）。
        """
        arm_state = self.step(u)
        return arm_state, self.get_ball_pos(), self.get_ball_vel()

    # ------------------------------------------------------------ 拍面速度

    def get_racket_face_speed(self) -> float:
        """拍面中心线速度标量 [m/s]（R1D 拍面中心 = 绑定 TCP site）。"""
        return float(np.linalg.norm(self.get_ee_vel()))

    def get_ee_angular_vel(self) -> np.ndarray:
        """拍面中心角速度 (3,) = J_r @ qdot（列序 = binding.qvel_indices）。"""
        return self.get_ee_jacr() @ self.data.qvel[self._qvel_indices]

    # ------------------------------------------------------------ 球—拍接触开关

    def set_arm_collision(self, enabled: bool) -> None:
        """仅切换球—拍接触（兼容旧命名；不动其他保护碰撞）。

        通道为 3.2 的显式 <pair> + ``model.pair_margin``：
        enabled=False → margin=-1.0（等效关闭球—拍接触）；
        enabled=True → 恢复构造时快照的编译默认 margin。
        臂/拍—机身等保护碰撞掩码不受影响（不改 contype/conaffinity）；
        切换后刷新 FK 使 data.ncon 与新 margin 一致。

        Args:
            enabled: True=开启球—拍接触，False=关闭。
        """
        if enabled:
            self.model.pair_margin[self._ball_racket_pair_ids] = (
                self._pair_margin_ref
            )
        else:
            self.model.pair_margin[self._ball_racket_pair_ids] = (
                _BALL_RACKET_PAIR_MARGIN_DISABLED
            )
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------ 球轨迹预测

    def predict_ball_trajectory(
        self, ball_pos: Array, ball_vel: Array, n_steps: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """解析抛物线预测球轨迹（含地面弹跳），不改物理球。

        委托内部 ``BallPredictor.predict_from``（纯数学单元，无 MuJoCo
        副作用）；live qpos/qvel/物理球逐项不变。

        Args:
            ball_pos: 初始位置 (3,)。
            ball_vel: 初始速度 (3,)。
            n_steps: 预测步数。

        Returns:
            (positions (n,3), velocities (n,3))。
        """
        return self._ball_predictor.predict_from(ball_pos, ball_vel, n_steps)

    # ------------------------------------------------------------ 内部实现

    def _locate_ball_racket_pairs(
        self, model: mujoco.MjModel
    ) -> np.ndarray:
        """定位带球模型中全部球—拍显式 pair 的 id。

        Args:
            model: 带球模型（pair 按 pair_geom1/2 与球/拍 geom id 匹配）。

        Returns:
            (k,) 球—拍 pair id 数组（k = 拍 geom 数）。

        Raises:
            ValueError: 模型中无任何球—拍显式 pair（控制通道缺失）。
        """
        racket_set = set(self._racket_geom_ids)
        pair_ids = [
            pid
            for pid in range(model.npair)
            if (
                int(model.pair_geom1[pid]) in racket_set
                and int(model.pair_geom2[pid]) == self._ball_geom_id
            )
            or (
                int(model.pair_geom2[pid]) in racket_set
                and int(model.pair_geom1[pid]) == self._ball_geom_id
            )
        ]
        if not pair_ids:
            raise ValueError(
                "带球模型中未找到球—拍显式 <pair>（set_arm_collision 的"
                "单独控制通道缺失；不应静默改 contype/conaffinity，"
                f"球 geom id={self._ball_geom_id}, "
                f"拍 geom ids={self._racket_geom_ids}）"
            )
        return np.asarray(pair_ids, dtype=int)

    def _write_ball_state(self, pos: np.ndarray, vel: np.ndarray) -> None:
        """把球位置/单位四元数/线速度/零角速度写入 live data（校验后调用）。"""
        bq, bv = self._ball_qpos_start, self._ball_qvel_start
        self.data.qpos[bq : bq + 3] = pos
        self.data.qpos[bq + 3 : bq + 7] = _BALL_QUAT_UNIT
        self.data.qvel[bv : bv + 3] = vel
        self.data.qvel[bv + 3 : bv + 6] = 0.0

    def _validate_ball_state(
        self, pos: Array, vel: Array
    ) -> tuple[np.ndarray, np.ndarray]:
        """校验球位置/速度（shape/有限性），失败在写入前抛出。

        Raises:
            ValueError: shape 不符或含非有限值。
        """
        pos_arr = np.asarray(pos, dtype=np.float64)
        vel_arr = np.asarray(vel, dtype=np.float64)
        if pos_arr.shape != (3,):
            raise ValueError(f"球位置 shape 应为 (3,)，实际 {pos_arr.shape}")
        if vel_arr.shape != (3,):
            raise ValueError(f"球速度 shape 应为 (3,)，实际 {vel_arr.shape}")
        if not bool(np.all(np.isfinite(pos_arr))):
            raise ValueError("球位置含非有限值")
        if not bool(np.all(np.isfinite(vel_arr))):
            raise ValueError("球速度含非有限值")
        return pos_arr, vel_arr
