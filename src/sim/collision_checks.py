"""独立预测碰撞保护查询 — has_forbidden_contact + ForbiddenContactChecker。

阶段 3 · Task 3.7 交付（施工计划 §共享接口契约"保护查询"条目，逐字实现）：

- ``has_forbidden_contact(model, data, pairs) -> bool``：只检查**预先解析**
  的禁止 geom 对（3.2 ``TennisSceneBundle.protected_geom_pairs`` 的
  (int,int) 元组），不把球拍击球等其他接触视作禁止接触 —— 球—拍接触
  不在禁止对列表内即被忽略；
- ``ForbiddenContactChecker``：面向预测消费方（PredictiveSafetyFilter /
  3.9 共享管线）的检查器封装 —— 在**独立 MjData**（非 live data、非
  env 的 step_from_state scratch）上按臂 qpos 地址写入关节角并查询禁止
  对接触，查询过程不触碰 live 球/状态/时钟；
- 禁止对缺失（空元组）或无效（id 越界、两端同 geom）必须失败，不允许
  以空集冒充机身保护后空真放行。

语义声明（不宣称的能力）：
- 检查为**离散位形查询**：调用方逐候选预测一步后查询终态接触。离散
  一步检查**不保证步间连续无碰撞**（步内扫掠穿透可能不被观测），也不
  构成真机安全认证 —— 仅是仿真闭环中的保护性拒绝条件；
- 独立数据上旁观球冻结在 ``qpos0`` 初位姿（与 3.3 scratch 基态一致）：
  本查询检查的是**保护对**（臂/拍—机身）碰撞，不是球；禁止对不含球
  时球位姿不影响查询结果。
"""

from __future__ import annotations

from typing import Any

import mujoco
import numpy as np
from numpy.typing import NDArray

from src.robot.model_binding import ModelBinding

Array = NDArray[np.float64]


def has_forbidden_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pairs: tuple[tuple[int, int], ...],
) -> bool:
    """查询 data 当前接触中是否出现任一禁止 geom 对。

    只读取 data.contact（调用方须保证 data 已对目标位形执行过
    mj_forward/mj_step 使接触数组最新）；只匹配预先解析的禁止对，
    其余接触（如球拍击球）一律忽略。空 pairs 为合法空集语义（返回
    False）；"禁止对缺失必须失败"的约束由 ForbiddenContactChecker
    构造校验承担。

    Args:
        model: MuJoCo 模型（用于禁止对 id 校验）。
        data: 已计算接触的 MuJoCo 数据（只读，不被修改）。
        pairs: 禁止 geom id 对元组（无序对语义，(g1,g2) 与 (g2,g1) 等价）。

    Returns:
        True 表示任一禁止对当前存在接触。

    Raises:
        ValueError: 禁止对含越界 id 或两端为同一 geom（无效对必须失败）。
    """
    forbidden = _normalize_pairs(model, pairs)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = int(con.geom1), int(con.geom2)
        key = (g1, g2) if g1 <= g2 else (g2, g1)
        if key in forbidden:
            return True
    return False


class ForbiddenContactChecker:
    """在独立 MjData 上按臂关节角查询禁止对接触的检查器。

    用法::

        checker = ForbiddenContactChecker(model, qpos_indices, pairs)
        checker.has_contact_at_q(q)          # 独立数据，live 不受影响

    预测消费方（PredictiveSafetyFilter）逐候选 ``step_from_state`` 得到
    预测态后，用 ``has_contact_at_q(x_pred[:n])`` 验证该候选是否引入
    禁止接触 —— 力矩 beta 与位置插值候选均逐一经此验证。

    语义：每次查询在独立 data 上 ``mj_resetData`` 后仅写入主动臂
    qpos（旁观球冻结在 qpos0 初位姿），再 mj_forward 计算接触；离散
    位形查询不保证步间连续无碰撞，不宣称真机安全认证。
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        qpos_indices: Array,
        pairs: tuple[tuple[int, int], ...],
    ) -> None:
        """初始化检查器（校验通过后才分配独立数据）。

        Args:
            model: 带保护碰撞掩码的模型（如 3.2 带球执行模型）。
            qpos_indices: (n,) 主动臂 qpos 地址（binding.qpos_indices）。
            pairs: 禁止 geom id 对（通常 = scene.protected_geom_pairs）。

        Raises:
            ValueError: pairs 为空（禁止对缺失必须失败，不得以空集
                冒充保护）、pairs 含无效对、或 qpos_indices 非法。
        """
        if len(pairs) == 0:
            raise ValueError(
                "禁止对缺失：pairs 为空元组，不得以空集冒充机身保护"
                "（应传入场景解析的 protected_geom_pairs）"
            )
        self._forbidden = _normalize_pairs(model, pairs)
        self._pairs = tuple(pairs)

        idx = np.asarray(qpos_indices, dtype=np.int64).reshape(-1)
        if idx.size == 0:
            raise ValueError("qpos_indices 为空，无法定位主动臂关节")
        if bool(np.any(idx < 0)) or bool(np.any(idx >= model.nq)):
            raise ValueError(
                f"qpos_indices 含越界地址（模型 nq={model.nq}，"
                f"收到 {idx.tolist()}）"
            )
        self._model = model
        self._qpos_indices = np.array(idx, dtype=np.int64)
        self._n = int(idx.size)
        # 独立数据：预测碰撞检查绝不触碰 live data / env scratch
        self._data = mujoco.MjData(model)

    @classmethod
    def from_env(
        cls,
        env: Any,
        pairs: tuple[tuple[int, int], ...],
    ) -> ForbiddenContactChecker:
        """从绑定环境构建检查器（model 取 env.model，臂地址取 binding）。

        Args:
            env: BoundRobotEnv 族环境（须有 ``model`` 与 ``binding``）。
            pairs: 禁止 geom id 对。

        Returns:
            检查器实例。

        Raises:
            ValueError: 环境缺少 model/binding（不静默假设 qpos 前缀布局）。
        """
        model = getattr(env, "model", None)
        if model is None:
            raise ValueError(
                "环境缺少 model 属性，无法构建禁止接触检查器"
                f"（收到 {type(env).__name__}）"
            )
        binding: ModelBinding | None = getattr(env, "binding", None)
        if binding is None:
            raise ValueError(
                "环境缺少 binding（ModelBinding），无法确定主动臂 qpos 地址；"
                "非绑定环境请显式用 ForbiddenContactChecker(model, "
                "qpos_indices, pairs) 构造，不假设前缀布局"
            )
        return cls(model, np.asarray(binding.qpos_indices), pairs)

    @property
    def pairs(self) -> tuple[tuple[int, int], ...]:
        """构造时传入的禁止 geom 对（只读）。"""
        return self._pairs

    def has_contact_at_q(self, q: Array) -> bool:
        """查询给定臂关节角位形下是否出现任一禁止对接触（独立数据）。

        每次查询从 reset 基态重建独立 data（重复输入输出相同），live
        环境/球/时钟不受影响；旁观球冻结在 qpos0 初位姿（本查询只关心
        保护对，禁止对不含球）。

        Args:
            q: (n,) 主动臂关节角（rad），n = qpos_indices 维数。

        Returns:
            True 表示该位形存在禁止对接触。

        Raises:
            ValueError: q 形状不符或含非有限值。
        """
        q_arr = np.asarray(q, dtype=np.float64)
        if q_arr.shape != (self._n,):
            raise ValueError(
                f"关节角 shape 应为 ({self._n},)，实际 {q_arr.shape}"
            )
        if not bool(np.all(np.isfinite(q_arr))):
            raise ValueError("关节角含非有限值")

        data = self._data
        mujoco.mj_resetData(self._model, data)
        data.qpos[self._qpos_indices] = q_arr
        mujoco.mj_forward(self._model, data)
        for i in range(data.ncon):
            con = data.contact[i]
            g1, g2 = int(con.geom1), int(con.geom2)
            key = (g1, g2) if g1 <= g2 else (g2, g1)
            if key in self._forbidden:
                return True
        return False


def _normalize_pairs(
    model: mujoco.MjModel, pairs: tuple[tuple[int, int], ...]
) -> frozenset[tuple[int, int]]:
    """校验禁止对并归一化为无序对集合（小 id 在前）。

    Raises:
        ValueError: id 越界或两端为同一 geom（无效对必须失败）。
    """
    normalized: set[tuple[int, int]] = set()
    ngeom = int(model.ngeom)
    for pair in pairs:
        g1, g2 = int(pair[0]), int(pair[1])
        if not (0 <= g1 < ngeom) or not (0 <= g2 < ngeom):
            raise ValueError(
                f"禁止对 ({g1}, {g2}) 含越界 geom id（模型 ngeom={ngeom}）"
            )
        if g1 == g2:
            raise ValueError(
                f"禁止对 ({g1}, {g2}) 两端为同一 geom，无法构成禁止接触对"
            )
        normalized.add((g1, g2) if g1 <= g2 else (g2, g1))
    return frozenset(normalized)
