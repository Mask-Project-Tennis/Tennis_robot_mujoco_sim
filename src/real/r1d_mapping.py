"""R1-A7-D 七关节名称映射 — 模型关节名与整机 35 槽位互查（阶段 4 Task 4.2）。

按 profile.active_joint_names（模型序）逐名查硬件配置的槽位与校准表，
不依赖前缀或顺序猜测；换算契约（Task 4.2 冻结）：

- 模型 → SDK：q_sdk = sign * q_model + offset
- SDK → 模型：q_model = sign * (q_sdk - offset)
- 速度只变符号：dq_model = sign * dq_sdk（不加偏置）

sign 仅允许 ±1，offset 为实测/核验值；未知校准（null）不能构造映射，
即未知校准不能启用运动。纯数据层：不导入 SDK/MuJoCo、不启动线程。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.real.r1d_config import R1D_ARM_JOINT_COUNT, R1DHardwareConfig, R1D_SLOT_COUNT
from src.real.r1d_types import R1DStateFrame
from src.robot.robot_profile import RobotProfile


@dataclass(frozen=True)
class R1DJointMapping:
    """R1-A7-D 单臂七关节映射（模型序）。

    Attributes:
        joint_names: 模型关节名（与 profile.active_joint_names 同序）。
        sdk_indices: 各模型关节对应的整机 SDK 槽位。
        signs: 各关节方向符号（仅 ±1）。
        offsets_rad: 各关节零位偏移（rad，SDK 侧）。
    """

    joint_names: tuple[str, ...]
    sdk_indices: tuple[int, ...]
    signs: tuple[int, ...]
    offsets_rad: tuple[float, ...]

    @classmethod
    def from_profile(
        cls, profile: RobotProfile, config: R1DHardwareConfig
    ) -> R1DJointMapping:
        """从机器人描述与硬件配置构建映射（按关节名精确匹配）。

        Args:
            profile: R1D 单侧 7 关节描述。
            config: 已加载的硬件配置（须含非 null 校准与槽位布局）。

        Returns:
            按模型序排列的 R1DJointMapping。

        Raises:
            ValueError: profile 不是 7 关节、校准为 null（未知校准不能
                构造映射）、joint_slots/signs/offsets 键与 profile 关节名
                不一致、槽位越界/重复、sign 非 ±1 或 offset 非有限。
        """
        if profile.ndof != R1D_ARM_JOINT_COUNT:
            raise ValueError(
                f"R1D 映射要求 {R1D_ARM_JOINT_COUNT} 关节 profile，"
                f"实际 ndof={profile.ndof}"
            )
        signs_map = config.calibration.joint_signs
        offsets_map = config.calibration.joint_offsets_rad
        if signs_map is None or offsets_map is None:
            raise ValueError(
                "校准为 null（joint_signs/joint_offsets_rad 未知），"
                "不能构造映射——未知校准不能启用运动"
            )

        names = profile.active_joint_names
        profile_names = set(names)
        # 三处按名表必须与 profile 关节名集合精确一致（多/少/拼写错误均拒绝）
        for table_name, table in (
            ("layout.joint_slots", config.layout.joint_slots),
            ("calibration.joint_signs", signs_map),
            ("calibration.joint_offsets_rad", offsets_map),
        ):
            diff = sorted(profile_names ^ set(table))
            if diff:
                raise ValueError(
                    f"{table_name} 与 profile.active_joint_names 不一致"
                    f"（差异: {diff[:4]}）"
                )

        # 防御性复查（直接构造 config 时 load 层校验不经过）
        seen_slots: set[int] = set()
        indices: list[int] = []
        signs: list[int] = []
        offsets: list[float] = []
        for name in names:
            slot = config.layout.joint_slots[name]
            if not isinstance(slot, int) or not (0 <= slot < R1D_SLOT_COUNT):
                raise ValueError(
                    f"关节 {name} 槽位 {slot} 越界（整机 {R1D_SLOT_COUNT} 槽）"
                )
            if slot in seen_slots:
                raise ValueError(f"槽位 {slot} 被重复映射（关节 {name}）")
            seen_slots.add(slot)
            sign = signs_map[name]
            if sign not in (1, -1):
                raise ValueError(f"关节 {name} 的 sign 只能取 ±1，实际 {sign}")
            offset = float(offsets_map[name])
            if not bool(np.isfinite(offset)):
                raise ValueError(f"关节 {name} 的 offset 必须是有限值，实际 {offset}")
            indices.append(slot)
            signs.append(sign)
            offsets.append(offset)

        return cls(
            joint_names=tuple(names),
            sdk_indices=tuple(indices),
            signs=tuple(signs),
            offsets_rad=tuple(offsets),
        )

    def extract(self, frame: R1DStateFrame) -> np.ndarray:
        """从整机状态帧提取本臂 7 关节位置（模型序，rad）。

        Args:
            frame: 完整 35 槽整机状态帧快照。

        Returns:
            形状 (7,) 的新数组：q_model = sign * (q_sdk - offset)。
        """
        idx = np.asarray(self.sdk_indices, dtype=np.intp)
        signs = np.asarray(self.signs, dtype=np.float64)
        offsets = np.asarray(self.offsets_rad, dtype=np.float64)
        return signs * (frame.q[idx] - offsets)

    def extract_velocities(self, frame: R1DStateFrame) -> np.ndarray:
        """从整机状态帧提取本臂 7 关节速度（模型序，rad/s）。

        速度只变符号不加偏置：dq_model = sign * dq_sdk。

        Args:
            frame: 完整 35 槽整机状态帧快照。

        Returns:
            形状 (7,) 的新数组。
        """
        idx = np.asarray(self.sdk_indices, dtype=np.intp)
        signs = np.asarray(self.signs, dtype=np.float64)
        return signs * frame.dq[idx]

    def to_sdk_positions(self, q: np.ndarray) -> np.ndarray:
        """把模型序 7 关节位置换算为 SDK 侧位置（模型序排列）。

        Args:
            q: 形状 (7,) 的模型序位置（rad），须有限。

        Returns:
            形状 (7,) 的新数组：q_sdk = sign * q_model + offset
            （仅换算数值，槽位放置由 4.3 命令策略显式完成）。

        Raises:
            ValueError: 长度不是 7 或含 NaN/Inf。
        """
        arr = np.asarray(q, dtype=np.float64)
        if arr.shape != (R1D_ARM_JOINT_COUNT,):
            raise ValueError(
                f"q 必须是一维长度 {R1D_ARM_JOINT_COUNT} 数组，实际形状 {arr.shape}"
            )
        if not bool(np.all(np.isfinite(arr))):
            raise ValueError("q 含 NaN 或 Inf")
        signs = np.asarray(self.signs, dtype=np.float64)
        offsets = np.asarray(self.offsets_rad, dtype=np.float64)
        return signs * arr + offsets
