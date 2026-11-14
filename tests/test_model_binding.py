"""ModelBinding 模型名称绑定器测试。

验证 RM65 真实模型与乱序七关节合成模型的地址解析，
以及缺名/错配/非标量关节的拒绝行为。绑定器只读，不修改模型。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from src.robot.model_binding import ModelBinding, bind_robot_model
from src.robot.robot_profile import RobotProfile, load_robot_profile
from src.utils.mujoco_loader import load_mujoco_model

# 仓库根目录
REPO_ROOT = Path(__file__).resolve().parent.parent


def _synthetic_profile(
    active: tuple[str, ...] = tuple(f"j{i}" for i in range(1, 8)),
    actuators: tuple[str, ...] = tuple(f"act_j{i}" for i in range(1, 8)),
    held: tuple[str, ...] = (),
    tcp: str = "tcp",
) -> RobotProfile:
    """构造乱序七关节合成模型对应的描述（直接构造，不经 YAML）。"""
    return RobotProfile(
        schema_version=1,
        robot_id="synthetic7",
        model_path=Path("unused.xml"),
        active_joint_names=active,
        actuator_names=actuators,
        held_joint_names=held,
        tcp_site_name=tcp,
    )


# freejoint 在前 + 7 个乱序 hinge + 倒序执行器定义
_SCRAMBLED_XML = """
<mujoco model="scrambled7">
  <worldbody>
    <body name="ball" pos="0 0 2">
      <freejoint name="ball_free"/>
      <geom type="sphere" size="0.01"/>
    </body>
    <!-- MuJoCo 限制单 body ≤6 dof：7 个 hinge 拆成嵌套 body，文档序不变 -->
    <body name="arm" pos="0 0 1">
      <joint name="j7" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
      <body name="seg1" pos="0 0 0.1">
        <joint name="j1" type="hinge" axis="1 0 0"/>
        <geom type="sphere" size="0.01" pos="0 0 0.05"/>
        <body name="seg2" pos="0 0 0.1">
          <joint name="j4" type="hinge" axis="0 0 1"/>
          <geom type="sphere" size="0.01" pos="0 0 0.05"/>
          <body name="seg3" pos="0 0 0.1">
            <joint name="j2" type="hinge" axis="1 0 0"/>
            <geom type="sphere" size="0.01" pos="0 0 0.05"/>
            <body name="seg4" pos="0 0 0.1">
              <joint name="j6" type="hinge" axis="0 1 0"/>
              <geom type="sphere" size="0.01" pos="0 0 0.05"/>
              <body name="seg5" pos="0 0 0.1">
                <joint name="j3" type="hinge" axis="1 0 0"/>
                <geom type="sphere" size="0.01" pos="0 0 0.05"/>
                <body name="seg6" pos="0 0 0.1">
                  <joint name="j5" type="hinge" axis="0 0 1"/>
                  <geom type="sphere" size="0.01" pos="0 0 0.05"/>
                  <site name="tcp" pos="0 0 0.1"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="act_j7" joint="j7"/>
    <motor name="act_j6" joint="j6"/>
    <motor name="act_j5" joint="j5"/>
    <motor name="act_j4" joint="j4"/>
    <motor name="act_j3" joint="j3"/>
    <motor name="act_j2" joint="j2"/>
    <motor name="act_j1" joint="j1"/>
  </actuator>
</mujoco>
"""


class TestRm65RealModel:
    """RM65 实际模型绑定。"""

    def test_bind_rm65_real_model(self) -> None:
        """真实模型：主动 0..5、保持 6..11、执行器 0..5、TCP 有效。"""
        profile = load_robot_profile(
            REPO_ROOT / "configs" / "robots" / "rm65.yaml"
        )
        model = load_mujoco_model(REPO_ROOT / "src" / "robot" / "rm65_model.xml")
        binding = bind_robot_model(profile, model)

        assert binding.joint_ids == tuple(range(6))
        assert binding.qpos_indices == tuple(range(6))
        assert binding.qvel_indices == tuple(range(6))
        assert binding.actuator_ids == tuple(range(6))
        assert binding.held_qpos_indices == tuple(range(6, 12))
        assert binding.held_qvel_indices == tuple(range(6, 12))
        assert binding.tcp_site_id >= 0
        expected_tcp = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "racket_center"
        )
        assert binding.tcp_site_id == expected_tcp


class TestScrambledSyntheticModel:
    """乱序七关节合成模型 — 证明按名称解析而非取前 N 列/范围切片。"""

    @pytest.fixture()
    def scrambled_model(self) -> mujoco.MjModel:
        """加载乱序合成模型。"""
        return mujoco.MjModel.from_xml_string(_SCRAMBLED_XML)

    def test_addresses_resolved_by_name(self, scrambled_model: mujoco.MjModel) -> None:
        """按 profile 名称顺序解析：地址乱序、执行器倒序，均非连续切片。"""
        binding = bind_robot_model(_synthetic_profile(), scrambled_model)

        # 关节 id：j1..j7 在 XML 中的定义序为 j7,j1,j4,j2,j6,j3,j5 → id 2,4,6,3,7,5,1
        assert binding.joint_ids == (2, 4, 6, 3, 7, 5, 1)
        # qpos 地址 = 7（freejoint 占 0..6）+ 各 hinge 自身偏移
        assert binding.qpos_indices == (8, 10, 12, 9, 13, 11, 7)
        # qvel 地址 = 6（freejoint 占 0..5）+ 各 hinge 自身偏移
        assert binding.qvel_indices == (7, 9, 11, 8, 12, 10, 6)
        # 执行器倒序定义：act_j1 的 id 是 6，而非范围切片的 0
        assert binding.actuator_ids == (6, 5, 4, 3, 2, 1, 0)
        assert binding.tcp_site_id == mujoco.mj_name2id(
            scrambled_model, mujoco.mjtObj.mjOBJ_SITE, "tcp"
        )

    def test_first_hinge_offsets_after_freejoint(
        self, scrambled_model: mujoco.MjModel
    ) -> None:
        """首个 hinge（j7）qpos 偏移=7、qvel 偏移=6，与 freejoint 布局一致。"""
        binding = bind_robot_model(_synthetic_profile(), scrambled_model)
        j7_index = _synthetic_profile().active_joint_names.index("j7")
        assert binding.qpos_indices[j7_index] == 7
        assert binding.qvel_indices[j7_index] == 6

    def test_held_joints_resolved(self, scrambled_model: mujoco.MjModel) -> None:
        """保持关节同样按名称解析地址。"""
        profile = _synthetic_profile(
            active=("j1", "j2"),
            actuators=("act_j1", "act_j2"),
            held=("j5", "j3"),
        )
        binding = bind_robot_model(profile, scrambled_model)
        # j5 qpos=13、j3 qpos=12；qvel=12、11
        assert binding.held_qpos_indices == (13, 12)
        assert binding.held_qvel_indices == (12, 11)

    def test_model_arrays_unchanged_after_bind(
        self, scrambled_model: mujoco.MjModel
    ) -> None:
        """绑定器只读：调用前后模型数组完全一致。"""
        before = {
            "jnt_qposadr": scrambled_model.jnt_qposadr.copy(),
            "jnt_dofadr": scrambled_model.jnt_dofadr.copy(),
            "jnt_type": scrambled_model.jnt_type.copy(),
            "actuator_trnid": scrambled_model.actuator_trnid.copy(),
            "qpos0": scrambled_model.qpos0.copy(),
        }
        bind_robot_model(_synthetic_profile(), scrambled_model)
        for key, snapshot in before.items():
            assert np.array_equal(getattr(scrambled_model, key), snapshot), key


class TestRejection:
    """缺名/错配/非标量关节拒绝。"""

    @pytest.fixture()
    def two_joint_model(self) -> mujoco.MjModel:
        """两个 hinge + 两个执行器的最小模型。"""
        xml = """
<mujoco model="two">
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="jA" type="hinge" axis="0 1 0"/>
      <joint name="jB" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
      <site name="tcp" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="actA" joint="jA"/>
    <motor name="actB" joint="jB"/>
  </actuator>
</mujoco>
"""
        return mujoco.MjModel.from_xml_string(xml)

    def test_missing_joint_rejected(self, two_joint_model: mujoco.MjModel) -> None:
        """主动关节名查不到 → ValueError 含名称。"""
        profile = _synthetic_profile(
            active=("jA", "jX"), actuators=("actA", "actB"), tcp="tcp"
        )
        with pytest.raises(ValueError, match="jX"):
            bind_robot_model(profile, two_joint_model)

    def test_missing_actuator_rejected(self, two_joint_model: mujoco.MjModel) -> None:
        """执行器名查不到 → ValueError 含名称。"""
        profile = _synthetic_profile(
            active=("jA", "jB"), actuators=("actA", "act_missing"), tcp="tcp"
        )
        with pytest.raises(ValueError, match="act_missing"):
            bind_robot_model(profile, two_joint_model)

    def test_missing_tcp_rejected(self, two_joint_model: mujoco.MjModel) -> None:
        """TCP site 查不到 → ValueError 含名称。"""
        profile = _synthetic_profile(
            active=("jA", "jB"), actuators=("actA", "actB"), tcp="nope"
        )
        with pytest.raises(ValueError, match="nope"):
            bind_robot_model(profile, two_joint_model)

    def test_actuator_wrong_joint_rejected(self) -> None:
        """执行器存在但驱动错误关节 → ValueError。"""
        xml = """
<mujoco model="wrongtrn">
  <worldbody>
    <body name="arm" pos="0 0 1">
      <joint name="jA" type="hinge" axis="0 1 0"/>
      <joint name="jB" type="hinge" axis="1 0 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
      <site name="tcp" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <!-- actA 实际驱动 jB：与 profile 期望错配 -->
    <motor name="actA" joint="jB"/>
    <motor name="actB" joint="jB"/>
  </actuator>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        profile = _synthetic_profile(
            active=("jA", "jB"), actuators=("actA", "actB"), tcp="tcp"
        )
        with pytest.raises(ValueError, match="actA"):
            bind_robot_model(profile, model)

    def test_free_joint_as_active_rejected(self) -> None:
        """主动关节为 free（非标量）→ ValueError。"""
        xml = """
<mujoco model="withfree">
  <worldbody>
    <body name="ball" pos="0 0 2">
      <freejoint name="ball_free"/>
      <geom type="sphere" size="0.01"/>
    </body>
    <body name="arm" pos="0 0 1">
      <joint name="jA" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
      <site name="tcp" pos="0 0 0.2"/>
    </body>
  </worldbody>
  <actuator>
    <motor name="actA" joint="jA"/>
    <motor name="actB" joint="jA"/>
  </actuator>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        profile = _synthetic_profile(
            active=("ball_free", "jA"), actuators=("actA", "actB"), tcp="tcp"
        )
        with pytest.raises(ValueError, match="ball_free"):
            bind_robot_model(profile, model)


class TestTypeExport:
    """类型可导入性。"""

    def test_model_binding_exported(self) -> None:
        """ModelBinding 可从模块直接导入（供后续任务类型标注）。"""
        assert ModelBinding is not None
