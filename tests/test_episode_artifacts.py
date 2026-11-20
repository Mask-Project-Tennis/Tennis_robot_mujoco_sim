"""T6 接触事件采集 + ``v12_episode_v2`` 产物测试（先红后绿）。

本文件覆盖 brief §6.1-6.3 与计划 T6 行 148-158：

* ``SimComponent.execute()`` 在 ``step_full`` 之后、弹性反弹之前，用同一
  物理步的 ``env.data.contact`` 采集**全部**球—拍接触事件（含正距离
  margin 事件），并保留 legacy 第一接触判定语义（判据/阈值/break 不动）；
* 新旧命中口径独立：``physical_hit`` / ``physical_hit_type`` 与
  ``legacy_hit`` / ``legacy_hit_type`` 分别计算，分歧写入
  ``legacy_vs_physical_divergence``；零接触回合即使 ``pos_error`` 很小
  也判 miss；
* ``write_episode_artifacts(result, output_dir)`` 写 ``v12_episode_v2``
  JSON + NPZ，含解析配置快照、输入哈希与场景快照，失败回合同样落盘；
* 真实 ``run_v12_episode`` 回合走真实 ``env.data.contact`` 路径（不是
  合成回放/末态推断）。

测试使用一个最小 fake 环境驱动 ``SimComponent.execute``（合成接触仅作
补充单测），真实执行证据由模块级 fixture 跑一次 RM65 短回合提供。
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

pytest.importorskip("mujoco")

import rm65_mpc_v12 as v12  # noqa: E402
from src.ilqt.components.sim_component import (  # noqa: E402
    LEGACY_HIT_RULE,
    PHYSICAL_HIT_RULE,
    ContactEvent,
    SimComponent,
    classify_contact_hit_type,
    classify_surface_kind,
)
from src.ilqt.tube_types import TubeConfig  # noqa: E402
from src.sim import episode_artifacts as artifacts  # noqa: E402
from src.sim.v12_episode import V12EpisodeResult, run_v12_episode  # noqa: E402

# RM65 最短闭环参数（与 T5 回归冒烟一致，仅用于产物 schema 证据）
_RM65_SHORT_ARGV = [
    "--serve-box",
    "--ball-speed", "7",
    "--seed", "42",
    "--no-plot",
    "--horizon", "60",
    "--iter", "2",
    "--near-iters", "2",
    "--first-plan-iters", "2",
    "--near-threshold", "30",
    "--replan-interval", "20",
]

_JOINT_NAMES_RM65 = (
    "r_joint1", "r_joint2", "r_joint3",
    "r_joint4", "r_joint5", "r_joint6",
)

# ── 最小 fake 环境（合成接触补充单测）────────────────────────────────


_FAKE_BALL_POS = np.array([0.30, 0.0, 1.00])


class _FakeData:
    """最小 MuJoCo data 替身（只暴露 execute 读取的字段）。"""

    def __init__(self, size: int) -> None:
        self.ncon = 0
        self.contact: list[object] = []
        self.xpos = np.zeros((size, 3), dtype=np.float64)


class _FakeModel:
    """最小 model 替身（只暴露 geom 名查询）。"""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def geom(self, geom_id: int) -> SimpleNamespace:
        return SimpleNamespace(name=self._names[int(geom_id)])


class _FakeMPC:
    """最小 MPC 替身（读取 _k_hit / _p_hit / _n_des）。"""

    def __init__(self, *, k_hit: int = 1, p_hit: np.ndarray | None = None) -> None:
        self._k_hit = int(k_hit)
        self._p_hit = (
            np.array([0.30, 0.0, 1.00], dtype=np.float64)
            if p_hit is None
            else np.asarray(p_hit, dtype=np.float64)
        )
        self._n_des = np.array([0.0, -1.0, 0.0], dtype=np.float64)


class _FakeLimits:
    """最小限位替身（execute 只读 qdot_max）。"""

    def __init__(self, n: int) -> None:
        self.qdot_max = np.ones(n, dtype=np.float64)


class _FakeEnv:
    """最小仿真环境替身。

    接触仅在 ``set_arm_collision(True)`` 后的 ``step_full`` 中可见——与
    MuJoCo 的接触对开关语义一致，用于验证"窗口关闭不扫描接触"。
    """

    def __init__(
        self,
        names: list[str],
        *,
        n: int = 6,
        ee_pos: np.ndarray | None = None,
        ee_vel: np.ndarray | None = None,
        face_speed: float = 0.0,
        ball_pos: np.ndarray | None = None,
        pending_contacts: list[object] | None = None,
        dt: float = 0.005,
    ) -> None:
        self.NQ = n
        self.NU = n
        self.dt = dt
        self.model = _FakeModel(names)
        self.data = _FakeData(size=n)
        self._ee_pos = (
            np.asarray(ee_pos, dtype=np.float64)
            if ee_pos is not None
            else _FAKE_BALL_POS.copy()
        )
        self._ee_vel = (
            np.asarray(ee_vel, dtype=np.float64)
            if ee_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        self._ee_normal = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        self._face_speed = float(face_speed)
        self._ball_pos = (
            np.asarray(ball_pos, dtype=np.float64)
            if ball_pos is not None
            else _FAKE_BALL_POS.copy()
        )
        self._ball_vel = np.zeros(3, dtype=np.float64)
        self.pending_contacts: list[object] = list(pending_contacts or [])
        self.collision_enabled = True
        self.control_range = np.tile(
            np.array([-100.0, 100.0], dtype=np.float64), (n, 1),
        )
        self.arm_state = np.zeros(2 * n, dtype=np.float64)
        self.step_count = 0

    # -- 只读接口 ------------------------------------------------------
    def update_kinematics(self) -> None:
        return None

    def get_ball_pos(self) -> np.ndarray:
        return self._ball_pos.copy()

    def get_ball_vel(self) -> np.ndarray:
        return self._ball_vel.copy()

    def get_ee_pos(self) -> np.ndarray:
        return self._ee_pos.copy()

    def get_ee_vel(self) -> np.ndarray:
        return self._ee_vel.copy()

    def get_ee_normal(self) -> np.ndarray:
        return self._ee_normal.copy()

    def get_racket_face_speed(self) -> float:
        return float(self._face_speed)

    def get_arm_state(self) -> np.ndarray:
        return self.arm_state.copy()

    # -- 写接口 --------------------------------------------------------
    def set_arm_collision(self, enabled: bool) -> None:
        self.collision_enabled = bool(enabled)

    def set_ball_vel(self, velocity: np.ndarray) -> None:
        self._ball_vel = np.asarray(velocity, dtype=np.float64).copy()

    def step_full(
        self, u_cmd: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.step_count += 1
        # 碰撞窗口关闭时 MuJoCo 物理上没有球—拍接触对
        if self.collision_enabled:
            self.data.contact = list(self.pending_contacts)
            self.data.ncon = len(self.pending_contacts)
        else:
            self.data.contact = []
            self.data.ncon = 0
        self._ball_pos = self._ball_pos + self._ball_vel * self.dt
        return (
            self.arm_state.copy(),
            self._ball_pos.copy(),
            self._ball_vel.copy(),
        )


def _contact(geom1: int, geom2: int, dist: float) -> SimpleNamespace:
    """构造 MuJoCo 接触替身。"""
    return SimpleNamespace(geom1=geom1, geom2=geom2, dist=float(dist))


def _make_component(
    env: _FakeEnv,
    *,
    k_hit: int = 1,
    ball_geom_id: int | None = None,
    racket_geom_ids: tuple[int, ...] | None = None,
) -> SimComponent:
    """构造被测 SimComponent（接触 ID 走 R1D 分支，缺省走 RM65 分支）。"""
    return SimComponent(
        env=env,
        mpc=_FakeMPC(k_hit=k_hit),
        robot_limits=_FakeLimits(env.NQ),
        init_q=np.zeros(env.NQ, dtype=np.float64),
        is_position_mode=False,
        hard_x_body_ids=[],
        initial_hitting_tube=None,
        tube_cfg=TubeConfig(),
        dt=env.dt,
        ball_geom_id=ball_geom_id,
        racket_geom_ids=racket_geom_ids,
    )


def _r1d_names() -> list[str]:
    """R1D 风格 geom 名（球 + 拍面/框/喉/柄）。"""
    return ["ball", "racket_face", "racket_frame", "racket_throat", "racket_handle"]


def _rm65_names() -> list[str]:
    """RM65 风格 geom 名（球 + 拍面/柄）。"""
    return ["ball", "r_racket_face", "r_racket_frame", "r_racket_handle"]


# ── 1. ContactEvent 契约 ─────────────────────────────────────────────


def test_contact_event_to_dict_has_fixed_contract_fields() -> None:
    """ContactEvent.to_dict() 必须暴露 brief 规定的全部字段名。"""
    event = ContactEvent(
        step=7,
        time_s=0.035,
        ball_geom_name="ball",
        other_geom_name="racket_face",
        surface_kind="face",
        distance_m=-0.001,
        face_speed_m_s=1.5,
        physical_contact=True,
        hit_type="active",
    )
    data = event.to_dict()
    assert set(data) == {
        "step",
        "time_s",
        "ball_geom_name",
        "other_geom_name",
        "surface_kind",
        "distance_m",
        "face_speed_m_s",
        "physical_contact",
        "hit_type",
    }
    assert data["step"] == 7
    assert data["time_s"] == pytest.approx(0.035)
    assert data["physical_contact"] is True
    assert data["hit_type"] == "active"


def test_surface_kind_and_hit_type_classification() -> None:
    """拍 geom 名 → face/frame/handle/other；命中分类按速度阈值。"""
    assert classify_surface_kind("r_racket_face") == "face"
    assert classify_surface_kind("racket_face") == "face"
    assert classify_surface_kind("racket_frame") == "frame"
    assert classify_surface_kind("racket_throat") == "frame"
    assert classify_surface_kind("r_racket_handle") == "handle"
    assert classify_surface_kind("ball") == "other"

    assert classify_contact_hit_type("face", True, 0.31) == "active"
    assert classify_contact_hit_type("face", True, 0.30) == "passive"
    assert classify_contact_hit_type("face", False, 2.0) == "miss"
    assert classify_contact_hit_type("frame", True, 2.0) == "miss"


# ── 2. 真实物理步接触采集（R1D 按 ID 分支）───────────────────────────


def test_active_face_contact_recorded_from_same_physics_step() -> None:
    """球—拍面真实物理接触：事件取当步拍速，physical_hit=active。"""
    env = _FakeEnv(
        _r1d_names(),
        face_speed=0.55,
        ee_vel=np.array([0.0, 0.55, 0.0]),
        pending_contacts=[_contact(0, 1, -0.002)],
    )
    component = _make_component(env, ball_geom_id=0, racket_geom_ids=(1, 2, 4))
    component.execute(np.zeros(env.NU))

    assert len(component.contact_events) == 1
    event = component.contact_events[0]
    assert event.step == 0
    assert event.time_s == pytest.approx(0.0)
    assert event.ball_geom_name == "ball"
    assert event.other_geom_name == "racket_face"
    assert event.surface_kind == "face"
    assert event.distance_m == pytest.approx(-0.002)
    assert event.face_speed_m_s == pytest.approx(0.55)
    assert event.physical_contact is True
    assert event.hit_type == "active"
    # 聚合：physical 口径独立于 legacy，二者此处一致
    assert component.physical_hit is True
    assert component.physical_hit_type == "active"
    assert component.physical_hit_step == 0
    assert component.active_contact is True
    summary = component.get_metrics()
    assert summary["legacy_hit"] is True
    assert summary["legacy_hit_type"] == "active"
    assert summary["legacy_hit_rule"] == LEGACY_HIT_RULE
    assert summary["physical_hit_rule"] == PHYSICAL_HIT_RULE
    assert summary["legacy_vs_physical_divergence"] is None
    assert summary["contact_events"][0]["step"] == 0


def test_low_speed_face_contact_is_passive() -> None:
    """拍面真实接触但拍速 ≤0.3 → passive（新旧口径同判）。"""
    env = _FakeEnv(
        _r1d_names(),
        face_speed=0.10,
        pending_contacts=[_contact(0, 1, -0.001)],
    )
    component = _make_component(env, ball_geom_id=0, racket_geom_ids=(1, 2, 4))
    component.execute(np.zeros(env.NU))

    assert component.physical_hit is True
    assert component.physical_hit_type == "passive"
    assert component.passive_contact is True
    assert component.active_contact is False
    assert component.contact_events[0].hit_type == "passive"


def test_zero_contact_is_miss_even_with_tiny_pos_error() -> None:
    """零接触回合：pos_error≈0 也不得判命中（physical/legacy 均 miss）。"""
    env = _FakeEnv(
        _r1d_names(),
        ee_pos=_FAKE_BALL_POS.copy(),  # pos_error ≈ 0
        pending_contacts=[],
    )
    component = _make_component(env, ball_geom_id=0, racket_geom_ids=(1, 2, 4))
    component.execute(np.zeros(env.NU))

    assert component.contact_events == []
    assert component.physical_hit is False
    assert component.physical_hit_type == "miss"
    assert component.ball_was_hit is False
    metrics = component.get_metrics()
    assert metrics["legacy_hit"] is False
    assert metrics["legacy_hit_type"] == "miss"
    assert np.linalg.norm(np.asarray(metrics["pos_error_history"]) ) == pytest.approx(0.0)


def test_positive_margin_contact_not_counted_as_physical_hit() -> None:
    """正距离 margin 接触记为 proximity 事件，不进入 physical_hit（R1D）。"""
    env = _FakeEnv(
        _r1d_names(),
        face_speed=2.0,
        pending_contacts=[_contact(0, 1, 0.004)],
    )
    component = _make_component(env, ball_geom_id=0, racket_geom_ids=(1, 2, 4))
    component.execute(np.zeros(env.NU))

    assert len(component.contact_events) == 1
    event = component.contact_events[0]
    assert event.distance_m == pytest.approx(0.004)
    assert event.physical_contact is False
    assert event.hit_type == "miss"
    assert component.physical_hit is False
    # R1D legacy 判定要求 dist<=0，因此 legacy 也不命中，无分歧
    assert component.ball_was_hit is False
    assert component.get_metrics()["legacy_vs_physical_divergence"] is None


# ── 3. legacy 第一接触语义与新旧分歧（RM65 名子串分支）───────────────


def test_multiple_contacts_same_step_all_recorded_once() -> None:
    """同一步多接触：事件全保留，回合只聚合一次 physical_hit。"""
    env = _FakeEnv(
        _rm65_names(),
        face_speed=2.0,
        ee_vel=np.array([0.0, 2.0, 0.0]),
        pending_contacts=[
            _contact(0, 1, -0.002),  # ball-face
            _contact(0, 2, -0.001),  # ball-frame
            _contact(0, 3, -0.001),  # ball-handle
        ],
    )
    component = _make_component(env)
    component.execute(np.zeros(env.NU))

    assert len(component.contact_events) == 3
    assert {e.surface_kind for e in component.contact_events} == {
        "face", "frame", "handle",
    }
    assert component.physical_hit is True
    assert component.physical_hit_type == "active"
    # legacy 仍是"第一个球—拍接触即命中"（RM65 行为逐字保留）
    assert component.active_contact is True
    # 第二步：已击中，碰撞窗口关闭，不再追加事件
    component.execute(np.zeros(env.NU))
    assert len(component.contact_events) == 3
    assert component.hit_step == 0


def test_positive_margin_legacy_divergence_recorded() -> None:
    """RM65 名子串路径：正距离接触 legacy 命中但 physical 不命中，须记录分歧。"""
    env = _FakeEnv(
        _rm65_names(),
        face_speed=2.0,
        ee_vel=np.array([0.0, 2.0, 0.0]),
        pending_contacts=[_contact(0, 1, 0.003)],
    )
    component = _make_component(env)
    component.execute(np.zeros(env.NU))

    assert component.ball_was_hit is True  # legacy：无 dist 过滤
    assert component.physical_hit is False
    divergence = component.get_metrics()["legacy_vs_physical_divergence"]
    assert divergence is not None
    assert divergence["reason"] == "legacy_hit_without_physical_face_contact"


def test_frame_only_contact_recorded_but_not_physical_hit() -> None:
    """仅框/柄接触：事件保留、legacy 命中，但 physical 口径为 miss 且记分歧。"""
    env = _FakeEnv(
        _rm65_names(),
        face_speed=2.0,
        ee_vel=np.array([0.0, 2.0, 0.0]),
        pending_contacts=[_contact(0, 3, -0.001)],  # ball-handle
    )
    component = _make_component(env)
    component.execute(np.zeros(env.NU))

    assert len(component.contact_events) == 1
    assert component.contact_events[0].surface_kind == "handle"
    assert component.ball_was_hit is True
    assert component.physical_hit is False
    divergence = component.get_metrics()["legacy_vs_physical_divergence"]
    assert divergence is not None
    assert divergence["reason"] == "legacy_hit_without_physical_face_contact"


def test_collision_window_closed_records_no_events() -> None:
    """碰撞窗口关闭（k_hit 远、距离远）时即使有潜在接触也不得扫描。"""
    env = _FakeEnv(
        _r1d_names(),
        ee_pos=np.array([1.5, 0.0, 1.0]),  # dist 远 >0.35
        ball_pos=np.array([0.3, 0.0, 1.0]),
        pending_contacts=[_contact(0, 1, -0.002)],
    )
    component = _make_component(
        env, k_hit=500, ball_geom_id=0, racket_geom_ids=(1, 2, 4),
    )
    component.execute(np.zeros(env.NU))

    assert component.contact_events == []
    assert component.physical_hit is False
    # execute 结束时恢复碰撞（legacy 行为），但本步窗口内不应扫描接触
    assert env.collision_enabled is True


# ── 4. 产物 writer（合成 result）─────────────────────────────────────


def _parse_and_resolve(argv: list[str]):
    """用真实 CLI 解析器 + 解析层构造 ResolvedV12Config。"""
    parser = v12._build_parser()
    args = parser.parse_args(argv)
    explicit = v12._explicit_cli_actions(parser, argv)
    return v12.resolve_v12_config(args, explicit)


def _fabricated_result(
    config: object,
    *,
    frames: int = 4,
    failure_reason: str | None = None,
    with_contacts: bool = True,
) -> V12EpisodeResult:
    """构造最小 V12EpisodeResult（合成产物 schema 单测用）。"""
    n = len(_JOINT_NAMES_RM65)
    n_rows = 1 if failure_reason == "ball_unreachable" else frames + 1
    n_control = 0 if failure_reason == "ball_unreachable" else frames
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n_rows, 2 * n)) * 0.1
    U = rng.normal(size=(n_control, n)) * 0.01
    ball_pos = np.tile(_FAKE_BALL_POS, (n_rows, 1))
    ball_vel = np.tile(np.array([0.0, -5.0, 1.0]), (n_rows, 1))
    time = np.arange(n_rows, dtype=np.float64) * float(config.dt)
    racket_pos = np.tile(np.array([0.30, 0.0, 1.00]), (n_rows, 1))

    contact_events = []
    if with_contacts:
        contact_events = [
            ContactEvent(
                step=2,
                time_s=2 * float(config.dt),
                ball_geom_name="ball",
                other_geom_name="r_racket_face",
                surface_kind="face",
                distance_m=-0.002,
                face_speed_m_s=1.2,
                physical_contact=True,
                hit_type="active",
            ).to_dict()
        ]
    metrics = {
        "total_steps": n_control,
        "safe_steps": n_control,
        "mpc_done": failure_reason is None,
        "first_plan_ms": 1.0,
        "wall_time_s": 2.0,
        "mpc_time_s": 1.0,
        "phase_timings": {"far": 0.1, "mid": 0.2, "near": 0.3},
        "backend": {"requested": "cpp", "effective": "python-analytic"},
        "failure_reason": failure_reason,
        "ablation_mode": "off",
        "is_position_mode": config.actuator_mode == "position",
        "n_mpc_steps": n_control,
        "min_dist": 0.02,
        "max_qdot_ratio": 0.5,
        "max_tcp_speed": 1.0,
        "max_racket_face_speed": 1.2,
        "pos_error": 0.05,
        "shoulder_pos": [0.0, 0.0, 1.0],
        "shoulder_pos_source": "cli/scene.shoulder_pos",
        "proximity": {
            "ball_radius": 0.033,
            "racket_radius": 0.12,
            "proximity_margin": 0.03,
            "ball_near_threshold": 0.183,
        },
        "hit_window": None,
        "serve": {
            "p0": np.asarray(_FAKE_BALL_POS, dtype=np.float64),
            "v0": np.array([0.0, -5.0, 1.0], dtype=np.float64),
        },
    }
    if with_contacts:
        metrics.update(
            {
                "physical_hit": True,
                "physical_hit_type": "active",
                "physical_hit_step": 2,
                "legacy_hit": True,
                "legacy_hit_type": "active",
                "legacy_hit_rule": LEGACY_HIT_RULE,
                "physical_hit_rule": PHYSICAL_HIT_RULE,
                "legacy_vs_physical_divergence": None,
                "contact_events": contact_events,
            }
        )
    hit_type = "active" if with_contacts else "miss"
    hit_step = 2 if with_contacts else -1
    return V12EpisodeResult(
        config=config,
        robot_id="rm65",
        mode=config.actuator_mode,
        joint_names=_JOINT_NAMES_RM65,
        dt=float(config.dt),
        racket_id=None,
        X=X,
        U=U,
        ball_pos=ball_pos,
        ball_vel=ball_vel,
        time=time,
        racket_pos=racket_pos,
        hit_type=hit_type,
        hit_step=hit_step,
        active_contact=with_contacts,
        passive_contact=False,
        v_ee_at_hit=1.2 if with_contacts else None,
        p_ee_at_hit=np.array([0.30, 0.0, 1.00]) if with_contacts else None,
        ball_pos_at_hit=np.asarray(_FAKE_BALL_POS) if with_contacts else None,
        ball_out_speed=5.0 if with_contacts else None,
        ball_out_vy=1.0 if with_contacts else None,
        metrics=metrics,
    )


def _sha256(path: Path) -> str:
    """文件 SHA256。"""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_write_episode_artifacts_roundtrip(tmp_path: Path) -> None:
    """JSON schema/键/单位/NPZ 形状与快照清单完整可回读。"""
    config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = _fabricated_result(config, frames=4)
    out = tmp_path / "episode"
    payload = artifacts.write_episode_artifacts(result, out)

    json_path = out / artifacts.ARTIFACT_JSON_NAME
    npz_path = out / artifacts.ARTIFACT_NPZ_NAME
    assert json_path.is_file() and npz_path.is_file()

    on_disk = json.loads(json_path.read_text(encoding="utf-8"))
    assert on_disk["schema"] == "v12_episode_v2"
    assert on_disk["simulation_only"] is True
    assert on_disk["robot_id"] == "rm65"
    assert on_disk["mode"] == "torque"
    assert on_disk["joint_names"] == list(_JOINT_NAMES_RM65)
    assert on_disk["dt"] == pytest.approx(float(config.dt))
    assert on_disk["units"]["position"] == "m"
    assert on_disk["units"]["linear_velocity"] == "m/s"
    assert on_disk["units"]["angle"] == "rad"
    assert on_disk["units"]["control_torque"] == "N*m"
    assert on_disk["control"]["mode"] == "torque"
    assert on_disk["control"]["units"] == "N*m"
    assert set(on_disk["hit"]) >= {
        "legacy_hit", "legacy_hit_type", "legacy_hit_rule",
        "physical_hit", "physical_hit_type", "physical_hit_rule",
        "physical_hit_step", "legacy_vs_physical_divergence",
        "contact_events", "pos_error", "min_dist",
        "ball_out_speed", "ball_out_vy",
    }
    assert on_disk["hit"]["legacy_hit"] is True
    assert on_disk["hit"]["physical_hit"] is True
    assert on_disk["hit"]["physical_hit_type"] == "active"
    assert on_disk["hit"]["legacy_hit_rule"] == LEGACY_HIT_RULE
    assert on_disk["hit"]["contact_events"][0]["surface_kind"] == "face"
    assert on_disk["input_sha256"]["model"]
    assert on_disk["input_sha256"]["scene_snapshot"]
    assert "config" in on_disk and "seeds" not in on_disk
    assert on_disk["config"]["robot_id"] == "rm65"
    assert on_disk["scene_snapshot"]["proximity"]["racket_radius"] == pytest.approx(0.12)
    assert on_disk["backend"]["solver_backend"] == config.solver_backend
    assert on_disk["episode"]["failure_reason"] is None
    # JSON 全可序列化（无 numpy / Path 泄漏）
    json.dumps(on_disk, ensure_ascii=False, allow_nan=False)

    with np.load(npz_path) as npz:
        assert set(npz.files) >= {
            "X", "U", "ball_pos", "ball_vel", "time", "racket_pos",
        }
        assert npz["X"].shape == (5, 12)
        assert npz["U"].shape == (4, 6)
        assert npz["ball_pos"].shape == (5, 3)
        assert npz["ball_vel"].shape == (5, 3)
        assert npz["time"].shape == (5,)
        assert npz["racket_pos"].shape == (5, 3)

    loaded = artifacts.load_episode_artifacts(out)
    assert loaded.payload["schema"] == "v12_episode_v2"
    assert loaded.X.shape == (5, 12)
    assert loaded.U.shape == (4, 6)
    np.testing.assert_allclose(loaded.ball_pos, result.ball_pos)

    # payload 返回值与磁盘一致
    assert payload["schema"] == on_disk["schema"]

    # 快照目录与清单
    snapshot = out / artifacts.SNAPSHOT_DIR_NAME
    manifest_path = snapshot / artifacts.SNAPSHOT_MANIFEST_NAME
    assert manifest_path.is_file()
    assert (snapshot / artifacts.SNAPSHOT_CONFIG_NAME).is_file()
    assert (snapshot / artifacts.SNAPSHOT_SCENE_NAME).is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["files"], "快照清单不得为空"
    for entry in manifest["files"]:
        copied = snapshot / entry["relative_path"]
        assert copied.is_file()
        assert _sha256(copied) == entry["sha256"]
        assert entry["sha256"] == on_disk["input_sha256"][entry["role"]]


def test_write_episode_artifacts_failed_episode_also_written(tmp_path: Path) -> None:
    """不可达/失败回合同样落盘（U 为空、failure_reason 保留）。"""
    config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = _fabricated_result(
        config, failure_reason="ball_unreachable", with_contacts=False,
    )
    out = tmp_path / "failed"
    artifacts.write_episode_artifacts(result, out)

    on_disk = json.loads(
        (out / artifacts.ARTIFACT_JSON_NAME).read_text(encoding="utf-8")
    )
    assert on_disk["episode"]["failure_reason"] == "ball_unreachable"
    assert on_disk["hit"]["legacy_hit"] is False
    assert on_disk["hit"]["physical_hit"] is False
    assert on_disk["hit"]["contact_events"] == []
    with np.load(out / artifacts.ARTIFACT_NPZ_NAME) as npz:
        assert npz["X"].shape == (1, 12)
        assert npz["U"].shape == (0, 6)


def test_write_episode_artifacts_sanitizes_non_finite_metrics(
    tmp_path: Path,
) -> None:
    """失败回合里的 inf/nan 指标转 null，产物仍是合法 JSON。

    回归：R1D ``ball_unreachable`` 回合的 ``min_dist``/``pos_error`` 为 ``inf``，
    ``json.dumps(..., allow_nan=False)`` 会直接抛 ``ValueError``（真实复跑中命中）。
    """
    config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = _fabricated_result(
        config, failure_reason="ball_unreachable", with_contacts=False,
    )
    metrics = dict(result.metrics)
    metrics["min_dist"] = float("inf")
    metrics["pos_error"] = float("nan")
    broken = V12EpisodeResult(**{**result.__dict__, "metrics": metrics})

    out = tmp_path / "non_finite"
    payload = artifacts.write_episode_artifacts(broken, out)

    raw = (out / artifacts.ARTIFACT_JSON_NAME).read_text(encoding="utf-8")
    assert "Infinity" not in raw and "NaN" not in raw
    assert payload["hit"]["min_dist"] is None
    assert payload["hit"]["pos_error"] is None


def test_write_episode_artifacts_rejects_misaligned_arrays(tmp_path: Path) -> None:
    """X/U/球轨迹行数不对齐时必须显式拒绝，不得写出坏产物。"""
    config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = _fabricated_result(config, frames=4)
    broken = V12EpisodeResult(
        **{
            **result.__dict__,
            "U": result.U[:-1],
        }
    )
    with pytest.raises(ValueError):
        artifacts.write_episode_artifacts(broken, tmp_path / "broken")


def test_position_mode_units_do_not_write_torque(tmp_path: Path) -> None:
    """position 模式的 U 单位是 rad，不得声称 torque。"""
    config = _parse_and_resolve([*_RM65_SHORT_ARGV, "--position-mode"])
    result = _fabricated_result(config, frames=3)
    out = tmp_path / "position"
    payload = artifacts.write_episode_artifacts(result, out)
    assert payload["mode"] == "position"
    assert payload["control"]["units"] == "rad"
    assert payload["units"]["control_position"] == "rad"
    assert payload["hit"]["legacy_hit_type"] == result.hit_type


# ── 5. 真实执行证据：真实 env.data.contact 路径 ───────────────────────


@pytest.fixture(scope="module")
def real_episode_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """跑一次真实 RM65 短回合并写 v2 产物（真实接触路径证据）。"""
    config = _parse_and_resolve(_RM65_SHORT_ARGV)
    result = run_v12_episode(config)
    out = tmp_path_factory.mktemp("real_rm65_episode")
    artifacts.write_episode_artifacts(result, out)
    return out


def test_real_rm65_episode_artifacts_match_runtime_result(
    real_episode_dir: Path,
) -> None:
    """真实 run_v12_episode 产物：contact_events 来自真实物理步。"""
    payload = json.loads(
        (real_episode_dir / artifacts.ARTIFACT_JSON_NAME).read_text(
            encoding="utf-8"
        )
    )
    assert payload["schema"] == "v12_episode_v2"
    assert payload["robot_id"] == "rm65"
    assert payload["mode"] in {"torque", "position"}
    events = payload["hit"]["contact_events"]
    assert isinstance(events, list)
    for event in events:
        assert set(event) >= {
            "step", "time_s", "ball_geom_name", "other_geom_name",
            "surface_kind", "distance_m", "face_speed_m_s",
            "physical_contact", "hit_type",
        }
        assert event["time_s"] == pytest.approx(
            event["step"] * payload["dt"], abs=1e-12,
        )
        assert event["surface_kind"] in {"face", "frame", "handle", "other"}
        assert event["hit_type"] in {"active", "passive", "miss"}
    # physical 聚合必须与事件自洽
    physical_faces = [
        e for e in events
        if e["physical_contact"] and e["surface_kind"] == "face"
    ]
    assert payload["hit"]["physical_hit"] is bool(physical_faces)
    if not physical_faces:
        assert payload["hit"]["physical_hit_type"] == "miss"
    # 旧 New 口径分歧字段必须存在（可能为 None）
    assert "legacy_vs_physical_divergence" in payload["hit"]
    assert payload["hit"]["legacy_hit"] == (
        payload["hit"]["legacy_hit_type"] in {"active", "passive"}
    )
    # 快照与 NPZ 行对齐
    with np.load(real_episode_dir / artifacts.ARTIFACT_NPZ_NAME) as npz:
        n_rows = npz["X"].shape[0]
        assert npz["U"].shape[0] == n_rows - 1
        assert npz["ball_pos"].shape[0] == n_rows
        assert npz["time"].shape[0] == n_rows
        assert npz["racket_pos"].shape[0] == n_rows


# ── 6. 依赖资产哈希（T6 追加：设计 §6 依赖资产哈希快照）────────────────


def _load_payload(directory: Path) -> dict:
    """读取产物 JSON。"""
    return json.loads(
        (directory / artifacts.ARTIFACT_JSON_NAME).read_text(encoding="utf-8")
    )


def _write_payload(directory: Path, payload: dict) -> None:
    """写回产物 JSON（测试篡改用）。"""
    (directory / artifacts.ARTIFACT_JSON_NAME).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _manifest_path(directory: Path) -> Path:
    """产物快照 manifest 路径。"""
    return (
        directory
        / artifacts.SNAPSHOT_DIR_NAME
        / artifacts.SNAPSHOT_MANIFEST_NAME
    )


def _write_temp_robot_profile(
    tmp_path: Path,
    model_xml: str,
    asset_files: dict[str, bytes],
) -> tuple[Path, dict[str, Path]]:
    """用临时模型 XML + 资产写一个可加载的机器人描述，返回 (profile, 资产路径)。"""
    model_dir = tmp_path / "temp_robot"
    model_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for name, content in asset_files.items():
        path = model_dir / name
        path.write_bytes(content)
        written[name] = path
    model_path = model_dir / "temp_model.xml"
    model_path.write_text(model_xml, encoding="utf-8")
    template = (ROOT / "configs" / "robots" / "rm65.yaml").read_text(encoding="utf-8")
    profile_text = re.sub(
        r"(?m)^\s*model_path:.*$",
        f"model_path: {model_path.as_posix()}",
        template,
        count=1,
    )
    profile_path = tmp_path / "temp_profile.yaml"
    profile_path.write_text(profile_text, encoding="utf-8")
    return profile_path, written


def test_real_rm65_artifact_records_model_asset_hashes(
    real_episode_dir: Path,
) -> None:
    """真实 RM65 产物：manifest/payload 记录所有 mesh 依赖逐文件哈希。"""
    payload = _load_payload(real_episode_dir)
    assets = payload["assets"]
    assert assets["status"] == "ok"
    assert assets["count"] >= 1
    assert payload["input_sha256"]["assets"] == assets["digest"]
    assert payload["input_sha256"]["assets"]
    assert {item["kind"] for item in assets["files"]} <= {"mesh", "texture"}
    for item in assets["files"]:
        assert item["role"] == "assets"
        source_path = ROOT / item["relative_path"]
        assert source_path.is_file()
        assert artifacts._sha256_file(source_path) == item["sha256"]
        assert source_path.stat().st_size == item["size"]
    # manifest 逐文件条目与 payload 完全一致，聚合哈希可重算
    manifest = json.loads(_manifest_path(real_episode_dir).read_text(encoding="utf-8"))
    assert manifest["assets"]["digest"] == assets["digest"]
    assert manifest["assets"]["files"] == assets["files"]
    assert artifacts._assets_digest(assets["files"]) == assets["digest"]


def test_replay_rejects_tampered_manifest_asset_hash(
    tmp_path: Path,
    real_episode_dir: Path,
) -> None:
    """篡改 manifest 中某 asset 的 sha256 后，回放必须明确拒绝。"""
    target = tmp_path / "tampered_manifest"
    shutil.copytree(real_episode_dir, target)
    manifest = json.loads(_manifest_path(target).read_text(encoding="utf-8"))
    assert manifest["assets"]["files"]
    manifest["assets"]["files"][0]["sha256"] = "0" * 64
    _manifest_path(target).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="依赖资产|哈希|不一致"):
        artifacts.replay_episode_artifacts(target, viewer=False)


def test_replay_rejects_tampered_mesh_file(
    tmp_path: Path,
) -> None:
    """新产物记录临时 mesh 哈希；mesh 被改后回放必须拒绝（含路径）。"""
    profile_path, written = _write_temp_robot_profile(
        tmp_path,
        model_xml=(
            '<mujoco model="temp">'
            '<compiler angle="degree"/>'
            '<asset><mesh name="m" file="temp_mesh.STL"/></asset>'
            "</mujoco>\n"
        ),
        asset_files={"temp_mesh.STL": b"solid temp\nendsolid temp\n"},
    )
    config = replace(
        _parse_and_resolve(_RM65_SHORT_ARGV), robot_profile_path=profile_path
    )
    out = tmp_path / "temp_asset_episode"
    payload = artifacts.write_episode_artifacts(
        _fabricated_result(config, frames=4), out
    )
    assets = payload["assets"]
    assert assets["status"] == "ok"
    assert assets["count"] == 1
    entry = assets["files"][0]
    mesh_path = written["temp_mesh.STL"]
    assert entry["kind"] == "mesh"
    assert Path(entry["source"]).resolve() == mesh_path.resolve()
    assert entry["sha256"] == _sha256(mesh_path)
    assert payload["input_sha256"]["assets"]

    mesh_path.write_bytes(b"solid tampered\nendsolid tampered\n")
    with pytest.raises(ValueError, match="依赖资产|哈希|不一致"):
        artifacts.replay_episode_artifacts(out, viewer=False)


def test_write_episode_artifacts_marks_missing_asset_references(
    tmp_path: Path,
) -> None:
    """模型引用缺失资产时显式记录 missing 状态（非静默跳过）。"""
    profile_path, _written = _write_temp_robot_profile(
        tmp_path,
        model_xml=(
            '<mujoco model="temp">'
            '<compiler angle="degree"/>'
            '<asset><mesh name="m" file="not_here.STL"/></asset>'
            "</mujoco>\n"
        ),
        asset_files={},
    )
    config = replace(
        _parse_and_resolve(_RM65_SHORT_ARGV), robot_profile_path=profile_path
    )
    out = tmp_path / "missing_asset_episode"
    payload = artifacts.write_episode_artifacts(
        _fabricated_result(config, frames=4), out
    )
    assets = payload["assets"]
    assert assets["status"] == "missing"
    assert assets["files"] == []
    assert assets["missing"]
    assert "not_here.STL" in assets["missing"][0]
    assert payload["input_sha256"]["assets"] == artifacts._assets_digest([])
    with pytest.raises(ValueError, match="依赖资产|缺失|哈希"):
        artifacts.replay_episode_artifacts(out, viewer=False)


def test_replay_accepts_legacy_payload_without_asset_hashes(
    tmp_path: Path,
    real_episode_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """旧 artifact 无 assets 键时仍可回放（仅警告，兼容 T6 之前产物）。"""
    target = tmp_path / "legacy_without_assets"
    shutil.copytree(real_episode_dir, target)
    payload = _load_payload(target)
    payload["input_sha256"].pop("assets", None)
    payload.pop("assets", None)
    _write_payload(target, payload)
    manifest = json.loads(_manifest_path(target).read_text(encoding="utf-8"))
    manifest.pop("assets", None)
    _manifest_path(target).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with caplog.at_level(logging.WARNING, logger="src.sim.episode_artifacts"):
        replay = artifacts.replay_episode_artifacts(target, viewer=False)
    assert replay.viewer_opened is False
    with np.load(target / artifacts.ARTIFACT_NPZ_NAME) as npz:
        assert replay.racket_positions.shape == npz["racket_pos"].shape
    assert any("旧 artifact" in record.getMessage() for record in caplog.records)



def test_hit_message_uses_robot_display_name() -> None:
    """评审 N1：R1D 失败回合文案不得误标 "RM-65"；RM65 旧文案保持。"""
    def _fake(robot_id: str) -> SimpleNamespace:
        return SimpleNamespace(
            robot_id=robot_id,
            metrics={"pos_error": 0.0},
            hit_type="miss",
            active_contact=False,
            passive_contact=False,
            hit_step=-1,
            config=SimpleNamespace(dt=0.005),
            v_ee_at_hit=None,
            ball_out_speed=None,
            ball_out_vy=None,
        )

    r1d = artifacts._hit_section(_fake("unitree_r1d"))
    assert r1d["message"].startswith("R1-A7-D 未命中")
    assert "RM-65" not in r1d["message"]

    rm65 = artifacts._hit_section(_fake("rm65"))
    assert rm65["message"].startswith("RM-65 未命中")
