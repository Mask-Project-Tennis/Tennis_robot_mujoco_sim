"""ReplanConfig 类型安全配置测试。

验证：
- from_mpc_config 从 MPCConfig 构建 ReplanConfig，所有字段正确映射
- to_dict 产生与旧 _build_replan_cfg() 等价的 dict（向后兼容 do_replan）
- 字段完整性：51 个键全覆盖，无 None 遗漏（除 fix_joint5_angle / Q_p_base / Q_v_base 等可空字段）
"""

from __future__ import annotations

import numpy as np

from src.ilqt.mpc_controller import MPCConfig
from src.ilqt.planning_env import PlanningEnv
from src.ilqt.replan_config import ReplanConfig
from src.real.config import RealRobotConfig
from src.real.runner_factory import build_replan_cfg, build_robot_limits, build_solver

_CFG = RealRobotConfig()


def _build_env() -> PlanningEnv:
    """构建位置模式 PlanningEnv（复用 runner_factory 常量）。"""
    from src.real.runner_factory import DT, INIT_Q, INIT_Q_LEFT, KD, KP

    env = PlanningEnv(dt=DT)
    env.init_q_left = INIT_Q_LEFT.copy()
    env.configure_actuator_mode("position", kp=KP, kd=KD)
    env.configure_feedforward(True)
    env.reset(INIT_Q)
    env.data.qpos[env.NQ : env.NQ + env.LEFT_ARM_NQ] = env.init_q_left
    return env


def _build_config() -> MPCConfig:
    """构建测试用 MPCConfig（含显式非默认值以便断言映射）。"""
    return MPCConfig(
        version="test",
        is_position_mode=True,
        dt=0.005,
        total_horizon=200,
        fixed_horizon=60,
        replan_interval=20,
        max_iter_per_plan=3,
        first_plan_iters=5,
        near_plan_iters=2,
        near_threshold=80,
        R=0.0001,
        normal_weight=500000.0,
        racket_speed=5.0,
        follow_through_length=0.0,
        follow_through_steps=0,
        Q_p_base=np.array([100000.0, 100000.0, 100000.0]),
        Q_v_base=np.array([400.0, 400.0, 400.0]),
        Q_qdot_base=0.001,
        Q_qddot_base=0.0005,
        Q_du_base=0.001,
        far_threshold=50,
    )


def _build_replan_config() -> ReplanConfig:
    """构建 ReplanConfig 实例（复用 env / limits / solver）。"""
    env = _build_env()
    config = _build_config()
    robot_limits = build_robot_limits(env, _CFG)
    solver = build_solver()
    d_hat = np.array([0.0, -1.0, -0.5])
    d_hat /= float(np.linalg.norm(d_hat))
    v_hit_desired = 1.8 * d_hat
    return ReplanConfig.from_mpc_config(
        config,
        robot_limits=robot_limits,
        solver=solver,
        d_hat=d_hat,
        v_hit_desired=v_hit_desired,
    )


class TestFromMpcConfig:
    """from_mpc_config 构造正确性。"""

    def test_all_fields_populated(self) -> None:
        """所有非可空字段非 None，可空字段（fix_joint5_angle/Q_p_base/Q_v_base）允许 None。"""
        rc = _build_replan_config()
        from dataclasses import fields as dc_fields

        nullable = {"fix_joint5_angle", "Q_p_base", "Q_v_base"}
        for f in dc_fields(rc):
            val = getattr(rc, f.name)
            if f.name in nullable:
                continue
            assert val is not None, f"字段 {f.name} 不应为 None"

    def test_scalar_mapping(self) -> None:
        """标量字段从 MPCConfig 正确映射。"""
        rc = _build_replan_config()
        c = _build_config()
        assert rc.dt == c.dt
        assert rc.total_horizon == c.total_horizon
        assert rc.fixed_horizon == c.fixed_horizon
        assert rc.replan_interval == c.replan_interval
        assert rc.max_iter_per_plan == c.max_iter_per_plan
        assert rc.first_plan_iters == c.first_plan_iters
        assert rc.near_plan_iters == c.near_plan_iters
        assert rc.near_threshold == c.near_threshold
        assert rc.R == c.R
        assert rc.normal_weight == c.normal_weight
        assert rc.racket_speed == c.racket_speed
        assert rc.max_tcp_speed == c.max_tcp_speed
        assert rc.ablation_mode == c.ablation_mode
        assert rc.is_position_mode == c.is_position_mode
        assert rc.use_backswing == c.use_backswing
        assert rc.use_r_decay == c.use_r_decay
        assert rc.r_decay_ratio == c.r_decay_ratio
        assert rc.backswing_offset == c.backswing_offset
        assert rc.backswing_ratio == c.backswing_ratio
        assert rc.normal_flip == c.normal_flip
        assert rc.workspace_radius == c.workspace_radius
        assert rc.follow_through_length == c.follow_through_length
        assert rc.follow_through_steps == c.follow_through_steps
        assert rc.follow_through_v_terminal == c.follow_through_v_terminal
        assert rc.far_threshold == c.far_threshold

    def test_array_mapping(self) -> None:
        """数组字段正确映射（值相等）。"""
        rc = _build_replan_config()
        c = _build_config()
        np.testing.assert_array_almost_equal(rc.shoulder_pos, c.shoulder_pos)
        # Q_p_base / Q_v_base 在本测试配置中非 None（显式传入数组）
        assert rc.Q_p_base is not None and c.Q_p_base is not None
        assert rc.Q_v_base is not None and c.Q_v_base is not None
        np.testing.assert_array_almost_equal(rc.Q_p_base, c.Q_p_base)
        np.testing.assert_array_almost_equal(rc.Q_v_base, c.Q_v_base)

    def test_direction_fields(self) -> None:
        """方向字段 d_hat / d_follow / v_hit_desired / v_hit_at_contact 正确设置。"""
        d_hat = np.array([0.0, -1.0, -0.5])
        d_hat /= float(np.linalg.norm(d_hat))
        v_hit_desired = 1.8 * d_hat
        env = _build_env()
        config = _build_config()
        rc = ReplanConfig.from_mpc_config(
            config,
            robot_limits=build_robot_limits(env, _CFG),
            solver=build_solver(),
            d_hat=d_hat,
            v_hit_desired=v_hit_desired,
        )
        np.testing.assert_array_almost_equal(rc.d_hat, d_hat)
        np.testing.assert_array_almost_equal(rc.d_follow, d_hat)  # 默认 d_follow = d_hat
        np.testing.assert_array_almost_equal(rc.v_hit_desired, v_hit_desired)
        np.testing.assert_array_almost_equal(rc.v_hit_at_contact, v_hit_desired)

    def test_d_follow_override(self) -> None:
        """显式传入 d_follow 时覆盖默认（d_follow != d_hat）。"""
        env = _build_env()
        config = _build_config()
        d_hat = np.array([0.0, 1.0, 0.0])
        d_follow_custom = np.array([1.0, 0.0, 0.0])
        rc = ReplanConfig.from_mpc_config(
            config,
            robot_limits=build_robot_limits(env, _CFG),
            solver=build_solver(),
            d_hat=d_hat,
            v_hit_desired=1.8 * d_hat,
            d_follow=d_follow_custom,
        )
        np.testing.assert_array_almost_equal(rc.d_follow, d_follow_custom)

    def test_derived_fields(self) -> None:
        """派生字段：hit_shift = follow_through_length, v_hit_at_contact = v_hit_desired, k_hit_total = total_horizon。"""
        rc = _build_replan_config()
        c = _build_config()
        assert rc.hit_shift == c.follow_through_length
        assert rc.k_hit_total == c.total_horizon

    def test_external_objects_attached(self) -> None:
        """外部对象 robot_limits / solver 正确附加。"""
        rc = _build_replan_config()
        assert rc.robot_limits is not None
        assert rc.solver is not None
        assert hasattr(rc.robot_limits, "q_lower")
        assert hasattr(rc.solver, "solve_few_iters")


class TestToDict:
    """to_dict 向后兼容性。"""

    def test_to_dict_returns_dict(self) -> None:
        """to_dict 返回 dict 类型。"""
        rc = _build_replan_config()
        d = rc.to_dict()
        assert isinstance(d, dict)

    def test_to_dict_key_count(self) -> None:
        """to_dict 包含全部 51 个键（覆盖 do_replan 所需所有 cfg 键）。"""
        rc = _build_replan_config()
        d = rc.to_dict()
        expected_keys = {
            # 时间与迭代
            "dt", "total_horizon", "fixed_horizon", "replan_interval",
            "max_iter_per_plan", "first_plan_iters", "near_plan_iters", "near_threshold",
            # 代价
            "R", "Q_p_scale_far", "Q_v_scale_far", "Q_p_scale_near", "Q_v_scale_near",
            "normal_weight", "racket_speed", "max_tcp_speed",
            "Q_p_base", "Q_v_base", "Q_qdot_base", "Q_qddot_base", "Q_du_base",
            # 几何
            "shoulder_pos", "workspace_radius",
            # 方向与随挥
            "d_hat", "d_follow", "v_hit_desired", "v_hit_at_contact",
            "hit_shift", "follow_through_length", "follow_through_steps", "follow_through_v_terminal",
            # Tube
            "tube_cfg", "ablation_mode",
            # 模式
            "is_position_mode", "use_backswing", "use_r_decay", "r_decay_ratio",
            "fix_joint5_angle", "backswing_offset", "backswing_ratio", "normal_flip",
            # 扰动
            "time_perturb_s", "space_perturb_m", "perturb_alpha_min",
            # 调度
            "smooth_far", "smooth_mid", "smooth_near", "k_hit_total",
            # 远段阈值
            "far_threshold",
            # 外部对象
            "robot_limits", "solver",
        }
        assert set(d.keys()) == expected_keys, (
            f"键集不匹配。缺失: {expected_keys - set(d.keys())}, "
            f"多余: {set(d.keys()) - expected_keys}"
        )

    def test_to_dict_values_match_config(self) -> None:
        """to_dict 值与 MPCConfig 输入一致。"""
        rc = _build_replan_config()
        c = _build_config()
        d = rc.to_dict()
        assert d["dt"] == c.dt
        assert d["total_horizon"] == c.total_horizon
        assert d["R"] == c.R
        assert d["is_position_mode"] == c.is_position_mode
        np.testing.assert_array_almost_equal(d["shoulder_pos"], c.shoulder_pos)


class TestRoundtrip:
    """from_mpc_config → to_dict 等价性（消除 _build_replan_cfg 重复的核心保证）。"""

    def test_dict_matches_legacy_build_replan_cfg(self) -> None:
        """to_dict() 与旧 _build_replan_cfg() 逻辑等价。

        旧 _build_replan_cfg() = runner_factory.build_replan_cfg() 基础字典
        + MPCConfig 覆盖（51 键）。本测试内联旧逻辑作为对照基线。
        """
        env = _build_env()
        config = _build_config()
        robot_limits = build_robot_limits(env, _CFG)
        solver = build_solver()
        d_hat = np.array([0.0, -1.0, -0.5])
        d_hat /= float(np.linalg.norm(d_hat))
        v_hit_desired = 1.8 * d_hat

        # 新路径
        rc = ReplanConfig.from_mpc_config(
            config,
            robot_limits=robot_limits,
            solver=solver,
            d_hat=d_hat,
            v_hit_desired=v_hit_desired,
        )
        new_dict = rc.to_dict()

        # 旧路径基线（内联 _build_replan_cfg 的 cfg.update 逻辑）
        legacy_dict: dict = {
            "dt": config.dt,
            "total_horizon": config.total_horizon,
            "fixed_horizon": config.fixed_horizon,
            "replan_interval": config.replan_interval,
            "max_iter_per_plan": config.max_iter_per_plan,
            "first_plan_iters": config.first_plan_iters,
            "near_plan_iters": config.near_plan_iters,
            "near_threshold": config.near_threshold,
            "R": config.R,
            "Q_p_scale_far": config.Q_p_scale_far,
            "Q_v_scale_far": config.Q_v_scale_far,
            "Q_p_scale_near": config.Q_p_scale_near,
            "Q_v_scale_near": config.Q_v_scale_near,
            "normal_weight": config.normal_weight,
            "racket_speed": config.racket_speed,
            "max_tcp_speed": config.max_tcp_speed,
            "is_position_mode": config.is_position_mode,
            "ablation_mode": config.ablation_mode,
            "use_backswing": config.use_backswing,
            "use_r_decay": config.use_r_decay,
            "r_decay_ratio": config.r_decay_ratio,
            "fix_joint5_angle": config.fix_joint5_angle,
            "backswing_offset": config.backswing_offset,
            "backswing_ratio": config.backswing_ratio,
            "normal_flip": config.normal_flip,
            "shoulder_pos": config.shoulder_pos,
            "workspace_radius": config.workspace_radius,
            "d_hat": d_hat,
            "d_follow": d_hat,
            "v_hit_desired": v_hit_desired,
            "v_hit_at_contact": v_hit_desired,
            "hit_shift": config.follow_through_length,
            "follow_through_length": config.follow_through_length,
            "follow_through_steps": config.follow_through_steps,
            "follow_through_v_terminal": config.follow_through_v_terminal,
            "tube_cfg": config.tube_cfg,
            "smooth_far": config.smooth_far,
            "smooth_mid": config.smooth_mid,
            "smooth_near": config.smooth_near,
            "time_perturb_s": config.time_perturb_s,
            "space_perturb_m": config.space_perturb_m,
            "perturb_alpha_min": config.perturb_alpha_min,
            "Q_p_base": config.Q_p_base,
            "Q_v_base": config.Q_v_base,
            "Q_qdot_base": config.Q_qdot_base,
            "Q_qddot_base": config.Q_qddot_base,
            "Q_du_base": config.Q_du_base,
            "far_threshold": config.far_threshold,
            "k_hit_total": config.total_horizon,
            "robot_limits": robot_limits,
            "solver": solver,
        }

        assert set(new_dict.keys()) == set(legacy_dict.keys())
        for k in legacy_dict:
            nv, lv = new_dict[k], legacy_dict[k]
            if isinstance(lv, np.ndarray):
                np.testing.assert_array_almost_equal(nv, lv)
            else:
                assert nv == lv, f"键 {k} 不匹配: {nv!r} vs {lv!r}"

    def test_to_dict_safe_to_mutate(self) -> None:
        """to_dict() 返回的 dict 可安全修改，不影响原 ReplanConfig。"""
        rc = _build_replan_config()
        d = rc.to_dict()
        original_k = rc.k_hit_total
        d["k_hit_total"] = 999
        assert rc.k_hit_total == original_k  # 原对象未受影响


class TestRunnerFactoryUnification:
    """B3: runner_factory.build_replan_cfg 与 MPCController 路径统一后行为保留。

    build_replan_cfg 现经 ReplanConfig.from_mpc_config().to_dict() 构建，
    产出 51-key dict（旧路径 42 键 + 9 个 do_replan .get() 默认等价键）。
    本测试验证旧 42 键值完全不变，9 个新键值与 do_replan 默认等价。
    """

    def test_build_replan_cfg_legacy_values_preserved(self) -> None:
        """旧 build_replan_cfg 的 42 个键值在统一后完全不变。"""
        env = _build_env()
        robot_limits = build_robot_limits(env, _CFG)
        solver = build_solver()
        d_hat = np.array([0.0, -1.0, -0.5])
        d_hat /= float(np.linalg.norm(d_hat))
        v_hit_desired = 1.8 * d_hat

        d = build_replan_cfg(env, robot_limits, solver, d_hat, v_hit_desired, _CFG)

        # 旧路径硬编码值（runner_factory.build_replan_cfg 原始 42 键）
        assert d["dt"] == 0.005
        assert d["total_horizon"] == 200
        assert d["fixed_horizon"] == 60
        assert d["replan_interval"] == 20
        assert d["max_iter_per_plan"] == 3
        assert d["first_plan_iters"] == 5
        assert d["near_plan_iters"] == 2
        assert d["near_threshold"] == 80
        assert d["R"] == 0.0001
        assert d["Q_p_scale_far"] == 5.0
        assert d["Q_v_scale_far"] == 3.0
        assert d["Q_p_scale_near"] == 8.0
        assert d["Q_v_scale_near"] == 120.0
        assert d["normal_weight"] == 500000.0
        assert d["racket_speed"] == 1.8  # config.target_hit_speed 默认值
        assert d["max_tcp_speed"] == 1.0  # config.max_tcp_speed 默认值
        assert d["ablation_mode"] == "full"
        assert d["is_position_mode"] is True
        assert d["use_backswing"] is False
        assert d["use_r_decay"] is False
        assert d["r_decay_ratio"] == 0.3
        assert d["fix_joint5_angle"] is None
        assert d["backswing_offset"] == 0.0
        assert d["backswing_ratio"] == 0.3
        assert d["normal_flip"] is False
        assert d["workspace_radius"] == 0.90
        assert d["hit_shift"] == 0.0
        assert d["follow_through_length"] == 0.0
        assert d["time_perturb_s"] == 0.0
        assert d["space_perturb_m"] == 0.0
        assert d["perturb_alpha_min"] == 0.0
        assert d["k_hit_total"] == 200
        assert d["smooth_far"] == {"Q_qdot_mult": 1.0, "Q_qddot_mult": 1.0, "Q_du_mult": 1.0}
        assert d["smooth_mid"] == {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 2.0}
        assert d["smooth_near"] == {"Q_qdot_mult": 2.0, "Q_qddot_mult": 2.0, "Q_du_mult": 3.0}
        np.testing.assert_array_almost_equal(d["shoulder_pos"], np.array([-0.1, -0.22693, 1.302645]))
        np.testing.assert_array_almost_equal(d["d_hat"], d_hat)
        np.testing.assert_array_almost_equal(d["v_hit_desired"], v_hit_desired)
        np.testing.assert_array_almost_equal(d["v_hit_at_contact"], v_hit_desired)
        assert d["robot_limits"] is robot_limits
        assert d["solver"] is solver

    def test_build_replan_cfg_new_keys_behavior_preserving(self) -> None:
        """统一后新增的 9 个键值与 do_replan .get() 默认等价（行为不变）。"""
        env = _build_env()
        robot_limits = build_robot_limits(env, _CFG)
        solver = build_solver()
        d_hat = np.array([0.0, 1.0, 0.0])
        v_hit_desired = 1.8 * d_hat

        d = build_replan_cfg(env, robot_limits, solver, d_hat, v_hit_desired, _CFG)

        # do_replan 用 .get("Q_p_base", None) — 显式 None 等价于键缺失
        assert d["Q_p_base"] is None
        assert d["Q_v_base"] is None
        # do_replan 用 .get("Q_qdot_base", 0.0) — 显式 0.0 等价于键缺失
        assert d["Q_qdot_base"] == 0.0
        assert d["Q_qddot_base"] == 0.0
        assert d["Q_du_base"] == 0.0
        # do_replan 用 .get("far_threshold", 50) — 与 MPCConfig 默认一致
        assert d["far_threshold"] == 50
        # from_mpc_config 默认 d_follow = d_hat（与 do_replan cfg.get("d_follow", cfg["d_hat"]) 等价）
        np.testing.assert_array_almost_equal(d["d_follow"], d_hat)
        # follow_through_steps / follow_through_v_terminal 不被 do_replan 引用，值无行为影响
        assert d["follow_through_steps"] == 0

    def test_build_replan_cfg_returns_full_key_set(self) -> None:
        """build_replan_cfg 产出与 ReplanConfig.to_dict() 相同的 51-key 集合。"""
        env = _build_env()
        robot_limits = build_robot_limits(env, _CFG)
        solver = build_solver()
        d_hat = np.array([0.0, 1.0, 0.0])
        v_hit_desired = 1.8 * d_hat

        d = build_replan_cfg(env, robot_limits, solver, d_hat, v_hit_desired, _CFG)

        rc = ReplanConfig.from_mpc_config(
            _build_config(),
            robot_limits=robot_limits,
            solver=solver,
            d_hat=d_hat,
            v_hit_desired=v_hit_desired,
        )
        assert set(d.keys()) == set(rc.to_dict().keys())
