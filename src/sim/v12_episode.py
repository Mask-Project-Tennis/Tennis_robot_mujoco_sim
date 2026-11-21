"""V12 单次回合的稳定入口（facade）。

历史原因，``run_v12_episode`` 的实现曾以约 778 行的单体函数存放在本模块。
B2 拆分后，真正的阶段实现位于 :mod:`src.sim.episode` 包：

- :mod:`src.sim.episode.preparation`：配置分辨率、机器人/场景事实、发球、
  击打点/IK 过滤、候选窗口/Tube、球速扰动、环境重置；
- :mod:`src.sim.episode.assembly`：MPC 配置/控制器、感知/安全/执行组件、
  runner 与轨迹记录 hook 的组装；
- :mod:`src.sim.episode.post_hit`：击打后 20 步保持；
- :mod:`src.sim.episode.results`：指标汇总、``V12EpisodeResult`` 构造、落盘；
- :mod:`src.sim.episode.execution`：``run_v12_episode_impl`` 五段式编排。

本模块只保留：

1. ``run_v12_episode`` 的**精确旧签名**（含所有 ``UNSET`` 覆盖参数）；
2. 旧调用方/测试可见的名字（含私有 helper）的原路径重导出；
3. 调用时从**本模块**读取 ``build_v12_robot`` / ``MPCController`` /
   ``SimComponent`` 并注入 impl —— 这样旧测试对
   ``src.sim.v12_episode.<name>`` 的 monkeypatch 仍然生效。

警告：
    不要在本模块重新实现任何阶段逻辑；所有新增行为都应落到 ``episode``
    包的对应阶段模块，否则本 facade 会重新长成第二个单体函数。
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from src.ilqt import episode_runner as _episode_runner_module
from src.ilqt.components.sim_component import SimComponent
from src.ilqt.mpc_controller import MPCController
from src.sim.episode import assembly as _assembly_module
from src.sim.episode import hooks as _hooks_module
from src.sim.episode import post_hit as _post_hit_module
from src.sim.episode import preparation as _preparation_module
from src.sim.episode import results as _results_module
from src.sim.episode.execution import run_v12_episode_impl
from src.sim.episode.types import V12EpisodeResult
from src.sim.episode_runtime import UNSET, UnsetType
from src.sim.v12_config import ResolvedV12Config
from src.sim.v12_robot_adapter import build_v12_robot

logger = logging.getLogger(__name__)

__all__ = ["V12EpisodeResult", "run_v12_episode"]

# ---------------------------------------------------------------------------
# 旧路径兼容重导出（保持对象同一性；测试会直接 import 这些私有名）
# ---------------------------------------------------------------------------

EpisodeRunner = _episode_runner_module.EpisodeRunner

# preparation 阶段
_DEFAULT_HIT_DIRECTION = _preparation_module._DEFAULT_HIT_DIRECTION
_effective_backend = _preparation_module._effective_backend
_model_shoulder_pos = _preparation_module._model_shoulder_pos
_mpc_workspace_center_pos = _preparation_module._mpc_workspace_center_pos
_normalize_hit_direction = _preparation_module._normalize_hit_direction
_predict_effective_backend = _preparation_module._predict_effective_backend
_resolve_hit_direction = _preparation_module._resolve_hit_direction
_resolve_serve_box_center = _preparation_module._resolve_serve_box_center
_resolve_serve_box_halfsize = _preparation_module._resolve_serve_box_halfsize
_resolve_shoulder_pos = _preparation_module._resolve_shoulder_pos
_select_follow_direction = _preparation_module._select_follow_direction
_set_rm65_left_arm = _preparation_module._set_rm65_left_arm
_shoulder_body_name = _preparation_module._shoulder_body_name
_build_tube_config = _preparation_module._build_tube_config

# assembly 阶段
_build_mpc_config = _assembly_module._build_mpc_config
_build_obs_gate = _assembly_module._build_obs_gate
_racket_radius_from_bundle = _assembly_module._racket_radius_from_bundle
_resolve_body_ids = _assembly_module._resolve_body_ids
_resolve_geom_ids = _assembly_module._resolve_geom_ids
_resolve_geom_pairs = _assembly_module._resolve_geom_pairs

# post_hit 阶段
_POST_HIT_ALLOWED_STOP_REASONS = _post_hit_module._POST_HIT_ALLOWED_STOP_REASONS
_POST_HIT_STEPS = _post_hit_module._POST_HIT_STEPS
_apply_post_hit_phase = _post_hit_module._apply_post_hit_phase
_run_post_hit_hold = _post_hit_module._run_post_hit_hold
_seed_ball_history = _hooks_module._seed_ball_history

# results 阶段
_apply_limits_overrides = _results_module._apply_limits_overrides
_explicit_runtime_options = _results_module._explicit_runtime_options
_model_dependency_manifest = _results_module._model_dependency_manifest
_r1d_artifact_metadata = _results_module._r1d_artifact_metadata
_real_config_snapshot = _results_module._real_config_snapshot
_resolve_failure_reason = _results_module._resolve_failure_reason
_runtime_option_overrides = _results_module._runtime_option_overrides
_scene_to_jsonable = _results_module._scene_to_jsonable
_serialize_hit_window = _results_module._serialize_hit_window
_sha256_file = _results_module._sha256_file


def run_v12_episode(
    config: ResolvedV12Config,
    *,
    dump_trajectory: Path | None = None,
    first_plan_flight: bool | UnsetType = UNSET,
    always_collision: bool | UnsetType = UNSET,
    ball_speed_perturb_pct: float | UnsetType = UNSET,
    fixed_direction: bool | UnsetType = UNSET,
    hit_direction: tuple[float, float, float] | None | UnsetType = UNSET,
    post_exec_hooks: tuple[Callable[[object], None], ...] = (),
    limits_config: Path | None | UnsetType = UNSET,
    max_tcp: float | None | UnsetType = UNSET,
    terminal_exempt_steps: int | None | UnsetType = UNSET,
    dq_max_fraction: float | None | UnsetType = UNSET,
) -> V12EpisodeResult:
    """运行一次 V12 共享闭环回合。

    流程与旧 ``scripts/rm65_mpc_v12.py`` 主流程逐段对应：初始化（bundle）→
    原发球（同 seed 同 RNG 顺序）→ 击打点/IK 可达过滤 → 候选窗口/Tube →
    MPC 与组件组装 → EpisodeRunner 执行 → 击打后 20 步保持 → 指标汇总。

    Args:
        config: 解析后的唯一配置来源；``config.runtime_overrides`` 里的已保存
            覆盖会在未显式传参时被继承（F01）。
        dump_trajectory: 轨迹 npz 落盘路径（None 不记录/不落盘）。
        first_plan_flight: 首个规划期间是否让球飞行
            （旧 ``--no-freeze-first-plan``）；``UNSET`` 继承已保存值/默认。
        always_collision: 击球前球-拍碰撞常开（批量实验用）；``UNSET`` 继承。
        ball_speed_perturb_pct: 球速耦合扰动百分比（旧 CLI 同名参数）；
            ``UNSET`` 继承。
        fixed_direction: 随挥方向是否用配置固定方向（旧 ``--fixed-direction``）；
            ``UNSET`` 继承。
        hit_direction: 固定击打方向覆盖（runtime ``hit_direction``）。优先级为
            本参数 > ``config.hitting_config["hit_direction"]`` > 内置默认
            （见 :func:`_resolve_hit_direction`）；``None`` 表示本参数不覆盖
            （回落到配置/默认），非法向量报错而不是静默回退。
        post_exec_hooks: 额外 post_exec hook（脚本层计时/诊断用）。
        limits_config: RM65 真机限位配置（旧 ``--limits-config``）；``None``
            清除已保存文件层，``UNSET`` 用已保存内容恢复（不要求原文件存在）。
        max_tcp: TCP 速度上限覆盖（0 表示不限）；``None`` 清除已保存覆盖。
        terminal_exempt_steps: 终段豁免步数覆盖；``None`` 清除已保存覆盖。
        dq_max_fraction: 关节位置增量上限分数覆盖；``None`` 清除已保存覆盖。

    Returns:
        :class:`V12EpisodeResult`：轨迹、击球事实与指标。

    Raises:
        TypeError: 传入的配置不是 :class:`ResolvedV12Config`。
        ValueError: 机器人/配置组合不受支持，或固定击打方向非法。
    """
    # 关键：在调用时读取本模块属性，而不是把实现模块的 import 绑定死。
    # tests/test_v12_runtime_restore.py 直接 monkeypatch 这三个名字。
    return run_v12_episode_impl(
        config,
        dump_trajectory=dump_trajectory,
        first_plan_flight=first_plan_flight,
        always_collision=always_collision,
        ball_speed_perturb_pct=ball_speed_perturb_pct,
        fixed_direction=fixed_direction,
        hit_direction=hit_direction,
        post_exec_hooks=post_exec_hooks,
        limits_config=limits_config,
        max_tcp=max_tcp,
        terminal_exempt_steps=terminal_exempt_steps,
        dq_max_fraction=dq_max_fraction,
        robot_builder=build_v12_robot,
        mpc_controller_cls=MPCController,
        sim_component_cls=SimComponent,
    )
