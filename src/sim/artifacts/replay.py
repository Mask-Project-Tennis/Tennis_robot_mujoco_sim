# ruff: noqa: I001, PLC0414, BLE001
# D1 契约 §6 要求显式 `import X as X` 重导出；生成器保留原文件 typing
# 导入与分组风格。BLE001/TRY004/SIM102 为拆分前旧代码既有告警，
# 行为不变，仅在此保留（记入报告）。

"""产物回放（构建 bundle、按关节名重建状态、viewer 可视化）。

依赖图最上层：``... ← materialization ← replay``。MuJoCo / viewer 只在
函数内延迟导入，保证 ``import src.sim.artifacts`` 不触发重型依赖。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from src.sim.v12_robot_adapter import V12RobotBundle

from src.sim.artifacts.types import (
    EpisodeArtifacts as EpisodeArtifacts,
    EpisodeReplayResult as EpisodeReplayResult,
)
from src.sim.artifacts.validation import (
    _verify_and_resolve_sources as _verify_and_resolve_sources,
    _verify_loaded_model as _verify_loaded_model,
    load_episode_artifacts as load_episode_artifacts,
)
from src.sim.artifacts.materialization import (
    _build_replay_bundle_with_snapshot as _build_replay_bundle_with_snapshot,
    _replay_config as _replay_config,
)

_LOGGER = logging.getLogger(__name__)

def build_replay_bundle(output_dir: Path) -> V12RobotBundle:
    """读取产物并构建回放用机器人 bundle（不打开 viewer）。

    快照模型优先：``_resolved_model_path`` 选中模型快照时，先把快照 XML
    物化为临时 XML/profile 再构建（快照目录不含 mesh 资产），保证加载的
    几何来自快照而不是被改动/删除的外部模型（评审 R11）。只有快照模型
    缺失、回退到外部模型时才直接使用原配置。

    Raises:
        ValueError: 快照/外部来源哈希不一致、实际加载模型被篡改，或
            快照模型物化失败。
    """
    artifacts = load_episode_artifacts(output_dir)
    sources = _verify_and_resolve_sources(artifacts.payload, Path(output_dir))
    config = _replay_config(artifacts.payload, sources)
    model_path = _verify_loaded_model(artifacts.payload, sources)
    return _build_replay_bundle_with_snapshot(
        artifacts, sources, config, model_path
    )
def _replay_frames(
    execution: Any,
    bundle: V12RobotBundle,
    artifacts: EpisodeArtifacts,
    joint_names: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """逐帧按名写入状态并 FK，返回 (ball_positions, racket_positions)。"""
    from src.sim.v12_robot_adapter import ReplayState  # 延迟导入

    n = len(joint_names)
    X = artifacts.X
    ball_pos = artifacts.ball_pos
    ball_vel = artifacts.ball_vel
    frames = int(X.shape[0])
    racket_positions = np.zeros((frames, 3), dtype=float)
    for frame in range(frames):
        state = ReplayState(
            joint_positions={
                name: float(X[frame, index]) for index, name in enumerate(joint_names)
            },
            joint_velocities={
                name: float(X[frame, n + index])
                for index, name in enumerate(joint_names)
            },
            ball_position=np.asarray(ball_pos[frame], dtype=float),
            ball_velocity=np.asarray(ball_vel[frame], dtype=float),
        )
        bundle.set_replay_state(state)
        execution.update_kinematics()
        racket_positions[frame] = np.asarray(execution.get_ee_pos(), dtype=float)
    return np.asarray(ball_pos, dtype=float), racket_positions
def _visualize_replay(
    execution: Any,
    bundle: V12RobotBundle,
    artifacts: EpisodeArtifacts,
    joint_names: list[str],
    speed: float,
    dt: float,
) -> None:
    """用 passive viewer 按 ``speed`` 倍速循环播放已保存的轨迹。"""
    import mujoco
    import mujoco.viewer

    X = artifacts.X
    frames = int(X.shape[0])
    if frames <= 0:
        return
    playback = float(speed) if speed and speed > 0 else 1.0
    with mujoco.viewer.launch_passive(
        execution.model, execution.data
    ) as viewer:
        viewer.cam.distance = 2.8
        viewer.cam.elevation = -15.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        start = time.perf_counter()
        frame = 0
        while viewer.is_running():
            elapsed = time.perf_counter() - start
            idx = int(elapsed * playback / max(float(dt), 1e-9))
            if idx >= frames:
                start = time.perf_counter()
                idx = 0
            if idx != frame:
                frame = idx
                from src.sim.v12_robot_adapter import ReplayState

                n = len(joint_names)
                state = ReplayState(
                    joint_positions={
                        name: float(X[frame, i])
                        for i, name in enumerate(joint_names)
                    },
                    joint_velocities={
                        name: float(X[frame, n + i])
                        for i, name in enumerate(joint_names)
                    },
                    ball_position=np.asarray(artifacts.ball_pos[frame], dtype=float),
                    ball_velocity=np.asarray(artifacts.ball_vel[frame], dtype=float),
                )
                bundle.set_replay_state(state)
                execution.update_kinematics()
                mujoco.mj_forward(execution.model, execution.data)
                viewer.sync()
            time.sleep(min(float(dt) / playback, 0.02))
def replay_episode_artifacts(
    output_dir: Path,
    *,
    speed: float = 1.0,
    viewer: bool = True,
    bundle: V12RobotBundle | None = None,
) -> EpisodeReplayResult:
    """只读回放一个已保存的 v2 回合产物。

    回放按关节名写入状态（不依赖 qpos 布局），默认 1.0 倍速；慢放必须
    显式传入 ``speed < 1``。回放**不会**改写 JSON/NPZ、也不会重新判定
    命中或合成反弹。

    Args:
        output_dir: 产物目录。
        speed: 播放倍速（默认 1.0；仅 viewer=True 时生效）。
        viewer: 是否打开被动查看器（False 时纯计算）。
        bundle: 可选的已构建 bundle（避免重复建模）。

    Returns:
        :class:`EpisodeReplayResult`（含 payload、球/拍位置历史与倍速）。

    Raises:
        ValueError: 快照/外部来源哈希不一致，或缺 ``input_sha256`` 校验项。
    """
    directory = Path(output_dir)
    artifacts = load_episode_artifacts(directory)
    payload = artifacts.payload
    sources = _verify_and_resolve_sources(payload, directory)
    config = _replay_config(payload, sources)
    model_path = _verify_loaded_model(payload, sources)
    if bundle is None:
        bundle = _build_replay_bundle_with_snapshot(
            artifacts, sources, config, model_path
        )
    execution = bundle.execution
    joint_names = [str(name) for name in payload["joint_names"]]
    ball_positions, racket_positions = _replay_frames(
        execution, bundle, artifacts, joint_names
    )
    viewer_opened = False
    if viewer:
        try:
            _visualize_replay(
                execution,
                bundle,
                artifacts,
                joint_names,
                speed=speed,
                dt=float(payload.get("dt", 0.005)),
            )
            viewer_opened = True
        except Exception as exc:  # pragma: no cover - 无显示环境时降级
            _LOGGER.warning("回放 viewer 不可用（降级为纯计算）：%s", exc)
    return EpisodeReplayResult(
        payload=payload,
        ball_positions=ball_positions,
        racket_positions=racket_positions,
        speed=float(speed),
        viewer_opened=viewer_opened,
        directory=directory,
    )
