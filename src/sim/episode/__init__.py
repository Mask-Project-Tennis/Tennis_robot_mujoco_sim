"""V12 回合阶段拆分包（B2）。

本包在导入时**只**暴露共享类型 :class:`V12EpisodeResult`；阶段实现按文件
放置，需要时由调用方直接导入子模块：

- :mod:`~src.sim.episode.preparation`：``prepare_episode`` 返回
  :class:`EpisodePreparation`，持有 bundle / execution / planning / RNG；
- :mod:`~src.sim.episode.assembly`：``assemble_episode`` 返回
  :class:`EpisodeAssembly`，持有 MPC / 组件 / runner / recorder / hooks；
- :mod:`~src.sim.episode.post_hit`：击打后保持段；
- :mod:`~src.sim.episode.results`：指标与 ``V12EpisodeResult`` 构造；
- :mod:`~src.sim.episode.execution`：``run_v12_episode_impl`` 五段式编排。

导入本包**不会**导入阶段实现、构建 MuJoCo 模型、加载旧实现模块
（``src.sim.v12_config`` / ``src.sim.v12_episode``）或写文件；本模块顶层只
依赖不启动环境的共享类型定义。这是 ``tests/test_refactor_shared_types.py``
的导入契约，也是 ``src.sim.episode`` 能被轻量消费者安全导入的前提。
"""

from __future__ import annotations

from src.sim.episode.types import V12EpisodeResult

__all__ = ["V12EpisodeResult"]
