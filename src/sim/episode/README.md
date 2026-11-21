# episode — V12 闭环 episode 层

## 当前阶段（B2：阶段拆分）

`run_v12_episode` 原本是约 778 行的单体函数（`src/sim/v12_episode.py`
L703-1480）。B2 把它拆成下列阶段模块；旧路径只保留 facade：

| 模块 | 职责 | 稳定入口 |
|------|------|----------|
| `preparation.py` | 运行时覆盖解析、`build_v12_robot`、限位应用、机器人/场景事实、发球（RNG 唯一创建/消费）、击打点 + IK 余量过滤、候选窗口/Tube、球速扰动、环境重置与球初态 | `prepare_episode(config, *, 9 个覆盖参数, robot_builder=...)` → `EpisodePreparation` |
| `assembly.py` | MPCConfig/MPCController、感知/安全/执行组件、`EpisodeRunner`、轨迹记录 hook 组装；RM65/R1D 分支差异 | `assemble_episode(prep, *, mpc_controller_cls, sim_component_cls, dump_trajectory, post_exec_hooks)` → `EpisodeAssembly` |
| `hooks.py` | `TrajectoryRecorder` 构造与 `post_exec_hooks` 合并、球初态 history 播种 | `build_post_exec_hooks(...)`、`_seed_ball_history(...)` |
| `post_hit.py` | 击打后 20 步保持（stop_reason 门控 + 逐帧 append） | `run_post_hit(prep=, assembly=, run_metrics=)`、`_apply_post_hit_phase`、`_run_post_hit_hold` |
| `results.py` | 指标汇总、R1D 产物元数据、`V12EpisodeResult` 构造、dump_trajectory | `build_result(...)`、`build_unreachable_result(prep)` |
| `execution.py` | 五段式编排：prepare → assemble → run → post_hit → result | `run_v12_episode_impl(...)` |
| `types.py` | 共享结果类型（B1/P1 迁入，唯一类定义） | `V12EpisodeResult` |

## 状态所有权（单一所有者）

| 有状态对象 | 唯一所有者 | 说明 |
|------------|-----------|------|
| `bundle` / `execution` / `planning` | `EpisodePreparation` | 组装、运行、指标阶段只持有引用，不重建 |
| RNG（`np.random.default_rng`） | `EpisodePreparation` | **只创建一次**；准备阶段消费次数/顺序与旧实现一致，后续阶段不得新增采样 |
| `x0` / `p0_real` / `v0_real` | `EpisodePreparation` | 组装阶段写入 SimComponent histories 时使用副本 |
| `mpc` / `perception` / `safety` / `sim_component` / `runner` | `EpisodeAssembly` | `runner.run` 由编排层调用，阶段内不重复创建 |
| `TrajectoryRecorder` | `EpisodeAssembly.recorder` | 由 `hooks.build_post_exec_hooks` 创建 |
| `SimComponent.X_history` / `ball_*_history` | `post_hit` 阶段追加 | `assembly` 只做初始播种；`build_result` 只读并拷贝成数组 |
| 指标 dict 与最终结果 | `results.build_result` | 其余阶段不构造 `V12EpisodeResult` |

**释放/退出**：没有需要显式 `close()` 的资源；MuJoCo 环境随 `bundle`
生命周期回收，函数返回即结束。`run_v12_episode_impl` 是唯一退出点：
早退（`hit_info is None`）只经 `build_unreachable_result` 返回，不进入
`assemble`；正常路径在 `build_result` 返回后结束。模块级无缓存/无全局状态。

## 行数审查（contracts §1.7：>150 行函数 / >600 行文件）

| 位置 | 行数 | 职责边界 | 不再拆的理由（摘要） |
|------|------|----------|----------------------|
| `preparation.prepare_episode` | ~350 | 配置→机器人→RNG/发球→击打点/IK→阈值/方向→窗口/Tube→扰动→重置 的单一路径协议 | 共享局部状态 + 严格 RNG/调用顺序；抽段需可变上下文或 30+ 参数，顺序漂移会改 RNG 消费。纯逻辑已抽 helper |
| `results.build_result` | ~370 | 指标汇总 + 拍心轨迹重建 + metrics 字面量 + dump | 主体是 40+ 键、必须逐字保持的 schema 字面量；抽 helper 会变成 40 参数转发层且更易漂移。纯判定已抽 helper |
| `results._unreachable_result` | ~170 | 早退结果 + 独立 metrics 字面量镜像 | 与主路径字面量相邻便于 schema 对照；抽共享构造会掩盖两条口径独立 |
| `assembly.assemble_episode` | ~156 | RM65 / R1D 两条互斥组件构造分支 | 再抽会把 `prep` 十余字段当参数透传；同处对照两套安全/接触参数更利于审查 |
| `preparation.py` | 852 | 准备阶段 helper + 快照 dataclass + 上述协议函数 | 文件由多个职责单一 helper/dataclass 组成，不是单体函数 |
| `results.py` | 961 | 指标/artifact helper + 主/早退两条结果构造 | 与旧实现逐字对齐的 schema 集中地；后续若继续增长，可按 “artifact 元数据 / 指标” 拆文件（见下） |

后续可选拆分点（需先补 golden 测试，不在 B2 做）：

1. `preparation`：把 IK 余量搜索抽为 `_refine_hit_point_with_ik(...)`（输入
   `execution/p_hit/k_hit_total/init_q/limits/ball_positions_all`，输出新
   `k_hit/p_hit/v_ball_hit`）——纯计算、无 RNG，风险最低。
2. `preparation`：把候选窗口 + Tube 构造抽为 `_build_hit_window(prep 局部事实)`。
3. `results`：把 `_r1d_artifact_metadata` 及 R1D 产物子结构移到
   `episode/r1d_artifact.py`，把 `results.py` 压回指标主路径。
4. `results`：若未来 metrics schema 稳定，可定义 `EpisodeMetrics` dataclass
   一次性承载键集合；当前阶段为避免口径漂移仍保留字面量。

原则：不为行数制造转发层；任何拆分不得改变调用顺序、RNG 消费或 metrics schema。

## 旧路径 facade 契约（`src/sim/v12_episode.py`）

- 保留 `run_v12_episode` **逐字签名**（含 9 个 `UNSET` 覆盖参数）与
  `__all__ = ["V12EpisodeResult", "run_v12_episode"]`。
- 重导出旧调用方/测试可见的名字（含私有 helper），对象与阶段模块内**同一**。
- `run_v12_episode` 在**调用时**读取本模块 `build_v12_robot` /
  `MPCController` / `SimComponent` 并注入 `run_v12_episode_impl`，因此
  `tests/test_v12_runtime_restore.py` 的 monkeypatch 仍然生效。
- 禁止把阶段逻辑搬回 facade：`run_v12_episode` 函数体只有 docstring +
  一次 `return run_v12_episode_impl(...)`（由
  `tests/test_episode_decomposition.py` 的 AST 断言守住）。

## 顺序红线

1. 配置分辨率 → 构建 bundle → 应用限位 → 唯一 RNG → 重置/发球 → 击打点/IK →
   候选窗口 → 扰动 → 重置/球初态 → 组装 → `runner.run` → 击打后保持 → 指标/落盘。
2. 显式覆盖 > `config.runtime_overrides` > 默认；`False` / `0` 是有效值。
3. 早退 `failure_reason="ball_unreachable"` 与 `post_hit_skipped_reason` 不变。

## 允许依赖

- 标准库、`numpy`、`src.ilqt.*` 组件、`src.sim.episode_runtime`、
  `src.sim.configuration.types`、`src.sim.v12_robot_adapter`、`src.robot.constants`。
- 不依赖 `src.sim.v12_episode`（防循环）；`import src.sim.episode` **只**导出
  `V12EpisodeResult`，不急切导入任何阶段实现（`tests/test_refactor_shared_types.py`
  的干净子进程契约），也不构建模型、不启动仿真、不写文件。需要阶段 API 时
  直接 `from src.sim.episode.preparation import prepare_episode` 等。

## 测试

```bash
python -m pytest tests/test_episode_decomposition.py -q
python -m ruff check src/sim/episode src/sim/v12_episode.py tests/test_episode_decomposition.py
```


## 何时读

改 V12 回合阶段顺序、RNG 消费、配置应用、post-hit/hold 或结果字段时；
只调用回合时用 `from src.sim.v12_episode import run_v12_episode`，不必进入阶段模块。

## 最小用例

```python
from src.sim.configuration import resolve_v12_config
from src.sim.v12_episode import run_v12_episode

config = resolve_v12_config()
result = run_v12_episode(config)
print(result.hit_type, result.hit_step, len(result.X))
```
