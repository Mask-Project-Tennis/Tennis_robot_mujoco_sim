# exp18: 图资产收集（实时计时 + 停顿对比 + 4 类轨迹 NPZ）

> **日期**: 2026-09-11
> **编号**: exp18
> **数据目录**: [experiment_data/exp18_fig_assets/](../../../experiment_data/exp18_fig_assets/)
> **状态**: ✅ 完成
> **背景**: 冲刺补跑 Phase 1（前台快速任务，产物先行先用）。旧 fig 数据源（exp1-12 raw）
> 已随系统重装丢失，本实验重建 fig3/4/7/8 的全部数据依赖
> **产物**: `timing.json`（fig7）、`manifest.json` + `raw/*.npz`（fig3/4/8 + 补充视频）

## 目的

<!-- 人工填写 -->

## 参数

| 参数 | 值 |
|------|-----|
| 脚本 | `scripts/exp/collect_fig_assets.py` |
| 计时 | 30 sync episodes + 10 async episodes（--async-replan），9 m/s，seed 1-30/1-10 |
| 求解耗时埋点 | sync: `REPLAN step=N … t=XXms`（DEBUG，--log-level）；async: `ASYNC_PLAN done … t=XXms`（INFO） |
| **停顿埋点** | `--dump-step-timing`：post_exec_hooks 间隔 = 主循环每步墙钟（超 5ms 控制周期计一次） |
| NPZ 扫描 | 4 类目标 × 逐 seed，要求 episode 时长 ≥1.0s（滤除早退） |
| 测量条件 | **机器空闲时测量**（避免 16-worker 批量实验的 CPU 争用污染计时） |

## 结果

### 1. 重规划求解耗时（按迭代数分解）

| 重规划类型 | 迭代数 | 样本 | 均值 | 说明 |
|------------|--------|------|------|------|
| sync 首次规划 | 30 | 28 | **919.8 ms** | 启动阶段（runner.run 之前），需提前量/离线预热 |
| sync 稳态 far | 0（纯 JT） | 200 | **3.4 ms** | 绝大多数 |
| sync 稳态 near | 5 | 63 | **24.8 ms** | 少量 |
| sync 稳态整体 | — | 235 | 中位 4ms / p95 33ms / max 41ms | 预算 150ms（30 步×5ms） |
| async 全部 | — | 91 | 中位 6ms / p95 49ms / max 63ms | 后台线程求解 |
| async 中 iters=5 | 5 | 5 | 56.0 ms | 后台线程求解偏慢（CPU 争用） |

### 2. 主循环停顿对比（fig7 第二序列，关键）

| 模式 | episodes | 总步数 | **超 5ms 周期步数** | p50 | p95 | p99 max | **max** |
|------|----------|--------|---------------------|-----|-----|---------|---------|
| sync | 28 | 6790 | **68（1.0%）** | 0.25 ms | 0.57 ms | 19.5 ms | **41.9 ms** |
| async | 10 | 2406 | **0（0%）** | 0.38 ms | 0.82 ms | 2.19 ms | **2.23 ms** |

> sync 的阻塞来自 `SyncReplanMode.submit` 内联求解（近场 5 iters ≈25ms）；async 的
> `submit` 仅入队，主循环每步均 ≤2.23ms（不超过控制周期）。

### 3. 轨迹 NPZ 资产（manifest.json）

| 资产 | 条件 | seed | 结果 | 用途 |
|------|------|------|------|------|
| `a_hit_clean.npz` | 基线 9 m/s | 1 | 命中 | fig3 走廊示意 + fig4 基准 + 视频 |
| `b_hit_space_perturb.npz` | 空间扰动 0.1 m | 1 | 命中 | fig4 对照 + fig8 走廊激活 |
| `c_miss_combined_perturb.npz` | t=50ms + s=0.2m | 3 | miss | fig8 失败模式 + 视频 |
| `d_hit_noise_kf.npz` | 噪声 σp=0.02 + KF | 3 | 命中 | fig8 感知面板备选 |

## 关键数字

| 指标 | 值 |
|------|-----|
| 实时性瓶颈 | 首次规划 919.8 ms（30 iters），是稳态 far 的 ~270 倍 |
| 稳态余量 | 稳态 max 41 ms / 预算 150 ms → 余量 ≥73% |
| **停顿对比（核心）** | sync 1.0% 的步超周期（max 41.9ms）vs async **0%**（max 2.23ms） |
| 求解耗时对比 | 0 iters: sync 3.4ms vs async 6.3ms；5 iters: 24.8ms vs 56.0ms——**同量级**，异步价值在非阻塞 |
| NPZ 命中率 | 4/4 类目标扫到（c/d 各需 3 seed；d 的 seed 1-2 为 0.2s 早退被过滤） |

## 结论

### 数据观察（Agent 生成）

1. **同步模式实时性瓶颈在首次规划**：30 iters 首次规划 919.8ms（启动阶段，不在控制循环内），稳态 far/near 分别 3.4/24.8ms，全部 ≤41ms——装进 150ms 重规划预算余量 ≥73%
2. **异步的价值被直接量化**：两种模式的**求解耗时同量级**（0 iters: 3.4 vs 6.3ms），但主循环停顿差异显著——sync 有 1.0% 的步超过 5ms 控制周期（最深 41.9ms），async 为 **0%（max 2.23ms）**
3. **异步后台求解在 near 阶段偏慢**（5 iters: 56.0ms vs sync 24.8ms），原因推断为后台线程与主循环的 CPU 争用；但因非阻塞，不影响控制循环
4. **NPZ 四类资产齐备**（manifest 记录 seed 与条件），fig3/4/8 与补充视频素材已解锁

### 分析与决策

<!-- 人工填写，格式：
1. 核心发现 1
2. 核心发现 2
3. 下一步行动
-->
