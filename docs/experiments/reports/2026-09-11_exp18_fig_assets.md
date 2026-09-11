# exp18: 图资产收集（实时计时 + 4 类轨迹 NPZ）

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
| 计时埋点 | sync: `REPLAN step=N … t=XXms`（DEBUG，--log-level）；async: `ASYNC_PLAN done … t=XXms`（INFO） |
| NPZ 扫描 | 4 类目标 × 逐 seed，要求 episode 时长 ≥1.0s（滤除早退） |
| 原始日志 | `raw/timing_logs/{sync,async}_seedNN.log` |

## 结果

### 1. 重规划耗时（fig7 数据源，按迭代数分解）

| 重规划类型 | 迭代数 | 均值 | 中位 | p95 | 次数 |
|------------|--------|------|------|-----|------|
| **sync 首次规划** | 30 | 926.5 ms | — | — | 少量 |
| **sync 稳态（far）** | 0（纯 JT） | 3.5 ms | — | — | 绝大多数 |
| **sync 稳态（near）** | 5 | 24.9 ms | — | — | 少量 |
| sync 稳态整体 | — | 9.2 ms | 4 ms | 33 ms | 235 |
| sync 全部 | — | 96.9 ms | 4 ms | 927 ms | 291 |
| async 全部 | — | 7.9 ms | 6 ms | 11 ms | 94 |
| async 中 iters=5 | 5 | 56.5 ms | 52 ms | 75 ms | 4 |

> 重规划间隔 30 步 = 150 ms（5ms × 30），sync 稳态 ≤42ms（max）占预算 ≤28%。

### 2. 轨迹 NPZ 资产（manifest.json）

| 资产 | 条件 | seed | 结果 | 用途 |
|------|------|------|------|------|
| `a_hit_clean.npz` | 基线 9 m/s | 1 | 命中 | fig3 走廊示意 + fig4 基准 + 视频 |
| `b_hit_space_perturb.npz` | 空间扰动 0.1 m | 1 | 命中 | fig4 对照 + fig8 走廊激活 |
| `c_miss_combined_perturb.npz` | t=50ms + s=0.2m | 3 | miss | fig8 失败模式 + 视频 |
| `d_hit_noise_kf.npz` | 噪声 σp=0.02 + KF | 3 | 命中 | fig8 感知面板备选 |

## 关键数字

| 指标 | 值 |
|------|-----|
| 实时性瓶颈 | 首次规划 926 ms（30 iters），是稳态的 ~265 倍 |
| 稳态余量 | 稳态 max 42 ms / 预算 150 ms → 余量 ≥72% |
| async vs sync（求解耗时） | 中位 6 ms vs 4 ms——**求解耗时无实质差异** |
| NPZ 命中率 | 4/4 类目标一次扫到（c/d 各需 3 个 seed；d 的 seed 1-2 为 0.2s 早退被过滤） |

## 结论

### 数据观察（Agent 生成）

1. **同步模式实时性瓶颈在首次规划**：30 iters 首次规划 926ms，而稳态（far 0 iters 3.5ms / near 5 iters 24.9ms）全部 ≤42ms——稳态装进 150ms 重规划预算余量充足
2. **异步不改变求解耗时**：0 iters 阶段 sync 3.5ms vs async 5.8ms；5 iters 阶段 24.9ms vs 56.5ms——两模式求解成本同量级，异步价值在非阻塞而非加速
3. **NPZ 四类资产齐备且可复现**：manifest 记录 seed 与条件，fig3/4/8 与补充视频素材已解锁
4. **早退现象被显式过滤**：噪声 σp=0.02 下 seed 1-2 出现 0.2s 早退（空采样防御路径），不计入图资产

### 分析与决策

<!-- 人工填写，格式：
1. 核心发现 1
2. 核心发现 2
3. 下一步行动
-->
