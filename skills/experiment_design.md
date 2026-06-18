# Skill: 实验设计与数据管理（experiment_design）

## 目的
围绕 MPC+iLQR+Tube 网球击打框架，系统化设计、执行和管理论文实验数据。
调用时机：运行批量实验、收集论文数据、分析实验结果、创建新实验时。

## 实验设计索引

实验设计文档（目的/假设/参数矩阵）存放在 `docs/experiments/design/`：
- 索引: `docs/experiments/README.md`
- 设计文档: `docs/experiments/design/expN_<name>.md`
- 实验报告: `docs/experiments/reports/YYYY-MM-DD_expN_<name>.md`
- 可执行参数: `experiment_data/expN/config.yaml`
- 结果数据: `experiment_data/expN/results.csv`

## 核心脚本映射

| 脚本 | 用途 |
|------|------|
| `scripts/rm65_mpc_v12.py` | ★ 当前活跃仿真主脚本（V12, EpisodeRunner 架构） |
| `scripts/rm65_mpc_tube_constraint.py` | 离线仿真（exp1-7 使用） |
| `scripts/exp/_run_expN_*.py` | 实验包装脚本（monkey-patch 约束） |
| `scripts/exp/run_expN_batch.py` | 批量运行器 |
| `scripts/extract/extract_expN_results.py` | 日志→CSV 提取 |

## 数据存储规范

### 目录结构
```
experiment_data/
├── expN_<name>/
│   ├── config.yaml          # 实验参数
│   ├── results.csv          # 汇总表
│   └── raw/                 # 原始日志
│       ├── speed9_seed0_tube_true.log
│       └── ...
```

### CSV 通用列

```csv
seed,ball_speed,hit,pos_error,min_distance,max_qdot_ratio,max_tcp_speed,mpc_steps,total_time_s,hit_time_error_ms,tube_ready_ms,ball_near_ms
```

| 列名 | 单位 | 说明 |
|------|------|------|
| seed | — | 随机种子 |
| ball_speed | m/s | 球到达击打点时的水平速度 |
| hit | bool | 命中判定（pos_error < 0.153） |
| pos_error | m | 末端位置误差 |
| min_distance | m | 全程最小球拍-球距离 |
| max_qdot_ratio | × | 最大关节速度 / 额定限速 |
| max_tcp_speed | m/s | 最大 TCP 线速度 |
| mpc_steps | — | MPC 执行步数 |
| total_time_s | s | 总墙钟时间 |

实验特有列在每组实验的 `config.yaml` 中定义。

### NPZ 原始数据格式（可选）

```python
np.savez_compressed(
    f"raw/seed{seed}_ballspeed{ball_speed}.npz",
    X_history=np.array(X_history),       # (N+1, 12)
    U_history=np.array(U_history),       # (N, 6)
    ball_pos_history=np.array(ball_pos), # (M, 3)
    distances=np.array(distances_history),
    step_times=np.array(step_times),
    replan_times=np.array(replan_times),
)
```

---

## 新建实验工作流

### 三层架构

| 层 | 文件模板 | 职责 |
|----|---------|------|
| 包装 | `scripts/exp/_run_expN_*.py` | monkey-patch 约束 → 构建 sys.argv → 调主脚本 |
| 运行 | `scripts/exp/run_expN_batch.py` | 遍历参数矩阵 → subprocess → UTF-8 日志 |
| 提取 | `scripts/extract/extract_expN_results.py` | regex 解析日志 → results.csv |

### 参考实现（复制即改）

| 用途 | 参考文件 |
|------|---------|
| 豁免约束包装 | `scripts/exp/_run_exp1_v3_exempt.py` |
| 严格约束包装 | `scripts/exp/_run_exp2_v3_strict.py` |
| 批量运行器 | `scripts/exp/run_exp2_v3_batch.py` |
| 提取脚本（离线） | `scripts/extract/extract_exp2_v3_results.py` |
| 提取脚本（实时 V12） | `scripts/extract/extract_exp1_results.py` |
| 默认约束参数 | `configs/default.yaml` |

### 包装脚本模板

**关键**：monkey-patch 必须在 `import main_mod` **之前**。

```python
"""实验N 辅助包装脚本：<description>。"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ball_speed = sys.argv[1]
seed = sys.argv[2]
use_tube = sys.argv[3]

# === Monkey-patch（必须在 import main_mod 之前）===
from src.ilqt.robot_limits import RobotLimits
_orig_from_config = RobotLimits.from_config

@classmethod
def _patched(cls, config, dt, ctrlrange):
    config = dict(config)
    config["forward_pass_margin"] = <值>
    config["qdot_scale"] = <值>
    config["forward_pass_q_tol_deg"] = <值>
    config["max_tcp_speed"] = <值>
    return _orig_from_config(config, dt, ctrlrange)

RobotLimits.from_config = _patched

sys.argv = [
    "rm65_mpc_v12.py",
    "--serve-box",
    "--ball-speed", ball_speed,
    "--seed", seed,
    "--no-plot",
]

import scripts.rm65_mpc_v12 as main_mod  # noqa: E402
main_mod.main()
```

### 批量运行器模板

```python
"""批量运行 expN 实验（多进程并行）。"""
import argparse, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "experiment_data" / "<experiment_id>" / "raw"
WRAPPER = PROJECT_ROOT / "scripts" / "exp" / "_run_expN_<name>.py"
PYTHON_EXE = str(Path(sys.executable))

SPEEDS = [8, 9, 10]
SEEDS = list(range(15))
TUBE_MODES = ["true", "false"]

def run_one(args):
    speed, seed, tube = args
    tag = f"speed{speed}_seed{seed}_tube_{tube}"
    log_path = RAW_DIR / f"{tag}.log"
    if log_path.exists():          # 断点续传
        return tag, True
    cmd = [PYTHON_EXE, str(WRAPPER), str(speed), str(seed), tube]
    try:
        result = subprocess.run(
            cmd, cwd=str(PROJECT_ROOT), capture_output=True,
            timeout=180, encoding="utf-8",
            env={**os.environ, "PYTHONUTF8": "1"},
        )
        content = result.stderr if result.stderr.strip() else result.stdout
        log_path.write_text(content, encoding="utf-8")
        return tag, True
    except subprocess.TimeoutExpired:
        return tag, False
    except Exception as e:
        log_path.write_text(f"ERROR: {e}", encoding="utf-8")
        return tag, False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = [(s, d, t) for s in SPEEDS for t in TUBE_MODES for d in SEEDS]
    total = len(tasks)
    print(f"<experiment_id>: {len(SPEEDS)} 球速 × {len(TUBE_MODES)} tube × {len(SEEDS)} seeds = {total} runs")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok, failed = 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, t): t for t in tasks}
        for i, f in enumerate(as_completed(futures), 1):
            tag, success = f.result()
            if success: ok += 1
            else: failed += 1
            if i % 20 == 0 or i == total:
                elapsed = time.time() - t0
                print(f"[{i}/{total}] ok={ok} fail={failed} elapsed={elapsed:.0f}s")
    print(f"完成: {ok} ok, {failed} failed, {(time.time()-t0)/60:.1f}min")

if __name__ == "__main__":
    main()
```

### config.yaml 模板

```yaml
experiment: <experiment_id>
description: "<description>"
purpose: "<purpose>"
constraint_type: <exempt|strict|custom>
seeds: <N>
ball_speeds: [<speeds>]
tube_modes: [<modes>]
script_type: <v12|offline|realtime>
constraints:
  forward_pass_margin: <值>
  qdot_scale: <值>
  forward_pass_q_tol_deg: <值>
  max_tcp_speed: <值>
total_runs: <计算值>
```

---

## 常见易错点

| # | 易错点 | 处理 |
|---|--------|------|
| 1 | monkey-patch 不生效 | patch 必须在 `import main_mod` 之前 |
| 2 | 离线脚本输出到 stderr | `result.stderr if result.stderr.strip() else result.stdout` |
| 3 | 并行 worker >4 导致 segfault | 默认 4 worker，`--no-plot` 关闭渲染 |
| 4 | 日志编码乱码 | `PYTHONUTF8=1` + `encoding="utf-8"` |
| 5 | 单次超时 | `timeout=180`，超时记为失败 |
| 6 | 实时脚本无 `__RESULT__` | 用实时专用提取脚本（匹配 `__RESULT__` JSON 行） |
| 7 | conda activate 失败 | 直接用 `python`，不 activate |
| 8 | tmux session 重名 | 先 `tmux kill-session -t <ID> 2>/dev/null` |

## 数据完整性检查

每次实验运行后检查：
- [ ] 每个 (ball_speed, seed, condition) 组合都有对应的日志文件
- [ ] pos_error 为有效数值（非 NaN、非 Inf）
- [ ] 无异常退出（日志末尾有完整输出）
- [ ] CSV 行数 = 预期运行数

## Dispatch subagent

主 Agent 创建完上述脚本后，dispatch experiment-runner subagent，传入：
```
experiment_id: expN_<name>
data_dir: <绝对路径>/experiment_data/expN_<name>
raw_dir: <绝对路径>/experiment_data/expN_<name>/raw
batch_script: scripts/exp/run_expN_<name>_batch.py
extract_script: scripts/extract/extract_expN_<name>_results.py
workers: 4
```
