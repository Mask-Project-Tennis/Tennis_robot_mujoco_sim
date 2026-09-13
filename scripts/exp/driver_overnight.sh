#!/usr/bin/env bash
# ============================================================================
# driver_overnight.sh —— 连夜重跑总控（在 tmux 会话内运行, agent 关闭不影响）
#
# 阶段:
#   P0 门禁    overnight_gate.py（复现性抽查, FAIL 立即停止不烧算力）
#   P1 主线    exp13 → exp15 → exp16 → exp14（batch_runner, 16 workers）
#   P2 改进    exp17a 噪声 → exp17b 扰动 → exp17c 观测频率
#   P3 收尾    汇总对照表 + git 提交数据（防再次丢失）
#
# 特性:
#   - 断点续跑: batch_runner --resume 跳过已完成 (config, seed),
#     整个 driver 可重复启动（重启后 tmux 里再跑一遍本脚本即可接着跑）
#   - 单实验失败不阻塞后续实验（记录进状态文件）
#   - 状态文件: experiment_data/_driver_state/{state.json, gate.json, summary.md}
#
# 用法:
#   bash scripts/exp/driver_overnight.sh          # 全流程
#   RUN_FROM=2 bash scripts/exp/driver_overnight.sh  # 跳过门禁, 从 P1 开始
# ============================================================================
set -u
set -o pipefail  # 管道中前一命令（gate/batch_runner）的失败不被 tee 吞掉
# 项目根目录按脚本位置解析；PY 可用环境变量覆盖（默认本机 conda 环境）
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${PY:-$HOME/.local/miniforge3/envs/mujoco_tennis/bin/python}"
export OMP_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$PROJECT"

STATE_DIR="$PROJECT/experiment_data/_driver_state"
LOG_DIR="$PROJECT/logs"
mkdir -p "$STATE_DIR" "$LOG_DIR"
TS="$(date +%m%d-%H%M)"
LOG="$LOG_DIR/overnight-$TS.log"
STATE="$STATE_DIR/state.json"

# ---------------------------------------------------------------- 工具函数
write_state() {  # write_state <phase> <status> <detail>
  "$PY" - "$1" "$2" "$3" "$STATE" <<'PYEOF'
import json, sys, time
from pathlib import Path
phase, status, detail, path = sys.argv[1:5]
p = Path(path)
state = json.loads(p.read_text()) if p.exists() else {"phases": {}}
state["phases"][phase] = {"status": status, "detail": detail,
                          "time": time.strftime("%m-%d %H:%M:%S")}
state["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
p.write_text(json.dumps(state, ensure_ascii=False, indent=2))
PYEOF
}

run_batch() {  # run_batch <exp_name>
  local exp="$1"
  echo "════════ [$(date +%H:%M:%S)] $exp 开始 ════════"
  if "$PY" scripts/exp/batch_runner.py "$exp" --workers 16 2>&1 | tee -a "$LOG"; then
    write_state "$exp" "done" "exit=0"
  else
    write_state "$exp" "failed" "exit!=0, 详见 $LOG"
    echo "!!! $exp 失败, 继续下一个实验"
  fi
}

echo "==> driver 启动 $(date), 日志: $LOG" | tee -a "$LOG"

# ---------------------------------------------------------------- P0 门禁
if [ "${RUN_FROM:-0}" -lt 1 ]; then
  echo "════════ P0 门禁 ════════" | tee -a "$LOG"
  if "$PY" scripts/exp/overnight_gate.py 2>&1 | tee -a "$LOG"; then
    write_state "P0_gate" "pass" "复现性抽查通过"
  else
    write_state "P0_gate" "fail" "复现不达标 —— 停止, 等人工分析 gate.json"
    echo "!!! 门禁失败, 停止全部后续实验（不烧算力）" | tee -a "$LOG"
    exit 1
  fi
fi

# ---------------------------------------------------------------- P1 主线
run_batch exp13_arch
run_batch exp15_speed_v2
run_batch exp16_limits_v2
run_batch exp14_pd_v2

# ---------------------------------------------------------------- P2 改进实验
run_batch exp17a_noise
run_batch exp17b_perturb
run_batch exp17c_obsfreq

# ---------------------------------------------------------------- P3 收尾
echo "════════ P3 汇总 + 归档 ════════" | tee -a "$LOG"
"$PY" scripts/exp/overnight_summary.py 2>&1 | tee -a "$LOG" \
  && write_state "P3_summary" "done" "summary.md 已生成" \
  || write_state "P3_summary" "failed" "汇总脚本异常"

if git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
  git add experiment_data/*/results.csv experiment_data/*/config.yaml \
          experiment_data/_driver_state/ \
          scripts/exp/batch_runner.py scripts/exp/overnight_experiments.py \
          scripts/exp/overnight_gate.py scripts/exp/overnight_summary.py \
          scripts/exp/driver_overnight.sh \
          scripts/rm65_mpc_v12.py src/ilqt/components/sim_perception.py \
          src/ilqt/episode_runner.py tests/test_batch_runner.py \
    && git commit -m "data(exp): 连夜重跑 exp13-17 —— 观测管线修复 + 200/100 seeds + V12 鲁棒性实验

- fix(ilqt): SimPerception 兼容 BallObservationGate(step 驱动), EpisodeRunner 透传 step
- fix(v12): 空采样序列 min_dist 防御（大噪声早退出）
- batch_runner: 16 workers 并行 + resume 断点续跑 + CSV 增量落盘
- exp13(V12 双模式 200s) exp15(9 球速 200s) exp16(限位 200s) exp14(PD 50s)
- exp17a/b/c: V12 首次噪声/扰动/观测频率鲁棒性网格" \
    && write_state "P3_git" "done" "数据与代码已提交" \
    || write_state "P3_git" "failed" "git 提交异常（可能无可提交变更）"
fi

write_state "driver" "done" "全部阶段结束, 见 summary.md"
echo "════════ driver 全部完成 $(date) ════════" | tee -a "$LOG"
