#!/usr/bin/env bash
# ============================================================================
# sprint_supplement.sh —— ICRA 冲刺补跑总控（一次性拉起, 结果先行先用）
#
# 结构（讨论确认的顺序 A）:
#   Phase 1（前台, ~3-5 分钟, 完成后立即可用于出图/写稿）:
#     collect_fig_assets.py —— 30ep 重规划计时 + 4 类 NPZ 图资产扫描
#   Phase 2（tmux 后台 detach, 本脚本随即返回, 与出图写稿互不阻塞）:
#     exp17d_mechanism（四档消融补齐 600 runs）→ exp17e_perturb7（4000 runs）
#     每个实验完成后写 experiment_data/<exp>/_.COMPLETE（或 _.FAILED）
#
# 幂等性: batch_runner --resume 跳过已完成 (config, seed);
#         Phase 2 重入时已 COMPLETE 的实验直接跳过。整脚本可安全重跑。
#
# 教训: 无 LD_LIBRARY_PATH 时子进程加载 libmujoco.so 失败（9/9 晚 gate 曾踩）,
#       本脚本显式定位并注入（site-packages 探测, 不 import mujoco）。
#
# 用法:
#   bash scripts/exp/sprint_supplement.sh          # Phase 1 + 拉起 Phase 2
#   SPRINT_TMUX=1 bash scripts/exp/sprint_supplement.sh   # 仅 Phase 2（tmux 内）
# ============================================================================
set -u
set -o pipefail
PROJECT="/data/fxy/research/Tennis_robot_mujoco_sim"
PY="/home/fxy/.local/miniforge3/envs/mujoco_tennis/bin/python"
export OMP_NUM_THREADS=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$PROJECT"

LOG_DIR="$PROJECT/logs"
STATE_DIR="$PROJECT/experiment_data/_driver_state"
mkdir -p "$LOG_DIR" "$STATE_DIR"
TS="$(date +%m%d-%H%M)"
LOG="$LOG_DIR/sprint-supplement-$TS.log"

# ----------------------------------------------------------- mujoco 库路径
MUJOCO_LIB="$("$PY" -c 'import site, pathlib; p = pathlib.Path(site.getsitepackages()[0]) / "mujoco"; print(p if any(p.glob("libmujoco.so*")) else "")')"
if [ -n "$MUJOCO_LIB" ]; then
  export LD_LIBRARY_PATH="$MUJOCO_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  echo "==> LD_LIBRARY_PATH 注入: $MUJOCO_LIB" | tee -a "$LOG"
fi

# ---------------------------------------------------------------- 工具函数
run_batch() {  # run_batch <exp_name>
  local exp="$1"
  if [ -f "experiment_data/$exp/_.COMPLETE" ]; then
    echo "==> $exp 已完成, 跳过（断点续跑）" | tee -a "$LOG"
    return 0
  fi
  echo "════════ [$(date +%H:%M:%S)] $exp 开始 ════════" | tee -a "$LOG"
  if "$PY" scripts/exp/batch_runner.py "$exp" --workers 16 2>&1 | tee -a "$LOG"; then
    touch "experiment_data/$exp/_.COMPLETE"
    echo "==> $exp COMPLETE $(date +%H:%M:%S)" | tee -a "$LOG"
  else
    touch "experiment_data/$exp/_.FAILED"
    echo "!!! $exp 失败（继续下一个, 详见 $LOG）" | tee -a "$LOG"
  fi
}

# ---------------------------------------------------------------- Phase 2
if [ "${SPRINT_TMUX:-0}" = "1" ]; then
  echo "==> Phase 2 启动 $(date)" | tee -a "$LOG"
  run_batch exp17d_mechanism
  run_batch exp17e_perturb7
  run_batch exp17f_mechanism_perturb   # 机制归因补测（标称四档发现 softmin 主导后追加）
  run_batch exp17g_spatial_power       # 强化版: s=0.2 角点高 seeds（走廊空间鲁棒性显著性）
  echo "════════ sprint 补跑全部完成 $(date) ════════" | tee -a "$LOG"
  exit 0
fi

# ---------------------------------------------------------------- Phase 1
echo "==> sprint_supplement 启动 $(date), 日志: $LOG" | tee -a "$LOG"
echo "════════ Phase 1 资产收集（前台）════════" | tee -a "$LOG"
if "$PY" scripts/exp/collect_fig_assets.py 2>&1 | tee -a "$LOG"; then
  echo "==> Phase 1 完成: experiment_data/exp18_fig_assets/{timing.json, manifest.json, raw/*.npz}" | tee -a "$LOG"
else
  echo "!!! Phase 1 失败, Phase 2 仍将拉起（详见 $LOG）" | tee -a "$LOG"
fi

# ---------------------------------------------------------- Phase 2 拉起
echo "════════ Phase 2 后台批量（tmux: sprint）════════" | tee -a "$LOG"
if tmux has-session -t sprint 2>/dev/null; then
  echo "==> tmux sprint 会话已存在, 不再重复拉起" | tee -a "$LOG"
else
  tmux new-session -d -s sprint "SPRINT_TMUX=1 bash $0 2>&1 | tee -a '$LOG'"
  echo "==> Phase 2 已拉起（tmux -t sprint）, 本脚本返回; 进度看 $LOG" | tee -a "$LOG"
fi
