# archive/ — 已归档脚本

> 归档日期: 2026-06-22（首批）+ 2026-07-09（HittingCost 组合化重构）| 活跃版本: `scripts/rm65_mpc_v12.py`

## 结构

- `root/` — V6-V10 + tube 变体（原 `scripts/` 根目录，9 个）
- `sim/` — realtime v4/v5 + V8/V9 消融变体 + HittingCost 子类脚本 + 旧批量/实时脚本（原 `scripts/sim/`，12 个）
- `exp/` — 仅引用上述脚本的旧实验脚本 + 配套 batch（原 `scripts/exp/`，38 个）

## 为什么归档

V11 已重构为 V12 薄壳（29 行），V6-V10 行为已被 exp13 数据归档。
2026-07-09 追加归档：HittingCost 组合化重构（CompositeCost 替代继承式 HittingCost），
含 HittingCost 子类的脚本（ConstrainedHittingCost / JointLimitHittingCost /
WorkspaceHittingCost）与旧批量/实时脚本一并归档，保留供历史追溯
（`git log --follow`），不再维护。

## 如何重跑归档的实验

```bash
# 归档的 batch 脚本已修复内部路径，可直接运行：
python scripts/archive/exp/run_exp1_batch.py
python scripts/archive/exp/run_exp7_batch.py
```
