# Handoff 给下一个 AI

## 当前目标

目标项目：`D:\skywalker\sjtu\大三下课程\Research\MouseV1\V1_simulation`

本轮完成的是“外围诊断脚本迁移”：不把旧 `train_simulation_analysis/scripts/` 整体搬入新项目，只把有复用价值的计算能力放入 `src/v1_research`，把可配置的行为接到现有 workflow/CLI/run bundle。

## 建议使用的 skills

- `refactor-v1-research-code`：继续改 V1 simulation 项目时先读。
- `document-v1-research-code`：完成一个逻辑子系统后更新 docs。
- `superpowers:test-driven-development`：新增行为或修 bug 时先写失败测试。
- `superpowers:systematic-debugging`：遇到测试失败先定位根因。
- `superpowers:verification-before-completion`：最终答复、提交或声称完成前跑验证。

## 已完成工作

新增训练诊断计算：

- `src/v1_research/learning/diagnostics.py`
- 主要入口：`active_rate_stats`、`plastic_weight_stats`、`row_sum_pressure`、`cap_fraction`、`weight_delta_stats`、`theta_stats`、`sample_tracked_weights`、`record_tracked_weights`
- 这些函数只接 `ModelState`、`BCMState`、`RateBatch` 或数组，不接 root config，不读写磁盘。

训练 workflow 接入 inspection：

- `src/v1_research/workflows/train.py`
- 新增 `TrainingInspectionConfig`
- `TrainingWorkflowConfig.inspection` 默认关闭。
- `inspection.enabled=True` 时写 `tables/training_diagnostics.csv`。
- 有 tracked edges 时写 `tables/tracked_weights.csv`。
- `inspection.save_plots=True` 才写 `figures/training_overview.png` 和 `figures/tracked_weights.png`。
- `inspection.save_per_batch_arrays=True` 才写 per-probe rates/weights `.npy`。

新增分析诊断能力：

- `src/v1_research/analysis/overlap.py`：坐标匹配、contingency、best label match、ARI、overlap surrogate significance。
- `src/v1_research/analysis/temporal.py`：按 tail fraction 或 end time 切窗口并复用 `run_analysis(...)`。

新增只读汇总入口：

- `src/v1_research/workflows/summarize.py`
- CLI：`v1-simulation summarize --run <run_dir> [--output path]`
- 只读新 run bundle，不兼容旧 artifact 命名。

增强 sweep 输出：

- `src/v1_research/workflows/analyze.py` 会把 analysis metrics summary 中的标量放入 workflow summary。
- `src/v1_research/workflows/sweep.py` 会把目标 workflow 的 summary 展平到 CSV 的 `summary.*` 字段。
- 旧 DG/Louvain/window/spatial sweep 脚本的常用需求应通过现有 `sweep` 参数网格表达。

文档：

- 更新了 `docs/learning.md`、`docs/analysis.md`、`docs/workflows.md`、`docs/sweeps.md`。
- 新增 `docs/diagnostics.md`，按当前逻辑说明本轮迁移后的诊断入口。
- 当前这份 `docs/handoff.md` 用于交接，不是项目正式逻辑文档。

## 遇到的坑与处理

1. `compileall` 会在 `src/` 和 `tests/` 下生成 `__pycache__`。
   - 每次验证后都清理：只删除 `src/` 和 `tests/` 下名为 `__pycache__` 的目录，不碰 `.venv`。

2. `.gitignore` 在本轮开始前已有未提交修改，并且包含 `docs/` ignore。
   - 不要随手提交 `.gitignore`。
   - 提交 docs 时用 `git add -f docs/...`。

3. `cap_fraction(...)` 初版曾对 cap 数组长度不匹配使用 `np.resize`。
   - 已改为直接 `ValueError`，这是数学上容易静默出错的检查，符合本项目取舍。

4. tracked weights 初版的 `sample_index` 使用候选边编号，抽样子集时会跳号。
   - 已改为对抽中的 tracked edges 重新编号 `0..n-1`，CSV 和图例更清楚。

5. 不能引入局部 RNG。
   - `sample_tracked_weights(...)` 使用全局 `np.random.choice`。
   - `overlap_significance(...)` 使用全局 `np.random.shuffle`。
   - 不新增 `seed` 字段、不使用 `np.random.default_rng(...)`。

## 最近验证结果

在目标项目根目录运行：

```powershell
uv run pytest -q
uv run python -m compileall src tests
rg "default_rng|SeedSequence|seed\s*:|seed\s*=|np\.random\.Generator" src tests configs
rg "v1_simulation|RootConfig|NetworkState|run_config|aE_all|frames_sorted" src tests
```

结果：

- `pytest`：`71 passed, 1 skipped, 4 warnings`
- `compileall`：通过
- RNG/seed 扫描：无命中
- 旧兼容依赖扫描：无命中

验证后已清理 `src/` 和 `tests/` 下的 `__pycache__`。

## 提交注意

用户要求提交 git commit。提交时建议包含本轮迁移相关的源码、测试和 docs，但排除 `.gitignore`，除非用户明确要求提交该文件。

推荐 commit message：

```text
Migrate peripheral diagnostics into workflows
```
