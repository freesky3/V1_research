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
   - 当时未新增 `seed` 字段，也不使用 `np.random.default_rng(...)`。

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
- 当时 RNG/seed 扫描：无命中
- 旧兼容依赖扫描：无命中

验证后已清理 `src/` 和 `tests/` 下的 `__pycache__`。

## 提交注意

用户要求提交 git commit。提交时建议包含本轮迁移相关的源码、测试和 docs，但排除 `.gitignore`，除非用户明确要求提交该文件。

推荐 commit message：

```text
Migrate peripheral diagnostics into workflows
```

## 2026-06-25 Frames-sorted 方向选择性迁移交接

本轮完成的是旧 `frames_sorted` 中“ensemble 是否具有方向选择性”的主科学结果迁移。没有原样搬旧脚本和大图，而是把数据流改成：

```text
simulate 生成 8 方向、多 trial drifting-grating bundle
-> analyze 读取 bundle 并按方向聚合 trial response
-> Louvain 使用 trial-resolved steady traces
-> direction_tuning 输出 ensemble 方向选择性 summary/table/figure
```

主要代码：

- `src/v1_research/workflows/simulate.py`
  - 新增 `TrialScheduleConfig`、`TrialSchedule`、`build_trial_schedule(...)`。
  - `run_grating_simulation(...)` 的 batch 语义改为 trial。
  - 新增保存 `trial_direction_indices.npy`、`trial_orientation_angles.npy`、`trial_phase_offsets.npy`。
  - `simulation_health` 仍检查 trial-resolved dynamics，画图时传入 `trial_orientation_angles`。
- `src/v1_research/analysis/pipeline.py`
  - `AnalysisInputs` 新增可选 `trial_responses` 和 `trial_direction_indices`。
  - `load_analysis_inputs_from_simulation(...)` 有 trial metadata 时把 trial response 聚合为 direction response。
  - OSI/preferred direction/direction tuning 使用按方向平均后的 steady response；Louvain 优先使用 trial-resolved steady traces。
- `src/v1_research/analysis/direction_tuning.py`
  - 新增 `DirectionTuningConfig` 和 `summarize_direction_tuning(...)`。
  - 只统计非零 ensemble label。
  - modulation index 使用 `(max - min) / (max + min)`，默认阈值 `0.2`。
- `src/v1_research/analysis/artifacts.py`、`src/v1_research/workflows/analysis_figures.py`、`src/v1_research/workflows/analyze.py`、`src/v1_research/workflows/summarize.py`
  - 写 `analysis/direction_tuning.json`。
  - 写 `tables/ensemble_direction_tuning.csv`。
  - `inspection.save_plots=true` 时写 `figures/ensemble_direction_tuning.png`。
  - direction tuning 标量进入 manifest summary 和 summarize 输出。

文档：

- 新增 `docs/direction_tuning.md`，作为当前正式逻辑说明。
- 已同步更新 `docs/analysis.md`、`docs/diagnostics.md`、`docs/sweeps.md`、`docs/workflows.md`。

测试：

- 新增 `tests/test_analysis_direction_tuning.py`。
- 更新 `tests/test_workflows_simulate.py`、`tests/test_workflows_analyze.py`、`tests/test_workflows_summarize.py`。

遇到的坑和处理：

1. Windows 下并行跑多个 pytest 命令时会抢同一个 `.pytest` basetemp，出现清理目录权限错误。
   - 处理：最终验证顺序运行，不并行跑多个 pytest。
2. `compileall` 会重新生成 `src/` 和 `tests/` 下的 `__pycache__`。
   - 处理：验证后只清理 `src` 和 `tests` 内确认过路径的 `__pycache__`。
3. 用户要求主程序统一设置随机种子。
   - 处理：trial schedule 使用全局 `np.random.permutation/uniform`，当时没有新增局部 RNG、`seed` 字段或 `np.random.default_rng(...)`。
4. 旧 `frames_sorted` 有很多 plot/diagnostic 逻辑。
   - 处理：本轮只迁移主统计量，不搬 sorted-trial trace、variance 图、single-neuron diagnostics 或 random-blocks。
5. `.gitignore` 在本轮之前已有未提交修改，且 `.superpowers/` 是未跟踪目录。
   - 处理：提交时不要包含 `.gitignore` 和 `.superpowers/`，除非用户另行明确要求。

后续建议：

- 默认继续用 `jax-rk4` 作为 GPU 主性能路径；只有遇到明确 adaptive/steady-state stop 需求时再实现最小 `diffrax` backend。
- 若继续迁移旧项目，优先考虑 DG/OU all-cell paired analysis 或 BCM 深度诊断；不要恢复 Hydra/RootConfig 兼容层。

## 2026-06-26 全局 seed 控制交接

本轮已新增命令级全局 seed 控制：

- `src/v1_research/seed.py` 提供 `set_global_seed(...)`，设置 Python `random`、NumPy、`PYTHONHASHSEED`，并在安装 Torch 时设置 Torch/CUDA/cuDNN deterministic 开关。
- `TrainingWorkflowConfig`、`SimulationWorkflowConfig`、`AnalysisWorkflowConfig`、`FullWorkflowConfig` 和 `SweepConfig` 都有顶层 `seed: int | None`。
- 单独 workflow 在入口调用 `set_global_seed(cfg.seed)`；`full` 和 `sweep` 先设置根 seed，再用 `preserve_global_seed()` 防止子 workflow 或 grid point 重新设 seed，因此整次命令消耗一条连续随机流。
- 示例 YAML 已加 `seed: null`。临时覆盖用 `-o seed=123`，没有单独的 `--seed` 选项。
- 新增 `tests/test_seed_control.py` 覆盖 helper、config override、simulate 可复现、full 连续流和 sweep 连续流。
