# Direction tuning 方向选择性分析

本文档说明当前 `v1_research` 中 ensemble 方向选择性分析的代码路径。它不是旧 `frames_sorted` 的逐文件迁移，而是把其中“ensemble 是否具有方向选择性”的科学核心放进新的 `simulate -> analyze` 数据流。

## 推荐阅读顺序

1. `src/v1_research/workflows/simulate.py`：看 `TrialScheduleConfig`、`build_trial_schedule(...)` 和 `run_grating_simulation(...)` 如何生成多 trial run bundle。
2. `src/v1_research/analysis/pipeline.py`：看 `load_analysis_inputs_from_simulation(...)` 如何读取 trial metadata，并把 trial response 聚合成按方向平均的 response。
3. `src/v1_research/analysis/direction_tuning.py`：看 `DirectionTuningConfig` 和 `summarize_direction_tuning(...)` 如何计算 ensemble direction tuning。
4. `src/v1_research/workflows/analyze.py` 与 `src/v1_research/analysis/artifacts.py`：看结果如何写入 run bundle 和 manifest summary。
5. `src/v1_research/workflows/analysis_figures.py`：看 `inspection.save_plots=true` 时的方向选择性图。

## simulate 生成 trial bundle

入口函数是 `run_grating_simulation(...)`。`SimulationWorkflowConfig` 中新增本地配置：

```python
TrialScheduleConfig(
    repeats_per_direction=4,
    shuffle=True,
    random_phase=True,
    phase_jitter=0.35,
)
```

`build_trial_schedule(...)` 接收 `DriftingGratingInput.orientation_angles`，为每个方向重复生成 trial。随机顺序和相位扰动使用全局 `np.random`，由主程序统一 seed 控制；这里不创建局部 RNG，也没有 `seed` 字段。

当前数据流：

```text
SimulationWorkflowConfig
-> DriftingGratingInput(...)
-> build_trial_schedule(...)
-> make_batched_drive_func(trial_orientation_angles, phase_offsets)
-> solve_rates(..., n_batch=n_trials)
-> arrays/*.npy
```

核心数组：

- `orientation_angles.npy`: 唯一方向，shape `(n_directions,)`。
- `trial_direction_indices.npy`: 每个 trial 属于哪个方向，shape `(n_trials,)`。
- `trial_orientation_angles.npy`: 每个 trial 的实际方向，shape `(n_trials,)`。
- `trial_phase_offsets.npy`: 每个 trial 的相位偏移，shape `(n_trials,)`。
- `excitatory_rates.npy`: steady rate，shape `(n_trials, n_exc)`。
- `excitatory_trajectory.npy`: 完整轨迹，shape `(n_time, n_trials, n_exc)`，仅在 `solver.store_trajectory=true` 时写。

`simulation_health.json` 仍然检查 trial-resolved dynamics。也就是说 health 的 batch 维度现在是 trial，而不是唯一方向。

## analyze 读取和聚合

入口函数是 `load_analysis_inputs_from_simulation(...)`。它优先读取 `excitatory_trajectory.npy`，没有 trajectory 时退化到 `excitatory_rates.npy` 的单时间点 response。

有 `trial_direction_indices.npy` 时，loader 会同时保留两份数据：

- `AnalysisInputs.trial_responses`: trial-resolved response，shape `(n_neurons, n_trials, n_time)`。
- `AnalysisInputs.responses`: 按方向平均后的 response，shape `(n_neurons, n_directions, n_time)`。

聚合函数是 `direction_average_trial_responses(...)`：

```text
trial_responses[:, trial_mask, :]
-> mean over repeated trials
-> responses[:, direction, :]
```

OSI、preferred direction 和 direction tuning 使用按方向平均后的 steady response。Louvain 在有 trial data 时优先使用 `trial_responses[:, :, steady_start:]` 展平成特征；没有 trial metadata 时保持旧的新项目行为，把 batch 当作方向。

## ensemble direction tuning

纯计算入口是 `summarize_direction_tuning(...)`，配置在 `DirectionTuningConfig`：

```python
DirectionTuningConfig(
    enabled=True,
    modulation_threshold=0.2,
)
```

输入是已筛选细胞的 community labels、按方向 steady mean response 和方向角。函数只统计非零 ensemble label；label `0` 表示未分类细胞，不计入 ensemble。

每个 ensemble 的方向曲线为：

```text
mean response of member neurons for each direction
```

主要字段：

- `preferred_direction_deg`: mean response 最大的方向。
- `modulation_index`: `(max - min) / (max + min)`。
- `direction_selective`: `modulation_index >= modulation_threshold`。
- `mean_rate_<deg>deg`: 每个方向的 ensemble mean response。

汇总字段：

- `direction_selective_ensembles`
- `direction_selective_fraction`
- `covered_direction_count`
- `min_ensemble_modulation`
- `mean_ensemble_modulation`

这里使用 steady-state mean，而不是旧 `frames_sorted` 中一些图默认使用的 max response。

## 输出位置

`analyze` 默认写核心结果：

```text
analysis/direction_tuning.json
tables/ensemble_direction_tuning.csv
```

当 `AnalysisInspectionConfig.save_plots=true` 时，额外写：

```text
figures/ensemble_direction_tuning.png
```

`run_analysis_workflow(...)` 会把 `direction_tuning.json` 中的标量汇总合并进 manifest summary。`workflows/summarize.py` 也会读取该 JSON，并展开成 `analysis.*` 字段，供 sweep CSV 直接使用。

## 和旧 frames_sorted 的边界

已迁移的是主科学问题：训练后分出的 ensemble 是否具有方向选择性。

没有迁移的内容：

- 大型 sorted-trial trace 图。
- trial variance 图。
- single-neuron diagnostics。
- `frames_sorted_blocks` 的 random-block 可视化路线。

后续如果需要这些诊断，应先明确新的科学问题，再把纯计算放入 `analysis/`，把调度和保存结果放入 `workflows/`。不要把旧脚本的 CLI、局部 seed、兼容 artifact 名称或大型 plotting pipeline 原样搬回新项目。
