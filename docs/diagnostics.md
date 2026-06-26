# 训练、仿真与分析诊断报告

本文说明当前训练、drifting-grating 仿真和 analyze 阶段的诊断代码如何组织，以及 `inspection.enabled=true` 时 run bundle 会写出哪些健康报告、表格和机制图。它面向需要判断训练或单次仿真是否过于静默、过度活跃、活动过度集中，是否接近 firing-rate cap，BCM row-sum cap 压力是否过大，或 OSI/Louvain 分析为什么没有得到清晰 ensemble 的研究者。

## 推荐阅读顺序

1. `src/v1_research/workflows/train.py`：从 `TrainingWorkflowConfig`、`TrainingInspectionConfig` 和 `run_training(...)` 看训练入口与落盘逻辑。
2. `src/v1_research/learning/diagnostics.py`：看纯诊断函数、`TrainingHealthConfig` 和 `evaluate_training_health(...)`。
3. `src/v1_research/workflows/simulation_health.py`：看 simulation 专用的 `SimulationHealthConfig`、`compute_simulation_health(...)` 和 `save_simulation_figures(...)`。
4. `src/v1_research/workflows/simulate.py`：看 `SimulationInspectionConfig` 如何把健康报告接入单次 grating simulation。
5. `src/v1_research/analysis/diagnostics.py`：看 analyze 的 selection funnel、graph health 和 unclassified 原因。
6. `src/v1_research/analysis/robustness.py`：看 analyze 的窗口与 Louvain 参数 robustness 复跑。
7. `src/v1_research/workflows/analyze.py`：看 `AnalysisInspectionConfig` 如何接入单次 analysis workflow。
8. `src/v1_research/workflows/analysis_figures.py`：看 analysis 结果图谱与 robustness 图如何生成。
9. `src/v1_research/learning/bcm.py`：看 BCM state、theta、row-sum cap 与 plastic weight 更新机制。
10. `src/v1_research/workflows/summarize.py`：看 run bundle 如何被压缩成 sweep 友好的 summary。
11. `docs/learning.md`、`docs/analysis.md` 与 `docs/workflows.md`：补充理解训练规则、分析链路和 CLI workflow 边界。

## 入口与配置

训练 workflow 的入口函数是 `run_training(cfg: TrainingWorkflowConfig, *, show_progress=True)`。本地诊断配置挂在 `TrainingWorkflowConfig.inspection`，定义于 `src/v1_research/workflows/train.py`：

```python
TrainingInspectionConfig(
    enabled=False,
    probe_every=1,
    health=TrainingHealthConfig(),
    tracked_weight_count=0,
    save_plots=False,
    save_per_batch_arrays=False,
    active_rate_threshold=1.0,
    steady_state=TrainingSteadyStateConfig(enabled=False),
)
```

健康阈值由 `src/v1_research/learning/diagnostics.py` 中的 `TrainingHealthConfig` 定义，默认偏向“早提醒”：

```python
TrainingHealthConfig(
    min_active_neuron_fraction=0.05,
    max_active_neuron_fraction=0.95,
    max_top1_activity_fraction=0.35,
    max_top5_activity_fraction=0.75,
    max_row_sum_cap_fraction=0.05,
    max_row_sum_cap_ratio=0.95,
    near_rate_cap_ratio=0.95,
    max_near_rate_cap_fraction=0.05,
)
```

这些字段可以写进 YAML，也可以用 CLI override 修改，例如：

```powershell
uv run v1-simulation train --config configs/train_smoke.yaml `
  -o inspection.enabled=true `
  -o inspection.health.min_active_neuron_fraction=0.2
```

`inspection.enabled=false` 时不写训练诊断表和健康报告。`inspection.save_plots=false` 是默认值，用于避免 sweep 或批量实验自动生成大量图片。需要看机制图时显式打开 `inspection.save_plots=true`。

`show_progress=True` 时，训练会在每个 inspected batch 后向 `stderr` 打印一到两行实时健康状态。CLI 默认等价于 `--progress`，会显示这些状态；`--no-progress` 会关闭实时状态打印，但不会影响训练结束后的 CSV、JSON 和图片产物。CLI 的最终 run directory 仍写到 `stdout`，因此脚本可以继续用 stdout 捕获 run 路径。

## 训练诊断数据流

每个 inspected batch 的数据流如下：

```text
run_training(...)
-> build natural-image batch drive
-> solve_rates(...) 得到 RateBatch(exc, inh, external)
-> apply_learning_rule(...)
-> _training_diagnostic_row(...)
   -> active_rate_stats(...)
   -> extended_active_rate_stats(...)
   -> plastic_weight_stats(...)
   -> weight_delta_stats(previous_model, model)
   -> theta_stats(...) / theta_distribution_stats(...)
   -> row_sum_pressure(...)
   -> extended_plastic_weight_stats(...)
   -> bcm_signal_stats(...)
-> if inspection.steady_state.enabled:
   -> keep full E/I trajectory for this probe
   -> compute tail stability metrics
   -> optionally save population and sampled-neuron traces
-> if show_progress:
   -> evaluate_training_health([current_row], ...)
   -> print live status to stderr
-> evaluate_training_health(all_rows, ...)
-> write run bundle
```

主要数组形状：

- `RateBatch.exc`: `(n_batch, n_exc)`，每个样本的 excitatory rate。
- `RateBatch.inh`: `(n_batch, n_inh)`，当 `n_inh=0` 时是空 population。
- `RateBatch.external`: `(n_batch, n_input)`，来自当前 batch 的 L4 external drive。
- `W_EE`: `(n_exc, n_exc)`，BCM 管理的 `E <- E` plastic block。
- `W_IE`: `(n_inh, n_exc)`，BCM 管理的 `I <- E` plastic block。
- `theta_exc`: `(n_exc,)`，`theta_inh`: `(n_inh,)`。
- `RateResult.exc_trajectory`: `(n_time, n_batch, n_exc)`，只在 `inspection.steady_state.enabled=true` 的 probe batch 上强制保留用于诊断。
- `RateResult.inh_trajectory`: `(n_time, n_batch, n_inh)`。

空 population 的扩展诊断字段使用 `None`，并且健康判定会跳过这些字段。这样 `inhibitory_fraction=0.0` 的 smoke 配置不会因为不存在 inhibitory population 而误报。

## training_diagnostics.csv

`tables/training_diagnostics.csv` 每行对应一次 inspected batch。字段按含义分为几类：

- 基本 activity：`exc_mean`、`exc_median`、`exc_max`、`exc_active_fraction`，以及对应的 `inh_*` 字段。
- per-neuron activity：`exc_active_neuron_count`、`exc_active_neuron_fraction`、`exc_silent_neuron_fraction`，以及对应的 `inh_*` 字段。
- activity concentration：`exc_top1_activity_fraction`、`exc_top5_activity_fraction`，用于判断少数 neuron 是否吸收了大部分活动。
- rate cap 压力：`exc_near_rate_cap_fraction`、`inh_near_rate_cap_fraction`。阈值来自 `solver.transfer.rate_max * inspection.health.near_rate_cap_ratio`。
- external drive 分布：`external_mean`、`external_median`、`external_p05`、`external_p95`、`external_max`。
- plastic weights：`W_EE_*`、`W_IE_*`，包含非零数量、均值、中位数、最大值、p05/p95、row-sum mean/p95/max。
- connected delta sign：`W_EE_delta_positive_fraction`、`W_EE_delta_negative_fraction`、`W_EE_delta_zero_fraction`，以及 `W_IE_*` 对应字段。
- row-sum cap 压力：`row_sum_EE_cap_fraction`、`row_sum_EE_cap_max_ratio`、`row_sum_IE_cap_fraction`、`row_sum_IE_cap_max_ratio`。
- BCM theta：`theta_exc_mean`、`theta_exc_median`、`theta_exc_p05`、`theta_exc_p95`，以及 `theta_inh_*`。
- BCM signal：`bcm_exc_above_theta_fraction`、`bcm_exc_signal_mean`、`bcm_exc_signal_abs_mean`，以及 `bcm_inh_*`。
- trial 内稳态：`steady_exc_tail_mean`、`steady_exc_tail_variance`、`steady_exc_final_vs_tail_abs_mean`、`steady_exc_final_vs_tail_relative_mean`、`steady_exc_tail_step_p95_abs_change`、`steady_exc_tail_population_relative_drift`，以及 `steady_inh_*`。这些字段只有 `inspection.steady_state.enabled=true` 时出现，用于判断固定模拟时长末尾是否仍在漂移。

旧的 `active_rate_stats(...)`、`plastic_weight_stats(...)`、`theta_stats(...)` 仍保留，用于兼容已有轻量诊断；扩展函数会补充更适合训练机制解释的字段。

## 训练 steady-state 诊断

`inspection.steady_state.enabled=true` 时，训练不会改变求解时长或学习规则，只会在 inspected batch 上临时要求 solver 保留完整 trajectory。这个诊断适合回答“当前 `time[-1]` 是否足够接近稳态”，不是 early stop。

主要读数：

- `final_vs_tail_abs_mean`：最后一个时间点与 tail window 均值的平均绝对差。越小表示结尾越接近当前训练用于学习的 tail mean。
- `final_vs_tail_relative_mean`：上述差值除以 tail mean 的相对量。低 firing rate 时这个值会被小分母放大，需要结合绝对差一起看。
- `tail_step_p95_abs_change`：tail window 内相邻时间点变化的 95 分位。持续偏大通常说明还在漂移或振荡。
- `tail_population_relative_drift`：tail window 中 population mean 的线性漂移量除以 tail mean。接近 0 更像稳定尾段。

数组产物：

- `training_probe_XXXXXX_population_trace.npz`：包含 `time`、`exc_population_mean`、`inh_population_mean`。
- `training_probe_XXXXXX_sampled_neuron_traces.npz`：包含抽样神经元 index 和 trace。默认每个 population 一半取 tail mean 最高神经元，一半随机抽取剩余神经元。

图片产物：

- `training_steady_population.png`：多个 probe 的 E/I population mean firing-rate trace。
- `training_steady_sampled_neurons.png`：最后一个 probe 中抽样神经元的 trial 内 trace。

## 训练中的实时状态

实时状态打印与 `inspection.probe_every` 对齐：哪一个 batch 会写入 `training_diagnostics.csv`，就在哪一个 batch 后打印状态。默认 CLI `--progress` 会显示，`--no-progress` 会静默。`full` 和 `sweep` 继续透传同一个 progress 开关。

状态行示例：

```text
[train] step=12 epoch=1 batch=12 health=warn warn_total=3 fail_total=0 exc_active=0.420 inh_active=- top1=0.180 top5=0.640 exc_rate_cap=0.000 inh_rate_cap=- row_EE_cap=0.510 row_IE_cap=- theta_exc=1.030 theta_inh=- bcm_exc_above=0.380 bcm_inh_above=- bcm_exc_signal=-0.024 bcm_inh_signal=- W_EE_delta+=0.410 W_EE_delta-=0.090 W_IE_delta+=- W_IE_delta-=-
```

如果当前 probe 触发健康事件，会额外打印一行事件摘要：

```text
[train] events: warn exc_top5_activity_fraction=0.820>0.750; fail exc_active_neuron_fraction=0.000>0.000
```

字段读取方式：

- `health` 是当前 probe 的 `ok|warn|fail`，不是整个训练最终状态。
- `warn_total` 和 `fail_total` 是截至当前 probe 的累计事件数量。
- `exc_active`、`inh_active` 是 per-neuron active fraction；空 population 显示为 `-`。
- `top1`、`top5` 描述 excitatory activity concentration。
- `exc_rate_cap`、`inh_rate_cap` 描述接近 firing-rate cap 的比例。
- `row_EE_cap`、`row_IE_cap` 是 row-sum/cap 最大比值。
- `theta_*`、`bcm_*` 和 `W_*_delta*` 用于快速判断 BCM signal 与权重更新方向。

实时状态只用于人工观察，不会触发 early stop，也不会改变训练完成语义。完整回顾仍以 `training_diagnostics.csv`、`training_health.json` 和机制图为准。

## 健康报告语义

`evaluate_training_health(...)` 只读取 diagnostic rows 和 `TrainingHealthConfig`，不读写磁盘。它会生成：

- `schema_version`
- `status`: `ok | warn | fail`
- `thresholds`: 本次使用的阈值快照
- `warning_count`、`failure_count`
- `first_warning_step`、`first_failure_step`
- `final_metrics`: 最后一条 inspected row 的数值字段
- `worst_metrics`: 每个指标在训练过程中的最差值
- `events`: 每个触发阈值的事件

健康状态只记录，不 raise，不改变训练完成语义。普通阈值越界是 `warn`，例如 top1/top5 activity concentration 太高、near-rate-cap fraction 太高、row-sum cap 压力过高。明显坏状态是 `fail`，目前包括有效 population 的 `active_neuron_fraction <= 0.0` 或 `>= 1.0`。

事件写入 `tables/training_health_events.csv`。每行包含 `step`、`severity`、`metric`、`value`、`threshold`、`rule` 和 `message`，方便 sweep 后筛选最早异常或最高频异常。

## Simulation 健康报告

单次 grating simulation 的入口函数仍是 `run_grating_simulation(cfg: SimulationWorkflowConfig)`。本地诊断配置挂在 `SimulationWorkflowConfig.inspection`，定义于 `src/v1_research/workflows/simulate.py`：

```python
SimulationInspectionConfig(
    enabled=True,
    save_plots=True,
    health=SimulationHealthConfig(),
)
```

`simulate` 默认开启完整诊断，前提是 `solver.store_trajectory=True`，因为健康报告直接从完整 E/I trajectory 计算。`sweep` 在构造 simulate config 时会默认覆盖为轻量模式：`inspection.enabled=false`、`inspection.save_plots=false`，且在没有显式要求完整诊断时设置 `solver.store_trajectory=false`。这样单次 `simulate` 面向检查，批量 `sweep` 面向吞吐。

仿真诊断数据流：

```text
run_grating_simulation(...)
-> DriftingGratingInput.make_batched_drive_func(orientation_angles)
-> solve_rates(...) 得到 RateResult
-> _sample_stimulus_trace(...)
-> compute_simulation_health(...)
   -> population activity metrics
   -> activity concentration metrics
   -> near-rate-cap metrics
   -> front/tail drift and step-to-step change
   -> stimulus/background distribution metrics
-> analysis/simulation_health.json
-> optional figures/simulate_*.png
```

主要数组形状：

- `RateResult.exc_trajectory`: `(n_time, n_trials, n_exc)`。
- `RateResult.inh_trajectory`: `(n_time, n_trials, n_inh)`。
- sampled stimulus trace: `(n_time, n_trials, n_input)`。
- `BackgroundTrace.exc`: `(n_time, n_trials, n_exc)`，`BackgroundTrace.inh`: `(n_time, n_trials, n_inh)`。

`compute_simulation_health(...)` 会生成 `analysis/simulation_health.json`，包含：

- `schema_version`
- `status`: `ok | warn | fail`
- `thresholds`: 本次使用的 `SimulationHealthConfig`
- `warning_count`、`failure_count`
- `metrics`: 展开的 activity、concentration、stability、stimulus/background 指标
- `events`: 触发阈值的健康事件

核心 metric 前缀为 `exc_*` 和 `inh_*`。活动指标包括 `active_fraction`、`silent_fraction`、`mean`、`median`、`p95`、`max`。集中度指标包括 `top1_activity_fraction` 和 `top5_activity_fraction`。若 `solver.transfer.rate_max` 存在，还会用 `rate_max * near_rate_cap_ratio` 计算 `near_rate_cap_fraction`。稳定性指标包括 `front_mean`、`back_mean`、`tail_mean`、`tail_variance`、`mean_drift`、`relative_mean_drift`、`step_mean_abs_change`、`step_p95_abs_change` 和 `step_max_abs_change`。

输入上下文指标使用 `stimulus_*`、`background_exc_*`、`background_inh_*` 前缀记录 mean、median、p05、p95、max。它们不替代 OSI/community 分析，只帮助解释仿真本身的输入强度与背景噪声范围。

## Run Bundle 产物

开启 `inspection.enabled=true` 后，训练 run 目录会额外包含：

```text
runs/train/<timestamp>/
  tables/
    training_diagnostics.csv
    training_health_events.csv
    tracked_weights.csv             # 有 tracked rows 时
  analysis/
    training_health.json
  arrays/
    training_probe_000001_exc_rates.npy
    training_probe_000001_inh_rates.npy
    training_probe_000001_weights.npy
    training_probe_000001_population_trace.npz      # steady_state.enabled=true 且 save_arrays=true
    training_probe_000001_sampled_neuron_traces.npz # steady_state.enabled=true 且 save_arrays=true
    ...
  figures/
    training_overview.png           # save_plots=True
    training_activity.png           # save_plots=True
    training_bcm.png                # save_plots=True
    training_plasticity.png         # save_plots=True
    training_row_sums.png           # save_plots=True
    training_steady_population.png  # steady_state.enabled=True 且 save_plots=True
    training_steady_sampled_neurons.png
    tracked_weights.png             # save_plots=True 且有 tracked rows
```

`manifest.json.outputs` 会记录 `training_health` 和 `training_health_events` 的相对路径。`manifest.json.summary` 与 `TrainingRun.summary` 会加入 sweep 友好的标量：

- `health_status`
- `health_warning_count`
- `health_failure_count`
- `first_health_warning_step`
- `first_health_failure_step`
- `final_exc_active_neuron_fraction`
- `final_inh_active_neuron_fraction`
- `final_exc_top1_activity_fraction`
- `final_exc_top5_activity_fraction`

`workflows/summarize.py` 会读取 `analysis/training_health.json`，并把健康状态、计数、首个事件 step 和 `final_metrics` 展开为 `training_health.*` 字段。这样 `summarize` 和 `sweep` 可以在不解析大 CSV 的情况下比较训练健康状态。

开启 `simulate.inspection.enabled=true` 后，simulation run 目录会额外包含：

```text
runs/simulate/<timestamp>/
  analysis/
    simulation_health.json
  figures/
    simulate_overview.png                 # save_plots=True
    simulate_orientation_heatmaps.png     # save_plots=True
    simulate_traces.png                   # save_plots=True
```

`manifest.json.outputs` 会记录 `simulation_health` 与三张核心图的相对路径。`manifest.json.summary` 与 `SimulationRun.summary` 会加入 sweep 友好的标量，例如 `health_status`、`health_warning_count`、`health_failure_count`、`final_exc_active_fraction`、`final_exc_top1_activity_fraction`、`final_exc_near_rate_cap_fraction` 和 `final_exc_relative_mean_drift`。

`workflows/summarize.py` 会读取 `analysis/simulation_health.json`，并把健康状态、计数和 `metrics` 展开为 `simulation_health.*` 字段。这样 CLI `summarize`、`summary.json` 和 sweep CSV 都可以直接看到单次仿真的健康状态。

开启 `analyze.inspection.enabled=true` 后，analysis run 会额外包含：

```text
runs/simulate/<timestamp>/
  analysis/
    selection_funnel.json
    graph_diagnostics.json
    unclassified_diagnostics.json
    direction_tuning.json
    robustness_summary.json             # robustness.enabled=True 时
  tables/
    selection_funnel.csv
    ensemble_direction_tuning.csv
    robustness_windows.csv              # robustness.enabled=True 且有窗口复跑时
    robustness_louvain.csv              # robustness.enabled=True 且有 Louvain grid 时
  figures/
    analysis_summary.png                # save_plots=True
    analysis_cortical_map.png           # save_plots=True
    analysis_similarity.png             # save_plots=True
    analysis_tuning.png                 # save_plots=True
    ensemble_direction_tuning.png       # save_plots=True
    analysis_failure_diagnosis.png      # save_plots=True
    analysis_robustness.png             # robustness.enabled=True 且 save_plots=True
```

如果 `output_run_root` 非空，这些文件会写到指定 analysis 输出目录；否则写回原 simulation run。`manifest.json.analysis_outputs` 会记录相对路径，方便后续脚本从 manifest 找到诊断产物。

## 机制图

`inspection.save_plots=true` 时，`_save_training_figures(...)` 会基于 `training_diagnostics.csv` 的同一组 rows 生成机制图：

- `training_overview.png`：active fraction、activity concentration、row-sum cap ratio 和 theta 的健康总览。
- `training_activity.png`：E/I rate 的 mean、median、p95、max，用于判断整体太静默还是接近 rate cap。
- `training_bcm.png`：theta p05/median/p95 与 BCM `y * (y - theta)` signal，用于看 rate-vs-theta 的学习方向。
- `training_plasticity.png`：`W_EE` 和 `W_IE` 的 p05/median/p95/max，用于看权重分布是否塌缩或爆发。
- `training_row_sums.png`：`W_EE` 和 `W_IE` 的 row-sum mean/p95/max，用于看 row-sum cap 前的压力。
- `tracked_weights.png`：只在 `tracked_weight_count > 0` 且找到可跟踪 plastic edge 时生成，展示抽样连接随 step 的权重轨迹。

图像函数不会重新运行训练，也不会重新求解 dynamics；它只消费当前 run 内存中的 diagnostic rows 和 tracked rows。

simulation 图像由 `save_simulation_figures(...)` 生成，也不会重新求解 dynamics。三张图的职责是：

- `simulate_overview.png`：活动比例、静默比例、top1/top5 集中度、near-rate-cap 和稳定性总览。
- `simulate_orientation_heatmaps.png`：按 orientation 和 cell 展示最终 E/I rates，快速看方向间是否有强烈不均衡。
- `simulate_traces.png`：按 orientation 展示 E/I population mean trajectory，快速看是否稳定、漂移或震荡。

analysis 图像由 `save_analysis_figures(...)` 生成，同样不会重新求解 dynamics。它消费当前 `AnalysisResult`、selection funnel、graph diagnostics 和 unclassified diagnostics：

- `analysis_summary.png`：selection funnel、OSI 分布、selected mean activity 和 ensemble size 的总览。
- `analysis_cortical_map.png`：按 L2/3 坐标显示 community label、OSI、preferred orientation 和 mean activity。
- `analysis_similarity.png`：按 community 排序后的 similarity 和 agreement matrix。
- `analysis_tuning.png`：每个非零 ensemble 的平均 tuning 曲线。
- `ensemble_direction_tuning.png`：每个 ensemble 的方向 tuning 曲线和 direction-selective ensemble 覆盖的方向 bin。
- `analysis_failure_diagnosis.png`：把 selection funnel、graph health、population dropout 和 cleanup dropout 放在一张图里，适合排查为什么没有清晰 ensemble。
- `analysis_robustness.png`：可选 robustness 总览，左边是不同响应窗口，右边是 Louvain 参数网格复跑。

analysis 的 JSON/CSV 诊断更适合脚本筛选：

- `selection_funnel.json` 与 `selection_funnel.csv`：记录 total、active、finite OSI、OSI pass、selected、classified 和 unclassified selected 的数量。
- `direction_tuning.json` 与 `ensemble_direction_tuning.csv`：记录 ensemble preferred direction、modulation index、direction-selective count 和 coverage。
- `graph_diagnostics.json`：记录 similarity kind、positive similarity fraction、thresholded edge density、degree 分布、isolated node 和 weak-module-degree 候选数。
- `unclassified_diagnostics.json`：记录 not active、OSI 不可用、低于 OSI 阈值、随机抽样移除、当前 filter 未选中、Louvain 未分类、weak module degree 移除和 small cluster 移除。
- `robustness_windows.csv` 与 `robustness_louvain.csv`：记录每个复跑 variant 的 status、selected neurons 和 metrics summary 标量。

## 随机性边界

本诊断路径不新增 seed。`sample_tracked_weights(...)` 继续使用全局 `np.random`，由外层实验入口统一控制随机性。健康报告、CSV 诊断和机制图本身都是确定性地从当前 batch rates、model、BCM state 与 config 计算出来。
