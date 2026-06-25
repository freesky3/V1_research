# 训练与仿真诊断健康报告

本文说明当前训练与 drifting-grating 仿真阶段的诊断代码如何组织，以及 `inspection.enabled=true` 时 run bundle 会写出哪些健康报告、表格和机制图。它面向需要判断训练或单次仿真是否过于静默、过度活跃、活动过度集中，是否接近 firing-rate cap，或 BCM row-sum cap 压力过大的研究者。

## 推荐阅读顺序

1. `src/v1_research/workflows/train.py`：从 `TrainingWorkflowConfig`、`TrainingInspectionConfig` 和 `run_training(...)` 看训练入口与落盘逻辑。
2. `src/v1_research/learning/diagnostics.py`：看纯诊断函数、`TrainingHealthConfig` 和 `evaluate_training_health(...)`。
3. `src/v1_research/workflows/simulation_health.py`：看 simulation 专用的 `SimulationHealthConfig`、`compute_simulation_health(...)` 和 `save_simulation_figures(...)`。
4. `src/v1_research/workflows/simulate.py`：看 `SimulationInspectionConfig` 如何把健康报告接入单次 grating simulation。
5. `src/v1_research/learning/bcm.py`：看 BCM state、theta、row-sum cap 与 plastic weight 更新机制。
6. `src/v1_research/workflows/summarize.py`：看 run bundle 如何被压缩成 sweep 友好的 summary。
7. `docs/learning.md` 与 `docs/workflows.md`：补充理解训练规则和 CLI workflow 边界。

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

旧的 `active_rate_stats(...)`、`plastic_weight_stats(...)`、`theta_stats(...)` 仍保留，用于兼容已有轻量诊断；扩展函数会补充更适合训练机制解释的字段。

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

- `RateResult.exc_trajectory`: `(n_time, n_orientations, n_exc)`。
- `RateResult.inh_trajectory`: `(n_time, n_orientations, n_inh)`。
- sampled stimulus trace: `(n_time, n_orientations, n_input)`。
- `BackgroundTrace.exc`: `(n_time, n_orientations, n_exc)`，`BackgroundTrace.inh`: `(n_time, n_orientations, n_inh)`。

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
    ...
  figures/
    training_overview.png           # save_plots=True
    training_activity.png           # save_plots=True
    training_bcm.png                # save_plots=True
    training_plasticity.png         # save_plots=True
    training_row_sums.png           # save_plots=True
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

## 随机性边界

本诊断路径不新增 seed。`sample_tracked_weights(...)` 继续使用全局 `np.random`，由外层实验入口统一控制随机性。健康报告、CSV 诊断和机制图本身都是确定性地从当前 batch rates、model、BCM state 与 config 计算出来。
