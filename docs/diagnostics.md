# 训练与分析诊断说明

本文说明当前外围诊断脚本迁移后的正式入口。旧 `scripts/` 没有整体搬入新项目；可复用的计算被拆进 `learning/`、`analysis/` 和 `workflows/`，一次性画图或硬编码 run 名的脚本仍应留在旧项目或 scratch。

## 推荐阅读顺序

1. `src/v1_research/learning/diagnostics.py`：训练中间过程的纯计算指标。
2. `src/v1_research/workflows/train.py`：`TrainingInspectionConfig` 如何把指标接到 run bundle。
3. `src/v1_research/analysis/overlap.py`：ensemble label overlap、contingency、ARI 和 surrogate significance。
4. `src/v1_research/analysis/temporal.py`：复用主分析 pipeline 的稳态窗口敏感性计算。
5. `src/v1_research/workflows/summarize.py`：只读 run bundle 的统一 summary 入口。

## 训练 inspection

入口配置位于 `src/v1_research/workflows/train.py`：

```python
TrainingInspectionConfig(
    enabled=False,
    probe_every=1,
    tracked_weight_count=0,
    save_plots=False,
    save_per_batch_arrays=False,
    active_rate_threshold=1.0,
)
```

默认关闭，且 `save_plots=False`。这保证 sweep 或批量训练不会自动生成大量图片。研究者做单次诊断时显式打开即可。

数据流：

```text
run_training(...)
-> solve dynamics for one natural-image batch
-> RateBatch(exc, inh, external)
-> LearningRule.initialize(...) or step(...)
-> if inspection enabled:
   -> active_rate_stats(...)
   -> plastic_weight_stats(...)
   -> weight_delta_stats(previous_model, model)
   -> theta_stats(...) and row_sum_pressure(...)
   -> optional sample_tracked_weights(...) / record_tracked_weights(...)
   -> optional per-probe arrays
-> write run bundle
```

写盘结果：

```text
runs/train/<timestamp>/
  tables/
    training_log.csv
    training_diagnostics.csv        # inspection.enabled=True
    tracked_weights.csv             # 存在可跟踪 plastic edge 时
  arrays/
    training_probe_000001_exc_rates.npy
    training_probe_000001_inh_rates.npy
    training_probe_000001_weights.npy
    ...
  figures/
    training_overview.png           # save_plots=True
    tracked_weights.png             # save_plots=True 且有 tracked weights
```

`manifest.json` 只记录这些输出路径和短 summary，不塞大数组。

## Learning 诊断函数

`learning/diagnostics.py` 中的函数只接收 `ModelState`、`BCMState`、`RateBatch` 或数组，不接 root config，不读写磁盘：

- `active_rate_stats(...)`：E/I firing rate 的 mean、median、max 和 active fraction。
- `plastic_weight_stats(...)`：`E <- E` 与 `I <- E` plastic block 的非零数量和权重统计。
- `row_sum_pressure(...)`：每行正权重和，以及相对 BCM row-sum cap 的压力。
- `cap_fraction(...)`：数组中达到 cap 的数量和比例；cap 数组长度不匹配会直接报错，避免静默错算。
- `weight_delta_stats(...)`：相邻模型状态之间的 plastic block delta。
- `theta_stats(...)`：BCM theta 的 mean/median。
- `sample_tracked_weights(...)`：用全局 `np.random` 从已连接 plastic edges 中抽样。
- `record_tracked_weights(...)`：读取 tracked edges 的当前权重和相对初值 delta。

`sample_tracked_weights(...)` 不创建局部 RNG，也不接收 seed。抽中的 tracked sample 在表格内重新编号为 `0..n-1`，便于 CSV 和图例阅读。

## Overlap 和 ARI

`analysis/overlap.py` 替代旧 `compare_ensemble_overlap.py` 一类脚本中的可复用计算：

```python
from v1_research.analysis.overlap import compare_label_sets

result = compare_label_sets(
    reference_labels=labels_a,
    reference_coords=coords_a,
    query_labels=labels_b,
    query_coords=coords_b,
)
```

它会先按坐标匹配 cell，再计算：

- matched / unmatched counts
- non-zero label contingency table
- one-to-one best label matches
- adjusted Rand index

`overlap_significance(...)` 用全局 `np.random.shuffle` 生成 query-label surrogate。这里同样不引入局部 RNG 或 seed 字段。

## 稳态窗口敏感性

`analysis/temporal.py` 提供：

```python
from v1_research.analysis.temporal import run_window_analysis
```

它接收已经加载好的 `AnalysisInputs`，按 `tail_fractions` 或 `end_times` 切出响应窗口，然后复用同一个 `run_analysis(...)`。这样 DG orientation window robustness 不需要新建专用脚本，也不会复制 OSI、Louvain 或 metrics 逻辑。

典型数据流：

```text
load_analysis_inputs_from_simulation(...)
-> AnalysisInputs(responses, coords, distance, orientation_angles)
-> run_window_analysis(cfg, inputs, tail_fractions=(0.25, 0.5))
-> list[summary rows]
```

这些 summary rows 可以被现有 sweep 的 `summary.*` 字段自然收集。

## Run summarize CLI

统一只读汇总入口位于 `workflows/summarize.py` 和 CLI：

```powershell
uv run v1-simulation summarize --run runs/train/...
uv run v1-simulation summarize --run runs/simulate/... --output summary.json
```

它读取新 run bundle 中常见的：

- `manifest.json`
- `model/state.npz`
- `arrays/excitatory_rates.npy`
- `arrays/inhibitory_rates.npy`
- `analysis/metrics.json`
- `tables/training_log.csv`
- `tables/training_diagnostics.csv`

该命令替代旧的 `summarize_simulation_run.py`、`summarize_analysis_artifact.py` 和 `summarize_bcm_diagnostics.py` 的常用只读汇总场景。它不兼容旧 artifact 命名。

## Sweep 边界

DG orientation coverage、Louvain 参数、window robustness、spatial gates 等扫描都应继续使用 `workflows/sweep.py`：

```yaml
workflow: analyze
base:
  simulation_run: runs/simulate/example
  analysis:
    filter_by_osi: false
parameters:
  analysis.louvain.thr_prop: [0.08, 0.12, 0.16]
  analysis.louvain.gamma: [0.7, 0.9]
```

成功的 workflow 会把短 summary 暴露给 sweep CSV，例如 `summary.n_ensembles`、`summary.classified_fraction`、`summary.osi_mean`。不要为了每个旧 sweep 脚本再加专用 workflow。

## 随机性边界

本轮迁移遵守项目统一 seed 约定：

- 不新增 `seed` 配置字段。
- 不调用 `np.random.default_rng(...)`。
- 不接收或保存 `np.random.Generator`。
- 需要抽样时使用全局 `np.random`，由主程序在进入 workflow 前统一 `set_seed(CONFIG["seed"])`。

这意味着同一个 sweep 内各 grid point 会按顺序消费全局随机状态。如果需要完全可重复的实验顺序，由调用方在 workflow 外统一设置 seed，而不是在诊断函数内部重置随机状态。
