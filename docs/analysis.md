# Analysis 层说明

本文档说明当前 `v1_research.analysis` 的主分析链路。Analysis 层只负责把 simulation run bundle 里的兴奋性响应转换成 OSI、Louvain community labels、spatial/community metrics、ensemble direction tuning 和紧凑 artifact；它不重新求解动力学、不训练权重，也不迁移旧项目的大型诊断绘图。本轮另外补上了轻量 `inspection` 诊断：selection funnel、graph health、unclassified 原因、结果图谱和可选 robustness 复跑，但这些都仍然是 analysis 的附属读数，不是新的动力学计算。

## 推荐阅读顺序

1. `src/v1_research/analysis/pipeline.py`：主数据流、`AnalysisConfig`、`AnalysisInputs`、`run_analysis(...)`。
2. `src/v1_research/analysis/osi.py`：OSI 和 preferred orientation 计算。
3. `src/v1_research/analysis/communities.py`：`LouvainConfig`、similarity matrix、agreement matrix、consensus Louvain。
4. `src/v1_research/analysis/direction_tuning.py`：ensemble 方向偏好、调制指数和方向覆盖。
5. `src/v1_research/analysis/metrics.py`：activity health、OSI distribution、community summary rows。
6. `src/v1_research/analysis/diagnostics.py`：selection funnel、graph health 和 unclassified 原因。
7. `src/v1_research/analysis/artifacts.py`：分析结果和 inspection 产物写盘。
8. `src/v1_research/analysis/robustness.py`：窗口复跑和 Louvain 参数网格复跑。
9. `src/v1_research/workflows/analysis_figures.py`：analysis 结果图谱和 robustness 图。
10. `src/v1_research/analysis/overlap.py`：两个 analysis label set 的坐标匹配、overlap 和 ARI。
11. `src/v1_research/analysis/temporal.py`：对不同稳态窗口复用主分析 pipeline。
12. `src/v1_research/workflows/analyze.py`：从 simulation run bundle 读取输入、调用 analysis、inspection 和 robustness、更新 manifest。

## 当前入口

分析计算入口：

```python
from v1_research.analysis import AnalysisConfig, run_analysis
from v1_research.analysis.pipeline import load_analysis_inputs_from_simulation

inputs = load_analysis_inputs_from_simulation("runs/simulate/...")
result = run_analysis(AnalysisConfig(), inputs)
```

workflow 入口：

```python
from v1_research.workflows import AnalysisWorkflowConfig, run_analysis_workflow

run = run_analysis_workflow(AnalysisWorkflowConfig(simulation_run="runs/simulate/..."))
```

CLI 入口：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml
uv run v1-simulation analyze --config configs/analyze_louvain.yaml -o analysis.osi_threshold=0.3
```

## 数据流

```text
Simulation run bundle
-> load_analysis_inputs_from_simulation(...)
   -> arrays/excitatory_trajectory.npy or arrays/excitatory_rates.npy
   -> arrays/orientation_angles.npy
   -> optional arrays/trial_direction_indices.npy
   -> model/state.npz
   -> excitatory L2/3 coords and distance
-> run_analysis(...)
   -> steady-state window
   -> compute_osi(...)
   -> select_analysis_neuron_indices(...)
   -> ensemble_activity_trace(...)
   -> identify_communities(...)
   -> summarize_communities(...)
-> write_analysis_result(...)
-> optional write_analysis_inspection(...)
   -> selection_funnel.json / graph_diagnostics.json / unclassified_diagnostics.json
   -> selection_funnel.csv
-> optional save_analysis_figures(...)
   -> analysis_summary.png / analysis_cortical_map.png / analysis_similarity.png
   -> analysis_tuning.png / analysis_failure_diagnosis.png
-> optional run_window_analysis(...) / run_louvain_parameter_grid(...)
   -> robustness_windows.csv / robustness_louvain.csv
   -> robustness_summary.json / analysis_robustness.png
-> analysis/ arrays and JSON
-> tables/ensemble_metrics.csv
-> analysis/direction_tuning.json and tables/ensemble_direction_tuning.csv
-> manifest.json analysis summary
```

如果 simulation run 保存了新格式 trial `excitatory_trajectory.npy`，shape 应为：

```text
(n_time, n_trial, n_exc)
```

analysis 会保留 trial-resolved trace 给 Louvain，并根据 `trial_direction_indices.npy` 聚合成内部方向 shape：

```text
responses: (n_exc, n_orientation, n_time)
```

如果没有 trajectory，则退回读取 `excitatory_rates.npy`：

```text
excitatory_rates.npy: (n_trial, n_exc)
responses: (n_exc, n_orientation, 1)
```

这个 fallback 可以计算 OSI、direction tuning 和 metrics，但 Louvain 只有均值响应轨迹，时间信息较少。没有 trial metadata 的旧式新项目 run 仍按一方向一 batch 读取。

## OSI

`compute_osi(responses_mean, orientation_angles, min_osi=0.4)` 使用 circular variance：

```text
vector = sum(response(theta) * exp(2j * theta))
osi = abs(vector) / sum(response(theta))
```

`preferred_orientation` 是仿真方向 bins 中 mean response 最大的方向。若 OSI 低于阈值或总响应为 0，则 preferred orientation 记为 `NaN`。

## Louvain Communities

`LouvainConfig` 位于 `analysis/communities.py`，因为这些字段只由 community detection 解释：

```python
LouvainConfig(
    thr_prop=0.12,
    gamma=0.55,
    num_runs=50,
    consensus_tau=0.5,
    consensus_reps=100,
    min_module_degree=1.0,
    min_cluster_size=8,
    similarity_kind="cosine",
)
```

`identify_communities(...)` 的步骤：

```text
activity_trace: (n_selected, n_features)
-> cosine or pearson similarity
-> threshold_proportional
-> repeated bct.community_louvain
-> agreement matrix
-> bct.consensus_und
-> drop weak or small clusters
-> relabel consecutive labels
```

label `0` 表示 unclassified。非零 community labels 会重新映射成连续整数。

随机性仍由 workflow 入口统一控制：analysis 纯计算层不创建局部 RNG，也不接收 `np.random.Generator`。`AnalysisWorkflowConfig.seed` 会在 `run_analysis_workflow(...)` 入口设置全局 seed；BCT 在未显式传 seed 时继续使用 NumPy 全局随机状态。

## Metrics 和 Artifacts

`summarize_communities(...)` 输出两类结果：

- global summary：ensemble 数量、classified fraction、ensemble size、similarity 和 OSI distribution。
- per-ensemble rows：size、centroid、within/outside similarity、spatial compactness、member OSI、preferred orientation coherence。

`summarize_direction_tuning(...)` 对每个非零 ensemble 输出方向响应均值、preferred direction、`modulation_index=(max-min)/(max+min)` 和是否超过 `modulation_threshold`。全局 summary 记录 direction-selective ensemble 数、比例、覆盖到的 direction bin 数和 ensemble modulation 的 min/mean。

默认写盘位置：

```text
runs/simulate/<timestamp>/
  analysis/
    config.yaml
    selected_indices.npy
    osi.npy
    preferred_orientation.npy
    responses_mean.npy
    steady_state_responses.npy
    coords.npy
    distance.npy
    community_labels.npy
    similarity.npy
    agreement.npy
    diagnostics.json
    community_diagnostics.json
    metrics.json
    direction_tuning.json
    inputs.json
  tables/
    ensemble_metrics.csv
    ensemble_direction_tuning.csv
  manifest.json
```

`output_run_root` 非空时，analysis workflow 会把结果写到指定目录，而不是直接写回 simulation run 的 `analysis/` 子目录。

## Overlap 和时间窗口

`compare_label_sets(...)` 用坐标匹配两个 analysis 的 selected cells，然后计算 non-zero community label contingency、best one-to-one label matches 和 adjusted Rand index。`overlap_significance(...)` 用全局 `np.random.shuffle` 生成 query-label surrogate，不创建局部 RNG。

`run_window_analysis(...)` 接收已经加载好的 `AnalysisInputs`，按 `tail_fractions` 或 `end_times` 切出响应窗口，再调用同一个 `run_analysis(...)`。它用于检查 steady-state window 对 OSI/Louvain 结果的影响，不复制 OSI、Louvain 或 metrics 逻辑。

## 当前边界

本轮只迁移主分析链路：

- 已迁移：OSI、Louvain、activity/spatial/community metrics、ensemble direction tuning、compact artifacts、overlap/ARI、window sensitivity、`analyze` workflow 和 CLI。
- 未迁移：旧 `frames_sorted.py` 的大型 sorted-trial trace/variance 图和 single-neuron diagnostics、DG/OU all-cell 专用脚本。

这些未迁移部分如果以后需要，应先明确科学问题，再把纯计算拆到 `analysis/`，把调度放到 `workflows/` 或 sweep 模块中。

## 分析诊断输出

`AnalysisWorkflowConfig` 在 `src/v1_research/workflows/analyze.py` 中额外挂了一个 `inspection` block：

```python
AnalysisInspectionConfig(
    enabled=True,
    save_plots=True,
    save_tables=True,
    robustness=AnalysisRobustnessConfig(),
)
```

`AnalysisRobustnessConfig` 默认是：

```python
AnalysisRobustnessConfig(
    enabled=False,
    tail_fractions=(0.25, 0.5, 0.75, 1.0),
    end_times=(),
    louvain_parameter_grid={},
    overlap_surrogates=0,
)
```

默认 `analyze` 会在 compact arrays 之外再写轻量 inspection 产物：

- `analysis/selection_funnel.json`
- `analysis/graph_diagnostics.json`
- `analysis/unclassified_diagnostics.json`
- `tables/selection_funnel.csv`
- `figures/analysis_summary.png`
- `figures/analysis_cortical_map.png`
- `figures/analysis_similarity.png`
- `figures/analysis_tuning.png`
- `figures/analysis_failure_diagnosis.png`

其中：

- `selection_funnel.*` 记录候选细胞如何经过 active / OSI / random sampling / Louvain 分类各阶段。
- `graph_diagnostics.json` 记录 similarity graph 的密度、degree 分布、孤立点和弱模块候选数。
- `unclassified_diagnostics.json` 记录没被选中或没被分类的原因。
- `analysis_summary.png` 是结果总览图。
- `analysis_cortical_map.png` 是 cortical scatter map。
- `analysis_similarity.png` 是 similarity / agreement 矩阵图。
- `analysis_tuning.png` 是各 ensemble 的 tuning 曲线。
- `analysis_failure_diagnosis.png` 是 selection、graph 和 dropout 的失败诊断图。

`inspection.save_plots=false` 时仍保留 JSON/CSV 诊断，但不写 PNG。`inspection.save_tables=false` 时只保留 compact arrays 和图，不写这些诊断表。`inspection.enabled=false` 时恢复为以前的 compact-output 行为。

`inspection.robustness.enabled=true` 时，workflow 还会跑窗口复用和 Louvain 参数网格复跑，并写：

- `tables/robustness_windows.csv`
- `tables/robustness_louvain.csv`
- `analysis/robustness_summary.json`
- `figures/analysis_robustness.png`

窗口复跑走 `run_window_analysis(...)`，依赖 `tail_fractions` 或 `end_times`。Louvain 复跑走 `run_louvain_parameter_grid(...)`，只接受 `louvain.*` 参数网格。`analysis_robustness.png` 是一个两栏图：左边看不同窗口，右边看不同 Louvain 参数组合。
