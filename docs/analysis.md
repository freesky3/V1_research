# Analysis 层说明

本文档说明当前 `v1_research.analysis` 的主分析链路。Analysis 层只负责把 simulation run bundle 里的兴奋性响应转换成 OSI、Louvain community labels、spatial/community metrics 和紧凑 artifact；它不重新求解动力学、不训练权重，也不迁移旧项目的大型诊断绘图。

## 推荐阅读顺序

1. `src/v1_research/analysis/pipeline.py`：主数据流、`AnalysisConfig`、`AnalysisInputs`、`run_analysis(...)`。
2. `src/v1_research/analysis/osi.py`：OSI 和 preferred orientation 计算。
3. `src/v1_research/analysis/communities.py`：`LouvainConfig`、similarity matrix、agreement matrix、consensus Louvain。
4. `src/v1_research/analysis/metrics.py`：activity health、OSI distribution、community summary rows。
5. `src/v1_research/analysis/artifacts.py`：分析结果写盘。
6. `src/v1_research/analysis/overlap.py`：两个 analysis label set 的坐标匹配、overlap 和 ARI。
7. `src/v1_research/analysis/temporal.py`：对不同稳态窗口复用主分析 pipeline。
8. `src/v1_research/workflows/analyze.py`：从 simulation run bundle 读取输入、调用 analysis、更新 manifest。

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
-> analysis/ arrays and JSON
-> tables/ensemble_metrics.csv
-> manifest.json analysis summary
```

如果 simulation run 保存了 `excitatory_trajectory.npy`，shape 应为：

```text
(n_time, n_orientation, n_exc)
```

analysis 会转成内部 shape：

```text
responses: (n_exc, n_orientation, n_time)
```

如果没有 trajectory，则退回读取 `excitatory_rates.npy`：

```text
excitatory_rates.npy: (n_orientation, n_exc)
responses: (n_exc, n_orientation, 1)
```

这个 fallback 可以计算 OSI 和 metrics，但 Louvain 只有均值响应轨迹，时间信息较少。

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

随机性仍由主程序统一控制：analysis 层没有 `seed` 字段，不创建局部 RNG，也不接收 `np.random.Generator`。BCT 在未显式传 seed 时使用 NumPy 全局随机状态，因此入口处的 `set_seed(CONFIG["seed"])` 仍是复现边界。

## Metrics 和 Artifacts

`summarize_communities(...)` 输出两类结果：

- global summary：ensemble 数量、classified fraction、ensemble size、similarity 和 OSI distribution。
- per-ensemble rows：size、centroid、within/outside similarity、spatial compactness、member OSI、preferred orientation coherence。

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
    inputs.json
  tables/
    ensemble_metrics.csv
  manifest.json
```

`output_run_root` 非空时，analysis workflow 会把结果写到指定目录，而不是直接写回 simulation run 的 `analysis/` 子目录。

## Overlap 和时间窗口

`compare_label_sets(...)` 用坐标匹配两个 analysis 的 selected cells，然后计算 non-zero community label contingency、best one-to-one label matches 和 adjusted Rand index。`overlap_significance(...)` 用全局 `np.random.shuffle` 生成 query-label surrogate，不创建局部 RNG。

`run_window_analysis(...)` 接收已经加载好的 `AnalysisInputs`，按 `tail_fractions` 或 `end_times` 切出响应窗口，再调用同一个 `run_analysis(...)`。它用于检查 steady-state window 对 OSI/Louvain 结果的影响，不复制 OSI、Louvain 或 metrics 逻辑。

## 当前边界

本轮只迁移主分析链路：

- 已迁移：OSI、Louvain、activity/spatial/community metrics、compact artifacts、overlap/ARI、window sensitivity、`analyze` workflow 和 CLI。
- 未迁移：`frames_sorted.py`、plotting-heavy diagnostics、DG/OU all-cell 专用脚本。

这些未迁移部分如果以后需要，应先明确科学问题，再把纯计算拆到 `analysis/`，把调度放到 `workflows/` 或 sweep 模块中。
