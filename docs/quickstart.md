# 快速上手使用说明

本文面向希望尽快完成一次训练、仿真、分析并拿到结果表格的人。它不解释每个模块的内部实现；如果要查每个命令有哪些可调参数，见 `docs/parameters.md`。如果要理解代码结构，可以再读 `docs/model.md`、`docs/inputs.md`、`docs/dynamics.md`、`docs/learning.md`、`docs/analysis.md`、`docs/workflows.md` 和 `docs/sweeps.md`。

所有命令都假设你已经进入项目根目录：

```powershell
cd D:\skywalker\sjtu\大三下课程\Research\MouseV1\V1_simulation
```

## 1. 准备环境和数据

项目使用 `uv` 管理环境，命令入口在 `pyproject.toml` 中注册为 `v1-simulation`。

```powershell
uv sync
uv run v1-simulation --help
```

默认配置里的小规模示例都使用 SciPy 后端，因此只需要基础依赖即可。若你要使用 `solver.backend=jax-rk4`，需要额外安装 JAX：

```powershell
uv sync --extra jax
```

或在 CUDA 12 环境中使用：

```powershell
uv sync --extra jax-cuda12
```

项目当前需要两类数据：

- `data/sample_data.pkl`：模型构建所需的经验比例、连接比例和权重样本。仿真、训练和 full workflow 都会读取它。
- `data/vanhateren_iml/*.iml`：natural-image training 使用的 Van Hateren 图像。只跑 drifting-grating simulation 时不需要这些图像。

如果你只是想确认流程能跑通，建议先运行 `simulate`，因为它只依赖 `sample_data.pkl`。

## 2. 最短路径：仿真到分析

### 2.1 跑一次 drifting-grating 仿真

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml
```

命令会打印新生成的 run 目录，例如：

```text
runs\simulate\20260624-224900
```

这个目录就是后续分析的输入。主要输出包括：

```text
runs/simulate/<timestamp>/
  config.yaml
  manifest.json
  model/
    state.npz
    metadata.json
  arrays/
    excitatory_rates.npy
    inhibitory_rates.npy
    time.npy
    orientation_angles.npy
    trial_direction_indices.npy
    trial_orientation_angles.npy
    trial_phase_offsets.npy
    excitatory_trajectory.npy      # solver.store_trajectory=true 时存在
    inhibitory_trajectory.npy      # solver.store_trajectory=true 时存在
```

`orientation_angles.npy` 保存唯一的 stimulus directions。`excitatory_rates.npy` 的 shape 是 `(n_trials, n_exc)`，其中 `n_trials = n_orientations * trials.repeats_per_direction`。如果保存 trajectory，`excitatory_trajectory.npy` 的 shape 是 `(n_time, n_trials, n_exc)`。

### 2.2 分析刚才的仿真结果

把 `simulation_run` 改成真实 run 目录：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml -o simulation_run=runs/simulate/<timestamp>
```

默认 `output_run_root: null`，因此分析结果会写回 simulation run 的 `analysis/` 子目录：

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
    selection_funnel.csv             # inspection.save_tables=true 时
  figures/
    analysis_summary.png             # inspection.save_plots=true 时
    analysis_cortical_map.png
    analysis_similarity.png
    analysis_tuning.png
    ensemble_direction_tuning.png
    analysis_failure_diagnosis.png
  manifest.json
```

最常看的结果：

- `analysis/metrics.json`：整体 summary，如 ensemble 数量、分类比例、OSI 统计等。
- `tables/ensemble_metrics.csv`：每个 ensemble 的 size、空间紧凑性、方向偏好一致性等。
- `analysis/direction_tuning.json` 和 `tables/ensemble_direction_tuning.csv`：每个 ensemble 的方向偏好、调制指数和方向覆盖 summary。
- `analysis/community_labels.npy`：每个被选中神经元的 community label，`0` 表示 unclassified。
- `figures/analysis_summary.png`：默认生成的结果总览图。
- `figures/analysis_failure_diagnosis.png`：默认生成的筛选、图结构和未分类原因诊断图。

如果希望把分析结果写到单独目录，而不是写回 simulation run：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o output_run_root=runs/analyze/my_analysis
```

如果只想要 JSON/CSV 诊断而不生成 PNG：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o inspection.save_plots=false
```

### 2.3 生成一个 compact summary

```powershell
uv run v1-simulation summarize --run runs/simulate/<timestamp>
```

默认会写：

```text
runs/simulate/<timestamp>/summary.json
```

这个 summary 会汇总 `manifest.json`、模型规模、rate 数组统计、analysis metrics 和训练日志里常见的末行指标。

## 3. 训练 natural-image 模型

训练入口：

```powershell
uv run v1-simulation train --config configs/train_smoke.yaml
```

这个 smoke 配置是小模型、小时间网格、小图像样本，用来检查流程。它依赖 `data/vanhateren_iml` 中存在 `.iml` 图像。

训练输出目录类似：

```text
runs/train/<timestamp>/
  config.yaml
  manifest.json
  model/
    state.npz
    metadata.json
  tables/
    training_log.csv
```

如果 `inspection.enabled=true`，还会额外写训练诊断表、可选数组和图：

```text
runs/train/<timestamp>/
  tables/
    training_diagnostics.csv
    training_health_events.csv
    tracked_weights.csv
  analysis/
    training_health.json
  arrays/
    training_probe_000001_exc_rates.npy
    training_probe_000001_inh_rates.npy
    training_probe_000001_weights.npy
  figures/
    training_overview.png
    training_activity.png
    training_bcm.png
    training_plasticity.png
    training_row_sums.png
    tracked_weights.png
```

训练后仿真时，把训练输出的 `model/` 目录作为 checkpoint：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o model_checkpoint=runs/train/<timestamp>/model
```

也可以直接运行 full workflow：先训练，再用训练得到的 checkpoint 做 grating simulation。

```powershell
uv run v1-simulation full --config configs/full_smoke.yaml --no-progress
```

`full` 命令会打印 simulation run 目录；训练 run 和仿真 run 分别保存在 `runs/train/<timestamp>/` 与 `runs/simulate/<timestamp>/`。

## 4. 配置文件怎么改

CLI 的基本形式是：

```powershell
uv run v1-simulation <workflow> --config <yaml> -o key=value
```

`--config/-c` 读取 YAML，`--override/-o` 使用 OmegaConf dot-list 覆盖 YAML 字段。可以写多个 `-o`：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o model.l4.n_side=20 `
  -o model.l23.n_side=20 `
  -o grating.n_orientations=8 `
  -o solver.store_trajectory=true
```

建议把常用实验复制成自己的 YAML，例如：

```powershell
Copy-Item configs/simulate_grating.yaml configs/simulate_my_experiment.yaml
```

然后在 YAML 中保存主参数，只把临时改动放到 `-o`。

### 4.1 模型规模和连接

这些字段在 `model:` 下，影响细胞数量、连接拓扑和初始权重。

```yaml
model:
  l4:
    n_side: 40              # L4 输入 sheet 边长，n_input = n_side^2
    region_size: 2.0
    all_tuned: true
    n_orientations: 8
  l23:
    n_side: 40              # L2/3 sheet 边长；不写时可由经验比例推导
    inhibitory_fraction: 0.2
    region_size: 2.0
    random_inhibitory: false
  p_ee: 0.12                # E <- E 目标连接概率的基准
  equalize_indegree: true
  periodic: true
  weight:
    base_strength: 3.0
    inhibitory_ratio: 5.5
```

常用改法：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o model.l4.n_side=30 `
  -o model.l23.n_side=30 `
  -o model.p_ee=0.08 `
  -o model.weight.base_strength=2.0
```

注意：细胞数随 `n_side^2` 增长，连接矩阵和求解时间会很快变大。先用小模型确认参数，再扩大规模。

### 4.2 Solver 和时间网格

这些字段控制 Wilson-Cowan rate dynamics 的积分。

```yaml
solver:
  backend: scipy            # scipy 或 jax-rk4
  scipy_method: RK4         # scipy backend 下可用 RK4 或 solve_ivp 方法名
  store_trajectory: true    # analysis 推荐 true；false 只保存稳态均值 rates
  jax_dtype: float64
time: [0.0, 0.01, 0.02]
```

常用改法：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o solver.backend=scipy `
  -o solver.store_trajectory=true `
  -o time=[0.0,0.005,0.01,0.015,0.02]
```

如果你希望做 Louvain/时间轨迹分析，建议保留 `solver.store_trajectory=true`。如果只需要最终平均响应，可以设为 `false` 减少磁盘占用。

### 4.3 Drifting grating 参数

这些字段在 `grating:` 下，只影响 grating stimulus 和 L4 RF projection。

```yaml
grating:
  baseline_rate: 0.1
  visual_gain: 1.0
  luminance: 1.0
  contrast: 1.0
  temporal_frequency: 6.283185307179586
  n_orientations: 4
  receptive_field:
    stimulus_size: 0.5
    resolution: 7
    gabor:
      sigma: 0.2
      gamma: 1.0
      spatial_frequency: 1.0
      phase: 0.0
```

常用改法：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o grating.visual_gain=2.0 `
  -o grating.contrast=0.8 `
  -o grating.n_orientations=8 `
  -o grating.receptive_field.resolution=15
```

`n_orientations` 越大，仿真 batch 越大；`receptive_field.resolution` 越大，输入积分越细，但更慢。

### 4.4 Natural-image training 参数

这些字段在 `natural_images:`、`learning:`、`batch_size` 和 `epochs` 下。

```yaml
natural_images:
  image_dir: data/vanhateren_iml
  image_shape: [1024, 1536]
  crop_size: 16
  patches_per_image: 1
  limit: 1
  preprocess:
    resolution: 16
    normalization: maxscale
  drive:
    visual_gain: 0.01
    baseline_rate: 0.1
  cache_dir: data/.projection_cache
learning:
  kind: bcm
  bcm:
    theta_init: 1.0
    eta: 0.00001
    row_sum_max_scale: null
batch_size: 1
epochs: 1
```

常用改法：

```powershell
uv run v1-simulation train --config configs/train_smoke.yaml `
  -o natural_images.limit=20 `
  -o natural_images.patches_per_image=4 `
  -o batch_size=8 `
  -o epochs=5 `
  -o learning.bcm.eta=0.00003
```

如果你反复使用同一批图像 crop 和 RF 参数，保留 `natural_images.cache_dir=data/.projection_cache` 可以复用 natural-image projection cache。只要模型几何、RF、预处理或 sample crop 变了，cache key 会改变，不会误用旧投影。

### 4.5 训练诊断参数

默认训练只写 `training_log.csv`。如果想观察训练中间过程：

```powershell
uv run v1-simulation train --config configs/train_smoke.yaml `
  -o inspection.enabled=true `
  -o inspection.probe_every=1 `
  -o inspection.tracked_weight_count=20 `
  -o inspection.save_plots=true
```

常用字段：

- `inspection.enabled`：是否记录诊断。
- `inspection.probe_every`：每隔多少个 batch 记录一次。
- `inspection.tracked_weight_count`：随机抽样跟踪多少条 plastic connection。
- `inspection.save_plots`：是否保存 `figures/training_overview.png` 和 tracked weights 图。
- `inspection.save_per_batch_arrays`：是否保存每次 probe 的 rates/weights 数组；批量实验中建议保持 `false`。
- `inspection.health.*`：训练健康报告阈值；默认偏早提醒，可用 `-o inspection.health.min_active_neuron_fraction=0.1` 等方式覆盖。

### 4.6 分析参数

分析配置在 `analysis:` 下，Louvain 参数在 `analysis.louvain:` 下。

```yaml
analysis:
  osi_threshold: 0.4
  active_threshold: 0.000001
  filter_by_osi: true
  random_sample_fraction: 1.0
  center_side_fraction: 1.0
  trace_source_hz: 100.0
  trace_target_hz: 4.0
  louvain:
    thr_prop: 0.12
    gamma: 0.55
    num_runs: 50
    consensus_tau: 0.5
    consensus_reps: 100
    min_module_degree: 1.0
    min_cluster_size: 8
    similarity_kind: cosine
```

常用改法：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o analysis.filter_by_osi=true `
  -o analysis.osi_threshold=0.3 `
  -o analysis.louvain.thr_prop=0.10 `
  -o analysis.louvain.gamma=0.7
```

如果小模型或 smoke run 中通过 OSI 筛选后神经元太少，可以先用：

```powershell
-o analysis.filter_by_osi=false -o analysis.osi_threshold=0.0 -o analysis.louvain.min_cluster_size=1
```

正式实验再提高 `osi_threshold`、`min_cluster_size` 和 Louvain 重复次数。

## 5. 扫参

扫参入口是：

```powershell
uv run v1-simulation sweep --config configs/sweep_simulate.yaml
```

一个 sweep YAML 的结构是：

```yaml
workflow: simulate
run_root: runs
base:
  run_root: runs
  empirical_data_path: data/sample_data.pkl
  solver:
    backend: scipy
    scipy_method: RK4
    store_trajectory: true
  grating:
    visual_gain: 1.0
    n_orientations: 4
  time: [0.0, 0.01, 0.02]
parameters:
  grating.visual_gain: [0.5, 1.0, 2.0]
  solver.store_trajectory: [false, true]
```

`workflow` 可以是 `train`、`simulate`、`analyze` 或 `full`。`base` 是目标 workflow 的基础配置，`parameters` 是要展开的网格。上面例子会生成 `3 x 2 = 6` 个 run。

输出目录：

```text
runs/sweep/<timestamp>/
  config.yaml
  manifest.json
  summary.json
  tables/
    runs.csv
```

`tables/runs.csv` 每行对应一个 grid point，包含：

- `index`
- 每个被扫参数的取值，例如 `grating.visual_gain`
- `workflow`
- `status`
- `run_dir`
- `error`
- 目标 workflow 的 `summary.*` 字段

某个 grid point 报错时，sweep 会记录 `status=error` 和 `error`，然后继续下一个点。

### 5.1 用 CLI 临时改 sweep 范围

可以直接覆盖 `parameters`：

```powershell
uv run v1-simulation sweep --config configs/sweep_simulate.yaml `
  -o parameters.grating.visual_gain=[0.5,1.0,2.0,4.0] `
  -o parameters.grating.n_orientations=[4,8]
```

OmegaConf 会把 `parameters.grating.visual_gain` 解析成嵌套 dict，`SweepConfig` 会在内部重新压平成 dot path。

### 5.2 扫 grating 仿真参数

建议从小模型、小时间网格开始：

```yaml
workflow: simulate
run_root: runs
base:
  run_root: runs
  empirical_data_path: data/sample_data.pkl
  model:
    l4:
      n_side: 10
      region_size: 1.0
      all_tuned: true
      n_orientations: 8
    l23:
      n_side: 10
      inhibitory_fraction: 0.2
      region_size: 1.0
      random_inhibitory: false
    p_ee: 0.08
    weight:
      base_strength: 1.0
  solver:
    backend: scipy
    scipy_method: RK4
    store_trajectory: true
  grating:
    baseline_rate: 0.1
    visual_gain: 1.0
    n_orientations: 8
  background:
    enabled: false
  time: [0.0, 0.005, 0.01, 0.015, 0.02]
parameters:
  grating.visual_gain: [0.5, 1.0, 2.0]
  grating.contrast: [0.5, 1.0]
  model.weight.base_strength: [0.5, 1.0]
```

运行：

```powershell
uv run v1-simulation sweep --config configs/sweep_my_simulate.yaml --no-progress
```

### 5.3 扫分析参数

先跑一个 simulation，再扫分析参数：

```yaml
workflow: analyze
run_root: runs
base:
  simulation_run: runs/simulate/<timestamp>
  output_run_root: null
  save_inputs: true
  analysis:
    filter_by_osi: false
    osi_threshold: 0.0
    random_sample_fraction: 1.0
    center_side_fraction: 1.0
    louvain:
      thr_prop: 0.12
      gamma: 0.55
      num_runs: 20
      consensus_tau: 0.5
      consensus_reps: 50
      min_module_degree: 1.0
      min_cluster_size: 8
      similarity_kind: cosine
parameters:
  analysis.louvain.thr_prop: [0.08, 0.12, 0.16]
  analysis.louvain.gamma: [0.5, 0.7, 0.9]
  analysis.osi_threshold: [0.2, 0.3, 0.4]
```

分析 workflow 的 summary 会进入 sweep CSV，例如：

```text
summary.n_ensembles
summary.classified_neurons
summary.classified_fraction
summary.osi_mean
```

这样可以直接用 `runs/sweep/<timestamp>/tables/runs.csv` 比较不同 Louvain 参数。

### 5.4 扫训练参数

训练扫参通常更耗时，建议先关掉图和 per-batch 数组：

```yaml
workflow: train
run_root: runs
base:
  run_root: runs
  empirical_data_path: data/sample_data.pkl
  model:
    l4:
      n_side: 6
      region_size: 1.0
      all_tuned: true
      n_orientations: 4
    l23:
      n_side: 6
      inhibitory_fraction: 0.2
      region_size: 1.0
      random_inhibitory: false
    p_ee: 0.08
    weight:
      base_strength: 0.5
  natural_images:
    image_dir: data/vanhateren_iml
    image_shape: [1024, 1536]
    crop_size: 32
    patches_per_image: 2
    limit: 10
    receptive_field:
      stimulus_size: 0.5
      resolution: 7
      gabor:
        sigma: 0.2
        gamma: 1.0
        spatial_frequency: 1.0
        phase: 0.0
    preprocess:
      resolution: 32
      normalization: maxscale
    drive:
      visual_gain: 0.01
      baseline_rate: 0.1
    cache_dir: data/.projection_cache
  solver:
    backend: scipy
    scipy_method: RK4
    store_trajectory: false
  learning:
    kind: bcm
    bcm:
      theta_init: 1.0
      eta: 0.00001
      row_sum_max_scale: 1.0
  inspection:
    enabled: true
    probe_every: 5
    tracked_weight_count: 0
    save_plots: false
    save_per_batch_arrays: false
  batch_size: 4
  epochs: 1
parameters:
  learning.bcm.eta: [0.00001, 0.00003]
  learning.bcm.row_sum_max_scale: [0.8, 1.0, 1.2]
```

如果后续要比较每个训练 run 的 grating response，可以先扫 `train`，再把每行 `run_dir/model` 作为 `simulate.model_checkpoint` 单独仿真。

## 6. 结果怎么看

### 6.1 每个 run 都先看 manifest

```text
runs/<workflow>/<timestamp>/manifest.json
```

`manifest.json` 记录 workflow 类型、solver、模型规模、主要输出路径和短 summary。它是判断 run 是否正常完成的第一入口。

### 6.2 训练结果

常用文件：

- `tables/training_log.csv`：每个 batch 的学习规则统计。
- `tables/training_diagnostics.csv`：开启 inspection 后的 rate、theta、plastic weight 和 row-sum cap 指标。
- `analysis/training_health.json`：训练健康状态、告警事件、最终指标和最差指标。
- `tables/training_health_events.csv`：每个健康告警事件一行；告警不会中断训练。
- `model/state.npz`：训练后的模型 checkpoint。

常用命令：

```powershell
uv run v1-simulation summarize --run runs/train/<timestamp>
```

### 6.3 仿真结果

常用文件：

- `arrays/excitatory_rates.npy`：每个 trial 的 E cell 平均响应。
- `arrays/inhibitory_rates.npy`：每个 trial 的 I cell 平均响应。
- `arrays/excitatory_trajectory.npy`：完整 E cell 时间轨迹，开启 `store_trajectory` 时存在。
- `arrays/orientation_angles.npy`：orientation bins。
- `arrays/trial_direction_indices.npy`：每个 trial 对应哪个 orientation bin。

常用命令：

```powershell
uv run v1-simulation summarize --run runs/simulate/<timestamp>
```

### 6.4 分析结果

常用文件：

- `analysis/osi.npy`：被选中细胞的 OSI。
- `analysis/preferred_orientation.npy`：被选中细胞的 preferred orientation。
- `analysis/community_labels.npy`：Louvain community label。
- `analysis/metrics.json`：整体 summary。
- `analysis/direction_tuning.json`：ensemble 方向调制和覆盖 summary。
- `tables/ensemble_metrics.csv`：每个 ensemble 的指标表。
- `tables/ensemble_direction_tuning.csv`：每个 ensemble 的方向 tuning 表。
- `analysis/selection_funnel.json` 与 `tables/selection_funnel.csv`：候选细胞经过 active、OSI、sampling 和 Louvain 分类的数量漏斗。
- `analysis/graph_diagnostics.json`：Louvain similarity graph 的边密度、degree 分布和孤立点。
- `analysis/unclassified_diagnostics.json`：细胞没有被选中或没有被分类的原因。
- `figures/analysis_summary.png`、`figures/analysis_cortical_map.png`、`figures/analysis_similarity.png`、`figures/analysis_tuning.png`、`figures/ensemble_direction_tuning.png`、`figures/analysis_failure_diagnosis.png`：默认生成的分析结果图谱。

如果 `manifest.json` 里的 `analysis.status` 是 `not_enough_neurons`，常见原因是 `analysis.filter_by_osi=true` 且 `osi_threshold` 太高，或者小模型里 active neuron 太少。可以先降低 `osi_threshold` 或设置 `filter_by_osi=false` 检查流程。

## 7. 常见问题

### 找不到 Van Hateren 图像

报错类似：

```text
No Van Hateren .iml files found in data/vanhateren_iml
```

说明你正在跑 `train` 或 `full`，但 `natural_images.image_dir` 下没有 `.iml` 文件。解决方式：

- 放入 Van Hateren `.iml` 数据。
- 或先只运行 `simulate`。
- 或把 `natural_images.image_dir` 改成真实图像目录。

### 分析没有 trajectory

如果 `solver.store_trajectory=false`，analysis 会退回使用 `excitatory_rates.npy`，也就是每个 trial 一个稳态均值点；若存在 `trial_direction_indices.npy`，仍会先按方向平均后计算 OSI 和 ensemble direction tuning。这样仍可计算 summary，但 Louvain 的时间信息较少。正式做 community 分析时建议仿真阶段使用：

```powershell
-o solver.store_trajectory=true
```

### 分析诊断输出太多

默认 `analyze` 会额外写 selection/graph/unclassified 诊断和五张 `analysis_*.png` 图。如果批量分析时不想生成图片，保留 JSON/CSV 但关闭图：

```powershell
-o inspection.save_plots=false
```

如果只想要以前的 compact arrays、`metrics.json` 和 `ensemble_metrics.csv`：

```powershell
-o inspection.enabled=false
```

### 想检查分析结果是否稳定

可选 robustness 需要显式打开。它会复用同一个 simulation run，比较不同时间窗口和 Louvain 参数下的结果：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o inspection.robustness.enabled=true `
  -o inspection.robustness.louvain_parameter_grid.louvain.gamma=[0.5,0.7]
```

可能写出的文件包括 `tables/robustness_windows.csv`、`tables/robustness_louvain.csv`、`analysis/robustness_summary.json` 和 `figures/analysis_robustness.png`。`end_times` 模式需要 simulation run 里有 `arrays/time.npy`。

### 扫参结果有失败行

查看：

```text
runs/sweep/<timestamp>/tables/runs.csv
```

失败行的 `error` 列会保存异常信息。sweep 不会因为某个点失败而中断，这是为了保留其他 grid point 的结果。

### 复现实验

当前 model、input、learning 和 analysis 底层仍不创建局部 RNG，随机性来自全局随机状态。可执行 workflow 现在接收顶层 `seed` 字段；可以在 YAML 写 `seed: 123`，也可以用 CLI override：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml -o seed=123
```

`full` 和 `sweep` 会把根 `seed` 作为整次命令的唯一随机流，不会在 train/simulate 子流程或 grid point 之间重置。
