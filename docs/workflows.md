# Workflows 层说明

本文档说明当前 `v1_research.workflows` 和 `v1_research.runs` 的逻辑。Workflow 层只负责调度和 IO：组装模型、输入、solver、learning rule，写 run bundle。科学计算仍留在 `model/`、`inputs/`、`dynamics/`、`learning/` 和后续 `analysis/`。

## 推荐阅读顺序

1. `src/v1_research/runs.py`：run 目录、manifest、CSV、模型 checkpoint 的轻量 IO。
2. `src/v1_research/workflows/train.py`：natural-image training workflow 和已有 batch helper。
3. `src/v1_research/workflows/simulate.py`：drifting-grating simulation workflow。
4. `src/v1_research/workflows/analyze.py`：simulation run bundle 的 OSI/Louvain/metrics 分析 workflow。
5. `src/v1_research/workflows/full.py`：先 train 后 simulate 的组合 workflow。
6. `src/v1_research/workflows/sweep.py`：轻量 grid sweep，展开 YAML 参数并调用已有 workflow。
7. `src/v1_research/cli.py`：Typer + OmegaConf 的命令入口。

## Run Bundle

`create_run_dir(run_root, workflow)` 创建：

```text
runs/<workflow>/<timestamp>/
  config.yaml
  manifest.json
  model/
    state.npz
    metadata.json
  arrays/
  tables/
  analysis/
  figures/
```

`state.npz` 保存 `ModelState` 的 layout、connection mask 和 weights。mask/weights 以 CSR component 保存，不会为了 checkpoint 把大矩阵转成 dense array。`load_model_state(...)` 是当前唯一的模型 checkpoint 读取入口。

`manifest.json` 只记录 workflow 类型、solver、learning rule、模型尺寸、关键输出路径和短 summary。不要把所有数组塞进 manifest，也不要恢复旧项目的 `aE_all.npy`、`run_config.json` 等兼容命名。

## Training Workflow

入口：

```python
from v1_research.workflows import TrainingWorkflowConfig, run_training

result = run_training(TrainingWorkflowConfig(), show_progress=False)
```

局部 config 位于 `workflows/train.py`：

- `NaturalImageWorkflowConfig`：自然图像目录、shape、crop、RF config、preprocess config、drive config、projection cache 路径。
- `TrainingWorkflowConfig`：组合 `ModelConfig`、`SolverConfig`、`LearningConfig`、`BackgroundConfig`、time grid、batch size 和 epoch 数。
- `TrainingInspectionConfig`：可选训练中间过程诊断，默认关闭；`save_plots=False` 保证 sweep 默认不生图。

数据流：

```text
TrainingWorkflowConfig
-> ExperimentalData.from_path(...)
-> build_model(cfg.model, empirical)
-> VanHaterenImageDataset / NaturalImageSampler
-> NaturalImagePreprocessor / L4NaturalImageProjector / NaturalImageL4Drive
-> optional NaturalImageProjectionCache preload
-> solve_and_learn_batch(...)
   -> solve_rates(...)
   -> RateBatch
   -> LearningRule.initialize(...) or step(...)
-> tables/training_log.csv
-> optional tables/training_diagnostics.csv / tracked_weights.csv
-> model/state.npz
-> manifest.json
```

`solve_and_learn_batch(...)` 仍然是一个薄 helper，只依赖 `LearningRule` 协议，不包含 BCM 公式。首次 batch 初始化 learning state，`updated=False`；后续 batch 调用 rule step，`updated=True`。

开启 `inspection.enabled=True` 时，workflow 会记录 active-rate、theta、plastic weight、row-sum/cap pressure 和可选 tracked weights。只有 `inspection.save_plots=True` 时才写 `figures/training_overview.png` 和 `figures/tracked_weights.png`。

重要 shape：

- natural-image drive 返回 `(n_input, n_batch)`。
- solver 输出 `RateResult.exc=(n_batch, n_exc)`、`RateResult.inh=(n_batch, n_inh)`。
- learning 使用 `RateBatch.external=(n_batch, n_input)`。

## Simulation Workflow

入口：

```python
from v1_research.workflows import SimulationWorkflowConfig, run_grating_simulation

result = run_grating_simulation(SimulationWorkflowConfig(model_checkpoint="runs/train/.../model"))
```

`SimulationWorkflowConfig` 位于 `workflows/simulate.py`，组合 `ModelConfig`、`SolverConfig`、`DriftingGratingConfig`、`BackgroundConfig`、time grid、可选 `model_checkpoint` 和 `SimulationInspectionConfig`。单次 `simulate` 默认 `inspection.enabled=True`、`inspection.save_plots=True`，用于把一次仿真直接变成健康检查入口；如果要只保存轻量 rate 数组，可以显式关闭 `inspection.enabled` 并把 `solver.store_trajectory=false`。

数据流：

```text
SimulationWorkflowConfig
-> load_model_state(model_checkpoint) or build_model(...)
-> DriftingGratingInput(cfg.grating, model.layout)
-> stimulus.make_batched_drive_func(orientation_angles)
-> solve_rates(...)
-> optional compute_simulation_health(...) from full trajectory
-> arrays/excitatory_rates.npy
-> arrays/inhibitory_rates.npy
-> arrays/time.npy
-> arrays/orientation_angles.npy
-> optional trajectory arrays
-> optional analysis/simulation_health.json
-> optional figures/simulate_overview.png / simulate_orientation_heatmaps.png / simulate_traces.png
-> model/state.npz
-> manifest.json
```

重要 shape：

- orientation batch size 等于 `grating.n_orientations`。
- `excitatory_rates.npy` shape 为 `(n_orientations, n_exc)`。
- `inhibitory_rates.npy` shape 为 `(n_orientations, n_inh)`。
- 若 `SolverConfig.store_trajectory=True`，trajectory shape 为 `(n_time, n_orientations, n_exc/n_inh)`。
- `simulation_health.json` 直接从 trajectory 计算 activity、silent fraction、top1/top5 concentration、near-rate-cap、front/tail drift、tail variance、step-to-step change，以及 stimulus/background 输入分布。OSI、community 和 ensemble metrics 仍由 `analyze` workflow 负责。

## Full Workflow

入口：

```python
from v1_research.workflows import FullWorkflowConfig, run_train_then_simulate

result = run_train_then_simulate(FullWorkflowConfig())
```

`run_train_then_simulate(...)` 先运行 `run_training(...)`，再把训练输出的 `model/` checkpoint 作为 `SimulationWorkflowConfig.model_checkpoint` 传给 `run_grating_simulation(...)`。它只是组合两个 workflow，不新增训练或仿真的科学逻辑。

## Sweep Workflow

入口：

```python
from v1_research.workflows import SweepConfig, run_sweep

result = run_sweep(
    SweepConfig(
        workflow="simulate",
        base={"solver": {"backend": "scipy"}},
        parameters={"grating.visual_gain": [100.0, 200.0]},
    )
)
```

`SweepConfig` 位于 `workflows/sweep.py`，只包含目标 workflow、base config、显式参数网格和 sweep run 根目录。它不恢复 Hydra config group，也不引入调度器、resume 或随机种子。参数网格使用 dot path 展开，例如：

```yaml
workflow: simulate
run_root: runs
base:
  solver:
    backend: scipy
    scipy_method: RK4
  grating:
    n_orientations: 4
parameters:
  grating.visual_gain: [100.0, 200.0]
  solver.store_trajectory: [false, true]
```

数据流：

```text
SweepConfig
-> expand_grid(parameters)
-> merge each grid point into base
-> construct target workflow dataclass
-> call train/simulate/analyze/full workflow
-> tables/runs.csv
-> summary.json
-> manifest.json
```

sweep 失败边界很简单：单个 grid point 报错时，在 `tables/runs.csv` 记录 `status=error` 和 `error`，然后继续下一个点。成功行记录目标 workflow 的 `run_dir` 和扁平化的 `summary.*` 字段。`simulate` sweep 有一层专门的轻量默认：如果 base 没有显式写 `inspection`，会关闭 simulation inspection 和 plot，并在没有显式要求完整诊断时设置 `solver.store_trajectory=false`。需要完整仿真健康报告时，在 sweep base 或参数网格中显式设置 `inspection.enabled=true` 和 `solver.store_trajectory=true`。

`analyze` workflow 会把 metrics summary 中的标量也放进 `summary`，所以 sweep 可以直接记录 `summary.n_ensembles`、`summary.classified_fraction`、`summary.osi_mean` 等字段，不需要专用分析 sweep 脚本。

## Analysis Workflow

入口：

```python
from v1_research.workflows import AnalysisWorkflowConfig, run_analysis_workflow

result = run_analysis_workflow(AnalysisWorkflowConfig(simulation_run="runs/simulate/..."))
```

`AnalysisWorkflowConfig` 位于 `workflows/analyze.py`，组合 simulation run 路径、可选 `output_run_root`、`AnalysisConfig` 和 `save_inputs`。底层 OSI、Louvain、spatial/community metrics 都在 `analysis/` 模块内，workflow 只负责读取 bundle、调用计算、写结果。

数据流：

```text
AnalysisWorkflowConfig
-> load_analysis_inputs_from_simulation(...)
   -> arrays/excitatory_trajectory.npy or arrays/excitatory_rates.npy
   -> arrays/orientation_angles.npy
   -> model/state.npz
-> run_analysis(...)
-> analysis/ compact arrays and JSON
-> tables/ensemble_metrics.csv
-> manifest.json analysis summary
```

重要 shape：

- `excitatory_trajectory.npy` shape 为 `(n_time, n_orientations, n_exc)`，analysis 内部转成 `(n_exc, n_orientations, n_time)`。
- 若没有 trajectory，则用 `excitatory_rates.npy` 的 `(n_orientations, n_exc)` 作为单时间点响应，内部 shape 为 `(n_exc, n_orientations, 1)`。
- analysis 当前只分析 excitatory L2/3 cells，坐标来自 `model.layout.l23.coords[layout.exc_idx]`。

## CLI

入口在 `src/v1_research/cli.py`：

```powershell
uv run v1-simulation train --config configs/train_bcm.yaml -o batch_size=4
uv run v1-simulation simulate --config configs/simulate_grating.yaml -o solver.backend=scipy
uv run v1-simulation analyze --config configs/analyze_louvain.yaml -o analysis.osi_threshold=0.3
uv run v1-simulation full --config configs/full.yaml --no-progress
uv run v1-simulation sweep --config configs/sweep_simulate.yaml -o parameters.grating.visual_gain=[100.0,200.0]
uv run v1-simulation summarize --run runs/simulate/...
```

CLI 使用 `OmegaConf.load(...)` 和 `OmegaConf.from_dotlist(...)` 做 YAML + `key=value` override，然后递归构造对应 workflow dataclass。这里没有全局 schema，也不接管 random seed；主程序仍应在进入 workflow 前统一设置全局 seed。

`summarize` 是只读入口，读取新 run bundle 的 manifest、model checkpoint、常见 arrays、analysis metrics、`analysis/training_health.json`、`analysis/simulation_health.json` 和训练表，写出 compact `summary.json` 或用户指定输出路径。不兼容旧 artifact 名。

## 随机性和边界

Workflow 层不创建局部 RNG，也没有 `seed` 字段。随机性来自 model/input/background 中已有的全局 `np.random` 调用，由主程序统一设置 seed。

当前没有恢复 old scripts 的 plotting-only diagnostics、early stop 或 Diffrax 验证脚本。后续迁移这些能力时，应把纯计算放到对应科学模块，workflow 只负责读取 run bundle、调用计算、保存结果。
