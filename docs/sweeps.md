# Sweeps 与常用实验入口说明

本文档说明当前轻量 sweep 和常用 YAML 实验入口的逻辑。它们只负责把显式参数网格展开成一组已有 workflow 调用，不恢复 Hydra config group、实验调度器、resume 系统或局部随机种子。

## 推荐阅读顺序

1. `src/v1_research/workflows/sweep.py`：`SweepConfig`、`expand_grid(...)`、`run_sweep(...)` 的主逻辑。
2. `src/v1_research/cli.py`：`v1-simulation sweep` 如何加载 YAML 和 `-o key=value` override。
3. `configs/sweep_simulate.yaml`：轻量 grating simulation sweep 示例。
4. `configs/train_smoke.yaml`、`simulate_grating.yaml`、`analyze_louvain.yaml`、`full_smoke.yaml`：常用单次实验入口。

## Sweep 入口

Python 入口：

```python
from v1_research.workflows import SweepConfig, run_sweep

result = run_sweep(
    SweepConfig(
        workflow="simulate",
        base={"solver": {"backend": "scipy"}},
        parameters={"grating.visual_gain": [0.5, 1.0]},
    )
)
```

CLI 入口：

```powershell
uv run v1-simulation sweep --config configs/sweep_simulate.yaml
uv run v1-simulation sweep --config configs/sweep_simulate.yaml -o parameters.grating.visual_gain=[0.5,1.0]
```

`SweepConfig` 位于 `workflows/sweep.py`，字段很少：

- `workflow`：目标 workflow，取值为 `train`、`simulate`、`analyze` 或 `full`。
- `base`：目标 workflow 的基础 YAML payload。
- `parameters`：dot-path 到候选值列表的映射。
- `run_root`：sweep 自己的 run 根目录，默认 `runs`。

## 数据流

```text
SweepConfig
-> expand_grid(parameters)
-> 每个 grid point merge 到 base
-> 用现有 CLI dataclass 构造逻辑生成目标 workflow config
-> 若 workflow=simulate 且未显式请求 inspection，则覆盖为轻量仿真配置
-> 调用 run_training / run_grating_simulation / run_analysis_workflow / run_train_then_simulate
-> 写 sweep run bundle
```

`parameters` 支持两种 YAML 写法。推荐直接使用 dot path：

```yaml
parameters:
  grating.visual_gain: [0.5, 1.0]
  solver.store_trajectory: [false, true]
```

CLI override 经过 OmegaConf 时会变成嵌套 dict，因此 `SweepConfig.__post_init__` 会把嵌套形式重新压平成 dot path：

```yaml
parameters:
  grating:
    visual_gain: [0.5, 1.0]
```

### simulate sweep 的轻量默认

普通 `simulate` workflow 默认保存完整 trajectory、写 `analysis/simulation_health.json` 并生成三张诊断图。`sweep` 调用 `simulate` 时默认相反：如果 `base` 没有显式写 `inspection`，会补上轻量 inspection 配置；如果同时没有显式写 `solver.store_trajectory`，也会补成轻量 trajectory 配置：

```yaml
inspection:
  enabled: false
  save_plots: false
solver:
  store_trajectory: false
```

这样扫参默认只保存均值 rate、orientation、time、model 和 manifest，避免每个 grid point 都写大 trajectory 或三张图。需要把某个 sweep 当作完整诊断批处理时，必须在 `base` 或 `parameters` 中显式打开：

```yaml
base:
  inspection:
    enabled: true
    save_plots: true
  solver:
    store_trajectory: true
```

如果只想保存 trajectory 供后续 `analyze` 使用，但不想生成 simulation health 和图，可以显式设置 `solver.store_trajectory=true`、`inspection.enabled=false`。

## 输出

sweep 输出目录：

```text
runs/sweep/<timestamp>/
  config.yaml
  manifest.json
  summary.json
  tables/
    runs.csv
```

`tables/runs.csv` 每一行对应一个 grid point，包含：

- `index`
- 每个 swept parameter 的 dot-path 列
- `workflow`
- `status`
- `run_dir`
- `error`
- 目标 workflow 返回的 `summary.*` 字段

如果某个 grid point 抛错，sweep 记录 `status=error` 和 `error`，然后继续下一个点。CSV 会先归一化列名，所以即使第一行失败、后续行成功，后续 `summary.*` 列也不会丢失。

当目标 workflow 是 `simulate` 且显式开启 inspection 时，`summary.*` 中会包含 `summary.health_status`、`summary.health_warning_count`、`summary.final_exc_active_fraction`、`summary.final_exc_top1_activity_fraction` 等健康摘要。默认轻量 simulate sweep 不生成这些列。

对 `analyze` workflow 做 sweep 时，analysis metrics summary 中的标量会自动进入 `summary.*` 列，例如：

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

生成的 CSV 会包含 `summary.n_ensembles`、`summary.classified_fraction`、`summary.osi_mean` 等字段，可直接替代旧的 DG spatial / orientation coverage 参数扫脚本中的主要筛选表。

## 常用 YAML 入口

当前 `configs/` 下提供轻量入口：

- `train_smoke.yaml`：小模型 natural-image training 配置，依赖 `data/vanhateren_iml` 中存在外部 `.iml` 图像。
- `simulate_grating.yaml`：小模型 drifting-grating simulation，可直接从 `data/sample_data.pkl` 构建模型。
- `analyze_louvain.yaml`：分析已有 simulation run 的 OSI/Louvain/metrics 示例，需要把 `simulation_run` 改成真实 run 路径。
- `full_smoke.yaml`：先 train 后 simulate 的小配置，同样依赖外部 natural-image 数据。
- `sweep_simulate.yaml`：对 `simulate` workflow 做 `visual_gain` 和 `store_trajectory` 的显式 grid sweep。

这些文件是可编辑的实验起点，不是新的配置层。字段仍由各 workflow 自己的 dataclass 解释。

## 随机性边界

sweep 层不创建局部 RNG，不接 `seed` 字段，也不在每个 grid point 内重设随机状态。复现边界仍然在主程序入口，由用户统一调用全局 `set_seed(CONFIG["seed"])`。

如果同一个 sweep 内不同 grid point 需要严格可复现，调用方应在进入 `run_sweep(...)` 前设置全局 seed，并理解每个 grid point 会按顺序消耗全局随机状态。不要在 sweep 内部重新引入 `np.random.default_rng(...)` 或 per-run seed。
