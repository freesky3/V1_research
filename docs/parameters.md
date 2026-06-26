# 命令参数参考

本文档说明 `v1-simulation` 每个命令能调整哪些参数，以及这些参数代表什么。它面向已经知道要跑哪个 workflow、但需要查字段名和含义的人。快速运行流程见 `docs/quickstart.md`。

参数来源有两层：

- CLI 选项：例如 `--config`、`--override/-o`、`--no-progress`。
- YAML 配置字段：例如 `model.l4.n_side`、`solver.store_trajectory`、`analysis.louvain.gamma`。

配置型命令都使用 OmegaConf dot-list 覆盖，因此可以这样临时改参数：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o model.l4.n_side=20 `
  -o grating.visual_gain=2.0 `
  -o solver.store_trajectory=true
```

多级字段用点号连接。列表可以直接写成 `[0.0,0.01,0.02]`，空值写 `null`，布尔值写 `true` 或 `false`。

## 1. CLI 命令选项

### `v1-simulation train`

用途：运行 natural-image training，保存训练后的模型 checkpoint 和训练日志。

```powershell
uv run v1-simulation train --config configs/train_smoke.yaml
```

CLI 选项：

| 选项 | 必填 | 含义 |
| --- | --- | --- |
| `--config`, `-c` | 是 | 训练 workflow 的 YAML 配置文件。 |
| `--override`, `-o` | 否 | 覆盖 YAML 中的任意字段，可重复使用。 |
| `--progress/--no-progress` | 否 | 是否显示训练进度；默认显示。 |
| `--help` | 否 | 显示命令帮助。 |

可调 YAML 字段见第 3 节，顶层配置类型是 `TrainingWorkflowConfig`。

### `v1-simulation simulate`

用途：运行 drifting-grating simulation，保存模型、rate 数组和可选轨迹。

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml
```

CLI 选项：

| 选项 | 必填 | 含义 |
| --- | --- | --- |
| `--config`, `-c` | 是 | 仿真 workflow 的 YAML 配置文件。 |
| `--override`, `-o` | 否 | 覆盖 YAML 中的任意字段，可重复使用。 |
| `--help` | 否 | 显示命令帮助。 |

可调 YAML 字段见第 4 节，顶层配置类型是 `SimulationWorkflowConfig`。

### `v1-simulation analyze`

用途：读取一个 simulation run bundle，计算 OSI、Louvain community 和 ensemble metrics。

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp>
```

CLI 选项：

| 选项 | 必填 | 含义 |
| --- | --- | --- |
| `--config`, `-c` | 是 | 分析 workflow 的 YAML 配置文件。 |
| `--override`, `-o` | 否 | 覆盖 YAML 中的任意字段，可重复使用。 |
| `--help` | 否 | 显示命令帮助。 |

可调 YAML 字段见第 5 节，顶层配置类型是 `AnalysisWorkflowConfig`。

### `v1-simulation full`

用途：先训练 natural-image 模型，再把训练得到的 `model/` checkpoint 传给 grating simulation。

```powershell
uv run v1-simulation full --config configs/full_smoke.yaml --no-progress
```

CLI 选项：

| 选项 | 必填 | 含义 |
| --- | --- | --- |
| `--config`, `-c` | 是 | full workflow 的 YAML 配置文件，包含 `train:` 和 `simulate:` 两棵配置树。 |
| `--override`, `-o` | 否 | 覆盖 YAML 中的任意字段，可重复使用。 |
| `--progress/--no-progress` | 否 | 是否显示训练阶段进度；默认显示。 |
| `--help` | 否 | 显示命令帮助。 |

可调 YAML 字段见第 6 节，顶层配置类型是 `FullWorkflowConfig`。

### `v1-simulation sweep`

用途：对 `train`、`simulate`、`analyze` 或 `full` 做显式 grid sweep。

```powershell
uv run v1-simulation sweep --config configs/sweep_simulate.yaml --no-progress
```

CLI 选项：

| 选项 | 必填 | 含义 |
| --- | --- | --- |
| `--config`, `-c` | 是 | sweep workflow 的 YAML 配置文件。 |
| `--override`, `-o` | 否 | 覆盖 sweep YAML 字段，例如 `parameters.grating.visual_gain=[0.5,1.0]`。 |
| `--progress/--no-progress` | 否 | 是否把进度传给目标 workflow 中会显示进度的部分；默认显示。 |
| `--help` | 否 | 显示命令帮助。 |

可调 YAML 字段见第 7 节，顶层配置类型是 `SweepConfig`。

### `v1-simulation summarize`

用途：只读汇总一个新格式 run bundle，写出 compact `summary.json`。

```powershell
uv run v1-simulation summarize --run runs/simulate/<timestamp>
```

CLI 选项：

| 选项 | 必填 | 含义 |
| --- | --- | --- |
| `--run` | 是 | 要汇总的 run 目录，例如 `runs/train/...` 或 `runs/simulate/...`。 |
| `--output`, `-o` | 否 | 输出 JSON 路径；不写时默认为 `<run>/summary.json`。 |
| `--help` | 否 | 显示命令帮助。 |

`summarize` 不读取 workflow YAML，也没有 `--config`。它会读取 `manifest.json`、`model/state.npz`、常见 rate 数组、`analysis/metrics.json`、`analysis/direction_tuning.json` 和训练表格。

## 2. 共享配置树

下面这些配置会被多个命令复用。字段路径以它们在 workflow YAML 中的位置为准。

### 2.1 `model`

用于 `train`、`simulate`，也会通过 `full.train.model` 或 `full.simulate.model` 出现。它控制模型几何、连接拓扑和初始权重。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `model.l4.n_side` | `40` | L4 输入 sheet 边长，输入细胞数为 `n_side^2`。 |
| `model.l4.region_size` | `2.0` | L4 sheet 的物理边长，影响坐标和距离。 |
| `model.l4.z_pos` | `0.0` | L4 sheet 的 z 位置，跨层距离会用到。 |
| `model.l4.all_tuned` | `true` | 是否让所有 L4 cells 都有方向 tuning；否则按经验比例抽 tuned cells。 |
| `model.l4.n_orientations` | `8` | L4 preferred orientation 的离散方向数。 |
| `model.l23.n_side` | `null` | L2/3 sheet 边长；为空时由经验比例和 L4 输入数量推导。 |
| `model.l23.inhibitory_fraction` | `null` | L2/3 inhibitory cell 比例；为空时由经验数据推导。 |
| `model.l23.region_size` | `2.0` | L2/3 sheet 的物理边长。 |
| `model.l23.z_pos` | `0.1` | L2/3 sheet 的 z 位置。 |
| `model.l23.random_inhibitory` | `false` | 是否随机放置 inhibitory cells；否则近似均匀放置。 |
| `model.p_ee` | `0.12` | `E <- E` 连接概率基准，其它 block 概率由经验比例推导。 |
| `model.equalize_indegree` | `true` | 是否按 target row 等化连接概率，使每行平均概率接近目标值。 |
| `model.periodic` | `true` | 是否在 sheet 边界使用周期距离。 |
| `model.connectivity_kernel.sigma_narrow` | `0.075` | 双高斯空间连接核的窄核 sigma。 |
| `model.connectivity_kernel.sigma_broad` | `0.225` | 双高斯空间连接核的宽核 sigma；必须大于 `sigma_narrow`。 |
| `model.connectivity_kernel.kappa` | `0.45` | 窄核权重，范围 `[0, 1]`。 |
| `model.weight.base_strength` | `3.0` | 初始兴奋性权重全局尺度。 |
| `model.weight.inhibitory_ratio` | `5.5` | inhibitory source 权重相对 excitatory source 的强度倍数。 |
| `model.weight.scales.ee` | `1.0` | `E <- E` block 权重缩放。 |
| `model.weight.scales.ei` | `1.08` | `E <- I` block 权重缩放，最终符号为负。 |
| `model.weight.scales.ex` | `1.0` | `E <- X/L4` block 权重缩放。 |
| `model.weight.scales.ie` | `1.0` | `I <- E` block 权重缩放。 |
| `model.weight.scales.ii` | `1.0` | `I <- I` block 权重缩放，最终符号为负。 |
| `model.weight.scales.ix` | `1.0` | `I <- X/L4` block 权重缩放。 |

常用调整：

- 增大 `model.l4.n_side` 和 `model.l23.n_side`：扩大网络规模，但计算和内存开销按平方甚至连接数增长。
- 降低 `model.p_ee`：降低 recurrent E-E 连接密度。
- 调整 `model.weight.base_strength`：整体改变初始输入电流强度。
- 调整 `model.connectivity_kernel.*`：改变连接的空间局域性。

### 2.2 `solver`

用于 `train`、`simulate` 和 `full`。它控制 Wilson-Cowan dynamics 的数值求解。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `solver.backend` | `jax-rk4` | 求解后端：`scipy` 或 `jax-rk4`。示例配置通常用 `scipy`。 |
| `solver.scipy_method` | `RK4` | SciPy 后端的方法；`RK4` 是固定步长，也可使用 `solve_ivp` 支持的方法名。 |
| `solver.store_trajectory` | `true` | 是否保存完整 E/I 时间轨迹。分析 Louvain 时建议为 `true`。 |
| `solver.jax_dtype` | `float64` | JAX 后端使用的数据类型。 |
| `solver.transfer.sigma_t` | `10.0` | Siegert transfer 中的噪声尺度。 |
| `solver.transfer.tau_exc` | `0.02` | excitatory population 的膜时间常数。 |
| `solver.transfer.tau_inh` | `0.01` | inhibitory population 的膜时间常数。 |
| `solver.transfer.refractory_tau` | `0.002` | 不应期时间常数。 |
| `solver.transfer.threshold` | `20.0` | Siegert transfer 的阈值。 |
| `solver.transfer.reset_potential` | `10.0` | Siegert transfer 的 reset potential。 |
| `solver.transfer.mu_table_max` | `100.0` | transfer table 的输入范围为 `[-mu_table_max, mu_table_max]`。 |
| `solver.transfer.rate_max` | `null` | firing rate 上限；为空时不额外裁剪。 |

常用调整：

- 小规模调试：`solver.backend=scipy`、`solver.scipy_method=RK4`。
- 性能路径：`solver.backend=jax-rk4`，同时安装 JAX optional dependency。
- 节省磁盘：`solver.store_trajectory=false`，但 analysis 会失去完整时间轨迹。

### 2.3 `time`

用于 `train`、`simulate` 和 `full` 中对应子 workflow。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `time` | `np.linspace(0.0, 0.2, 51)` | 一维、有限、严格递增的时间网格。YAML 中可写成列表，或写成 `start`/`stop` 加 `step` 或 `num` 的 mapping。 |

例子：

```yaml
time: [0.0, 0.005, 0.01, 0.015, 0.02]
```

```yaml
time:
  start: 0.0
  stop: 0.8
  step: 0.004
```

时间点越多，RK4 或 JAX scan 步数越多；`store_trajectory=true` 时输出数组也更大。

### 2.4 `background`

用于 `train`、`simulate` 和 `full`。它给 E/I populations 叠加 OU background trace。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `background.enabled` | `false` | 是否启用 OU background。 |
| `background.exc.mean` | `0.0` | E cells background 均值。 |
| `background.exc.stationary_std` | `0.0` | E cells stationary 标准差；为 0 时是常数。 |
| `background.exc.tau` | `0.05` | E cells OU 时间常数。 |
| `background.inh.mean` | `0.0` | I cells background 均值。 |
| `background.inh.stationary_std` | `0.0` | I cells stationary 标准差。 |
| `background.inh.tau` | `0.05` | I cells OU 时间常数。 |
| `background.interpolation` | `linear` | solver 内插值方式：`linear` 或 `sample_hold`。 |

例子：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml `
  -o background.enabled=true `
  -o background.exc.stationary_std=0.1 `
  -o background.inh.stationary_std=0.1
```

### 2.5 `receptive_field` 和 `gabor`

这些字段出现在 `natural_images.receptive_field` 和 `grating.receptive_field` 下。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `*.receptive_field.stimulus_size` | `2.0` | 每个 L4 RF 周围的视觉积分区域边长。 |
| `*.receptive_field.resolution` | `300` | 视觉积分网格每边采样数。越大越精细也越慢。 |
| `*.receptive_field.gabor.sigma` | `0.085` | Gabor envelope 的空间尺度。 |
| `*.receptive_field.gabor.gamma` | `1.0` | Gabor aspect ratio。 |
| `*.receptive_field.gabor.spatial_frequency` | `14.137166941154069` | Gabor 空间频率。 |
| `*.receptive_field.gabor.phase` | `0.0` | Gabor 相位。 |

示例配置为了跑得快，常把 `resolution` 降到 `7` 或 `16`。

## 3. `train` YAML 参数

`train` 的顶层配置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `run_root` | `runs` | run bundle 根目录，输出到 `<run_root>/train/<timestamp>/`。 |
| `empirical_data_path` | `data/sample_data.pkl` | 经验数据 pkl 路径。 |
| `model` | 见第 2.1 节 | 训练前构建的模型。 |
| `natural_images` | 见下表 | natural-image 数据、预处理和 L4 drive。 |
| `solver` | 见第 2.2 节 | 每个 batch 的 dynamics solver。 |
| `learning` | 见下表 | 学习规则配置。 |
| `background` | 见第 2.4 节 | batch dynamics 的 OU background。 |
| `time` | 见第 2.3 节 | 每个 batch 的时间网格。 |
| `batch_size` | `8` | 每次 solver 并行处理多少 natural-image samples。 |
| `epochs` | `1` | 遍历 natural-image samples 的轮数。 |
| `inspection` | 见下表 | 可选训练诊断。 |

### 3.1 `natural_images`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `natural_images.image_dir` | `data/vanhateren_iml` | Van Hateren `.iml` 图像目录。 |
| `natural_images.image_shape` | `[1024, 1536]` | 原始 `.iml` 图像 shape。 |
| `natural_images.pattern` | `*.iml` | 搜索图像的 glob pattern。 |
| `natural_images.crop_size` | `null` | 随机 crop 的边长；为空时使用整张图。 |
| `natural_images.patches_per_image` | `1` | 每张图每个 epoch 抽多少个 crop sample。 |
| `natural_images.limit` | `null` | 每个 epoch 最多使用多少张图；用于 smoke run 或子集实验。 |
| `natural_images.receptive_field.*` | 见第 2.5 节 | natural-image projection 的 L4 RF 参数。 |
| `natural_images.preprocess.resolution` | `128` | crop 后 resize 到的方形分辨率。 |
| `natural_images.preprocess.normalization` | `log-zscore` | 归一化方式：`log-zscore`、`zscore` 或 `maxscale`。 |
| `natural_images.preprocess.clip_zscore` | `3.0` | z-score 模式下的裁剪阈值；为空时不裁剪。 |
| `natural_images.preprocess.frame_scale` | `1.0` | 归一化后整体缩放。 |
| `natural_images.preprocess.frame_offset` | `0.0` | 归一化后整体平移。 |
| `natural_images.preprocess.antialias` | `true` | 下采样前是否做高斯抗混叠。 |
| `natural_images.preprocess.zscore_eps` | `1e-8` | 判断标准差或最大值接近零的 eps。 |
| `natural_images.drive.visual_gain` | `400.0` | RF 积分转 L4 firing-rate drive 的缩放。 |
| `natural_images.drive.baseline_rate` | `0.0` | L4 drive baseline。 |
| `natural_images.drive.periodic` | `true` | projection 时图像采样是否周期边界。 |
| `natural_images.drive.projection_chunk_size` | `64` | 构建 projection matrix 时按多少个 L4 cells 分块。 |
| `natural_images.cache_dir` | `null` | projection cache 目录；例如 `data/.projection_cache`。 |

### 3.2 `learning`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `learning.kind` | `bcm` | 学习规则类型；当前只有 `bcm`。 |
| `learning.bcm.eta` | `3.0e-5` | BCM 学习率。 |
| `learning.bcm.theta_beta` | `0.01` | sliding threshold 的更新混合系数。 |
| `learning.bcm.theta_eps` | `1.0e-6` | theta 下限和除零保护。 |
| `learning.bcm.theta_y0` | `20.0` | 用于把均方响应转换为 theta 的尺度。 |
| `learning.bcm.theta_init` | `1.0` | 初始 theta；为空时由第一批响应均方初始化。 |
| `learning.bcm.theta_floor` | `1.0e-3` | theta 的额外下限；为空时只用 `theta_eps`。 |
| `learning.bcm.theta_update_order` | `pre` | 更新权重时使用更新前还是更新后的 theta：`pre` 或 `post`。 |
| `learning.bcm.w_max` | `null` | plastic excitatory-source block 的单权重上限。 |
| `learning.bcm.row_sum_max_scale` | `1.0` | 基于初始 row sum 的行和上限缩放；为空时不限制。 |

当前 BCM 只更新 excitatory source 的 recurrent efferents：`E <- E` 和 `I <- E`。L4 input 权重不由 BCM 修改。

### 3.3 `inspection`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `inspection.enabled` | `false` | 是否记录训练诊断。 |
| `inspection.probe_every` | `1` | 每隔多少个 batch 记录一次。 |
| `inspection.tracked_weight_count` | `0` | 抽样跟踪多少条 plastic connections。 |
| `inspection.save_plots` | `false` | 是否保存训练诊断图。 |
| `inspection.save_per_batch_arrays` | `false` | 是否保存每次 probe 的 rates 和 weights 数组。 |
| `inspection.active_rate_threshold` | `1.0` | 统计 active fraction 时的 firing-rate 阈值。 |
| `inspection.health.min_active_neuron_fraction` | `0.05` | 低于该比例时记录静默告警；完全静默记为 fail。 |
| `inspection.health.max_active_neuron_fraction` | `0.95` | 高于该比例时记录过度活跃告警；完全活跃记为 fail。 |
| `inspection.health.max_top1_activity_fraction` | `0.35` | 单个神经元活动占比过高时记录集中度告警。 |
| `inspection.health.max_top5_activity_fraction` | `0.75` | 前 5 个神经元活动占比过高时记录集中度告警。 |
| `inspection.health.max_row_sum_cap_fraction` | `0.05` | 超过该比例的行到达 row-sum cap 时记录告警。 |
| `inspection.health.max_row_sum_cap_ratio` | `0.95` | row-sum/cap 最大比值超过该值时记录告警。 |
| `inspection.health.near_rate_cap_ratio` | `0.95` | 若 `solver.transfer.rate_max` 存在，用它定义 near-rate-cap 阈值。 |
| `inspection.health.max_near_rate_cap_fraction` | `0.05` | 超过该比例的 rate 接近 rate cap 时记录告警。 |

批量实验和 sweep 中建议保持 `save_plots=false`、`save_per_batch_arrays=false`，避免生成大量文件。

## 4. `simulate` YAML 参数

`simulate` 的顶层配置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `run_root` | `runs` | run bundle 根目录，输出到 `<run_root>/simulate/<timestamp>/`。 |
| `empirical_data_path` | `data/sample_data.pkl` | 不使用 checkpoint 时，用它构建新模型。 |
| `model` | 见第 2.1 节 | `model_checkpoint=null` 时使用的模型配置。 |
| `model_checkpoint` | `null` | 已训练或已保存模型目录，通常是 `runs/train/.../model` 或 `runs/simulate/.../model`。非空时忽略 `model` 并读取 checkpoint。 |
| `solver` | 见第 2.2 节 | grating dynamics solver。 |
| `grating` | 见下表 | drifting-grating stimulus 参数。 |
| `trials` | 见第 4.2 节 | 每个方向重复多少 trial、是否打乱 trial 顺序以及 trial 相位抖动。 |
| `background` | 见第 2.4 节 | grating dynamics 的 OU background。 |
| `time` | 见第 2.3 节 | grating 仿真的时间网格。 |

### 4.1 `grating`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `grating.receptive_field.*` | 见第 2.5 节 | grating 输入投影到 L4 的 RF 参数。 |
| `grating.baseline_rate` | `0.0` | L4 grating drive baseline。 |
| `grating.visual_gain` | `400.0` | RF 积分转 L4 firing-rate drive 的缩放。 |
| `grating.visual_gain_ramp_duration` | `0.0` | visual gain 的 smoothstep ramp 时长，单位同 time；`0.0` 表示从 `t=0` 直接使用完整 `visual_gain`。 |
| `grating.luminance` | `1.0` | grating luminance。 |
| `grating.contrast` | `1.0` | grating contrast。 |
| `grating.temporal_frequency` | `2*pi` | grating 时间频率，单位按模型时间解释。 |
| `grating.n_orientations` | `8` | 均匀采样的 stimulus orientation 数。 |

`simulate` 会先生成唯一的 `orientation_angles`，再按 `trials.repeats_per_direction` 展开成 trial batch 并行求解。`excitatory_rates.npy` 的第一维是 trial 数，不再是唯一方向数。

### 4.2 `trials`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `trials.repeats_per_direction` | `4` | 每个 stimulus direction 重复的独立 trial 数。 |
| `trials.shuffle` | `true` | 是否打乱 trial 顺序；随机性来自全局 NumPy seed。 |
| `trials.random_phase` | `true` | 是否给每个 trial 加 grating phase offset。 |
| `trials.phase_jitter` | `0.35` | phase offset 从 `[-phase_jitter, phase_jitter]` 均匀抽样。 |

仿真会额外写：

- `arrays/trial_direction_indices.npy`：每个 trial 对应的 direction index。
- `arrays/trial_orientation_angles.npy`：每个 trial 的实际方向角。
- `arrays/trial_phase_offsets.npy`：每个 trial 的相位偏移。

## 5. `analyze` YAML 参数

`analyze` 的顶层配置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `simulation_run` | 必填 | 要分析的 simulation run 目录。 |
| `output_run_root` | `null` | 为空时写回 `<simulation_run>/analysis`；非空时写到指定目录。 |
| `analysis` | 见下表 | OSI、筛选、Louvain 和 metrics 参数。 |
| `save_inputs` | `true` | 是否写 `inputs.json`，记录输入数组 shape 和来源。 |
| `inspection` | 见第 5.4 节 | 分析诊断、结果图谱和可选 robustness 复跑配置。 |

### 5.1 `analysis`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `analysis.osi_threshold` | `0.4` | OSI 阈值；低于阈值的 preferred orientation 记为 `NaN`。 |
| `analysis.active_threshold` | `1.0e-6` | active neuron 筛选阈值，也用于 activity health metrics。 |
| `analysis.filter_by_osi` | `true` | 是否用 OSI 阈值筛选进入 Louvain 的神经元；否则用 activity 阈值。 |
| `analysis.random_sample_fraction` | `1.0` | 从候选神经元中随机抽样的比例，范围 `(0, 1]`。 |
| `analysis.center_side_fraction` | `1.0` | 只分析中心区域的边长比例，范围 `(0, 1]`。小于 1 时裁掉边缘 cells。 |
| `analysis.trace_source_hz` | `100.0` | 输入 trajectory 的原始采样率，用于 decimation。 |
| `analysis.trace_target_hz` | `4.0` | Louvain activity trace 的目标采样率。 |
| `analysis.direction_tuning.*` | 见第 5.2 节 | ensemble 方向选择性 summary。 |
| `analysis.louvain.*` | 见下表 | community detection 参数。 |

注意：`configs/analyze_louvain.yaml` 中若出现旧字段或额外字段，只有 dataclass 中存在的字段会被构造进 `AnalysisConfig`；当前正式字段以上表为准。

### 5.2 `analysis.direction_tuning`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `analysis.direction_tuning.enabled` | `true` | 是否为非零 ensemble 计算方向调制和覆盖。 |
| `analysis.direction_tuning.modulation_threshold` | `0.2` | `modulation_index=(max-min)/(max+min)` 超过该值时记为 direction-selective。 |

输出为 `analysis/direction_tuning.json`、`tables/ensemble_direction_tuning.csv`，并在 `inspection.save_plots=true` 时写 `figures/ensemble_direction_tuning.png`。

### 5.3 `analysis.louvain`

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `analysis.louvain.thr_prop` | `0.12` | 对 similarity graph 做 proportional threshold 的保留比例，范围 `(0, 1]`。 |
| `analysis.louvain.gamma` | `0.55` | Louvain resolution 参数；越大通常越倾向更多小 community。 |
| `analysis.louvain.num_runs` | `50` | 重复 Louvain 的次数，用于 agreement matrix。 |
| `analysis.louvain.consensus_tau` | `0.5` | consensus clustering 的 agreement 阈值，范围 `[0, 1]`。 |
| `analysis.louvain.consensus_reps` | `100` | consensus 迭代次数。 |
| `analysis.louvain.min_module_degree` | `1.0` | community 内每个节点最低内部连接度，低于则丢为 unclassified。 |
| `analysis.louvain.min_cluster_size` | `8` | 最小 community 大小，小于该值的 cluster 丢为 unclassified。 |
| `analysis.louvain.similarity_kind` | `cosine` | activity trace 相似度：`cosine` 或 `pearson`。 |

小模型调试时可以降低 `min_cluster_size`、`num_runs` 和 `consensus_reps`；正式分析再提高。

### 5.4 `inspection`

`analyze` 的 `inspection` block 定义在 `AnalysisInspectionConfig` 和 `AnalysisRobustnessConfig` 中。它不改变 baseline `run_analysis(...)` 的科学计算，只控制诊断表、图像和可选复跑。

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `inspection.enabled` | `true` | 是否在 compact arrays 外额外写分析诊断。 |
| `inspection.save_plots` | `true` | 是否写 `figures/analysis_*.png` 结果图谱和失败诊断图。 |
| `inspection.save_tables` | `true` | 是否写 selection funnel、graph、unclassified 和 robustness 的 JSON/CSV 输出。 |
| `inspection.robustness.enabled` | `false` | 是否运行可选 robustness 复跑。 |
| `inspection.robustness.tail_fractions` | `[0.25, 0.5, 0.75, 1.0]` | 用响应尾段比例切窗口并复用主分析 pipeline。 |
| `inspection.robustness.end_times` | `[]` | 用绝对结束时间切 prefix 窗口；需要 simulation run 里有 `arrays/time.npy`。 |
| `inspection.robustness.louvain_parameter_grid` | `{}` | 显式 Louvain 参数网格，只支持 `louvain.*` 字段。 |
| `inspection.robustness.overlap_surrogates` | `0` | 预留字段；当前 robustness 汇总不使用 surrogate overlap。 |

常用轻量分析：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o inspection.save_plots=false
```

恢复 compact-only 行为：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o inspection.enabled=false
```

开启 robustness 复跑：

```powershell
uv run v1-simulation analyze --config configs/analyze_louvain.yaml `
  -o simulation_run=runs/simulate/<timestamp> `
  -o inspection.robustness.enabled=true `
  -o inspection.robustness.louvain_parameter_grid.louvain.gamma=[0.5,0.7]
```

## 6. `full` YAML 参数

`full` 是组合配置：

```yaml
train:
  ...
simulate:
  ...
```

| 字段 | 含义 |
| --- | --- |
| `train.*` | 完整的 `train` 配置树，见第 3 节。 |
| `simulate.*` | 完整的 `simulate` 配置树，见第 4 节。 |

运行时 `full` 会先调用 `run_training(train)`，然后把 `train` 输出的 `model/` checkpoint 自动写入 `simulate.model_checkpoint`。因此你通常不需要在 full YAML 里手动设置 `simulate.model_checkpoint`。

如果 `simulate.run_root` 保持默认 `runs`，代码会使用 `train.run_root` 作为 simulation 输出根目录，方便两个 run 落在同一个 run root 下。

覆盖参数时要带上前缀：

```powershell
uv run v1-simulation full --config configs/full_smoke.yaml `
  -o train.epochs=5 `
  -o train.learning.bcm.eta=0.00003 `
  -o simulate.grating.n_orientations=8
```

## 7. `sweep` YAML 参数

`sweep` 的顶层配置：

| 字段 | 默认值 | 含义 |
| --- | --- | --- |
| `workflow` | 必填 | 目标 workflow：`train`、`simulate`、`analyze` 或 `full`。 |
| `base` | `{}` | 目标 workflow 的基础配置 payload。字段结构必须和目标 workflow 一致。 |
| `parameters` | `{}` | dot-path 到候选值列表的映射，会做笛卡尔积。 |
| `run_root` | `runs` | sweep 自己的输出根目录，输出到 `<run_root>/sweep/<timestamp>/`。 |

例子：

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

`parameters` 支持两种写法。推荐 dot path：

```yaml
parameters:
  analysis.louvain.gamma: [0.5, 0.7, 0.9]
```

CLI override 常会形成嵌套 dict，代码会自动压平成 dot path：

```powershell
uv run v1-simulation sweep --config configs/sweep_simulate.yaml `
  -o parameters.grating.visual_gain=[0.5,1.0,2.0]
```

sweep 会把每个 grid point merge 到 `base`，构造目标 workflow dataclass，然后运行目标 workflow。失败的点写入 `status=error` 和 `error`，不会中断后续点。

## 8. 输出路径参数

这些参数不会改变科学计算，但会改变文件组织：

| 字段或选项 | 命令 | 含义 |
| --- | --- | --- |
| `run_root` | `train`, `simulate`, `sweep` | 对应 workflow run bundle 的根目录。 |
| `train.run_root` | `full` | full 中训练 run 的根目录。 |
| `simulate.run_root` | `full` | full 中仿真 run 的根目录；默认时会跟随 `train.run_root`。 |
| `output_run_root` | `analyze` | 分析结果输出目录；为空时写回 simulation run。 |

run bundle 一般形如：

```text
runs/<workflow>/<timestamp>/
  config.yaml
  manifest.json
  model/
  arrays/
  tables/
  analysis/
  figures/
```

## 9. 随机性参数

每个可执行 workflow 都支持顶层 `seed`：`train`、`simulate`、`analyze`、`full` 和 `sweep`。`summarize` 只读，不需要 seed。

可以在 YAML 里写：

```yaml
seed: 123
```

也可以用 CLI override：

```powershell
uv run v1-simulation simulate --config configs/simulate_grating.yaml -o seed=123
```

当前没有单独的 `--seed` 选项。workflow 入口会在开始时调用全局 seed helper，同时设置 Python `random`、NumPy、`PYTHONHASHSEED`，如果安装了 `torch` 也会顺带设置 CPU/CUDA 与 cuDNN 确定性开关。

随机性来源仍然是这些全局调用：

- model connection 和 weight sampling。
- L4 tuning 抽样。
- natural-image path 顺序和 crop。
- background OU noise。
- analysis 中的随机抽样和 BCT Louvain 内部随机性。

`full` 和 `sweep` 会把根 `seed` 作为整次命令的唯一随机流，子 workflow 和 grid point 不会在中途重置。`seed` 应该写在命令顶层，而不是放进 `base` 或 `parameters` 里。

如果需要在 Python 里复现同样的行为，也可以直接构造带 `seed` 的 workflow config：

```python
from v1_research.workflows import SimulationWorkflowConfig, run_grating_simulation

run = run_grating_simulation(SimulationWorkflowConfig(seed=123))
```
