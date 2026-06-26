# Inputs 层说明

本文档说明当前 `v1_research.inputs` 的逻辑。输入层只负责把外部刺激转换为 L4 外部 drive；它不解释 Wilson-Cowan 动力学、不更新权重，也不负责 workflow 调度。

## 推荐阅读顺序

1. `src/v1_research/inputs/gabor.py`：理解视觉积分网格、Gabor RF、L4 tuning 如何进入输入层。
2. `src/v1_research/inputs/grating.py`：drifting grating 的解析积分路径，用于 orientation 仿真。
3. `src/v1_research/inputs/natural_images.py`：Van Hateren 图像、crop sample、预处理和 L4 projection。
4. `src/v1_research/inputs/background.py`：E/I population 的 OU background trace。
5. `docs/cache.md`：Gabor projection matrix 和 natural-image rates cache。

## 共同边界

输入层入口都接收明确对象，不接收顶层全局 config：

- `DriftingGratingInput(cfg, layout)` 接收 `DriftingGratingConfig` 和 `PopulationLayout`。
- `L4NaturalImageProjector(layout, rf_cfg, drive_cfg)` 接收 `PopulationLayout`、`ReceptiveFieldConfig`、`NaturalImageDriveConfig`。
- `generate_background_trace(cfg, n_exc, n_inh, n_batch, time)` 接收 `BackgroundConfig` 和明确 shape 参数。

`PopulationLayout` 必须提供：

- `layout.l4.coords`：shape `(n_input, 2)`。
- `layout.l4_tuning_labels`：shape `(n_input,)`，值为 `"T"` 或 `"U"`。
- `layout.l4_preferred_orientations`：shape `(n_input,)`，未调谐 cell 为 `NaN`。

输入层不再兼容旧项目的 `L4.N`、`tunings`、`pref_dirs` 等属性探测。缺少 tuning 信息时直接报错，因为这会让 RF 计算静默错误。

## Gabor RF

局部 config 在 `inputs/gabor.py`：

- `GaborConfig`：`sigma`、`gamma`、`spatial_frequency`、`phase`。
- `ReceptiveFieldConfig`：`stimulus_size`、`resolution`、`gabor`。

核心数据流：

```text
PopulationLayout
-> l4_tuning_arrays(layout)
-> VisualGrid.centered_midpoint(...)
-> gabor_kernel(...) per L4 cell
-> L4GaborBank.filters
```

`VisualGrid` 使用 centered midpoint grid，`area_element = dx * dy`。`L4GaborBank.filters` 懒加载后设为 read-only，shape 为：

```text
(n_input, resolution, resolution)
```

调谐 cell 使用 oriented Gabor；未调谐 cell 使用 isotropic Gaussian envelope，并忽略 preferred orientation。

## Drifting Grating

局部 config 在 `inputs/grating.py`：`DriftingGratingConfig`。它拥有 grating stimulus 参数，也嵌入 `ReceptiveFieldConfig`。

当前入口：

```python
stimulus = DriftingGratingInput(cfg, model.layout)
drive = stimulus.make_batched_drive_func(theta_angles)
a_x = drive(t)  # shape: (n_input, n_theta)
```

解析路径：

```text
orientation theta
-> _precompute_integrals(theta)
-> base, cos_coeff, sin_coeff cached by theta
-> external_drive(theta, t)
```

每个 orientation 只做一次 RF/grid 空间积分。之后任意时间点只代入 `temporal_frequency * t`，避免在 solver 循环里反复生成 frame 和投影。

`visual_gain_ramp_duration > 0` 时，L4 drive 末端的 `visual_gain` 会乘一个 smoothstep 时间包络：`t<=0` 为 0，`t>=duration` 为 1，中间使用 `s*s*(3-2*s)`。这用于减轻 drifting-grating 在 `t=0` 硬切入造成的共同 transient；`stimulus_frame(...)` 仍渲染完整 grating frame，不应用该 gain ramp。

重要 shape：

- `external_drive(theta, t)` 返回 `(n_input,)`。
- `make_batched_drive_func(theta_angles)(t)` 返回 `(n_input, n_theta)`。
- `stimulus_frame(theta, t)` 返回 `(n_input, resolution, resolution)`，用于检查或可视化每个 L4 RF center 周围的 grating frame。

## Natural Images

局部 config 在 `inputs/natural_images.py`：

- `NaturalImagePreprocessConfig`：resize、normalization、clip、frame scale/offset。
- `NaturalImageDriveConfig`：`visual_gain`、`baseline_rate`、periodic sampling、projection chunk size。
- RF 参数仍来自 `ReceptiveFieldConfig`。

训练路径建议：

```text
VanHaterenImageDataset
-> NaturalImageSampler.make_epoch(...)
-> NaturalImagePreprocessor.transform(image, sample)
-> L4NaturalImageProjector.project(frame)
-> NaturalImageL4Drive.rates_for_sample(sample)
```

`NaturalImageSample` 由 image path 和可选 `CropBox` 组成。`NaturalImageSampler` 使用全局 `np.random` 进行 image order、crop 和 sample order 采样；主程序在 workflow 入口统一调用 `set_global_seed(CONFIG["seed"])`。

`L4NaturalImageProjector.projection_matrix(frame_shape)` 返回矩阵：

```text
(n_input, height * width)
```

`project(frame)` 要求 frame 是 2D，返回 `(n_input,)`。`project_frames(frames)` 要求同 batch frame shape 一致，返回 `(n_frames, n_input)`。

`NaturalImageL4Drive.make_static_batch_func(samples)(t)` 返回 solver 更常用的 batch drive shape：

```text
(n_input, n_batch)
```

## Background

局部 config 在 `inputs/background.py`：

- `OUParams(mean, stationary_std, tau)` 描述一个 stationary OU process。
- `BackgroundConfig(enabled, exc, inh, interpolation)` 分别配置 E/I background。
- `BackgroundTrace(time, exc, inh, interpolation)` 是 solver 消费的 time-major trace。

入口：

```python
trace = generate_background_trace(
    cfg,
    n_exc=layout.n_exc,
    n_inh=layout.n_inh,
    n_batch=n_batch,
    time=time,
)
```

重要 shape：

```text
trace.time.shape == (n_time,)
trace.exc.shape == (n_time, n_batch, n_exc)
trace.inh.shape == (n_time, n_batch, n_inh)
```

`trace.value_at(t)` 支持 `linear` 和 `sample_hold` interpolation。`trace.rk4_samples()` 预采样 left/mid/right stage，供 RK4 solver 避免每个 stage 重复插值。

## 随机性约定

输入层不创建局部 RNG，也没有 `seed` 字段。当前随机路径只有：

- `NaturalImageSampler`：使用 `np.random.permutation` 和 `np.random.randint`。
- `generate_ou_background`：使用 `np.random.standard_normal`。

复现边界是 workflow/main 入口统一设置全局 seed。不要在这些底层函数中重新引入 `np.random.default_rng(...)` 或 `seed` 参数。

## 保留的检查

输入层不做大规模配置防御，只保留容易造成 silent wrong result 的检查：

- Gabor `sigma/gamma` 必须为正。
- visual grid size/resolution 合法。
- L4 tuning 数组必须存在且长度匹配。
- natural-image frame 必须是 2D；batch projection 的 frame shape 必须一致。
- background time grid 必须有限且严格递增。
- background trace shape 必须与 time/batch/population 维度匹配。
