# Model 层说明

本文档说明当前 `v1_research.model` 和 `v1_research.data` 的逻辑。它描述现在代码如何工作，不记录重构流水账。

## 推荐阅读顺序

1. `src/v1_research/model/build.py`：从配置和实验数据构建完整 `ModelState` 的入口。
2. `src/v1_research/data/experimental.py`：读取 `data/sample_data.pkl`，得到经验比例和权重样本。
3. `src/v1_research/model/geometry.py`：构建 L4、L2/3 sheet 几何和细胞标签。
4. `src/v1_research/model/connectivity.py`：从经验连接比例推导 block probability，并采样连接拓扑。
5. `src/v1_research/model/weights.py`：在拓扑上采样带符号权重。
6. `src/v1_research/model/state.py`：保存最终 layout、connection mask 和 weights。

## 当前入口

模型层的主入口是：

```python
from v1_research.data import ExperimentalData
from v1_research.model import ModelConfig, build_model

empirical = ExperimentalData.from_path("data/sample_data.pkl")
model = build_model(ModelConfig(), empirical)
```

随机性不在 model 层单独传 `seed` 或 `rng`。主程序应在进入工作流前统一设置全局随机种子，例如：

```python
set_global_seed(CONFIG["seed"])
model = build_model(cfg.model, empirical)
```

当前 model 层用到的是 `np.random` 全局状态。以后如果引入 Torch/JAX 路径，也应在 workflow 入口统一设置 seed，而不是在底层函数里分散创建局部 RNG。

## 数据流

```text
ExperimentalData.from_path(...)
-> ModelConfig
-> build_population_layout(...)
   -> SheetGeometry(L4)
   -> derive_population_counts(...)
   -> SheetGeometry(L2/3)
   -> assign_l23_cell_types(...)
   -> assign_l4_tuning(...)
-> build_model_spec(...)
   -> derive_connection_probabilities(...)
   -> ConnectivityConfig
   -> WeightConfig
-> sample_connectivity(...)
-> sample_weights(...)
-> ModelState
```

`build_model(...)` 只做组装，不解释 stimulus、dynamics、training 或 analysis 逻辑。

## 经验数据

`ExperimentalData.from_path(...)` 读取 `data/sample_data.pkl`。当前期望的关键字段包括：

- population ratio：`eta_I`、`eta_X`、`etaT_E`、`etaT_X`
- connection ratio：`gamma_EE`、`gamma_EI`、`gamma_EX`、`gamma_IE`、`gamma_II`、`gamma_IX`、`chi`
- weight samples：`sampled_J_EE`、`sampled_J_EI`、`sampled_J_EX`、`sampled_J_IE`、`sampled_J_II`、`sampled_J_IX`

读取后代码暴露 snake_case 字段，例如 `eta_i`、`eta_x`、`gamma_ee`、`eta_t_x`。旧字段名只在加载 sample data 时出现，不作为 model 层 API。

`derive_population_counts(...)` 根据 L4 输入数量和经验比例得到：

- `n_input`：L4 cell 数量。
- `l23_n_side`：L2/3 sheet 边长。若 `L23Config.n_side` 未指定，则由 `eta_x` 和 `eta_i` 推导。
- `n_exc` / `n_inh`：L2/3 兴奋/抑制细胞数量。

## 几何和 layout

`L4Config` 和 `L23Config` 位于 `model/geometry.py`，因为它们的字段由几何构建逻辑解释。

`SheetGeometry` 表示一个二维规则方格 sheet：

- `n_cells = n_side * n_side`
- `coords.shape == (n_cells, 2)`
- 坐标以 sheet 中心为原点。
- `distance_matrix(periodic=True)` 计算同层距离。
- `distance_to(other, periodic=True)` 计算跨层距离，并包含 z 方向距离。

`PopulationLayout` 位于 `model/state.py`，它组合几何和标签：

- `l23`：L2/3 sheet。
- `l4`：L4 input sheet。
- `l23_cell_types`：长度为 `l23.n_cells`，取值为 `"E"` 或 `"I"`。
- `l4_tuning_labels`：长度为 `l4.n_cells`，取值为 `"T"` 或 `"U"`。
- `l4_preferred_orientations`：长度为 `l4.n_cells`，未调谐细胞为 `NaN`。

常用索引：

- `layout.exc_idx`：L2/3 excitatory cell 行/列索引。
- `layout.inh_idx`：L2/3 inhibitory cell 行/列索引。
- `layout.input_idx`：L4 input 在矩阵 source 轴上的列索引。

## 矩阵 shape

模型矩阵只对 L2/3 细胞求动态；L4 是外部输入 source。因此：

```text
rows = L2/3 targets = n_exc + n_inh
cols = L2/3 recurrent sources + L4 input sources
shape = (layout.l23.n_cells, layout.l23.n_cells + layout.l4.n_cells)
```

`ModelState` 保存：

- `layout: PopulationLayout`
- `connection_mask: scipy.sparse.csr_matrix[bool]`
- `weights: scipy.sparse.csr_matrix[float]` 或 dense array

`ModelState.shape` 等于 `layout.shape`。

## 连接概率

`ConnectivityConfig` 位于 `model/connectivity.py`。block 命名始终是 target/source：

- `ee`：target E receives from source E。
- `ei`：target E receives from source I。
- `ex`：target E receives from source L4 input。
- `ie`、`ii`、`ix` 同理。

`derive_connection_probabilities(...)` 先用 `p_ee` 和经验比例推导每个 block 的目标连接概率。`probability_matrix(...)` 再把空间 kernel 缩放到这些目标概率。

空间 kernel 是 narrow/broad Gaussian 混合：

```text
score = kappa * narrow(distance) + (1 - kappa) * broad(distance)
```

`probability_block(...)` 默认做 row equalization，使每个 target row 在 valid source 上的平均概率接近该 block 的目标概率。同类 recurrent block 会去掉 self-connection。

## 权重

`WeightConfig` 位于 `model/weights.py`：

- `base_strength`：全局兴奋性权重尺度。
- `inhibitory_ratio`：抑制性 source 相对兴奋性 source 的强度倍数。
- `BlockWeightScales`：`ee`、`ei`、`ex`、`ie`、`ii`、`ix` 的 block 级缩放。

`sample_weights(...)` 在 `connection_mask` 指定的非零拓扑上采样经验权重样本：

- source E 和 source X 生成非负权重。
- source I 生成非正权重。
- 输出 CSR sparse matrix。

训练阶段若需要 dense 权重、row sum 限制或索引检查，可以复用 `as_dense_weights(...)`、`limit_row_sums(...)`、`validate_indices(...)` 等小工具。

## 关于校验

当前代码避免在配置 dataclass 中堆防御式 `__post_init__`。例如 `L4Config`、`L23Config`、`WeightConfig`、`ModelConfig` 不做一层层字段正负检查。

仍保留的检查主要属于数学或矩阵不变量：

- `PopulationLayout` 的标签长度和值域。
- `ModelState` 的 `connection_mask` / `weights` shape。
- 概率必须在 `[0, 1]`。
- spatial kernel 的 sigma 必须正且 `sigma_narrow < sigma_broad`。
- row equalization 中不能对 zero-score row 分配正概率。

这些检查的目的不是兼容或防呆，而是避免产生 silent wrong numerical result。

## 测试位置

当前 model 层测试在：

- `tests/test_model_geometry.py`
- `tests/test_model_connectivity.py`
- `tests/test_model_build.py`

覆盖内容包括几何坐标、周期距离、连接概率 row target、seed 复现、矩阵 shape 和权重符号。
