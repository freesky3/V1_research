# Dynamics 层说明

本文档说明当前 `v1_research.dynamics` 的逻辑。Dynamics 层只负责 Wilson-Cowan 方程、transfer table 和 rate solver；它不构建 stimulus、不更新权重，也不调度 workflow。

## 推荐阅读顺序

1. `src/v1_research/dynamics/wilson_cowan.py`：共享 Wilson-Cowan 方程。NumPy 和 JAX 后端都复用这里的 RHS 逻辑。
2. `src/v1_research/dynamics/solvers.py`：统一 solver 配置、结果对象、后端工厂和 `solve_rates(...)` 入口。
3. `src/v1_research/dynamics/scipy_solver.py`：SciPy 调试/数值对照路径，包含固定步长 RK4 和 `solve_ivp` 方法。
4. `src/v1_research/dynamics/jax_rk4.py`：JAX RK4 性能路径。
5. `src/v1_research/dynamics/transfer.py`：Siegert transfer table。

## 当前入口

```python
from v1_research.dynamics import SolverConfig, solve_rates

result = solve_rates(
    model,
    drive=l4_drive,
    time=time,
    n_batch=n_batch,
    cfg=SolverConfig(backend="jax-rk4"),
    background_trace=background_trace,
)
```

`drive(t)` 返回 L4 外部输入，shape 为 `(n_input, n_batch)`。当 `n_batch == 1` 时，也可以返回 `(n_input,)`。

## 数据流

```text
ModelState
-> WilsonCowanEquation
   -> 按 E/I/L4 source 拆分 weights
   -> 填入动态 L2/3 rates 和外部 L4 drive
   -> 计算 mu = weights @ sources
   -> 应用 E/I transfer functions
   -> 返回 dy/dt
-> ScipySolver 或 JaxRK4Solver
-> RateResult
```

Solver 后端不复制 Wilson-Cowan 方程逻辑。SciPy 后端调用 `WilsonCowanEquation.rhs(...)`；JAX 后端调用同一模块里的 `jax_wilson_cowan_rhs(...)`，只负责 RK4 scan 和 device array 准备。

## Shape 约定

- 动态 state：`(n_l23, n_batch)`。
- L4 drive：`(n_input, n_batch)`。
- 完整 solver trajectory：`(n_time, n_l23, n_batch)`。
- 对外 E/I trajectory：`(n_time, n_batch, n_exc)` 和 `(n_time, n_batch, n_inh)`。
- 对外 E/I summary：`(n_batch, n_exc)` 和 `(n_batch, n_inh)`，当前取最后三分之一时间点均值。
- Background trace 复用 `inputs.background.BackgroundTrace`，原始 shape 是 `(n_time, n_batch, n_units)`。

## Backend

- `ScipySolver(method="RK4")`：固定步长 RK4，适合调试、数值对照和小网络测试。
- `ScipySolver(method="RK45")` 或其他 `solve_ivp` 方法：可读的 SciPy adaptive 路径。
- `JaxRK4Solver(dtype="float64")`：主性能路径。JAX 仍是 optional extra；只有选择该 backend 时才要求安装。

`SolverConfig` 保持很窄：

```python
SolverConfig(
    backend="jax-rk4",
    scipy_method="RK4",
    store_trajectory=True,
    jax_dtype="float64",
    transfer=TransferConfig(),
)
```

没有迁移旧项目里的 RootConfig bridge、Diffrax backend、early-stop、diagnostics 或 training_bcm fallback。后续如果确实需要 steady-state stop，应先在 workflow/learning 场景里明确科学需求，再加最小实现。

## Transfer

`TransferConfig` 位于 `dynamics/transfer.py`，因为这些字段只由 transfer table 解释。字段使用新名字：

- `tau_exc` / `tau_inh`
- `refractory_tau`
- `threshold`
- `reset_potential`
- `mu_table_max`
- `rate_max`

不保留旧 `kind`、`tau_e`、`tau_i`、`tau_rp`、`theta`、`v_r` 等兼容字段。

## 随机性约定

Dynamics 层不创建 RNG，也没有 `seed` 参数。Solver 对固定 model、drive、time grid、transfer table 和 background trace 是确定性的；随机性属于 model/input/workflow 层，由主程序统一设置全局 seed。

## 保留的检查

只保留容易造成 silent wrong result 的数学检查：

- time grid 必须有限且严格递增。
- weights shape 必须匹配 `ModelState.layout.shape`。
- state、drive、background shape 必须匹配 batch 和 E/I/L4 数量。
- transfer table 的 `mu` 必须严格递增，且与 `rate` shape 一致。
- tau、sigma、refractory time、table range 等数学参数必须为正。
