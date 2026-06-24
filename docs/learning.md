# Learning 层说明

本文档说明当前 `v1_research.learning` 和薄训练 workflow 的逻辑。Learning 层只负责学习规则本身；它不构建输入、不求解动力学、不保存 artifact，也不决定训练 epoch 或 batch schedule。

## 推荐阅读顺序

1. `src/v1_research/learning/rules.py`：学习规则协议、`RateBatch` 和 `LearningUpdate`。
2. `src/v1_research/learning/bcm.py`：BCM sliding threshold 和 excitatory-source efferent weight update。
3. `src/v1_research/learning/config.py`：显式 learning-rule factory。
4. `src/v1_research/workflows/train.py`：只依赖 `LearningRule` 协议的 batch helper。

## 当前入口

```python
from v1_research.learning import LearningConfig, make_learning_rule
from v1_research.workflows import apply_learning_rule

rule = make_learning_rule(LearningConfig(kind="bcm"))
update = apply_learning_rule(model, rates, rule, state=None)
```

首次 batch 只初始化 rule state，`updated=False`，不修改权重。后续 batch 调用 `rule.step(...)`，workflow 只消费 `LearningUpdate(model, state, updated)`。

## 数据流

```text
RateResult from dynamics
-> RateBatch(exc, inh, external)
-> LearningRule.initialize(...) or LearningRule.step(...)
-> LearningUpdate
-> workflow keeps next model/state
```

`RateBatch` 使用 batch-first shape：

```text
exc:      (n_batch, n_exc)
inh:      (n_batch, n_inh)
external: (n_batch, n_input) or None
```

BCM 当前只使用 `exc` 和 `inh`。`external` 留给后续 Oja、Hebbian 或 feedforward learning rule；workflow 不需要知道具体规则是否使用它。

## LearningRule 协议

```python
class LearningRule(Protocol):
    def initialize(self, model: ModelState, rates: RateBatch) -> LearningState: ...
    def step(self, model: ModelState, rates: RateBatch, state: LearningState) -> LearningUpdate: ...
    def stats(self, model: ModelState, state: LearningState) -> dict[str, float]: ...
```

这个协议是训练 workflow 的唯一依赖。新增规则时实现同一协议，并在 `make_learning_rule(...)` 加一个显式分支即可；不要引入 registry、entrypoint 或旧配置兼容层。

## BCM 规则

局部配置在 `learning/bcm.py`：

```python
BCMConfig(
    eta=3.0e-5,
    theta_beta=0.01,
    theta_eps=1.0e-6,
    theta_y0=20.0,
    theta_init=1.0,
    theta_floor=1.0e-3,
    theta_update_order="pre",
    w_max=None,
    row_sum_max_scale=1.0,
)
```

BCM state 保存 E/I target populations 的 sliding thresholds，以及初始化时从模型权重得到的 row-sum caps：

```text
theta_exc: (n_exc,)
theta_inh: (n_inh,)
row_sum_limits.target_exc_source_exc: (n_exc,) or None
row_sum_limits.target_inh_source_exc: (n_inh,) or None
```

阈值初始化：

```text
theta = mean(rates ** 2, axis=batch) / theta_y0
```

若 `theta_init` 不是 `None`，则使用常数初始化。阈值更新：

```text
theta_next = (1 - theta_beta) * theta + theta_beta * mean(rates ** 2) / theta_y0
theta_next = max(theta_next, max(theta_eps, theta_floor))
```

权重更新：

```text
gain = eta * y * (y - max(theta, theta_eps))
delta = gain.T @ x / n_batch
```

矩阵约定仍是 `weights[target, source]`。当前 BCM 只更新 excitatory source 的 recurrent efferents：

- `E <- E`
- `I <- E`

`E <- I`、`I <- I` 和所有 L4 input source block 不由 BCM 修改。更新只写入已有 connection topology；topology 外权重强制保持 0。更新后权重下限为 0，可选 `w_max` 上限和基于初始 row sum 的 `row_sum_max_scale`。

`BCMLearningRule` 本身不保存训练状态。row-sum caps 跟随 `BCMState`，因此同一个 rule 对象可以在不同 model/state 上复用，不会混用别的模型初始化得到的 row-sum limit。

## Workflow 边界

`src/v1_research/workflows/train.py` 当前只提供两个小 helper：

- `apply_learning_rule(...)`：初始化或执行一个 rule step。
- `solve_and_learn_batch(...)`：用 solver 得到 rates，组装 `RateBatch`，再调用 `apply_learning_rule(...)`。

完整 natural-image train workflow、run bundle、checkpoint、CSV log 和 CLI 还没有迁移。后续迁移时应继续让 workflow 只依赖 `LearningRule`，不要重新把 BCM 公式写进 trainer。

## 随机性约定

Learning 层不创建局部 RNG，也没有 `seed` 字段。BCM 对给定 model 和 rates 是确定性的。复现边界仍由主程序在进入 workflow 前统一设置全局 seed。

## 保留的检查

Learning 层不做旧项目那种大规模配置防御，只保留会导致 silent wrong result 的数学检查：

- rates 必须是有限的 batch-first 2D matrix。
- E/I/external rate width 必须匹配 `ModelState.layout`。
- theta shape 必须匹配目标 population。
- topology shape 必须匹配 weight block。
- `theta_beta`、`theta_eps`、`theta_y0` 等 BCM 数学参数必须落在公式有意义的范围内。
