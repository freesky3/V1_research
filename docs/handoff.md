# Handoff 给下一个 AI

## 当前任务状态

目标项目是：`D:\skywalker\sjtu\大三下课程\Research\MouseV1\V1_simulation`。

当前已经完成第一阶段 model 地基：

- 包名统一为 `v1_research`。
- 新建 `src/v1_research/data/experimental.py`。
- 新建 `src/v1_research/model/geometry.py`、`connectivity.py`、`weights.py`、`state.py`、`build.py`。
- 新建 package wiring：`src/v1_research/__init__.py`、`cli.py`、`model/__init__.py`、`data/__init__.py`。
- `pyproject.toml` 已改为项目名 `v1-research`，console script 仍叫 `v1-simulation`，入口指向 `v1_research:main`。
- 加了 model 层测试：`tests/test_model_geometry.py`、`tests/test_model_connectivity.py`、`tests/test_model_build.py`。
- 新增当前逻辑文档：`docs/model.md`。

目标仓库目前还没有初始 commit，所以 `git status` 会显示整个项目为 untracked。这不是本轮把所有文件都改坏了，而是这个新项目本身还没提交。

## 建议先读

1. `docs/PLAN.md`：总体重构方向。
2. `docs/model.md`：当前 model/data 层逻辑说明。
3. `src/v1_research/model/build.py`：当前 model 构建入口。
4. `tests/test_model_build.py`：seed、shape、权重符号的行为约束。

## 建议使用的 skills

- `refactor-v1-research-code`：继续迁移 research code 时使用。
- `document-v1-research-code`：每完成一个科学子系统后更新逻辑文档。
- `superpowers:test-driven-development`：改行为前先写测试。
- `superpowers:systematic-debugging`：测试或构建失败时先定位根因。
- `superpowers:verification-before-completion`：最终回答前跑验证命令。

## 当前 model 层关键设计

主入口：

```python
from v1_research.data import ExperimentalData
from v1_research.model import ModelConfig, build_model

empirical = ExperimentalData.from_path("data/sample_data.pkl")
model = build_model(ModelConfig(), empirical)
```

随机性设计：

- model 层不再接 `seed`、`rng`、`rngs`、`np.random.default_rng(...)`。
- 注意删掉配置字段那种工程化防御，只保留少数数学上容易静默出错的检查。
- 统一由上层 workflow / main 调 `set_seed(CONFIG["seed"])`。
- 当前 model 层随机调用均走全局 `np.random`：L4 tuning、随机 inhibitory placement、connectivity sampling、weight sampling。

校验取舍：

- 已删除 `L4Config`、`L23Config`、`WeightConfig`、`ModelConfig` 里偏防御式的字段正负检查。
- 仍保留数学/矩阵不变量检查：probability range、kernel sigma 关系、label shape/value、matrix shape、row equalization zero-score 检查。
- 用户明确觉得配置 dataclass 的 `__post_init__` 太啰嗦，不利于简洁性；后续不要重新加回“大型 validation framework”。

## 已踩过的坑和解决方式

1. `uv run pytest` 一开始找不到 pytest。
   - 解决：在 `pyproject.toml` 加了 `[dependency-groups] dev = ["pytest>=9.0.2"]`。

2. 改包名后 `uv_build` 仍期待 `src/v1_simulation/__init__.py`。
   - 根因：`[project].name = "v1-simulation"` 会映射到 `v1_simulation`。
   - 解决：项目名改成 `v1-research`，包目录为 `src/v1_research`。

3. PowerShell 5 的 `Set-Content -Encoding UTF8` 会给 `pyproject.toml` 写入 BOM，导致 pytest/TOML 报 `Invalid statement (at line 1, column 1)`。
   - 解决：用 `.NET` 的 `System.Text.UTF8Encoding($false)` 无 BOM 写回。之后尽量避免用 BOM 写 TOML。

4. 目标项目在当前 workspace root 外。
   - 需要工具写入时用 `sandbox_permissions="require_escalated"`。
   - 不要误改旧项目 `train_simulation_analysis/src/v1_simulation/network`；它只是迁移参考。

5. `src/V1_simulation/data` 曾在早期检查中出现过，但后来确认 `V1_simulation/src` 是空目录。
   - 当前实际源码全部在 `src/v1_research`。

6. 一次机械替换把 `weights.py` 写进了字面量 `` `r`n``。
   - 已通过整文件重写修复。后续做机械替换后务必读回关键文件。

## 已验证命令

最近一次相关验证：

```powershell
cd D:\skywalker\sjtu\大三下课程\Research\MouseV1\V1_simulation
uv run pytest
uv run python -m compileall src tests
```

结果：

- `uv run pytest`：10 passed。
- `compileall`：通过。

## 后续建议

下一步可以继续按 `docs/PLAN.md` 的顺序迁移：

1. `inputs/`：grating、Gabor receptive fields、natural image L4 drive。
2. `dynamics/`：Wilson-Cowan 方程和 SciPy/JAX solver。
3. `learning/`：BCM rule 和后续可切换 learning rule。
4. `workflows/`：train/simulate/analyze 的薄调度层。

迁移时不要复制旧兼容 API。每个子系统完成后都在 `docs/` 下按逻辑补文档，不要按日期写 refactor diary。
## Inputs/cache 层迁移更新

本轮新增：

- `src/v1_research/inputs/gabor.py`：Gabor RF、visual grid、`L4GaborBank`。
- `src/v1_research/inputs/grating.py`：解析 drifting-grating L4 drive。
- `src/v1_research/inputs/natural_images.py`：Van Hateren dataset、crop sampler、preprocessor、L4 projector、static natural-image drive。
- `src/v1_research/inputs/background.py`：OU background trace 和 RK4 stage samples。
- `src/v1_research/cache/keys.py`、`storage.py`、`projections.py`：projection matrix 与 natural-image rates cache。
- `docs/inputs.md`：输入层逻辑说明。

随机性约定继续保持：底层不接 `seed`，不创建局部 RNG。natural-image sampler 和 background 都走全局 `np.random`，由 workflow/main 在入口统一设置 seed。

迁移时删掉了旧项目的兼容探测：输入层直接接收 `PopulationLayout`，使用 `layout.l4.coords`、`layout.l4_tuning_labels`、`layout.l4_preferred_orientations`。不要再加回旧 `L4.N/tunings/pref_dirs` 适配。
## 文档整理与提交交接

本轮补充了输入/cache 层的逻辑文档，并准备项目首次 git commit。

已完成：

- 重写 `docs/inputs.md`，按当前代码逻辑说明 Gabor RF、drifting grating、natural images、background 的入口、数据流、局部 config 和关键 shape。
- 新建 `docs/cache.md`，说明 projection matrix cache、natural-image rates cache、cache key 内容、磁盘布局和训练前预热方式。
- 更新 `.gitignore`，忽略 `.pytest/`、`.pytest_cache/`、环境变量文件、run/output/artifact 目录、外部 Van Hateren `.iml` 数据和 projection cache 目录。
- 保留 `data/sample_data.pkl` 作为可提交样本数据，因为现有 model tests 依赖它。

遇到的坑和解决方式：

- 目标 repo 位于当前 workspace root 外，读写和 git 操作需要 `sandbox_permissions="require_escalated"`。
- `apply_patch` 和普通 `Get-Content` 有时会在中文路径/中文文档上触发 Windows sandbox `CryptUnprotectData failed`。解决方式是使用受控提升后的 PowerShell，并用 `[System.Text.UTF8Encoding]::new($false)` 写 UTF-8 no BOM 文本。
- 目标 repo 当前没有历史提交，`git log` 会报 `current branch 'master' does not have any commits yet`。这是正常初始状态，不代表工作区损坏。
- `data/vanhateren_iml` 是外部自然图像数据目录，应继续忽略；不要把真实 `.iml` 数据提交进 repo。

后续注意：

- 输入/cache 层仍然遵守全局 seed 约定；不要在底层重新加入 `seed` 字段或 `np.random.default_rng(...)`。
- 如果后续实现 workflow，cache 路径应由 workflow 显式传入，例如 `data/.projection_cache/`，不要隐藏在训练循环内部。
# Dynamics 层迁移交接

本轮新增 `src/v1_research/dynamics/`，把 Wilson-Cowan 方程从 solver 后端中独立出来，并接上两个后端：

- `transfer.py`：Siegert transfer table，本地 `TransferConfig` 字段为 `tau_exc`、`tau_inh`、`refractory_tau`、`threshold`、`reset_potential`、`mu_table_max`、`rate_max`。没有保留旧 `tau_e/tau_i/tau_rp/theta/v_r/kind` 兼容字段。
- `wilson_cowan.py`：独立 Wilson-Cowan 方程。`WilsonCowanEquation` 负责 NumPy RHS、shape 检查、weight block 拆分；`jax_wilson_cowan_rhs` 是 JAX 后端复用的同一方程逻辑。
- `scipy_solver.py`：`ScipySolver`，包含固定步长 RK4 和 `solve_ivp` 调试路径。
- `jax_rk4.py`：`JaxRK4Solver`，JAX 仍是 optional extra，只有选择该 backend 时才要求安装。
- `solvers.py`：`SolverConfig`、`RateResult`、`RateSolver`、`make_solver(...)`、`solve_rates(...)`。
- `docs/dynamics.md`：当前 dynamics 层逻辑文档。

关键约定：

- solver 不复制 Wilson-Cowan 方程逻辑；SciPy/JAX 都复用 `dynamics/wilson_cowan.py`。
- dynamic state shape 为 `(n_l23, n_batch)`，drive shape 为 `(n_input, n_batch)`，public E/I trajectory shape 为 `(n_time, n_batch, n_exc/n_inh)`。
- dynamics 层不创建局部 RNG，也不接 seed。随机性继续放在 workflow/model/input 入口，由主程序统一设置全局 seed。
- 没有迁移旧 early-stop、diagnostics、Diffrax 或 training_bcm fallback；后续若需要，应按具体 workflow 科学需求重新加最小实现。

新增测试：

- `tests/test_dynamics_transfer.py`
- `tests/test_dynamics_wilson_cowan.py`
- `tests/test_dynamics_solvers.py`

遇到的坑和解决方式：

- 目标仓库 `.gitignore` 忽略了整个 `docs/`，所以 `docs/dynamics.md` 和 `docs/handoff.md` 需要用 `git add -f` 显式纳入提交。
- 当前环境没有安装 JAX；JAX/SciPy 数值对照测试用 `pytest.skip(...)` 显式跳过。安装 optional JAX extra 后，这个测试会实际比较 `JaxRK4Solver` 与 `ScipySolver(method="RK4")`。
- `compileall` 会在 `src/` 和 `tests/` 下生成 `__pycache__`，提交前已清理；后续验证后也要留意清理生成缓存。

# Learning 层迁移交接

本轮新增 `src/v1_research/learning/`，把 BCM 从旧项目 trainer 风格逻辑里拆成一个可切换的 `LearningRule` 实现，并新增一个薄训练 workflow helper：

- `learning/rules.py`：`RateBatch`、`LearningUpdate`、`LearningRule` 协议。`RateBatch` 使用 batch-first shape：`exc=(n_batch, n_exc)`、`inh=(n_batch, n_inh)`、`external=(n_batch, n_input)`。
- `learning/bcm.py`：`BCMConfig`、`BCMState`、`BCMRowSumLimits`、`BCMLearningRule` 和 BCM 纯计算函数。
- `learning/config.py`：`LearningConfig(kind="bcm")` 和 `make_learning_rule(...)`。当前只用显式 `if cfg.kind == "bcm"`，不要引入 registry。
- `workflows/train.py`：`apply_learning_rule(...)` 和 `solve_and_learn_batch(...)`。这里不出现 BCM 公式，只依赖 `LearningRule` 协议。
- `docs/learning.md`：当前 learning 层逻辑文档。

关键约定：

- BCM 只更新 excitatory source recurrent efferents：`E <- E` 和 `I <- E`。不更新 inhibitory source block，也不更新 L4 input block。
- `BCMState` 保存 theta 和初始化时从模型权重得到的 row-sum caps；`BCMLearningRule` 本身不保存训练状态，避免同一个 rule 对象复用时串模型状态。
- 本轮没有迁移旧 `JAXBCMUpdater`、bad-batch 过滤、steady-state 诊断、checkpoint、CSV log 或完整 natural-image training loop。
- Learning 层不创建局部 RNG，也没有 seed 字段。给定 model 和 rates 后 BCM 是确定性的。

新增测试：

- `tests/test_learning_bcm.py`：theta 初始化、pre/post theta update、plastic block 选择、topology 外保持 0、`w_max`、row-sum caps、factory。
- `tests/test_workflows_train.py`：用 fake learning rule 验证 workflow 只依赖协议，首次 batch 只初始化，后续 batch 替换 model/state。

遇到的坑和解决方式：

- 一开始 `BCMLearningRule` 内部缓存了 row-sum caps，后来用测试发现同一个 rule 对象如果初始化两个不同 model，会混用第二个模型的 caps。已改为把 caps 放进 `BCMState`。
- 测试里手算 inhibitory theta 第二列时曾算错，targeted test 暴露后按 `mean(rates ** 2)` 公式修正了期望。
- 当前 `.gitignore` 忽略了整个 `docs/`，所以 `docs/learning.md` 和更新后的 `docs/handoff.md` 提交时需要 `git add -f`。
- `compileall` 仍会生成 `__pycache__`；提交前已清理，后续也要注意。

最近一次验证：

```powershell
uv run pytest tests/test_learning_bcm.py tests/test_workflows_train.py -q
uv run pytest
uv run python -m compileall src tests
```

结果：

- targeted learning/workflow tests：9 passed。
- 全量 pytest：37 passed, 1 skipped。
- compileall：通过。

后续建议：

- 下一步如果迁移完整训练 workflow，让自然图像 batch、solver、artifact 保存都在 `workflows/` 或 run bundle 层组装；不要把 BCM 公式写回 trainer。
- 如果要加入 `JaxBCMLearningRule`，实现同一个 `LearningRule` 协议，并只扩展 `make_learning_rule(...)` 分支。
- 后续 Oja/Hebbian 规则可以复用 `RateBatch.external`，但 workflow 不应该知道某个规则是否使用 external rates。
