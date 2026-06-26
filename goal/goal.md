## 目标：完成 `v1_research` 的 BCM 训练、DG/frames 分析、OU 对照与最终收口

这个文件只描述 AI 应如何执行实验、判断结果和记录证据。运行经验、参数建议和历史日志不要继续堆在这里，写入 `notebooks/`。

### 先读 Notebook

每次继续实验前，先读取以下文件中的最新结论：

* 当前建议与经验索引：`notebooks/CURRENT_EXPERIMENT_GUIDANCE_20260626.md`
* 严格审计与阶段性解释：`notebooks/STRICT_AUDIT_20260626.md`
* 训练参数日志：`notebooks/strict_training_audit_log.csv`
* DG 参数日志：`notebooks/DG_parameter_search_log.csv`
* OU/background 日志：`notebooks/OU_parameter_search_log.csv`
* frames/direction-display 日志：`notebooks/frames_parameter_search_log.csv`
* 可疑问题记录：`notebooks/problem.md`

如果 notebook 与本文件冲突，以本文件的实验协议和硬性约束为准；以 notebook 的最新参数建议和历史结果为准。

### 当前执行原则

* 不要仅凭 shared-nvme 中某个历史 run 看起来不错就停止。
* 不要把单 seed、medium-scale 或 fixed-checkpoint 结果写成最终成功。
* 不要直接进入 full all-image 5 epoch，除非 notebook 中的 medium validation、DG diagnostic 和必要的 OU/background diagnostic 都已通过。
* 搜索阶段优先固定 seed，例如 `seed=42`；final 复现阶段再做多 seed，例如 `[42, 43, 44, 45]`。
* 每轮只改少数参数，保持 run 名称可解释，并写入 notebook/CSV。

### 硬性约束

* **不改变**核心训练、模拟和分析算法，除非用户明确批准。
* 允许修改：配置文件、命令行覆盖、诊断绘图、日志记录脚本、summary/报告辅助脚本。
* 训练和模拟优先使用 `solver.backend=jax-rk4`；JAX RK4 运行时使用 `JAX_ENABLE_X64=1`。
* 搜索 run 写入 `/root/shared-nvme/v1_research_runs`，不要占满 root FS。
* **不要覆盖**已有运行目录；使用有描述性的任务名。
* 遇到疑似代码或方法问题，写入 `notebooks/problem.md`：代码位置、疑似错误原因、建议修改方案。若会直接影响结果，只记录并标注结果不可用；未经批准不要改核心算法。

最终训练候选必须满足：

* 全量自然图像（all natural images）
* `epochs=5`
* `natural_images.patches_per_image=8`
* `solver.backend=jax-rk4`
* `background.enabled=false`

### 可执行入口

* `uv run v1-simulation train --config ...`
* `uv run v1-simulation simulate --config ...`
* `uv run v1-simulation analyze --config ...`
* `uv run v1-simulation full --config ...`
* `uv run v1-simulation sweep --config ...`
* `uv run v1-simulation summarize --run ...`

必要时先读：

* `docs/parameters.md`
* `docs/workflows.md`
* `docs/analysis.md`
* `docs/direction_tuning.md`
* `docs/quickstart.md`

### 通用运行流程

1. 读取本文件和 notebook 最新状态。
2. 明确本轮只测试什么假设，以及接受/拒绝门槛。
3. 用描述性 `run_root` 运行训练、模拟或分析。
4. 运行 `uv run v1-simulation summarize --run <run_dir>`。
5. 抽取关键指标，不只看图。
6. 将参数、指标和判断追加到对应 CSV。
7. 如果结果改变搜索方向，更新 `notebooks/STRICT_AUDIT_20260626.md` 或当前 guidance notebook。
8. 明确下一步是接受、拒绝、局部继续搜索、做 DG/OU 诊断，还是升级到更大验证。

### 训练搜索协议

使用 `uv run v1-simulation train`，并尽量开启训练诊断。

可调参数优先包括：

* `learning.bcm.eta`
* `learning.bcm.theta_init`
* `learning.bcm.theta_y0`
* `learning.bcm.row_sum_max_scale`
* `solver.transfer.rate_max`
* `natural_images.preprocess.frame_scale`
* `natural_images.preprocess.frame_offset`
* `model.p_ee`
* `model.weight.scales.ee`
* `model.weight.scales.ei`

训练判断必须检查：

* E/I median、p95、max rate
* E/I neuron fraction `>0.1 Hz`、`>0.5 Hz`、`>1 Hz`、`>2 Hz`
* E/I near-rate-cap fraction
* top1/top5 activity concentration
* row-sum cap ratio 是否持续上升
* tracked-weight relative median/max，不只看少数 max 样本
* 全局权重变化和 BCM signal/theta 轨迹

训练候选至少应满足：

* final E/I `>1 Hz` fraction 约 `>=0.6`
* tracked-weight relative median `>1e-4`
* E p95 不进入 `20+ Hz` 过热区
* near-cap fraction 低
* row cap ratio 不持续恶化

如果大部分神经元均值低于 `1 Hz`，即使 health 没有 fail，也标记为 `candidate_with_caveat` 或 reject。若 tracked weights 和全局 `W_EE_mean` / `W_IE_mean` 几乎不变，不能作为最终训练。

### DG / frames 诊断协议

使用最终候选或 medium candidate checkpoint 做 drifting-grating simulation，然后 analyze。

原则：

* **仅调优**模拟和分析参数，不要更改已训练好的网络。
* 运行 8 个方向、多个 trial；DG diagnostic 默认优先用每方向 8 个 trial。
* DG diagnostic 默认优先使用较长 trial，例如 `time.start=0.0`、`time.stop=0.8`、`time.step=0.004`，避免只看到 1 Hz grating 周期前 20%。
* 优先沿用训练线参数，谨慎调 `grating.visual_gain`；若开头 transient 主导，优先使用 `grating.visual_gain_ramp_duration=0.1` 的 smoothstep ramp，而不是手工设置 L2/3 初始 firing-rate。
* 不要把提高 DG gain 当作训练问题的替代修复。

DG/frames 判断必须检查：

* simulation health、near-cap、activity concentration
* all-time 和 tail-only 多阈值活动
* population peak/tail ratio、front/tail ratio
* selected-neuron rate median/p95 和 `>0.5/1/2 Hz` fraction
* Louvain ensemble count、classified fraction
* modulation-based direction selectivity
* amplitude-gated `effective_direction_selective`
* `effective_covered_direction_count`
* `low_amplitude_ensembles`

如果 `simulate_traces.png` 显示开头 transient 主导、尾段明显降低，必须用 tail/steady 窗口复查并记录 drift。若多个 ensemble 只偏好少数角度，或 effective coverage 明显低于原始 modulation coverage，需要降级结果。

### OU / background 对照协议

新项目没有单独 OU workflow；通过 `simulate` 中的 `background.enabled=true` 和背景参数做对照。

判断必须检查：

* simulation health、near-cap、classified fraction
* Louvain 分组是否与 DG 线数量接近
* within/between similarity 是否提示 common-mode/background 主导
* 若有 overlap 指标，优先用它量化 DG/background 对照关系

OU 对照不能只凭 ensemble 数接近就接受；必须解释 health warning、near-cap、classified fraction、overlap 和 common-mode 风险。

### 停止条件

只有满足以下条件才可收口：

* 找到当前最好的训练、DG、frames 和 OU/background 参数组合。
* 合理相邻参数已经尝试，未发现更优 tradeoff。
* all-image 5 epoch final 训练完成并通过训练门槛。
* DG/frames 分析通过 effective direction coverage 和 tail/transient 检查。
* OU/background 对照不显示明显 common-mode 或背景主导风险。
* 多 seed 全重训练复现通过；不能用 fixed checkpoint 的 simulation seed 变化代替全重训练复现。

### 最终报告要求

最终报告必须包含：

* 最终训练命令及确切 overrides。
* 最终 frames_sorted / direction-tuning 命令。
* 最终 DG 模拟与分析命令。
* 最终 OU/background 对照命令。
* 训练、DG、OU/background 及分析的运行目录。
* 关键诊断图表和 CSV/JSON summary。
* 参数尝试表：接受/拒绝原因。
* 明确指出是否仍有目标未实现。
