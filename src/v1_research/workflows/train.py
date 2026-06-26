"""Natural-image training workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any
import sys

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from v1_research.cache import NaturalImageProjectionCache
from v1_research.data import ExperimentalData
from v1_research.dynamics import ExternalDrive, RateResult, SolverConfig, solve_rates
from v1_research.inputs.background import BackgroundConfig, BackgroundTrace, generate_background_trace, validate_time_grid
from v1_research.inputs.gabor import ReceptiveFieldConfig
from v1_research.inputs.natural_images import (
    L4NaturalImageProjector,
    NaturalImageDriveConfig,
    NaturalImageL4Drive,
    NaturalImagePreprocessConfig,
    NaturalImagePreprocessor,
    NaturalImageSampler,
    VanHaterenImageDataset,
)
from v1_research.learning import LearningConfig, LearningRule, LearningUpdate, RateBatch, make_learning_rule
from v1_research.learning.bcm import BCMState
from v1_research.learning.diagnostics import (
    TrainingHealthConfig,
    TrackedWeight,
    active_rate_stats,
    bcm_signal_stats,
    evaluate_training_health,
    extended_active_rate_stats,
    extended_plastic_weight_stats,
    plastic_weight_stats,
    record_tracked_weights,
    row_sum_pressure,
    sample_tracked_weights,
    theta_distribution_stats,
    theta_stats,
    weight_delta_stats,
)
from v1_research.model import ModelConfig, ModelState, build_model
from v1_research.model.weights import as_dense_weights
from v1_research.runs import (
    create_run_dir,
    model_summary,
    relative_output_path,
    save_model_state,
    write_config,
    write_csv_rows,
    write_json,
    write_manifest,
)
from v1_research.seed import set_global_seed

SolverCallable = Callable[..., RateResult]


@dataclass(frozen=True, slots=True)
class NaturalImageWorkflowConfig:
    """Natural-image input configuration owned by the training workflow."""

    image_dir: str | Path = Path("data/vanhateren_iml")
    image_shape: tuple[int, int] = (1024, 1536)
    pattern: str = "*.iml"
    crop_size: int | None = None
    patches_per_image: int = 1
    limit: int | None = None
    receptive_field: ReceptiveFieldConfig = field(default_factory=ReceptiveFieldConfig)
    preprocess: NaturalImagePreprocessConfig = field(default_factory=NaturalImagePreprocessConfig)
    drive: NaturalImageDriveConfig = field(default_factory=NaturalImageDriveConfig)
    cache_dir: str | Path | None = None


@dataclass(frozen=True, slots=True)
class TrainingSteadyStateConfig:
    """Optional within-trial steady-state diagnostics for training probes."""

    enabled: bool = False
    tail_fraction: float = 1.0 / 3.0
    stability_window_fraction: float = 0.25
    sample_neuron_count: int = 8
    batch_sample_index: int = 0
    save_arrays: bool = True
    max_plotted_probes: int = 12


@dataclass(frozen=True, slots=True)
class TrainingInspectionConfig:
    """Optional intermediate diagnostics for training runs."""

    enabled: bool = False
    probe_every: int = 1
    health: TrainingHealthConfig = field(default_factory=TrainingHealthConfig)
    tracked_weight_count: int = 0
    save_plots: bool = False
    save_per_batch_arrays: bool = False
    active_rate_threshold: float = 1.0
    steady_state: TrainingSteadyStateConfig = field(default_factory=TrainingSteadyStateConfig)


@dataclass(frozen=True, slots=True)
class TrainingWorkflowConfig:
    """Top-level configuration for natural-image learning runs."""

    seed: int | None = None
    run_root: str | Path = Path("runs")
    empirical_data_path: str | Path = Path("data/sample_data.pkl")
    model: ModelConfig = field(default_factory=ModelConfig)
    natural_images: NaturalImageWorkflowConfig = field(default_factory=NaturalImageWorkflowConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    time: NDArray[np.float64] | tuple[float, ...] = field(default_factory=lambda: np.linspace(0.0, 0.2, 51))
    batch_size: int = 8
    epochs: int = 1
    inspection: TrainingInspectionConfig = field(default_factory=TrainingInspectionConfig)


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """In-memory summary of a saved training run."""

    run_dir: Path
    model: ModelState
    learning_state: object
    summary: dict[str, Any]
    log_path: Path
    model_path: Path


@dataclass(frozen=True, slots=True)
class BatchRateSolve:
    """Rate batch plus optional trajectory-bearing solver result."""

    rates: RateBatch
    result: RateResult


@dataclass(frozen=True, slots=True)
class TrainingSteadyStateProbe:
    """Compact within-trial traces captured for one training probe."""

    step: int
    time: NDArray[np.float64]
    batch_sample_index: int
    exc_population_mean: NDArray[np.float64]
    inh_population_mean: NDArray[np.float64]
    exc_sample_indices: NDArray[np.int64]
    inh_sample_indices: NDArray[np.int64]
    exc_sample_traces: NDArray[np.float64]
    inh_sample_traces: NDArray[np.float64]


def apply_learning_rule(
    model: ModelState,
    rates: RateBatch,
    rule: LearningRule,
    state=None,
) -> LearningUpdate:
    """Initializes or applies one learning-rule step."""

    if state is None:
        return LearningUpdate(model=model, state=rule.initialize(model, rates), updated=False)
    return rule.step(model, rates, state)


def solve_and_learn_batch(
    model: ModelState,
    *,
    drive: ExternalDrive,
    time: Sequence[float] | np.ndarray,
    n_batch: int,
    solver_cfg: SolverConfig,
    rule: LearningRule,
    state=None,
    background_trace: BackgroundTrace | None = None,
    solver: SolverCallable | None = None,
) -> LearningUpdate:
    """Solves dynamics for one batch and applies the selected learning rule."""

    rates = _solve_batch_rates(
        model,
        drive=drive,
        time=time,
        n_batch=n_batch,
        solver_cfg=solver_cfg,
        background_trace=background_trace,
        solver=solver,
    )
    return apply_learning_rule(model, rates, rule, state=state)


def _solve_batch_rates(
    model: ModelState,
    *,
    drive: ExternalDrive,
    time: Sequence[float] | np.ndarray,
    n_batch: int,
    solver_cfg: SolverConfig,
    background_trace: BackgroundTrace | None = None,
    solver: SolverCallable | None = None,
) -> RateBatch:
    return _solve_batch_rate_result(
        model,
        drive=drive,
        time=time,
        n_batch=n_batch,
        solver_cfg=solver_cfg,
        background_trace=background_trace,
        solver=solver,
    ).rates


def _solve_batch_rate_result(
    model: ModelState,
    *,
    drive: ExternalDrive,
    time: Sequence[float] | np.ndarray,
    n_batch: int,
    solver_cfg: SolverConfig,
    background_trace: BackgroundTrace | None = None,
    solver: SolverCallable | None = None,
) -> BatchRateSolve:
    solver_fn = solve_rates if solver is None else solver
    time_values = np.asarray(time, dtype=float)
    result = solver_fn(
        model,
        drive=drive,
        time=time_values,
        n_batch=n_batch,
        cfg=solver_cfg,
        background_trace=background_trace,
    )
    external = np.asarray(drive(float(time_values[0])), dtype=float)
    if external.ndim == 1:
        external = external[:, np.newaxis]
    return BatchRateSolve(
        rates=RateBatch(exc=result.exc, inh=result.inh, external=external.T),
        result=result,
    )


def run_training(cfg: TrainingWorkflowConfig, *, show_progress: bool = True) -> TrainingRun:
    """Runs natural-image learning and writes a run bundle."""

    set_global_seed(cfg.seed)
    run_dir = create_run_dir(cfg.run_root, "train")
    write_config(run_dir, cfg)
    empirical = ExperimentalData.from_path(cfg.empirical_data_path)
    model = build_model(cfg.model, empirical)
    time = validate_time_grid(np.asarray(cfg.time, dtype=float), copy=True)
    drive, sampler = _build_natural_image_drive(cfg.natural_images, model)
    rule = make_learning_rule(cfg.learning)

    state = None
    tracked: list[TrackedWeight] = []
    batches = 0
    samples_seen = 0
    log_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    tracked_rows: list[dict[str, object]] = []
    per_batch_arrays: list[Path] = []
    steady_state_arrays: list[Path] = []
    steady_state_probes: list[TrainingSteadyStateProbe] = []
    live_warning_count = 0
    live_failure_count = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        samples = sampler.make_epoch(limit=cfg.natural_images.limit)
        if cfg.natural_images.cache_dir is not None:
            cached = NaturalImageProjectionCache(cfg.natural_images.cache_dir).load_or_build(drive, samples)
            drive = NaturalImageL4Drive(
                dataset=drive.dataset,
                preprocessor=drive.preprocessor,
                projector=drive.projector,
                cached_rates=cached,
            )
        for batch_index, batch in enumerate(_iter_batches(samples, int(cfg.batch_size)), start=1):
            background = generate_background_trace(
                cfg.background,
                n_exc=model.layout.n_exc,
                n_inh=model.layout.n_inh,
                n_batch=len(batch),
                time=time,
            )
            batch_drive = drive.make_static_batch_func(batch)
            next_step = batches + 1
            inspect_this_step = _should_inspect(cfg.inspection, next_step)
            steady_this_step = _should_collect_steady_state(cfg.inspection, next_step)
            solver_cfg = _trajectory_solver_config(cfg.solver) if steady_this_step else cfg.solver
            solved = _solve_batch_rate_result(
                model,
                drive=batch_drive,
                time=time,
                n_batch=len(batch),
                solver_cfg=solver_cfg,
                background_trace=background,
            )
            rates = solved.rates
            previous_model = model
            update = apply_learning_rule(model, rates, rule, state=state)
            model = update.model
            state = update.state
            batches += 1
            samples_seen += len(batch)
            log_rows.append(
                _training_log_row(
                    epoch=epoch,
                    batch=batch_index,
                    batch_size=len(batch),
                    updated=update.updated,
                    stats=rule.stats(model, state),
                )
            )
            if inspect_this_step:
                if not tracked and int(cfg.inspection.tracked_weight_count) > 0:
                    tracked = sample_tracked_weights(previous_model, count=int(cfg.inspection.tracked_weight_count))
                diagnostic_row = _training_diagnostic_row(
                    epoch=epoch,
                    batch=batch_index,
                    step=batches,
                    previous_model=previous_model,
                    model=model,
                    state=state,
                    rates=rates,
                    active_threshold=cfg.inspection.active_rate_threshold,
                    near_rate_cap=_near_rate_cap(cfg.solver, cfg.inspection.health),
                    bcm_cfg=cfg.learning.bcm,
                )
                if steady_this_step:
                    diagnostic_row.update(_training_steady_state_row(solved.result, cfg.inspection.steady_state))
                diagnostic_rows.append(diagnostic_row)
                if show_progress:
                    current_health = evaluate_training_health([diagnostic_row], cfg.inspection.health)
                    live_warning_count += int(current_health.get("warning_count", 0) or 0)
                    live_failure_count += int(current_health.get("failure_count", 0) or 0)
                    print(
                        _format_live_training_status(
                            diagnostic_row,
                            current_health,
                            warn_total=live_warning_count,
                            fail_total=live_failure_count,
                        ),
                        file=sys.stderr,
                    )
                    event_summary = _format_health_event_summary(current_health)
                    if event_summary:
                        print(event_summary, file=sys.stderr)
                if cfg.inspection.save_per_batch_arrays:
                    per_batch_arrays.extend(_save_training_probe_arrays(run_dir, batches, rates, model))
                if steady_this_step and (cfg.inspection.steady_state.save_arrays or cfg.inspection.save_plots):
                    steady_probe = _training_steady_state_probe(
                        batches,
                        solved.result,
                        cfg.inspection.steady_state,
                    )
                    if cfg.inspection.save_plots:
                        steady_state_probes.append(steady_probe)
                    if cfg.inspection.steady_state.save_arrays:
                        steady_state_arrays.extend(
                            _save_training_steady_state_arrays(
                                run_dir,
                                steady_probe,
                            )
                        )
                tracked_rows.extend(record_tracked_weights(model, tracked, step=batches))

    log_path = write_csv_rows(run_dir / "tables" / "training_log.csv", log_rows)
    diagnostic_path = None
    tracked_path = None
    health_path = None
    health_events_path = None
    health_report: dict[str, object] | None = None
    figure_paths: dict[str, Path] = {}
    if cfg.inspection.enabled:
        diagnostic_path = write_csv_rows(run_dir / "tables" / "training_diagnostics.csv", diagnostic_rows)
        health_report = evaluate_training_health(diagnostic_rows, cfg.inspection.health)
        health_path = write_json(run_dir / "analysis" / "training_health.json", health_report)
        health_events = health_report.get("events", []) if isinstance(health_report, dict) else []
        health_events_path = write_csv_rows(
            run_dir / "tables" / "training_health_events.csv",
            health_events if isinstance(health_events, list) else [],
        )
        if tracked_rows:
            tracked_path = write_csv_rows(run_dir / "tables" / "tracked_weights.csv", tracked_rows)
        if cfg.inspection.save_plots:
            figure_paths = _save_training_figures(
                run_dir,
                diagnostic_rows,
                tracked_rows,
                steady_state_probes=steady_state_probes,
                steady_cfg=cfg.inspection.steady_state,
                health_report=health_report,
            )
    model_path = save_model_state(run_dir / "model", model, metadata={"batches": batches, "samples_seen": samples_seen})
    summary: dict[str, Any] = {
        "epochs": int(cfg.epochs),
        "batches": batches,
        "samples_seen": samples_seen,
        "time_steps": int(time.size),
    }
    if health_report is not None:
        summary.update(_training_health_summary(health_report))
    outputs = {
        "training_log": relative_output_path(log_path, run_dir),
        "model": relative_output_path(model_path, run_dir),
    }
    if diagnostic_path is not None:
        outputs["training_diagnostics"] = relative_output_path(diagnostic_path, run_dir)
    if health_path is not None:
        outputs["training_health"] = relative_output_path(health_path, run_dir)
    if health_events_path is not None:
        outputs["training_health_events"] = relative_output_path(health_events_path, run_dir)
    if tracked_path is not None:
        outputs["tracked_weights"] = relative_output_path(tracked_path, run_dir)
    if per_batch_arrays:
        outputs["training_probe_arrays"] = [relative_output_path(path, run_dir) for path in per_batch_arrays]
    if steady_state_arrays:
        outputs["training_steady_state_arrays"] = [relative_output_path(path, run_dir) for path in steady_state_arrays]
    for name, path in figure_paths.items():
        outputs[name] = relative_output_path(path, run_dir)
    write_manifest(
        run_dir,
        {
            "workflow": "train",
            "seed": cfg.seed,
            "solver": cfg.solver.backend,
            "learning_rule": cfg.learning.kind,
            "dtype": cfg.solver.jax_dtype,
            "model": model_summary(model),
            "outputs": outputs,
            "summary": summary,
        },
    )
    return TrainingRun(
        run_dir=run_dir,
        model=model,
        learning_state=state,
        summary=summary,
        log_path=log_path,
        model_path=model_path,
    )


def _build_natural_image_drive(
    cfg: NaturalImageWorkflowConfig,
    model: ModelState,
) -> tuple[NaturalImageL4Drive, NaturalImageSampler]:
    dataset = VanHaterenImageDataset(cfg.image_dir, shape=cfg.image_shape, pattern=cfg.pattern)
    sampler = NaturalImageSampler(dataset, crop_size=cfg.crop_size, patches_per_image=cfg.patches_per_image)
    preprocessor = NaturalImagePreprocessor(cfg.preprocess)
    projector = L4NaturalImageProjector(model.layout, cfg.receptive_field, cfg.drive)
    drive = NaturalImageL4Drive(dataset=dataset, preprocessor=preprocessor, projector=projector)
    return drive, sampler


def _iter_batches(items: Sequence[object], batch_size: int) -> Sequence[tuple[object, ...]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return tuple(tuple(items[start : start + batch_size]) for start in range(0, len(items), batch_size))


def _training_log_row(
    *,
    epoch: int,
    batch: int,
    batch_size: int,
    updated: bool,
    stats: dict[str, float],
) -> dict[str, object]:
    row: dict[str, object] = {
        "epoch": int(epoch),
        "batch": int(batch),
        "batch_size": int(batch_size),
        "updated": bool(updated),
    }
    row.update(stats)
    return row


def _should_inspect(cfg: TrainingInspectionConfig, step: int) -> bool:
    return bool(cfg.enabled) and int(cfg.probe_every) > 0 and step % int(cfg.probe_every) == 0


def _should_collect_steady_state(cfg: TrainingInspectionConfig, step: int) -> bool:
    return _should_inspect(cfg, step) and bool(cfg.steady_state.enabled)


def _trajectory_solver_config(cfg: SolverConfig) -> SolverConfig:
    if bool(cfg.store_trajectory):
        return cfg
    return replace(cfg, store_trajectory=True)


def _training_diagnostic_row(
    *,
    epoch: int,
    batch: int,
    step: int,
    previous_model: ModelState,
    model: ModelState,
    state: object,
    rates: RateBatch,
    active_threshold: float,
    near_rate_cap: float | None,
    bcm_cfg: object,
) -> dict[str, object]:
    row: dict[str, object] = {"epoch": int(epoch), "batch": int(batch), "step": int(step)}
    row.update(active_rate_stats(rates, active_threshold=active_threshold))
    row.update(
        extended_active_rate_stats(
            rates,
            active_threshold=active_threshold,
            near_rate_cap=near_rate_cap,
        )
    )
    row.update(plastic_weight_stats(model))
    row.update(weight_delta_stats(previous_model, model))
    if isinstance(state, BCMState):
        row.update(theta_stats(state))
        row.update(theta_distribution_stats(state))
        row.update(row_sum_pressure(model, state=state))
        row.update(_extended_model_plastic_stats(previous_model, model, state=state))
        row.update(bcm_signal_stats(rates, state, bcm_cfg))
    else:
        row.update(row_sum_pressure(model))
        row.update(_extended_model_plastic_stats(previous_model, model, state=None))
    return row


def _save_training_figures(
    run_dir: Path,
    diagnostics: list[dict[str, object]],
    tracked: list[dict[str, object]],
    *,
    steady_state_probes: list[TrainingSteadyStateProbe] | None = None,
    steady_cfg: TrainingSteadyStateConfig = TrainingSteadyStateConfig(),
    health_report: dict[str, object] | None = None,
) -> dict[str, Path]:
    if not diagnostics:
        return {}

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    steps = [int(row["step"]) for row in diagnostics]

    overview = figure_dir / "training_overview.png"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), dpi=140)
    active_has_series = _plot_series(axes[0, 0], steps, diagnostics, ["exc_active_neuron_fraction", "inh_active_neuron_fraction"])
    axes[0, 0].set_title("Active fraction")
    concentration_has_series = _plot_series(axes[0, 1], steps, diagnostics, ["exc_top1_activity_fraction", "exc_top5_activity_fraction"])
    axes[0, 1].set_title("Activity concentration")
    cap_has_series = _plot_series(axes[1, 0], steps, diagnostics, ["row_sum_EE_cap_max_ratio", "row_sum_IE_cap_max_ratio"])
    axes[1, 0].set_title("Row-sum cap ratio")
    theta_has_series = _plot_series(axes[1, 1], steps, diagnostics, ["theta_exc_median", "theta_inh_median"])
    axes[1, 1].set_title(f"Health: {_health_status_text(health_report)}")
    for ax, has_series in zip(
        axes.ravel(),
        [active_has_series, concentration_has_series, cap_has_series, theta_has_series],
        strict=True,
    ):
        ax.set_xlabel("Batch")
        if has_series:
            ax.legend()
    fig.tight_layout()
    fig.savefig(overview)
    plt.close(fig)
    paths["training_overview"] = overview

    activity = figure_dir / "training_activity.png"
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=140)
    exc_rate_has_series = _plot_series(axes[0], steps, diagnostics, ["exc_mean", "exc_median", "exc_p95", "exc_max"])
    axes[0].set_title("Excitatory rates")
    inh_rate_has_series = _plot_series(axes[1], steps, diagnostics, ["inh_mean", "inh_median", "inh_p95", "inh_max"])
    axes[1].set_title("Inhibitory rates")
    for ax, has_series in zip(axes, [exc_rate_has_series, inh_rate_has_series], strict=True):
        ax.set_xlabel("Batch")
        if has_series:
            ax.legend()
    fig.tight_layout()
    fig.savefig(activity)
    plt.close(fig)
    paths["training_activity"] = activity

    bcm = figure_dir / "training_bcm.png"
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=140)
    theta_dist_has_series = _plot_series(axes[0], steps, diagnostics, ["theta_exc_p05", "theta_exc_median", "theta_exc_p95"])
    axes[0].set_title("Excitatory theta")
    bcm_signal_has_series = _plot_series(axes[1], steps, diagnostics, ["bcm_exc_above_theta_fraction", "bcm_exc_signal_mean"])
    axes[1].set_title("BCM signal")
    for ax, has_series in zip(axes, [theta_dist_has_series, bcm_signal_has_series], strict=True):
        ax.set_xlabel("Batch")
        if has_series:
            ax.legend()
    fig.tight_layout()
    fig.savefig(bcm)
    plt.close(fig)
    paths["training_bcm"] = bcm

    plasticity = figure_dir / "training_plasticity.png"
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=140)
    ee_has_series = _plot_series(axes[0], steps, diagnostics, ["W_EE_p05", "W_EE_median", "W_EE_p95", "W_EE_max"])
    axes[0].set_title("E<-E weights")
    ie_has_series = _plot_series(axes[1], steps, diagnostics, ["W_IE_p05", "W_IE_median", "W_IE_p95", "W_IE_max"])
    axes[1].set_title("I<-E weights")
    for ax, has_series in zip(axes, [ee_has_series, ie_has_series], strict=True):
        ax.set_xlabel("Batch")
        if has_series:
            ax.legend()
    fig.tight_layout()
    fig.savefig(plasticity)
    plt.close(fig)
    paths["training_plasticity"] = plasticity

    row_sums = figure_dir / "training_row_sums.png"
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 6.0), dpi=140)
    ee_row_has_series = _plot_series(axes[0], steps, diagnostics, ["W_EE_row_sum_mean", "W_EE_row_sum_p95", "W_EE_row_sum_max"])
    axes[0].set_title("E<-E row sums")
    ie_row_has_series = _plot_series(axes[1], steps, diagnostics, ["W_IE_row_sum_mean", "W_IE_row_sum_p95", "W_IE_row_sum_max"])
    axes[1].set_title("I<-E row sums")
    for ax, has_series in zip(axes, [ee_row_has_series, ie_row_has_series], strict=True):
        ax.set_xlabel("Batch")
        if has_series:
            ax.legend()
    fig.tight_layout()
    fig.savefig(row_sums)
    plt.close(fig)
    paths["training_row_sums"] = row_sums

    if tracked:
        tracked_path = figure_dir / "tracked_weights.png"
        fig, ax = plt.subplots(figsize=(7.0, 4.0), dpi=140)
        rows_by_sample: dict[str, list[dict[str, object]]] = {}
        for row in tracked:
            rows_by_sample.setdefault(str(row["sample_index"]), []).append(row)
        for sample, rows in rows_by_sample.items():
            ordered = sorted(rows, key=lambda item: int(item["step"]))
            ax.plot(
                [int(row["step"]) for row in ordered],
                [float(row["current_weight"]) for row in ordered],
                label=sample,
            )
        ax.set_xlabel("Batch")
        ax.legend(title="sample")
        fig.tight_layout()
        fig.savefig(tracked_path)
        plt.close(fig)
        paths["tracked_weights_figure"] = tracked_path

    probes = list(steady_state_probes or [])
    if probes:
        steady_population = figure_dir / "training_steady_population.png"
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), dpi=140)
        _plot_steady_population_traces(
            axes[0],
            probes,
            population="exc",
            max_plotted=int(steady_cfg.max_plotted_probes),
        )
        _plot_steady_population_traces(
            axes[1],
            probes,
            population="inh",
            max_plotted=int(steady_cfg.max_plotted_probes),
        )
        fig.tight_layout()
        fig.savefig(steady_population)
        plt.close(fig)
        paths["training_steady_population"] = steady_population

        steady_neurons = figure_dir / "training_steady_sampled_neurons.png"
        fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), dpi=140)
        _plot_sampled_neuron_traces(axes[0], probes[-1], population="exc")
        _plot_sampled_neuron_traces(axes[1], probes[-1], population="inh")
        fig.tight_layout()
        fig.savefig(steady_neurons)
        plt.close(fig)
        paths["training_steady_sampled_neurons"] = steady_neurons
    return paths


def _extended_model_plastic_stats(
    previous_model: ModelState,
    model: ModelState,
    *,
    state: BCMState | None,
) -> dict[str, object]:
    previous = as_dense_weights(previous_model.weights)
    current = as_dense_weights(model.weights)
    exc = model.layout.exc_idx
    inh = model.layout.inh_idx
    ee_limits = state.row_sum_limits.target_exc_source_exc if state is not None else None
    ie_limits = state.row_sum_limits.target_inh_source_exc if state is not None else None
    return {
        **extended_plastic_weight_stats(
            "W_EE",
            current[np.ix_(exc, exc)],
            previous=previous[np.ix_(exc, exc)],
            row_sum_limits=ee_limits,
        ),
        **extended_plastic_weight_stats(
            "W_IE",
            current[np.ix_(inh, exc)],
            previous=previous[np.ix_(inh, exc)],
            row_sum_limits=ie_limits,
        ),
    }


def _training_steady_state_row(result: RateResult, cfg: TrainingSteadyStateConfig) -> dict[str, object]:
    row: dict[str, object] = {}
    row.update(_trajectory_stability_stats("steady_exc", result.exc_trajectory, result.time, cfg))
    row.update(_trajectory_stability_stats("steady_inh", result.inh_trajectory, result.time, cfg))
    return row


def _trajectory_stability_stats(
    prefix: str,
    trajectory: NDArray[np.float64] | None,
    time: NDArray[np.float64],
    cfg: TrainingSteadyStateConfig,
) -> dict[str, object]:
    if trajectory is None:
        return _empty_trajectory_stability_stats(prefix)
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 3:
        raise ValueError(f"{prefix} trajectory must have shape (n_time, n_batch, n_units).")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{prefix} trajectory contains NaN or infinite values.")
    if arr.size == 0 or arr.shape[0] == 0 or arr.shape[2] == 0:
        return _empty_trajectory_stability_stats(prefix)
    time_arr = validate_time_grid(np.asarray(time, dtype=float), copy=True)
    if time_arr.size != arr.shape[0]:
        raise ValueError(f"{prefix} trajectory time dimension does not match time.")

    tail_start = _tail_start_index(arr.shape[0], cfg.tail_fraction)
    tail = arr[tail_start:]
    tail_mean_by_batch_unit = np.mean(tail, axis=0)
    final = arr[-1]
    final_delta = np.abs(final - tail_mean_by_batch_unit)
    denominator = np.maximum(np.abs(tail_mean_by_batch_unit), 1.0e-12)
    relative_delta = final_delta / denominator
    population_mean = np.mean(arr, axis=(1, 2))
    tail_population_mean = population_mean[tail_start:]
    if tail_population_mean.size >= 2:
        slope = _linear_slope(time_arr[tail_start:], tail_population_mean)
        tail_duration = float(time_arr[-1] - time_arr[tail_start])
        population_relative_drift = (slope * tail_duration) / max(abs(float(np.mean(tail_population_mean))), 1.0e-12)
    else:
        slope = None
        population_relative_drift = None
    if tail.shape[0] >= 2:
        step = np.abs(np.diff(tail, axis=0))
        step_mean = float(np.mean(step))
        step_p95 = float(np.percentile(step, 95))
        step_max = float(np.max(step))
    else:
        step_mean = None
        step_p95 = None
        step_max = None

    return {
        f"{prefix}_tail_start_index": int(tail_start),
        f"{prefix}_tail_mean": float(np.mean(tail)),
        f"{prefix}_tail_variance": float(np.var(tail)),
        f"{prefix}_final_vs_tail_abs_mean": float(np.mean(final_delta)),
        f"{prefix}_final_vs_tail_abs_p95": float(np.percentile(final_delta, 95)),
        f"{prefix}_final_vs_tail_relative_mean": float(np.mean(relative_delta)),
        f"{prefix}_final_vs_tail_relative_p95": float(np.percentile(relative_delta, 95)),
        f"{prefix}_tail_step_mean_abs_change": step_mean,
        f"{prefix}_tail_step_p95_abs_change": step_p95,
        f"{prefix}_tail_step_max_abs_change": step_max,
        f"{prefix}_tail_population_mean_slope": slope,
        f"{prefix}_tail_population_relative_drift": population_relative_drift,
    }


def _empty_trajectory_stability_stats(prefix: str) -> dict[str, object]:
    return {
        f"{prefix}_tail_start_index": None,
        f"{prefix}_tail_mean": None,
        f"{prefix}_tail_variance": None,
        f"{prefix}_final_vs_tail_abs_mean": None,
        f"{prefix}_final_vs_tail_abs_p95": None,
        f"{prefix}_final_vs_tail_relative_mean": None,
        f"{prefix}_final_vs_tail_relative_p95": None,
        f"{prefix}_tail_step_mean_abs_change": None,
        f"{prefix}_tail_step_p95_abs_change": None,
        f"{prefix}_tail_step_max_abs_change": None,
        f"{prefix}_tail_population_mean_slope": None,
        f"{prefix}_tail_population_relative_drift": None,
    }


def _tail_start_index(n_time: int, fraction: float) -> int:
    if int(n_time) <= 1:
        return 0
    bounded = min(1.0, max(0.0, float(fraction)))
    tail_count = max(1, int(np.ceil(int(n_time) * bounded)))
    return max(0, int(n_time) - tail_count)


def _linear_slope(time: NDArray[np.float64], values: NDArray[np.float64]) -> float:
    if time.size != values.size:
        raise ValueError("time and values must have the same length.")
    centered_time = time - float(np.mean(time))
    denom = float(np.sum(centered_time**2))
    if denom <= 0.0:
        return 0.0
    centered_values = values - float(np.mean(values))
    return float(np.sum(centered_time * centered_values) / denom)


def _near_rate_cap(solver_cfg: SolverConfig, health_cfg: TrainingHealthConfig) -> float | None:
    rate_max = solver_cfg.transfer.rate_max
    if rate_max is None:
        return None
    return float(rate_max) * float(health_cfg.near_rate_cap_ratio)


def _training_health_summary(report: dict[str, object]) -> dict[str, object]:
    final_metrics = report.get("final_metrics", {})
    summary: dict[str, object] = {
        "health_status": report.get("status"),
        "health_warning_count": int(report.get("warning_count", 0) or 0),
        "health_failure_count": int(report.get("failure_count", 0) or 0),
        "first_health_warning_step": report.get("first_warning_step"),
        "first_health_failure_step": report.get("first_failure_step"),
    }
    if isinstance(final_metrics, dict):
        for key in (
            "exc_active_neuron_fraction",
            "inh_active_neuron_fraction",
            "exc_top1_activity_fraction",
            "exc_top5_activity_fraction",
            "steady_exc_final_vs_tail_abs_mean",
            "steady_inh_final_vs_tail_abs_mean",
            "steady_exc_final_vs_tail_relative_mean",
            "steady_inh_final_vs_tail_relative_mean",
            "steady_exc_tail_step_p95_abs_change",
            "steady_inh_tail_step_p95_abs_change",
            "steady_exc_tail_population_relative_drift",
            "steady_inh_tail_population_relative_drift",
        ):
            if key in final_metrics:
                summary[f"final_{key}"] = final_metrics[key]
    return summary


def _format_live_training_status(
    row: dict[str, object],
    report: dict[str, object],
    *,
    warn_total: int,
    fail_total: int,
) -> str:
    return " ".join(
        [
            "[train]",
            f"step={_format_int(row.get('step'))}",
            f"epoch={_format_int(row.get('epoch'))}",
            f"batch={_format_int(row.get('batch'))}",
            f"health={report.get('status', 'unknown')}",
            f"warn_total={int(warn_total)}",
            f"fail_total={int(fail_total)}",
            f"exc_active={_format_metric(row.get('exc_active_neuron_fraction'))}",
            f"inh_active={_format_metric(row.get('inh_active_neuron_fraction'))}",
            f"top1={_format_metric(row.get('exc_top1_activity_fraction'))}",
            f"top5={_format_metric(row.get('exc_top5_activity_fraction'))}",
            f"exc_rate_cap={_format_metric(row.get('exc_near_rate_cap_fraction'))}",
            f"inh_rate_cap={_format_metric(row.get('inh_near_rate_cap_fraction'))}",
            f"row_EE_cap={_format_metric(row.get('row_sum_EE_cap_max_ratio'))}",
            f"row_IE_cap={_format_metric(row.get('row_sum_IE_cap_max_ratio'))}",
            f"theta_exc={_format_metric(row.get('theta_exc_median'))}",
            f"theta_inh={_format_metric(row.get('theta_inh_median'))}",
            f"bcm_exc_above={_format_metric(row.get('bcm_exc_above_theta_fraction'))}",
            f"bcm_inh_above={_format_metric(row.get('bcm_inh_above_theta_fraction'))}",
            f"bcm_exc_signal={_format_metric(row.get('bcm_exc_signal_mean'))}",
            f"bcm_inh_signal={_format_metric(row.get('bcm_inh_signal_mean'))}",
            f"W_EE_delta+={_format_metric(row.get('W_EE_delta_positive_fraction'))}",
            f"W_EE_delta-={_format_metric(row.get('W_EE_delta_negative_fraction'))}",
            f"W_IE_delta+={_format_metric(row.get('W_IE_delta_positive_fraction'))}",
            f"W_IE_delta-={_format_metric(row.get('W_IE_delta_negative_fraction'))}",
        ]
    )


def _format_health_event_summary(report: dict[str, object], *, limit: int = 3) -> str | None:
    events = report.get("events", [])
    if not isinstance(events, list) or not events:
        return None
    shown = events[: int(limit)]
    parts = [_format_health_event(event) for event in shown if isinstance(event, dict)]
    remaining = len(events) - len(shown)
    if remaining > 0:
        parts.append(f"+{remaining} more")
    return "[train] events: " + "; ".join(parts)


def _format_health_event(event: dict[str, object]) -> str:
    severity = str(event.get("severity", "event"))
    metric = str(event.get("metric", "metric"))
    value = _format_metric(event.get("value"))
    threshold = _format_metric(event.get("threshold"))
    return f"{severity} {metric}={value}>{threshold}"


def _format_metric(value: object) -> str:
    converted = _numeric_value(value)
    if converted is None:
        return "-"
    return f"{converted:.3f}"


def _format_int(value: object) -> str:
    converted = _numeric_value(value)
    if converted is None:
        return "-"
    return str(int(converted))


def _plot_series(ax, steps: list[int], rows: list[dict[str, object]], names: list[str]) -> bool:
    plotted = False
    for name in names:
        x: list[int] = []
        y: list[float] = []
        for step, row in zip(steps, rows, strict=True):
            value = _numeric_value(row.get(name))
            if value is None:
                continue
            x.append(step)
            y.append(value)
        if y:
            ax.plot(x, y, label=name)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    return plotted


def _plot_steady_population_traces(
    ax,
    probes: list[TrainingSteadyStateProbe],
    *,
    population: str,
    max_plotted: int,
) -> None:
    shown = probes[-max(1, int(max_plotted)) :]
    plotted = False
    for probe in shown:
        values = probe.exc_population_mean if population == "exc" else probe.inh_population_mean
        if values.size == 0:
            continue
        ax.plot(probe.time, values, alpha=0.7, label=f"step {probe.step}")
        plotted = True
    if plotted:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean rate")
    ax.set_title(f"{population.upper()} population mean")


def _plot_sampled_neuron_traces(ax, probe: TrainingSteadyStateProbe, *, population: str) -> None:
    if population == "exc":
        traces = probe.exc_sample_traces
        indices = probe.exc_sample_indices
    else:
        traces = probe.inh_sample_traces
        indices = probe.inh_sample_indices
    if traces.size == 0 or traces.shape[1] == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    else:
        for col, index in enumerate(indices):
            ax.plot(probe.time, traces[:, col], alpha=0.75, label=str(int(index)))
        ax.legend(title="cell")
    ax.set_xlabel("Time")
    ax.set_ylabel("Rate")
    ax.set_title(f"{population.upper()} sampled neurons, step {probe.step}, batch item {probe.batch_sample_index}")


def _numeric_value(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _health_status_text(report: dict[str, object] | None) -> str:
    if report is None:
        return "unknown"
    return str(report.get("status", "unknown"))


def _save_training_probe_arrays(
    run_dir: Path,
    step: int,
    rates: RateBatch,
    model: ModelState,
) -> list[Path]:
    arrays = run_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    stem = f"training_probe_{int(step):06d}"
    paths = [
        arrays / f"{stem}_exc_rates.npy",
        arrays / f"{stem}_inh_rates.npy",
        arrays / f"{stem}_weights.npy",
    ]
    np.save(paths[0], rates.exc)
    np.save(paths[1], rates.inh)
    weights = model.weights.toarray() if sparse.issparse(model.weights) else np.asarray(model.weights, dtype=float)
    np.save(paths[2], weights)
    return paths


def _training_steady_state_probe(
    step: int,
    result: RateResult,
    cfg: TrainingSteadyStateConfig,
) -> TrainingSteadyStateProbe:
    time = validate_time_grid(np.asarray(result.time, dtype=float), copy=True)
    exc = _trajectory_or_empty(result.exc_trajectory, n_time=time.size)
    inh = _trajectory_or_empty(result.inh_trajectory, n_time=time.size)
    n_batch = max(_trajectory_batch_size(exc), _trajectory_batch_size(inh), 1)
    batch_index = min(max(0, int(cfg.batch_sample_index)), n_batch - 1)
    exc_population = _population_mean_by_time(exc)
    inh_population = _population_mean_by_time(inh)
    exc_indices = _sample_trace_neurons(exc, count=int(cfg.sample_neuron_count), tail_fraction=float(cfg.tail_fraction))
    inh_indices = _sample_trace_neurons(inh, count=int(cfg.sample_neuron_count), tail_fraction=float(cfg.tail_fraction))
    return TrainingSteadyStateProbe(
        step=int(step),
        time=time,
        batch_sample_index=batch_index,
        exc_population_mean=exc_population,
        inh_population_mean=inh_population,
        exc_sample_indices=exc_indices,
        inh_sample_indices=inh_indices,
        exc_sample_traces=_sample_traces_for_batch(exc, batch_index=batch_index, indices=exc_indices),
        inh_sample_traces=_sample_traces_for_batch(inh, batch_index=batch_index, indices=inh_indices),
    )


def _save_training_steady_state_arrays(
    run_dir: Path,
    probe: TrainingSteadyStateProbe,
) -> list[Path]:
    arrays = run_dir / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)
    stem = f"training_probe_{int(probe.step):06d}"
    population_path = arrays / f"{stem}_population_trace.npz"
    sampled_path = arrays / f"{stem}_sampled_neuron_traces.npz"
    np.savez_compressed(
        population_path,
        time=probe.time,
        step=np.array(probe.step, dtype=np.int64),
        batch_sample_index=np.array(probe.batch_sample_index, dtype=np.int64),
        exc_population_mean=probe.exc_population_mean,
        inh_population_mean=probe.inh_population_mean,
    )
    np.savez_compressed(
        sampled_path,
        time=probe.time,
        step=np.array(probe.step, dtype=np.int64),
        batch_sample_index=np.array(probe.batch_sample_index, dtype=np.int64),
        exc_indices=probe.exc_sample_indices,
        inh_indices=probe.inh_sample_indices,
        exc_traces=probe.exc_sample_traces,
        inh_traces=probe.inh_sample_traces,
    )
    return [population_path, sampled_path]


def _trajectory_or_empty(trajectory: NDArray[np.float64] | None, *, n_time: int) -> NDArray[np.float64]:
    if trajectory is None:
        return np.empty((int(n_time), 0, 0), dtype=float)
    arr = np.asarray(trajectory, dtype=float)
    if arr.ndim != 3:
        raise ValueError("training steady-state trajectory must have shape (n_time, n_batch, n_units).")
    if arr.shape[0] != int(n_time):
        raise ValueError("training steady-state trajectory time dimension does not match time.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("training steady-state trajectory contains NaN or infinite values.")
    return arr


def _trajectory_batch_size(trajectory: NDArray[np.float64]) -> int:
    return int(trajectory.shape[1]) if trajectory.ndim == 3 and trajectory.shape[1] > 0 else 0


def _population_mean_by_time(trajectory: NDArray[np.float64]) -> NDArray[np.float64]:
    if trajectory.size == 0 or trajectory.shape[1] == 0 or trajectory.shape[2] == 0:
        return np.empty((0,), dtype=float)
    return np.mean(trajectory, axis=(1, 2))


def _sample_trace_neurons(
    trajectory: NDArray[np.float64],
    *,
    count: int,
    tail_fraction: float,
) -> NDArray[np.int64]:
    if int(count) <= 0 or trajectory.size == 0 or trajectory.shape[2] == 0:
        return np.empty((0,), dtype=np.int64)
    n_units = int(trajectory.shape[2])
    sample_count = min(int(count), n_units)
    tail_start = _tail_start_index(trajectory.shape[0], tail_fraction)
    tail_mean = np.mean(trajectory[tail_start:], axis=(0, 1))
    active_count = min(sample_count, int(np.ceil(sample_count / 2)))
    active = np.argsort(tail_mean)[::-1][:active_count]
    remaining = np.setdiff1d(np.arange(n_units, dtype=np.int64), active.astype(np.int64), assume_unique=False)
    random_count = sample_count - int(active.size)
    if random_count > 0 and remaining.size > 0:
        random = np.random.choice(remaining, size=min(random_count, remaining.size), replace=False)
        chosen = np.concatenate([active.astype(np.int64), np.asarray(random, dtype=np.int64)])
    else:
        chosen = active.astype(np.int64)
    return np.asarray(chosen, dtype=np.int64)


def _sample_traces_for_batch(
    trajectory: NDArray[np.float64],
    *,
    batch_index: int,
    indices: NDArray[np.int64],
) -> NDArray[np.float64]:
    if trajectory.size == 0 or indices.size == 0 or trajectory.shape[1] == 0:
        return np.empty((trajectory.shape[0], 0), dtype=float)
    safe_batch = min(max(0, int(batch_index)), trajectory.shape[1] - 1)
    return np.asarray(trajectory[:, safe_batch, indices], dtype=float)
