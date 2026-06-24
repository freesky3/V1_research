"""Natural-image training workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

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
class TrainingInspectionConfig:
    """Optional intermediate diagnostics for training runs."""

    enabled: bool = False
    probe_every: int = 1
    health: TrainingHealthConfig = field(default_factory=TrainingHealthConfig)
    tracked_weight_count: int = 0
    save_plots: bool = False
    save_per_batch_arrays: bool = False
    active_rate_threshold: float = 1.0


@dataclass(frozen=True, slots=True)
class TrainingWorkflowConfig:
    """Top-level configuration for natural-image learning runs."""

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
    return RateBatch(exc=result.exc, inh=result.inh, external=external.T)


def run_training(cfg: TrainingWorkflowConfig, *, show_progress: bool = True) -> TrainingRun:
    """Runs natural-image learning and writes a run bundle."""

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
            rates = _solve_batch_rates(
                model,
                drive=batch_drive,
                time=time,
                n_batch=len(batch),
                solver_cfg=cfg.solver,
                background_trace=background,
            )
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
            if _should_inspect(cfg.inspection, batches):
                if not tracked and int(cfg.inspection.tracked_weight_count) > 0:
                    tracked = sample_tracked_weights(previous_model, count=int(cfg.inspection.tracked_weight_count))
                diagnostic_rows.append(
                    _training_diagnostic_row(
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
                )
                if cfg.inspection.save_per_batch_arrays:
                    per_batch_arrays.extend(_save_training_probe_arrays(run_dir, batches, rates, model))
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
            figure_paths = _save_training_figures(run_dir, diagnostic_rows, tracked_rows, health_report=health_report)
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
    for name, path in figure_paths.items():
        outputs[name] = relative_output_path(path, run_dir)
    write_manifest(
        run_dir,
        {
            "workflow": "train",
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
        ):
            if key in final_metrics:
                summary[f"final_{key}"] = final_metrics[key]
    return summary


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
