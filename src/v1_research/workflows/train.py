"""Natural-image training workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from collections.abc import Callable, Sequence

import numpy as np
from numpy.typing import NDArray

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
from v1_research.model import ModelConfig, ModelState, build_model
from v1_research.runs import (
    create_run_dir,
    model_summary,
    save_model_state,
    write_config,
    write_csv_rows,
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


@dataclass(frozen=True, slots=True)
class TrainingRun:
    """In-memory summary of a saved training run."""

    run_dir: Path
    model: ModelState
    learning_state: object
    summary: dict[str, int | float | list[int]]
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

    solver_fn = solve_rates if solver is None else solver
    result = solver_fn(
        model,
        drive=drive,
        time=np.asarray(time, dtype=float),
        n_batch=n_batch,
        cfg=solver_cfg,
        background_trace=background_trace,
    )
    external = np.asarray(drive(float(np.asarray(time, dtype=float)[0])), dtype=float)
    if external.ndim == 1:
        external = external[:, np.newaxis]
    rates = RateBatch(exc=result.exc, inh=result.inh, external=external.T)
    return apply_learning_rule(model, rates, rule, state=state)


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
    batches = 0
    samples_seen = 0
    log_rows: list[dict[str, object]] = []
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
            update = solve_and_learn_batch(
                model,
                drive=drive.make_static_batch_func(batch),
                time=time,
                n_batch=len(batch),
                solver_cfg=cfg.solver,
                rule=rule,
                state=state,
                background_trace=background,
            )
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

    log_path = write_csv_rows(run_dir / "tables" / "training_log.csv", log_rows)
    model_path = save_model_state(run_dir / "model", model, metadata={"batches": batches, "samples_seen": samples_seen})
    summary: dict[str, int | float | list[int]] = {
        "epochs": int(cfg.epochs),
        "batches": batches,
        "samples_seen": samples_seen,
        "time_steps": int(time.size),
    }
    write_manifest(
        run_dir,
        {
            "workflow": "train",
            "solver": cfg.solver.backend,
            "learning_rule": cfg.learning.kind,
            "dtype": cfg.solver.jax_dtype,
            "model": model_summary(model),
            "outputs": {
                "training_log": str(log_path.relative_to(run_dir)),
                "model": str(model_path.relative_to(run_dir)),
            },
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
