"""Drifting-grating simulation workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from v1_research.data import ExperimentalData
from v1_research.dynamics import RateResult, SolverConfig, solve_rates
from v1_research.inputs.background import BackgroundConfig, generate_background_trace, validate_time_grid
from v1_research.inputs.grating import DriftingGratingConfig, DriftingGratingInput
from v1_research.model import ModelConfig, ModelState, build_model
from v1_research.runs import (
    create_run_dir,
    load_model_state,
    model_summary,
    relative_output_path,
    save_model_state,
    write_config,
    write_json,
    write_manifest,
)
from v1_research.workflows.simulation_health import (
    SimulationHealthConfig,
    compute_simulation_health,
    save_simulation_figures,
)


@dataclass(frozen=True, slots=True)
class SimulationInspectionConfig:
    """Optional health diagnostics for simulation runs."""

    enabled: bool = True
    save_plots: bool = True
    health: SimulationHealthConfig = field(default_factory=SimulationHealthConfig)


@dataclass(frozen=True, slots=True)
class TrialScheduleConfig:
    """Repeated grating-trial schedule for direction tuning analysis."""

    repeats_per_direction: int = 4
    shuffle: bool = True
    random_phase: bool = True
    phase_jitter: float = 0.35


@dataclass(frozen=True, slots=True)
class TrialSchedule:
    """Concrete trial directions and phase offsets consumed by the solver."""

    orientation_angles: NDArray[np.float64]
    direction_indices: NDArray[np.int64]
    trial_orientation_angles: NDArray[np.float64]
    phase_offsets: NDArray[np.float64]

    @property
    def n_trials(self) -> int:
        """Number of independent trials."""

        return int(self.direction_indices.size)


@dataclass(frozen=True, slots=True)
class SimulationWorkflowConfig:
    """Top-level configuration for a grating simulation run."""

    run_root: str | Path = Path("runs")
    empirical_data_path: str | Path = Path("data/sample_data.pkl")
    model: ModelConfig = field(default_factory=ModelConfig)
    model_checkpoint: str | Path | None = None
    solver: SolverConfig = field(default_factory=SolverConfig)
    grating: DriftingGratingConfig = field(default_factory=DriftingGratingConfig)
    trials: TrialScheduleConfig = field(default_factory=TrialScheduleConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    time: NDArray[np.float64] | tuple[float, ...] = field(default_factory=lambda: np.linspace(0.0, 0.2, 51))
    inspection: SimulationInspectionConfig = field(default_factory=SimulationInspectionConfig)


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """In-memory summary of a saved grating simulation."""

    run_dir: Path
    model: ModelState
    rates: RateResult
    orientation_angles: NDArray[np.float64]
    trial_direction_indices: NDArray[np.int64]
    array_paths: dict[str, Path]
    summary: dict[str, Any]

    @property
    def exc_rates(self) -> NDArray[np.float64]:
        """Excitatory mean rates shaped ``(n_trial, n_exc)``."""

        return self.rates.exc

    @property
    def inh_rates(self) -> NDArray[np.float64]:
        """Inhibitory mean rates shaped ``(n_trial, n_inh)``."""

        return self.rates.inh


def run_grating_simulation(cfg: SimulationWorkflowConfig) -> SimulationRun:
    """Runs a drifting-grating batch simulation and writes a run bundle."""

    run_dir = create_run_dir(cfg.run_root, "simulate")
    write_config(run_dir, cfg)
    model = _load_or_build_model(cfg)
    time = validate_time_grid(np.asarray(cfg.time, dtype=float), copy=True)
    stimulus = DriftingGratingInput(cfg.grating, model.layout)
    schedule = build_trial_schedule(stimulus.orientation_angles, cfg.trials)
    background = generate_background_trace(
        cfg.background,
        n_exc=model.layout.n_exc,
        n_inh=model.layout.n_inh,
        n_batch=schedule.n_trials,
        time=time,
    )
    drive = stimulus.make_batched_drive_func(schedule.trial_orientation_angles, phase_offsets=schedule.phase_offsets)
    rates = solve_rates(
        model,
        drive=drive,
        time=time,
        n_batch=schedule.n_trials,
        cfg=cfg.solver,
        background_trace=background,
    )
    save_model_state(run_dir / "model", model, metadata={"source": _model_source(cfg)})
    array_paths = _save_simulation_arrays(run_dir, rates, schedule)
    health_report: dict[str, object] | None = None
    health_path: Path | None = None
    figure_paths: dict[str, Path] = {}
    if cfg.inspection.enabled:
        stimulus_trace = _sample_stimulus_trace(drive, time)
        health_report = compute_simulation_health(
            rates,
            stimulus_trace=stimulus_trace,
            background_trace=background,
            rate_max=cfg.solver.transfer.rate_max,
            cfg=cfg.inspection.health,
        )
        health_path = write_json(run_dir / "analysis" / "simulation_health.json", health_report)
        if cfg.inspection.save_plots:
            figure_paths = save_simulation_figures(run_dir, rates, schedule.trial_orientation_angles, health_report)
    summary: dict[str, Any] = {
        "n_orientations": int(schedule.orientation_angles.size),
        "n_trials": int(schedule.n_trials),
        "repeats_per_direction": int(cfg.trials.repeats_per_direction),
        "time_steps": int(time.size),
        "exc_shape": list(rates.exc.shape),
        "inh_shape": list(rates.inh.shape),
    }
    if health_report is not None:
        summary.update(_simulation_health_summary(health_report))
    outputs = {key: str(path.relative_to(run_dir)) for key, path in array_paths.items()}
    if health_path is not None:
        outputs["simulation_health"] = relative_output_path(health_path, run_dir)
    for key, path in figure_paths.items():
        outputs[key] = relative_output_path(path, run_dir)
    write_manifest(
        run_dir,
        {
            "workflow": "simulate",
            "solver": cfg.solver.backend,
            "dtype": cfg.solver.jax_dtype,
            "model": model_summary(model),
            "outputs": outputs,
            "summary": summary,
        },
    )
    return SimulationRun(
        run_dir=run_dir,
        model=model,
        rates=rates,
        orientation_angles=schedule.orientation_angles,
        trial_direction_indices=schedule.direction_indices,
        array_paths=array_paths,
        summary=summary,
    )


def build_trial_schedule(orientation_angles: NDArray[np.float64], cfg: TrialScheduleConfig) -> TrialSchedule:
    """Builds repeated grating trials using the global NumPy random state."""

    angles = np.asarray(orientation_angles, dtype=float).reshape(-1)
    if angles.size == 0 or not np.all(np.isfinite(angles)):
        raise ValueError("orientation_angles must be a non-empty finite vector.")
    if int(cfg.repeats_per_direction) < 1:
        raise ValueError("repeats_per_direction must be at least 1.")
    if float(cfg.phase_jitter) < 0.0 or not np.isfinite(float(cfg.phase_jitter)):
        raise ValueError("phase_jitter must be finite and non-negative.")

    direction_indices = np.repeat(np.arange(angles.size, dtype=np.int64), int(cfg.repeats_per_direction))
    if cfg.shuffle:
        direction_indices = np.random.permutation(direction_indices).astype(np.int64, copy=False)
    if cfg.random_phase:
        phase_offsets = np.random.uniform(
            -float(cfg.phase_jitter),
            float(cfg.phase_jitter),
            size=direction_indices.size,
        ).astype(float, copy=False)
    else:
        phase_offsets = np.zeros(direction_indices.size, dtype=float)
    return TrialSchedule(
        orientation_angles=angles,
        direction_indices=direction_indices,
        trial_orientation_angles=angles[direction_indices],
        phase_offsets=phase_offsets,
    )


def _sample_stimulus_trace(drive, time: NDArray[np.float64]) -> NDArray[np.float64]:
    values = [np.asarray(drive(float(t)), dtype=float).T for t in time]
    return np.asarray(values, dtype=np.float64)


def _simulation_health_summary(report: dict[str, object]) -> dict[str, object]:
    metrics = report.get("metrics", {})
    summary: dict[str, object] = {
        "health_status": report.get("status"),
        "health_warning_count": int(report.get("warning_count", 0) or 0),
        "health_failure_count": int(report.get("failure_count", 0) or 0),
    }
    if isinstance(metrics, dict):
        for key in (
            "exc_active_fraction",
            "inh_active_fraction",
            "exc_silent_fraction",
            "inh_silent_fraction",
            "exc_top1_activity_fraction",
            "exc_top5_activity_fraction",
            "exc_near_rate_cap_fraction",
            "exc_relative_mean_drift",
            "inh_relative_mean_drift",
        ):
            if key in metrics:
                summary[f"final_{key}"] = metrics[key]
    return summary


def _load_or_build_model(cfg: SimulationWorkflowConfig) -> ModelState:
    if cfg.model_checkpoint is not None:
        return load_model_state(cfg.model_checkpoint)
    empirical = ExperimentalData.from_path(cfg.empirical_data_path)
    return build_model(cfg.model, empirical)


def _model_source(cfg: SimulationWorkflowConfig) -> str:
    if cfg.model_checkpoint is not None:
        return str(cfg.model_checkpoint)
    return "built_from_config"


def _save_simulation_arrays(
    run_dir: Path,
    rates: RateResult,
    schedule: TrialSchedule,
) -> dict[str, Path]:
    arrays = run_dir / "arrays"
    paths = {
        "excitatory_rates": arrays / "excitatory_rates.npy",
        "inhibitory_rates": arrays / "inhibitory_rates.npy",
        "time": arrays / "time.npy",
        "orientation_angles": arrays / "orientation_angles.npy",
        "trial_direction_indices": arrays / "trial_direction_indices.npy",
        "trial_orientation_angles": arrays / "trial_orientation_angles.npy",
        "trial_phase_offsets": arrays / "trial_phase_offsets.npy",
    }
    np.save(paths["excitatory_rates"], rates.exc)
    np.save(paths["inhibitory_rates"], rates.inh)
    np.save(paths["time"], rates.time)
    np.save(paths["orientation_angles"], schedule.orientation_angles)
    np.save(paths["trial_direction_indices"], schedule.direction_indices)
    np.save(paths["trial_orientation_angles"], schedule.trial_orientation_angles)
    np.save(paths["trial_phase_offsets"], schedule.phase_offsets)
    if rates.exc_trajectory is not None:
        paths["excitatory_trajectory"] = arrays / "excitatory_trajectory.npy"
        np.save(paths["excitatory_trajectory"], rates.exc_trajectory)
    if rates.inh_trajectory is not None:
        paths["inhibitory_trajectory"] = arrays / "inhibitory_trajectory.npy"
        np.save(paths["inhibitory_trajectory"], rates.inh_trajectory)
    return paths


__all__ = [
    "SimulationInspectionConfig",
    "SimulationRun",
    "SimulationWorkflowConfig",
    "TrialSchedule",
    "TrialScheduleConfig",
    "build_trial_schedule",
    "run_grating_simulation",
]
