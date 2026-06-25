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
class SimulationWorkflowConfig:
    """Top-level configuration for a grating simulation run."""

    run_root: str | Path = Path("runs")
    empirical_data_path: str | Path = Path("data/sample_data.pkl")
    model: ModelConfig = field(default_factory=ModelConfig)
    model_checkpoint: str | Path | None = None
    solver: SolverConfig = field(default_factory=SolverConfig)
    grating: DriftingGratingConfig = field(default_factory=DriftingGratingConfig)
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
    array_paths: dict[str, Path]
    summary: dict[str, Any]

    @property
    def exc_rates(self) -> NDArray[np.float64]:
        """Excitatory mean rates shaped ``(n_orientation, n_exc)``."""

        return self.rates.exc

    @property
    def inh_rates(self) -> NDArray[np.float64]:
        """Inhibitory mean rates shaped ``(n_orientation, n_inh)``."""

        return self.rates.inh


def run_grating_simulation(cfg: SimulationWorkflowConfig) -> SimulationRun:
    """Runs a drifting-grating batch simulation and writes a run bundle."""

    run_dir = create_run_dir(cfg.run_root, "simulate")
    write_config(run_dir, cfg)
    model = _load_or_build_model(cfg)
    time = validate_time_grid(np.asarray(cfg.time, dtype=float), copy=True)
    stimulus = DriftingGratingInput(cfg.grating, model.layout)
    orientation_angles = stimulus.orientation_angles
    background = generate_background_trace(
        cfg.background,
        n_exc=model.layout.n_exc,
        n_inh=model.layout.n_inh,
        n_batch=orientation_angles.size,
        time=time,
    )
    drive = stimulus.make_batched_drive_func(orientation_angles)
    rates = solve_rates(
        model,
        drive=drive,
        time=time,
        n_batch=orientation_angles.size,
        cfg=cfg.solver,
        background_trace=background,
    )
    save_model_state(run_dir / "model", model, metadata={"source": _model_source(cfg)})
    array_paths = _save_simulation_arrays(run_dir, rates, orientation_angles)
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
            figure_paths = save_simulation_figures(run_dir, rates, orientation_angles, health_report)
    summary: dict[str, Any] = {
        "n_orientations": int(orientation_angles.size),
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
        orientation_angles=orientation_angles,
        array_paths=array_paths,
        summary=summary,
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
    orientation_angles: NDArray[np.float64],
) -> dict[str, Path]:
    arrays = run_dir / "arrays"
    paths = {
        "excitatory_rates": arrays / "excitatory_rates.npy",
        "inhibitory_rates": arrays / "inhibitory_rates.npy",
        "time": arrays / "time.npy",
        "orientation_angles": arrays / "orientation_angles.npy",
    }
    np.save(paths["excitatory_rates"], rates.exc)
    np.save(paths["inhibitory_rates"], rates.inh)
    np.save(paths["time"], rates.time)
    np.save(paths["orientation_angles"], orientation_angles)
    if rates.exc_trajectory is not None:
        paths["excitatory_trajectory"] = arrays / "excitatory_trajectory.npy"
        np.save(paths["excitatory_trajectory"], rates.exc_trajectory)
    if rates.inh_trajectory is not None:
        paths["inhibitory_trajectory"] = arrays / "inhibitory_trajectory.npy"
        np.save(paths["inhibitory_trajectory"], rates.inh_trajectory)
    return paths


__all__ = ["SimulationInspectionConfig", "SimulationRun", "SimulationWorkflowConfig", "run_grating_simulation"]
