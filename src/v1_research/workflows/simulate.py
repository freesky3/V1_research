"""Drifting-grating simulation workflow orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    save_model_state,
    write_config,
    write_manifest,
)


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


@dataclass(frozen=True, slots=True)
class SimulationRun:
    """In-memory summary of a saved grating simulation."""

    run_dir: Path
    model: ModelState
    rates: RateResult
    orientation_angles: NDArray[np.float64]
    array_paths: dict[str, Path]
    summary: dict[str, int | float | list[int]]

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
    rates = solve_rates(
        model,
        drive=stimulus.make_batched_drive_func(orientation_angles),
        time=time,
        n_batch=orientation_angles.size,
        cfg=cfg.solver,
        background_trace=background,
    )
    save_model_state(run_dir / "model", model, metadata={"source": _model_source(cfg)})
    array_paths = _save_simulation_arrays(run_dir, rates, orientation_angles)
    summary: dict[str, int | float | list[int]] = {
        "n_orientations": int(orientation_angles.size),
        "time_steps": int(time.size),
        "exc_shape": list(rates.exc.shape),
        "inh_shape": list(rates.inh.shape),
    }
    write_manifest(
        run_dir,
        {
            "workflow": "simulate",
            "solver": cfg.solver.backend,
            "dtype": cfg.solver.jax_dtype,
            "model": model_summary(model),
            "outputs": {key: str(path.relative_to(run_dir)) for key, path in array_paths.items()},
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


__all__ = ["SimulationRun", "SimulationWorkflowConfig", "run_grating_simulation"]
