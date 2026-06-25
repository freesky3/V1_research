"""End-to-end workflow composition."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from v1_research.seed import preserve_global_seed, set_global_seed
from v1_research.workflows.simulate import SimulationRun, SimulationWorkflowConfig, run_grating_simulation
from v1_research.workflows.train import TrainingRun, TrainingWorkflowConfig, run_training


@dataclass(frozen=True, slots=True)
class FullWorkflowConfig:
    """Configuration for train-then-simulate runs."""

    seed: int | None = None
    train: TrainingWorkflowConfig = field(default_factory=TrainingWorkflowConfig)
    simulate: SimulationWorkflowConfig = field(default_factory=SimulationWorkflowConfig)


@dataclass(frozen=True, slots=True)
class FullRun:
    """In-memory summary of a composed full run."""

    run_dir: Path
    train: TrainingRun
    simulate: SimulationRun
    summary: dict[str, str | int | None]


def run_train_then_simulate(cfg: FullWorkflowConfig, *, show_progress: bool = True) -> FullRun:
    """Runs natural-image training followed by grating simulation."""

    set_global_seed(cfg.seed)
    with preserve_global_seed():
        train_cfg = replace(cfg.train, seed=cfg.seed)
        train_run = run_training(train_cfg, show_progress=show_progress)
        simulate_cfg = replace(
            cfg.simulate,
            seed=cfg.seed,
            model_checkpoint=train_run.model_path,
            run_root=cfg.simulate.run_root if cfg.simulate.run_root != Path("runs") else cfg.train.run_root,
        )
        simulation_run = run_grating_simulation(simulate_cfg)
    return FullRun(
        run_dir=train_run.run_dir.parent.parent,
        train=train_run,
        simulate=simulation_run,
        summary={
            "seed": cfg.seed,
            "train_run": str(train_run.run_dir),
            "simulate_run": str(simulation_run.run_dir),
            "train_batches": int(train_run.summary["batches"]),
        },
    )


__all__ = ["FullRun", "FullWorkflowConfig", "run_train_then_simulate"]
