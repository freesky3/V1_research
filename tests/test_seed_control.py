from __future__ import annotations

import json
import random

import numpy as np
from scipy import sparse

from v1_research import cli
from v1_research.dynamics import SolverConfig
from v1_research.inputs.grating import DriftingGratingConfig
from v1_research.model import ModelState, PopulationLayout, SheetGeometry
from v1_research.runs import save_model_state
from v1_research.seed import set_global_seed
from v1_research.workflows.analyze import AnalysisWorkflowConfig
from v1_research.workflows.full import FullWorkflowConfig, run_train_then_simulate
from v1_research.workflows.simulate import (
    SimulationInspectionConfig,
    SimulationWorkflowConfig,
    TrialScheduleConfig,
    run_grating_simulation,
)
from v1_research.workflows.sweep import SweepConfig, run_sweep
from v1_research.workflows.train import TrainingWorkflowConfig


def _one_cell_model() -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = sparse.csr_matrix(np.array([[0.0, 0.05]], dtype=float))
    return ModelState(layout=layout, connection_mask=weights != 0.0, weights=weights)


def test_set_global_seed_controls_numpy_and_random_streams() -> None:
    set_global_seed(17)
    first = (float(np.random.random()), float(random.random()))

    set_global_seed(17)
    second = (float(np.random.random()), float(random.random()))

    assert first == second


def test_cli_load_workflow_config_reads_seed_and_override(tmp_path) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("seed: 11\nrun_root: runs\n", encoding="utf-8")

    cfg = cli.load_workflow_config(config_path, ["seed=22"], TrainingWorkflowConfig)

    assert cfg.seed == 22


def test_simulation_seed_makes_trial_schedule_reproducible(tmp_path) -> None:
    checkpoint = save_model_state(tmp_path / "input_model", _one_cell_model())
    cfg = SimulationWorkflowConfig(
        run_root=tmp_path / "runs",
        model_checkpoint=checkpoint,
        seed=17,
        solver=SolverConfig(backend="scipy", scipy_method="RK4", store_trajectory=True),
        grating=DriftingGratingConfig(n_orientations=3, visual_gain=1.0, baseline_rate=0.1),
        trials=TrialScheduleConfig(repeats_per_direction=2, shuffle=True, random_phase=True, phase_jitter=0.1),
        inspection=SimulationInspectionConfig(enabled=False, save_plots=False),
        time=np.array([0.0, 0.01, 0.02], dtype=float),
    )

    np.random.seed(1)
    first = run_grating_simulation(cfg)
    np.random.seed(999)
    second = run_grating_simulation(cfg)

    np.testing.assert_array_equal(
        np.load(first.array_paths["trial_direction_indices"]),
        np.load(second.array_paths["trial_direction_indices"]),
    )
    np.testing.assert_allclose(
        np.load(first.array_paths["trial_phase_offsets"]),
        np.load(second.array_paths["trial_phase_offsets"]),
    )
    assert first.summary["n_trials"] == 6

    manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 17


def test_full_workflow_uses_one_seed_across_train_and_simulate(monkeypatch, tmp_path) -> None:
    from v1_research.workflows import full as full_module

    calls: list[tuple[str, float]] = []

    class TrainResult:
        run_dir = tmp_path / "runs" / "train" / "demo"
        model_path = tmp_path / "runs" / "train" / "demo" / "model"
        summary = {"batches": 1}

    class SimResult:
        run_dir = tmp_path / "runs" / "simulate" / "demo"
        summary = {}

    def fake_run_training(cfg, *, show_progress: bool = True):
        calls.append(("train", float(np.random.random())))
        return TrainResult()

    def fake_run_grating_simulation(cfg):
        calls.append(("simulate", float(np.random.random())))
        return SimResult()

    monkeypatch.setattr(full_module, "run_training", fake_run_training)
    monkeypatch.setattr(full_module, "run_grating_simulation", fake_run_grating_simulation)

    result = run_train_then_simulate(FullWorkflowConfig(seed=123), show_progress=False)

    np.testing.assert_allclose(
        np.array([calls[0][1], calls[1][1]], dtype=float),
        np.array([0.6964691855978616, 0.28613933495037946], dtype=float),
    )
    assert calls[0][0] == "train"
    assert calls[1][0] == "simulate"
    assert result.summary["train_batches"] == 1


def test_sweep_uses_one_seed_across_grid_points(monkeypatch, tmp_path) -> None:
    from v1_research.workflows import sweep as sweep_module

    calls: list[float] = []

    class FakeRun:
        def __init__(self, run_dir: str) -> None:
            self.run_dir = tmp_path / run_dir
            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.summary = {"n_orientations": 4}

    def fake_run_grating_simulation(cfg):
        calls.append(float(np.random.random()))
        return FakeRun(f"runs/simulate/run_{len(calls)}")

    monkeypatch.setattr(sweep_module, "run_grating_simulation", fake_run_grating_simulation)

    result = run_sweep(
        SweepConfig(
            workflow="simulate",
            run_root=tmp_path / "runs",
            seed=123,
            base={
                "grating": {"n_orientations": 4},
                "solver": {"backend": "scipy"},
            },
            parameters={"grating.visual_gain": [100.0, 200.0]},
        )
    )

    np.testing.assert_allclose(
        np.array(calls, dtype=float),
        np.array([0.6964691855978616, 0.28613933495037946], dtype=float),
    )
    assert result.summary["runs"] == 2
    assert result.summary["failed"] == 0

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 123


def test_analysis_workflow_config_accepts_seed(tmp_path) -> None:
    config_path = tmp_path / "analysis.yaml"
    config_path.write_text("simulation_run: runs/simulate/example\nseed: 17\n", encoding="utf-8")

    cfg = cli.load_workflow_config(config_path, [], AnalysisWorkflowConfig)

    assert cfg.seed == 17
