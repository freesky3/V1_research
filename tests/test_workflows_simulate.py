from __future__ import annotations

import json

import numpy as np
from scipy import sparse

from v1_research.dynamics import SolverConfig
from v1_research.inputs.grating import DriftingGratingConfig
from v1_research.model import ModelState, PopulationLayout, SheetGeometry
from v1_research.runs import save_model_state
from v1_research.workflows.simulate import SimulationWorkflowConfig, run_grating_simulation


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


def test_grating_simulation_writes_run_bundle_from_checkpoint(tmp_path) -> None:
    checkpoint = save_model_state(tmp_path / "input_model", _one_cell_model())
    cfg = SimulationWorkflowConfig(
        run_root=tmp_path / "runs",
        model_checkpoint=checkpoint,
        solver=SolverConfig(backend="scipy", scipy_method="RK4", store_trajectory=True),
        grating=DriftingGratingConfig(n_orientations=3, visual_gain=1.0, baseline_rate=0.1),
        time=np.array([0.0, 0.01, 0.02], dtype=float),
    )

    result = run_grating_simulation(cfg)

    assert result.run_dir.parent == tmp_path / "runs" / "simulate"
    assert result.exc_rates.shape == (3, 1)
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "model" / "state.npz").is_file()
    np.testing.assert_allclose(np.load(result.array_paths["time"]), cfg.time)
    np.testing.assert_allclose(np.load(result.array_paths["orientation_angles"]), result.orientation_angles)
    assert np.load(result.array_paths["excitatory_rates"]).shape == (3, 1)
    assert np.load(result.array_paths["excitatory_trajectory"]).shape == (3, 3, 1)
    assert (result.run_dir / "analysis" / "simulation_health.json").is_file()
    for name in [
        "simulate_overview.png",
        "simulate_orientation_heatmaps.png",
        "simulate_traces.png",
    ]:
        assert (result.run_dir / "figures" / name).is_file()
    assert result.summary["health_status"] in {"ok", "warn", "fail"}

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "simulate"
    assert manifest["solver"] == "scipy"
    assert manifest["summary"]["n_orientations"] == 3
    assert manifest["outputs"]["simulation_health"] == "analysis/simulation_health.json"
    assert manifest["outputs"]["simulate_overview"] == "figures/simulate_overview.png"
