from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from typer.testing import CliRunner

from v1_research import cli
from v1_research.model import ModelState, PopulationLayout, SheetGeometry
from v1_research.runs import create_run_dir, save_model_state, write_csv_rows, write_json, write_manifest
from v1_research.workflows.summarize import summarize_run


def _model() -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = sparse.csr_matrix((1, 2), dtype=float)
    return ModelState(layout=layout, connection_mask=weights != 0.0, weights=weights)


def test_summarize_run_reads_simulation_analysis_and_training_tables(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path / "runs", "simulate")
    save_model_state(run_dir / "model", _model())
    np.save(run_dir / "arrays" / "excitatory_rates.npy", np.array([[1.0], [3.0]]))
    write_json(run_dir / "analysis" / "metrics.json", {"n_ensembles": 2, "classified_fraction": 0.5})
    write_json(
        run_dir / "analysis" / "training_health.json",
        {
            "status": "warn",
            "warning_count": 2,
            "failure_count": 0,
            "first_warning_step": 3,
            "final_metrics": {"exc_active_neuron_fraction": 0.4},
        },
    )
    write_csv_rows(run_dir / "tables" / "training_diagnostics.csv", [{"batch": 1, "exc_mean": 2.0}])
    write_manifest(run_dir, {"workflow": "simulate", "summary": {"batches": 1}})

    summary = summarize_run(run_dir)

    assert summary["workflow"] == "simulate"
    assert summary["model.n_exc"] == 1
    assert summary["rates.exc_mean"] == 2.0
    assert summary["analysis.n_ensembles"] == 2
    assert summary["training_health.status"] == "warn"
    assert summary["training_health.warning_count"] == 2
    assert summary["training_health.final.exc_active_neuron_fraction"] == 0.4
    assert summary["training_diagnostics.rows"] == 1


def test_cli_summarize_writes_json_output(tmp_path) -> None:
    run_dir = create_run_dir(tmp_path / "runs", "train")
    save_model_state(run_dir / "model", _model())
    write_manifest(run_dir, {"workflow": "train"})
    output = tmp_path / "summary.json"

    result = CliRunner().invoke(cli.app, ["summarize", "--run", str(run_dir), "--output", str(output)])

    assert result.exit_code == 0, result.output
    assert str(output) in result.output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["workflow"] == "train"
