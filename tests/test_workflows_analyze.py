from __future__ import annotations

import json

import numpy as np
from scipy import sparse
from typer.testing import CliRunner

from v1_research import cli
from v1_research.analysis.communities import LouvainConfig
from v1_research.analysis.pipeline import AnalysisConfig
from v1_research.model import ModelState, PopulationLayout, SheetGeometry
from v1_research.runs import create_run_dir, save_model_state, write_manifest
from v1_research.workflows.analyze import AnalysisWorkflowConfig, run_analysis_workflow


def _four_exc_model() -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=2, region_size=2.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E", "E", "E", "E"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = sparse.csr_matrix((4, 5), dtype=float)
    return ModelState(layout=layout, connection_mask=weights != 0.0, weights=weights)


def _simulation_bundle(tmp_path, *, with_trajectory: bool = True):
    run_dir = create_run_dir(tmp_path / "runs", "simulate")
    model = _four_exc_model()
    save_model_state(run_dir / "model", model)
    orientation_angles = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    np.save(run_dir / "arrays" / "orientation_angles.npy", orientation_angles)
    np.save(run_dir / "arrays" / "time.npy", np.array([0.0, 0.1, 0.2]))
    mean_rates = np.array(
        [
            [4.0, 4.0, 0.2, 0.2],
            [0.2, 0.2, 5.0, 5.0],
            [4.0, 4.0, 0.2, 0.2],
            [0.2, 0.2, 5.0, 5.0],
        ]
    )
    np.save(run_dir / "arrays" / "excitatory_rates.npy", mean_rates)
    if with_trajectory:
        trajectory = np.repeat(mean_rates[np.newaxis, :, :], 3, axis=0)
        np.save(run_dir / "arrays" / "excitatory_trajectory.npy", trajectory)
    write_manifest(run_dir, {"workflow": "simulate"})
    return run_dir


def test_analysis_workflow_reads_simulation_bundle_and_writes_outputs(tmp_path) -> None:
    run_dir = _simulation_bundle(tmp_path)
    cfg = AnalysisWorkflowConfig(
        simulation_run=run_dir,
        analysis=AnalysisConfig(
            osi_threshold=0.0,
            filter_by_osi=False,
            random_sample_fraction=1.0,
            louvain=LouvainConfig(num_runs=2, consensus_reps=2, min_cluster_size=1),
        ),
    )

    result = run_analysis_workflow(cfg)

    assert result.run_dir == run_dir
    assert result.result.status == "ok"
    assert result.result.osi.shape == (4,)
    assert (run_dir / "analysis" / "osi.npy").is_file()
    assert (run_dir / "analysis" / "preferred_orientation.npy").is_file()
    assert (run_dir / "analysis" / "community_labels.npy").is_file()
    assert (run_dir / "analysis" / "metrics.json").is_file()
    assert (run_dir / "tables" / "ensemble_metrics.csv").is_file()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["analysis"]["status"] == "ok"


def test_analysis_workflow_falls_back_to_mean_rates_without_trajectory(tmp_path) -> None:
    run_dir = _simulation_bundle(tmp_path, with_trajectory=False)

    result = run_analysis_workflow(
        AnalysisWorkflowConfig(
            simulation_run=run_dir,
            analysis=AnalysisConfig(
                osi_threshold=0.0,
                filter_by_osi=False,
                louvain=LouvainConfig(num_runs=1, consensus_reps=1, min_cluster_size=1),
            ),
        )
    )

    assert result.result.steady_state_responses.shape == (4, 4, 1)


def test_cli_analyze_loads_config_and_dispatches(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "analyze.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"simulation_run: {tmp_path / 'runs' / 'simulate' / 'demo'}",
                "analysis:",
                "  osi_threshold: 0.4",
                "  louvain:",
                "    num_runs: 2",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_analysis_workflow(cfg):
        captured["simulation_run"] = cfg.simulation_run
        captured["osi_threshold"] = cfg.analysis.osi_threshold
        captured["num_runs"] = cfg.analysis.louvain.num_runs

        class Result:
            run_dir = tmp_path / "runs" / "simulate" / "demo"

        return Result()

    monkeypatch.setattr(cli, "run_analysis_workflow", fake_run_analysis_workflow)

    result = CliRunner().invoke(
        cli.app,
        ["analyze", "--config", str(config_path), "-o", "analysis.osi_threshold=0.3"],
    )

    assert result.exit_code == 0, result.output
    assert captured["simulation_run"] == tmp_path / "runs" / "simulate" / "demo"
    assert captured["osi_threshold"] == 0.3
    assert captured["num_runs"] == 2
