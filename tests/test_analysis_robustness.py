from __future__ import annotations

import numpy as np

from v1_research.analysis.communities import LouvainConfig
from v1_research.analysis.pipeline import AnalysisConfig, AnalysisInputs
from v1_research.analysis.robustness import run_louvain_parameter_grid, summarize_robustness


def _inputs() -> AnalysisInputs:
    responses = np.array(
        [
            [[4.0, 4.0, 4.0], [0.2, 0.2, 0.2], [4.0, 4.0, 4.0], [0.2, 0.2, 0.2]],
            [[4.0, 4.0, 4.0], [0.2, 0.2, 0.2], [4.0, 4.0, 4.0], [0.2, 0.2, 0.2]],
            [[0.2, 0.2, 0.2], [5.0, 5.0, 5.0], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]],
            [[0.2, 0.2, 0.2], [5.0, 5.0, 5.0], [0.2, 0.2, 0.2], [5.0, 5.0, 5.0]],
        ],
        dtype=float,
    )
    coords = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=float)
    distance = np.linalg.norm(coords[:, np.newaxis, :] - coords[np.newaxis, :, :], axis=2)
    return AnalysisInputs(
        responses=responses,
        coords=coords,
        distance=distance,
        orientation_angles=np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]),
    )


def test_run_louvain_parameter_grid_records_varied_values_and_metrics() -> None:
    cfg = AnalysisConfig(
        osi_threshold=0.0,
        filter_by_osi=False,
        louvain=LouvainConfig(num_runs=1, consensus_reps=1, min_cluster_size=1),
    )

    rows = run_louvain_parameter_grid(
        cfg,
        _inputs(),
        parameter_grid={"louvain.gamma": [0.5, 0.7]},
    )

    assert [row["louvain.gamma"] for row in rows] == [0.5, 0.7]
    assert all(row["status"] == "ok" for row in rows)
    assert all("n_ensembles" in row for row in rows)


def test_run_louvain_parameter_grid_accepts_nested_omegaconf_mapping() -> None:
    cfg = AnalysisConfig(
        osi_threshold=0.0,
        filter_by_osi=False,
        louvain=LouvainConfig(num_runs=1, consensus_reps=1, min_cluster_size=1),
    )

    rows = run_louvain_parameter_grid(
        cfg,
        _inputs(),
        parameter_grid={"louvain": {"gamma": [0.5]}},
    )

    assert rows[0]["louvain.gamma"] == 0.5


def test_summarize_robustness_counts_rows() -> None:
    summary = summarize_robustness(
        window_rows=[{"status": "ok"}],
        louvain_rows=[{"status": "ok"}, {"status": "not_enough_neurons"}],
    )

    assert summary["window_row_count"] == 1
    assert summary["louvain_row_count"] == 2
    assert summary["ok_variant_count"] == 2
