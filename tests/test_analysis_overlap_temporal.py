from __future__ import annotations

import numpy as np

from v1_research.analysis.communities import LouvainConfig
from v1_research.analysis.overlap import compare_label_sets, overlap_significance
from v1_research.analysis.pipeline import AnalysisConfig, AnalysisInputs
from v1_research.analysis.temporal import run_window_analysis


def test_compare_label_sets_matches_coords_and_reports_ari() -> None:
    reference_coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    query_coords = np.array([[2.0, 0.0], [0.0, 0.0], [5.0, 5.0]])

    result = compare_label_sets(
        reference_labels=np.array([1, 1, 2]),
        reference_coords=reference_coords,
        query_labels=np.array([2, 1, 9]),
        query_coords=query_coords,
    )

    assert result.matched_count == 2
    assert result.reference_unmatched_count == 1
    assert result.query_unmatched_count == 1
    assert result.adjusted_rand_index == 1.0
    assert result.best_matches[0]["reference_label"] == 1
    assert result.best_matches[0]["query_label"] == 1
    assert result.best_matches[0]["overlap_count"] == 1


def test_overlap_significance_uses_global_random_state() -> None:
    np.random.seed(10)

    result = overlap_significance(np.array([1, 1, 2, 2]), np.array([1, 1, 2, 2]), num_surrogates=8)

    assert result["actual_adjusted_rand_index"] == 1.0
    assert result["num_surrogates"] == 8
    assert 0.0 <= result["p_ge"] <= 1.0


def test_run_window_analysis_reuses_analysis_pipeline_for_tail_and_prefix_windows() -> None:
    responses = np.array(
        [
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]],
            [[2.0, 2.0, 2.0, 2.0], [1.0, 1.0, 1.0, 1.0]],
        ],
        dtype=float,
    )
    inputs = AnalysisInputs(
        responses=responses,
        coords=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=float),
        distance=np.array([[0.0, 1.0], [1.0, 0.0]], dtype=float),
        orientation_angles=np.array([0.0, np.pi / 2.0]),
    )

    rows = run_window_analysis(
        AnalysisConfig(
            osi_threshold=0.0,
            filter_by_osi=False,
            louvain=LouvainConfig(num_runs=2, consensus_reps=2, min_cluster_size=1),
        ),
        inputs,
        tail_fractions=(0.5,),
        end_times=(0.2,),
        time=np.array([0.0, 0.1, 0.2, 0.3]),
    )

    assert [row["window_kind"] for row in rows] == ["tail", "prefix"]
    assert rows[0]["time_points"] == 2
    assert rows[1]["time_points"] == 3
    assert all("n_ensembles" in row for row in rows)
