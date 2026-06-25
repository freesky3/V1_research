from __future__ import annotations

import numpy as np
import pytest

from v1_research.analysis.communities import LouvainConfig
from v1_research.analysis.diagnostics import (
    graph_health_diagnostics,
    selection_funnel,
    unclassified_diagnostics,
)
from v1_research.analysis.pipeline import AnalysisConfig


def test_selection_funnel_counts_activity_osi_sampling_and_classification() -> None:
    cfg = AnalysisConfig(osi_threshold=0.4, active_threshold=0.5, filter_by_osi=True, random_sample_fraction=1.0)
    osi = np.array([0.1, 0.4, 0.8, np.nan])
    activity = np.array([0.0, 1.0, 2.0, 3.0])
    selected = np.array([1, 2])
    labels = np.array([1, 0])

    rows, summary = selection_funnel(
        osi=osi,
        activity=activity,
        selected_indices=selected,
        labels=labels,
        cfg=cfg,
    )

    assert summary["total_candidates"] == 4
    assert summary["active_candidates"] == 3
    assert summary["finite_osi_candidates"] == 3
    assert summary["osi_pass_candidates"] == 2
    assert summary["selected_neurons"] == 2
    assert summary["classified_neurons"] == 1
    assert summary["unclassified_selected_neurons"] == 1
    assert rows[0]["stage"] == "total_candidates"


def test_graph_health_diagnostics_reports_density_and_degree_distribution() -> None:
    similarity = np.array(
        [
            [0.0, 0.9, 0.0],
            [0.9, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    metrics = graph_health_diagnostics(similarity, LouvainConfig(thr_prop=0.5, min_module_degree=1.0))

    assert metrics["selected_neurons"] == 3
    assert metrics["positive_similarity_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["thresholded_edge_density"] == pytest.approx(1.0 / 3.0)
    assert metrics["degree_min"] == 0
    assert metrics["isolated_node_count"] == 1


def test_unclassified_diagnostics_explains_population_and_louvain_dropouts() -> None:
    cfg = AnalysisConfig(osi_threshold=0.5, active_threshold=0.5, filter_by_osi=True)
    reasons = unclassified_diagnostics(
        osi=np.array([0.1, 0.7, np.nan, 0.8]),
        activity=np.array([0.0, 1.0, 2.0, 3.0]),
        selected_indices=np.array([1, 3]),
        labels=np.array([0, 2]),
        cfg=cfg,
        community_diagnostics={"weak_module_degree_removed": 1, "small_cluster_removed": 0},
    )

    assert reasons["not_active_enough"] == 1
    assert reasons["finite_osi_unavailable"] == 1
    assert reasons["below_osi_threshold"] == 1
    assert reasons["selected_but_louvain_unclassified"] == 1
    assert reasons["weak_module_degree_removed"] == 1
