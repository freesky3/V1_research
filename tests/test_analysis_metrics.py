from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from v1_research.analysis.metrics import (
    activity_health_metrics,
    osi_distribution_metrics,
    summarize_communities,
    write_analysis_metrics,
)


def test_activity_health_metrics_summarize_per_neuron_activity() -> None:
    responses = np.array(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[1.0, 3.0], [1.0, 3.0]],
            [[2.0, 2.0], [2.0, 2.0]],
        ]
    )

    metrics = activity_health_metrics(responses, active_threshold=0.5)

    assert metrics["active_neuron_count"] == 2
    assert metrics["active_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["silent_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["rate_mean"] == pytest.approx(4.0 / 3.0)
    assert metrics["top1_activity_fraction"] == pytest.approx(0.5)


def test_osi_distribution_metrics_counts_finite_thresholds() -> None:
    metrics = osi_distribution_metrics(np.array([0.1, 0.4, 0.5, np.nan]))

    assert metrics["n_neurons"] == 4
    assert metrics["osi_finite_count"] == 3
    assert metrics["osi_fraction_gt_0_4"] == pytest.approx(1.0 / 3.0)
    assert metrics["osi_count_gt_0_5"] == 0


def test_summarize_communities_combines_similarity_spatial_and_tuning() -> None:
    labels = np.array([1, 1, 2, 0])
    similarity = np.array(
        [
            [0.0, 0.8, 0.2, 0.1],
            [0.8, 0.0, 0.3, 0.2],
            [0.2, 0.3, 0.0, 0.1],
            [0.1, 0.2, 0.1, 0.0],
        ]
    )
    distance = np.array(
        [
            [0.0, 2.0, 4.0, 5.0],
            [2.0, 0.0, 3.0, 6.0],
            [4.0, 3.0, 0.0, 7.0],
            [5.0, 6.0, 7.0, 0.0],
        ]
    )
    coords = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0], [6.0, 0.0]])
    osi = np.array([0.5, 0.7, 0.2, 0.1])
    preferred = np.array([0.0, 0.0, np.pi / 2.0, np.nan])

    summary, rows = summarize_communities(
        labels,
        similarity=similarity,
        distance=distance,
        coords=coords,
        osi=osi,
        preferred_orientation=preferred,
    )

    assert summary["n_ensembles"] == 2
    assert summary["classified_neurons"] == 3
    assert rows[0]["ensemble_id"] == 1
    assert rows[0]["size"] == 2
    assert rows[0]["within_similarity_mean"] == pytest.approx(0.8)
    assert rows[0]["mean_pairwise_distance"] == pytest.approx(2.0)
    assert rows[0]["member_preferred_orientation_coherence"] == pytest.approx(1.0)


def test_write_analysis_metrics_writes_json_and_csv(tmp_path) -> None:
    summary_path, rows_path = write_analysis_metrics(
        {"n_ensembles": 1, "classified_fraction": 0.5},
        [{"ensemble_id": 1, "size": 2, "centroid_x": 0.5}],
        tmp_path,
    )

    assert json.loads(summary_path.read_text(encoding="utf-8"))["n_ensembles"] == 1
    with rows_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["ensemble_id"] == "1"
    assert rows[0]["size"] == "2"
