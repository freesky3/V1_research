from __future__ import annotations

import numpy as np
import pytest

from v1_research.analysis.clusters import cluster_members, labels_array, relabel_consecutive
from v1_research.analysis.communities import (
    _drop_weak_or_small_clusters,
    agreement_matrix,
    cosine_similarity_matrix,
    pearson_correlation_matrix,
)
from v1_research.analysis.spatial import cluster_spatial_metrics, distance_matrix, select_center_indices


def test_similarity_matrices_zero_diagonal_and_handle_constant_traces() -> None:
    trace = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [2.0, 2.0, 2.0],
        ]
    )

    cosine = cosine_similarity_matrix(trace)
    pearson = pearson_correlation_matrix(trace)

    assert cosine.shape == (3, 3)
    assert pearson.shape == (3, 3)
    np.testing.assert_allclose(np.diag(cosine), 0.0)
    np.testing.assert_allclose(np.diag(pearson), 0.0)
    assert np.all(np.isfinite(cosine))
    assert np.all(np.isfinite(pearson))


def test_agreement_matrix_counts_shared_labels_across_partitions() -> None:
    partitions = np.array(
        [
            [1, 1, 2],
            [1, 2, 2],
            [2, 2, 3],
        ]
    )

    agreement = agreement_matrix(partitions)

    expected = np.array(
        [
            [3.0, 2.0, 0.0],
            [2.0, 3.0, 1.0],
            [0.0, 1.0, 3.0],
        ]
    )
    np.testing.assert_allclose(agreement, expected)


def test_labels_and_spatial_metrics_use_nonzero_communities() -> None:
    labels = relabel_consecutive(np.array([10, 10, 0, 20]))
    coords = np.array(
        [
            [0.0, 0.0],
            [3.0, 4.0],
            [0.0, 2.0],
            [1.0, 1.0],
        ]
    )

    assert labels.tolist() == [1, 1, 0, 2]
    np.testing.assert_array_equal(labels_array(labels, n_neurons=4), labels)
    assert set(cluster_members(labels)) == {1, 2}

    distances = distance_matrix(coords)
    metrics = cluster_spatial_metrics(labels, distances)

    assert metrics[1]["mean_pairwise_distance"] == pytest.approx(5.0)
    assert metrics[1]["nearest_neighbor_distance"] == pytest.approx(5.0)
    assert 2 not in metrics


def test_select_center_indices_returns_middle_square() -> None:
    np.testing.assert_array_equal(select_center_indices(4, side_fraction=0.5), np.array([5, 6, 9, 10]))


def test_louvain_cleanup_reports_weak_degree_and_small_cluster_counts() -> None:
    labels = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    graph_binary = np.array(
        [
            [False, True, False, False, False],
            [True, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
            [False, False, False, False, False],
        ]
    )

    cleaned, diagnostics = _drop_weak_or_small_clusters(
        labels,
        graph_binary,
        min_module_degree=1.0,
        min_cluster_size=2,
    )

    np.testing.assert_array_equal(np.nan_to_num(cleaned, nan=0.0), np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
    assert diagnostics["weak_module_degree_removed"] == 2
    assert diagnostics["small_cluster_removed"] == 1
