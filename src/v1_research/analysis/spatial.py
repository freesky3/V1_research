"""Spatial metrics for analyzed neuron ensembles."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_research.analysis.clusters import cluster_members, labels_array


def distance_matrix(
    coords: ArrayLike,
    *,
    region_size: float | None = None,
    periodic: bool = False,
) -> NDArray[np.float64]:
    """Computes pairwise distances between two-dimensional coordinates."""

    points = np.asarray(coords, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("coords must have shape (n_neurons, 2).")
    if not np.all(np.isfinite(points)):
        raise ValueError("coords contains NaN or infinite values.")

    delta = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    if periodic:
        if region_size is None or float(region_size) <= 0.0:
            raise ValueError("region_size must be positive when periodic=True.")
        abs_delta = np.abs(delta)
        delta = np.minimum(abs_delta, float(region_size) - abs_delta)
    return np.sqrt(np.sum(delta**2, axis=2))


def select_center_indices(n_side: int, *, side_fraction: float = 0.5) -> NDArray[np.int64]:
    """Returns flattened indices in the center square of an ``n_side`` grid."""

    if int(n_side) <= 0:
        raise ValueError("n_side must be positive.")
    if not 0.0 < float(side_fraction) <= 1.0:
        raise ValueError("side_fraction must be in (0, 1].")
    center_side = max(1, int(round(int(n_side) * float(side_fraction))))
    start = (int(n_side) - center_side) // 2
    end = start + center_side
    grid = np.arange(int(n_side) * int(n_side), dtype=np.int64).reshape(int(n_side), int(n_side))
    return grid[start:end, start:end].ravel().copy()


def cluster_spatial_metrics(labels: ArrayLike, distances: ArrayLike) -> dict[int, dict[str, float]]:
    """Computes compactness metrics for non-zero communities with at least 2 cells."""

    dist = np.asarray(distances, dtype=float)
    if dist.ndim != 2 or dist.shape[0] != dist.shape[1]:
        raise ValueError("distances must be a square matrix.")
    label_values = labels_array(labels, n_neurons=dist.shape[0])

    metrics: dict[int, dict[str, float]] = {}
    for c_id, members in cluster_members(label_values).items():
        if members.size < 2:
            continue
        mean_dist, nearest = _cluster_spatial_values(dist, members)
        metrics[c_id] = {
            "mean_pairwise_distance": float(mean_dist),
            "nearest_neighbor_distance": float(np.mean(nearest)),
        }
    return metrics


def _cluster_spatial_values(
    distances: NDArray[np.float64],
    members: NDArray[np.int64],
) -> tuple[float, NDArray[np.float64]]:
    sub_dist = np.array(distances[np.ix_(members, members)], dtype=float, copy=True)
    pairwise = sub_dist[np.triu_indices_from(sub_dist, k=1)]
    pairwise = pairwise[np.isfinite(pairwise)]
    if pairwise.size == 0:
        return np.nan, np.array([], dtype=float)
    np.fill_diagonal(sub_dist, np.inf)
    nearest = np.min(sub_dist, axis=1)
    return float(np.mean(pairwise)), nearest[np.isfinite(nearest)]
