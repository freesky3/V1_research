"""Small helpers for community label arrays."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def labels_array(labels: ArrayLike, *, n_neurons: int | None = None) -> NDArray[np.int64]:
    """Returns community labels as a validated one-dimensional integer array.

    Args:
        labels: Community labels where 0 denotes an unclassified neuron.
        n_neurons: Optional expected label count.

    Returns:
        A copy of the labels as ``int64``.
    """

    arr = np.nan_to_num(np.asarray(labels, dtype=float), nan=0.0).astype(np.int64, copy=False).reshape(-1)
    if n_neurons is not None and arr.shape != (int(n_neurons),):
        raise ValueError(f"labels must have shape ({int(n_neurons)},), got {arr.shape}.")
    if np.any(arr < 0):
        raise ValueError("labels must be non-negative; 0 is reserved for unclassified neurons.")
    return arr.copy()


def cluster_ids(labels: ArrayLike) -> list[int]:
    """Returns sorted non-zero community IDs."""

    arr = labels_array(labels)
    return [int(c_id) for c_id in np.unique(arr) if c_id != 0]


def cluster_members(labels: ArrayLike) -> dict[int, NDArray[np.int64]]:
    """Groups neuron indices by non-zero community label."""

    arr = labels_array(labels)
    return {c_id: np.flatnonzero(arr == c_id) for c_id in cluster_ids(arr)}


def relabel_consecutive(labels: ArrayLike) -> NDArray[np.int64]:
    """Maps arbitrary positive labels to ``1..n`` while leaving 0 unclassified."""

    arr = labels_array(labels)
    out = np.zeros_like(arr, dtype=np.int64)
    for new_id, old_id in enumerate(cluster_ids(arr), start=1):
        out[arr == old_id] = new_id
    return out
