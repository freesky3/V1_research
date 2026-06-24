"""Synaptic weight sampling and matrix utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from v1_research.data.experimental import EmpiricalWeightSamples
from v1_research.model.state import PopulationLayout


@dataclass(frozen=True, slots=True)
class BlockWeightScales:
    """Per-block multipliers for sampled synaptic weights."""

    ee: float = 1.0
    ei: float = 1.08
    ex: float = 1.0
    ie: float = 1.0
    ii: float = 1.0
    ix: float = 1.0


@dataclass(frozen=True, slots=True)
class WeightConfig:
    """Global and per-block synaptic weight scale configuration."""

    base_strength: float = 3.0
    inhibitory_ratio: float = 5.5
    scales: BlockWeightScales = BlockWeightScales()


def as_dense_weights(weights: ArrayLike | sparse.spmatrix, *, name: str = "weights") -> NDArray[np.float64]:
    """Convert dense or sparse weights to a finite dense float matrix."""

    if sparse.issparse(weights):
        arr = weights.toarray().astype(float, copy=False)
    else:
        arr = np.asarray(weights, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr.astype(float, copy=True)


def as_connection_mask(
    mask: ArrayLike | sparse.spmatrix,
    shape: tuple[int, int],
    *,
    name: str = "connection_mask",
) -> NDArray[np.bool_]:
    """Convert a boolean or binary connection mask to a dense bool matrix."""

    if sparse.issparse(mask):
        arr = mask.toarray()
    else:
        arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"{name} shape {arr.shape} does not match {shape}.")
    if arr.dtype == bool:
        return arr.astype(bool, copy=True)
    numeric = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(numeric)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if not np.all((numeric == 0.0) | (numeric == 1.0)):
        raise ValueError(f"{name} must be boolean or contain only 0/1 values.")
    return numeric.astype(bool)


def validate_indices(
    name: str,
    indices: ArrayLike,
    upper_bound: int,
    *,
    allow_empty: bool = False,
) -> NDArray[np.int64]:
    """Validate a one-dimensional array of unique integer indices."""

    raw = np.asarray(indices)
    if raw.ndim != 1:
        raise ValueError(f"{name} must be a 1D index array.")
    if raw.size == 0:
        if allow_empty:
            return np.array([], dtype=np.int64)
        raise ValueError(f"{name} must not be empty.")
    idx = raw.astype(np.int64, copy=False)
    if not np.all(np.asarray(raw, dtype=float) == idx):
        raise ValueError(f"{name} must contain integer indices.")
    if np.any(idx < 0) or np.any(idx >= int(upper_bound)):
        raise ValueError(f"{name} contains out-of-range indices.")
    if np.unique(idx).size != idx.size:
        raise ValueError(f"{name} contains duplicate indices.")
    return idx.astype(np.int64, copy=True)


def limit_row_sums(weights: ArrayLike, row_sum_max: float | ArrayLike | None) -> NDArray[np.float64]:
    """Scale rows down so their sums do not exceed configured limits."""

    arr = as_dense_weights(weights, name="weights")
    if row_sum_max is None:
        return arr

    limits = np.asarray(row_sum_max, dtype=float)
    if limits.ndim == 0:
        limits = np.full(arr.shape[0], float(limits), dtype=float)
    if limits.shape != (arr.shape[0],):
        raise ValueError(f"row_sum_max shape {limits.shape} does not match {(arr.shape[0],)}.")
    if not np.all(np.isfinite(limits)) or np.any(limits < 0.0):
        raise ValueError("row_sum_max must contain finite non-negative values.")

    row_totals = np.sum(arr, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.divide(limits, row_totals, out=np.ones_like(row_totals), where=row_totals > 0.0)
    scale = np.minimum(scale, 1.0)
    limited = arr * scale[:, np.newaxis]
    limited[(limits == 0.0) & (row_totals > 0.0), :] = 0.0
    return limited


def row_sums(weights: ArrayLike | sparse.spmatrix) -> NDArray[np.float64]:
    """Return dense row sums for dense or sparse weights."""

    if sparse.issparse(weights):
        return np.asarray(weights.sum(axis=1)).ravel().astype(float)
    return np.sum(as_dense_weights(weights, name="weights"), axis=1)


def sample_weights(
    layout: PopulationLayout,
    connection_mask: sparse.csr_matrix,
    cfg: WeightConfig,
    empirical_samples: EmpiricalWeightSamples,
) -> sparse.csr_matrix:
    """Sample signed sparse weights for an existing topology.

    Args:
        layout: Model population layout.
        connection_mask: Boolean topology matrix.
        cfg: Weight scale configuration.
        empirical_samples: Empirical magnitude samples by block.

    Returns:
        Signed CSR weight matrix.
    """

    mask = sparse.csr_matrix(connection_mask, dtype=bool)
    if mask.shape != layout.shape:
        raise ValueError(f"connection_mask shape {mask.shape} does not match layout shape {layout.shape}.")

    rows: list[NDArray[np.int64]] = []
    cols: list[NDArray[np.int64]] = []
    data: list[NDArray[np.float64]] = []
    scales = cfg.scales
    blocks = (
        (layout.exc_idx, layout.exc_idx, +cfg.base_strength * scales.ee, empirical_samples.ee),
        (layout.exc_idx, layout.inh_idx, -cfg.base_strength * cfg.inhibitory_ratio * scales.ei, empirical_samples.ei),
        (layout.exc_idx, layout.input_idx, +cfg.base_strength * scales.ex, empirical_samples.ex),
        (layout.inh_idx, layout.exc_idx, +cfg.base_strength * scales.ie, empirical_samples.ie),
        (layout.inh_idx, layout.inh_idx, -cfg.base_strength * cfg.inhibitory_ratio * scales.ii, empirical_samples.ii),
        (layout.inh_idx, layout.input_idx, +cfg.base_strength * scales.ix, empirical_samples.ix),
    )

    for target_idx, source_idx, gain, samples in blocks:
        _append_weight_block(rows, cols, data, mask, target_idx, source_idx, gain, samples)

    if not rows:
        return sparse.csr_matrix(layout.shape, dtype=float)
    weights = sparse.coo_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=layout.shape,
        dtype=float,
    ).tocsr()
    weights.eliminate_zeros()
    return weights


def _append_weight_block(
    rows: list[NDArray[np.int64]],
    cols: list[NDArray[np.int64]],
    data: list[NDArray[np.float64]],
    mask: sparse.csr_matrix,
    target_idx: NDArray[np.int64],
    source_idx: NDArray[np.int64],
    gain: float,
    samples: NDArray[np.float64],
) -> None:
    block = mask[np.ix_(target_idx, source_idx)]
    local_rows, local_cols = block.nonzero()
    if local_rows.size == 0:
        return
    rows.append(target_idx[local_rows])
    cols.append(source_idx[local_cols])
    data.append(float(gain) * np.random.choice(samples, size=local_rows.size))
