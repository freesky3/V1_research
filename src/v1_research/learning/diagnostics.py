"""Reusable diagnostics for learning workflows."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_research.learning.bcm import BCMState
from v1_research.learning.rules import RateBatch
from v1_research.model.state import ModelState
from v1_research.model.weights import as_connection_mask, as_dense_weights
from v1_research.runs import json_ready

PlasticBlockName = Literal["EE", "IE"]


@dataclass(frozen=True, slots=True)
class TrackedWeight:
    """One sampled plastic connection to track across training."""

    block: PlasticBlockName
    target_index: int
    source_index: int
    target_local_index: int
    source_local_index: int
    sample_index: int
    initial_weight: float


def active_rate_stats(rates: RateBatch, *, active_threshold: float = 1.0) -> dict[str, float]:
    """Summarizes excitatory and inhibitory response activity."""

    return {
        **_matrix_stats("exc", rates.exc, active_threshold=active_threshold),
        **_matrix_stats("inh", rates.inh, active_threshold=active_threshold),
    }


def plastic_weight_stats(model: ModelState) -> dict[str, float | int]:
    """Summarizes current ``E<-E`` and ``I<-E`` plastic weights."""

    weights = as_dense_weights(model.weights)
    return {
        **_block_weight_stats("W_EE", weights[np.ix_(model.layout.exc_idx, model.layout.exc_idx)]),
        **_block_weight_stats("W_IE", weights[np.ix_(model.layout.inh_idx, model.layout.exc_idx)]),
    }


def row_sum_pressure(model: ModelState, *, state: BCMState | None = None) -> dict[str, float | None]:
    """Reports row-sum pressure for BCM plastic blocks."""

    weights = as_dense_weights(model.weights)
    exc = model.layout.exc_idx
    inh = model.layout.inh_idx
    rows = {
        **_row_sum_stats("row_sum_EE", weights[np.ix_(exc, exc)]),
        **_row_sum_stats("row_sum_IE", weights[np.ix_(inh, exc)]),
    }
    if state is None:
        return rows
    rows.update(_cap_pressure("row_sum_EE", weights[np.ix_(exc, exc)], state.row_sum_limits.target_exc_source_exc))
    rows.update(_cap_pressure("row_sum_IE", weights[np.ix_(inh, exc)], state.row_sum_limits.target_inh_source_exc))
    return rows


def cap_fraction(values: ArrayLike, *, limit: float | ArrayLike | None, atol: float = 1.0e-8) -> dict[str, Any]:
    """Returns how many values are at or above a cap."""

    arr = _finite_array(values).reshape(-1)
    if limit is None or arr.size == 0:
        return {"count": 0, "fraction": 0.0, "limit": None}
    limits = np.asarray(limit, dtype=float)
    if limits.ndim == 0:
        capped = arr >= float(limits) - float(atol)
        return {"count": int(np.sum(capped)), "fraction": float(np.mean(capped)), "limit": float(limits)}
    flat_limits = limits.reshape(-1)
    if flat_limits.size != arr.size:
        raise ValueError(f"cap limits must have {arr.size} values, got {flat_limits.size}.")
    capped = arr >= flat_limits - float(atol)
    return {"count": int(np.sum(capped)), "fraction": float(np.mean(capped)), "limit": None}


def weight_delta_stats(before: ModelState, after: ModelState) -> dict[str, float]:
    """Summarizes plastic block weight deltas between two model states."""

    initial = as_dense_weights(before.weights, name="before.weights")
    final = as_dense_weights(after.weights, name="after.weights")
    if initial.shape != final.shape:
        raise ValueError(f"weight shapes must match, got {initial.shape} and {final.shape}.")
    exc = before.layout.exc_idx
    inh = before.layout.inh_idx
    delta = final - initial
    return {
        **_delta_stats("delta_EE", delta[np.ix_(exc, exc)]),
        **_delta_stats("delta_IE", delta[np.ix_(inh, exc)]),
    }


def theta_stats(state: BCMState) -> dict[str, float]:
    """Summarizes BCM threshold vectors."""

    return {
        **_vector_stats("theta_exc", state.theta_exc),
        **_vector_stats("theta_inh", state.theta_inh),
    }


def sample_tracked_weights(model: ModelState, *, count: int) -> list[TrackedWeight]:
    """Samples connected plastic weights using the global NumPy random state."""

    if int(count) <= 0:
        return []
    weights = as_dense_weights(model.weights)
    topology = as_connection_mask(model.connection_mask, weights.shape)
    candidates = _tracked_candidates(model, weights, topology)
    if not candidates:
        return []
    sample_size = min(int(count), len(candidates))
    chosen = np.random.choice(np.arange(len(candidates)), size=sample_size, replace=False)
    chosen.sort()
    return [replace(candidates[int(index)], sample_index=sample_index) for sample_index, index in enumerate(chosen)]


def record_tracked_weights(model: ModelState, tracked: list[TrackedWeight], *, step: int) -> list[dict[str, Any]]:
    """Reads current values for tracked plastic weights."""

    weights = as_dense_weights(model.weights)
    rows: list[dict[str, Any]] = []
    for item in tracked:
        current = float(weights[item.target_index, item.source_index])
        rows.append(
            json_ready(
                {
                    "step": int(step),
                    "block": item.block,
                    "sample_index": int(item.sample_index),
                    "target_index": int(item.target_index),
                    "source_index": int(item.source_index),
                    "target_local_index": int(item.target_local_index),
                    "source_local_index": int(item.source_local_index),
                    "initial_weight": float(item.initial_weight),
                    "current_weight": current,
                    "delta": current - float(item.initial_weight),
                }
            )
        )
    return rows


def _tracked_candidates(
    model: ModelState,
    weights: NDArray[np.float64],
    topology: NDArray[np.bool_],
) -> list[TrackedWeight]:
    candidates: list[TrackedWeight] = []
    for block, target_idx in (("EE", model.layout.exc_idx), ("IE", model.layout.inh_idx)):
        source_idx = model.layout.exc_idx
        local_rows, local_cols = np.nonzero(topology[np.ix_(target_idx, source_idx)])
        for sample_index, (row, col) in enumerate(zip(local_rows, local_cols, strict=True)):
            target = int(target_idx[row])
            source = int(source_idx[col])
            candidates.append(
                TrackedWeight(
                    block=block,
                    target_index=target,
                    source_index=source,
                    target_local_index=int(row),
                    source_local_index=int(col),
                    sample_index=len(candidates),
                    initial_weight=float(weights[target, source]),
                )
            )
    return candidates


def _matrix_stats(prefix: str, values: ArrayLike, *, active_threshold: float) -> dict[str, float]:
    arr = _finite_array(values)
    active = arr > float(active_threshold)
    return {
        f"{prefix}_mean": _safe_mean(arr),
        f"{prefix}_median": _safe_median(arr),
        f"{prefix}_max": _safe_max(arr),
        f"{prefix}_active_fraction": float(np.mean(active)) if arr.size else 0.0,
    }


def _block_weight_stats(prefix: str, values: ArrayLike) -> dict[str, float | int]:
    arr = _finite_array(values)
    connected = arr[arr != 0.0]
    return {
        f"{prefix}_nonzero": int(connected.size),
        f"{prefix}_mean": _safe_mean(connected),
        f"{prefix}_median": _safe_median(connected),
        f"{prefix}_max": _safe_max(connected),
    }


def _row_sum_stats(prefix: str, values: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values)
    rows = np.sum(np.maximum(arr, 0.0), axis=1) if arr.ndim == 2 else np.array([], dtype=float)
    return {
        f"{prefix}_mean": _safe_mean(rows),
        f"{prefix}_max": _safe_max(rows),
    }


def _cap_pressure(prefix: str, values: ArrayLike, limits: ArrayLike | None) -> dict[str, float | None]:
    if limits is None:
        return {f"{prefix}_cap_fraction": None, f"{prefix}_cap_max_ratio": None}
    rows = np.sum(np.maximum(_finite_array(values), 0.0), axis=1)
    cap = np.asarray(limits, dtype=float).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(rows, cap, out=np.zeros_like(rows), where=cap > 0.0)
    capped = ratio >= 1.0 - 1.0e-8
    return {
        f"{prefix}_cap_fraction": float(np.mean(capped)) if capped.size else 0.0,
        f"{prefix}_cap_max_ratio": _safe_max(ratio),
    }


def _delta_stats(prefix: str, values: ArrayLike) -> dict[str, float]:
    arr = _finite_array(values)
    return {
        f"{prefix}_mean": _safe_mean(arr),
        f"{prefix}_abs_mean": _safe_mean(np.abs(arr)),
        f"{prefix}_max": _safe_max(arr),
        f"{prefix}_abs_max": _safe_max(np.abs(arr)),
    }


def _vector_stats(prefix: str, values: ArrayLike) -> dict[str, float]:
    arr = _finite_array(values).reshape(-1)
    return {
        f"{prefix}_mean": _safe_mean(arr),
        f"{prefix}_median": _safe_median(arr),
    }


def _finite_array(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("diagnostic values contain NaN or infinite values.")
    return arr


def _safe_mean(values: ArrayLike) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.mean(arr))


def _safe_median(values: ArrayLike) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.median(arr))


def _safe_max(values: ArrayLike) -> float:
    arr = np.asarray(values, dtype=float)
    return 0.0 if arr.size == 0 else float(np.max(arr))


__all__ = [
    "TrackedWeight",
    "active_rate_stats",
    "cap_fraction",
    "plastic_weight_stats",
    "record_tracked_weights",
    "row_sum_pressure",
    "sample_tracked_weights",
    "theta_stats",
    "weight_delta_stats",
]
