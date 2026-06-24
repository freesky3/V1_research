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


@dataclass(frozen=True, slots=True)
class TrainingHealthConfig:
    """Thresholds for train-time health reporting."""

    min_active_neuron_fraction: float = 0.05
    max_active_neuron_fraction: float = 0.95
    max_top1_activity_fraction: float = 0.35
    max_top5_activity_fraction: float = 0.75
    max_row_sum_cap_fraction: float = 0.05
    max_row_sum_cap_ratio: float = 0.95
    near_rate_cap_ratio: float = 0.95
    max_near_rate_cap_fraction: float = 0.05


def active_rate_stats(rates: RateBatch, *, active_threshold: float = 1.0) -> dict[str, float]:
    """Summarizes excitatory and inhibitory response activity."""

    return {
        **_matrix_stats("exc", rates.exc, active_threshold=active_threshold),
        **_matrix_stats("inh", rates.inh, active_threshold=active_threshold),
    }


def extended_active_rate_stats(
    rates: RateBatch,
    *,
    active_threshold: float = 1.0,
    near_rate_cap: float | None = None,
) -> dict[str, float | None]:
    """Returns population activity, concentration, and external-drive stats."""

    stats: dict[str, float | None] = {
        **active_rate_stats(rates, active_threshold=active_threshold),
        **_population_activity_stats("exc", rates.exc, active_threshold=active_threshold, near_rate_cap=near_rate_cap),
        **_population_activity_stats("inh", rates.inh, active_threshold=active_threshold, near_rate_cap=near_rate_cap),
    }
    if rates.external is not None:
        stats.update(_array_distribution_stats("external", rates.external))
    return stats


def plastic_weight_stats(model: ModelState) -> dict[str, float | int]:
    """Summarizes current ``E<-E`` and ``I<-E`` plastic weights."""

    weights = as_dense_weights(model.weights)
    return {
        **_block_weight_stats("W_EE", weights[np.ix_(model.layout.exc_idx, model.layout.exc_idx)]),
        **_block_weight_stats("W_IE", weights[np.ix_(model.layout.inh_idx, model.layout.exc_idx)]),
    }


def extended_plastic_weight_stats(
    prefix: str,
    values: ArrayLike,
    *,
    previous: ArrayLike | None = None,
    row_sum_limits: ArrayLike | None = None,
) -> dict[str, float | int | None]:
    """Summarizes connected weights, signed deltas, and row-sum pressure."""

    arr = _finite_array(values)
    connected = arr[arr != 0.0]
    stats: dict[str, float | int | None] = {
        **_block_weight_stats(prefix, arr),
        **_connected_distribution_stats(prefix, connected),
        **_row_sum_distribution_stats(prefix, arr),
    }
    if connected.size == 0:
        stats[f"{prefix}_mean"] = None
        stats[f"{prefix}_median"] = None
        stats[f"{prefix}_max"] = None

    if row_sum_limits is not None:
        stats.update(_row_sum_cap_ratio_stats(prefix, arr, row_sum_limits))
    else:
        stats[f"{prefix}_row_sum_cap_max_ratio"] = None
        stats[f"{prefix}_row_sum_cap_fraction"] = None

    if previous is not None:
        before = _finite_array(previous)
        if before.shape != arr.shape:
            raise ValueError(f"previous shape {before.shape} does not match values shape {arr.shape}.")
        stats.update(_connected_delta_sign_stats(prefix, before, arr))
    return stats


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


def theta_distribution_stats(state: BCMState) -> dict[str, float | None]:
    """Returns BCM theta distribution percentiles for E and I populations."""

    return {
        **_vector_distribution_stats("theta_exc", state.theta_exc),
        **_vector_distribution_stats("theta_inh", state.theta_inh),
    }


def bcm_signal_stats(rates: RateBatch, state: BCMState, cfg) -> dict[str, float | None]:
    """Summarizes BCM ``y * (y - theta)`` learning signal."""

    theta_eps = float(getattr(cfg, "theta_eps", 1.0e-6))
    return {
        **_bcm_population_signal_stats("bcm_exc", rates.exc, state.theta_exc, theta_eps=theta_eps),
        **_bcm_population_signal_stats("bcm_inh", rates.inh, state.theta_inh, theta_eps=theta_eps),
    }


def evaluate_training_health(
    diagnostics: list[dict[str, object]],
    cfg: TrainingHealthConfig = TrainingHealthConfig(),
) -> dict[str, Any]:
    """Builds a structured train-health report from diagnostic rows."""

    events: list[dict[str, Any]] = []
    for row in diagnostics:
        step = int(row.get("step", 0) or 0)
        _append_active_health_events(events, row, step, cfg, prefix="exc")
        _append_active_health_events(events, row, step, cfg, prefix="inh")
        _append_upper_bound_event(
            events,
            row,
            step,
            metric="exc_top1_activity_fraction",
            threshold=cfg.max_top1_activity_fraction,
            rule="max_top1_activity_fraction",
        )
        _append_upper_bound_event(
            events,
            row,
            step,
            metric="exc_top5_activity_fraction",
            threshold=cfg.max_top5_activity_fraction,
            rule="max_top5_activity_fraction",
        )
        for prefix in ("exc", "inh"):
            _append_upper_bound_event(
                events,
                row,
                step,
                metric=f"{prefix}_near_rate_cap_fraction",
                threshold=cfg.max_near_rate_cap_fraction,
                rule="max_near_rate_cap_fraction",
            )
        for block in ("EE", "IE"):
            _append_upper_bound_event(
                events,
                row,
                step,
                metric=f"row_sum_{block}_cap_fraction",
                threshold=cfg.max_row_sum_cap_fraction,
                rule="max_row_sum_cap_fraction",
            )
            _append_upper_bound_event(
                events,
                row,
                step,
                metric=f"row_sum_{block}_cap_max_ratio",
                threshold=cfg.max_row_sum_cap_ratio,
                rule="max_row_sum_cap_ratio",
            )

    warning_events = [event for event in events if event["severity"] == "warn"]
    failure_events = [event for event in events if event["severity"] == "fail"]
    status = "fail" if failure_events else "warn" if warning_events else "ok"
    return json_ready(
        {
            "schema_version": 1,
            "status": status,
            "thresholds": cfg,
            "warning_count": len(warning_events),
            "failure_count": len(failure_events),
            "first_warning_step": _first_event_step(warning_events),
            "first_failure_step": _first_event_step(failure_events),
            "final_metrics": _final_metrics(diagnostics),
            "worst_metrics": _worst_metrics(diagnostics, cfg),
            "events": events,
        }
    )


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


def _population_activity_stats(
    prefix: str,
    values: ArrayLike,
    *,
    active_threshold: float,
    near_rate_cap: float | None,
) -> dict[str, float | None]:
    arr = _finite_array(values)
    stats = _array_distribution_stats(prefix, arr)
    if arr.size == 0 or arr.shape[-1] == 0:
        stats.update(
            {
                f"{prefix}_active_fraction": None,
                f"{prefix}_active_neuron_count": 0,
                f"{prefix}_active_neuron_fraction": None,
                f"{prefix}_silent_neuron_fraction": None,
                f"{prefix}_top1_activity_fraction": None,
                f"{prefix}_top5_activity_fraction": None,
                f"{prefix}_near_rate_cap_fraction": None,
            }
        )
        return stats

    per_neuron = np.mean(arr, axis=0)
    active = per_neuron > float(active_threshold)
    nonnegative = np.maximum(per_neuron, 0.0)
    total = float(np.sum(nonnegative))
    sorted_activity = np.sort(nonnegative)[::-1]
    stats.update(
        {
            f"{prefix}_active_neuron_count": int(np.sum(active)),
            f"{prefix}_active_neuron_fraction": float(np.mean(active)),
            f"{prefix}_silent_neuron_fraction": float(np.mean(~active)),
            f"{prefix}_top1_activity_fraction": (
                float(sorted_activity[0] / total) if total > 0.0 and sorted_activity.size else None
            ),
            f"{prefix}_top5_activity_fraction": (
                float(np.sum(sorted_activity[: min(5, sorted_activity.size)]) / total) if total > 0.0 else None
            ),
            f"{prefix}_near_rate_cap_fraction": (
                float(np.mean(arr >= float(near_rate_cap))) if near_rate_cap is not None else None
            ),
        }
    )
    return stats


def _block_weight_stats(prefix: str, values: ArrayLike) -> dict[str, float | int]:
    arr = _finite_array(values)
    connected = arr[arr != 0.0]
    return {
        f"{prefix}_nonzero": int(connected.size),
        f"{prefix}_mean": _safe_mean(connected),
        f"{prefix}_median": _safe_median(connected),
        f"{prefix}_max": _safe_max(connected),
    }


def _connected_distribution_stats(prefix: str, values: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values).reshape(-1)
    return {
        f"{prefix}_p05": _safe_percentile(arr, 5),
        f"{prefix}_p95": _safe_percentile(arr, 95),
    }


def _row_sum_distribution_stats(prefix: str, values: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values)
    rows = np.sum(np.maximum(arr, 0.0), axis=1) if arr.ndim == 2 else np.array([], dtype=float)
    return {
        f"{prefix}_row_sum_mean": _safe_optional_mean(rows),
        f"{prefix}_row_sum_p95": _safe_percentile(rows, 95),
        f"{prefix}_row_sum_max": _safe_optional_max(rows),
    }


def _row_sum_cap_ratio_stats(prefix: str, values: ArrayLike, limits: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values)
    rows = np.sum(np.maximum(arr, 0.0), axis=1) if arr.ndim == 2 else np.array([], dtype=float)
    cap = np.asarray(limits, dtype=float).reshape(-1)
    if cap.size != rows.size:
        raise ValueError(f"row_sum_limits must have {rows.size} values, got {cap.size}.")
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(rows, cap, out=np.zeros_like(rows), where=cap > 0.0)
    capped = ratio >= 1.0 - 1.0e-8
    return {
        f"{prefix}_row_sum_cap_max_ratio": _safe_optional_max(ratio),
        f"{prefix}_row_sum_cap_fraction": float(np.mean(capped)) if capped.size else None,
    }


def _connected_delta_sign_stats(prefix: str, before: NDArray[np.float64], after: NDArray[np.float64]) -> dict[str, float | None]:
    connected = (before != 0.0) | (after != 0.0)
    delta = (after - before)[connected]
    if delta.size == 0:
        return {
            f"{prefix}_delta_positive_fraction": None,
            f"{prefix}_delta_negative_fraction": None,
            f"{prefix}_delta_zero_fraction": None,
        }
    return {
        f"{prefix}_delta_positive_fraction": float(np.mean(delta > 0.0)),
        f"{prefix}_delta_negative_fraction": float(np.mean(delta < 0.0)),
        f"{prefix}_delta_zero_fraction": float(np.mean(delta == 0.0)),
    }


def _row_sum_stats(prefix: str, values: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values)
    rows = np.sum(np.maximum(arr, 0.0), axis=1) if arr.ndim == 2 else np.array([], dtype=float)
    return {
        f"{prefix}_mean": _safe_optional_mean(rows),
        f"{prefix}_max": _safe_optional_max(rows),
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
        f"{prefix}_cap_fraction": float(np.mean(capped)) if capped.size else None,
        f"{prefix}_cap_max_ratio": _safe_optional_max(ratio),
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


def _vector_distribution_stats(prefix: str, values: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values).reshape(-1)
    return {
        f"{prefix}_mean": _safe_optional_mean(arr),
        f"{prefix}_median": _safe_optional_median(arr),
        f"{prefix}_p05": _safe_percentile(arr, 5),
        f"{prefix}_p95": _safe_percentile(arr, 95),
    }


def _array_distribution_stats(prefix: str, values: ArrayLike) -> dict[str, float | None]:
    arr = _finite_array(values).reshape(-1)
    return {
        f"{prefix}_mean": _safe_optional_mean(arr),
        f"{prefix}_median": _safe_optional_median(arr),
        f"{prefix}_p05": _safe_percentile(arr, 5),
        f"{prefix}_p95": _safe_percentile(arr, 95),
        f"{prefix}_max": _safe_optional_max(arr),
    }


def _bcm_population_signal_stats(
    prefix: str,
    rates: ArrayLike,
    theta: ArrayLike,
    *,
    theta_eps: float,
) -> dict[str, float | None]:
    y = _finite_array(rates)
    theta_arr = np.maximum(_finite_array(theta).reshape(-1), float(theta_eps))
    if y.size == 0 or y.shape[-1] == 0:
        return {
            f"{prefix}_above_theta_fraction": None,
            f"{prefix}_signal_mean": None,
            f"{prefix}_signal_abs_mean": None,
        }
    if y.shape[1] != theta_arr.size:
        raise ValueError(f"rates width {y.shape[1]} does not match theta width {theta_arr.size}.")
    signal = y * (y - theta_arr[np.newaxis, :])
    return {
        f"{prefix}_above_theta_fraction": float(np.mean(y > theta_arr[np.newaxis, :])),
        f"{prefix}_signal_mean": _safe_mean(signal),
        f"{prefix}_signal_abs_mean": _safe_mean(np.abs(signal)),
    }


def _append_active_health_events(
    events: list[dict[str, Any]],
    row: dict[str, object],
    step: int,
    cfg: TrainingHealthConfig,
    *,
    prefix: str,
) -> None:
    metric = f"{prefix}_active_neuron_fraction"
    value = _maybe_float(row.get(metric))
    if value is None:
        return
    if value <= 0.0 or value >= 1.0:
        events.append(_health_event(step, "fail", metric, value, 0.0 if value <= 0.0 else 1.0, "activity_extreme"))
        return
    if value < float(cfg.min_active_neuron_fraction):
        events.append(_health_event(step, "warn", metric, value, cfg.min_active_neuron_fraction, "min_active_neuron_fraction"))
    if value > float(cfg.max_active_neuron_fraction):
        events.append(_health_event(step, "warn", metric, value, cfg.max_active_neuron_fraction, "max_active_neuron_fraction"))


def _append_upper_bound_event(
    events: list[dict[str, Any]],
    row: dict[str, object],
    step: int,
    *,
    metric: str,
    threshold: float,
    rule: str,
) -> None:
    value = _maybe_float(row.get(metric))
    if value is None:
        return
    if value > float(threshold):
        events.append(_health_event(step, "warn", metric, value, float(threshold), rule))


def _health_event(step: int, severity: str, metric: str, value: float, threshold: float, rule: str) -> dict[str, Any]:
    return {
        "step": int(step),
        "severity": severity,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "rule": rule,
        "message": f"{metric} triggered {rule}",
    }


def _first_event_step(events: list[dict[str, Any]]) -> int | None:
    if not events:
        return None
    return int(min(int(event["step"]) for event in events))


def _final_metrics(rows: list[dict[str, object]]) -> dict[str, Any]:
    if not rows:
        return {}
    return _numeric_row(rows[-1])


def _worst_metrics(rows: list[dict[str, object]], cfg: TrainingHealthConfig) -> dict[str, Any]:
    numeric_rows = [_numeric_row(row) for row in rows]
    keys = {key for row in numeric_rows for key in row}
    worst: dict[str, float] = {}
    for key in keys:
        values = [row[key] for row in numeric_rows if key in row]
        if not values:
            continue
        if key.endswith("active_neuron_fraction"):
            minimum = min(values)
            maximum = max(values)
            worst[key] = minimum if minimum < cfg.min_active_neuron_fraction else maximum
        else:
            worst[key] = max(values)
    return worst


def _numeric_row(row: dict[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in row.items():
        converted = _maybe_float(value)
        if converted is not None:
            values[str(key)] = converted
    return values


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


def _safe_percentile(values: ArrayLike, percentile: float) -> float | None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return None if arr.size == 0 else float(np.percentile(arr, percentile))


def _safe_optional_mean(values: ArrayLike) -> float | None:
    arr = np.asarray(values, dtype=float)
    return None if arr.size == 0 else float(np.mean(arr))


def _safe_optional_median(values: ArrayLike) -> float | None:
    arr = np.asarray(values, dtype=float)
    return None if arr.size == 0 else float(np.median(arr))


def _safe_optional_max(values: ArrayLike) -> float | None:
    arr = np.asarray(values, dtype=float)
    return None if arr.size == 0 else float(np.max(arr))


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


__all__ = [
    "TrainingHealthConfig",
    "TrackedWeight",
    "active_rate_stats",
    "bcm_signal_stats",
    "cap_fraction",
    "evaluate_training_health",
    "extended_active_rate_stats",
    "extended_plastic_weight_stats",
    "plastic_weight_stats",
    "record_tracked_weights",
    "row_sum_pressure",
    "sample_tracked_weights",
    "theta_distribution_stats",
    "theta_stats",
    "weight_delta_stats",
]
