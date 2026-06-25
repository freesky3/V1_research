"""Health diagnostics and figures for drifting-grating simulations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_research.dynamics import RateResult
from v1_research.inputs.background import BackgroundTrace
from v1_research.runs import json_ready


@dataclass(frozen=True, slots=True)
class SimulationHealthConfig:
    """Thresholds and windows for simulation health reporting."""

    active_rate_threshold: float = 1.0
    min_active_fraction: float = 0.05
    max_active_fraction: float = 0.95
    max_top1_activity_fraction: float = 0.35
    max_top5_activity_fraction: float = 0.75
    near_rate_cap_ratio: float = 0.95
    max_near_rate_cap_fraction: float = 0.05
    stability_window_fraction: float = 0.25
    tail_fraction: float = 1.0 / 3.0
    max_relative_mean_drift: float = 0.25


def compute_simulation_health(
    rates: RateResult,
    *,
    stimulus_trace: ArrayLike | None = None,
    background_trace: BackgroundTrace | None = None,
    rate_max: float | None = None,
    cfg: SimulationHealthConfig = SimulationHealthConfig(),
) -> dict[str, Any]:
    """Builds a structured health report from full simulation trajectories."""

    if rates.exc_trajectory is None or rates.inh_trajectory is None:
        raise ValueError("simulation health requires solver.store_trajectory=True.")

    near_rate_cap = None if rate_max is None else float(rate_max) * float(cfg.near_rate_cap_ratio)
    metrics: dict[str, Any] = {}
    metrics.update(_population_metrics("exc", rates.exc_trajectory, cfg=cfg, near_rate_cap=near_rate_cap))
    metrics.update(_population_metrics("inh", rates.inh_trajectory, cfg=cfg, near_rate_cap=near_rate_cap))
    if stimulus_trace is not None:
        metrics.update(_distribution_metrics("stimulus", stimulus_trace))
    if background_trace is not None:
        metrics.update(_distribution_metrics("background_exc", background_trace.exc))
        metrics.update(_distribution_metrics("background_inh", background_trace.inh))
    else:
        metrics.update(_empty_distribution_metrics("background_exc"))
        metrics.update(_empty_distribution_metrics("background_inh"))

    events = _health_events(metrics, cfg)
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
            "metrics": metrics,
            "events": events,
        }
    )


def save_simulation_figures(
    run_dir: str | Path,
    rates: RateResult,
    orientation_angles: NDArray[np.float64],
    health_report: dict[str, Any],
) -> dict[str, Path]:
    """Writes core simulation inspection figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    metrics = health_report.get("metrics", {})
    metrics = metrics if isinstance(metrics, dict) else {}

    overview = figure_dir / "simulate_overview.png"
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 6.0), dpi=140)
    _plot_metric_bars(
        axes[0, 0],
        metrics,
        ["exc_active_fraction", "inh_active_fraction", "exc_silent_fraction", "inh_silent_fraction"],
        "Activity",
    )
    _plot_metric_bars(
        axes[0, 1],
        metrics,
        [
            "exc_top1_activity_fraction",
            "exc_top5_activity_fraction",
            "inh_top1_activity_fraction",
            "inh_top5_activity_fraction",
        ],
        "Concentration",
    )
    _plot_metric_bars(
        axes[1, 0],
        metrics,
        ["exc_near_rate_cap_fraction", "inh_near_rate_cap_fraction"],
        "Near rate cap",
    )
    _plot_metric_bars(
        axes[1, 1],
        metrics,
        ["exc_relative_mean_drift", "inh_relative_mean_drift", "exc_step_p95_abs_change", "inh_step_p95_abs_change"],
        f"Stability: {health_report.get('status', 'unknown')}",
    )
    fig.tight_layout()
    fig.savefig(overview)
    plt.close(fig)
    paths["simulate_overview"] = overview

    heatmaps = figure_dir / "simulate_orientation_heatmaps.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=140)
    _plot_orientation_heatmap(fig, axes[0], rates.exc, orientation_angles, title="Excitatory final rates")
    _plot_orientation_heatmap(fig, axes[1], rates.inh, orientation_angles, title="Inhibitory final rates")
    fig.tight_layout()
    fig.savefig(heatmaps)
    plt.close(fig)
    paths["simulate_orientation_heatmaps"] = heatmaps

    traces = figure_dir / "simulate_traces.png"
    fig, axes = plt.subplots(2, 1, figsize=(9.0, 6.0), dpi=140)
    _plot_population_traces(axes[0], rates.time, rates.exc_trajectory, title="Excitatory mean trajectories")
    _plot_population_traces(axes[1], rates.time, rates.inh_trajectory, title="Inhibitory mean trajectories")
    fig.tight_layout()
    fig.savefig(traces)
    plt.close(fig)
    paths["simulate_traces"] = traces
    return paths


def _population_metrics(
    prefix: str,
    trajectory: ArrayLike,
    *,
    cfg: SimulationHealthConfig,
    near_rate_cap: float | None,
) -> dict[str, Any]:
    arr = _finite_trajectory(trajectory, prefix)
    metrics = _distribution_metrics(prefix, arr)
    metrics.update(_stability_metrics(prefix, arr, cfg=cfg))
    if arr.size == 0 or arr.shape[-1] == 0:
        metrics.update(
            {
                f"{prefix}_active_fraction": None,
                f"{prefix}_silent_fraction": None,
                f"{prefix}_active_neuron_count": 0,
                f"{prefix}_active_neuron_fraction": None,
                f"{prefix}_silent_neuron_fraction": None,
                f"{prefix}_top1_activity_fraction": None,
                f"{prefix}_top5_activity_fraction": None,
                f"{prefix}_near_rate_cap_fraction": None,
            }
        )
        return metrics

    active = arr > float(cfg.active_rate_threshold)
    per_neuron = np.mean(arr, axis=(0, 1))
    active_neurons = per_neuron > float(cfg.active_rate_threshold)
    nonnegative = np.maximum(per_neuron, 0.0)
    total = float(np.sum(nonnegative))
    sorted_activity = np.sort(nonnegative)[::-1]
    metrics.update(
        {
            f"{prefix}_active_fraction": float(np.mean(active)),
            f"{prefix}_silent_fraction": float(np.mean(~active)),
            f"{prefix}_active_neuron_count": int(np.sum(active_neurons)),
            f"{prefix}_active_neuron_fraction": float(np.mean(active_neurons)),
            f"{prefix}_silent_neuron_fraction": float(np.mean(~active_neurons)),
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
    return metrics


def _stability_metrics(prefix: str, trajectory: NDArray[np.float64], *, cfg: SimulationHealthConfig) -> dict[str, Any]:
    if trajectory.size == 0:
        return {
            f"{prefix}_front_mean": None,
            f"{prefix}_back_mean": None,
            f"{prefix}_tail_mean": None,
            f"{prefix}_tail_variance": None,
            f"{prefix}_mean_drift": None,
            f"{prefix}_relative_mean_drift": None,
            f"{prefix}_step_mean_abs_change": None,
            f"{prefix}_step_p95_abs_change": None,
            f"{prefix}_step_max_abs_change": None,
        }
    window = _window_size(trajectory.shape[0], cfg.stability_window_fraction)
    tail_window = _window_size(trajectory.shape[0], cfg.tail_fraction)
    front_mean = float(np.mean(trajectory[:window]))
    back_mean = float(np.mean(trajectory[-window:]))
    tail = trajectory[-tail_window:]
    mean_drift = back_mean - front_mean
    denominator = max(abs(front_mean), 1.0e-12)
    relative = mean_drift / denominator
    if trajectory.shape[0] < 2:
        step_mean = None
        step_p95 = None
        step_max = None
    else:
        step = np.abs(np.diff(trajectory, axis=0))
        step_mean = float(np.mean(step))
        step_p95 = float(np.percentile(step, 95))
        step_max = float(np.max(step))
    return {
        f"{prefix}_front_mean": front_mean,
        f"{prefix}_back_mean": back_mean,
        f"{prefix}_tail_mean": float(np.mean(tail)),
        f"{prefix}_tail_variance": float(np.var(tail)),
        f"{prefix}_mean_drift": mean_drift,
        f"{prefix}_relative_mean_drift": relative,
        f"{prefix}_step_mean_abs_change": step_mean,
        f"{prefix}_step_p95_abs_change": step_p95,
        f"{prefix}_step_max_abs_change": step_max,
    }


def _distribution_metrics(prefix: str, values: ArrayLike) -> dict[str, Any]:
    arr = _finite_array(values).reshape(-1)
    if arr.size == 0:
        return _empty_distribution_metrics(prefix)
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_p05": float(np.percentile(arr, 5)),
        f"{prefix}_p95": float(np.percentile(arr, 95)),
        f"{prefix}_max": float(np.max(arr)),
    }


def _empty_distribution_metrics(prefix: str) -> dict[str, None]:
    return {
        f"{prefix}_mean": None,
        f"{prefix}_median": None,
        f"{prefix}_p05": None,
        f"{prefix}_p95": None,
        f"{prefix}_max": None,
    }


def _health_events(metrics: dict[str, Any], cfg: SimulationHealthConfig) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for prefix in ("exc", "inh"):
        active = _maybe_float(metrics.get(f"{prefix}_active_fraction"))
        if active is not None:
            if active <= 0.0 or active >= 1.0:
                threshold = 0.0 if active <= 0.0 else 1.0
                events.append(_event("fail", f"{prefix}_active_fraction", active, threshold, "activity_extreme"))
            else:
                _append_lower_event(events, metrics, f"{prefix}_active_fraction", cfg.min_active_fraction, "min_active_fraction")
                _append_upper_event(events, metrics, f"{prefix}_active_fraction", cfg.max_active_fraction, "max_active_fraction")
        _append_upper_event(
            events,
            metrics,
            f"{prefix}_top1_activity_fraction",
            cfg.max_top1_activity_fraction,
            "max_top1_activity_fraction",
        )
        _append_upper_event(
            events,
            metrics,
            f"{prefix}_top5_activity_fraction",
            cfg.max_top5_activity_fraction,
            "max_top5_activity_fraction",
        )
        _append_upper_event(
            events,
            metrics,
            f"{prefix}_near_rate_cap_fraction",
            cfg.max_near_rate_cap_fraction,
            "max_near_rate_cap_fraction",
        )
        relative_drift = _maybe_float(metrics.get(f"{prefix}_relative_mean_drift"))
        if relative_drift is not None and abs(relative_drift) > float(cfg.max_relative_mean_drift):
            events.append(
                _event(
                    "warn",
                    f"{prefix}_relative_mean_drift",
                    relative_drift,
                    float(cfg.max_relative_mean_drift),
                    "max_relative_mean_drift",
                )
            )
    return events


def _append_lower_event(
    events: list[dict[str, Any]],
    metrics: dict[str, Any],
    metric: str,
    threshold: float,
    rule: str,
) -> None:
    value = _maybe_float(metrics.get(metric))
    if value is not None and value < float(threshold):
        events.append(_event("warn", metric, value, float(threshold), rule))


def _append_upper_event(
    events: list[dict[str, Any]],
    metrics: dict[str, Any],
    metric: str,
    threshold: float,
    rule: str,
) -> None:
    value = _maybe_float(metrics.get(metric))
    if value is not None and value > float(threshold):
        events.append(_event("warn", metric, value, float(threshold), rule))


def _event(severity: str, metric: str, value: float, threshold: float, rule: str) -> dict[str, Any]:
    return {
        "severity": severity,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "rule": rule,
        "message": f"{metric} triggered {rule}",
    }


def _finite_trajectory(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = _finite_array(values)
    if arr.ndim != 3:
        raise ValueError(f"{name} trajectory must have shape (n_time, n_orientation, n_units).")
    return arr


def _finite_array(values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("simulation health values contain NaN or infinite values.")
    return arr


def _window_size(n_time: int, fraction: float) -> int:
    if int(n_time) <= 0:
        return 0
    bounded = min(1.0, max(0.0, float(fraction)))
    return max(1, int(np.ceil(int(n_time) * bounded)))


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _plot_metric_bars(ax, metrics: dict[str, Any], names: list[str], title: str) -> None:
    labels: list[str] = []
    values: list[float] = []
    for name in names:
        value = _maybe_float(metrics.get(name))
        if value is None:
            continue
        labels.append(name.replace("_", "\n"))
        values.append(value)
    if not values:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
    else:
        ax.bar(np.arange(len(values)), values, color="#4C78A8")
        ax.set_xticks(np.arange(len(values)), labels, rotation=30, ha="right")
    ax.set_title(title)


def _plot_orientation_heatmap(fig, ax, values: ArrayLike, orientation_angles: NDArray[np.float64], *, title: str) -> None:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or arr.ndim != 2 or arr.shape[1] == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    image = ax.imshow(arr, aspect="auto", interpolation="nearest")
    ax.set_xlabel("Cell")
    ax.set_ylabel("Orientation")
    ax.set_yticks(np.arange(orientation_angles.size), [f"{angle:.2f}" for angle in orientation_angles])
    ax.set_title(title)
    fig.colorbar(image, ax=ax, shrink=0.85)


def _plot_population_traces(ax, time: NDArray[np.float64], trajectory: ArrayLike | None, *, title: str) -> None:
    if trajectory is None:
        ax.text(0.5, 0.5, "no trajectory", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    arr = np.asarray(trajectory, dtype=float)
    if arr.size == 0 or arr.ndim != 3 or arr.shape[2] == 0:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    mean_by_orientation = np.mean(arr, axis=2)
    for index in range(mean_by_orientation.shape[1]):
        ax.plot(time, mean_by_orientation[:, index], alpha=0.65)
    ax.set_xlabel("Time")
    ax.set_ylabel("Mean rate")
    ax.set_title(title)


__all__ = [
    "SimulationHealthConfig",
    "compute_simulation_health",
    "save_simulation_figures",
]
