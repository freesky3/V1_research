"""Temporal-window sensitivity helpers for analysis runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from v1_research.analysis.pipeline import AnalysisConfig, AnalysisInputs, run_analysis
from v1_research.runs import json_ready


def run_window_analysis(
    cfg: AnalysisConfig,
    inputs: AnalysisInputs,
    *,
    tail_fractions: Sequence[float] = (),
    end_times: Sequence[float] = (),
    time: ArrayLike | None = None,
) -> list[dict[str, Any]]:
    """Runs analysis on tail and prefix response windows."""

    inputs.validate()
    rows: list[dict[str, Any]] = []
    n_time = int(inputs.responses.shape[2])
    for fraction in tail_fractions:
        start = max(0, min(n_time - 1, int(round(n_time * (1.0 - float(fraction))))))
        window = _slice_inputs(inputs, start=start, stop=n_time)
        rows.append(_window_row("tail", f"{float(fraction):g}", start, n_time, run_analysis(cfg, window)))
    if end_times:
        if time is None:
            raise ValueError("time is required when end_times is not empty.")
        time_values = np.asarray(time, dtype=float).reshape(-1)
        if time_values.size != n_time:
            raise ValueError(f"time must have length {n_time}, got {time_values.size}.")
        for end_time in end_times:
            stop = int(np.searchsorted(time_values, float(end_time), side="right"))
            stop = max(1, min(n_time, stop))
            window = _slice_inputs(inputs, start=0, stop=stop)
            rows.append(_window_row("prefix", f"{float(end_time):g}", 0, stop, run_analysis(cfg, window)))
    return rows


def _slice_inputs(inputs: AnalysisInputs, *, start: int, stop: int) -> AnalysisInputs:
    return AnalysisInputs(
        responses=np.asarray(inputs.responses, dtype=float)[:, :, start:stop],
        coords=inputs.coords,
        distance=inputs.distance,
        orientation_angles=inputs.orientation_angles,
    )


def _window_row(kind: str, value: str, start: int, stop: int, result) -> dict[str, Any]:
    summary = result.diagnostics.get("metrics_summary", {})
    row: dict[str, Any] = {
        "window_kind": kind,
        "window_value": value,
        "time_start_index": int(start),
        "time_stop_index": int(stop),
        "time_points": int(stop - start),
        "status": result.status,
        "selected_neurons": int(result.selected_indices.size),
    }
    if isinstance(summary, dict):
        for key, metric in summary.items():
            row[str(key)] = metric
    return json_ready(row)


__all__ = ["run_window_analysis"]
