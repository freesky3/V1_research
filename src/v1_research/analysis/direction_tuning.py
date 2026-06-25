"""Ensemble direction tuning summaries from analyzed communities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from v1_research.analysis.clusters import cluster_members, labels_array
from v1_research.runs import json_ready


@dataclass(frozen=True, slots=True)
class DirectionTuningConfig:
    """Configuration for ensemble direction-selectivity summaries."""

    enabled: bool = True
    modulation_threshold: float = 0.2


def summarize_direction_tuning(
    labels: ArrayLike,
    responses_mean: ArrayLike,
    orientation_angles: ArrayLike,
    cfg: DirectionTuningConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarizes ensemble direction preference and modulation."""

    label_values = labels_array(labels)
    responses = np.asarray(responses_mean, dtype=float)
    angles = np.asarray(orientation_angles, dtype=float).reshape(-1)
    if responses.ndim != 2:
        raise ValueError("responses_mean must have shape (n_neurons, n_direction).")
    if responses.shape[0] != label_values.size:
        raise ValueError(f"labels must have shape ({responses.shape[0]},), got {label_values.shape}.")
    if angles.shape != (responses.shape[1],):
        raise ValueError(f"orientation_angles must have shape ({responses.shape[1]},), got {angles.shape}.")
    if not np.isfinite(float(cfg.modulation_threshold)) or float(cfg.modulation_threshold) < 0.0:
        raise ValueError("modulation_threshold must be finite and non-negative.")

    rows: list[dict[str, Any]] = []
    direction_labels = [_degree_label(angle) for angle in angles]
    for ensemble_id, members in cluster_members(label_values).items():
        tuning = np.mean(responses[members], axis=0)
        finite = tuning[np.isfinite(tuning)]
        if finite.size:
            max_mean = float(np.max(finite))
            min_mean = float(np.min(finite))
            mean_rate = float(np.mean(finite))
            preferred_idx = int(np.nanargmax(tuning))
            preferred_deg = _degrees(angles[preferred_idx])
        else:
            max_mean = min_mean = mean_rate = preferred_deg = float("nan")
            preferred_idx = -1
        denom = max_mean + min_mean
        modulation = (max_mean - min_mean) / denom if np.isfinite(denom) and denom > 0.0 else 0.0
        direction_selective = bool(modulation >= float(cfg.modulation_threshold))
        row: dict[str, Any] = {
            "ensemble_id": int(ensemble_id),
            "size": int(members.size),
            "preferred_direction_index": int(preferred_idx),
            "preferred_direction_deg": float(preferred_deg),
            "max_mean_rate": max_mean,
            "min_mean_rate": min_mean,
            "mean_rate_across_directions": mean_rate,
            "modulation_index": float(modulation),
            "direction_selective": direction_selective,
        }
        for label, value in zip(direction_labels, tuning, strict=True):
            row[f"mean_rate_{label}deg"] = float(value)
        rows.append(row)

    modulations = [float(row["modulation_index"]) for row in rows]
    selective_rows = [row for row in rows if bool(row["direction_selective"])]
    covered = {
        int(row["preferred_direction_index"])
        for row in selective_rows
        if int(row["preferred_direction_index"]) >= 0
    }
    summary = {
        "direction_tuning_enabled": bool(cfg.enabled),
        "direction_modulation_threshold": float(cfg.modulation_threshold),
        "direction_tuned_ensembles": int(len(rows)),
        "direction_selective_ensembles": int(len(selective_rows)),
        "direction_selective_fraction": float(len(selective_rows) / len(rows)) if rows else None,
        "covered_direction_count": int(len(covered)),
        "min_ensemble_modulation": float(np.min(modulations)) if modulations else None,
        "mean_ensemble_modulation": float(np.mean(modulations)) if modulations else None,
    }
    return json_ready(summary), json_ready(rows)


def _degree_label(angle: float) -> str:
    degrees = _degrees(angle)
    rounded = round(degrees)
    if abs(degrees - rounded) < 1.0e-9:
        return str(int(rounded))
    return f"{degrees:.3g}"


def _degrees(angle: float) -> float:
    return float(np.degrees(float(angle)) % 360.0)
