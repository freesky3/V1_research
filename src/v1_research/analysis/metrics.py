"""Population and community summary metrics for analysis results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import ArrayLike

from v1_research.analysis.clusters import cluster_members, labels_array
from v1_research.analysis.spatial import cluster_spatial_metrics
from v1_research.runs import json_ready, write_json

METRIC_SCHEMA_VERSION = 1


def activity_health_metrics(responses: ArrayLike, *, active_threshold: float = 1.0e-6) -> dict[str, Any]:
    """Computes rate statistics from per-neuron responses."""

    values = np.asarray(responses, dtype=float)
    if values.ndim == 1:
        per_neuron = values
    elif values.ndim == 2:
        per_neuron = np.nanmean(values, axis=1)
    elif values.ndim == 3:
        per_neuron = np.nanmean(values, axis=(1, 2))
    else:
        raise ValueError("responses must be 1D, 2D, or 3D.")

    per_neuron = np.nan_to_num(per_neuron, nan=0.0, posinf=0.0, neginf=0.0)
    active = per_neuron > float(active_threshold)
    nonnegative = np.maximum(per_neuron, 0.0)
    total_activity = float(np.sum(nonnegative))
    sorted_activity = np.sort(nonnegative)[::-1]
    top1 = float(sorted_activity[0] / total_activity) if total_activity > 0.0 else None
    top5 = (
        float(np.sum(sorted_activity[: min(5, sorted_activity.size)]) / total_activity)
        if total_activity > 0.0
        else None
    )
    return json_ready(
        {
            "active_threshold": float(active_threshold),
            "active_neuron_count": int(np.sum(active)),
            "active_fraction": _finite_float(np.mean(active)) if per_neuron.size else None,
            "silent_fraction": _finite_float(np.mean(~active)) if per_neuron.size else None,
            "rate_mean": _safe_mean(per_neuron),
            "rate_median": _safe_median(per_neuron),
            "rate_p95": _finite_float(np.percentile(per_neuron, 95)) if per_neuron.size else None,
            "rate_max": _finite_float(np.max(per_neuron)) if per_neuron.size else None,
            "top1_activity_fraction": top1,
            "top5_activity_fraction": top5,
        }
    )


def osi_distribution_metrics(osi: ArrayLike) -> dict[str, Any]:
    """Computes population-level OSI distribution statistics."""

    values = np.asarray(osi, dtype=float)
    finite = values[np.isfinite(values)]
    return json_ready(
        {
            "n_neurons": int(values.size),
            "osi_finite_count": int(finite.size),
            "osi_finite_fraction": _finite_float(finite.size / values.size) if values.size else None,
            "osi_mean": _safe_mean(finite),
            "osi_median": _safe_median(finite),
            "osi_std": _safe_std(finite),
            "osi_count_gt_0_2": int(np.sum(finite > 0.2)),
            "osi_count_gt_0_4": int(np.sum(finite > 0.4)),
            "osi_count_gt_0_5": int(np.sum(finite > 0.5)),
            "osi_count_gt_0_6": int(np.sum(finite > 0.6)),
            "osi_fraction_gt_0_2": _finite_float(np.mean(finite > 0.2)) if finite.size else None,
            "osi_fraction_gt_0_4": _finite_float(np.mean(finite > 0.4)) if finite.size else None,
            "osi_fraction_gt_0_5": _finite_float(np.mean(finite > 0.5)) if finite.size else None,
            "osi_fraction_gt_0_6": _finite_float(np.mean(finite > 0.6)) if finite.size else None,
        }
    )


def summarize_communities(
    labels: ArrayLike,
    *,
    similarity: ArrayLike | None = None,
    distance: ArrayLike | None = None,
    coords: ArrayLike | None = None,
    osi: ArrayLike | None = None,
    preferred_orientation: ArrayLike | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Builds global and per-community metric rows."""

    label_values = labels_array(labels)
    n_neurons = label_values.size
    classified = label_values != 0
    clusters = cluster_members(label_values)

    similarity_values = None if similarity is None else np.asarray(similarity, dtype=float)
    distance_values = None if distance is None else np.asarray(distance, dtype=float)
    coords_values = None if coords is None else np.asarray(coords, dtype=float)
    osi_values = None if osi is None else np.asarray(osi, dtype=float)
    preferred_values = None if preferred_orientation is None else np.asarray(preferred_orientation, dtype=float)
    spatial = cluster_spatial_metrics(label_values, distance_values) if distance_values is not None else {}

    rows: list[dict[str, Any]] = []
    for c_id, members in clusters.items():
        row: dict[str, Any] = {
            "ensemble_id": int(c_id),
            "size": int(members.size),
            "member_fraction": _finite_float(members.size / n_neurons) if n_neurons else None,
        }
        if similarity_values is not None:
            within = _offdiag_values(similarity_values[np.ix_(members, members)])
            outside = np.flatnonzero((label_values != c_id) & classified)
            between = similarity_values[np.ix_(members, outside)].ravel() if outside.size else []
            row["within_similarity_mean"] = _safe_mean(within)
            row["outside_similarity_mean"] = _safe_mean(between)
        if coords_values is not None:
            centroid = np.mean(coords_values[members], axis=0)
            row["centroid_x"] = _finite_float(centroid[0])
            row["centroid_y"] = _finite_float(centroid[1])
        row.update(spatial.get(c_id, {}))
        if osi_values is not None:
            row["member_osi_mean"] = _safe_mean(osi_values[members])
            row["member_osi_median"] = _safe_median(osi_values[members])
        if preferred_values is not None:
            values = preferred_values[members]
            values = values[np.isfinite(values)]
            row["member_preferred_orientation_coherence"] = (
                _finite_float(np.abs(np.mean(np.exp(2j * values)))) if values.size else None
            )
        rows.append(row)

    sizes = [row["size"] for row in rows]
    summary: dict[str, Any] = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "n_neurons": int(n_neurons),
        "n_ensembles": int(len(rows)),
        "classified_neurons": int(np.sum(classified)),
        "unclassified_neurons": int(np.sum(~classified)),
        "classified_fraction": _finite_float(np.mean(classified)) if n_neurons else None,
        "ensemble_size_mean": _safe_mean(sizes),
        "ensemble_size_median": _safe_median(sizes),
        "ensemble_size_min": int(np.min(sizes)) if sizes else None,
        "ensemble_size_max": int(np.max(sizes)) if sizes else None,
    }
    if similarity_values is not None and n_neurons > 1:
        upper = np.triu_indices(n_neurons, k=1)
        left = label_values[upper[0]]
        right = label_values[upper[1]]
        valid = (left != 0) & (right != 0)
        upper_similarity = similarity_values[upper]
        summary["within_similarity_mean"] = _safe_mean(upper_similarity[valid & (left == right)])
        summary["between_similarity_mean"] = _safe_mean(upper_similarity[valid & (left != right)])
    if osi_values is not None:
        summary.update(osi_distribution_metrics(osi_values))
    return json_ready(summary), json_ready(rows)


def write_analysis_metrics(
    summary: dict[str, Any],
    ensemble_rows: Iterable[dict[str, Any]],
    save_dir: str | Path,
) -> tuple[Path, Path]:
    """Writes analysis summary JSON and ensemble metric CSV."""

    target = Path(save_dir)
    target.mkdir(parents=True, exist_ok=True)
    summary_path = write_json(target / "metrics.json", summary)
    ensemble_path = target / "ensemble_metrics.csv"

    rows = [json_ready(row) for row in ensemble_rows]
    fieldnames = _ordered_fieldnames(rows)
    with ensemble_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return summary_path, ensemble_path


def _ordered_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "ensemble_id",
        "size",
        "member_fraction",
        "within_similarity_mean",
        "outside_similarity_mean",
        "mean_pairwise_distance",
        "nearest_neighbor_distance",
        "centroid_x",
        "centroid_y",
        "member_osi_mean",
        "member_osi_median",
        "member_preferred_orientation_coherence",
    ]
    present = {key for row in rows for key in row}
    return [key for key in preferred if key in present] + sorted(present - set(preferred))


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _safe_mean(values: Iterable[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.mean(finite)) if finite.size else None


def _safe_median(values: Iterable[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.median(finite)) if finite.size else None


def _safe_std(values: Iterable[Any]) -> float | None:
    finite = _finite_values(values)
    return float(np.std(finite, ddof=1)) if finite.size > 1 else None


def _finite_values(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray([np.nan if value is None else value for value in values], dtype=float)
    return arr[np.isfinite(arr)]


def _offdiag_values(matrix: np.ndarray) -> np.ndarray:
    if matrix.shape[0] < 2:
        return np.array([], dtype=float)
    values = matrix[np.triu_indices_from(matrix, k=1)]
    return values[np.isfinite(values)]
