"""Diagnostic summaries for analysis selection, graph health, and dropouts."""

from __future__ import annotations

from typing import Any, Mapping

import bct
import numpy as np
from numpy.typing import ArrayLike

from v1_research.analysis.communities import LouvainConfig
from v1_research.analysis.pipeline import AnalysisConfig
from v1_research.runs import json_ready


def selection_funnel(
    *,
    osi: ArrayLike,
    activity: ArrayLike,
    selected_indices: ArrayLike,
    labels: ArrayLike | None,
    cfg: AnalysisConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Counts neurons through the analysis selection and classification stages."""

    osi_values, activity_values, selected = _population_inputs(osi, activity, selected_indices)
    label_values = _selected_labels(labels, selected.size)
    finite_osi = np.isfinite(osi_values)
    active = np.isfinite(activity_values) & (activity_values > float(cfg.active_threshold))
    osi_pass = finite_osi & (osi_values >= float(cfg.osi_threshold))
    classified = label_values != 0
    summary = {
        "total_candidates": int(osi_values.size),
        "active_candidates": int(np.sum(active)),
        "finite_osi_candidates": int(np.sum(finite_osi)),
        "osi_pass_candidates": int(np.sum(osi_pass)),
        "selected_neurons": int(selected.size),
        "classified_neurons": int(np.sum(classified)),
        "unclassified_selected_neurons": int(np.sum(~classified)),
        "selection_filter": "osi" if cfg.filter_by_osi else "activity",
        "osi_threshold": float(cfg.osi_threshold),
        "active_threshold": float(cfg.active_threshold),
        "random_sample_fraction": float(cfg.random_sample_fraction),
    }
    rows = [
        _stage_row("total_candidates", summary["total_candidates"]),
        _stage_row("active_candidates", summary["active_candidates"]),
        _stage_row("finite_osi_candidates", summary["finite_osi_candidates"]),
        _stage_row("osi_pass_candidates", summary["osi_pass_candidates"]),
        _stage_row("selected_neurons", summary["selected_neurons"]),
        _stage_row("classified_neurons", summary["classified_neurons"]),
        _stage_row("unclassified_selected_neurons", summary["unclassified_selected_neurons"]),
    ]
    return json_ready(rows), json_ready(summary)


def graph_health_diagnostics(similarity: ArrayLike | None, cfg: LouvainConfig) -> dict[str, Any]:
    """Summarizes the positive and thresholded graph used by Louvain."""

    if similarity is None:
        return json_ready(_empty_graph_metrics())
    values = np.asarray(similarity, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("similarity must be a square matrix.")
    if not np.all(np.isfinite(values)):
        raise ValueError("similarity contains NaN or infinite values.")

    n_nodes = int(values.shape[0])
    n_pairs = n_nodes * (n_nodes - 1) // 2
    upper = np.triu_indices(n_nodes, k=1)
    positive = values[upper] > 0.0 if n_pairs else np.array([], dtype=bool)
    graph_similarity = np.maximum(values, 0.0)
    graph = bct.threshold_proportional(graph_similarity, float(cfg.thr_prop)) if n_nodes else graph_similarity
    graph = bct.weight_conversion(graph, "normalize") if np.any(graph) else graph
    graph_binary = graph > 0.0
    degree = np.sum(graph_binary, axis=1).astype(float) if n_nodes else np.array([], dtype=float)
    edge_count = int(np.sum(graph_binary[upper])) if n_pairs else 0
    return json_ready(
        {
            "similarity_kind": cfg.similarity_kind,
            "selected_neurons": n_nodes,
            "positive_similarity_fraction": _fraction(int(np.sum(positive)), n_pairs),
            "thresholded_edge_density": _fraction(edge_count, n_pairs),
            "thresholded_edge_count": edge_count,
            "degree_mean": _safe_stat(degree, np.mean),
            "degree_median": _safe_stat(degree, np.median),
            "degree_p05": _percentile(degree, 5),
            "degree_p95": _percentile(degree, 95),
            "degree_min": int(np.min(degree)) if degree.size else None,
            "degree_max": int(np.max(degree)) if degree.size else None,
            "isolated_node_count": int(np.sum(degree == 0.0)) if degree.size else 0,
            "weak_module_degree_candidate_count": int(np.sum(degree < float(cfg.min_module_degree))) if degree.size else 0,
            "thr_prop": float(cfg.thr_prop),
            "min_module_degree": float(cfg.min_module_degree),
        }
    )


def unclassified_diagnostics(
    *,
    osi: ArrayLike,
    activity: ArrayLike,
    selected_indices: ArrayLike,
    labels: ArrayLike | None,
    cfg: AnalysisConfig,
    community_diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Counts why cells were not selected or not classified by Louvain."""

    osi_values, activity_values, selected = _population_inputs(osi, activity, selected_indices)
    label_values = _selected_labels(labels, selected.size)
    finite_osi = np.isfinite(osi_values)
    active = np.isfinite(activity_values) & (activity_values > float(cfg.active_threshold))
    osi_pass = finite_osi & (osi_values >= float(cfg.osi_threshold))
    selected_mask = np.zeros(osi_values.size, dtype=bool)
    selected_mask[selected] = True
    if cfg.filter_by_osi:
        filter_candidates = osi_pass
    else:
        filter_candidates = active
    diagnostics = {
        "not_active_enough": int(np.sum(~active)),
        "finite_osi_unavailable": int(np.sum(~finite_osi)),
        "below_osi_threshold": int(np.sum(finite_osi & ~osi_pass)),
        "removed_by_random_sampling": int(np.sum(filter_candidates & ~selected_mask)),
        "not_selected_by_current_filter": int(np.sum(~filter_candidates)),
        "selected_but_louvain_unclassified": int(np.sum(label_values == 0)),
    }
    community = dict(community_diagnostics or {})
    diagnostics["weak_module_degree_removed"] = int(community.get("weak_module_degree_removed", 0) or 0)
    diagnostics["small_cluster_removed"] = int(community.get("small_cluster_removed", 0) or 0)
    return json_ready(diagnostics)


def _population_inputs(
    osi: ArrayLike,
    activity: ArrayLike,
    selected_indices: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    osi_values = np.asarray(osi, dtype=float).reshape(-1)
    activity_values = np.asarray(activity, dtype=float).reshape(-1)
    selected = np.asarray(selected_indices, dtype=np.int64).reshape(-1)
    if activity_values.shape != osi_values.shape:
        raise ValueError(f"activity must have shape {osi_values.shape}, got {activity_values.shape}.")
    if np.any((selected < 0) | (selected >= osi_values.size)):
        raise ValueError("selected_indices contains values outside the population.")
    return osi_values, activity_values, selected


def _selected_labels(labels: ArrayLike | None, n_selected: int) -> np.ndarray:
    if labels is None:
        return np.zeros(int(n_selected), dtype=np.int64)
    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    if values.shape != (int(n_selected),):
        raise ValueError(f"labels must have shape ({int(n_selected)},), got {values.shape}.")
    return values


def _stage_row(stage: str, count: int) -> dict[str, Any]:
    return {"stage": stage, "count": int(count)}


def _empty_graph_metrics() -> dict[str, Any]:
    return {
        "similarity_kind": None,
        "selected_neurons": 0,
        "positive_similarity_fraction": None,
        "thresholded_edge_density": None,
        "thresholded_edge_count": 0,
        "degree_mean": None,
        "degree_median": None,
        "degree_p05": None,
        "degree_p95": None,
        "degree_min": None,
        "degree_max": None,
        "isolated_node_count": 0,
        "weak_module_degree_candidate_count": 0,
    }


def _fraction(numerator: int, denominator: int) -> float | None:
    return None if int(denominator) == 0 else float(numerator) / float(denominator)


def _safe_stat(values: np.ndarray, fn) -> float | None:
    finite = values[np.isfinite(values)]
    return float(fn(finite)) if finite.size else None


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else None


__all__ = ["graph_health_diagnostics", "selection_funnel", "unclassified_diagnostics"]
