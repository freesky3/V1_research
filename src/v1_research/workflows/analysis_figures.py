"""Matplotlib figure writers for analysis workflow inspection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from v1_research.analysis.pipeline import AnalysisResult


def save_analysis_figures(
    run_dir: str | Path,
    result: AnalysisResult,
    *,
    selection_rows: list[dict[str, object]],
    graph_diagnostics: dict[str, object],
    unclassified_diagnostics: dict[str, object],
    orientation_angles: ArrayLike,
) -> dict[str, Path]:
    """Writes default analysis inspection figures."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis_summary_figure": figure_dir / "analysis_summary.png",
        "analysis_cortical_map_figure": figure_dir / "analysis_cortical_map.png",
        "analysis_similarity_figure": figure_dir / "analysis_similarity.png",
        "analysis_tuning_figure": figure_dir / "analysis_tuning.png",
        "analysis_failure_diagnosis_figure": figure_dir / "analysis_failure_diagnosis.png",
    }
    _save_summary(paths["analysis_summary_figure"], result, selection_rows, plt)
    _save_cortical_map(paths["analysis_cortical_map_figure"], result, plt)
    _save_similarity(paths["analysis_similarity_figure"], result, plt)
    _save_tuning(paths["analysis_tuning_figure"], result, orientation_angles, plt)
    _save_failure(paths["analysis_failure_diagnosis_figure"], selection_rows, graph_diagnostics, unclassified_diagnostics, plt)
    return paths


def save_analysis_robustness_figure(
    run_dir: str | Path,
    *,
    window_rows: list[dict[str, object]],
    louvain_rows: list[dict[str, object]],
) -> Path:
    """Writes a compact robustness summary figure."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(run_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "analysis_robustness.png"
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=140)
    _plot_metric_series(axes[0], window_rows, "time_points", ["n_ensembles", "classified_fraction"], "Window robustness")
    _plot_metric_series(axes[1], louvain_rows, None, ["n_ensembles", "classified_fraction"], "Louvain robustness")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _save_summary(path: Path, result: AnalysisResult, selection_rows: list[dict[str, object]], plt) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=140)
    _bar_rows(axes[0, 0], selection_rows, "Selection funnel")
    _hist(axes[0, 1], result.osi, "OSI distribution", "OSI")
    activity = np.mean(result.responses_mean, axis=1) if result.responses_mean.size else np.array([], dtype=float)
    _hist(axes[1, 0], activity, "Selected mean activity", "Mean response")
    labels = _labels(result)
    sizes = [int(np.sum(labels == label)) for label in np.unique(labels[labels != 0])]
    _hist(axes[1, 1], sizes, "Ensemble sizes", "Cells")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_cortical_map(path: Path, result: AnalysisResult, plt) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 7.0), dpi=140)
    coords = np.asarray(result.coords, dtype=float)
    labels = _labels(result)
    activity = np.mean(result.responses_mean, axis=1) if result.responses_mean.size else np.array([], dtype=float)
    _scatter(axes[0, 0], coords, labels, "Community label")
    _scatter(axes[0, 1], coords, result.osi, "OSI")
    _scatter(axes[1, 0], coords, result.preferred_orientation, "Preferred orientation")
    _scatter(axes[1, 1], coords, activity, "Mean activity")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_similarity(path: Path, result: AnalysisResult, plt) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), dpi=140)
    communities = result.communities
    if communities is None:
        _no_data(axes[0], "No community result")
        _no_data(axes[1], "No agreement matrix")
    else:
        labels = communities.labels
        order = _community_order(labels)
        _matrix(axes[0], communities.similarity[np.ix_(order, order)], "Similarity")
        if communities.agreement is None:
            _no_data(axes[1], "No agreement matrix")
        else:
            _matrix(axes[1], communities.agreement[np.ix_(order, order)], "Agreement")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_tuning(path: Path, result: AnalysisResult, orientation_angles: ArrayLike, plt) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.5), dpi=140)
    labels = _labels(result)
    angles = np.asarray(orientation_angles, dtype=float)
    plotted = False
    for label in np.unique(labels[labels != 0]):
        members = np.flatnonzero(labels == label)
        if members.size == 0 or result.responses_mean.size == 0:
            continue
        tuning = np.mean(result.responses_mean[members], axis=0)
        ax.plot(angles, tuning, marker="o", label=f"ensemble {int(label)}")
        plotted = True
    if not plotted:
        _no_data(ax, "No classified ensembles")
    else:
        ax.set_xlabel("Orientation (rad)")
        ax.set_ylabel("Mean response")
        ax.set_title("Ensemble tuning")
        ax.legend(loc="best", fontsize="small")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _save_failure(
    path: Path,
    selection_rows: list[dict[str, object]],
    graph_diagnostics: dict[str, object],
    unclassified_diagnostics: dict[str, object],
    plt,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.0), dpi=140)
    _bar_rows(axes[0, 0], selection_rows, "Selection funnel")
    _metric_bars(
        axes[0, 1],
        graph_diagnostics,
        ["thresholded_edge_density", "positive_similarity_fraction", "isolated_node_count", "weak_module_degree_candidate_count"],
        "Graph health",
    )
    _metric_bars(
        axes[1, 0],
        unclassified_diagnostics,
        ["not_active_enough", "finite_osi_unavailable", "below_osi_threshold", "selected_but_louvain_unclassified"],
        "Population dropouts",
    )
    _metric_bars(
        axes[1, 1],
        unclassified_diagnostics,
        ["weak_module_degree_removed", "small_cluster_removed", "removed_by_random_sampling"],
        "Cleanup dropouts",
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _labels(result: AnalysisResult) -> np.ndarray:
    if result.communities is None:
        return np.zeros(result.selected_indices.size, dtype=np.int64)
    return np.asarray(result.communities.labels, dtype=np.int64)


def _community_order(labels: np.ndarray) -> np.ndarray:
    return np.asarray(sorted(range(labels.size), key=lambda index: (labels[index] == 0, int(labels[index]), index)), dtype=np.int64)


def _bar_rows(ax, rows: list[dict[str, object]], title: str) -> None:
    if not rows:
        _no_data(ax, "No rows")
        return
    names = [str(row.get("stage", "")) for row in rows]
    values = [_maybe_float(row.get("count")) for row in rows]
    valid = [(name, value) for name, value in zip(names, values, strict=True) if value is not None]
    if not valid:
        _no_data(ax, "No numeric rows")
        return
    labels, heights = zip(*valid, strict=True)
    ax.bar(np.arange(len(heights)), heights, color="#4C78A8")
    ax.set_xticks(np.arange(len(labels)), [label.replace("_", "\n") for label in labels], rotation=30, ha="right")
    ax.set_title(title)


def _metric_bars(ax, metrics: dict[str, object], names: list[str], title: str) -> None:
    labels: list[str] = []
    values: list[float] = []
    for name in names:
        value = _maybe_float(metrics.get(name))
        if value is None:
            continue
        labels.append(name.replace("_", "\n"))
        values.append(value)
    if not values:
        _no_data(ax, "No metrics")
        return
    ax.bar(np.arange(len(values)), values, color="#F58518")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=30, ha="right")
    ax.set_title(title)


def _hist(ax, values: ArrayLike, title: str, xlabel: str) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        _no_data(ax, "No data")
        return
    ax.hist(arr, bins=min(20, max(1, arr.size)), color="#54A24B", alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")


def _scatter(ax, coords: np.ndarray, values: ArrayLike, title: str) -> None:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if coords.size == 0 or coords.shape[0] == 0 or arr.size != coords.shape[0]:
        _no_data(ax, "No coordinates")
        return
    image = ax.scatter(coords[:, 0], coords[:, 1], c=arr, cmap="viridis", s=36)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.figure.colorbar(image, ax=ax, shrink=0.8)


def _matrix(ax, values: np.ndarray, title: str) -> None:
    if values.size == 0:
        _no_data(ax, "No matrix")
        return
    image = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Cell")
    ax.set_ylabel("Cell")
    ax.figure.colorbar(image, ax=ax, shrink=0.8)


def _no_data(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def _maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _plot_metric_series(
    ax,
    rows: list[dict[str, object]],
    x_name: str | None,
    y_names: list[str],
    title: str,
) -> None:
    if not rows:
        _no_data(ax, "No robustness rows")
        ax.set_title(title)
        return
    x = np.arange(len(rows), dtype=float) if x_name is None else np.array([_maybe_float(row.get(x_name)) or index for index, row in enumerate(rows)], dtype=float)
    plotted = False
    for y_name in y_names:
        y = np.array([np.nan if _maybe_float(row.get(y_name)) is None else _maybe_float(row.get(y_name)) for row in rows], dtype=float)
        if np.any(np.isfinite(y)):
            ax.plot(x, y, marker="o", label=y_name)
            plotted = True
    if not plotted:
        _no_data(ax, "No numeric metrics")
    else:
        ax.legend(loc="best", fontsize="small")
    ax.set_title(title)
    ax.set_xlabel(x_name or "variant")


__all__ = ["save_analysis_figures", "save_analysis_robustness_figure"]
