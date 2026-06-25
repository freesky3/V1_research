"""Persistence for compact analysis outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from v1_research.analysis.metrics import write_analysis_metrics
from v1_research.analysis.pipeline import AnalysisResult
from v1_research.runs import write_csv_rows, write_json


def write_analysis_result(result: AnalysisResult, output_dir: str | Path, *, tables_dir: str | Path | None = None) -> dict[str, Path]:
    """Writes compact arrays, diagnostics, and metric tables for an analysis result."""

    analysis_dir = Path(output_dir)
    table_dir = analysis_dir if tables_dir is None else Path(tables_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "selected_indices": analysis_dir / "selected_indices.npy",
        "osi": analysis_dir / "osi.npy",
        "preferred_orientation": analysis_dir / "preferred_orientation.npy",
        "responses_mean": analysis_dir / "responses_mean.npy",
        "steady_state_responses": analysis_dir / "steady_state_responses.npy",
        "coords": analysis_dir / "coords.npy",
        "distance": analysis_dir / "distance.npy",
        "diagnostics": analysis_dir / "diagnostics.json",
    }
    np.save(paths["selected_indices"], result.selected_indices)
    np.save(paths["osi"], result.osi)
    np.save(paths["preferred_orientation"], result.preferred_orientation)
    np.save(paths["responses_mean"], result.responses_mean)
    np.save(paths["steady_state_responses"], result.steady_state_responses)
    np.save(paths["coords"], result.coords)
    np.save(paths["distance"], result.distance)
    write_json(paths["diagnostics"], result.diagnostics)

    if result.communities is not None:
        paths["community_labels"] = analysis_dir / "community_labels.npy"
        paths["similarity"] = analysis_dir / "similarity.npy"
        np.save(paths["community_labels"], result.communities.labels)
        np.save(paths["similarity"], result.communities.similarity)
        if result.communities.agreement is not None:
            paths["agreement"] = analysis_dir / "agreement.npy"
            np.save(paths["agreement"], result.communities.agreement)
        paths["community_diagnostics"] = analysis_dir / "community_diagnostics.json"
        write_json(paths["community_diagnostics"], result.communities.diagnostics)

    summary = result.diagnostics.get("metrics_summary", {})
    rows = result.diagnostics.get("ensemble_metrics", [])
    if isinstance(summary, dict) and isinstance(rows, list):
        metrics_path, ensemble_path = write_analysis_metrics(summary, rows, table_dir)
        paths["metrics"] = metrics_path
        paths["ensemble_metrics"] = ensemble_path
        if table_dir != analysis_dir:
            paths["analysis_metrics"] = write_json(analysis_dir / "metrics.json", summary)
    return paths


def write_analysis_inspection(
    *,
    output_dir: str | Path,
    tables_dir: str | Path,
    selection_rows: list[dict[str, object]],
    selection_summary: dict[str, object],
    graph_diagnostics: dict[str, object],
    unclassified_diagnostics: dict[str, object],
) -> dict[str, Path]:
    """Writes JSON and CSV diagnostics for analysis inspection."""

    analysis_dir = Path(output_dir)
    table_dir = Path(tables_dir)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    return {
        "selection_funnel": write_json(analysis_dir / "selection_funnel.json", selection_summary),
        "graph_diagnostics": write_json(analysis_dir / "graph_diagnostics.json", graph_diagnostics),
        "unclassified_diagnostics": write_json(analysis_dir / "unclassified_diagnostics.json", unclassified_diagnostics),
        "selection_funnel_table": write_csv_rows(table_dir / "selection_funnel.csv", selection_rows),
    }
