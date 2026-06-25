"""Analysis workflow orchestration for simulation run bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from v1_research.analysis.artifacts import write_analysis_inspection, write_analysis_result
from v1_research.analysis.diagnostics import graph_health_diagnostics, selection_funnel, unclassified_diagnostics
from v1_research.analysis.osi import compute_osi
from v1_research.analysis.pipeline import (
    AnalysisConfig,
    AnalysisInputs,
    AnalysisResult,
    load_analysis_inputs_from_simulation,
    run_analysis,
)
from v1_research.analysis.robustness import run_louvain_parameter_grid, summarize_robustness
from v1_research.analysis.temporal import run_window_analysis
from v1_research.runs import relative_output_path, write_config, write_csv_rows, write_json, write_manifest
from v1_research.workflows.analysis_figures import save_analysis_figures, save_analysis_robustness_figure


@dataclass(frozen=True, slots=True)
class AnalysisRobustnessConfig:
    """Optional repeated-analysis diagnostics for stability checks."""

    enabled: bool = False
    tail_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    end_times: tuple[float, ...] = ()
    louvain_parameter_grid: dict[str, tuple[object, ...]] = field(default_factory=dict)
    overlap_surrogates: int = 0


@dataclass(frozen=True, slots=True)
class AnalysisInspectionConfig:
    """Optional diagnostics and figures for analysis runs."""

    enabled: bool = True
    save_plots: bool = True
    save_tables: bool = True
    robustness: AnalysisRobustnessConfig = field(default_factory=AnalysisRobustnessConfig)


@dataclass(frozen=True, slots=True)
class AnalysisWorkflowConfig:
    """Top-level configuration for analyzing a grating simulation run."""

    simulation_run: str | Path
    output_run_root: str | Path | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    save_inputs: bool = True
    inspection: AnalysisInspectionConfig = field(default_factory=AnalysisInspectionConfig)


@dataclass(frozen=True, slots=True)
class AnalysisRun:
    """In-memory summary of a saved analysis workflow."""

    run_dir: Path
    result: AnalysisResult
    output_paths: dict[str, Path]
    summary: dict[str, int | str | float | None]


def run_analysis_workflow(cfg: AnalysisWorkflowConfig) -> AnalysisRun:
    """Loads a simulation bundle, runs analysis, and writes compact outputs."""

    source_run = Path(cfg.simulation_run)
    output_dir = _analysis_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = output_dir.parent / "tables" if output_dir.name == "analysis" else output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    write_config(output_dir, cfg)
    inputs = load_analysis_inputs_from_simulation(source_run, center_side_fraction=cfg.analysis.center_side_fraction)
    result = run_analysis(cfg.analysis, inputs)
    paths = write_analysis_result(result, output_dir, tables_dir=tables_dir)
    inspection = None
    if cfg.inspection.enabled and (cfg.inspection.save_tables or cfg.inspection.save_plots):
        inspection = _compute_analysis_inspection(cfg.analysis, inputs, result)
    if inspection is not None and cfg.inspection.save_tables:
        paths.update(
            write_analysis_inspection(
                output_dir=output_dir,
                tables_dir=tables_dir,
                selection_rows=inspection["selection_rows"],
                selection_summary=inspection["selection_summary"],
                graph_diagnostics=inspection["graph_diagnostics"],
                unclassified_diagnostics=inspection["unclassified_diagnostics"],
            )
        )
    if inspection is not None and cfg.inspection.save_plots:
        paths.update(
            save_analysis_figures(
                _figure_output_root(cfg, source_run, output_dir),
                result,
                selection_rows=inspection["selection_rows"],
                graph_diagnostics=inspection["graph_diagnostics"],
                unclassified_diagnostics=inspection["unclassified_diagnostics"],
                orientation_angles=inputs.orientation_angles,
            )
        )
    if cfg.inspection.enabled and cfg.inspection.robustness.enabled:
        robustness = _run_analysis_robustness(cfg, source_run, inputs)
        if cfg.inspection.save_tables:
            if robustness["window_rows"]:
                paths["robustness_windows"] = write_csv_rows(tables_dir / "robustness_windows.csv", robustness["window_rows"])
            if robustness["louvain_rows"]:
                paths["robustness_louvain"] = write_csv_rows(tables_dir / "robustness_louvain.csv", robustness["louvain_rows"])
            paths["robustness_summary"] = write_json(output_dir / "robustness_summary.json", robustness["summary"])
        if cfg.inspection.save_plots:
            paths["analysis_robustness_figure"] = save_analysis_robustness_figure(
                _figure_output_root(cfg, source_run, output_dir),
                window_rows=robustness["window_rows"],
                louvain_rows=robustness["louvain_rows"],
            )
    if cfg.save_inputs:
        paths["inputs"] = write_json(
            output_dir / "inputs.json",
            {
                "source_run": str(source_run),
                "responses_shape": list(inputs.responses.shape),
                "coords_shape": list(inputs.coords.shape),
                "distance_shape": list(inputs.distance.shape),
                "orientation_count": int(inputs.orientation_angles.size),
            },
        )
    summary = {
        "status": result.status,
        "selected_neurons": int(result.selected_indices.size),
        "n_ensembles": (
            int(result.communities.n_ensembles) if result.communities is not None else 0
        ),
        "classified_neurons": (
            int(result.communities.classified_neurons) if result.communities is not None else 0
        ),
    }
    metrics_summary = result.diagnostics.get("metrics_summary", {})
    if isinstance(metrics_summary, dict):
        for key, value in metrics_summary.items():
            if isinstance(value, (int, float, str)) or value is None:
                summary[str(key)] = value
    if cfg.output_run_root is None:
        _update_source_manifest(source_run, summary, paths)
    else:
        write_manifest(
            output_dir,
            {
                "workflow": "analyze",
                "source_run": str(source_run),
                "analysis": summary,
                "outputs": {
                    key: relative_output_path(path, output_dir)
                    for key, path in paths.items()
                    if path.is_relative_to(output_dir)
                },
            },
        )
    run_dir = source_run if cfg.output_run_root is None else output_dir
    return AnalysisRun(run_dir=run_dir, result=result, output_paths=paths, summary=summary)


def _analysis_output_dir(cfg: AnalysisWorkflowConfig) -> Path:
    if cfg.output_run_root is None:
        return Path(cfg.simulation_run) / "analysis"
    return Path(cfg.output_run_root)


def _figure_output_root(cfg: AnalysisWorkflowConfig, source_run: Path, output_dir: Path) -> Path:
    if cfg.output_run_root is None:
        return source_run
    return output_dir


def _update_source_manifest(source_run: Path, summary: dict[str, int | str | float | None], paths: dict[str, Path]) -> None:
    manifest_path = source_run / "manifest.json"
    payload = {}
    if manifest_path.exists():
        import json

        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["analysis"] = summary
    payload["analysis_outputs"] = {
        key: relative_output_path(path, source_run)
        for key, path in paths.items()
        if _is_relative_to(path, source_run)
    }
    write_manifest(source_run, payload)


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _compute_analysis_inspection(
    cfg: AnalysisConfig,
    inputs: AnalysisInputs,
    result: AnalysisResult,
) -> dict[str, object]:
    steady_start = int(result.diagnostics.get("steady_state_start", int(inputs.responses.shape[2] * 2 / 3)))
    responses_mean = np.mean(np.asarray(inputs.responses, dtype=float)[:, :, steady_start:], axis=2)
    osi, _ = compute_osi(responses_mean, inputs.orientation_angles, min_osi=cfg.osi_threshold)
    activity = np.mean(responses_mean, axis=1)
    labels = result.communities.labels if result.communities is not None else None
    community_diagnostics = result.communities.diagnostics if result.communities is not None else {}
    selection_rows, selection_summary = selection_funnel(
        osi=osi,
        activity=activity,
        selected_indices=result.selected_indices,
        labels=labels,
        cfg=cfg,
    )
    graph = graph_health_diagnostics(
        result.communities.similarity if result.communities is not None else None,
        cfg.louvain,
    )
    unclassified = unclassified_diagnostics(
        osi=osi,
        activity=activity,
        selected_indices=result.selected_indices,
        labels=labels,
        cfg=cfg,
        community_diagnostics=community_diagnostics,
    )
    return {
        "selection_rows": selection_rows,
        "selection_summary": selection_summary,
        "graph_diagnostics": graph,
        "unclassified_diagnostics": unclassified,
    }


def _run_analysis_robustness(
    cfg: AnalysisWorkflowConfig,
    source_run: Path,
    inputs: AnalysisInputs,
) -> dict[str, object]:
    robustness = cfg.inspection.robustness
    time = _load_time(source_run) if robustness.end_times else None
    window_rows = run_window_analysis(
        cfg.analysis,
        inputs,
        tail_fractions=robustness.tail_fractions,
        end_times=robustness.end_times,
        time=time,
    )
    louvain_rows = run_louvain_parameter_grid(
        cfg.analysis,
        inputs,
        parameter_grid=robustness.louvain_parameter_grid,
    )
    return {
        "window_rows": window_rows,
        "louvain_rows": louvain_rows,
        "summary": summarize_robustness(window_rows=window_rows, louvain_rows=louvain_rows),
    }


def _load_time(source_run: Path) -> np.ndarray:
    path = source_run / "arrays" / "time.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing time array for robustness end_times: {path}")
    return np.asarray(np.load(path), dtype=float)


__all__ = [
    "AnalysisInspectionConfig",
    "AnalysisRobustnessConfig",
    "AnalysisRun",
    "AnalysisWorkflowConfig",
    "run_analysis_workflow",
]
