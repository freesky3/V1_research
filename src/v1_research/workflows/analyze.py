"""Analysis workflow orchestration for simulation run bundles."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from v1_research.analysis.artifacts import write_analysis_result
from v1_research.analysis.pipeline import (
    AnalysisConfig,
    AnalysisResult,
    load_analysis_inputs_from_simulation,
    run_analysis,
)
from v1_research.runs import relative_output_path, write_config, write_json, write_manifest


@dataclass(frozen=True, slots=True)
class AnalysisWorkflowConfig:
    """Top-level configuration for analyzing a grating simulation run."""

    simulation_run: str | Path
    output_run_root: str | Path | None = None
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    save_inputs: bool = True


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


__all__ = ["AnalysisRun", "AnalysisWorkflowConfig", "run_analysis_workflow"]
