"""Lightweight grid sweeps over existing V1 workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Literal

from omegaconf import OmegaConf

from v1_research.runs import (
    create_run_dir,
    json_ready,
    relative_output_path,
    write_config,
    write_csv_rows,
    write_json,
    write_manifest,
)
from v1_research.workflows.analyze import AnalysisWorkflowConfig, run_analysis_workflow
from v1_research.workflows.full import FullWorkflowConfig, run_train_then_simulate
from v1_research.workflows.simulate import SimulationWorkflowConfig, run_grating_simulation
from v1_research.workflows.train import TrainingWorkflowConfig, run_training

WorkflowName = Literal["train", "simulate", "analyze", "full"]


@dataclass(frozen=True, slots=True)
class SweepConfig:
    """Configuration for an explicit grid sweep over one workflow."""

    workflow: WorkflowName
    base: dict[str, Any] = field(default_factory=dict)
    parameters: dict[str, list[Any]] = field(default_factory=dict)
    run_root: str | Path = Path("runs")

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _flatten_parameter_map(self.parameters))


@dataclass(frozen=True, slots=True)
class SweepRun:
    """In-memory summary of a saved sweep."""

    run_dir: Path
    csv_path: Path
    summary_path: Path
    summary: dict[str, int | str]
    rows: list[dict[str, Any]]


def run_sweep(cfg: SweepConfig, *, show_progress: bool = True) -> SweepRun:
    """Expands a grid, runs each workflow config, and writes sweep tables."""

    run_dir = create_run_dir(cfg.run_root, "sweep")
    write_config(run_dir, cfg)
    rows: list[dict[str, Any]] = []
    for index, point in enumerate(expand_grid(cfg.parameters), start=1):
        rows.append(_run_one(index, cfg, point, show_progress=show_progress))

    rows = _normalize_rows(rows)
    csv_path = write_csv_rows(run_dir / "tables" / "runs.csv", rows)
    summary = {
        "workflow": cfg.workflow,
        "runs": len(rows),
        "failed": sum(1 for row in rows if row["status"] == "error"),
    }
    summary_path = write_json(run_dir / "summary.json", summary)
    write_manifest(
        run_dir,
        {
            "workflow": "sweep",
            "target_workflow": cfg.workflow,
            "outputs": {
                "runs": relative_output_path(csv_path, run_dir),
                "summary": relative_output_path(summary_path, run_dir),
            },
            "summary": summary,
        },
    )
    return SweepRun(run_dir=run_dir, csv_path=csv_path, summary_path=summary_path, summary=summary, rows=rows)


def expand_grid(parameters: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Expands dot-path parameter lists into grid points."""

    if not parameters:
        return [{}]
    names = list(parameters)
    values = [list(parameters[name]) for name in names]
    return [dict(zip(names, items, strict=True)) for items in product(*values)]


def _run_one(index: int, cfg: SweepConfig, point: dict[str, Any], *, show_progress: bool) -> dict[str, Any]:
    row: dict[str, Any] = {"index": index, **point, "workflow": cfg.workflow}
    try:
        workflow_cfg = _workflow_config(cfg.workflow, _merge_point(cfg.base, point))
        result = _dispatch(cfg.workflow, workflow_cfg, show_progress=show_progress)
    except Exception as exc:  # noqa: BLE001 - sweep records failures and continues by design.
        row.update({"status": "error", "run_dir": "", "error": str(exc)})
        return row

    row.update({"status": "ok", "run_dir": str(result.run_dir), "error": ""})
    for key, value in _summary_items(result).items():
        row[f"summary.{key}"] = value
    return row


def _merge_point(base: Mapping[str, Any], point: Mapping[str, Any]) -> dict[str, Any]:
    merged = OmegaConf.create(json_ready(base))
    for key, value in point.items():
        OmegaConf.update(merged, key, json_ready(value), merge=False)
    payload = OmegaConf.to_container(merged, resolve=True)
    return payload if isinstance(payload, dict) else {}


def _flatten_parameter_map(parameters: Mapping[str, Any], prefix: str = "") -> dict[str, list[Any]]:
    flattened: dict[str, list[Any]] = {}
    for key, value in parameters.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_parameter_map(value, name))
        else:
            flattened[name] = list(value)
    return flattened


def _workflow_config(workflow: WorkflowName, payload: dict[str, Any]) -> Any:
    from v1_research.cli import dataclass_from_mapping

    cls = {
        "train": TrainingWorkflowConfig,
        "simulate": SimulationWorkflowConfig,
        "analyze": AnalysisWorkflowConfig,
        "full": FullWorkflowConfig,
    }[workflow]
    return dataclass_from_mapping(cls, payload)


def _dispatch(workflow: WorkflowName, cfg: Any, *, show_progress: bool) -> Any:
    if workflow == "train":
        return run_training(cfg, show_progress=show_progress)
    if workflow == "simulate":
        return run_grating_simulation(cfg)
    if workflow == "analyze":
        return run_analysis_workflow(cfg)
    if workflow == "full":
        return run_train_then_simulate(cfg, show_progress=show_progress)
    raise ValueError(f"Unknown workflow: {workflow!r}")


def _summary_items(result: Any) -> dict[str, Any]:
    summary = getattr(result, "summary", {})
    if not isinstance(summary, Mapping):
        return {}
    return {str(key): value for key, value in json_ready(summary).items()}


def _normalize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return [{key: row.get(key, "") for key in columns} for row in rows]


__all__ = ["SweepConfig", "SweepRun", "expand_grid", "run_sweep"]
