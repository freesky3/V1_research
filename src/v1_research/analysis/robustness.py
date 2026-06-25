"""Optional robustness helpers for analysis runs."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any, Mapping, Sequence

from v1_research.analysis.pipeline import AnalysisConfig, AnalysisInputs, run_analysis
from v1_research.runs import json_ready


def run_louvain_parameter_grid(
    cfg: AnalysisConfig,
    inputs: AnalysisInputs,
    *,
    parameter_grid: Mapping[str, Sequence[object]],
) -> list[dict[str, Any]]:
    """Runs analysis over a small explicit grid of Louvain parameter values."""

    flattened = _flatten_parameter_grid(parameter_grid)
    if not flattened:
        return []
    names = list(flattened)
    values = [tuple(flattened[name]) for name in names]
    rows: list[dict[str, Any]] = []
    for combination in product(*values):
        variant_cfg = cfg
        row: dict[str, Any] = {}
        for name, value in zip(names, combination, strict=True):
            variant_cfg = _with_parameter(variant_cfg, name, value)
            row[name] = value
        try:
            result = run_analysis(variant_cfg, inputs)
        except Exception as exc:  # pragma: no cover - exercised by workflow-level resilience
            row.update({"status": "error", "error": str(exc)})
        else:
            row.update(_result_row(result))
        rows.append(row)
    return json_ready(rows)


def _flatten_parameter_grid(parameter_grid: Mapping[str, Any]) -> dict[str, Sequence[object]]:
    flattened: dict[str, Sequence[object]] = {}
    for name, value in parameter_grid.items():
        if isinstance(value, Mapping):
            for child_name, child_value in value.items():
                flattened[f"{name}.{child_name}"] = _as_sequence(child_value)
        else:
            flattened[str(name)] = _as_sequence(value)
    return flattened


def summarize_robustness(
    *,
    window_rows: list[dict[str, object]],
    louvain_rows: list[dict[str, object]],
) -> dict[str, Any]:
    """Builds compact counters for robustness outputs."""

    all_rows = [*window_rows, *louvain_rows]
    ok_count = sum(1 for row in all_rows if row.get("status") == "ok")
    error_count = sum(1 for row in all_rows if row.get("status") == "error")
    return json_ready(
        {
            "window_row_count": len(window_rows),
            "louvain_row_count": len(louvain_rows),
            "variant_count": len(all_rows),
            "ok_variant_count": ok_count,
            "error_variant_count": error_count,
        }
    )


def _with_parameter(cfg: AnalysisConfig, name: str, value: object) -> AnalysisConfig:
    if not name.startswith("louvain."):
        raise ValueError(f"Only louvain.* robustness parameters are supported, got {name!r}.")
    field_name = name.split(".", 1)[1]
    if not hasattr(cfg.louvain, field_name):
        raise ValueError(f"Unknown Louvain parameter: {field_name}")
    louvain = replace(cfg.louvain, **{field_name: value})
    return replace(cfg, louvain=louvain)


def _as_sequence(value: Any) -> Sequence[object]:
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return value
    return (value,)


def _result_row(result) -> dict[str, Any]:
    row: dict[str, Any] = {
        "status": result.status,
        "selected_neurons": int(result.selected_indices.size),
    }
    summary = result.diagnostics.get("metrics_summary", {})
    if isinstance(summary, dict):
        for key, value in summary.items():
            row[str(key)] = value
    return row


__all__ = ["run_louvain_parameter_grid", "summarize_robustness"]
