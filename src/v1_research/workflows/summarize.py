"""Read-only summaries for new-format run bundles."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from v1_research.runs import json_ready, load_model_state, model_summary, write_json


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """Builds a compact summary from a run bundle."""

    root = Path(run_dir)
    manifest = _read_json(root / "manifest.json")
    summary: dict[str, Any] = {
        "run_dir": str(root),
        "workflow": manifest.get("workflow"),
    }
    if isinstance(manifest.get("summary"), dict):
        for key, value in manifest["summary"].items():
            summary[f"manifest.{key}"] = value

    model_dir = root / "model"
    if (model_dir / "state.npz").exists():
        for key, value in model_summary(load_model_state(model_dir)).items():
            summary[f"model.{key}"] = value

    _add_rate_summary(summary, root / "arrays" / "excitatory_rates.npy", prefix="rates.exc")
    _add_rate_summary(summary, root / "arrays" / "inhibitory_rates.npy", prefix="rates.inh")
    _add_analysis_summary(summary, root / "analysis" / "metrics.json")
    _add_table_summary(summary, root / "tables" / "training_log.csv", prefix="training_log")
    _add_table_summary(summary, root / "tables" / "training_diagnostics.csv", prefix="training_diagnostics")
    return json_ready(summary)


def write_run_summary(run_dir: str | Path, output: str | Path | None = None) -> Path:
    """Writes ``summary.json`` or the requested output path."""

    root = Path(run_dir)
    target = root / "summary.json" if output is None else Path(output)
    return write_json(target, summarize_run(root))


def _add_rate_summary(summary: dict[str, Any], path: Path, *, prefix: str) -> None:
    if not path.exists():
        return
    values = np.asarray(np.load(path), dtype=float)
    summary[f"{prefix}_shape"] = list(values.shape)
    summary[f"{prefix}_mean"] = _finite_float(np.mean(values))
    summary[f"{prefix}_max"] = _finite_float(np.max(values))


def _add_analysis_summary(summary: dict[str, Any], path: Path) -> None:
    if not path.exists():
        return
    metrics = _read_json(path)
    for key, value in metrics.items():
        summary[f"analysis.{key}"] = value


def _add_table_summary(summary: dict[str, Any], path: Path, *, prefix: str) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    summary[f"{prefix}.rows"] = len(rows)
    if rows:
        last = rows[-1]
        for key, value in last.items():
            converted = _number_or_text(value)
            if isinstance(converted, int | float):
                summary[f"{prefix}.last.{key}"] = converted


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _number_or_text(value: str) -> int | float | str:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        try:
            return float(value)
        except (TypeError, ValueError):
            return value
    return integer


def _finite_float(value: Any) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


__all__ = ["summarize_run", "write_run_summary"]
