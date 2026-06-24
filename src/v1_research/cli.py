"""Command-line entrypoint for V1 research workflows."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Annotated, Any, get_args, get_origin, get_type_hints

import numpy as np
import typer
from omegaconf import OmegaConf

from v1_research.workflows.full import FullWorkflowConfig, run_train_then_simulate
from v1_research.workflows.simulate import SimulationWorkflowConfig, run_grating_simulation
from v1_research.workflows.train import TrainingWorkflowConfig, run_training

app = typer.Typer(help="Run V1 research workflows.")


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    overrides: Annotated[list[str] | None, typer.Option("--override", "-o")] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
) -> None:
    """Run natural-image training."""

    cfg = load_workflow_config(config, overrides or [], TrainingWorkflowConfig)
    result = run_training(cfg, show_progress=progress)
    typer.echo(result.run_dir)


@app.command()
def simulate(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    overrides: Annotated[list[str] | None, typer.Option("--override", "-o")] = None,
) -> None:
    """Run drifting-grating simulation."""

    cfg = load_workflow_config(config, overrides or [], SimulationWorkflowConfig)
    result = run_grating_simulation(cfg)
    typer.echo(result.run_dir)


@app.command()
def full(
    config: Annotated[Path, typer.Option("--config", "-c", exists=True, dir_okay=False)],
    overrides: Annotated[list[str] | None, typer.Option("--override", "-o")] = None,
    progress: Annotated[bool, typer.Option("--progress/--no-progress")] = True,
) -> None:
    """Run training followed by grating simulation."""

    cfg = load_workflow_config(config, overrides or [], FullWorkflowConfig)
    result = run_train_then_simulate(cfg, show_progress=progress)
    typer.echo(result.summary["simulate_run"])


def main() -> None:
    """Runs the Typer application."""

    app()


def load_workflow_config(path: Path, overrides: list[str], cls: type[Any]) -> Any:
    """Loads YAML config plus OmegaConf dot-list overrides into a dataclass."""

    loaded = OmegaConf.load(path)
    merged = OmegaConf.merge(loaded, OmegaConf.from_dotlist(overrides))
    payload = OmegaConf.to_container(merged, resolve=True)
    return dataclass_from_mapping(cls, payload if isinstance(payload, dict) else {})


def dataclass_from_mapping(cls: type[Any], payload: dict[str, Any]) -> Any:
    """Constructs a dataclass, recursively handling local config dataclasses."""

    if not is_dataclass(cls):
        return payload
    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for item in fields(cls):
        if item.name not in payload:
            continue
        kwargs[item.name] = _coerce_value(type_hints.get(item.name, item.type), payload[item.name])
    return cls(**kwargs)


def _coerce_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    target = _resolve_dataclass_type(annotation)
    if target is not None and isinstance(value, dict):
        return dataclass_from_mapping(target, value)
    if _is_path_annotation(annotation):
        return Path(value)
    if _is_time_annotation(annotation):
        return np.asarray(value, dtype=float)
    if _is_tuple_annotation(annotation) and isinstance(value, list):
        return tuple(value)
    return value


def _resolve_dataclass_type(annotation: Any) -> type[Any] | None:
    if isinstance(annotation, type) and is_dataclass(annotation):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        if isinstance(arg, type) and is_dataclass(arg):
            return arg
    return None


def _is_path_annotation(annotation: Any) -> bool:
    if annotation is Path:
        return True
    return Path in get_args(annotation)


def _is_time_annotation(annotation: Any) -> bool:
    args = get_args(annotation)
    return any(arg is np.ndarray or get_origin(arg) is np.ndarray for arg in args)


def _is_tuple_annotation(annotation: Any) -> bool:
    return get_origin(annotation) is tuple


__all__ = [
    "app",
    "dataclass_from_mapping",
    "load_workflow_config",
    "main",
    "run_grating_simulation",
    "run_train_then_simulate",
    "run_training",
]
