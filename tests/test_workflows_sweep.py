from __future__ import annotations

import csv
import json
from pathlib import Path

from v1_research.workflows.sweep import SweepConfig, expand_grid, run_sweep


class FakeRun:
    def __init__(self, run_dir: Path, summary: dict[str, object]) -> None:
        self.run_dir = run_dir
        self.summary = summary


def test_expand_grid_preserves_parameter_order() -> None:
    rows = expand_grid(
        {
            "grating.visual_gain": [100.0, 200.0],
            "solver.store_trajectory": [False, True],
        }
    )

    assert rows == [
        {"grating.visual_gain": 100.0, "solver.store_trajectory": False},
        {"grating.visual_gain": 100.0, "solver.store_trajectory": True},
        {"grating.visual_gain": 200.0, "solver.store_trajectory": False},
        {"grating.visual_gain": 200.0, "solver.store_trajectory": True},
    ]


def test_run_sweep_writes_csv_and_manifest(tmp_path, monkeypatch) -> None:
    captured = []

    def fake_simulation(cfg):
        captured.append((cfg.grating.visual_gain, cfg.solver.store_trajectory))
        run_dir = tmp_path / "runs" / "simulate" / f"run_{len(captured)}"
        run_dir.mkdir(parents=True)
        return FakeRun(run_dir, {"n_orientations": cfg.grating.n_orientations})

    monkeypatch.setattr("v1_research.workflows.sweep.run_grating_simulation", fake_simulation)

    result = run_sweep(
        SweepConfig(
            workflow="simulate",
            run_root=tmp_path / "runs",
            base={
                "solver": {"backend": "scipy"},
                "grating": {"n_orientations": 4},
            },
            parameters={
                "grating.visual_gain": [100.0, 200.0],
                "solver.store_trajectory": [False],
            },
        )
    )

    assert captured == [(100.0, False), (200.0, False)]
    assert result.run_dir.parent == tmp_path / "runs" / "sweep"
    assert result.summary["runs"] == 2
    assert result.summary["failed"] == 0

    with result.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["grating.visual_gain"] == "100.0"
    assert rows[0]["solver.store_trajectory"] == "False"
    assert rows[0]["status"] == "ok"
    assert rows[0]["workflow"] == "simulate"
    assert rows[0]["summary.n_orientations"] == "4"

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "sweep"
    assert manifest["target_workflow"] == "simulate"
    assert manifest["summary"]["runs"] == 2


def test_simulate_sweep_defaults_to_lightweight_inspection(tmp_path, monkeypatch) -> None:
    captured = []

    def fake_simulation(cfg):
        captured.append((cfg.solver.store_trajectory, cfg.inspection.enabled, cfg.inspection.save_plots))
        run_dir = tmp_path / "runs" / "simulate" / "run_1"
        run_dir.mkdir(parents=True)
        return FakeRun(run_dir, {"n_orientations": cfg.grating.n_orientations})

    monkeypatch.setattr("v1_research.workflows.sweep.run_grating_simulation", fake_simulation)

    run_sweep(
        SweepConfig(
            workflow="simulate",
            run_root=tmp_path / "runs",
            base={
                "solver": {"backend": "scipy"},
                "grating": {"n_orientations": 4},
            },
            parameters={"grating.visual_gain": [100.0]},
        )
    )

    assert captured == [(False, False, False)]


def test_run_sweep_records_failure_and_continues(tmp_path, monkeypatch) -> None:
    calls = []

    def fake_simulation(cfg):
        calls.append(cfg.grating.visual_gain)
        if cfg.grating.visual_gain == 200.0:
            raise RuntimeError("unstable")
        run_dir = tmp_path / "runs" / "simulate" / "ok"
        run_dir.mkdir(parents=True, exist_ok=True)
        return FakeRun(run_dir, {"n_orientations": cfg.grating.n_orientations})

    monkeypatch.setattr("v1_research.workflows.sweep.run_grating_simulation", fake_simulation)

    result = run_sweep(
        SweepConfig(
            workflow="simulate",
            run_root=tmp_path / "runs",
            base={"grating": {"n_orientations": 4}},
            parameters={"grating.visual_gain": [100.0, 200.0]},
        )
    )

    assert calls == [100.0, 200.0]
    assert result.summary["failed"] == 1
    with result.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[1]["status"] == "error"
    assert rows[1]["error"] == "unstable"


def test_run_sweep_keeps_summary_columns_when_first_run_fails(tmp_path, monkeypatch) -> None:
    def fake_simulation(cfg):
        if cfg.grating.visual_gain == 100.0:
            raise RuntimeError("first failed")
        run_dir = tmp_path / "runs" / "simulate" / "ok"
        run_dir.mkdir(parents=True, exist_ok=True)
        return FakeRun(run_dir, {"n_orientations": cfg.grating.n_orientations})

    monkeypatch.setattr("v1_research.workflows.sweep.run_grating_simulation", fake_simulation)

    result = run_sweep(
        SweepConfig(
            workflow="simulate",
            run_root=tmp_path / "runs",
            base={"grating": {"n_orientations": 4}},
            parameters={"grating.visual_gain": [100.0, 200.0]},
        )
    )

    with result.csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["status"] == "error"
    assert rows[1]["summary.n_orientations"] == "4"
