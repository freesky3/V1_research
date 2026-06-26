from __future__ import annotations

import numpy as np
from typer.testing import CliRunner

from v1_research import cli
from v1_research.workflows.simulate import SimulationWorkflowConfig


def test_cli_loads_config_applies_override_and_dispatches_train(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        "\n".join(
            [
                "run_root: runs",
                "empirical_data_path: data/sample_data.pkl",
                "batch_size: 2",
                "epochs: 1",
                "model:",
                "  l4:",
                "    n_side: 1",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_training(cfg, *, show_progress: bool = True):
        captured["batch_size"] = cfg.batch_size
        captured["l4_n_side"] = cfg.model.l4.n_side
        captured["min_active"] = cfg.inspection.health.min_active_neuron_fraction
        captured["show_progress"] = show_progress

        class Result:
            run_dir = tmp_path / "runs" / "train" / "demo"

        return Result()

    monkeypatch.setattr(cli, "run_training", fake_run_training)

    result = CliRunner().invoke(
        cli.app,
        [
            "train",
            "--config",
            str(config_path),
            "-o",
            "batch_size=3",
            "-o",
            "inspection.health.min_active_neuron_fraction=0.2",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"batch_size": 3, "l4_n_side": 1, "min_active": 0.2, "show_progress": False}
    assert str(tmp_path / "runs" / "train" / "demo") in result.output


def test_cli_config_loader_keeps_optional_null_paths_as_none(tmp_path) -> None:
    config_path = tmp_path / "simulate.yaml"
    config_path.write_text("model_checkpoint: null\n", encoding="utf-8")

    cfg = cli.load_workflow_config(config_path, [], SimulationWorkflowConfig)

    assert cfg.model_checkpoint is None


def test_cli_config_loader_accepts_compact_time_grid_mapping(tmp_path) -> None:
    config_path = tmp_path / "simulate.yaml"
    config_path.write_text("time:\n  start: 0.0\n  stop: 0.8\n  step: 0.004\n", encoding="utf-8")

    cfg = cli.load_workflow_config(config_path, [], SimulationWorkflowConfig)

    assert cfg.time.shape == (201,)
    np.testing.assert_allclose(cfg.time[[0, -1]], [0.0, 0.8])
    assert np.all(np.diff(cfg.time) > 0.0)


def test_cli_train_passes_progress_for_live_status_and_prints_run_dir(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "train.yaml"
    config_path.write_text("run_root: runs\ninspection:\n  enabled: true\n", encoding="utf-8")
    run_dir = tmp_path / "runs" / "train" / "demo"
    captured = {}

    def fake_run_training(cfg, *, show_progress: bool = True):
        captured["show_progress"] = show_progress

        class Result:
            pass

        result = Result()
        result.run_dir = run_dir
        return result

    monkeypatch.setattr(cli, "run_training", fake_run_training)

    result = CliRunner().invoke(cli.app, ["train", "--config", str(config_path)])

    assert result.exit_code == 0, result.output
    assert captured == {"show_progress": True}
    assert result.output.strip() == str(run_dir)


def test_cli_sweep_loads_config_applies_override_and_dispatches(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "sweep.yaml"
    config_path.write_text(
        "\n".join(
            [
                "workflow: simulate",
                "run_root: runs",
                "base:",
                "  grating:",
                "    n_orientations: 4",
                "parameters:",
                "  grating.visual_gain: [100.0]",
            ]
        ),
        encoding="utf-8",
    )
    captured = {}

    def fake_run_sweep(cfg, *, show_progress: bool = True):
        captured["workflow"] = cfg.workflow
        captured["visual_gain"] = cfg.parameters["grating.visual_gain"]
        captured["show_progress"] = show_progress

        class Result:
            run_dir = tmp_path / "runs" / "sweep" / "demo"

        return Result()

    monkeypatch.setattr(cli, "run_sweep", fake_run_sweep)

    result = CliRunner().invoke(
        cli.app,
        [
            "sweep",
            "--config",
            str(config_path),
            "-o",
            "parameters.grating.visual_gain=[200.0,300.0]",
            "--no-progress",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured == {
        "workflow": "simulate",
        "visual_gain": [200.0, 300.0],
        "show_progress": False,
    }
    assert str(tmp_path / "runs" / "sweep" / "demo") in result.output
