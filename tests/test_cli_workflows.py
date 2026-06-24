from __future__ import annotations

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
        captured["show_progress"] = show_progress

        class Result:
            run_dir = tmp_path / "runs" / "train" / "demo"

        return Result()

    monkeypatch.setattr(cli, "run_training", fake_run_training)

    result = CliRunner().invoke(
        cli.app,
        ["train", "--config", str(config_path), "-o", "batch_size=3", "--no-progress"],
    )

    assert result.exit_code == 0, result.output
    assert captured == {"batch_size": 3, "l4_n_side": 1, "show_progress": False}
    assert str(tmp_path / "runs" / "train" / "demo") in result.output


def test_cli_config_loader_keeps_optional_null_paths_as_none(tmp_path) -> None:
    config_path = tmp_path / "simulate.yaml"
    config_path.write_text("model_checkpoint: null\n", encoding="utf-8")

    cfg = cli.load_workflow_config(config_path, [], SimulationWorkflowConfig)

    assert cfg.model_checkpoint is None


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
