from __future__ import annotations

import csv
import json
from dataclasses import dataclass

import numpy as np

from v1_research.dynamics import SolverConfig
from v1_research.inputs.background import BackgroundConfig
from v1_research.inputs.gabor import GaborConfig, ReceptiveFieldConfig
from v1_research.inputs.natural_images import NaturalImageDriveConfig, NaturalImagePreprocessConfig
from v1_research.learning import LearningConfig
from v1_research.learning.bcm import BCMConfig
from v1_research.model import L23Config, L4Config
from v1_research.model.build import ModelConfig
from v1_research.model.weights import WeightConfig
from v1_research.workflows.train import (
    NaturalImageWorkflowConfig,
    TrainingHealthConfig,
    TrainingInspectionConfig,
    TrainingWorkflowConfig,
    _format_health_event_summary,
    _format_live_training_status,
    run_training,
)


@dataclass(frozen=True, slots=True)
class FakeSolverConfig:
    backend: str = "fake"
    store_trajectory: bool = False


def _write_tiny_iml(path, value: int = 10) -> None:
    pixels = np.full((4, 4), value, dtype=">u2")
    pixels.tofile(path)


def _training_cfg(tmp_path, *, probe_every: int = 1, inhibitory_fraction: float = 0.0) -> TrainingWorkflowConfig:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_tiny_iml(image_dir / "sample_001.iml")

    return TrainingWorkflowConfig(
        run_root=tmp_path / "runs",
        empirical_data_path="data/sample_data.pkl",
        model=ModelConfig(
            l4=L4Config(n_side=1, region_size=1.0, all_tuned=True, n_orientations=1),
            l23=L23Config(n_side=1, inhibitory_fraction=inhibitory_fraction, region_size=1.0, random_inhibitory=False),
            p_ee=0.0,
            weight=WeightConfig(base_strength=0.1),
        ),
        natural_images=NaturalImageWorkflowConfig(
            image_dir=image_dir,
            image_shape=(4, 4),
            crop_size=None,
            patches_per_image=1,
            limit=1,
            receptive_field=ReceptiveFieldConfig(
                stimulus_size=0.5,
                resolution=3,
                gabor=GaborConfig(sigma=0.2, gamma=1.0, spatial_frequency=1.0, phase=0.0),
            ),
            preprocess=NaturalImagePreprocessConfig(resolution=4, normalization="maxscale"),
            drive=NaturalImageDriveConfig(visual_gain=0.01, baseline_rate=0.1),
            cache_dir=tmp_path / "projection_cache",
        ),
        solver=SolverConfig(backend="scipy", scipy_method="RK4", store_trajectory=False),
        learning=LearningConfig(
            kind="bcm",
            bcm=BCMConfig(theta_init=1.0, eta=1e-5, row_sum_max_scale=None),
        ),
        background=BackgroundConfig(enabled=False),
        time=np.array([0.0, 0.01, 0.02], dtype=float),
        batch_size=1,
        epochs=2,
        inspection=TrainingInspectionConfig(
            enabled=True,
            health=TrainingHealthConfig(min_active_neuron_fraction=0.2, max_top1_activity_fraction=0.2),
            tracked_weight_count=1,
            probe_every=probe_every,
            save_plots=False,
            save_per_batch_arrays=True,
        ),
    )


def test_live_training_status_formats_detail_and_missing_values() -> None:
    row = {
        "step": 3,
        "epoch": 1,
        "batch": 3,
        "exc_active_neuron_fraction": 0.25,
        "inh_active_neuron_fraction": None,
        "exc_top1_activity_fraction": 0.2,
        "exc_top5_activity_fraction": 0.7,
        "exc_near_rate_cap_fraction": 0.0,
        "row_sum_EE_cap_max_ratio": 0.5,
        "theta_exc_median": 1.0,
        "bcm_exc_above_theta_fraction": 0.4,
        "bcm_exc_signal_mean": -0.125,
        "W_EE_delta_positive_fraction": 0.6,
        "W_EE_delta_negative_fraction": 0.1,
    }
    report = {"status": "warn", "warning_count": 2, "failure_count": 1}

    line = _format_live_training_status(row, report, warn_total=5, fail_total=2)

    assert line.startswith("[train] step=3 epoch=1 batch=3 health=warn warn_total=5 fail_total=2")
    assert "exc_active=0.250" in line
    assert "inh_active=-" in line
    assert "top1=0.200" in line
    assert "top5=0.700" in line
    assert "row_EE_cap=0.500" in line
    assert "theta_exc=1.000" in line
    assert "bcm_exc_signal=-0.125" in line
    assert "W_EE_delta+=0.600" in line


def test_health_event_summary_truncates_events() -> None:
    report = {
        "events": [
            {"severity": "warn", "metric": "a", "value": 0.1, "threshold": 0.05},
            {"severity": "warn", "metric": "b", "value": 0.2, "threshold": 0.1},
            {"severity": "fail", "metric": "c", "value": 0.0, "threshold": 0.0},
            {"severity": "warn", "metric": "d", "value": 1.0, "threshold": 0.9},
        ]
    }

    line = _format_health_event_summary(report, limit=3)

    assert line == "[train] events: warn a=0.100>0.050; warn b=0.200>0.100; fail c=0.000>0.000; +1 more"


def test_training_workflow_writes_log_checkpoint_and_manifest(tmp_path) -> None:
    cfg = _training_cfg(tmp_path)

    result = run_training(cfg, show_progress=False)

    assert result.run_dir.parent == tmp_path / "runs" / "train"
    assert result.summary["batches"] == 2
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "model" / "state.npz").is_file()
    with (result.run_dir / "tables" / "training_log.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["updated"] == "False"
    assert rows[1]["updated"] == "True"
    with (result.run_dir / "tables" / "training_diagnostics.csv").open(encoding="utf-8", newline="") as handle:
        diagnostic_rows = list(csv.DictReader(handle))
    assert len(diagnostic_rows) == 2
    assert "theta_exc_median" in diagnostic_rows[0]
    assert "exc_active_neuron_fraction" in diagnostic_rows[0]
    assert "bcm_exc_above_theta_fraction" in diagnostic_rows[0]
    assert (result.run_dir / "analysis" / "training_health.json").is_file()
    assert (result.run_dir / "tables" / "training_health_events.csv").is_file()
    health = json.loads((result.run_dir / "analysis" / "training_health.json").read_text(encoding="utf-8"))
    assert not any(str(event["metric"]).startswith("inh_") for event in health["events"])
    assert not (result.run_dir / "figures" / "training_overview.png").exists()
    assert (result.run_dir / "arrays" / "training_probe_000001_exc_rates.npy").is_file()
    assert (result.run_dir / "arrays" / "training_probe_000002_weights.npy").is_file()

    manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["workflow"] == "train"
    assert manifest["learning_rule"] == "bcm"
    assert manifest["summary"]["samples_seen"] == 2
    assert manifest["summary"]["health_status"] in {"ok", "warn", "fail"}
    assert "health_warning_count" in manifest["summary"]
    assert "final_exc_active_neuron_fraction" in manifest["summary"]
    assert manifest["outputs"]["training_diagnostics"] == "tables/training_diagnostics.csv"
    assert manifest["outputs"]["training_health"] == "analysis/training_health.json"
    assert manifest["outputs"]["training_health_events"] == "tables/training_health_events.csv"
    assert manifest["outputs"]["training_probe_arrays"] == [
        "arrays/training_probe_000001_exc_rates.npy",
        "arrays/training_probe_000001_inh_rates.npy",
        "arrays/training_probe_000001_weights.npy",
        "arrays/training_probe_000002_exc_rates.npy",
        "arrays/training_probe_000002_inh_rates.npy",
        "arrays/training_probe_000002_weights.npy",
    ]
    assert result.summary["health_status"] in {"ok", "warn", "fail"}


def test_training_workflow_prints_live_status_to_stderr(tmp_path, capsys) -> None:
    cfg = _training_cfg(tmp_path)

    result = run_training(cfg, show_progress=True)
    captured = capsys.readouterr()

    assert str(result.run_dir) not in captured.out
    assert captured.err.count("[train] step=") == 2
    assert "health=" in captured.err
    assert "warn_total=" in captured.err
    assert "fail_total=" in captured.err
    assert "inh_active=-" in captured.err
    assert "[train] events:" in captured.err


def test_training_workflow_suppresses_live_status_without_progress(tmp_path, capsys) -> None:
    run_training(_training_cfg(tmp_path), show_progress=False)
    captured = capsys.readouterr()

    assert "[train] step=" not in captured.err
    assert "[train] events:" not in captured.err


def test_training_workflow_prints_live_status_only_on_probe_steps(tmp_path, capsys) -> None:
    run_training(_training_cfg(tmp_path, probe_every=2), show_progress=True)
    captured = capsys.readouterr()

    assert captured.err.count("[train] step=") == 1
    assert "step=2" in captured.err
    assert "step=1" not in captured.err


def test_training_workflow_writes_full_inspection_figures(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_tiny_iml(image_dir / "sample_001.iml")

    cfg = TrainingWorkflowConfig(
        run_root=tmp_path / "runs",
        empirical_data_path="data/sample_data.pkl",
        model=ModelConfig(
            l4=L4Config(n_side=1, region_size=1.0, all_tuned=True, n_orientations=1),
            l23=L23Config(n_side=4, inhibitory_fraction=0.5, region_size=1.0, random_inhibitory=False),
            p_ee=0.5,
            weight=WeightConfig(base_strength=0.1),
        ),
        natural_images=NaturalImageWorkflowConfig(
            image_dir=image_dir,
            image_shape=(4, 4),
            crop_size=None,
            patches_per_image=1,
            limit=1,
            receptive_field=ReceptiveFieldConfig(
                stimulus_size=0.5,
                resolution=3,
                gabor=GaborConfig(sigma=0.2, gamma=1.0, spatial_frequency=1.0, phase=0.0),
            ),
            preprocess=NaturalImagePreprocessConfig(resolution=4, normalization="maxscale"),
            drive=NaturalImageDriveConfig(visual_gain=0.01, baseline_rate=0.1),
        ),
        solver=SolverConfig(backend="scipy", scipy_method="RK4", store_trajectory=False),
        learning=LearningConfig(
            kind="bcm",
            bcm=BCMConfig(theta_init=1.0, eta=1e-5, row_sum_max_scale=1.0),
        ),
        background=BackgroundConfig(enabled=False),
        time=np.array([0.0, 0.01, 0.02], dtype=float),
        batch_size=1,
        epochs=2,
        inspection=TrainingInspectionConfig(
            enabled=True,
            tracked_weight_count=2,
            save_plots=True,
        ),
    )

    result = run_training(cfg, show_progress=False)

    for name in [
        "training_overview.png",
        "training_activity.png",
        "training_bcm.png",
        "training_plasticity.png",
        "training_row_sums.png",
        "tracked_weights.png",
    ]:
        assert (result.run_dir / "figures" / name).is_file()
