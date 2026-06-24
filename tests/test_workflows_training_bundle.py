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
    run_training,
)


@dataclass(frozen=True, slots=True)
class FakeSolverConfig:
    backend: str = "fake"
    store_trajectory: bool = False


def _write_tiny_iml(path, value: int = 10) -> None:
    pixels = np.full((4, 4), value, dtype=">u2")
    pixels.tofile(path)


def test_training_workflow_writes_log_checkpoint_and_manifest(tmp_path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_tiny_iml(image_dir / "sample_001.iml")

    cfg = TrainingWorkflowConfig(
        run_root=tmp_path / "runs",
        empirical_data_path="data/sample_data.pkl",
        model=ModelConfig(
            l4=L4Config(n_side=1, region_size=1.0, all_tuned=True, n_orientations=1),
            l23=L23Config(n_side=1, inhibitory_fraction=0.0, region_size=1.0, random_inhibitory=False),
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
            save_plots=False,
            save_per_batch_arrays=True,
        ),
    )

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
