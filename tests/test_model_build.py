from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import sparse

from v1_research.data import ExperimentalData
from v1_research.model import L4Config, L23Config
from v1_research.model.build import ModelConfig, build_model

ROOT = Path(__file__).resolve().parents[1]


def _small_config() -> ModelConfig:
    return ModelConfig(
        l4=L4Config(n_side=8, region_size=2.0, z_pos=0.0, all_tuned=True, n_orientations=8),
        l23=L23Config(n_side=14, inhibitory_fraction=None, region_size=2.0, z_pos=0.1, random_inhibitory=False),
    )


def test_build_model_uses_sample_data_to_create_expected_shapes() -> None:
    empirical = ExperimentalData.from_path(ROOT / "data" / "sample_data.pkl")
    np.random.seed(7)
    model = build_model(_small_config(), empirical)

    assert model.layout.n_exc == 166
    assert model.layout.n_inh == 30
    assert model.layout.n_input == 64
    assert model.shape == (196, 260)
    assert sparse.isspmatrix_csr(model.connection_mask)
    assert sparse.isspmatrix_csr(model.weights)


def test_build_model_reproducibility_comes_from_global_numpy_seed() -> None:
    empirical = ExperimentalData.from_path(ROOT / "data" / "sample_data.pkl")
    cfg = _small_config()

    np.random.seed(7)
    first = build_model(cfg, empirical)
    np.random.seed(7)
    second = build_model(cfg, empirical)
    np.random.seed(8)
    third = build_model(cfg, empirical)

    assert (first.connection_mask != second.connection_mask).nnz == 0
    assert (first.weights != second.weights).nnz == 0
    assert (first.connection_mask != third.connection_mask).nnz + (first.weights != third.weights).nnz > 0


def test_build_model_weight_signs_follow_source_population() -> None:
    empirical = ExperimentalData.from_path(ROOT / "data" / "sample_data.pkl")
    np.random.seed(7)
    model = build_model(_small_config(), empirical)

    q = model.connection_mask.toarray()
    weights = model.weights.toarray()
    active_exc_sources = weights[:, model.layout.exc_idx][q[:, model.layout.exc_idx]]
    active_inh_sources = weights[:, model.layout.inh_idx][q[:, model.layout.inh_idx]]
    active_input_sources = weights[:, model.layout.input_idx][q[:, model.layout.input_idx]]

    assert np.all(active_exc_sources >= 0.0)
    assert np.all(active_inh_sources <= 0.0)
    assert np.all(active_input_sources >= 0.0)
