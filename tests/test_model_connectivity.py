from __future__ import annotations

import numpy as np

from v1_research.model.connectivity import (
    ConnectivityConfig,
    ConnectionProbabilities,
    SpatialKernelConfig,
    probability_block,
    probability_matrix,
)
from v1_research.model.geometry import SheetGeometry
from v1_research.model.state import PopulationLayout


def _layout() -> PopulationLayout:
    return PopulationLayout(
        l23=SheetGeometry(n_side=3, region_size=3.0, z_pos=0.1),
        l4=SheetGeometry(n_side=2, region_size=3.0, z_pos=0.0),
        l23_cell_types=np.array(["E", "I", "E", "E", "I", "E", "E", "I", "E"]),
        l4_tuning_labels=np.array(["T", "U", "T", "U"]),
        l4_preferred_orientations=np.array([0.0, np.nan, np.pi / 2.0, np.nan]),
    )


def test_probability_block_masks_before_row_equalization() -> None:
    score = np.ones((4, 4), dtype=float)
    valid = ~np.eye(4, dtype=bool)

    prob = probability_block(score, 0.25, valid_mask=valid, equalize_rows=True)

    assert np.all(prob.diagonal() == 0.0)
    assert np.allclose(prob.sum(axis=1) / valid.sum(axis=1), 0.25)


def test_probability_matrix_has_no_recurrent_self_connections() -> None:
    layout = _layout()
    cfg = ConnectivityConfig(
        probabilities=ConnectionProbabilities(ee=0.3, ei=0.2, ex=0.4, ie=0.25, ii=0.3, ix=0.35),
        kernel=SpatialKernelConfig(sigma_narrow=0.5, sigma_broad=1.5, kappa=0.4),
        periodic=True,
        equalize_indegree=True,
    )

    probabilities = probability_matrix(layout, cfg)

    ee = probabilities[np.ix_(layout.exc_idx, layout.exc_idx)]
    ii = probabilities[np.ix_(layout.inh_idx, layout.inh_idx)]
    assert np.all(ee.diagonal() == 0.0)
    assert np.all(ii.diagonal() == 0.0)


def test_probability_matrix_row_means_match_target_probabilities() -> None:
    layout = _layout()
    cfg = ConnectivityConfig(
        probabilities=ConnectionProbabilities(ee=0.3, ei=0.2, ex=0.4, ie=0.25, ii=0.3, ix=0.35),
        kernel=SpatialKernelConfig(sigma_narrow=0.5, sigma_broad=1.5, kappa=0.4),
        periodic=True,
        equalize_indegree=True,
    )

    probabilities = probability_matrix(layout, cfg)

    assert np.allclose(
        probabilities[np.ix_(layout.exc_idx, layout.exc_idx)].sum(axis=1) / (layout.n_exc - 1),
        cfg.probabilities.ee,
    )
    assert np.allclose(
        probabilities[np.ix_(layout.inh_idx, layout.inh_idx)].sum(axis=1) / (layout.n_inh - 1),
        cfg.probabilities.ii,
    )
    assert np.allclose(probabilities[np.ix_(layout.exc_idx, layout.inh_idx)].mean(axis=1), cfg.probabilities.ei)
    assert np.allclose(probabilities[np.ix_(layout.inh_idx, layout.exc_idx)].mean(axis=1), cfg.probabilities.ie)
    assert np.allclose(probabilities[np.ix_(layout.exc_idx, layout.input_idx)].mean(axis=1), cfg.probabilities.ex)
    assert np.allclose(probabilities[np.ix_(layout.inh_idx, layout.input_idx)].mean(axis=1), cfg.probabilities.ix)
