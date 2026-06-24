from __future__ import annotations

import numpy as np
from scipy import sparse

from v1_research.learning import BCMConfig, BCMState, RateBatch
from v1_research.learning.diagnostics import (
    active_rate_stats,
    cap_fraction,
    plastic_weight_stats,
    record_tracked_weights,
    row_sum_pressure,
    sample_tracked_weights,
    theta_stats,
    weight_delta_stats,
)
from v1_research.model import ModelState, PopulationLayout, SheetGeometry


def _small_model(weights: np.ndarray | None = None) -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=2, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E", "I", "E", "I"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    values = (
        np.array(
            [
                [0.2, -0.4, 0.1, 0.0, 0.8],
                [0.3, -0.5, 0.2, -0.1, 0.7],
                [0.0, -0.2, 0.4, -0.3, 0.6],
                [0.5, 0.0, 0.1, -0.4, 0.5],
            ],
            dtype=float,
        )
        if weights is None
        else weights
    )
    return ModelState(
        layout=layout,
        connection_mask=sparse.csr_matrix(values != 0.0),
        weights=sparse.csr_matrix(values),
    )


def test_active_rate_stats_reports_active_fraction_and_percentiles() -> None:
    rates = RateBatch(
        exc=np.array([[0.0, 2.0], [4.0, 6.0]], dtype=float),
        inh=np.array([[1.0, 3.0], [5.0, 7.0]], dtype=float),
    )

    stats = active_rate_stats(rates, active_threshold=3.0)

    assert stats["exc_active_fraction"] == 0.5
    assert stats["inh_active_fraction"] == 0.5
    assert stats["exc_mean"] == 3.0
    assert stats["inh_max"] == 7.0


def test_plastic_weight_and_cap_stats_use_bcm_blocks_only() -> None:
    model = _small_model()

    stats = plastic_weight_stats(model)
    pressure = row_sum_pressure(model, state=BCMState(theta_exc=np.ones(2), theta_inh=np.ones(2)))
    caps = cap_fraction(np.array([[0.1, 0.9], [1.0, 0.99]]), limit=1.0, atol=0.02)

    assert stats["W_EE_nonzero"] == 3
    assert stats["W_IE_nonzero"] == 4
    assert stats["W_EE_mean"] > 0.0
    assert pressure["row_sum_EE_max"] > 0.0
    assert caps["fraction"] == 0.5
    assert caps["count"] == 2


def test_cap_fraction_rejects_unmatched_array_limits() -> None:
    values = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=float)

    with np.testing.assert_raises(ValueError):
        cap_fraction(values, limit=np.array([0.2, 0.3, 0.4], dtype=float))


def test_delta_theta_and_tracked_weight_diagnostics_are_deterministic_under_global_seed() -> None:
    np.random.seed(3)
    initial = _small_model()
    updated_values = initial.weights.toarray()
    updated_values[0, 0] += 0.05
    updated_values[1, 2] += 0.07
    updated = _small_model(updated_values)

    tracked = sample_tracked_weights(initial, count=2)
    rows = record_tracked_weights(updated, tracked, step=4)

    assert len(tracked) == 2
    assert [item.sample_index for item in tracked] == [0, 1]
    assert [item.sample_index for item in sample_tracked_weights(initial, count=100)] == list(range(7))
    assert {row["step"] for row in rows} == {4}
    assert all(row["current_weight"] >= row["initial_weight"] for row in rows)
    assert np.isclose(weight_delta_stats(initial, updated)["delta_EE_max"], 0.05)
    assert theta_stats(BCMState(theta_exc=np.array([1.0, 3.0]), theta_inh=np.array([2.0, 4.0]))) == {
        "theta_exc_mean": 2.0,
        "theta_exc_median": 2.0,
        "theta_inh_mean": 3.0,
        "theta_inh_median": 3.0,
    }
