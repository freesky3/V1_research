from __future__ import annotations

import numpy as np
import pytest

from v1_research.dynamics import TransferConfig, TransferGrid, TransferTable, build_transfer_tables


def test_transfer_table_interpolates_and_clamps_rates() -> None:
    table = TransferTable(mu=np.array([-1.0, 0.0, 1.0]), rate=np.array([0.0, 2.0, 5.0]), rate_max=4.0)

    assert table(-0.5) == pytest.approx(1.0)
    assert table(2.0) == pytest.approx(4.0)
    np.testing.assert_allclose(table(np.array([-2.0, 0.5, 2.0])), [0.0, 3.5, 4.0])


def test_siegert_transfer_tables_are_finite_and_monotonic() -> None:
    cfg = TransferConfig(mu_table_max=4.0, tau_exc=0.02, tau_inh=0.01, rate_max=200.0)
    tables = build_transfer_tables(cfg, grid=TransferGrid.symmetric(4.0, points_per_unit=20))

    exc_mu, exc_rate = tables.excitatory.as_arrays()
    inh_mu, inh_rate = tables.inhibitory.as_arrays()

    assert exc_mu.shape == exc_rate.shape
    assert inh_mu.shape == inh_rate.shape
    assert np.all(np.isfinite(exc_rate))
    assert np.all(np.isfinite(inh_rate))
    assert np.all(np.diff(exc_mu) > 0.0)
    assert np.all(np.diff(exc_rate) >= -1.0e-9)
    assert np.all(np.diff(inh_rate) >= -1.0e-9)
