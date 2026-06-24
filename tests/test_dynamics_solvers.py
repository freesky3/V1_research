from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from scipy import sparse

from v1_research.dynamics import JaxRK4Solver, ScipySolver, SolverConfig, TransferTable, solve_rates
from v1_research.inputs import BackgroundTrace
from v1_research.model import ModelState, PopulationLayout, SheetGeometry


def _small_model() -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=2, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E", "I", "E", "I"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = np.array(
        [
            [0.0, -0.2, 0.0, 0.0, 0.5],
            [0.1, -0.3, 0.0, 0.0, 0.4],
            [0.0, 0.0, 0.0, -0.2, 0.3],
            [0.0, 0.0, 0.1, -0.3, 0.2],
        ],
        dtype=float,
    )
    return ModelState(layout=layout, connection_mask=sparse.csr_matrix(weights != 0.0), weights=sparse.csr_matrix(weights))


def _transfer_table() -> TransferTable:
    return TransferTable(mu=np.array([-10.0, 0.0, 10.0]), rate=np.array([0.0, 0.0, 10.0]), rate_max=50.0)


def _drive(t: float) -> np.ndarray:
    return np.array([[1.0 + 0.1 * float(t), 0.5]], dtype=float)


def test_scipy_solver_returns_finite_rates_and_trajectory() -> None:
    time = np.linspace(0.0, 0.2, 6)
    result = ScipySolver(method="RK4").solve(
        _small_model(),
        drive=_drive,
        time=time,
        n_batch=2,
        phi_exc=_transfer_table(),
        phi_inh=_transfer_table(),
        tau_exc=0.02,
        tau_inh=0.01,
        store_trajectory=True,
    )

    assert result.exc.shape == (2, 2)
    assert result.inh.shape == (2, 2)
    assert result.exc_trajectory is not None
    assert result.exc_trajectory.shape == (6, 2, 2)
    assert np.all(np.isfinite(result.exc))
    assert np.all(np.isfinite(result.inh))


def test_solve_rates_uses_solver_config_backend() -> None:
    time = np.linspace(0.0, 0.1, 4)
    result = solve_rates(
        _small_model(),
        drive=_drive,
        time=time,
        n_batch=2,
        cfg=SolverConfig(backend="scipy", scipy_method="RK4", store_trajectory=False),
        phi_exc=_transfer_table(),
        phi_inh=_transfer_table(),
    )

    assert result.exc_trajectory is None
    assert result.inh_trajectory is None
    assert result.time.shape == (4,)


def test_background_trace_changes_solver_result() -> None:
    time = np.linspace(0.0, 0.2, 6)
    no_background = ScipySolver(method="RK4").solve(
        _small_model(),
        drive=_drive,
        time=time,
        n_batch=2,
        phi_exc=_transfer_table(),
        phi_inh=_transfer_table(),
        tau_exc=0.02,
        tau_inh=0.01,
    )
    background = BackgroundTrace(
        time=time,
        exc=np.full((time.size, 2, 2), 0.2),
        inh=np.full((time.size, 2, 2), 0.1),
    )
    with_background = ScipySolver(method="RK4").solve(
        _small_model(),
        drive=_drive,
        time=time,
        n_batch=2,
        phi_exc=_transfer_table(),
        phi_inh=_transfer_table(),
        tau_exc=0.02,
        tau_inh=0.01,
        background_trace=background,
    )

    assert not np.allclose(no_background.exc, with_background.exc)


def test_jax_rk4_matches_scipy_rk4_when_jax_is_installed() -> None:
    if importlib.util.find_spec("jax") is None:
        pytest.skip("JAX is not installed.")
    time = np.linspace(0.0, 0.2, 6)
    kwargs = dict(
        model=_small_model(),
        drive=_drive,
        time=time,
        n_batch=2,
        phi_exc=_transfer_table(),
        phi_inh=_transfer_table(),
        tau_exc=0.02,
        tau_inh=0.01,
        store_trajectory=True,
    )

    scipy_result = ScipySolver(method="RK4").solve(**kwargs)
    jax_result = JaxRK4Solver(dtype="float64").solve(**kwargs)

    np.testing.assert_allclose(jax_result.exc, scipy_result.exc, rtol=1.0e-6, atol=1.0e-6)
    np.testing.assert_allclose(jax_result.inh, scipy_result.inh, rtol=1.0e-6, atol=1.0e-6)
