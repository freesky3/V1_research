from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from v1_research.dynamics import TransferTable, WilsonCowanEquation
from v1_research.inputs import BackgroundTrace
from v1_research.model import ModelState, PopulationLayout, SheetGeometry


def _model(weights: np.ndarray) -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=2, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E", "I", "E", "I"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    return ModelState(
        layout=layout,
        connection_mask=sparse.csr_matrix(weights != 0.0),
        weights=sparse.csr_matrix(weights),
    )


def _linear_table() -> TransferTable:
    return TransferTable(mu=np.array([-100.0, 0.0, 100.0]), rate=np.array([-100.0, 0.0, 100.0]))


def test_wilson_cowan_equation_uses_model_drive_background_and_transfer() -> None:
    weights = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 0.0, 4.0],
            [0.0, 0.0, 0.0, 0.0, 5.0],
        ]
    )
    equation = WilsonCowanEquation(
        _model(weights),
        phi_exc=_linear_table(),
        phi_inh=_linear_table(),
        tau_exc=2.0,
        tau_inh=4.0,
        n_batch=2,
    )
    y = np.ones((4, 2))
    drive = np.array([[1.0, 2.0]])
    background = (np.array([[0.5, 1.0], [1.5, 2.0]]), np.array([[3.0, 4.0], [5.0, 6.0]]))

    dy = equation.rhs(y, drive, background=background)

    expected = np.array(
        [
            [(2.0 * 2.0 + 0.5 - 1.0) / 2.0, (2.0 * 4.0 + 1.5 - 1.0) / 2.0],
            [(4.0 * 3.0 + 3.0 - 1.0) / 4.0, (4.0 * 6.0 + 5.0 - 1.0) / 4.0],
            [(2.0 * 4.0 + 1.0 - 1.0) / 2.0, (2.0 * 8.0 + 2.0 - 1.0) / 2.0],
            [(4.0 * 5.0 + 4.0 - 1.0) / 4.0, (4.0 * 10.0 + 6.0 - 1.0) / 4.0],
        ]
    )
    np.testing.assert_allclose(dy, expected)


def test_wilson_cowan_equation_rejects_wrong_drive_shape() -> None:
    equation = WilsonCowanEquation(
        _model(np.zeros((4, 5))),
        phi_exc=_linear_table(),
        phi_inh=_linear_table(),
        tau_exc=1.0,
        tau_inh=1.0,
        n_batch=2,
    )

    with pytest.raises(ValueError, match="external drive shape"):
        equation.rhs(np.zeros((4, 2)), np.zeros((1, 1)))


def test_wilson_cowan_equation_checks_background_trace_shape() -> None:
    equation = WilsonCowanEquation(
        _model(np.zeros((4, 5))),
        phi_exc=_linear_table(),
        phi_inh=_linear_table(),
        tau_exc=1.0,
        tau_inh=1.0,
        n_batch=2,
    )
    trace = BackgroundTrace(
        time=np.array([0.0, 1.0]),
        exc=np.zeros((2, 1, 2)),
        inh=np.zeros((2, 1, 2)),
    )

    with pytest.raises(ValueError, match="Background trace shape mismatch"):
        equation.validate_background_trace(trace, np.array([0.0, 1.0]))
