from __future__ import annotations

import numpy as np
import pytest

from v1_research.analysis.osi import compute_osi


def test_compute_osi_marks_flat_cells_without_preferred_orientation() -> None:
    theta = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])
    responses = np.array(
        [
            [4.0, 0.0, 4.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )

    osi, preferred = compute_osi(responses, theta, min_osi=0.4)

    assert osi[0] == pytest.approx(1.0)
    assert preferred[0] == pytest.approx(0.0)
    assert osi[1] == pytest.approx(0.0)
    assert np.isnan(preferred[1])


def test_compute_osi_uses_first_peak_as_unfolded_direction() -> None:
    theta = np.array([0.0, np.pi / 2.0, np.pi])
    responses = np.array(
        [
            [1.0, 2.0, 5.0],
            [1.0, 5.0, 1.0],
        ]
    )

    osi, preferred = compute_osi(responses, theta, min_osi=0.0)

    assert osi.shape == (2,)
    assert preferred[0] == pytest.approx(np.pi)
    assert preferred[1] == pytest.approx(np.pi / 2.0)


def test_compute_osi_rejects_mismatched_orientation_count() -> None:
    with pytest.raises(ValueError, match="orientation_angles"):
        compute_osi(np.ones((2, 3)), np.ones(2))
