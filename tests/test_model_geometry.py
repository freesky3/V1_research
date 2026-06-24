from __future__ import annotations

import numpy as np

from v1_research.model.geometry import (
    L4Config,
    SheetGeometry,
    assign_l4_tuning,
    uniform_grid_indices,
)


def test_sheet_geometry_coordinates_are_centered_grid() -> None:
    sheet = SheetGeometry(n_side=2, region_size=2.0, z_pos=0.1)

    assert sheet.n_cells == 4
    assert sheet.coords.shape == (4, 2)
    assert np.allclose(
        sheet.coords,
        np.array(
            [
                [-0.5, -0.5],
                [0.5, -0.5],
                [-0.5, 0.5],
                [0.5, 0.5],
            ]
        ),
    )


def test_periodic_distance_wraps_across_sheet_boundary() -> None:
    sheet = SheetGeometry(n_side=4, region_size=4.0, z_pos=0.0)

    periodic = sheet.distance_matrix(periodic=True)
    nonperiodic = sheet.distance_matrix(periodic=False)

    assert periodic[0, 3] == 1.0
    assert nonperiodic[0, 3] == 3.0


def test_uniform_grid_indices_are_deterministic_and_spread_out() -> None:
    indices = uniform_grid_indices(n_side=4, count=4)

    assert np.array_equal(indices, np.array([0, 3, 12, 15]))


def test_l4_tuning_assignment_uses_global_numpy_seed() -> None:
    cfg = L4Config(n_side=3, region_size=2.0, z_pos=0.0, all_tuned=False, n_orientations=4)

    np.random.seed(12)
    first = assign_l4_tuning(cfg, tuned_fraction=0.5)
    np.random.seed(12)
    second = assign_l4_tuning(cfg, tuned_fraction=0.5)
    np.random.seed(13)
    third = assign_l4_tuning(cfg, tuned_fraction=0.5)

    assert np.array_equal(first.tuning_labels, second.tuning_labels)
    assert np.allclose(first.preferred_orientations, second.preferred_orientations, equal_nan=True)
    assert not (
        np.array_equal(first.tuning_labels, third.tuning_labels)
        and np.allclose(first.preferred_orientations, third.preferred_orientations, equal_nan=True)
    )
