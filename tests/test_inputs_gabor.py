from __future__ import annotations

import numpy as np
import pytest

from v1_research.inputs import GaborConfig, L4GaborBank, ReceptiveFieldConfig, VisualGrid, gabor_bank, gabor_kernel
from v1_research.model import PopulationLayout, SheetGeometry


def _layout() -> PopulationLayout:
    return PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=2.0, z_pos=0.1),
        l4=SheetGeometry(n_side=2, region_size=2.0, z_pos=0.0),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["T", "U", "T", "U"]),
        l4_preferred_orientations=np.array([0.0, np.nan, np.pi / 2.0, np.nan]),
    )


def test_visual_grid_and_gabor_bank_shapes() -> None:
    grid = VisualGrid.centered_midpoint(size=2.0, resolution=4)
    assert np.allclose(grid.x_axis, [-0.75, -0.25, 0.25, 0.75])
    assert grid.area_element == pytest.approx(0.25)

    cfg = GaborConfig(sigma=0.5, gamma=0.5, spatial_frequency=2.0, phase=0.0)
    tuned = gabor_kernel(grid, cfg, preferred_orientation=0.0, tuned=True)
    untuned = gabor_kernel(grid, cfg, preferred_orientation=np.pi / 4.0, tuned=False)
    assert tuned.shape == (4, 4)
    assert np.allclose(untuned, untuned.T)

    bank = gabor_bank(
        grid,
        cfg,
        preferred_orientations=np.array([0.0, np.pi / 2.0]),
        tuned=np.array([True, False]),
    )
    assert bank.shape == (2, 4, 4)


def test_l4_gabor_bank_uses_population_layout_tuning() -> None:
    bank = L4GaborBank(ReceptiveFieldConfig(stimulus_size=2.0, resolution=5, gabor=GaborConfig(sigma=0.5)), _layout())
    assert bank.filters.shape == (4, 5, 5)
    assert not bank.filters.flags.writeable

    missing = PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=2.0, z_pos=0.1),
        l4=SheetGeometry(n_side=1, region_size=2.0, z_pos=0.0),
        l23_cell_types=np.array(["E"]),
    )
    with pytest.raises(ValueError, match="L4 tuning"):
        L4GaborBank(ReceptiveFieldConfig(resolution=3), missing)
