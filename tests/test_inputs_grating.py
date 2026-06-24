from __future__ import annotations

import numpy as np

from v1_research.inputs import DriftingGratingConfig, DriftingGratingInput, GaborConfig, ReceptiveFieldConfig
from v1_research.model import PopulationLayout, SheetGeometry


def _layout() -> PopulationLayout:
    return PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=2.0, z_pos=0.1),
        l4=SheetGeometry(n_side=2, region_size=2.0, z_pos=0.0),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["T", "U", "T", "U"]),
        l4_preferred_orientations=np.array([0.0, np.nan, np.pi / 2.0, np.nan]),
    )


def test_drifting_grating_single_and_batched_drive() -> None:
    cfg = DriftingGratingConfig(
        receptive_field=ReceptiveFieldConfig(
            stimulus_size=2.0,
            resolution=5,
            gabor=GaborConfig(sigma=0.5, gamma=1.0, spatial_frequency=1.0, phase=0.0),
        ),
        baseline_rate=2.0,
        visual_gain=1.5,
        temporal_frequency=2.0 * np.pi,
        luminance=1.0,
        contrast=0.8,
        n_orientations=8,
    )
    stimulus = DriftingGratingInput(cfg, _layout())

    drive_0 = stimulus.external_drive(theta_stim=0.0, t=0.0)
    assert drive_0.shape == (4,)
    assert np.all(drive_0 >= 0.0)
    np.testing.assert_allclose(drive_0, stimulus.external_drive(theta_stim=0.0, t=0.0))

    batched = stimulus.make_batched_drive_func([0.0, np.pi / 2.0])
    assert batched(0.1).shape == (4, 2)

    shifted = stimulus.make_batched_drive_func([0.0], phase_offsets=[np.pi])
    np.testing.assert_allclose(shifted(0.0), stimulus.make_batched_drive_func([0.0])(0.5))
    assert not np.allclose(shifted(0.0), stimulus.make_batched_drive_func([0.0])(0.0))

    frame = stimulus.stimulus_frame(theta_stim=0.0, t=0.0)
    assert frame.shape == (4, 5, 5)
    assert np.min(frame) >= 0.19
    assert np.max(frame) <= 1.81
