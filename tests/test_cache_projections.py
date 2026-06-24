from __future__ import annotations

import numpy as np

from v1_research.cache import GaborProjectionMatrixCache, NaturalImageProjectionCache
from v1_research.inputs import (
    GaborConfig,
    L4NaturalImageProjector,
    NaturalImageDriveConfig,
    NaturalImageL4Drive,
    NaturalImagePreprocessConfig,
    NaturalImagePreprocessor,
    NaturalImageSample,
    ReceptiveFieldConfig,
)
from v1_research.model import PopulationLayout, SheetGeometry


def _layout(region_size: float = 2.0) -> PopulationLayout:
    return PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=region_size, z_pos=0.1),
        l4=SheetGeometry(n_side=1, region_size=region_size, z_pos=0.0),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["U"]),
        l4_preferred_orientations=np.array([np.nan]),
    )


def _drive(dataset) -> NaturalImageL4Drive:
    preprocessor = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="maxscale"))
    projector = L4NaturalImageProjector(
        _layout(),
        ReceptiveFieldConfig(stimulus_size=2.0, resolution=3, gabor=GaborConfig(sigma=0.5, spatial_frequency=1.0)),
        NaturalImageDriveConfig(visual_gain=1.0, baseline_rate=0.0),
    )
    return NaturalImageL4Drive(dataset=dataset, preprocessor=preprocessor, projector=projector)


class Dataset:
    def __init__(self) -> None:
        self.calls = 0

    def read(self, path):
        self.calls += 1
        return np.ones((3, 3), dtype=float)


def test_gabor_projection_matrix_cache_key_changes_with_layout(tmp_path) -> None:
    dataset = Dataset()
    drive = _drive(dataset)
    cache = GaborProjectionMatrixCache(tmp_path)
    first_key = cache.key(drive.projector, (3, 3))

    changed = L4NaturalImageProjector(
        _layout(region_size=3.0),
        drive.projector.rf_cfg,
        drive.projector.drive_cfg,
    )
    second_key = cache.key(changed, (3, 3))
    assert first_key != second_key

    matrix = cache.load_or_build(drive.projector, (3, 3))
    assert matrix.shape == (1, 9)
    assert not matrix.flags.writeable
    np.testing.assert_allclose(matrix, cache.load_or_build(drive.projector, (3, 3)))


def test_natural_image_projection_cache_hit_avoids_dataset_reads(tmp_path) -> None:
    samples = [NaturalImageSample("dummy1.iml"), NaturalImageSample("dummy2.iml")]

    dataset = Dataset()
    drive = _drive(dataset)
    cache = NaturalImageProjectionCache(tmp_path)
    miss = cache.load_or_build(drive, samples)
    assert dataset.calls == 2
    assert set(miss) == set(samples)

    dataset_hit = Dataset()
    drive_hit = _drive(dataset_hit)
    hit = cache.load_or_build(drive_hit, samples)
    assert dataset_hit.calls == 0
    for sample in samples:
        np.testing.assert_allclose(miss[sample], hit[sample])
