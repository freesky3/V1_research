from __future__ import annotations

import numpy as np
import pytest

from v1_research.inputs import (
    CropBox,
    GaborConfig,
    L4NaturalImageProjector,
    NaturalImageDriveConfig,
    NaturalImageL4Drive,
    NaturalImagePreprocessConfig,
    NaturalImagePreprocessor,
    NaturalImageSample,
    NaturalImageSampler,
    ReceptiveFieldConfig,
    VanHaterenImageDataset,
    apply_crop,
    read_van_hateren_iml,
)
from v1_research.model import PopulationLayout, SheetGeometry


def _layout() -> PopulationLayout:
    return PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=2.0, z_pos=0.1),
        l4=SheetGeometry(n_side=1, region_size=2.0, z_pos=0.0),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["U"]),
        l4_preferred_orientations=np.array([np.nan]),
    )


def test_read_dataset_sampler_and_crop_use_global_seed(tmp_path) -> None:
    shape = (3, 4)
    first = tmp_path / "im1.iml"
    second = tmp_path / "im2.iml"
    np.arange(12, dtype=">u2").tofile(first)
    np.arange(12, 24, dtype=">u2").tofile(second)

    image = read_van_hateren_iml(first, shape=shape)
    assert image.shape == shape
    np.testing.assert_array_equal(apply_crop(image, CropBox(1, 1, 2, 2)), np.array([[5, 6], [9, 10]]))

    dataset = VanHaterenImageDataset(tmp_path, shape=shape)
    sampler = NaturalImageSampler(dataset, crop_size=2, patches_per_image=2)
    np.random.seed(123)
    first_epoch = sampler.make_epoch(limit=1)
    np.random.seed(123)
    second_epoch = sampler.make_epoch(limit=1)
    assert first_epoch == second_epoch
    assert len(first_epoch) == 2
    assert all(sample.crop is not None for sample in first_epoch)

    bad = tmp_path / "bad.iml"
    np.array([1, 2], dtype=">u2").tofile(bad)
    with pytest.raises(ValueError, match="pixels, expected"):
        read_van_hateren_iml(bad, shape=shape)


def test_preprocess_projector_and_static_drive() -> None:
    class Dataset:
        def read(self, path):
            return np.ones((3, 3), dtype=float)

    preprocessor = NaturalImagePreprocessor(NaturalImagePreprocessConfig(resolution=3, normalization="maxscale"))
    frame = preprocessor.transform(
        np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        NaturalImageSample("x.iml"),
    )
    assert np.max(frame) == pytest.approx(1.0)

    projector = L4NaturalImageProjector(
        _layout(),
        ReceptiveFieldConfig(stimulus_size=2.0, resolution=3, gabor=GaborConfig(sigma=0.5, spatial_frequency=1.0)),
        NaturalImageDriveConfig(visual_gain=2.0, baseline_rate=1.0, projection_chunk_size=1),
    )
    rates = projector.project(np.ones((3, 3), dtype=float))
    assert rates.shape == (1,)
    assert rates[0] >= 0.0
    with pytest.raises(ValueError, match="two-dimensional"):
        projector.project(np.ones(9))

    drive = NaturalImageL4Drive(dataset=Dataset(), preprocessor=preprocessor, projector=projector)  # type: ignore[arg-type]
    samples = [NaturalImageSample("a.iml"), NaturalImageSample("b.iml")]
    batch = drive.make_static_batch_func(samples)
    assert batch(0.0).shape == (1, 2)
    assert not batch(0.0).flags.writeable
