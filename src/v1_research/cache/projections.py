"""Disk caches for Gabor and natural-image input projections."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from v1_research.cache.keys import (
    gabor_projection_key,
    natural_image_projection_key,
    sample_manifest,
    sorted_unique_samples,
)
from v1_research.cache.storage import CacheStore
from v1_research.inputs.natural_images import (
    L4NaturalImageProjector,
    NaturalImageL4Drive,
    NaturalImageSample,
    rates_from_integrals,
)

BackendName = Literal["numpy", "jax"]


class GaborProjectionMatrixCache:
    """Cache for matrices mapping visual frame pixels to L4 RF integrals."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.store = CacheStore(cache_dir)

    def key(
        self,
        projector: L4NaturalImageProjector,
        frame_shape: tuple[int, int],
        *,
        dtype: str = "float64",
        backend: BackendName = "numpy",
    ) -> str:
        """Returns the projection-matrix cache key."""

        return gabor_projection_key(
            layout=projector.layout,
            rf_cfg=projector.rf_cfg,
            drive_cfg=projector.drive_cfg,
            frame_shape=frame_shape,
            dtype=dtype,
            backend=backend,
        )

    def load_or_build(
        self,
        projector: L4NaturalImageProjector,
        frame_shape: tuple[int, int],
        *,
        dtype: str = "float64",
        backend: BackendName = "numpy",
    ) -> NDArray[np.float64]:
        """Loads or builds a projection matrix."""

        key = self.key(projector, frame_shape, dtype=dtype, backend=backend)
        if self.store.has(key):
            matrix = self.store.load_array(key)
            expected = (projector.layout.n_input, int(frame_shape[0]) * int(frame_shape[1]))
            if matrix.shape == expected:
                return matrix

        matrix = projector.projection_matrix(frame_shape).astype(dtype, copy=False)
        matrix.setflags(write=False)
        self.store.save_array(
            key,
            matrix,
            {
                "kind": "gabor_projection_matrix",
                "key": key,
                "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
                "dtype": dtype,
                "backend": backend,
            },
        )
        return matrix


class NaturalImageProjectionCache:
    """Cache for projected natural-image sample rates."""

    def __init__(self, cache_dir: str | Path, *, matrix_cache: GaborProjectionMatrixCache | None = None) -> None:
        self.store = CacheStore(cache_dir)
        self.matrix_cache = matrix_cache or GaborProjectionMatrixCache(Path(cache_dir) / "matrices")

    def key(
        self,
        drive: NaturalImageL4Drive,
        samples: list[NaturalImageSample] | tuple[NaturalImageSample, ...],
        *,
        dtype: str = "float64",
        backend: BackendName = "numpy",
    ) -> str:
        """Returns the natural-image projection cache key."""

        return natural_image_projection_key(
            layout=drive.projector.layout,
            rf_cfg=drive.projector.rf_cfg,
            drive_cfg=drive.projector.drive_cfg,
            preprocess_cfg=drive.preprocessor.cfg,
            samples=samples,
            dtype=dtype,
            backend=backend,
        )

    def load_or_build(
        self,
        drive: NaturalImageL4Drive,
        samples: list[NaturalImageSample] | tuple[NaturalImageSample, ...],
        *,
        dtype: str = "float64",
        backend: BackendName = "numpy",
    ) -> dict[NaturalImageSample, NDArray[np.float64]]:
        """Loads or computes projected rates for the unique requested samples."""

        unique_samples = sorted_unique_samples(tuple(samples))
        key = self.key(drive, unique_samples, dtype=dtype, backend=backend)
        if self.store.has(key):
            rates = self.store.load_array(key)
            if rates.shape == (len(unique_samples), drive.projector.layout.n_input):
                return _sample_rate_map(unique_samples, rates)

        rates = self._build_rates(drive, unique_samples, dtype=dtype, backend=backend)
        rates.setflags(write=False)
        self.store.save_array(
            key,
            rates,
            {
                "kind": "natural_image_projection",
                "key": key,
                "samples": sample_manifest(unique_samples),
                "dtype": dtype,
                "backend": backend,
            },
        )
        return _sample_rate_map(unique_samples, rates)

    def _build_rates(
        self,
        drive: NaturalImageL4Drive,
        samples: tuple[NaturalImageSample, ...],
        *,
        dtype: str,
        backend: BackendName,
    ) -> NDArray[np.float64]:
        if not samples:
            return np.empty((0, drive.projector.layout.n_input), dtype=dtype)

        frames: list[NDArray[np.float64]] = []
        current_path = None
        current_image = None
        for sample in samples:
            if sample.path != current_path:
                current_path = sample.path
                current_image = drive.dataset.read(sample.path)
            frames.append(drive.preprocessor.transform(current_image, sample))

        first_shape = frames[0].shape
        if not all(frame.shape == first_shape for frame in frames):
            return np.asarray([drive.projector.project(frame) for frame in frames], dtype=dtype)

        matrix = self.matrix_cache.load_or_build(drive.projector, first_shape, dtype=dtype, backend=backend)
        flattened = np.vstack([np.asarray(frame, dtype=dtype).ravel() for frame in frames])
        integrals = flattened @ matrix.T
        return rates_from_integrals(integrals, drive.projector.drive_cfg).astype(dtype, copy=False)


def _sample_rate_map(
    samples: tuple[NaturalImageSample, ...],
    rates: NDArray[np.float64],
) -> dict[NaturalImageSample, NDArray[np.float64]]:
    result: dict[NaturalImageSample, NDArray[np.float64]] = {}
    for index, sample in enumerate(samples):
        row = np.asarray(rates[index])
        row.setflags(write=False)
        result[sample] = row
    return result
