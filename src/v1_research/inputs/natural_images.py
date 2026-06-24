"""Natural-image inputs and L4 projection utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter, zoom

from v1_research.inputs.gabor import L4GaborBank, ReceptiveFieldConfig
from v1_research.model.state import PopulationLayout


VAN_HATEREN_SHAPE = (1024, 1536)
NormalizationMode = Literal["log-zscore", "zscore", "maxscale"]


@dataclass(frozen=True, slots=True)
class CropBox:
    """Pixel crop in top-left-height-width form."""

    top: int
    left: int
    height: int
    width: int


@dataclass(frozen=True, slots=True)
class NaturalImageSample:
    """One image path and optional crop to project onto L4."""

    path: Path
    crop: CropBox | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))


@dataclass(frozen=True, slots=True)
class NaturalImagePreprocessConfig:
    """Preprocessing applied before natural-image projection."""

    resolution: int = 128
    normalization: NormalizationMode = "log-zscore"
    clip_zscore: float | None = 3.0
    frame_scale: float = 1.0
    frame_offset: float = 0.0
    antialias: bool = True
    zscore_eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.resolution <= 1:
            raise ValueError("resolution must be greater than 1.")
        if self.clip_zscore is not None and self.clip_zscore <= 0.0:
            raise ValueError("clip_zscore must be positive when provided.")
        if not np.isfinite(float(self.frame_scale)) or float(self.frame_scale) < 0.0:
            raise ValueError("frame_scale must be finite and non-negative.")
        if not np.isfinite(float(self.frame_offset)):
            raise ValueError("frame_offset must be finite.")


@dataclass(frozen=True, slots=True)
class NaturalImageDriveConfig:
    """Parameters for converting RF integrals into L4 rates."""

    visual_gain: float = 400.0
    baseline_rate: float = 0.0
    periodic: bool = True
    projection_chunk_size: int = 64

    def __post_init__(self) -> None:
        if self.projection_chunk_size <= 0:
            raise ValueError("projection_chunk_size must be positive.")


def read_van_hateren_iml(path: str | Path, *, shape: tuple[int, int] = VAN_HATEREN_SHAPE) -> NDArray[np.uint16]:
    """Reads one big-endian Van Hateren ``.iml`` image."""

    path = Path(path)
    image = np.fromfile(path, dtype=">u2")
    expected = int(shape[0]) * int(shape[1])
    if image.size != expected:
        raise ValueError(f"{path} has {image.size} pixels, expected {expected}.")
    return image.reshape(shape)


class VanHaterenImageDataset:
    """Small file-backed Van Hateren image dataset wrapper."""

    def __init__(self, image_dir: str | Path, *, shape: tuple[int, int] = VAN_HATEREN_SHAPE, pattern: str = "*.iml") -> None:
        self.image_dir = Path(image_dir)
        self.shape = tuple(shape)
        self.paths = tuple(sorted(self.image_dir.glob(pattern)))
        if not self.paths:
            raise FileNotFoundError(f"No Van Hateren .iml files found in {self.image_dir}.")

    def read(self, path: str | Path) -> NDArray[np.uint16]:
        """Reads one dataset image."""

        return read_van_hateren_iml(path, shape=self.shape)

    def iter_paths(self, *, limit: int | None = None, shuffle: bool = True) -> tuple[Path, ...]:
        """Returns image paths, optionally shuffled by the global NumPy RNG."""

        paths = np.array(self.paths, dtype=object)
        if shuffle and paths.size:
            paths = paths[np.random.permutation(paths.size)]
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative.")
            paths = paths[:limit]
        return tuple(Path(path) for path in paths)


class NaturalImageSampler:
    """Draws natural-image crop samples using the global NumPy RNG."""

    def __init__(self, dataset: VanHaterenImageDataset, *, crop_size: int | None, patches_per_image: int = 1) -> None:
        if patches_per_image <= 0:
            raise ValueError("patches_per_image must be positive.")
        if crop_size is not None and crop_size <= 0:
            raise ValueError("crop_size must be positive when provided.")
        self.dataset = dataset
        self.crop_size = crop_size
        self.patches_per_image = int(patches_per_image)

    def make_epoch(
        self,
        *,
        limit: int | None = None,
        shuffle_paths: bool = True,
        shuffle_samples: bool = True,
    ) -> tuple[NaturalImageSample, ...]:
        """Builds one epoch of image crop samples."""

        samples: list[NaturalImageSample] = []
        for path in self.dataset.iter_paths(limit=limit, shuffle=shuffle_paths):
            for _ in range(self.patches_per_image):
                samples.append(NaturalImageSample(path=path, crop=self._sample_crop()))
        if shuffle_samples and samples:
            order = np.random.permutation(len(samples))
            samples = [samples[int(i)] for i in order]
        return tuple(samples)

    def _sample_crop(self) -> CropBox | None:
        if self.crop_size is None:
            return None
        height, width = self.dataset.shape
        if self.crop_size > height or self.crop_size > width:
            raise ValueError(f"crop_size={self.crop_size} exceeds image shape {self.dataset.shape}.")
        top = int(np.random.randint(0, height - self.crop_size + 1))
        left = int(np.random.randint(0, width - self.crop_size + 1))
        return CropBox(top=top, left=left, height=self.crop_size, width=self.crop_size)


class NaturalImagePreprocessor:
    """Crops, resizes, and normalizes natural images."""

    def __init__(self, cfg: NaturalImagePreprocessConfig) -> None:
        self.cfg = cfg

    def transform(self, image: NDArray[np.generic], sample: NaturalImageSample) -> NDArray[np.float64]:
        """Transforms a raw image into a normalized square frame."""

        frame = apply_crop(np.asarray(image), sample.crop)
        frame = self._resize(np.asarray(frame, dtype=float))
        frame = self._normalize(frame)
        frame = frame * float(self.cfg.frame_scale) + float(self.cfg.frame_offset)
        return frame.astype(float, copy=False)

    def _resize(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        target = int(self.cfg.resolution)
        if image.shape == (target, target):
            return image
        if self.cfg.antialias:
            image = self._antialias_before_downsample(image, target)
        factors = (target / image.shape[0], target / image.shape[1])
        return zoom(image, factors, order=1)

    def _antialias_before_downsample(self, image: NDArray[np.float64], target: int) -> NDArray[np.float64]:
        downsample = max(image.shape[0] / target, image.shape[1] / target)
        if downsample <= 1.0:
            return image
        sigma = max(0.0, (downsample - 1.0) / 2.0)
        return gaussian_filter(image, sigma=sigma, mode="nearest")

    def _normalize(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.cfg.normalization == "log-zscore":
            return self._zscore(np.log1p(np.maximum(image, 0.0)))
        if self.cfg.normalization == "zscore":
            return self._zscore(image)
        if self.cfg.normalization == "maxscale":
            max_value = float(np.max(image))
            if max_value <= float(self.cfg.zscore_eps):
                return np.zeros_like(image, dtype=float)
            return image / max_value
        raise ValueError(f"Unsupported natural image normalization: {self.cfg.normalization}")

    def _zscore(self, image: NDArray[np.float64]) -> NDArray[np.float64]:
        std = float(np.std(image))
        if std <= float(self.cfg.zscore_eps):
            return np.zeros_like(image, dtype=float)
        normalized = (image - float(np.mean(image))) / std
        if self.cfg.clip_zscore is not None:
            normalized = np.clip(normalized, -float(self.cfg.clip_zscore), float(self.cfg.clip_zscore))
        return normalized


class L4NaturalImageProjector:
    """Projects preprocessed visual frames onto L4 Gabor receptive fields."""

    def __init__(self, layout: PopulationLayout, rf_cfg: ReceptiveFieldConfig, drive_cfg: NaturalImageDriveConfig) -> None:
        self.layout = layout
        self.rf_cfg = rf_cfg
        self.drive_cfg = drive_cfg
        self.rf_bank = L4GaborBank(rf_cfg, layout)
        self._matrix_cache: dict[tuple[int, int], NDArray[np.float64]] = {}

        coords = np.asarray(layout.l4.coords, dtype=float)
        self.x_i = coords[:, 0]
        self.y_i = coords[:, 1]
        grid = self.rf_bank.grid
        self.image_x_min = float(np.min(self.x_i) + grid.x_axis[0])
        self.image_x_max = float(np.max(self.x_i) + grid.x_axis[-1])
        self.image_y_min = float(np.min(self.y_i) + grid.y_axis[0])
        self.image_y_max = float(np.max(self.y_i) + grid.y_axis[-1])

    def projection_matrix(self, frame_shape: tuple[int, int]) -> NDArray[np.float64]:
        """Returns the matrix that maps flattened frame pixels to L4 integrals."""

        if len(frame_shape) != 2:
            raise ValueError("frame_shape must be a two-dimensional shape.")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        key = (height, width)
        if key not in self._matrix_cache:
            matrix = self._build_projection_matrix(height, width)
            matrix.setflags(write=False)
            self._matrix_cache[key] = matrix
        return self._matrix_cache[key]

    def project(self, frame: NDArray[np.float64]) -> NDArray[np.float64]:
        """Projects one preprocessed image frame to L4 firing-rate drive."""

        frame = np.asarray(frame, dtype=float)
        if frame.ndim != 2:
            raise ValueError("frame must be a two-dimensional image.")
        matrix = self.projection_matrix(frame.shape)
        return rates_from_integrals(matrix @ frame.ravel(), self.drive_cfg)

    def project_frames(self, frames: Sequence[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Projects a same-shaped frame batch to ``(n_frames, n_l4)`` rates."""

        if not frames:
            return np.empty((0, self.layout.n_input), dtype=float)
        first = np.asarray(frames[0], dtype=float)
        if first.ndim != 2:
            raise ValueError("natural-image frames must be two-dimensional.")
        matrix = self.projection_matrix(first.shape)
        flattened = np.empty((len(frames), first.size), dtype=float)
        flattened[0] = first.ravel()
        for index, frame in enumerate(frames[1:], start=1):
            arr = np.asarray(frame, dtype=float)
            if arr.shape != first.shape:
                raise ValueError("all frames in a projection batch must have matching shapes.")
            flattened[index] = arr.ravel()
        integrals = flattened @ matrix.T
        return rates_from_integrals(integrals, self.drive_cfg)

    def _build_projection_matrix(self, height: int, width: int) -> NDArray[np.float64]:
        n_l4 = self.layout.n_input
        matrix = np.zeros((n_l4, height * width), dtype=float)
        grid = self.rf_bank.grid
        filters = self.rf_bank.filters
        chunk_size = int(self.drive_cfg.projection_chunk_size)
        dx_dy = grid.dx * grid.dy

        for start in range(0, n_l4, chunk_size):
            stop = min(start + chunk_size, n_l4)
            x = self.x_i[start:stop, np.newaxis, np.newaxis] + grid.x[np.newaxis, :, :]
            y = self.y_i[start:stop, np.newaxis, np.newaxis] + grid.y[np.newaxis, :, :]
            cols = self._coord_to_pixel(x, self.image_x_min, self.image_x_max, width)
            rows = self._coord_to_pixel(y, self.image_y_min, self.image_y_max, height)

            r0 = np.floor(rows).astype(np.int64)
            c0 = np.floor(cols).astype(np.int64)
            dr = rows - r0
            dc = cols - c0
            r1 = r0 + 1
            c1 = c0 + 1

            if self.drive_cfg.periodic:
                r0 = r0 % height
                r1 = r1 % height
                c0 = c0 % width
                c1 = c1 % width
            else:
                r0 = np.clip(r0, 0, height - 1)
                r1 = np.clip(r1, 0, height - 1)
                c0 = np.clip(c0, 0, width - 1)
                c1 = np.clip(c1, 0, width - 1)

            w00 = (1.0 - dr) * (1.0 - dc)
            w01 = (1.0 - dr) * dc
            w10 = dr * (1.0 - dc)
            w11 = dr * dc
            kernel = filters[start:stop] * dx_dy
            idx00 = r0 * width + c0
            idx01 = r0 * width + c1
            idx10 = r1 * width + c0
            idx11 = r1 * width + c1

            for local_idx in range(stop - start):
                row = start + local_idx
                np.add.at(matrix[row], idx00[local_idx].ravel(), (kernel[local_idx] * w00[local_idx]).ravel())
                np.add.at(matrix[row], idx01[local_idx].ravel(), (kernel[local_idx] * w01[local_idx]).ravel())
                np.add.at(matrix[row], idx10[local_idx].ravel(), (kernel[local_idx] * w10[local_idx]).ravel())
                np.add.at(matrix[row], idx11[local_idx].ravel(), (kernel[local_idx] * w11[local_idx]).ravel())
        return matrix

    @staticmethod
    def _coord_to_pixel(coord: NDArray[np.float64], coord_min: float, coord_max: float, n_pixels: int) -> NDArray[np.float64]:
        if coord_max <= coord_min:
            return np.zeros_like(coord, dtype=float)
        return (coord - coord_min) * (n_pixels - 1) / (coord_max - coord_min)


class NaturalImageL4Drive:
    """Dataset, preprocessing, projection, and optional in-memory rates cache."""

    def __init__(
        self,
        *,
        dataset: VanHaterenImageDataset,
        preprocessor: NaturalImagePreprocessor,
        projector: L4NaturalImageProjector,
        cached_rates: dict[NaturalImageSample, NDArray[np.float64]] | None = None,
    ) -> None:
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.projector = projector
        self._cached_rates: dict[NaturalImageSample, NDArray[np.float64]] = dict(cached_rates or {})

    def rates_for_sample(self, sample: NaturalImageSample) -> NDArray[np.float64]:
        """Returns static L4 rates for one natural-image sample."""

        if sample not in self._cached_rates:
            image = self.dataset.read(sample.path)
            frame = self.preprocessor.transform(image, sample)
            rates = self.projector.project(frame)
            rates.setflags(write=False)
            self._cached_rates[sample] = rates
        return self._cached_rates[sample]

    def make_static_func(self, sample: NaturalImageSample) -> Callable[[float], NDArray[np.float64]]:
        """Returns a time-independent drive function for one sample."""

        rates = self.rates_for_sample(sample)

        def drive(_t: float) -> NDArray[np.float64]:
            return rates

        drive.is_time_dependent = False  # type: ignore[attr-defined]
        return drive

    def make_static_batch_func(self, samples: Sequence[NaturalImageSample]) -> Callable[[float], NDArray[np.float64]]:
        """Returns a time-independent ``(n_l4, n_batch)`` drive function."""

        rate_matrix = np.column_stack([self.rates_for_sample(sample) for sample in samples])
        rate_matrix.setflags(write=False)

        def drive(_t: float) -> NDArray[np.float64]:
            return rate_matrix

        drive.is_time_dependent = False  # type: ignore[attr-defined]
        return drive


def apply_crop(image: NDArray[np.generic], crop: CropBox | None) -> NDArray[np.generic]:
    """Applies a natural-image crop, or returns the full image."""

    if crop is None:
        return image
    return image[crop.top : crop.top + crop.height, crop.left : crop.left + crop.width]


def rates_from_integrals(integrals: NDArray[np.float64], cfg: NaturalImageDriveConfig) -> NDArray[np.float64]:
    """Converts RF/image integrals into non-negative L4 rates."""

    return np.maximum(0.0, float(cfg.baseline_rate) + integrals) * float(cfg.visual_gain)
