"""Readable cache keys for expensive input projections."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
from numpy.typing import NDArray

from v1_research.inputs.natural_images import NaturalImageSample
from v1_research.model.state import PopulationLayout


def stable_digest(payload: Any) -> str:
    """Hashes JSON-like payloads with stable key ordering."""

    data = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def array_signature(array: NDArray[Any]) -> dict[str, Any]:
    """Compact signature for arrays used in cache keys."""

    arr = np.ascontiguousarray(array)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "sha256": hashlib.sha256(arr.tobytes()).hexdigest(),
    }


def layout_signature(layout: PopulationLayout) -> dict[str, Any]:
    """Returns the L4 geometry and tuning signature relevant to input projections."""

    return {
        "l4": {
            "n_side": layout.l4.n_side,
            "region_size": layout.l4.region_size,
            "z_pos": layout.l4.z_pos,
            "coords": array_signature(np.asarray(layout.l4.coords, dtype=float)),
        },
        "tuning_labels": array_signature(np.asarray(layout.l4_tuning_labels)),
        "preferred_orientations": array_signature(
            np.nan_to_num(np.asarray(layout.l4_preferred_orientations, dtype=float), nan=0.0)
        ),
    }


def sample_signature(sample: NaturalImageSample) -> dict[str, Any]:
    """Returns the path/crop identity for one natural-image sample."""

    crop = None
    if sample.crop is not None:
        crop = {
            "top": sample.crop.top,
            "left": sample.crop.left,
            "height": sample.crop.height,
            "width": sample.crop.width,
        }
    return {"path": str(Path(sample.path)), "crop": crop}


def sample_manifest(samples: list[NaturalImageSample] | tuple[NaturalImageSample, ...]) -> list[dict[str, Any]]:
    """Returns a sorted unique sample manifest for order-independent cache keys."""

    return [sample_signature(sample) for sample in sorted_unique_samples(samples)]


def sorted_unique_samples(samples: list[NaturalImageSample] | tuple[NaturalImageSample, ...]) -> tuple[NaturalImageSample, ...]:
    """Sorts and deduplicates natural-image samples by path and crop."""

    return tuple(sorted(set(samples), key=lambda sample: json.dumps(sample_signature(sample), sort_keys=True)))


def gabor_projection_key(
    *,
    layout: PopulationLayout,
    rf_cfg: Any,
    drive_cfg: Any,
    frame_shape: tuple[int, int],
    dtype: str,
    backend: str,
) -> str:
    """Builds a cache key for an L4 Gabor projection matrix."""

    return stable_digest(
        {
            "kind": "gabor_projection_matrix",
            "layout": layout_signature(layout),
            "rf_cfg": _jsonable(rf_cfg),
            "drive_cfg": _jsonable(drive_cfg),
            "frame_shape": [int(frame_shape[0]), int(frame_shape[1])],
            "dtype": dtype,
            "backend": backend,
        }
    )


def natural_image_projection_key(
    *,
    layout: PopulationLayout,
    rf_cfg: Any,
    drive_cfg: Any,
    preprocess_cfg: Any,
    samples: list[NaturalImageSample] | tuple[NaturalImageSample, ...],
    dtype: str,
    backend: str,
) -> str:
    """Builds a cache key for projected natural-image sample rates."""

    return stable_digest(
        {
            "kind": "natural_image_projection",
            "layout": layout_signature(layout),
            "rf_cfg": _jsonable(rf_cfg),
            "drive_cfg": _jsonable(drive_cfg),
            "preprocess_cfg": _jsonable(preprocess_cfg),
            "samples": sample_manifest(samples),
            "frame_shape": [int(preprocess_cfg.resolution), int(preprocess_cfg.resolution)],
            "dtype": dtype,
            "backend": backend,
        }
    )


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return array_signature(value)
    if isinstance(value, np.generic):
        return value.item()
    return value
