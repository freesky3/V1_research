"""Orientation-selectivity analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def compute_osi(
    responses_mean: ArrayLike,
    orientation_angles: ArrayLike,
    *,
    min_osi: float = 0.4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Computes OSI and preferred simulated direction for each neuron.

    Args:
        responses_mean: Mean responses shaped ``(n_neurons, n_orientation)``.
        orientation_angles: Simulated directions in radians shaped ``(n_orientation,)``.
        min_osi: Preferred direction is reported only when OSI reaches this value.

    Returns:
        ``(osi, preferred_orientation)`` arrays shaped ``(n_neurons,)``. Preferred
        orientation is ``NaN`` for cells below threshold or with zero total response.
    """

    responses = np.asarray(responses_mean, dtype=float)
    angles = np.asarray(orientation_angles, dtype=float)
    if responses.ndim != 2:
        raise ValueError("responses_mean must have shape (n_neurons, n_orientation).")
    if angles.shape != (responses.shape[1],):
        raise ValueError(f"orientation_angles must have shape ({responses.shape[1]},), got {angles.shape}.")
    if not 0.0 <= float(min_osi) <= 1.0:
        raise ValueError("min_osi must be in [0, 1].")
    if not np.all(np.isfinite(responses)):
        raise ValueError("responses_mean contains NaN or infinite values.")
    if not np.all(np.isfinite(angles)):
        raise ValueError("orientation_angles contains NaN or infinite values.")

    wrapped = np.mod(angles, 2.0 * np.pi)
    vector = np.sum(responses * np.exp(2j * wrapped), axis=1)
    scalar = np.sum(responses, axis=1)
    osi = np.divide(np.abs(vector), scalar, out=np.zeros_like(scalar, dtype=float), where=scalar > 0.0)
    preferred = np.full(responses.shape[0], np.nan, dtype=float)
    valid = (osi >= float(min_osi)) & (scalar > 0.0)
    if np.any(valid):
        preferred[valid] = wrapped[np.argmax(responses[valid], axis=1)]
    return osi.astype(float, copy=False), preferred.astype(float, copy=False)
