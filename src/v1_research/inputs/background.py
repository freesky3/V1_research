"""Background drive traces for rate solvers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
InterpolationMode = Literal["linear", "sample_hold"]


@dataclass(frozen=True, slots=True)
class OUParams:
    """Stationary Ornstein-Uhlenbeck process parameters."""

    mean: float = 0.0
    stationary_std: float = 0.0
    tau: float = 0.05

    def __post_init__(self) -> None:
        if not np.isfinite(float(self.mean)):
            raise ValueError("OU mean must be finite.")
        if not np.isfinite(float(self.stationary_std)) or float(self.stationary_std) < 0.0:
            raise ValueError("OU stationary_std must be finite and non-negative.")
        if not np.isfinite(float(self.tau)) or float(self.tau) <= 0.0:
            raise ValueError("OU tau must be finite and positive.")


@dataclass(frozen=True, slots=True)
class BackgroundConfig:
    """Optional OU background configuration for E and I populations."""

    enabled: bool = False
    exc: OUParams = field(default_factory=OUParams)
    inh: OUParams = field(default_factory=OUParams)
    interpolation: InterpolationMode = "linear"


@dataclass(frozen=True, slots=True)
class RK4BackgroundSamples:
    """Pre-sampled background arrays for RK4 left/mid/right stages."""

    exc_left: FloatArray
    inh_left: FloatArray
    exc_mid: FloatArray
    inh_mid: FloatArray
    exc_right: FloatArray
    inh_right: FloatArray


@dataclass(frozen=True, slots=True)
class BackgroundTrace:
    """Time-major background drive consumed by solvers."""

    time: FloatArray
    exc: FloatArray
    inh: FloatArray
    interpolation: InterpolationMode = "linear"

    def __post_init__(self) -> None:
        time = validate_time_grid(self.time, copy=True)
        exc = _trace_array(self.exc, "exc")
        inh = _trace_array(self.inh, "inh")
        interpolation = _interpolation_mode(self.interpolation)
        if exc.shape[0] != time.size:
            raise ValueError(f"exc time dimension {exc.shape[0]} does not match time size {time.size}.")
        if inh.shape[0] != time.size:
            raise ValueError(f"inh time dimension {inh.shape[0]} does not match time size {time.size}.")
        if exc.shape[1] != inh.shape[1]:
            raise ValueError(f"exc batch size {exc.shape[1]} does not match inh batch size {inh.shape[1]}.")
        time.setflags(write=False)
        exc.setflags(write=False)
        inh.setflags(write=False)
        object.__setattr__(self, "time", time)
        object.__setattr__(self, "exc", exc)
        object.__setattr__(self, "inh", inh)
        object.__setattr__(self, "interpolation", interpolation)

    @property
    def n_time(self) -> int:
        """Number of time samples."""

        return self.time.size

    @property
    def n_batch(self) -> int:
        """Number of parallel trials."""

        return self.exc.shape[1]

    @property
    def n_exc(self) -> int:
        """Number of excitatory cells."""

        return self.exc.shape[2]

    @property
    def n_inh(self) -> int:
        """Number of inhibitory cells."""

        return self.inh.shape[2]

    def validate_shape(self, *, n_exc: int, n_inh: int, n_batch: int) -> None:
        """Checks trace dimensions against solver expectations."""

        expected = (int(n_batch), int(n_exc), int(n_inh))
        actual = (self.n_batch, self.n_exc, self.n_inh)
        if actual != expected:
            raise ValueError(f"Background trace shape mismatch: got batch/exc/inh={actual}, expected {expected}.")

    def value_at(self, t: float) -> tuple[FloatArray, FloatArray]:
        """Interpolates E/I background values at time ``t``."""

        if self.interpolation == "sample_hold":
            index = int(np.searchsorted(self.time, float(t), side="right") - 1)
            index = min(max(index, 0), self.n_time - 1)
            return self.exc[index], self.inh[index]
        return _linear_value_at(self.time, self.exc, float(t)), _linear_value_at(self.time, self.inh, float(t))

    def rk4_samples(self) -> RK4BackgroundSamples:
        """Precomputes background values at RK4 left/mid/right stages."""

        if self.n_time < 2:
            raise ValueError("RK4 background samples require at least two time points.")
        if self.interpolation == "linear":
            exc_mid = 0.5 * (self.exc[:-1] + self.exc[1:])
            inh_mid = 0.5 * (self.inh[:-1] + self.inh[1:])
        else:
            exc_mid = self.exc[:-1]
            inh_mid = self.inh[:-1]
        return RK4BackgroundSamples(
            exc_left=self.exc[:-1],
            inh_left=self.inh[:-1],
            exc_mid=exc_mid,
            inh_mid=inh_mid,
            exc_right=self.exc[1:],
            inh_right=self.inh[1:],
        )


def generate_background_trace(
    cfg: BackgroundConfig,
    *,
    n_exc: int,
    n_inh: int,
    n_batch: int,
    time: FloatArray,
) -> BackgroundTrace | None:
    """Generates OU background, or ``None`` when disabled."""

    if not cfg.enabled:
        return None
    return generate_ou_background(
        n_exc=n_exc,
        n_inh=n_inh,
        n_batch=n_batch,
        time=time,
        exc=cfg.exc,
        inh=cfg.inh,
        interpolation=cfg.interpolation,
    )


def generate_ou_background(
    *,
    n_exc: int,
    n_inh: int,
    n_batch: int,
    time: FloatArray,
    exc: OUParams,
    inh: OUParams,
    interpolation: InterpolationMode = "linear",
) -> BackgroundTrace:
    """Generates E/I OU traces using the global NumPy RNG."""

    time = validate_time_grid(time, copy=True)
    n_exc = _non_negative_int(n_exc, "n_exc")
    n_inh = _non_negative_int(n_inh, "n_inh")
    n_batch = _positive_int(n_batch, "n_batch")
    return BackgroundTrace(
        time=time,
        exc=_generate_ou_population(n_units=n_exc, n_batch=n_batch, time=time, params=exc),
        inh=_generate_ou_population(n_units=n_inh, n_batch=n_batch, time=time, params=inh),
        interpolation=interpolation,
    )


def validate_time_grid(value: FloatArray, *, copy: bool = True) -> FloatArray:
    """Returns a finite, strictly increasing one-dimensional time grid."""

    time = np.array(value, dtype=np.float64, copy=copy)
    if time.ndim != 1 or time.size == 0:
        raise ValueError("time must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(time)):
        raise ValueError("time must contain only finite values.")
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    return time


def _generate_ou_population(*, n_units: int, n_batch: int, time: FloatArray, params: OUParams) -> FloatArray:
    values = np.empty((time.size, n_batch, n_units), dtype=np.float64)
    if n_units == 0:
        return values
    if params.stationary_std == 0.0:
        values.fill(float(params.mean))
        return values
    values[0] = float(params.mean) + float(params.stationary_std) * np.random.standard_normal((n_batch, n_units))
    for step, dt in enumerate(np.diff(time), start=1):
        alpha = float(np.exp(-float(dt) / float(params.tau)))
        innovation_std = float(params.stationary_std) * np.sqrt(max(0.0, 1.0 - alpha * alpha))
        values[step] = (
            float(params.mean)
            + alpha * (values[step - 1] - float(params.mean))
            + innovation_std * np.random.standard_normal((n_batch, n_units))
        )
    return values


def _linear_value_at(time: FloatArray, values: FloatArray, t: float) -> FloatArray:
    if t <= time[0]:
        return values[0]
    if t >= time[-1]:
        return values[-1]
    right = int(np.searchsorted(time, t, side="right"))
    left = right - 1
    weight = (t - time[left]) / (time[right] - time[left])
    return (1.0 - weight) * values[left] + weight * values[right]


def _trace_array(value: FloatArray, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True, order="C")
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (n_time, n_batch, n_units).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def _interpolation_mode(value: str) -> InterpolationMode:
    if value not in {"linear", "sample_hold"}:
        raise ValueError("interpolation must be 'linear' or 'sample_hold'.")
    return value  # type: ignore[return-value]


def _non_negative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be non-negative.")
    return value


def _positive_int(value: int, name: str) -> int:
    value = _non_negative_int(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
