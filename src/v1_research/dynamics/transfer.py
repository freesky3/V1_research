"""Siegert transfer tables for Wilson-Cowan rate dynamics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import cumulative_trapezoid
from scipy.special import erf

FloatArray = NDArray[np.float64]

DEFAULT_ASYMPTOTIC_THRESHOLD = 5.5
DEFAULT_INTEGRAL_GRID_STEP = 1.0e-3
DEFAULT_MAX_INTEGRAL_UPPER = 26.0


@dataclass(frozen=True, slots=True)
class TransferConfig:
    """Siegert transfer parameters shared by E and I populations."""

    sigma_t: float = 10.0
    tau_exc: float = 0.02
    tau_inh: float = 0.01
    refractory_tau: float = 0.002
    threshold: float = 20.0
    reset_potential: float = 10.0
    mu_table_max: float = 100.0
    rate_max: float | None = None


@dataclass(frozen=True, slots=True)
class TransferGrid:
    """Grid of mean membrane inputs used to tabulate transfer functions."""

    mu_min: float
    mu_max: float
    n_points: int

    @classmethod
    def symmetric(cls, mu_max: float, points_per_unit: int = 1000) -> "TransferGrid":
        """Creates a symmetric grid spanning ``[-mu_max, mu_max]``."""

        if float(mu_max) <= 0.0:
            raise ValueError("mu_max must be positive.")
        n_points = max(2, int(float(points_per_unit) * float(mu_max)))
        return cls(mu_min=-float(mu_max), mu_max=float(mu_max), n_points=n_points)

    def values(self) -> FloatArray:
        """Returns the grid values."""

        return np.linspace(float(self.mu_min), float(self.mu_max), int(self.n_points), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class TransferTable:
    """Interpolated firing-rate table."""

    mu: FloatArray
    rate: FloatArray
    rate_max: float | None = None

    def __post_init__(self) -> None:
        mu = np.asarray(self.mu, dtype=np.float64).reshape(-1).copy()
        rate = np.asarray(self.rate, dtype=np.float64).reshape(-1).copy()
        if mu.shape != rate.shape:
            raise ValueError("mu and rate must have the same shape.")
        if mu.size < 2:
            raise ValueError("transfer table needs at least two points.")
        if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(rate)):
            raise ValueError("transfer table values must be finite.")
        if np.any(np.diff(mu) <= 0.0):
            raise ValueError("mu values must be strictly increasing.")
        if self.rate_max is not None and float(self.rate_max) <= 0.0:
            raise ValueError("rate_max must be positive when set.")
        mu.setflags(write=False)
        rate.setflags(write=False)
        object.__setattr__(self, "mu", mu)
        object.__setattr__(self, "rate", rate)

    def __call__(self, mu: ArrayLike) -> float | FloatArray:
        """Interpolates firing rates for scalar or array input."""

        values = np.asarray(mu, dtype=np.float64)
        was_scalar = values.ndim == 0
        out = np.interp(values, self.mu, self.rate, left=self.rate[0], right=self.rate[-1])
        if self.rate_max is not None:
            out = np.clip(out, 0.0, float(self.rate_max))
        return float(out) if was_scalar else out

    def as_arrays(self) -> tuple[FloatArray, FloatArray]:
        """Returns raw table arrays."""

        return self.mu, self.rate


@dataclass(frozen=True, slots=True)
class TransferTables:
    """Excitatory and inhibitory transfer tables."""

    excitatory: TransferTable
    inhibitory: TransferTable


def siegert_kernel(
    x: ArrayLike,
    *,
    asymptotic_threshold: float = DEFAULT_ASYMPTOTIC_THRESHOLD,
) -> float | FloatArray:
    """Evaluates the Siegert integral kernel."""

    arr = np.asarray(x, dtype=np.float64)
    was_scalar = arr.ndim == 0
    values = np.atleast_1d(arr)
    out = np.full_like(values, np.nan, dtype=np.float64)

    large_pos = values > float(asymptotic_threshold)
    mid = (~large_pos) & (values >= -float(asymptotic_threshold))
    large_neg = values < -float(asymptotic_threshold)

    if np.any(large_pos):
        val = values[large_pos]
        out[large_pos] = 2.0 * np.exp(val * val)
    if np.any(mid):
        val = values[mid]
        out[mid] = np.exp(val * val) * (1.0 + erf(val))
    if np.any(large_neg):
        val = values[large_neg]
        val2 = val * val
        out[large_neg] = -1.0 / (np.sqrt(np.pi) * val) * (1.0 - 0.5 / val2 + 0.75 / (val2 * val2))

    return float(out[0]) if was_scalar else out


def integrate_siegert_kernel(
    lower: ArrayLike,
    upper: ArrayLike,
    *,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
    max_upper: float = DEFAULT_MAX_INTEGRAL_UPPER,
) -> FloatArray:
    """Integrates the Siegert kernel between broadcast lower/upper bounds."""

    if float(grid_step) <= 0.0:
        raise ValueError("grid_step must be positive.")
    lower_arr, upper_arr = np.broadcast_arrays(
        np.atleast_1d(np.asarray(lower, dtype=np.float64)),
        np.atleast_1d(np.asarray(upper, dtype=np.float64)),
    )
    out = np.empty_like(upper_arr, dtype=np.float64)
    finite = upper_arr <= float(max_upper)
    out[~finite] = np.inf

    nans = np.isnan(lower_arr) | np.isnan(upper_arr)
    out[nans] = np.nan
    if not np.any(finite):
        return out

    bounds = np.concatenate([lower_arr[finite], upper_arr[finite]])
    valid = bounds[~np.isnan(bounds)]
    if valid.size == 0:
        out[finite] = np.nan
        return out

    grid_min = float(np.min(valid))
    grid_max = float(np.max(valid))
    if grid_min == grid_max:
        out[finite] = 0.0
        out[nans] = np.nan
        return out

    n_grid = max(2, int(np.ceil((grid_max - grid_min) / float(grid_step))) + 1)
    grid = np.linspace(grid_min, grid_max, n_grid, dtype=np.float64)
    antiderivative = cumulative_trapezoid(siegert_kernel(grid), grid, initial=0.0)
    lower_vals = np.interp(lower_arr[finite], grid, antiderivative)
    upper_vals = np.interp(upper_arr[finite], grid, antiderivative)
    out[finite] = upper_vals - lower_vals
    out[nans] = np.nan
    return out


def siegert_rate(
    mu: ArrayLike,
    *,
    tau_m: float,
    cfg: TransferConfig,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
) -> float | FloatArray:
    """Computes steady-state firing rate from the Siegert approximation."""

    if float(tau_m) <= 0.0:
        raise ValueError("tau_m must be positive.")
    if float(cfg.sigma_t) <= 0.0:
        raise ValueError("sigma_t must be positive.")
    if float(cfg.refractory_tau) <= 0.0:
        raise ValueError("refractory_tau must be positive.")

    values = np.asarray(mu, dtype=np.float64)
    was_scalar = values.ndim == 0
    arr = np.atleast_1d(values)
    if np.any(np.abs(arr) > 100.0):
        raise ValueError("mu contains values outside the supported range [-100, 100].")

    lower = (float(cfg.reset_potential) - arr) / float(cfg.sigma_t)
    upper = (float(cfg.threshold) - arr) / float(cfg.sigma_t)
    integral = integrate_siegert_kernel(lower, upper, grid_step=grid_step)
    rate = 1.0 / (float(cfg.refractory_tau) + float(tau_m) * np.sqrt(np.pi) * integral)
    return float(rate[0]) if was_scalar else rate


def build_transfer_table(
    *,
    tau_m: float,
    cfg: TransferConfig,
    grid: TransferGrid,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
) -> TransferTable:
    """Builds one transfer table."""

    mu = grid.values()
    rate = siegert_rate(mu, tau_m=float(tau_m), cfg=cfg, grid_step=grid_step)
    return TransferTable(mu=mu, rate=np.asarray(rate, dtype=np.float64), rate_max=cfg.rate_max)


def build_transfer_tables(
    cfg: TransferConfig,
    *,
    grid: TransferGrid | None = None,
    grid_step: float = DEFAULT_INTEGRAL_GRID_STEP,
) -> TransferTables:
    """Builds excitatory and inhibitory Siegert tables."""

    table_grid = TransferGrid.symmetric(cfg.mu_table_max) if grid is None else grid
    return TransferTables(
        excitatory=build_transfer_table(tau_m=cfg.tau_exc, cfg=cfg, grid=table_grid, grid_step=grid_step),
        inhibitory=build_transfer_table(tau_m=cfg.tau_inh, cfg=cfg, grid=table_grid, grid_step=grid_step),
    )
