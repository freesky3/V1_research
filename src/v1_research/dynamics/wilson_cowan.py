"""Wilson-Cowan equations independent of solver backends."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import sparse

from v1_research.inputs.background import BackgroundTrace, validate_time_grid
from v1_research.model.state import ModelState, PopulationLayout
from v1_research.dynamics.transfer import TransferTable

FloatArray = NDArray[np.float64]
TransferFunction = Callable[[ArrayLike], ArrayLike]


class ExternalDrive(Protocol):
    """Continuous-time L4 drive returning ``(n_input, n_batch)`` values."""

    def __call__(self, t: float) -> ArrayLike:
        ...


@dataclass(frozen=True, slots=True)
class RateLayout:
    """Index arrays used by dynamics code."""

    exc_idx: NDArray[np.int64]
    inh_idx: NDArray[np.int64]
    input_idx: NDArray[np.int64]

    @classmethod
    def from_population_layout(cls, layout: PopulationLayout) -> "RateLayout":
        """Builds dynamics indexes from model layout."""

        return cls(
            exc_idx=_index_array(layout.exc_idx, "exc_idx"),
            inh_idx=_index_array(layout.inh_idx, "inh_idx"),
            input_idx=_index_array(layout.input_idx, "input_idx"),
        )

    @property
    def n_exc(self) -> int:
        """Number of excitatory dynamic cells."""

        return int(self.exc_idx.size)

    @property
    def n_inh(self) -> int:
        """Number of inhibitory dynamic cells."""

        return int(self.inh_idx.size)

    @property
    def n_input(self) -> int:
        """Number of external L4 input cells."""

        return int(self.input_idx.size)

    @property
    def n_rates(self) -> int:
        """Number of dynamic L2/3 cells."""

        return self.n_exc + self.n_inh


@dataclass(frozen=True, slots=True)
class WeightBlocks:
    """Weights split by source population."""

    exc: FloatArray
    inh: FloatArray
    external: FloatArray


class WilsonCowanEquation:
    """Batched Wilson-Cowan right-hand side for a fixed model."""

    def __init__(
        self,
        model: ModelState,
        *,
        phi_exc: TransferFunction,
        phi_inh: TransferFunction,
        tau_exc: float,
        tau_inh: float,
        n_batch: int,
    ) -> None:
        self.model = model
        self.layout = RateLayout.from_population_layout(model.layout)
        self.phi_exc = phi_exc
        self.phi_inh = phi_inh
        self.tau_exc = _positive_float(tau_exc, "tau_exc")
        self.tau_inh = _positive_float(tau_inh, "tau_inh")
        self.n_batch = _positive_int(n_batch, "n_batch")
        self.weights = _dense_weights(model.weights, shape=model.layout.shape)
        if self.layout.input_idx.size and int(self.layout.input_idx.max()) >= self.weights.shape[1]:
            raise ValueError("input_idx contains source columns outside the weights matrix.")
        if self.weights.shape[0] != self.layout.n_rates:
            raise ValueError("weights row count must match the number of dynamic rates.")

    def rhs(
        self,
        y: ArrayLike,
        external_drive: ArrayLike,
        *,
        background: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> FloatArray:
        """Evaluates ``dy/dt`` for one state and one L4 drive value."""

        rates = self.validate_state(y)
        drive = self.validate_external_drive_value(external_drive)
        sources = np.zeros((self.weights.shape[1], self.n_batch), dtype=np.float64)
        sources[self.layout.exc_idx, :] = rates[self.layout.exc_idx, :]
        sources[self.layout.inh_idx, :] = rates[self.layout.inh_idx, :]
        sources[self.layout.input_idx, :] = drive

        mu = self.weights @ sources
        exc_drive = self.tau_exc * mu[self.layout.exc_idx, :]
        inh_drive = self.tau_inh * mu[self.layout.inh_idx, :]
        if background is not None:
            bg_exc, bg_inh = self.validate_background_value(background)
            exc_drive = exc_drive + bg_exc
            inh_drive = inh_drive + bg_inh

        exc_rate = np.asarray(self.phi_exc(exc_drive), dtype=np.float64)
        inh_rate = np.asarray(self.phi_inh(inh_drive), dtype=np.float64)
        if exc_rate.shape != exc_drive.shape:
            raise ValueError(f"phi_exc returned shape {exc_rate.shape}, expected {exc_drive.shape}.")
        if inh_rate.shape != inh_drive.shape:
            raise ValueError(f"phi_inh returned shape {inh_rate.shape}, expected {inh_drive.shape}.")

        dy = np.empty_like(rates)
        dy[self.layout.exc_idx, :] = (-rates[self.layout.exc_idx, :] + exc_rate) / self.tau_exc
        dy[self.layout.inh_idx, :] = (-rates[self.layout.inh_idx, :] + inh_rate) / self.tau_inh
        return dy

    def rhs_flat(
        self,
        t: float,
        y_flat: ArrayLike,
        drive: ExternalDrive,
        *,
        background: tuple[ArrayLike, ArrayLike] | None = None,
    ) -> FloatArray:
        """Evaluates flattened ``dy/dt`` for SciPy adaptive solvers."""

        return self.rhs(np.asarray(y_flat, dtype=np.float64).reshape(self.layout.n_rates, self.n_batch), drive(t), background=background).ravel()

    def validate_state(self, y: ArrayLike) -> FloatArray:
        """Returns a dense rate state with shape ``(n_rates, n_batch)``."""

        rates = np.asarray(y, dtype=np.float64)
        if rates.shape != (self.layout.n_rates, self.n_batch):
            raise ValueError(f"rate state shape {rates.shape} != ({self.layout.n_rates}, {self.n_batch}).")
        if not np.all(np.isfinite(rates)):
            raise ValueError("rate state contains NaN or infinite values.")
        return rates

    def validate_external_drive_value(self, value: ArrayLike) -> FloatArray:
        """Returns a dense external drive with shape ``(n_input, n_batch)``."""

        drive = np.asarray(value, dtype=np.float64)
        if drive.ndim == 1:
            if self.n_batch != 1:
                raise ValueError(
                    f"external drive shape {drive.shape} != ({self.layout.n_input}, {self.n_batch})."
                )
            drive = drive[:, np.newaxis]
        if drive.shape != (self.layout.n_input, self.n_batch):
            raise ValueError(f"external drive shape {drive.shape} != ({self.layout.n_input}, {self.n_batch}).")
        if not np.all(np.isfinite(drive)):
            raise ValueError("external drive contains NaN or infinite values.")
        return drive

    def validate_background_value(self, background: tuple[ArrayLike, ArrayLike]) -> tuple[FloatArray, FloatArray]:
        """Returns background arrays transposed to equation state order."""

        bg_exc = np.asarray(background[0], dtype=np.float64)
        bg_inh = np.asarray(background[1], dtype=np.float64)
        if bg_exc.shape != (self.n_batch, self.layout.n_exc):
            raise ValueError(f"background exc shape {bg_exc.shape} != ({self.n_batch}, {self.layout.n_exc}).")
        if bg_inh.shape != (self.n_batch, self.layout.n_inh):
            raise ValueError(f"background inh shape {bg_inh.shape} != ({self.n_batch}, {self.layout.n_inh}).")
        return bg_exc.T, bg_inh.T

    def validate_background_trace(self, trace: BackgroundTrace | None, time: ArrayLike) -> None:
        """Checks that a background trace matches this equation and time grid."""

        if trace is None:
            return
        trace.validate_shape(n_exc=self.layout.n_exc, n_inh=self.layout.n_inh, n_batch=self.n_batch)
        solver_time = validate_time_grid(np.asarray(time, dtype=np.float64), copy=True)
        if trace.time.shape != solver_time.shape or not np.allclose(trace.time, solver_time, rtol=1.0e-12, atol=1.0e-15):
            raise ValueError("Background trace time grid does not match solver time grid.")

    def weight_blocks(self) -> WeightBlocks:
        """Returns dense weight blocks by source population."""

        return WeightBlocks(
            exc=self.weights[:, self.layout.exc_idx].copy(),
            inh=self.weights[:, self.layout.inh_idx].copy(),
            external=self.weights[:, self.layout.input_idx].copy(),
        )

    def transfer_table_arrays(self) -> tuple[FloatArray, FloatArray, float, FloatArray, FloatArray, float]:
        """Returns transfer tables in a JAX-friendly shape."""

        exc_mu, exc_rate, exc_max = _transfer_arrays(self.phi_exc, "phi_exc")
        inh_mu, inh_rate, inh_max = _transfer_arrays(self.phi_inh, "phi_inh")
        return exc_mu, exc_rate, exc_max, inh_mu, inh_rate, inh_max


def jax_transfer_interpolator(jnp):
    """Creates a small table interpolator for JAX kernels."""

    def interp_phi(x, xp, fp, rate_max):
        out = jnp.interp(x, xp, fp, left=fp[0], right=fp[-1])
        return jnp.where(jnp.isfinite(rate_max), jnp.clip(out, 0.0, rate_max), out)

    return interp_phi


def jax_wilson_cowan_rhs(jnp):
    """Creates the shared JAX Wilson-Cowan RHS used by JAX solvers."""

    interp_phi = jax_transfer_interpolator(jnp)

    def rhs(
        y,
        drive,
        bg_exc,
        bg_inh,
        weights_exc,
        weights_inh,
        weights_ext,
        idx_exc,
        idx_inh,
        phi_exc_mu,
        phi_exc_rate,
        phi_exc_rate_max,
        phi_inh_mu,
        phi_inh_rate,
        phi_inh_rate_max,
        tau_exc,
        tau_inh,
    ):
        mu = weights_exc @ y[idx_exc, :] + weights_inh @ y[idx_inh, :] + weights_ext @ drive
        dy = jnp.zeros_like(y)
        dy = dy.at[idx_exc, :].set(
            (-y[idx_exc, :] + interp_phi(tau_exc * mu[idx_exc, :] + bg_exc, phi_exc_mu, phi_exc_rate, phi_exc_rate_max))
            / tau_exc
        )
        dy = dy.at[idx_inh, :].set(
            (-y[idx_inh, :] + interp_phi(tau_inh * mu[idx_inh, :] + bg_inh, phi_inh_mu, phi_inh_rate, phi_inh_rate_max))
            / tau_inh
        )
        return dy

    return rhs


def _dense_weights(weights: sparse.spmatrix | ArrayLike, *, shape: tuple[int, int]) -> FloatArray:
    if sparse.issparse(weights):
        arr = weights.toarray().astype(np.float64, copy=False)
    else:
        arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(f"weights shape {arr.shape} does not match layout shape {shape}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError("weights contain NaN or infinite values.")
    return arr.astype(np.float64, copy=True)


def _transfer_arrays(phi: TransferFunction, name: str) -> tuple[FloatArray, FloatArray, float]:
    if not isinstance(phi, TransferTable) and not hasattr(phi, "as_arrays"):
        raise ValueError(f"{name} must be a TransferTable-like object for JAX solvers.")
    mu, rate = phi.as_arrays()
    rate_max = getattr(phi, "rate_max", None)
    return np.asarray(mu, dtype=np.float64), np.asarray(rate, dtype=np.float64), (
        float(rate_max) if rate_max is not None else float("inf")
    )


def _index_array(values: ArrayLike, name: str) -> NDArray[np.int64]:
    arr = np.asarray(values, dtype=np.int64).reshape(-1).copy()
    if arr.size and np.any(arr < 0):
        raise ValueError(f"{name} must contain non-negative indices.")
    if np.unique(arr).size != arr.size:
        raise ValueError(f"{name} must not contain duplicate indices.")
    arr.setflags(write=False)
    return arr


def _positive_float(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer.")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be positive.")
    return value
