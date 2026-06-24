"""Shared solver interface for V1 rate dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np

from v1_research.dynamics.transfer import TransferConfig, TransferTables, build_transfer_tables
from v1_research.dynamics.wilson_cowan import ExternalDrive, FloatArray, RateLayout, TransferFunction
from v1_research.inputs.background import BackgroundTrace, validate_time_grid
from v1_research.model.state import ModelState

BackendName = Literal["scipy", "jax-rk4"]


@dataclass(frozen=True, slots=True)
class SolverConfig:
    """Local solver configuration for Wilson-Cowan rate dynamics."""

    backend: BackendName = "jax-rk4"
    scipy_method: str = "RK4"
    store_trajectory: bool = True
    jax_dtype: str = "float64"
    transfer: TransferConfig = field(default_factory=TransferConfig)


@dataclass(frozen=True, slots=True)
class RateResult:
    """Batched E/I rate result."""

    exc: FloatArray
    inh: FloatArray
    exc_trajectory: FloatArray | None
    inh_trajectory: FloatArray | None
    time: FloatArray


class RateSolver(Protocol):
    """Solver backend protocol."""

    def solve(
        self,
        model: ModelState,
        *,
        drive: ExternalDrive,
        time: FloatArray,
        n_batch: int,
        phi_exc: TransferFunction,
        phi_inh: TransferFunction,
        tau_exc: float,
        tau_inh: float,
        background_trace: BackgroundTrace | None = None,
        store_trajectory: bool = True,
    ) -> RateResult:
        ...


def make_solver(cfg: SolverConfig) -> RateSolver:
    """Creates the selected solver backend."""

    if cfg.backend == "scipy":
        from v1_research.dynamics.scipy_solver import ScipySolver

        return ScipySolver(method=cfg.scipy_method)
    if cfg.backend == "jax-rk4":
        from v1_research.dynamics.jax_rk4 import JaxRK4Solver

        return JaxRK4Solver(dtype=cfg.jax_dtype)
    raise ValueError(f"Unknown solver backend: {cfg.backend!r}")


def solve_rates(
    model: ModelState,
    *,
    drive: ExternalDrive,
    time: FloatArray,
    n_batch: int,
    cfg: SolverConfig,
    background_trace: BackgroundTrace | None = None,
    transfer_tables: TransferTables | None = None,
    phi_exc: TransferFunction | None = None,
    phi_inh: TransferFunction | None = None,
) -> RateResult:
    """Solves Wilson-Cowan dynamics using a local solver config."""

    phi_e, phi_i = _resolve_transfer(cfg, transfer_tables=transfer_tables, phi_exc=phi_exc, phi_inh=phi_inh)
    solver = make_solver(cfg)
    return solver.solve(
        model,
        drive=drive,
        time=validate_time_grid(np.asarray(time, dtype=np.float64), copy=True),
        n_batch=n_batch,
        phi_exc=phi_e,
        phi_inh=phi_i,
        tau_exc=cfg.transfer.tau_exc,
        tau_inh=cfg.transfer.tau_inh,
        background_trace=background_trace,
        store_trajectory=cfg.store_trajectory,
    )


def pack_rate_result(
    trajectory: FloatArray,
    layout: RateLayout,
    time: FloatArray,
    *,
    store_trajectory: bool,
) -> RateResult:
    """Packs ``(n_time, n_rates, n_batch)`` trajectory into E/I public shapes."""

    y_t = np.asarray(trajectory, dtype=np.float64)
    if y_t.ndim != 3:
        raise ValueError("trajectory must have shape (n_time, n_rates, n_batch).")
    if y_t.shape[0] != time.size:
        raise ValueError("trajectory time dimension must match time.")
    if y_t.shape[1] != layout.n_rates:
        raise ValueError("trajectory rate dimension must match layout.n_rates.")

    exc_t = np.transpose(y_t[:, layout.exc_idx, :], (0, 2, 1))
    inh_t = np.transpose(y_t[:, layout.inh_idx, :], (0, 2, 1))
    tail_start = int(time.size * 2 / 3)
    exc = np.mean(exc_t[tail_start:], axis=0)
    inh = np.mean(inh_t[tail_start:], axis=0)
    return RateResult(
        exc=exc,
        inh=inh,
        exc_trajectory=exc_t if store_trajectory else None,
        inh_trajectory=inh_t if store_trajectory else None,
        time=validate_time_grid(time, copy=True),
    )


def _resolve_transfer(
    cfg: SolverConfig,
    *,
    transfer_tables: TransferTables | None,
    phi_exc: TransferFunction | None,
    phi_inh: TransferFunction | None,
) -> tuple[TransferFunction, TransferFunction]:
    if phi_exc is not None and phi_inh is not None:
        return phi_exc, phi_inh
    if (phi_exc is None) != (phi_inh is None):
        raise ValueError("phi_exc and phi_inh must be provided together.")
    tables = build_transfer_tables(cfg.transfer) if transfer_tables is None else transfer_tables
    return tables.excitatory, tables.inhibitory
