"""BCM learning rule for excitatory efferent plasticity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from v1_research.learning.rules import LearningUpdate, RateBatch
from v1_research.model.state import ModelState
from v1_research.model.weights import as_connection_mask, as_dense_weights, limit_row_sums

ThetaUpdateOrder = Literal["pre", "post"]


@dataclass(frozen=True, slots=True)
class BCMConfig:
    """Local configuration for BCM plasticity math."""

    eta: float = 3.0e-5
    theta_beta: float = 0.01
    theta_eps: float = 1.0e-6
    theta_y0: float = 20.0
    theta_init: float | None = 1.0
    theta_floor: float | None = 1.0e-3
    theta_update_order: ThetaUpdateOrder = "pre"
    w_max: float | None = None
    row_sum_max_scale: float | None = 1.0


@dataclass(frozen=True, slots=True)
class BCMRowSumLimits:
    """Initial row-sum caps for BCM plastic blocks."""

    target_exc_source_exc: NDArray[np.float64] | None = None
    target_inh_source_exc: NDArray[np.float64] | None = None


@dataclass(frozen=True, slots=True)
class BCMState:
    """Sliding thresholds and row-sum caps for BCM plasticity."""

    theta_exc: NDArray[np.float64]
    theta_inh: NDArray[np.float64]
    row_sum_limits: BCMRowSumLimits = BCMRowSumLimits()

    def __post_init__(self) -> None:
        object.__setattr__(self, "theta_exc", _theta_vector("theta_exc", self.theta_exc))
        object.__setattr__(self, "theta_inh", _theta_vector("theta_inh", self.theta_inh))


class BCMLearningRule:
    """BCM rule that updates outgoing excitatory recurrent weights."""

    def __init__(self, cfg: BCMConfig = BCMConfig()) -> None:
        _check_config(cfg)
        self.cfg = cfg

    def initialize(self, model: ModelState, rates: RateBatch) -> BCMState:
        """Initializes BCM thresholds from the first response batch."""

        _check_rate_shapes(model, rates)
        row_sum_limits = initial_row_sum_limits(model, self.cfg)
        if self.cfg.theta_init is None:
            return BCMState(
                theta_exc=np.maximum(mean_squared_response(rates.exc) / float(self.cfg.theta_y0), float(self.cfg.theta_eps)),
                theta_inh=np.maximum(mean_squared_response(rates.inh) / float(self.cfg.theta_y0), float(self.cfg.theta_eps)),
                row_sum_limits=row_sum_limits,
            )
        theta_init = float(self.cfg.theta_init)
        return BCMState(
            theta_exc=np.full(model.layout.n_exc, theta_init, dtype=float),
            theta_inh=np.full(model.layout.n_inh, theta_init, dtype=float),
            row_sum_limits=row_sum_limits,
        )

    def step(self, model: ModelState, rates: RateBatch, state: BCMState) -> LearningUpdate[BCMState]:
        """Applies one BCM update to ``E<-E`` and ``I<-E`` weights."""

        _check_rate_shapes(model, rates)
        _check_state_shapes(model, state)
        theta_for_update = state
        next_state = state
        if self.cfg.theta_update_order == "pre":
            next_state = update_theta(state, rates, self.cfg)
            theta_for_update = next_state

        next_weights = update_excitatory_efferents(
            model=model,
            rates=rates,
            theta=theta_for_update,
            cfg=self.cfg,
            row_sum_limits=state.row_sum_limits,
        )

        if self.cfg.theta_update_order == "post":
            next_state = update_theta(state, rates, self.cfg)

        return LearningUpdate(
            model=ModelState(layout=model.layout, connection_mask=model.connection_mask, weights=next_weights),
            state=next_state,
            updated=True,
        )

    def stats(self, model: ModelState, state: BCMState) -> dict[str, float]:
        """Returns compact theta and plastic-weight diagnostics."""

        _check_state_shapes(model, state)
        weights = as_dense_weights(model.weights)
        exc = model.layout.exc_idx
        inh = model.layout.inh_idx
        return {
            "theta_exc_median": _median_or_zero(state.theta_exc),
            "theta_inh_median": _median_or_zero(state.theta_inh),
            "W_EE_mean": _nonzero_mean(weights[np.ix_(exc, exc)]),
            "W_IE_mean": _nonzero_mean(weights[np.ix_(inh, exc)]),
        }


def update_theta(state: BCMState, rates: RateBatch, cfg: BCMConfig) -> BCMState:
    """Updates BCM sliding thresholds from one response batch."""

    return BCMState(
        theta_exc=update_theta_vector(state.theta_exc, rates.exc, cfg),
        theta_inh=update_theta_vector(state.theta_inh, rates.inh, cfg),
        row_sum_limits=state.row_sum_limits,
    )


def update_theta_vector(theta: ArrayLike, response: ArrayLike, cfg: BCMConfig) -> NDArray[np.float64]:
    """Updates one population threshold vector."""

    theta_arr = _theta_vector("theta", theta)
    response_ms = mean_squared_response(response) / float(cfg.theta_y0)
    if theta_arr.shape != response_ms.shape:
        raise ValueError(f"theta shape {theta_arr.shape} does not match response width {response_ms.shape}.")
    updated = (1.0 - float(cfg.theta_beta)) * theta_arr + float(cfg.theta_beta) * response_ms
    return np.maximum(updated, _theta_floor(cfg))


def mean_squared_response(response: ArrayLike) -> NDArray[np.float64]:
    """Returns per-neuron mean squared response for a batch-first matrix."""

    arr = _batch_matrix("response", response)
    return np.mean(arr**2, axis=0)


def bcm_gain(response: ArrayLike, theta: ArrayLike, cfg: BCMConfig) -> NDArray[np.float64]:
    """Calculates ``eta * y * (y - theta)`` for a batch-first response matrix."""

    y = _batch_matrix("response", response)
    theta_arr = np.maximum(_theta_vector("theta", theta), float(cfg.theta_eps))
    if y.shape[1] != theta_arr.size:
        raise ValueError(f"response width {y.shape[1]} does not match theta width {theta_arr.size}.")
    return float(cfg.eta) * y * (y - theta_arr[np.newaxis, :])


def bcm_delta(x: ArrayLike, y: ArrayLike, theta: ArrayLike, cfg: BCMConfig) -> NDArray[np.float64]:
    """Calculates a target-by-source BCM weight delta matrix."""

    x_arr = _batch_matrix("x", x)
    gain = bcm_gain(y, theta, cfg)
    if x_arr.shape[0] != gain.shape[0]:
        raise ValueError("x and y batches must have the same leading dimension.")
    return gain.T @ x_arr / x_arr.shape[0]


def update_excitatory_efferents(
    *,
    model: ModelState,
    rates: RateBatch,
    theta: BCMState,
    cfg: BCMConfig,
    row_sum_limits: BCMRowSumLimits | None = None,
) -> NDArray[np.float64]:
    """Updates ``E<-E`` and ``I<-E`` plastic blocks and leaves all others unchanged."""

    weights = as_dense_weights(model.weights)
    topology = as_connection_mask(model.connection_mask, weights.shape)
    updated = weights.copy()
    exc = model.layout.exc_idx
    inh = model.layout.inh_idx
    limits = BCMRowSumLimits() if row_sum_limits is None else row_sum_limits

    updated[np.ix_(exc, exc)] = update_excitatory_block(
        weights=weights[np.ix_(exc, exc)],
        connection_mask=topology[np.ix_(exc, exc)],
        x=rates.exc,
        y=rates.exc,
        theta=theta.theta_exc,
        cfg=cfg,
        row_sum_max=limits.target_exc_source_exc,
    )
    updated[np.ix_(inh, exc)] = update_excitatory_block(
        weights=weights[np.ix_(inh, exc)],
        connection_mask=topology[np.ix_(inh, exc)],
        x=rates.exc,
        y=rates.inh,
        theta=theta.theta_inh,
        cfg=cfg,
        row_sum_max=limits.target_inh_source_exc,
    )
    updated[~topology] = 0.0
    return updated


def update_excitatory_block(
    *,
    weights: ArrayLike,
    connection_mask: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    theta: ArrayLike,
    cfg: BCMConfig,
    row_sum_max: float | ArrayLike | None,
) -> NDArray[np.float64]:
    """Applies BCM update and row-sum cap to one positive excitatory-source block."""

    weights_arr = as_dense_weights(weights)
    topology = np.asarray(connection_mask, dtype=bool)
    if topology.shape != weights_arr.shape:
        raise ValueError(f"connection_mask shape {topology.shape} does not match weights shape {weights_arr.shape}.")
    delta = bcm_delta(x, y, theta, cfg)
    if delta.shape != weights_arr.shape:
        raise ValueError(f"BCM delta shape {delta.shape} does not match weights shape {weights_arr.shape}.")
    updated = np.maximum(weights_arr + topology.astype(float) * delta, 0.0)
    if cfg.w_max is not None:
        updated = np.minimum(updated, float(cfg.w_max))
    updated *= topology
    return limit_row_sums(updated, row_sum_max)


def initial_row_sum_limits(model: ModelState, cfg: BCMConfig) -> BCMRowSumLimits:
    """Computes row-sum caps from initial plastic-block weights."""

    if cfg.row_sum_max_scale is None:
        return BCMRowSumLimits()
    weights = as_dense_weights(model.weights)
    scale = float(cfg.row_sum_max_scale)
    exc = model.layout.exc_idx
    inh = model.layout.inh_idx
    return BCMRowSumLimits(
        target_exc_source_exc=np.sum(weights[np.ix_(exc, exc)], axis=1) * scale,
        target_inh_source_exc=np.sum(weights[np.ix_(inh, exc)], axis=1) * scale,
    )


def _check_config(cfg: BCMConfig) -> None:
    _finite_nonnegative(cfg.eta, "eta")
    beta = _finite_positive(cfg.theta_beta, "theta_beta")
    if beta > 1.0:
        raise ValueError("theta_beta must be in (0.0, 1.0].")
    _finite_positive(cfg.theta_eps, "theta_eps")
    _finite_positive(cfg.theta_y0, "theta_y0")
    if cfg.theta_update_order not in {"pre", "post"}:
        raise ValueError("theta_update_order must be 'pre' or 'post'.")
    if cfg.theta_init is not None:
        _finite_positive(cfg.theta_init, "theta_init")
    if cfg.theta_floor is not None:
        _finite_positive(cfg.theta_floor, "theta_floor")
    if cfg.w_max is not None:
        _finite_positive(cfg.w_max, "w_max")
    if cfg.row_sum_max_scale is not None:
        _finite_nonnegative(cfg.row_sum_max_scale, "row_sum_max_scale")


def _check_rate_shapes(model: ModelState, rates: RateBatch) -> None:
    if rates.exc.shape[1] != model.layout.n_exc:
        raise ValueError(f"exc rates width {rates.exc.shape[1]} does not match {model.layout.n_exc}.")
    if rates.inh.shape[1] != model.layout.n_inh:
        raise ValueError(f"inh rates width {rates.inh.shape[1]} does not match {model.layout.n_inh}.")
    if rates.external is not None and rates.external.shape[1] != model.layout.n_input:
        raise ValueError(f"external rates width {rates.external.shape[1]} does not match {model.layout.n_input}.")


def _check_state_shapes(model: ModelState, state: BCMState) -> None:
    if state.theta_exc.shape != (model.layout.n_exc,):
        raise ValueError("theta_exc shape must match the number of excitatory cells.")
    if state.theta_inh.shape != (model.layout.n_inh,):
        raise ValueError("theta_inh shape must match the number of inhibitory cells.")


def _theta_floor(cfg: BCMConfig) -> float:
    floor = float(cfg.theta_eps)
    if cfg.theta_floor is not None:
        floor = max(floor, float(cfg.theta_floor))
    return floor


def _theta_vector(name: str, values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float).reshape(-1).copy()
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _batch_matrix(name: str, values: ArrayLike) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2D batch matrix.")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} batch must not be empty.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    return arr


def _finite_positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be positive and finite.")
    return value


def _finite_nonnegative(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be non-negative and finite.")
    return value


def _median_or_zero(values: NDArray[np.float64]) -> float:
    return 0.0 if values.size == 0 else float(np.median(values))


def _nonzero_mean(values: NDArray[np.float64]) -> float:
    nonzero = np.asarray(values, dtype=float).ravel()
    nonzero = nonzero[nonzero != 0.0]
    return 0.0 if nonzero.size == 0 else float(np.mean(nonzero))
