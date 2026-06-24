"""JAX RK4 solver backend for Wilson-Cowan dynamics."""

from __future__ import annotations

import importlib.util

import numpy as np

from v1_research.dynamics.wilson_cowan import ExternalDrive, FloatArray, WilsonCowanEquation, jax_wilson_cowan_rhs
from v1_research.inputs.background import BackgroundTrace


class JaxRK4Solver:
    """JIT-compiled RK4 path for Wilson-Cowan dynamics."""

    def __init__(self, *, dtype: str = "float64") -> None:
        self.dtype = str(dtype)

    def integrate(
        self,
        equation: WilsonCowanEquation,
        *,
        drive: ExternalDrive,
        time: FloatArray,
        background_trace: BackgroundTrace | None = None,
    ) -> FloatArray:
        """Returns rate trajectory with shape ``(n_time, n_rates, n_batch)``."""

        jax, jnp = _require_jax()
        if time.size < 2:
            raise ValueError("solver time grid must contain at least two points.")
        equation.validate_background_trace(background_trace, time)

        dtype = jnp.float32 if self.dtype == "float32" else jnp.float64
        blocks = equation.weight_blocks()
        ax_left, ax_mid, ax_right = _precompute_rk4_drive(drive, time, equation=equation)
        bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i = _precompute_rk4_background(
            background_trace,
            time=time,
            equation=equation,
        )
        exc_mu, exc_rate, exc_rate_max, inh_mu, inh_rate, inh_rate_max = equation.transfer_table_arrays()
        run = _compiled_rk4(jax, jnp)
        y_all = run(
            jnp.zeros((equation.layout.n_rates, equation.n_batch), dtype=dtype),
            jnp.asarray(blocks.exc, dtype=dtype),
            jnp.asarray(blocks.inh, dtype=dtype),
            jnp.asarray(blocks.external, dtype=dtype),
            jnp.asarray(equation.layout.exc_idx, dtype=jnp.int32),
            jnp.asarray(equation.layout.inh_idx, dtype=jnp.int32),
            jnp.asarray(time, dtype=dtype),
            jnp.asarray(ax_left, dtype=dtype),
            jnp.asarray(ax_mid, dtype=dtype),
            jnp.asarray(ax_right, dtype=dtype),
            jnp.asarray(bg_left_e, dtype=dtype),
            jnp.asarray(bg_mid_e, dtype=dtype),
            jnp.asarray(bg_right_e, dtype=dtype),
            jnp.asarray(bg_left_i, dtype=dtype),
            jnp.asarray(bg_mid_i, dtype=dtype),
            jnp.asarray(bg_right_i, dtype=dtype),
            jnp.asarray(exc_mu, dtype=dtype),
            jnp.asarray(exc_rate, dtype=dtype),
            jnp.asarray(exc_rate_max, dtype=dtype),
            jnp.asarray(inh_mu, dtype=dtype),
            jnp.asarray(inh_rate, dtype=dtype),
            jnp.asarray(inh_rate_max, dtype=dtype),
            jnp.asarray(equation.tau_exc, dtype=dtype),
            jnp.asarray(equation.tau_inh, dtype=dtype),
        )
        jax.block_until_ready(y_all)
        return np.asarray(y_all, dtype=np.float64)

    def solve(
        self,
        model,
        *,
        drive,
        time,
        n_batch,
        phi_exc,
        phi_inh,
        tau_exc=0.02,
        tau_inh=0.01,
        background_trace=None,
        store_trajectory=True,
    ):
        """Builds an equation and solves it."""

        from v1_research.dynamics.solvers import pack_rate_result
        from v1_research.inputs.background import validate_time_grid

        time_grid = validate_time_grid(np.asarray(time, dtype=np.float64), copy=True)
        equation = WilsonCowanEquation(
            model,
            phi_exc=phi_exc,
            phi_inh=phi_inh,
            tau_exc=tau_exc,
            tau_inh=tau_inh,
            n_batch=n_batch,
        )
        trajectory = self.integrate(equation, drive=drive, time=time_grid, background_trace=background_trace)
        return pack_rate_result(trajectory, equation.layout, time_grid, store_trajectory=store_trajectory)


_RUN_CACHE = {}


def _compiled_rk4(jax, jnp):
    if "rk4" in _RUN_CACHE:
        return _RUN_CACHE["rk4"]
    wc_rhs = jax_wilson_cowan_rhs(jnp)

    def run(
        y0,
        weights_exc,
        weights_inh,
        weights_ext,
        idx_exc,
        idx_inh,
        time,
        ax_left,
        ax_mid,
        ax_right,
        bg_left_e,
        bg_mid_e,
        bg_right_e,
        bg_left_i,
        bg_mid_i,
        bg_right_i,
        phi_exc_mu,
        phi_exc_rate,
        phi_exc_rate_max,
        phi_inh_mu,
        phi_inh_rate,
        phi_inh_rate_max,
        tau_exc,
        tau_inh,
    ):
        params = (
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
        )
        dts = time[1:] - time[:-1]

        def scan_step(y, xs):
            dt, ax_l, ax_m, ax_r, bg_l_e, bg_m_e, bg_r_e, bg_l_i, bg_m_i, bg_r_i = xs
            k1 = wc_rhs(y, ax_l, bg_l_e, bg_l_i, *params)
            k2 = wc_rhs(y + 0.5 * dt * k1, ax_m, bg_m_e, bg_m_i, *params)
            k3 = wc_rhs(y + 0.5 * dt * k2, ax_m, bg_m_e, bg_m_i, *params)
            k4 = wc_rhs(y + dt * k3, ax_r, bg_r_e, bg_r_i, *params)
            y_next = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            return y_next, y_next

        _, ys = jax.lax.scan(
            scan_step,
            y0,
            (dts, ax_left, ax_mid, ax_right, bg_left_e, bg_mid_e, bg_right_e, bg_left_i, bg_mid_i, bg_right_i),
        )
        return jnp.concatenate([y0[jnp.newaxis, :, :], ys], axis=0)

    _RUN_CACHE["rk4"] = jax.jit(run)
    return _RUN_CACHE["rk4"]


def _precompute_rk4_drive(
    drive: ExternalDrive,
    time: FloatArray,
    *,
    equation: WilsonCowanEquation,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    left = []
    mid = []
    right = []
    for t0, t1 in zip(time[:-1], time[1:]):
        dt = float(t1 - t0)
        left.append(equation.validate_external_drive_value(drive(float(t0))))
        mid.append(equation.validate_external_drive_value(drive(float(t0 + 0.5 * dt))))
        right.append(equation.validate_external_drive_value(drive(float(t1))))
    return np.stack(left), np.stack(mid), np.stack(right)


def _precompute_rk4_background(
    trace: BackgroundTrace | None,
    *,
    time: FloatArray,
    equation: WilsonCowanEquation,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    if trace is None:
        n_steps = time.size - 1
        return (
            np.zeros((n_steps, equation.layout.n_exc, equation.n_batch), dtype=np.float64),
            np.zeros((n_steps, equation.layout.n_exc, equation.n_batch), dtype=np.float64),
            np.zeros((n_steps, equation.layout.n_exc, equation.n_batch), dtype=np.float64),
            np.zeros((n_steps, equation.layout.n_inh, equation.n_batch), dtype=np.float64),
            np.zeros((n_steps, equation.layout.n_inh, equation.n_batch), dtype=np.float64),
            np.zeros((n_steps, equation.layout.n_inh, equation.n_batch), dtype=np.float64),
        )
    samples = trace.rk4_samples()
    return (
        np.transpose(samples.exc_left, (0, 2, 1)),
        np.transpose(samples.exc_mid, (0, 2, 1)),
        np.transpose(samples.exc_right, (0, 2, 1)),
        np.transpose(samples.inh_left, (0, 2, 1)),
        np.transpose(samples.inh_mid, (0, 2, 1)),
        np.transpose(samples.inh_right, (0, 2, 1)),
    )


def _require_jax():
    if importlib.util.find_spec("jax") is None:
        raise RuntimeError("JaxRK4Solver requires installing the optional JAX dependencies.")
    import jax
    import jax.numpy as jnp

    return jax, jnp
