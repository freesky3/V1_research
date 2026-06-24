"""SciPy solver backend for Wilson-Cowan dynamics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from v1_research.dynamics.wilson_cowan import ExternalDrive, FloatArray, WilsonCowanEquation
from v1_research.inputs.background import BackgroundTrace


class ScipySolver:
    """Debug-friendly SciPy solver for Wilson-Cowan dynamics."""

    def __init__(self, *, method: str = "RK4") -> None:
        self.method = str(method)

    def integrate(
        self,
        equation: WilsonCowanEquation,
        *,
        drive: ExternalDrive,
        time: FloatArray,
        background_trace: BackgroundTrace | None = None,
    ) -> FloatArray:
        """Returns rate trajectory with shape ``(n_time, n_rates, n_batch)``."""

        if time.size < 2:
            raise ValueError("solver time grid must contain at least two points.")
        equation.validate_background_trace(background_trace, time)
        if self.method == "RK4":
            return _fixed_rk4(equation, drive=drive, time=time, background_trace=background_trace)
        return _solve_ivp(equation, drive=drive, time=time, background_trace=background_trace, method=self.method)

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

        time_grid = _time_grid(time)
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


def _fixed_rk4(
    equation: WilsonCowanEquation,
    *,
    drive: ExternalDrive,
    time: FloatArray,
    background_trace: BackgroundTrace | None,
) -> FloatArray:
    y = np.zeros((equation.layout.n_rates, equation.n_batch), dtype=np.float64)
    trajectory = np.empty((time.size, equation.layout.n_rates, equation.n_batch), dtype=np.float64)
    trajectory[0] = y
    samples = background_trace.rk4_samples() if background_trace is not None else None

    for step, (t0, t1) in enumerate(zip(time[:-1], time[1:]), start=1):
        dt = float(t1 - t0)
        left = _background_stage(samples, "left", step - 1)
        mid = _background_stage(samples, "mid", step - 1)
        right = _background_stage(samples, "right", step - 1)
        k1 = equation.rhs(y, drive(float(t0)), background=left)
        k2 = equation.rhs(y + 0.5 * dt * k1, drive(float(t0 + 0.5 * dt)), background=mid)
        k3 = equation.rhs(y + 0.5 * dt * k2, drive(float(t0 + 0.5 * dt)), background=mid)
        k4 = equation.rhs(y + dt * k3, drive(float(t1)), background=right)
        y = y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step] = y
    return trajectory


def _solve_ivp(
    equation: WilsonCowanEquation,
    *,
    drive: ExternalDrive,
    time: FloatArray,
    background_trace: BackgroundTrace | None,
    method: str,
) -> FloatArray:
    y0 = np.zeros((equation.layout.n_rates, equation.n_batch), dtype=np.float64).ravel()

    def rhs(t: float, y_flat: FloatArray) -> FloatArray:
        background = None if background_trace is None else background_trace.value_at(float(t))
        return equation.rhs_flat(float(t), y_flat, drive, background=background)

    sol = solve_ivp(rhs, (float(time[0]), float(time[-1])), y0, method=method, t_eval=time)
    if not sol.success:
        raise RuntimeError(f"SciPy solver failed: {sol.message}")
    return np.asarray(sol.y.T, dtype=np.float64).reshape(sol.t.size, equation.layout.n_rates, equation.n_batch)


def _background_stage(samples, stage: str, index: int):
    if samples is None:
        return None
    if stage == "left":
        return samples.exc_left[index], samples.inh_left[index]
    if stage == "mid":
        return samples.exc_mid[index], samples.inh_mid[index]
    if stage == "right":
        return samples.exc_right[index], samples.inh_right[index]
    raise ValueError(f"Unsupported RK4 background stage: {stage!r}")


def _time_grid(time) -> FloatArray:
    from v1_research.inputs.background import validate_time_grid

    return validate_time_grid(np.asarray(time, dtype=np.float64), copy=True)
