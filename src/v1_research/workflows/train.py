"""Thin training helpers that depend only on the learning-rule protocol."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from v1_research.dynamics import ExternalDrive, RateResult, SolverConfig, solve_rates
from v1_research.inputs.background import BackgroundTrace
from v1_research.learning import LearningRule, LearningUpdate, RateBatch
from v1_research.model.state import ModelState

SolverCallable = Callable[..., RateResult]


def apply_learning_rule(
    model: ModelState,
    rates: RateBatch,
    rule: LearningRule,
    state=None,
) -> LearningUpdate:
    """Initializes or applies one learning-rule step."""

    if state is None:
        return LearningUpdate(model=model, state=rule.initialize(model, rates), updated=False)
    return rule.step(model, rates, state)


def solve_and_learn_batch(
    model: ModelState,
    *,
    drive: ExternalDrive,
    time: Sequence[float] | np.ndarray,
    n_batch: int,
    solver_cfg: SolverConfig,
    rule: LearningRule,
    state=None,
    background_trace: BackgroundTrace | None = None,
    solver: SolverCallable | None = None,
) -> LearningUpdate:
    """Solves dynamics for one batch and applies the selected learning rule."""

    solver_fn = solve_rates if solver is None else solver
    result = solver_fn(
        model,
        drive=drive,
        time=np.asarray(time, dtype=float),
        n_batch=n_batch,
        cfg=solver_cfg,
        background_trace=background_trace,
    )
    external = np.asarray(drive(float(np.asarray(time, dtype=float)[0])), dtype=float)
    if external.ndim == 1:
        external = external[:, np.newaxis]
    rates = RateBatch(exc=result.exc, inh=result.inh, external=external.T)
    return apply_learning_rule(model, rates, rule, state=state)
