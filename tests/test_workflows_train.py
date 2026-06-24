from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse

from v1_research.dynamics import RateResult, SolverConfig
from v1_research.learning import LearningUpdate, RateBatch
from v1_research.model import ModelState, PopulationLayout, SheetGeometry
from v1_research.workflows.train import apply_learning_rule, solve_and_learn_batch


def _small_model(weight: float = 0.2) -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = np.array([[0.0, weight]], dtype=float)
    return ModelState(layout=layout, connection_mask=sparse.csr_matrix(weights != 0.0), weights=weights)


@dataclass(frozen=True, slots=True)
class FakeState:
    calls: int


class FakeLearningRule:
    def initialize(self, model: ModelState, rates: RateBatch) -> FakeState:
        assert rates.exc.shape == (2, 1)
        return FakeState(calls=1)

    def step(self, model: ModelState, rates: RateBatch, state: FakeState) -> LearningUpdate[FakeState]:
        next_model = _small_model(weight=float(model.weights[0, 1]) + float(np.mean(rates.exc)))
        return LearningUpdate(model=next_model, state=FakeState(calls=state.calls + 1), updated=True)

    def stats(self, model: ModelState, state: FakeState) -> dict[str, float]:
        return {"calls": float(state.calls)}


def test_apply_learning_rule_initializes_without_weight_update() -> None:
    model = _small_model()
    rates = RateBatch(exc=np.array([[1.0], [2.0]]), inh=np.empty((2, 0)))

    update = apply_learning_rule(model, rates, FakeLearningRule())

    assert update.model is model
    assert update.state == FakeState(calls=1)
    assert update.updated is False


def test_apply_learning_rule_uses_protocol_step_after_initialization() -> None:
    model = _small_model()
    rates = RateBatch(exc=np.array([[1.0], [2.0]]), inh=np.empty((2, 0)))

    update = apply_learning_rule(model, rates, FakeLearningRule(), state=FakeState(calls=1))

    assert float(update.model.weights[0, 1]) == 1.7
    assert update.state == FakeState(calls=2)
    assert update.updated is True


def test_solve_and_learn_batch_builds_rate_batch_from_solver_result() -> None:
    model = _small_model()
    solver_result = RateResult(
        exc=np.array([[3.0], [5.0]]),
        inh=np.empty((2, 0)),
        exc_trajectory=None,
        inh_trajectory=None,
        time=np.array([0.0, 1.0]),
    )

    def solver(*args, **kwargs) -> RateResult:
        return solver_result

    def drive(_t: float) -> np.ndarray:
        return np.array([[9.0, 11.0]], dtype=float)

    update = solve_and_learn_batch(
        model,
        drive=drive,
        time=np.array([0.0, 1.0]),
        n_batch=2,
        solver_cfg=SolverConfig(backend="scipy"),
        rule=FakeLearningRule(),
        state=FakeState(calls=2),
        solver=solver,
    )

    assert float(update.model.weights[0, 1]) == 4.2
    assert update.state == FakeState(calls=3)
    assert update.updated is True
