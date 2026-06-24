from __future__ import annotations

import numpy as np
from scipy import sparse

from v1_research.learning import (
    BCMLearningRule,
    BCMConfig,
    BCMState,
    LearningConfig,
    RateBatch,
    make_learning_rule,
)
from v1_research.model import ModelState, PopulationLayout, SheetGeometry


def _small_model(*, weight_scale: float = 1.0) -> ModelState:
    layout = PopulationLayout(
        l23=SheetGeometry(n_side=2, region_size=1.0, z_pos=0.0),
        l4=SheetGeometry(n_side=1, region_size=1.0, z_pos=0.1),
        l23_cell_types=np.array(["E", "I", "E", "I"]),
        l4_tuning_labels=np.array(["T"]),
        l4_preferred_orientations=np.array([0.0]),
    )
    weights = weight_scale * np.array(
        [
            [0.2, -0.4, 0.1, 0.0, 0.8],
            [0.3, -0.5, 0.2, -0.1, 0.7],
            [0.0, -0.2, 0.4, -0.3, 0.6],
            [0.5, 0.0, 0.1, -0.4, 0.5],
        ],
        dtype=float,
    )
    mask = weights != 0.0
    return ModelState(layout=layout, connection_mask=sparse.csr_matrix(mask), weights=sparse.csr_matrix(weights))


def _rates() -> RateBatch:
    return RateBatch(
        exc=np.array([[2.0, 1.0], [4.0, 3.0]], dtype=float),
        inh=np.array([[1.5, 2.5], [3.5, 4.5]], dtype=float),
        external=np.array([[7.0], [8.0]], dtype=float),
    )


def test_bcm_initialize_uses_batch_mean_squared_response() -> None:
    rule = BCMLearningRule(BCMConfig(theta_init=None, theta_y0=2.0, theta_eps=0.1))

    state = rule.initialize(_small_model(), _rates())

    np.testing.assert_allclose(state.theta_exc, np.array([5.0, 2.5]))
    np.testing.assert_allclose(state.theta_inh, np.array([3.625, 6.625]))


def test_bcm_pre_update_uses_updated_theta_for_weight_delta() -> None:
    model = _small_model()
    cfg = BCMConfig(
        eta=0.1,
        theta_beta=1.0,
        theta_y0=1.0,
        theta_init=1.0,
        theta_update_order="pre",
        row_sum_max_scale=None,
    )
    rule = BCMLearningRule(cfg)
    state = BCMState(theta_exc=np.array([1.0, 1.0]), theta_inh=np.array([1.0, 1.0]))

    update = rule.step(model, _rates(), state)

    weights = np.asarray(update.model.weights)
    np.testing.assert_allclose(update.state.theta_exc, np.array([10.0, 5.0]))
    np.testing.assert_allclose(update.state.theta_inh, np.array([7.25, 13.25]))
    assert weights[0, 0] == 0.0
    assert weights[1, 0] == 0.0
    np.testing.assert_allclose(weights[:, [1, 3, 4]], model.weights.toarray()[:, [1, 3, 4]])
    assert weights[0, 4] == model.weights.toarray()[0, 4]


def test_bcm_post_update_uses_previous_theta_for_weight_delta() -> None:
    model = _small_model()
    rule = BCMLearningRule(
        BCMConfig(
            eta=0.1,
            theta_beta=1.0,
            theta_y0=1.0,
            theta_init=1.0,
            theta_update_order="post",
            row_sum_max_scale=None,
        )
    )
    state = BCMState(theta_exc=np.array([1.0, 1.0]), theta_inh=np.array([1.0, 1.0]))

    update = rule.step(model, _rates(), state)

    weights = np.asarray(update.model.weights)
    assert weights[0, 0] > model.weights.toarray()[0, 0]
    assert weights[1, 0] > model.weights.toarray()[1, 0]
    np.testing.assert_allclose(update.state.theta_exc, np.array([10.0, 5.0]))
    np.testing.assert_allclose(update.state.theta_inh, np.array([7.25, 13.25]))


def test_bcm_caps_weights_and_row_sums_on_plastic_blocks() -> None:
    model = _small_model(weight_scale=2.0)
    rule = BCMLearningRule(
        BCMConfig(
            eta=0.5,
            theta_init=0.1,
            theta_update_order="post",
            w_max=0.75,
            row_sum_max_scale=1.0,
        )
    )
    state = rule.initialize(model, _rates())

    update = rule.step(model, _rates(), state)

    initial = model.weights.toarray()
    weights = np.asarray(update.model.weights)
    exc = model.layout.exc_idx
    inh = model.layout.inh_idx
    ee_initial_rows = np.sum(initial[np.ix_(exc, exc)], axis=1)
    ie_initial_rows = np.sum(initial[np.ix_(inh, exc)], axis=1)
    assert np.max(weights[np.ix_(exc, exc)]) <= 0.75
    assert np.max(weights[np.ix_(inh, exc)]) <= 0.75
    np.testing.assert_array_less(np.sum(weights[np.ix_(exc, exc)], axis=1), ee_initial_rows + 1.0e-12)
    np.testing.assert_array_less(np.sum(weights[np.ix_(inh, exc)], axis=1), ie_initial_rows + 1.0e-12)


def test_bcm_state_carries_row_sum_limits_between_models() -> None:
    rule = BCMLearningRule(BCMConfig(theta_init=0.1, theta_update_order="post", row_sum_max_scale=1.0))
    first_model = _small_model(weight_scale=0.5)
    second_model = _small_model(weight_scale=3.0)
    first_state = rule.initialize(first_model, _rates())
    second_state = rule.initialize(second_model, _rates())

    first_update = rule.step(first_model, _rates(), first_state)
    second_update = rule.step(second_model, _rates(), second_state)

    exc = first_model.layout.exc_idx
    first_initial = first_model.weights.toarray()
    second_initial = second_model.weights.toarray()
    np.testing.assert_allclose(
        np.sum(np.asarray(first_update.model.weights)[np.ix_(exc, exc)], axis=1),
        np.sum(first_initial[np.ix_(exc, exc)], axis=1),
    )
    np.testing.assert_allclose(
        np.sum(np.asarray(second_update.model.weights)[np.ix_(exc, exc)], axis=1),
        np.sum(second_initial[np.ix_(exc, exc)], axis=1),
    )


def test_learning_factory_builds_bcm_rule() -> None:
    rule = make_learning_rule(LearningConfig(kind="bcm", bcm=BCMConfig(eta=0.2)))

    assert isinstance(rule, BCMLearningRule)
