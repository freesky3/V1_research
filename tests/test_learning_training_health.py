from __future__ import annotations

import numpy as np
import pytest

from v1_research.learning import BCMConfig, BCMState, RateBatch
from v1_research.learning.diagnostics import (
    TrainingHealthConfig,
    bcm_signal_stats,
    evaluate_training_health,
    extended_active_rate_stats,
    extended_plastic_weight_stats,
    theta_distribution_stats,
)


def test_extended_active_rate_stats_reports_concentration_and_empty_population() -> None:
    rates = RateBatch(
        exc=np.array([[0.0, 8.0, 2.0], [0.0, 0.0, 2.0]], dtype=float),
        inh=np.empty((2, 0), dtype=float),
        external=np.array([[1.0, 3.0], [2.0, 4.0]], dtype=float),
    )

    stats = extended_active_rate_stats(rates, active_threshold=1.0, near_rate_cap=7.0)

    assert stats["exc_active_neuron_fraction"] == pytest.approx(2.0 / 3.0)
    assert stats["exc_silent_neuron_fraction"] == pytest.approx(1.0 / 3.0)
    assert stats["exc_top1_activity_fraction"] == pytest.approx(2.0 / 3.0)
    assert stats["exc_top5_activity_fraction"] == pytest.approx(1.0)
    assert stats["exc_near_rate_cap_fraction"] == pytest.approx(1.0 / 6.0)
    assert stats["inh_mean"] is None
    assert stats["inh_median"] is None
    assert stats["inh_max"] is None
    assert stats["inh_active_fraction"] is None
    assert stats["inh_active_neuron_fraction"] is None
    assert stats["external_mean"] == pytest.approx(2.5)


def test_extended_plastic_weight_stats_reports_connected_delta_signs_and_percentiles() -> None:
    before = np.array([[0.1, 0.0, 0.3], [0.2, 0.4, 0.0]], dtype=float)
    after = np.array([[0.2, 0.0, 0.1], [0.2, 0.6, 0.0]], dtype=float)

    stats = extended_plastic_weight_stats(
        "W_TEST",
        after,
        previous=before,
        row_sum_limits=np.array([1.0, 2.0], dtype=float),
    )

    assert stats["W_TEST_p05"] == pytest.approx(0.115)
    assert stats["W_TEST_p95"] == pytest.approx(0.54)
    assert stats["W_TEST_delta_positive_fraction"] == pytest.approx(0.5)
    assert stats["W_TEST_delta_negative_fraction"] == pytest.approx(0.25)
    assert stats["W_TEST_delta_zero_fraction"] == pytest.approx(0.25)
    assert stats["W_TEST_row_sum_p95"] == pytest.approx(0.775)
    assert stats["W_TEST_row_sum_cap_max_ratio"] == pytest.approx(0.4)


def test_extended_plastic_weight_stats_marks_empty_rows_as_none() -> None:
    stats = extended_plastic_weight_stats(
        "W_EMPTY",
        np.empty((0, 3), dtype=float),
        previous=np.empty((0, 3), dtype=float),
        row_sum_limits=np.array([], dtype=float),
    )

    assert stats["W_EMPTY_p05"] is None
    assert stats["W_EMPTY_row_sum_mean"] is None
    assert stats["W_EMPTY_row_sum_p95"] is None
    assert stats["W_EMPTY_row_sum_max"] is None
    assert stats["W_EMPTY_row_sum_cap_max_ratio"] is None
    assert stats["W_EMPTY_row_sum_cap_fraction"] is None
    assert stats["W_EMPTY_delta_positive_fraction"] is None


def test_bcm_signal_and_theta_distribution_stats_handle_empty_inhibitory_state() -> None:
    state = BCMState(theta_exc=np.array([1.0, 3.0]), theta_inh=np.array([], dtype=float))
    rates = RateBatch(
        exc=np.array([[0.5, 4.0], [2.0, 1.0]], dtype=float),
        inh=np.empty((2, 0), dtype=float),
    )

    theta_stats = theta_distribution_stats(state)
    signal_stats = bcm_signal_stats(rates, state, BCMConfig(eta=0.5))

    assert theta_stats["theta_exc_p05"] == pytest.approx(1.1)
    assert theta_stats["theta_exc_p95"] == pytest.approx(2.9)
    assert theta_stats["theta_inh_mean"] is None
    assert theta_stats["theta_inh_median"] is None
    assert theta_stats["theta_inh_p05"] is None
    assert signal_stats["bcm_exc_above_theta_fraction"] == pytest.approx(0.5)
    assert signal_stats["bcm_inh_above_theta_fraction"] is None
    assert signal_stats["bcm_exc_signal_mean"] == pytest.approx(0.9375)


def test_evaluate_training_health_records_warns_and_fails_without_raising() -> None:
    diagnostics = [
        {
            "step": 1,
            "exc_active_neuron_fraction": 0.0,
            "exc_top1_activity_fraction": 0.0,
            "exc_top5_activity_fraction": 0.0,
            "exc_near_rate_cap_fraction": 0.0,
            "row_sum_EE_cap_fraction": 0.0,
            "row_sum_EE_cap_max_ratio": 0.0,
        },
        {
            "step": 2,
            "exc_active_neuron_fraction": 0.5,
            "exc_top1_activity_fraction": 0.6,
            "exc_top5_activity_fraction": 0.9,
            "exc_near_rate_cap_fraction": 0.1,
            "row_sum_EE_cap_fraction": 0.1,
            "row_sum_EE_cap_max_ratio": 1.2,
        },
    ]

    report = evaluate_training_health(diagnostics, TrainingHealthConfig())

    assert report["schema_version"] == 1
    assert report["status"] == "fail"
    assert report["warning_count"] >= 1
    assert report["failure_count"] == 1
    assert report["first_warning_step"] == 2
    assert report["first_failure_step"] == 1
    assert any(event["severity"] == "fail" and event["metric"] == "exc_active_neuron_fraction" for event in report["events"])
    assert report["final_metrics"]["exc_active_neuron_fraction"] == pytest.approx(0.5)
    assert report["worst_metrics"]["exc_top1_activity_fraction"] == pytest.approx(0.6)
