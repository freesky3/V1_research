from __future__ import annotations

import numpy as np
import pytest

from v1_research.dynamics import RateResult
from v1_research.inputs.background import BackgroundTrace
from v1_research.workflows.simulation_health import SimulationHealthConfig, compute_simulation_health


def test_compute_simulation_health_reports_activity_concentration_stability_and_inputs() -> None:
    exc_trajectory = np.array(
        [
            [[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]],
            [[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]],
            [[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]],
            [[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]],
        ],
        dtype=float,
    )
    inh_trajectory = np.array(
        [
            [[0.0], [0.0]],
            [[1.0], [1.0]],
            [[2.0], [2.0]],
            [[3.0], [3.0]],
        ],
        dtype=float,
    )
    rates = RateResult(
        exc=np.mean(exc_trajectory[2:], axis=0),
        inh=np.mean(inh_trajectory[2:], axis=0),
        exc_trajectory=exc_trajectory,
        inh_trajectory=inh_trajectory,
        time=np.array([0.0, 0.1, 0.2, 0.3], dtype=float),
    )
    stimulus_trace = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 3.0], [4.0, 5.0]],
            [[3.0, 4.0], [5.0, 6.0]],
            [[4.0, 5.0], [6.0, 7.0]],
        ],
        dtype=float,
    )
    background = BackgroundTrace(
        time=rates.time,
        exc=np.full_like(exc_trajectory, 0.2),
        inh=np.full_like(inh_trajectory, 0.4),
    )

    report = compute_simulation_health(
        rates,
        stimulus_trace=stimulus_trace,
        background_trace=background,
        rate_max=5.0,
        cfg=SimulationHealthConfig(
            active_rate_threshold=1.0,
            near_rate_cap_ratio=0.9,
            stability_window_fraction=0.5,
            tail_fraction=0.5,
        ),
    )

    metrics = report["metrics"]
    assert report["schema_version"] == 1
    assert report["status"] == "warn"
    assert metrics["exc_active_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["exc_silent_fraction"] == pytest.approx(1.0 / 3.0)
    assert metrics["exc_mean"] == pytest.approx(2.5)
    assert metrics["exc_median"] == pytest.approx(2.5)
    assert metrics["exc_p95"] == pytest.approx(5.0)
    assert metrics["exc_max"] == pytest.approx(5.0)
    assert metrics["exc_top1_activity_fraction"] == pytest.approx(0.6)
    assert metrics["exc_top5_activity_fraction"] == pytest.approx(1.0)
    assert metrics["exc_near_rate_cap_fraction"] == pytest.approx(1.0 / 6.0)
    assert metrics["exc_front_mean"] == pytest.approx(2.0)
    assert metrics["exc_tail_mean"] == pytest.approx(3.0)
    assert metrics["exc_mean_drift"] == pytest.approx(1.0)
    assert metrics["exc_relative_mean_drift"] == pytest.approx(0.5)
    assert metrics["exc_step_mean_abs_change"] == pytest.approx(1.0 / 3.0)
    assert metrics["inh_tail_variance"] == pytest.approx(0.25)
    assert metrics["stimulus_mean"] == pytest.approx(4.0)
    assert metrics["stimulus_p95"] == pytest.approx(6.25)
    assert metrics["background_exc_mean"] == pytest.approx(0.2)
    assert metrics["background_inh_mean"] == pytest.approx(0.4)
    assert any(event["metric"] == "exc_top1_activity_fraction" for event in report["events"])
