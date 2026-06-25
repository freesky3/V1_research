from __future__ import annotations

import numpy as np
import pytest

from v1_research.analysis.direction_tuning import DirectionTuningConfig, summarize_direction_tuning


def test_summarize_direction_tuning_reports_modulated_ensemble_coverage() -> None:
    labels = np.array([1, 1, 2, 2, 0])
    responses_mean = np.array(
        [
            [5.0, 1.0, 5.0, 1.0],
            [4.0, 1.0, 4.0, 1.0],
            [1.0, 6.0, 1.0, 6.0],
            [1.0, 5.0, 1.0, 5.0],
            [9.0, 9.0, 9.0, 9.0],
        ]
    )
    angles = np.array([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0])

    summary, rows = summarize_direction_tuning(
        labels,
        responses_mean,
        angles,
        DirectionTuningConfig(modulation_threshold=0.2),
    )

    assert summary["direction_selective_ensembles"] == 2
    assert summary["direction_selective_fraction"] == pytest.approx(1.0)
    assert summary["covered_direction_count"] == 2
    assert summary["mean_ensemble_modulation"] == pytest.approx(((3.5 / 5.5) + (4.5 / 6.5)) / 2.0)
    assert rows[0]["ensemble_id"] == 1
    assert rows[0]["preferred_direction_deg"] == 0.0
    assert rows[0]["modulation_index"] == pytest.approx(3.5 / 5.5)
    assert rows[0]["direction_selective"] is True
    assert rows[0]["mean_rate_0deg"] == pytest.approx(4.5)
    assert rows[1]["preferred_direction_deg"] == 90.0
