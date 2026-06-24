from __future__ import annotations

import numpy as np
import pytest

from v1_research.inputs import BackgroundConfig, BackgroundTrace, OUParams, generate_background_trace, generate_ou_background


def test_background_trace_generation_uses_global_seed() -> None:
    cfg = BackgroundConfig(
        enabled=True,
        exc=OUParams(mean=0.0, stationary_std=1.0, tau=0.05),
        inh=OUParams(mean=0.0, stationary_std=0.5, tau=0.05),
    )
    time = np.linspace(0.0, 0.2, 5)

    np.random.seed(7)
    first = generate_background_trace(cfg, n_exc=3, n_inh=2, n_batch=4, time=time)
    np.random.seed(7)
    second = generate_background_trace(cfg, n_exc=3, n_inh=2, n_batch=4, time=time)

    assert first is not None
    assert second is not None
    assert first.exc.shape == (5, 4, 3)
    assert first.inh.shape == (5, 4, 2)
    np.testing.assert_allclose(first.exc, second.exc)
    np.testing.assert_allclose(first.inh, second.inh)
    assert generate_background_trace(BackgroundConfig(enabled=False), n_exc=1, n_inh=1, n_batch=1, time=time) is None


def test_background_interpolation_and_rk4_samples() -> None:
    trace = BackgroundTrace(
        time=np.array([0.0, 1.0, 2.0]),
        exc=np.array([[[0.0]], [[2.0]], [[4.0]]]),
        inh=np.array([[[1.0]], [[3.0]], [[5.0]]]),
        interpolation="linear",
    )
    exc_mid, inh_mid = trace.value_at(0.5)
    assert exc_mid[0, 0] == pytest.approx(1.0)
    assert inh_mid[0, 0] == pytest.approx(2.0)

    samples = trace.rk4_samples()
    assert samples.exc_left.shape == (2, 1, 1)
    np.testing.assert_allclose(samples.exc_mid[:, 0, 0], [1.0, 3.0])

    held = BackgroundTrace(time=trace.time, exc=trace.exc, inh=trace.inh, interpolation="sample_hold")
    assert held.value_at(0.5)[0][0, 0] == pytest.approx(0.0)


def test_background_rejects_bad_time_grid() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_ou_background(
            n_exc=1,
            n_inh=1,
            n_batch=1,
            time=np.array([0.0, 0.0]),
            exc=OUParams(),
            inh=OUParams(),
        )
