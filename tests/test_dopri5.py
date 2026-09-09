"""Tests for the DOPRI5 integrator."""

import numpy as np
import pytest

from pbh.base import FloatArray
from pbh.dopri5 import DOPRI5, DopriIntegrationError


def decay(_t: float, y: FloatArray, _params: object) -> FloatArray:
    return -y


def oscillator(_t: float, y: FloatArray, _params: object) -> FloatArray:
    return np.array([y[1], -y[0]])


def integrate_to(integrator: DOPRI5, t_end: float) -> int:
    steps = 0
    while integrator.t < t_end:
        integrator.step(t_end)
        steps += 1
    return steps


def test_exponential_decay() -> None:
    integrator = DOPRI5(0.0, np.array([1.0]), decay, rtol=1e-10, atol=1e-10)
    integrate_to(integrator, 2.0)
    assert integrator.t == 2.0
    assert integrator.values[0] == pytest.approx(np.exp(-2.0), rel=1e-8)


def test_harmonic_oscillator() -> None:
    integrator = DOPRI5(0.0, np.array([1.0, 0.0]), oscillator, rtol=1e-10, atol=1e-10)
    integrate_to(integrator, 2 * np.pi)
    assert integrator.values == pytest.approx([1.0, 0.0], abs=1e-7)


def test_step_never_overshoots_target() -> None:
    integrator = DOPRI5(0.0, np.array([1.0]), decay, init_h=0.5, max_h=10.0)
    while integrator.t < 0.3:
        integrator.step(0.3)
        assert integrator.t <= 0.3
    assert integrator.t == 0.3


def test_adaptive_step_grows_for_smooth_problem() -> None:
    integrator = DOPRI5(0.0, np.array([1.0]), decay, init_h=1e-4, rtol=1e-6, atol=1e-6)
    steps = integrate_to(integrator, 5.0)
    assert steps < 100
    assert integrator.hdid > 1e-4


def test_params_are_forwarded() -> None:
    def derivs(_t: float, y: FloatArray, params: object) -> FloatArray:
        assert isinstance(params, float)
        return -params * y

    integrator = DOPRI5(0.0, np.array([1.0]), derivs, params=2.0, rtol=1e-10, atol=1e-10)
    integrate_to(integrator, 1.0)
    assert integrator.values[0] == pytest.approx(np.exp(-2.0), rel=1e-8)


def test_update_max_h_below_min_raises() -> None:
    integrator = DOPRI5(0.0, np.array([1.0]), decay, min_h=1e-3)
    with pytest.raises(DopriIntegrationError, match="max step size"):
        integrator.update_max_h(1e-4)


def test_update_max_h_clamps_next_step() -> None:
    integrator = DOPRI5(0.0, np.array([1.0]), decay, init_h=0.5)
    integrator.update_max_h(0.1)
    assert integrator.hnext == 0.1


def test_min_step_failure_raises() -> None:
    def rough(t: float, _y: FloatArray, _params: object) -> FloatArray:
        # A derivative that is wildly discontinuous defeats the error estimate
        return np.array([1e12 if (t * 1e6) % 2 < 1 else -1e12])

    integrator = DOPRI5(0.0, np.array([0.0]), rough, min_h=1e-3, rtol=1e-12, atol=1e-12)
    with pytest.raises(DopriIntegrationError, match="minimum threshold"):
        integrate_to(integrator, 1.0)


def test_set_init_values_clears_fsal() -> None:
    integrator = DOPRI5(0.0, np.array([1.0]), decay)
    integrator.step(0.1)
    assert integrator.dxdt is not None
    integrator.set_init_values(0.0, np.array([2.0]))
    assert integrator.dxdt is None
    assert integrator.t == 0.0
