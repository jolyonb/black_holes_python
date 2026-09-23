"""Tests of pbh.readout: the rate-corrected estimate and its bar on synthetic accretion histories (Section 6.3)."""

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pbh.eos import RADIATION, EquationOfState
from pbh.horizon import Horizon, HorizonReport
from pbh.readout import Epoch, ReadoutSettings, first_reading, readings, resample, starts_new_epoch
from pbh.types import FloatArray

EOS = EquationOfState(RADIATION)
LAMBDA_C = EOS.accretion_eigenvalue
SETTINGS = ReadoutSettings()
XI_FORM = 5.0


def step_times(xi_form: float, xi_end: float) -> FloatArray:
    """Step times as a run takes them after formation: steps shrinking like e^(-(xi - xi_form)), from 0.02 to 0.002."""
    t, times = xi_form, [xi_form]
    while t < xi_end:
        t += max(0.02 * math.exp(-(t - xi_form)), 0.002)
        times.append(t)
    return np.array(times)


def constant_efficiency(xi: FloatArray, M_inf: float, efficiency: float) -> FloatArray:
    """The law eq:exc:ahlaw at `efficiency` times the Michel value: M = M_inf / (1 + lambda M_inf e^-xi)."""
    return M_inf / (1.0 + efficiency * LAMBDA_C * M_inf * np.exp(-xi))


def accreted(xi: FloatArray, M_0: float, lambda_a: object) -> FloatArray:
    """`d ln M / d xi = lambda_a(xi) M e^(-xi)` integrated from `M_0` at `xi[0]`, to round-off, at the times `xi`."""
    assert callable(lambda_a)

    def rate(t: float, y: FloatArray) -> FloatArray:
        return np.array([lambda_a(t) * y[0] ** 2 * math.exp(-t)])

    sol = solve_ivp(rate, (float(xi[0]), float(xi[-1])), [M_0], t_eval=xi, rtol=1e-12, atol=1e-14, method="DOP853")
    return sol.y[0]


# --- the pieces ---


def test_resampling_is_uniform_from_the_first_sample_to_the_last():
    xi = np.array([1.0, 1.013, 1.02, 1.1])
    times, ln_M = resample(xi, np.exp(xi), 0.01)
    assert times[0] == 1.0
    assert times[-1] == pytest.approx(1.1)
    assert np.allclose(np.diff(times), 0.01)
    assert ln_M == pytest.approx(times)  # ln M is linear in xi here, so the interpolation is exact
    with pytest.raises(ValueError, match="increase strictly"):
        resample(np.array([1.0, 1.0, 1.1]), np.ones(3), 0.01)


def test_the_settings_refuse_nonsense():
    with pytest.raises(ValueError, match="positive"):
        ReadoutSettings(target=0.0)
    with pytest.raises(ValueError, match="too few points"):
        ReadoutSettings(window=0.01, spacing=0.005)


def test_on_pure_exponential_growth_the_rate_and_the_mass_are_exact():
    # ln M linear in xi: every window's line is the line itself.
    xi = step_times(XI_FORM, XI_FORM + 1.0)
    r = readings(xi, 2.0 * np.exp(0.1 * (xi - XI_FORM)), EOS, SETTINGS)
    assert r.omega == pytest.approx(0.1, rel=1e-12)
    assert r.M_AH == pytest.approx(2.0 * np.exp(0.1 * (r.xi - XI_FORM)), rel=1e-12)
    assert r.xi[0] == pytest.approx(XI_FORM + 0.15)  # the first centre has half a window behind it
    assert r.xi[-1] <= xi[-1] - 0.15 + 1e-9


def test_a_series_shorter_than_one_window_has_no_readings():
    xi = np.array([XI_FORM, XI_FORM + 0.1, XI_FORM + 0.2])
    r = readings(xi, np.ones(3), EOS, SETTINGS)
    assert r.xi.size == 0
    assert first_reading(r, XI_FORM, SETTINGS) is None


# --- the estimate on the accretion law (eq:exc:rateest) ---


@pytest.mark.parametrize("efficiency", [0.6, 1.0, 1.4])
def test_with_a_constant_efficiency_the_estimate_is_the_final_mass_at_every_time(efficiency: float):
    # M_est = M_inf identically in time whatever the efficiency, but for the straight line fitted across a curved
    # ln M: a systematic of order the window squared times the curvature, some 1.6e-3 omega, largest right after
    # formation where omega is large and a few 1e-5 from the floor on.
    M_inf = 3.0
    xi = step_times(XI_FORM, XI_FORM + 4.0)
    r = readings(xi, constant_efficiency(xi, M_inf, efficiency), EOS, SETTINGS)
    error = np.abs(r.M_est / M_inf - 1.0)
    after = r.xi >= XI_FORM + SETTINGS.floor
    assert np.max(error) < 5e-4
    assert np.max(error[after]) < 1e-4
    # ... which the reported systematic omega W^2 / 60 predicts, sign and size, to its leading order in omega
    low = r.M_est[after] / M_inf - 1.0
    assert np.all(low < 0.0)
    assert np.max(np.abs(low + r.systematic[after]) / r.systematic[after]) < 0.25
    # the horizon mass itself is low by about omega, which is large early
    assert r.M_AH[0] / M_inf - 1.0 < -0.05
    assert r.efficiency[after] == pytest.approx(efficiency, rel=5e-3)  # lambda_a / lambda_c, with the fit's systematic
    assert r.lambda_c_eps == pytest.approx(LAMBDA_C * r.M_AH * np.exp(-r.xi))
    # Q = ln(lambda_a M_inf) is constant, so the bar vanishes but for the fit's systematic, which it does not measure
    assert r.Q[after] == pytest.approx(math.log(efficiency * LAMBDA_C * M_inf), abs=5e-3)
    assert np.all(np.isfinite(r.bar[after]))
    assert np.max(r.bar[after]) < 2e-5


def test_where_the_efficiency_drifts_the_estimate_drifts_as_eq_exc_estdrift_says():
    # lambda_a falls from 1.5 to 1 times Michel: varsigma = d ln lambda_a / d xi < 0, so the estimate falls, at
    # varsigma omega / (1 - omega).
    M_0 = 0.8

    def lambda_a(t: float) -> float:
        return LAMBDA_C * (1.0 + 0.5 * math.exp(-(t - XI_FORM)))

    xi = step_times(XI_FORM, XI_FORM + 4.0)
    r = readings(xi, accreted(xi, M_0, lambda_a), EOS, SETTINGS)
    varsigma = np.array([-0.5 * math.exp(-(t - XI_FORM)) / (1.0 + 0.5 * math.exp(-(t - XI_FORM))) for t in r.xi])
    drift = np.gradient(np.log(r.M_est), r.xi)
    predicted = varsigma * r.omega / (1.0 - r.omega)
    inner = (r.xi > XI_FORM + 0.5) & (r.xi < r.xi[-1] - 0.1)  # past the coarse first steps, and the derivative's end
    assert np.max(np.abs(drift[inner] - predicted[inner])) < 1e-2 * np.max(np.abs(predicted))
    assert np.all(np.diff(r.M_est) < 0.0)  # it falls while the efficiency falls
    # dQ/d xi = varsigma / (1 - omega)
    dQ = np.gradient(r.Q, r.xi)
    assert np.max(np.abs(dQ[inner] - (varsigma / (1.0 - r.omega))[inner])) < 2e-3


def test_after_the_floor_the_bar_bounds_the_error_of_the_estimate():
    # The efficiency relaxes from 1.6 to 1 times Michel and the run is long enough to find the final mass: from the
    # floor on, wherever the bar is below five per cent, the error of the estimate never exceeds it (Section 8.5).
    def lambda_a(t: float) -> float:
        return LAMBDA_C * (1.0 + 0.6 * math.exp(-1.5 * (t - XI_FORM)))

    long = step_times(XI_FORM, XI_FORM + 14.0)
    M = accreted(long, 0.8, lambda_a)
    # by the end lambda_a = lambda_c to 1e-9, and the constant-efficiency law gives M_inf from the last value
    M_inf = float(M[-1] / (1.0 - LAMBDA_C * M[-1] * math.exp(-long[-1])))
    r = readings(long, M, EOS, SETTINGS)
    after = (r.xi >= XI_FORM + SETTINGS.floor) & (r.bar < 0.05)
    assert after.sum() > 100
    error = np.abs(r.M_est[after] / M_inf - 1.0)
    assert np.all(error <= r.bar[after] + 1e-6)
    i = first_reading(r, XI_FORM, SETTINGS)
    assert i is not None
    assert r.xi[i] >= XI_FORM + SETTINGS.floor - 1e-9
    assert r.bar[i] < SETTINGS.target
    assert abs(r.M_est[i] / M_inf - 1.0) < SETTINGS.target


# --- the read-out decision ---


def test_the_first_reading_waits_for_the_floor_and_for_the_bar():
    M_inf = 3.0
    xi = step_times(XI_FORM, XI_FORM + 4.0)
    r = readings(xi, constant_efficiency(xi, M_inf, 1.0), EOS, SETTINGS)
    # with a constant efficiency the bar is tiny at once, so the floor decides
    i = first_reading(r, XI_FORM, SETTINGS)
    assert i is not None
    assert r.xi[i] == pytest.approx(XI_FORM + SETTINGS.floor, abs=SETTINGS.spacing)
    # a target the bar never meets gives no reading
    assert first_reading(r, XI_FORM, ReadoutSettings(target=1e-9)) is None


def test_where_the_mass_falls_q_and_the_bar_are_undefined():
    # omega < 0: Q = ln of a negative number is not a number, and neither is any bar whose span holds it, so a mass
    # that falls, which an apparent-horizon mass never does, is never read.
    xi = step_times(XI_FORM, XI_FORM + 2.0)
    r = readings(xi, 2.0 * (1.0 - 0.01 * (xi - XI_FORM)), EOS, SETTINGS)
    assert np.all(r.omega < 0.0)
    assert np.all(np.isnan(r.Q))
    assert np.all(np.isnan(r.bar))
    assert first_reading(r, XI_FORM, SETTINGS) is None


# --- epochs ---


def test_an_epoch_ignores_a_time_it_already_has_and_a_new_epoch_needs_a_horizon():
    epoch = Epoch(xi_start=XI_FORM)
    epoch.add(XI_FORM, 1.0, 0.5)
    epoch.add(XI_FORM, 1.1, 0.6)  # a restart re-examines the state it starts from
    assert (epoch.xi, epoch.M_AH, epoch.X_AH) == ([XI_FORM], [1.0], 0.5)
    nan = float("nan")
    empty = HorizonReport(np.ones(3), 0, (), None, nan, nan, nan, 0, nan, 0, False)
    assert not starts_new_epoch(empty, 0.5)
    shell = Horizon(j=1, x=0.4, X=0.8, outer=True)
    alone = HorizonReport(np.ones(3), 1, (shell,), shell, 1.0, nan, nan, 0, nan, 0, False)
    assert not starts_new_epoch(alone, 0.5)  # nothing inside it: the old region, grown
