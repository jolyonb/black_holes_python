"""Tests of pbh.causal: the domain that keeps a radius out of the outer boundary's reach (Section 8.6)."""

import math
from fractions import Fraction

import pytest

from pbh.causal import isolation
from pbh.eos import RADIATION, EquationOfState

RAD = EquationOfState(RADIATION)


def test_sound_and_light_cross_the_sound_horizon_and_the_hubble_radius():
    # radiation: tau = e^(xi/2) / sqrt 3 (eq:numbh:causal) and tau_L = e^(xi/2) (eq:numbh:causalL)
    reach = isolation(RAD, 0.0, 7.0, 0.5, 30.0)
    assert reach["needed_sound"] == pytest.approx(0.5 + (math.exp(3.5) - 1.0) / math.sqrt(3.0), rel=1e-14)
    assert reach["needed_light"] == pytest.approx(0.5 + math.exp(3.5) - 1.0, rel=1e-14)
    assert reach["isolated_sound"] is True  # 19.2 against 30
    assert reach["isolated_light"] is False  # 32.6 against 30
    assert (reach["r"], reach["since"], reach["Rtilde_max"]) == (0.5, 0.0, 30.0)


def test_from_the_distant_past_sound_reaches_the_origin_from_the_boundary_at_the_printed_time():
    # Section 8.6: from the origin to the boundary takes until xi = 2 ln Rtilde_max + ln 3
    R = 24.0
    reach = isolation(RAD, -60.0, 2.0 * math.log(R) + math.log(3.0), 0.0, R)
    assert reach["needed_sound"] == pytest.approx(R, rel=1e-12)


def test_light_is_sound_over_the_sound_speed_for_any_fluid():
    eos = EquationOfState(Fraction(1, 5))
    reach = isolation(eos, -2.0, 3.0, 1.0, 50.0)
    assert reach["needed_light"] - 1.0 == pytest.approx((reach["needed_sound"] - 1.0) / math.sqrt(0.2), rel=1e-14)
