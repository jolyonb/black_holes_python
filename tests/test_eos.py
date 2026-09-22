"""Tests of pbh.eos: the constants of Section 3 as printed, checked for radiation and exactly for general w."""

import math
from fractions import Fraction

import numpy as np
import pytest

from pbh.eos import RADIATION, Background, EquationOfState, as_rational_w

# --- canonicalisation of w ---


def test_exact_inputs_are_taken_exactly():
    assert as_rational_w("1/3") == Fraction(1, 3)
    assert as_rational_w(Fraction(2, 6)) == Fraction(1, 3)
    assert as_rational_w("0.2") == Fraction(1, 5)
    assert as_rational_w(1) == Fraction(1)


def test_a_float_that_is_not_its_rounded_rational_warns():
    with pytest.warns(UserWarning, match="rounded to the rational 3/10"):
        assert as_rational_w(0.1 + 0.2) == Fraction(3, 10)  # 0.30000000000000004 is not the double nearest 3/10


def test_a_float_that_round_trips_is_taken_silently():
    assert as_rational_w(0.5) == Fraction(1, 2)
    assert as_rational_w(1 / 3) == Fraction(1, 3)  # the double nearest 1/3 rounds back to exactly 1/3


@pytest.mark.parametrize("bad", [0, "0", -1, Fraction(3, 2), 1.5])
def test_w_outside_the_admissible_range_is_refused(bad: object):
    with pytest.raises(ValueError, match="0 < w <= 1"):
        as_rational_w(bad)  # type: ignore[arg-type]


def test_the_dataclass_refuses_an_inadmissible_w_too():
    with pytest.raises(ValueError, match="0 < w <= 1"):
        EquationOfState(Fraction(0))


# --- radiation, the case every number in the paper is for ---


@pytest.fixture
def radiation() -> EquationOfState:
    return EquationOfState(RADIATION)


def test_radiation_constants_are_the_printed_values(radiation: EquationOfState):
    assert radiation.is_radiation
    assert radiation.alpha == Fraction(1, 2)  # eq:asol: alpha = 1/2 for w = 1/3
    assert radiation.sqrt_w == pytest.approx(1 / math.sqrt(3))
    assert radiation.lapse_exponent == -0.25  # eq:MSphinov: e^phi = rhotilde^(-1/4)
    assert radiation.energy_source_rate == 0.5  # 2 - 3 alpha
    assert radiation.growing_mode_rate == 1.0  # lambda_g = 2 (1 - alpha), Section 7.6
    assert radiation.accretion_eigenvalue == pytest.approx(6 * math.sqrt(3))  # eq:exc:lambdac
    assert radiation.sonic_radius_over_mass == 3.0  # eq:exc:sonic: r_s = 3M


def test_radiation_background_at_a_generic_time(radiation: EquationOfState):
    xi = 1.7
    bg = Background.at(radiation, xi)
    assert bg.xi == xi
    assert bg.a == pytest.approx(math.exp(xi / 2))
    assert bg.H == pytest.approx(math.exp(-xi))
    assert bg.hubble_radius == pytest.approx(math.exp(xi / 2))
    assert bg.Gammabar2 == pytest.approx(math.exp(xi))
    assert bg.c_s**2 == pytest.approx(math.exp(xi) / 12)  # Section 5.1: c_s^2 = e^xi / 12
    assert bg.tau == pytest.approx(math.exp(xi / 2) / math.sqrt(3))  # eq:lin:tau
    assert bg.tau == pytest.approx(2 * bg.c_s)


def test_a_non_radiation_fluid_is_flagged(radiation: EquationOfState):
    assert not EquationOfState(Fraction(1, 5)).is_radiation
    assert radiation.is_radiation


# --- general w: the identities the code relies on, in exact rational arithmetic where they are rational ---

SEVERAL_W = [Fraction(1, 5), Fraction(1, 3), Fraction(2, 3), Fraction(1)]


@pytest.mark.parametrize("w", SEVERAL_W)
def test_rational_identities_between_the_constants_hold_exactly(w: Fraction):
    alpha = EquationOfState(w).alpha
    assert alpha == Fraction(2, 3) / (1 + w)  # eq:asol
    assert 2 - 3 * alpha == 3 * alpha * w  # the energy source rate, Section 7.2
    assert -w / (1 + w) == -Fraction(3, 2) * alpha * w  # the lapse exponent, eq:MSphinov
    assert 2 * (1 - alpha) == (2 - 3 * alpha) * (1 + 3 * w) / (3 * w)  # lambda_g against the source rate


@pytest.mark.parametrize("w", SEVERAL_W)  # radiation and the stiff fluid by square roots, the rest by the power
def test_the_lapse_is_the_power_of_the_density_to_an_ulp(w: Fraction):
    eos = EquationOfState(w)
    rho = np.geomspace(1e-6, 1e6, 1001)
    assert np.max(np.abs(eos.lapse(rho) / rho**eos.lapse_exponent - 1.0)) <= 2.3e-16


@pytest.mark.parametrize("w", SEVERAL_W)
def test_the_lapse_deviation_is_formed_without_subtracting_one(w: Fraction):
    # Against expm1(k log1p(delta_rho)), accurate to an ulp of the deviation itself, for densities near and far from 1:
    # radiation and the stiff fluid by their square-root formulas, the rest by that formula.
    eos = EquationOfState(w)
    delta_rho = np.concatenate((np.geomspace(1e-12, 1e3, 200), -np.geomspace(1e-12, 0.999, 200)))
    rho = 1.0 + delta_rho
    ephi, delta_ephi = eos.lapse_and_deviation(rho, delta_rho)
    reference = np.expm1(eos.lapse_exponent * np.log1p(delta_rho))
    assert np.max(np.abs(delta_ephi / reference - 1.0)) < 1e-15
    assert np.array_equal(ephi, eos.lapse(rho))


@pytest.mark.parametrize("w", SEVERAL_W)
def test_floats_agree_with_the_exact_formulas_for_several_w(w: Fraction):
    eos = EquationOfState(w)
    alpha = eos.alpha
    assert eos.lapse_exponent == pytest.approx(float(-w / (1 + w)))
    assert eos.energy_source_rate == pytest.approx(float(3 * alpha * w))
    assert eos.growing_mode_rate == pytest.approx(float(2 * (1 - alpha)))
    one_plus_3w = float(1 + 3 * w)
    assert eos.accretion_eigenvalue == pytest.approx(
        one_plus_3w ** (one_plus_3w / (2 * float(w))) / (4 * float(w) ** 1.5)
    )
    xi = 0.9
    bg = Background.at(eos, xi)
    assert bg.c_s == pytest.approx(float(alpha) * math.sqrt(w) * math.exp((1 - float(alpha)) * xi))
    assert bg.Gammabar2 == pytest.approx(bg.hubble_radius**2)
    tau_printed = 2 * math.sqrt(w) / float(1 + 3 * w) * math.exp((1 - float(alpha)) * xi)  # eq:lin:tau as printed
    assert bg.tau == pytest.approx(tau_printed)
    assert bg.hubble_radius / tau_printed == pytest.approx(float(1 + 3 * w) / (2 * math.sqrt(w)))  # Section 5.1


def test_both_dataclasses_are_frozen(radiation: EquationOfState):
    with pytest.raises(AttributeError):
        radiation.w = Fraction(1, 2)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        Background.at(radiation, 0.0).xi = 1.0  # type: ignore[misc]
