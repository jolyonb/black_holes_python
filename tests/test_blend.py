"""Tests of the post-formation map of Section 8.1: the step, the ramp, the four identities, repeated switch-on, FRW."""

import math

import numpy as np
import pytest
from evolve import evolve

from pbh.eos import RADIATION, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME
from pbh.layout import Layout
from pbh.maps import BlendMap, IdentityMap, Map, PinnedMap, SinhStretch, Zone, fractions, quintic_step, ramp
from pbh.outer import OutgoingWave
from pbh.state import frw_state
from pbh.stencils import FaceClosure
from pbh.timestep import Scheme
from pbh.types import FloatArray

RAD = EquationOfState(RADIATION)
ALPHA = float(RAD.alpha)
BASES: list[Map] = [IdentityMap(12.0), SinhStretch(12.0, scale=3.0)]
ZONE = Zone(xi_on=4.7, tau_on=0.3, x_t=0.2, Delta_t=0.075)


def printed_map(base: Map, alpha: float, zone: Zone, xi: float, N: int) -> tuple[FloatArray, FloatArray]:
    """eq:numbh:map written out independently: X = B [chi + (1 - chi) e^-aT], d_xi X = -a T' B (1 - chi) e^-aT."""
    B, _ = base.radii(0.0, N)
    u = fractions(N)
    zeta = (u - zone.x_t) / zone.Delta_t
    chi = np.where(zeta <= -1.0, 0.0, np.where(zeta >= 1.0, 1.0, 0.5 + (15 * zeta - 10 * zeta**3 + 3 * zeta**5) / 16))
    s = xi - zone.xi_on
    T = s - zone.tau_on * (1.0 - math.exp(-s / zone.tau_on)) if s > 0 else 0.0
    dT = 1.0 - math.exp(-s / zone.tau_on) if s > 0 else 0.0
    return B * (chi + (1.0 - chi) * math.exp(-alpha * T)), -alpha * dT * B * (1.0 - chi) * math.exp(-alpha * T)


# --- the pieces ---


def second_difference(zeta: float, h: float) -> float:
    """The second difference of the step at zeta, an estimate of its second derivative."""
    values = quintic_step(np.array([zeta + h, zeta, zeta - h]))
    return float(values[0] - 2 * values[1] + values[2]) / h**2


def test_the_quintic_step_is_the_printed_polynomial_flat_outside_and_c2_at_the_joins():
    zeta = np.linspace(-1.5, 1.5, 3001)
    sigma = quintic_step(zeta)
    assert np.all(sigma[zeta <= -1.0] == 0.0)
    assert np.all(sigma[zeta >= 1.0] == 1.0)
    inside = np.abs(zeta) < 1.0
    assert sigma[inside] == pytest.approx(
        0.5 + (15 * zeta[inside] - 10 * zeta[inside] ** 3 + 3 * zeta[inside] ** 5) / 16
    )
    assert quintic_step(np.array([0.0]))[0] == 0.5
    # the printed derivative (15/16)(1 - zeta^2)^2, and the same polynomial as 6u^5 - 15u^4 + 10u^3, u = (zeta + 1)/2
    h = 1e-6
    numerical = (quintic_step(zeta + h) - quintic_step(zeta - h)) / (2 * h)
    printed = np.where(inside, 15.0 / 16.0 * (1.0 - zeta**2) ** 2, 0.0)
    assert numerical == pytest.approx(printed, abs=1e-8)
    u = (zeta + 1.0) / 2.0
    assert sigma[inside] == pytest.approx(6 * u[inside] ** 5 - 15 * u[inside] ** 4 + 10 * u[inside] ** 3)
    # C^2: the second difference is continuous across zeta = +-1 (the third derivative jumps there, as a quintic's must)
    for edge in (-1.0, 1.0):
        second = [second_difference(e, h) for e in (edge - 1e-3, edge + 1e-3)]
        assert abs(second[0] - second[1]) < 1e-2
    # and a tanh is never exactly flat: the reason it is not used
    assert 0.5 * (1.0 + np.tanh(3.0)) != 1.0


def test_the_ramp_starts_flat_with_zero_rate_never_exceeds_unit_rate_and_saturates():
    xi_on, tau_on = 4.7, 0.3
    assert ramp(4.0, xi_on, tau_on) == (0.0, 0.0)
    assert ramp(xi_on, xi_on, tau_on) == (0.0, 0.0)
    T, dT = ramp(xi_on + 1e-3, xi_on, tau_on)
    assert 0.0 < T < 1e-5  # second order in the offset: T ~ s^2 / (2 tau_on)
    assert 0.0 < dT < 0.01
    assert T == pytest.approx(1e-6 / (2 * tau_on), rel=1e-2)
    for s in (0.1, 0.5, 2.0):
        T, dT = ramp(xi_on + s, xi_on, tau_on)
        assert 0.0 <= dT < 1.0
        assert T == pytest.approx(s - tau_on * (1.0 - math.exp(-s / tau_on)))
        h = 1e-6
        assert dT == pytest.approx(
            (ramp(xi_on + s + h, xi_on, tau_on)[0] - ramp(xi_on + s - h, xi_on, tau_on)[0]) / (2 * h), rel=1e-6
        )
    T, dT = ramp(xi_on + 5.0, xi_on, tau_on)
    assert T == pytest.approx(5.0 - tau_on, rel=1e-6)
    assert dT == pytest.approx(1.0, abs=1e-6)


def test_a_zone_is_refused_at_the_origin_at_the_outer_face_or_with_a_bad_ramp():
    with pytest.raises(ValueError, match="outside the origin"):
        Zone(xi_on=1.0, tau_on=0.3, x_t=0.05, Delta_t=0.1)
    with pytest.raises(ValueError, match="outer face must be static"):
        Zone(xi_on=1.0, tau_on=0.3, x_t=0.9, Delta_t=0.15)
    with pytest.raises(ValueError, match="tau_on > 0"):
        Zone(xi_on=1.0, tau_on=0.0, x_t=0.5, Delta_t=0.1)


# --- the map with one zone: the printed map and its four identities ---


@pytest.mark.parametrize("base", BASES)
@pytest.mark.parametrize("xi", [4.0, 4.7, 4.75, 5.0, 7.0])
def test_one_zone_is_the_printed_map_on_both_bases(base: Map, xi: float):
    m = BlendMap(base, ALPHA, (ZONE,))
    X, X_xi = m.radii(xi, 200)
    X_printed, X_xi_printed = printed_map(base, ALPHA, ZONE, xi, 200)
    assert X == pytest.approx(X_printed, rel=1e-14, abs=1e-14)
    assert X_xi == pytest.approx(X_xi_printed, rel=1e-14, abs=1e-14)
    assert not m.is_static


@pytest.mark.parametrize("base", BASES)
def test_the_four_identities(base: Map):
    N = 400
    m = BlendMap(base, ALPHA, (ZONE,))
    B, _ = base.radii(0.0, N)
    u = fractions(N)
    inside, outside = u <= ZONE.inner_edge, u >= ZONE.outer_edge
    for xi in (4.7, 4.8, 5.5, 8.0):
        X, X_xi = m.radii(xi, N)
        T, dT = ramp(xi, ZONE.xi_on, ZONE.tau_on)
        # 1. inside the transition the coordinate is pinned up to the ramp, and odd if B is: X_0 = 0 exactly
        assert X[inside] == pytest.approx(math.exp(-ALPHA * T) * B[inside], rel=1e-15)
        assert X_xi[inside] == pytest.approx(-ALPHA * dT * X[inside], rel=1e-15, abs=1e-300)
        assert X[0] == 0.0
        # 2. outside it the far zone is the base and the outer face is static, exactly
        assert np.array_equal(X[outside], B[outside])
        assert np.all(X_xi[outside] == 0.0)
        assert X[N] == B[N]
        assert X_xi[N] == 0.0
        # 3. the radii increase strictly, so the geometry accepts the map
        assert np.all(np.diff(X) > 0.0)
        Geometry.of(X, X_xi)
        # 4. no coordinate line moves inward in physical radius
        assert np.all(ALPHA * X + X_xi >= 0.0)
    # at switch-on the map is the base in value and velocity: the state carries over unchanged
    X_on, X_xi_on = m.radii(ZONE.xi_on, N)
    assert np.array_equal(X_on, B)
    assert np.all(X_xi_on == 0.0)


def test_the_pinned_map_is_the_blend_with_the_step_pushed_beyond_the_grid():
    # PinnedMap pins everything; a blend whose transition sits beyond the labels pins everything on the grid too.
    N = 50
    pinned = PinnedMap(IdentityMap(6.0), ALPHA, xi_on=2.0)
    far = Zone(xi_on=2.0, tau_on=1e-6, x_t=0.99, Delta_t=0.005)  # a ramp so short the pin is immediate
    blend = BlendMap(IdentityMap(6.0), ALPHA, (far,))
    X_p, _ = pinned.radii(3.0, N)
    X_b, _ = blend.radii(3.0, N)
    inside = fractions(N) <= far.inner_edge
    assert X_b[inside] == pytest.approx(X_p[inside] * math.exp(ALPHA * far.tau_on), rel=1e-12)


# --- several zones: the repeated switch-on ---


def test_a_second_zone_continues_the_map_in_value_and_velocity_and_keeps_the_identities():
    N = 400
    base = SinhStretch(12.0, scale=3.0)
    first = BlendMap(base, ALPHA, (ZONE,))
    later = Zone(xi_on=6.0, tau_on=0.3, x_t=0.5, Delta_t=0.15)
    both = first.with_zone(later)
    assert both.zones == (ZONE, later)
    # at the second switch-on the two maps agree in value and velocity: the state carries over
    X1, V1 = first.radii(later.xi_on, N)
    X2, V2 = both.radii(later.xi_on, N)
    assert np.array_equal(X1, X2)
    assert np.array_equal(V1, V2)
    # before it they are the same map
    assert np.array_equal(first.radii(5.0, N)[0], both.radii(5.0, N)[0])
    # after it: the inner zone is unchanged, the annulus is pinned from 6.0, the exterior is static, all monotone
    B, _ = base.radii(0.0, N)
    u = fractions(N)
    for xi in (6.1, 7.0, 9.0):
        X, X_xi = both.radii(xi, N)
        T1, _ = ramp(xi, ZONE.xi_on, ZONE.tau_on)
        T2, _ = ramp(xi, later.xi_on, later.tau_on)
        inner, annulus, outer = (
            u <= ZONE.inner_edge,
            (u >= ZONE.outer_edge) & (u <= later.inner_edge),
            u >= later.outer_edge,
        )
        assert X[inner] == pytest.approx(math.exp(-ALPHA * T1) * B[inner], rel=1e-15)
        assert X[annulus] == pytest.approx(math.exp(-ALPHA * T2) * B[annulus], rel=1e-15)
        assert np.array_equal(X[outer], B[outer])
        assert np.all(X_xi[outer] == 0.0)
        assert np.all(np.diff(X) > 0.0)
        assert np.all(ALPHA * X + X_xi >= 0.0)
        assert np.sum(both.weights(N), axis=0) == pytest.approx(np.ones(N + 2))
    with pytest.raises(ValueError, match="must not overlap"):
        first.with_zone(Zone(xi_on=6.0, tau_on=0.3, x_t=0.3, Delta_t=0.1))
    with pytest.raises(ValueError, match="cannot be switched on before"):
        first.with_zone(Zone(xi_on=4.0, tau_on=0.3, x_t=0.6, Delta_t=0.1))
    with pytest.raises(ValueError, match="at least one zone"):
        BlendMap(base, ALPHA, ())
    with pytest.raises(ValueError, match="must be static"):
        BlendMap(first, ALPHA, (later,))


# --- FRW through a forced switch-on (tab:numbh:tests) ---


@pytest.mark.slow
@pytest.mark.parametrize("base", BASES)
def test_frw_passes_through_a_forced_switch_on_in_deviation_form_to_round_off(base: Map):
    # Table tab:numbh:tests: FRW through a forced switch-on stays FRW to 1e-13 in the deviation form, through the
    # ramp and after it, on the moving map, with the outgoing-wave closure at the static outer face.
    N = 100
    zone = Zone(xi_on=1.0, tau_on=0.3, x_t=0.3, Delta_t=0.1)
    m = BlendMap(base, ALPHA, (zone,))
    sch = Scheme(RAD, m, Layout(N), FaceClosure.FIRST_ORDER, OutgoingWave(), CENTRED_SCHEME)
    xi_0 = 0.8
    state = frw_state(sch.frame(xi_0).geo)
    for xi_end in (1.15, 2.5):  # inside the ramp, and well after it
        final = evolve(sch, state, xi_0, xi_end)
        geo = sch.frame(xi_end).geo
        assert np.max(np.abs(final.E / geo.dV - 1.0)) < 1e-13
        assert np.max(np.abs(final.U[1:] / geo.X[1 : N + 1] - 1.0)) < 1e-13
        assert abs(final.W) < 1e-13
