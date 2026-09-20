"""Tests of pbh.maps: the face fractions, the admissibility conditions, the face velocities, and each map."""

import numpy as np
import pytest

from pbh.geometry import Geometry
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch, fractions

# --- the face fractions ---


def test_fractions_run_from_the_origin_to_the_virtual_face():
    N = 5
    u = fractions(N)
    assert u.shape == (N + 2,)
    assert u[0] == 0.0
    assert u[N] == 1.0
    assert u[N + 1] == pytest.approx(1.0 + 1.0 / N)
    assert np.allclose(np.diff(u), 1.0 / N)


@pytest.mark.parametrize("N", [0, 1])
def test_a_degenerate_grid_is_refused(N: int):
    with pytest.raises(ValueError, match="N >= 2"):
        fractions(N)


# --- every map: admissibility, the outer radius, and the velocity against a central difference in time ---

MAPS: list[Map] = [
    IdentityMap(4.0),
    SinhStretch(4.0, scale=3.0),
    SinhStretch(4.0, scale=0.7),
    PinnedMap(IdentityMap(4.0), alpha=0.5),
    PinnedMap(SinhStretch(4.0, scale=2.0), alpha=0.4, xi_on=1.1),
]


@pytest.mark.parametrize("m", MAPS)
def test_the_map_vanishes_at_the_origin_exactly_and_increases(m: Map):
    for xi in (0.0, 1.3):
        X, _ = m.radii(xi, 40)
        assert X[0] == 0.0
        assert np.all(np.diff(X) > 0.0)


@pytest.mark.parametrize("m", MAPS)
def test_the_face_velocity_matches_a_central_difference_in_time(m: Map):
    eps = 1e-6
    _, X_xi = m.radii(0.9, 40)
    X_plus, _ = m.radii(0.9 + eps, 40)
    X_minus, _ = m.radii(0.9 - eps, 40)
    assert X_xi == pytest.approx((X_plus - X_minus) / (2 * eps), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("m", MAPS)
def test_every_map_feeds_the_geometry(m: Map):
    geo = Geometry.of(*m.radii(0.4, 16))
    assert geo.N == 16


@pytest.mark.parametrize("m", MAPS[:3])
def test_a_static_map_puts_the_outer_face_at_rtilde_max(m: Map):
    X, _ = m.radii(0.0, 25)
    assert X[25] == pytest.approx(4.0, rel=1e-15)


# --- the particular maps ---


def test_the_identity_map_is_static_and_uniform_in_the_radius():
    m = IdentityMap(3.0)
    X, X_xi = m.radii(2.0, 6)
    assert m.is_static
    assert X == pytest.approx(3.0 * np.arange(8) / 6)
    assert np.all(X_xi == 0.0)


def test_the_sinh_stretch_is_finest_at_the_origin_and_reaches_rtilde_max():
    m = SinhStretch(9.0, scale=3.0)
    X, X_xi = m.radii(0.0, 100)
    assert m.is_static
    assert X[100] == pytest.approx(9.0, rel=1e-15)
    assert np.all(X_xi == 0.0)
    assert np.all(np.diff(np.diff(X)) > 0.0)  # the cell widths in X increase outward: finest at the centre
    # Inside the scale the map is nearly uniform: X_1 = L sinh(asinh(3) / N) is L asinh(3) / N to relative order
    # (asinh(3) / N)^2.
    assert X[1] == pytest.approx(3.0 * np.arcsinh(3.0) / 100, rel=1e-3)


@pytest.mark.parametrize(("Rtilde_max", "scale"), [(0.0, 1.0), (4.0, 0.0), (-1.0, 1.0)])
def test_non_positive_lengths_are_refused(Rtilde_max: float, scale: float):
    with pytest.raises(ValueError, match="positive"):
        SinhStretch(Rtilde_max, scale=scale)
    if scale > 0.0:
        with pytest.raises(ValueError, match="positive"):
            IdentityMap(Rtilde_max)


@pytest.mark.parametrize("base", [IdentityMap(5.0), SinhStretch(5.0, scale=2.0)])
@pytest.mark.parametrize("xi_on", [0.0, 3.9])
def test_the_pinned_map_is_its_base_at_the_pin_on_time_and_holds_the_physical_radius_after(base: Map, xi_on: float):
    alpha = 0.5
    m = PinnedMap(base, alpha=alpha, xi_on=xi_on)
    B, _ = base.radii(0.0, 10)
    assert not m.is_static
    X_on, X_xi_on = m.radii(xi_on, 10)
    assert np.array_equal(X_on, B)  # the state on the base carries over unchanged at the switch
    assert X_xi_on == pytest.approx(-alpha * B)  # but the faces start moving at once
    for xi in (xi_on + 0.7, xi_on + 2.1):
        X, X_xi = m.radii(xi, 10)
        assert np.exp(alpha * xi) * X == pytest.approx(np.exp(alpha * xi_on) * B)  # Rbar constant in time
        assert X_xi == pytest.approx(-alpha * X)


def test_only_a_static_map_can_be_pinned():
    with pytest.raises(ValueError, match="static"):
        PinnedMap(PinnedMap(IdentityMap(1.0), alpha=0.5), alpha=0.5)
