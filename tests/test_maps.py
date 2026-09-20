"""Tests of pbh.maps: the label grid, the admissibility conditions and the derivatives of each map."""

import numpy as np
import pytest

from pbh.geometry import Geometry
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch, face_labels

# --- the label grid ---


def test_face_labels_are_the_uniform_grid_with_the_virtual_face():
    N, x_max = 5, 2.5
    x = face_labels(N, x_max)
    assert x.shape == (N + 2,)
    assert x[0] == 0.0
    assert x[N] == x_max
    assert x[N + 1] == pytest.approx(x_max + x_max / N)
    assert np.allclose(np.diff(x), x_max / N)


@pytest.mark.parametrize(("N", "x_max"), [(1, 1.0), (0, 1.0), (4, 0.0), (4, -1.0)])
def test_a_degenerate_grid_is_refused(N: int, x_max: float):
    with pytest.raises(ValueError, match="N >= 2"):
        face_labels(N, x_max)


# --- every map: admissibility, and the derivatives against finite differences ---

MAPS: list[Map] = [
    IdentityMap(),
    SinhStretch(scale=3.0),
    SinhStretch(scale=0.7),
    PinnedMap(IdentityMap(), alpha=0.5),
    PinnedMap(SinhStretch(scale=2.0), alpha=0.4),
]


@pytest.mark.parametrize("m", MAPS)
def test_the_map_vanishes_at_the_origin_exactly_and_increases(m: Map):
    x = face_labels(40, 8.0)
    for xi in (0.0, 1.3):
        X, _, X_x = m.at(xi, x)
        assert X[0] == 0.0
        assert np.all(np.diff(X) > 0.0)
        assert np.all(X_x > 0.0)


@pytest.mark.parametrize("m", MAPS)
def test_the_label_derivative_matches_a_central_difference(m: Map):
    x = face_labels(40, 8.0)
    eps = 1e-6
    _, _, X_x = m.at(0.9, x)
    X_plus, _, _ = m.at(0.9, x + eps)
    X_minus, _, _ = m.at(0.9, x - eps)
    assert X_x == pytest.approx((X_plus - X_minus) / (2 * eps), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("m", MAPS)
def test_the_time_derivative_matches_a_central_difference(m: Map):
    x = face_labels(40, 8.0)
    eps = 1e-6
    _, X_xi, _ = m.at(0.9, x)
    X_plus, _, _ = m.at(0.9 + eps, x)
    X_minus, _, _ = m.at(0.9 - eps, x)
    assert X_xi == pytest.approx((X_plus - X_minus) / (2 * eps), rel=1e-8, abs=1e-8)


@pytest.mark.parametrize("m", MAPS)
def test_every_map_feeds_the_geometry(m: Map):
    geo = Geometry.of(*m.at(0.4, face_labels(16, 4.0)))
    assert geo.N == 16


# --- the particular maps ---


def test_the_identity_map_is_static_and_the_identity():
    m = IdentityMap()
    x = face_labels(6, 3.0)
    X, X_xi, X_x = m.at(2.0, x)
    assert m.is_static
    assert np.array_equal(X, x)
    assert np.all(X_xi == 0.0)
    assert np.all(X_x == 1.0)
    assert X is not x  # a copy, so the caller's labels are never aliased


def test_the_sinh_stretch_is_the_paper_s_and_is_finest_at_the_origin():
    m = SinhStretch(scale=3.0)
    x = face_labels(100, 9.0)
    X, X_xi, X_x = m.at(0.0, x)
    assert m.is_static
    assert X == pytest.approx(3.0 * np.sinh(x / 3.0))
    assert np.all(X_xi == 0.0)
    assert X_x[0] == 1.0
    assert np.all(np.diff(X_x) > 0.0)  # the Jacobian grows outward: cells are finest at the centre
    assert np.all(np.diff(np.diff(X)) > 0.0)  # and the cell widths in X increase outward


def test_a_non_positive_stretch_scale_is_refused():
    with pytest.raises(ValueError, match="positive"):
        SinhStretch(scale=0.0)


@pytest.mark.parametrize("base", [IdentityMap(), SinhStretch(scale=2.0)])
@pytest.mark.parametrize("xi_on", [0.0, 3.9])
def test_the_pinned_map_is_its_base_at_the_pin_on_time_and_holds_the_physical_radius_after(base: Map, xi_on: float):
    alpha = 0.5
    m = PinnedMap(base, alpha=alpha, xi_on=xi_on)
    x = face_labels(10, 5.0)
    B, _, B_x = base.at(0.0, x)
    assert not m.is_static
    X_on, X_xi_on, X_x_on = m.at(xi_on, x)
    assert np.array_equal(X_on, B)  # the state on the base carries over unchanged at the switch
    assert np.array_equal(X_x_on, B_x)
    assert X_xi_on == pytest.approx(-alpha * B)  # but the faces start moving at once
    for xi in (xi_on + 0.7, xi_on + 2.1):
        X, X_xi, X_x = m.at(xi, x)
        assert np.exp(alpha * xi) * X == pytest.approx(np.exp(alpha * xi_on) * B)  # Rbar constant in time
        assert X_xi == pytest.approx(-alpha * X)
        assert X_x == pytest.approx(np.exp(-alpha * (xi - xi_on)) * B_x)


def test_only_a_static_map_can_be_pinned():
    with pytest.raises(ValueError, match="static"):
        PinnedMap(PinnedMap(IdentityMap(), alpha=0.5), alpha=0.5)
