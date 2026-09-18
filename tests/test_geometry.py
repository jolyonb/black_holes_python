"""Tests of pbh.geometry against exact rational integrals and the round-off claims of Section 7.2."""

import math
from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

from pbh.geometry import Geometry
from pbh.types import FloatArray

type MapAtFaces = tuple[FloatArray, FloatArray, FloatArray]
type MakeMap = Callable[[int, float], MapAtFaces]

# --- exact integrals: the printed forms of eq:num:geom equal int X^2 dX and int X^4 dX / int X^2 dX ---


def exact_shell_volume(a: Fraction, b: Fraction) -> Fraction:
    return (b**3 - a**3) / 3


def exact_mean_square_radius(a: Fraction, b: Fraction) -> Fraction:
    return (b**5 - a**5) / 5 / exact_shell_volume(a, b)


RATIONAL_SHELLS = [
    (Fraction(0), Fraction(1, 7)),  # the innermost cell, starting at the origin
    (Fraction(1, 7), Fraction(2, 7)),
    (Fraction(3, 2), Fraction(31, 20)),  # a thin shell far out, where the cancellation is worst
    (Fraction(12), Fraction(2401, 200)),
]


def one_shell(a: Fraction, b: Fraction) -> Geometry:
    """A grid whose cell 1 (cell 0 if a = 0) is the shell from a to b, plus the virtual cell beyond it."""
    X = (
        np.array([0.0, float(a), float(b), 2 * float(b) - float(a)])
        if a != 0
        else np.array([0.0, float(b), 2 * float(b)])
    )
    return Geometry.of(X, np.zeros_like(X), np.ones_like(X))


@pytest.mark.parametrize(("a", "b"), RATIONAL_SHELLS)
def test_the_printed_volume_and_mean_square_radius_are_the_exact_integrals(a: Fraction, b: Fraction):
    geo = one_shell(a, b)
    c = 1 if a != 0 else 0  # the cell from a to b
    assert geo.dV[c] == pytest.approx(float(exact_shell_volume(a, b)), rel=1e-15)
    assert geo.sbar[c] == pytest.approx(float(exact_mean_square_radius(a, b)), rel=1e-15)


@pytest.mark.parametrize(("a", "b"), RATIONAL_SHELLS)
def test_a_field_linear_in_s_has_its_shell_average_at_sbar_exactly(a: Fraction, b: Fraction):
    # The property Section 7.2 rests on: the average of f = f0 + f1 X^2 under X^2 dX is f(sbar), in exact arithmetic.
    f0, f1 = Fraction(3, 5), Fraction(-7, 11)
    average = (f0 * (b**3 - a**3) / 3 + f1 * (b**5 - a**5) / 5) / exact_shell_volume(a, b)
    assert average == f0 + f1 * exact_mean_square_radius(a, b)


# --- the assembled geometry on a grid ---


def identity_map(N: int, x_max: float) -> MapAtFaces:
    """X = x at the N + 2 labels 0 .. (N + 1) h: the pre-formation map, static."""
    X = np.linspace(0.0, x_max, N + 1)
    X = np.append(X, X[-1] + X[1])
    return X, np.zeros_like(X), np.ones_like(X)


def sinh_map(N: int, x_max: float) -> MapAtFaces:
    """X = 3 sinh(x / 3), the static stretch the paper verifies FRW on (Section 7.3)."""
    x = np.linspace(0.0, x_max, N + 1)
    x = np.append(x, x[-1] + x[1])
    return 3.0 * np.sinh(x / 3.0), np.zeros_like(x), np.cosh(x / 3.0)


@pytest.mark.parametrize("make_map", [identity_map, sinh_map])
def test_shapes_and_the_two_conventions(make_map: MakeMap):
    N = 8
    geo = Geometry.of(*make_map(N, 4.0))
    assert geo.N == N
    assert geo.X.shape == geo.X_xi.shape == geo.X_x.shape == (N + 1,)
    assert geo.dV.shape == geo.dV_xi.shape == geo.dX.shape == geo.Xm.shape == (N,)
    assert geo.sbar.shape == (N + 1,)  # the virtual outer cell rides along
    assert geo.dS.shape == (N + 1,)
    assert math.isnan(geo.dS[0])  # no gradient is ever formed at the origin
    assert np.all(np.isfinite(geo.dS[1:]))
    assert geo.X[0] == 0.0
    assert np.all(geo.dS[1:] > 0.0)
    assert np.all(geo.dX > 0.0)


@pytest.mark.parametrize("make_map", [identity_map, sinh_map])
def test_the_virtual_cell_enters_only_through_dS_at_the_outer_face(make_map: MakeMap):
    N = 8
    X, X_xi, X_x = make_map(N, 4.0)
    geo = Geometry.of(X, X_xi, X_x)
    a, b = Fraction(X[-2]), Fraction(X[-1])
    assert geo.sbar[N] == pytest.approx(float(exact_mean_square_radius(a, b)), rel=1e-14)
    assert geo.dS[N] == geo.sbar[N] - geo.sbar[N - 1]
    # Nothing else changes if the virtual face moves.
    X_moved = X.copy()
    X_moved[-1] *= 1.5
    geo_moved = Geometry.of(X_moved, X_xi, X_x)
    for name in ("X", "dV", "dV_xi", "dX", "Xm"):
        assert np.array_equal(getattr(geo, name), getattr(geo_moved, name))
    assert np.array_equal(geo.sbar[:-1], geo_moved.sbar[:-1])
    assert np.array_equal(geo.dS[1:-1], geo_moved.dS[1:-1])


@pytest.mark.parametrize("make_map", [identity_map, sinh_map])
def test_the_cumulative_volume_is_the_cube_of_the_face_radius(make_map: MakeMap):
    # 3 sum_{i<j} Delta V_i = X_j^3 is the FRW mass M_j = X_j^3 (Section 7.3); the paper verifies it to round-off.
    geo = Geometry.of(*make_map(2000, 20.0))
    M_frw = 3.0 * np.concatenate(([0.0], np.cumsum(geo.dV)))
    assert np.max(np.abs(M_frw[1:] / geo.X[1:] ** 3 - 1.0)) < 1e-13


def test_round_off_of_the_geometry_at_n_2000_is_at_the_quoted_floors():
    # Section 7.2: at N = 2000 the geometry carries about 3e-13 in Delta V and 5e-13 in Delta S relative to exact
    # arithmetic, growing like N eps toward the outer edge; the assertion is at the quoted floors with a margin of two.
    N, x_max = 2000, 20.0
    geo = Geometry.of(*identity_map(N, x_max))
    h = Fraction(x_max) / N
    faces = [h * j for j in range(N + 2)]
    dV_exact = np.array([float(exact_shell_volume(faces[c], faces[c + 1])) for c in range(N)])
    sbar_exact = [exact_mean_square_radius(faces[c], faces[c + 1]) for c in range(N + 1)]
    dS_exact = np.array([float(sbar_exact[j] - sbar_exact[j - 1]) for j in range(1, N + 1)])
    assert np.max(np.abs(geo.dV / dV_exact - 1.0)) < 6e-13
    assert np.max(np.abs(geo.dS[1:] / dS_exact - 1.0)) < 1e-12


def test_the_volume_rate_on_a_self_similar_moving_map_is_exact():
    # On X = e^(-alpha xi) x every radius scales together, so Delta V scales as e^(-3 alpha xi) and
    # d_xi Delta V = -3 alpha Delta V exactly.
    alpha = 0.5
    X, _, X_x = sinh_map(50, 6.0)
    geo = Geometry.of(X, -alpha * X, X_x)
    assert geo.dV_xi == pytest.approx(-3.0 * alpha * geo.dV, rel=1e-14)


def test_a_static_map_has_no_volume_rate():
    geo = Geometry.of(*sinh_map(50, 6.0))
    assert np.all(geo.dV_xi == 0.0)


def test_the_midpoint_and_width_are_what_they_say():
    X, X_xi, X_x = sinh_map(5, 2.0)
    geo = Geometry.of(X, X_xi, X_x)
    assert np.array_equal(geo.dX, X[1:-1] - X[:-2])
    assert np.array_equal(geo.Xm, 0.5 * (X[1:-1] + X[:-2]))


def test_an_inadmissible_map_is_refused():
    X, X_xi, X_x = identity_map(4, 1.0)
    with pytest.raises(ValueError, match="X_0 = 0"):
        Geometry.of(X + 1e-3, X_xi, X_x)
    X_folded = X.copy()
    X_folded[2] = X_folded[3]
    with pytest.raises(ValueError, match="increase strictly"):
        Geometry.of(X_folded, X_xi, X_x)
