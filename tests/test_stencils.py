"""Tests of pbh.stencils: exactness, orders at the first faces, and the excision rows."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, SinhStretch
from pbh.stencils import FaceClosure, StencilWeights
from pbh.types import FloatArray

O1, O2 = FaceClosure.FIRST_ORDER, FaceClosure.SECOND_ORDER
type Family = Callable[[float], Map]
FAMILIES: list[Family] = [IdentityMap, lambda R: SinhStretch(R, scale=1.5)]


def geometry(family: Family, N: int, Rtilde_max: float) -> Geometry:
    return Geometry.of(*family(Rtilde_max).radii(0.0, N))


def weights(geo: Geometry, j_e: int = 0, closure: FaceClosure = O1) -> StencilWeights:
    return StencilWeights.of(geo, Layout(geo.N, j_e), closure)


def shell_averages(f_of_X: Callable[[FloatArray], FloatArray], geo: Geometry) -> FloatArray:
    """The exact shell averages of f under X^2 dX, by 16-node Gauss-Legendre quadrature on every cell."""
    nodes, weights = np.polynomial.legendre.leggauss(16)
    lo, hi = geo.X[:-1], geo.X[1:]
    X = 0.5 * (hi - lo)[:, None] * nodes[None, :] + 0.5 * (hi + lo)[:, None]
    integrals: FloatArray = 0.5 * (hi - lo) * np.sum(weights[None, :] * X**2 * f_of_X(X), axis=1)
    return integrals / geo.dV


# --- exactness ---


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("closure", [O1, O2])
def test_the_gradient_in_s_is_exact_on_a_field_linear_in_s(family: Family, closure: FaceClosure):
    geo = geometry(family, 12, 3.0)
    a, b = 0.7, -0.4
    f = a + b * geo.sbar[:-1]  # the shell average of a + b X^2 is its value at sbar, exactly
    for j_e in (0, 3):
        grad = weights(geo, j_e, closure).gradient_s(f)
        expected = 2.0 * b * geo.X  # d/dX (a + b X^2)
        assert grad[j_e + 1 : 12] == pytest.approx(expected[j_e + 1 : 12], rel=1e-12)
        assert grad[j_e] == (0.0 if (j_e == 0 or closure is O1) else pytest.approx(expected[j_e], rel=1e-12))
        assert math.isnan(grad[12])


@pytest.mark.parametrize("family", FAMILIES)
def test_the_second_order_face_value_is_exact_on_a_field_linear_in_s_and_the_first_order_one_is_the_cell(
    family: Family,
):
    geo = geometry(family, 12, 3.0)
    a, b = 0.7, -0.4
    f = a + b * geo.sbar[:-1]
    assert weights(geo, 3, O2).face_average(f)[3] == pytest.approx(a + b * geo.X[3] ** 2, rel=1e-12)
    assert weights(geo, 3, O1).face_average(f)[3] == f[3]


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("closure", [O1, O2])
def test_every_velocity_row_gives_one_on_u_equals_x(family: Family, closure: FaceClosure):
    # (D_U X)_j = 1 exactly is what makes FRW exact (Section 7.3), at every row including the one-sided ones.
    geo = geometry(family, 12, 3.0)
    for j_e in (0, 3):
        grad = weights(geo, j_e, closure).velocity_gradient(geo.X)
        assert grad[max(j_e, 1) :] == pytest.approx(1.0, rel=1e-12)
        assert math.isnan(grad[0])


@pytest.mark.parametrize("family", FAMILIES)
def test_the_three_point_rows_are_exact_on_a_quadratic_and_the_first_order_row_is_not(family: Family):
    geo = geometry(family, 12, 3.0)
    X = geo.X
    U = 0.3 - 1.1 * X + 0.8 * X**2
    dU = -1.1 + 1.6 * X
    w2, w1 = weights(geo, 3, O2), weights(geo, 3, O1)
    assert w2.velocity_gradient(U)[12] == pytest.approx(dU[12], rel=1e-12)  # the outer row
    assert w2.velocity_gradient(U)[3] == pytest.approx(dU[3], rel=1e-12)  # its mirror at j_e
    assert w1.velocity_gradient(U)[3] == pytest.approx((U[4] - U[3]) / geo.dX[3])  # one difference
    assert w1.velocity_gradient(U)[3] != pytest.approx(dU[3], rel=1e-6)


# --- the two-cell average and the origin ---


def test_the_face_value_is_the_two_cell_average_with_the_printed_end_rows():
    geo = geometry(IdentityMap, 6, 3.0)
    f = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    avg = weights(geo).face_average(f)
    assert avg[0] == 1.0  # <f>_0 = f_0
    assert np.array_equal(avg[1:6], 0.5 * (f[:-1] + f[1:]))
    assert avg[6] == 1.5 * 32.0 - 0.5 * 16.0  # <f>_N = 3/2 f_{N-1} - 1/2 f_{N-2}


def test_entries_below_the_excision_face_are_nan():
    geo = geometry(IdentityMap, 8, 2.0)
    w = weights(geo, 3)
    f = np.ones(8)
    for out in (w.face_average(f), w.gradient_s(f), w.velocity_gradient(geo.X)):
        assert np.all(np.isnan(out[:3]))


# --- orders of accuracy (Section 7.3: second order uniformly to the origin in s; a difference in X is first order) ---


def face_errors(family: Family, N: int, face: int) -> tuple[float, float]:
    """Errors of the gradient at one face for a smooth even field: in s (the scheme) and in X (the naive stencil)."""
    geo = geometry(family, N, 2.0)
    f = shell_averages(lambda X: np.cos(1.3 * X), geo)
    exact = -1.3 * np.sin(1.3 * geo.X[face])
    in_s = weights(geo).gradient_s(f)[face]
    in_X = (f[face] - f[face - 1]) / (geo.Xm[face] - geo.Xm[face - 1])
    return abs(in_s - exact), abs(in_X - exact)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("face", [1, 2, 3, 4])
def test_the_gradient_in_s_is_second_order_down_to_the_first_face(family: Family, face: int):
    e_s = [face_errors(family, N, face)[0] for N in (20, 40, 80)]
    rate_s = np.log2(e_s[0] / e_s[1]), np.log2(e_s[1] / e_s[2])
    assert min(rate_s) > 1.9, f"gradient in s at face {face}: rates {rate_s}"


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("face", [1, 2])
def test_the_gradient_in_x_is_only_first_order_at_the_first_faces(family: Family, face: int):
    # Section 7.3: an even field is not linear in X near the origin, so a difference of shell averages in X is first
    # order at the first faces whatever the cell radius; further out the origin no longer dominates its error.
    e_X = [face_errors(family, N, face)[1] for N in (20, 40, 80)]
    rate_X = np.log2(e_X[0] / e_X[1]), np.log2(e_X[1] / e_X[2])
    assert max(rate_X) < 1.5, f"gradient in X at face {face}: rates {rate_X} (expected first order)"
