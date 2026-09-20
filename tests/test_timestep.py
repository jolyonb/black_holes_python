"""Tests of pbh.timestep: the integrators, the deviation form on a moving map, the step rules, Bessel convergence."""

import math
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from modes import J1_ZEROS, BesselMode

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch
from pbh.outer import HeldAtFrw
from pbh.stencils import FaceClosure
from pbh.timestep import (
    COURANT_NUMBER,
    RK4,
    SSPRK3,
    ButcherTableau,
    Integrator,
    Scheme,
    StepLimit,
    advance,
    courant_step,
    explicit_rk_step,
    step_cap,
    step_size,
)
from pbh.types import FloatArray

EOS = EquationOfState(RADIATION)


def scheme(m: Map, N: int, j_e: int = 0) -> Scheme:
    return Scheme(EOS, m, Layout(N, j_e), FaceClosure.FIRST_ORDER, HeldAtFrw())


# --- the tableaux: order conditions, and the stability polynomials on y' = lambda y ---


def order_conditions(t: ButcherTableau) -> dict[str, Fraction]:
    """The Butcher order conditions through fourth order, evaluated exactly on the tableau."""
    n = t.stages
    a = [list(row) + [Fraction(0)] * (n - len(row)) for row in t.a]
    b, c = t.b, t.c
    ac = [sum((a[i][j] * c[j] for j in range(n)), Fraction(0)) for i in range(n)]  # (a c)_i
    ac2 = [sum((a[i][j] * c[j] ** 2 for j in range(n)), Fraction(0)) for i in range(n)]
    aac = [sum((a[i][j] * ac[j] for j in range(n)), Fraction(0)) for i in range(n)]
    return {
        "b": sum(b, Fraction(0)),
        "b c": sum((b[i] * c[i] for i in range(n)), Fraction(0)),
        "b c^2": sum((b[i] * c[i] ** 2 for i in range(n)), Fraction(0)),
        "b a c": sum((b[i] * ac[i] for i in range(n)), Fraction(0)),
        "b c^3": sum((b[i] * c[i] ** 3 for i in range(n)), Fraction(0)),
        "b c a c": sum((b[i] * c[i] * ac[i] for i in range(n)), Fraction(0)),
        "b a c^2": sum((b[i] * ac2[i] for i in range(n)), Fraction(0)),
        "b a a c": sum((b[i] * aac[i] for i in range(n)), Fraction(0)),
    }


THIRD_ORDER = {"b": Fraction(1), "b c": Fraction(1, 2), "b c^2": Fraction(1, 3), "b a c": Fraction(1, 6)}
FOURTH_ORDER = {
    **THIRD_ORDER,
    "b c^3": Fraction(1, 4),
    "b c a c": Fraction(1, 8),
    "b a c^2": Fraction(1, 12),
    "b a a c": Fraction(1, 24),
}


def test_rk4_satisfies_the_order_conditions_through_fourth_order_exactly():
    assert order_conditions(RK4) == FOURTH_ORDER


def test_ssprk3_satisfies_the_order_conditions_through_third_order_and_not_fourth():
    conditions = order_conditions(SSPRK3)
    assert {k: conditions[k] for k in THIRD_ORDER} == THIRD_ORDER
    assert conditions["b a a c"] != Fraction(1, 24)  # a three-stage method cannot meet all eight; this one it fails


def test_an_inconsistent_tableau_is_refused():
    with pytest.raises(ValueError, match="inconsistent"):
        ButcherTableau(c=(Fraction(0), Fraction(1)), a=((), (Fraction(1, 2),)), b=(Fraction(1, 2), Fraction(1, 2)))
    with pytest.raises(ValueError, match="sum to one"):
        ButcherTableau(c=(Fraction(0), Fraction(1)), a=((), (Fraction(1),)), b=(Fraction(1, 2), Fraction(1, 3)))
    with pytest.raises(ValueError, match="explicit"):
        ButcherTableau(c=(Fraction(0),), a=((Fraction(0),),), b=(Fraction(1),))
    with pytest.raises(ValueError, match="per stage"):
        ButcherTableau(c=(Fraction(0), Fraction(1)), a=((),), b=(Fraction(1),))


@pytest.mark.parametrize("lam", [-0.7, 0.3])
def test_rk4_applies_its_fourth_order_stability_polynomial(lam: float):
    y = np.array([1.0])
    z = lam * 0.4
    out = explicit_rk_step(RK4, lambda _xi, y: lam * y, 0.0, y, 0.4)
    assert out[0] == pytest.approx(1 + z + z**2 / 2 + z**3 / 6 + z**4 / 24, rel=1e-14)


def test_ssprk3_applies_its_third_order_stability_polynomial():
    lam = -0.9
    y = np.array([1.0])
    z = lam * 0.4
    out = explicit_rk_step(SSPRK3, lambda _xi, y: lam * y, 0.0, y, 0.4)
    assert out[0] == pytest.approx(1 + z + z**2 / 2 + z**3 / 6, rel=1e-14)


def test_ssprk3_is_the_shu_osher_form():
    # u1 = u + h f(u); u2 = 3/4 u + 1/4 (u1 + h f(u1)); u3 = 1/3 u + 2/3 (u2 + h f(u2)), on a nonlinear problem.
    def f(_xi: float, y: FloatArray) -> FloatArray:
        return np.sin(y)

    y, h = np.array([0.7]), 0.3
    u1 = y + h * f(0.0, y)
    u2 = 0.75 * y + 0.25 * (u1 + h * f(0.0, u1))
    u3 = y / 3 + 2 / 3 * (u2 + h * f(0.0, u2))
    assert explicit_rk_step(SSPRK3, f, 0.0, y, h) == pytest.approx(u3, rel=1e-15)


def test_the_local_error_of_one_rk4_step_is_fifth_order_on_a_nonlinear_problem():
    # y' = y^2 with y(0) = 1 has y = 1 / (1 - xi): the error of a single step falls by 2^5 when the step is halved.
    def f(_xi: float, y: FloatArray) -> FloatArray:
        return y**2

    def local_error(h: float) -> float:
        return abs(explicit_rk_step(RK4, f, 0.0, np.array([1.0]), h)[0] - 1.0 / (1.0 - h))

    assert local_error(0.02) / local_error(0.01) == pytest.approx(32.0, rel=0.05)


def test_the_integrator_enum_carries_its_tableau():
    assert Integrator.RK4.tableau is RK4
    assert Integrator.SSPRK3.tableau is SSPRK3


# --- the deviation form: FRW through a moving map to round-off (Section 7.6; tab:num:tests row 1) ---


@pytest.mark.parametrize("integrator", [Integrator.RK4, Integrator.SSPRK3])
@pytest.mark.parametrize("j_e", [0, 3])
def test_frw_stays_frw_through_the_pinned_map_in_deviation_form(integrator: Integrator, j_e: int):
    sch = scheme(PinnedMap(SinhStretch(6.0, scale=2.0), alpha=float(EOS.alpha), xi_on=0.3), 40, j_e)
    dy = np.zeros(sch.layout.size)
    xi, dxi = 0.3, 0.05
    for _ in range(20):  # one e-fold
        dy = advance(sch, integrator, xi, dy, dxi)
        xi += dxi
    scale = np.max(np.abs(sch.frw(xi)))
    assert np.max(np.abs(dy)) < 1e-13 * scale, "FRW preserved through a moving map to round-off"


def test_on_a_static_map_the_frame_is_computed_once():
    sch = scheme(SinhStretch(4.0, scale=2.0), 20)
    assert sch.frame(0.0).geo is sch.frame(1.0).geo
    assert sch.frame(0.0).w is sch.frame(1.0).w
    moving = scheme(PinnedMap(IdentityMap(4.0), alpha=0.5), 20)
    assert moving.frame(0.0).geo is not moving.frame(1.0).geo


# --- the step rules ---


def test_the_first_courant_step_on_frw_is_the_printed_one():
    # Section 7.6: on the uniform grid the first step is C_CFL Delta X / c_s(xi_0), for radiation
    # C_CFL Delta X sqrt(12) e^(-xi_0 / 2), which is 15.6 ell / N for a Gaussian of width ell on Rtilde_max = 6 ell.
    N, Rtilde_max, xi_0 = 400, 48.0, 0.0
    sch = scheme(IdentityMap(Rtilde_max), N)
    res = sch.evaluate(xi_0, sch.frw(xi_0))
    dxi = courant_step(res, sch.frame(xi_0).geo, sch.layout, COURANT_NUMBER)
    dX = Rtilde_max / N
    assert dxi == pytest.approx(COURANT_NUMBER * dX * math.sqrt(12.0) * math.exp(-xi_0 / 2))
    assert dxi == pytest.approx(15.6 * 8.0 / N, rel=1e-2)


def test_the_courant_step_is_the_shortest_cell_crossing_time():
    # On a stretch the smallest cell sits at the origin, and on FRW every signal speed is c_s.
    sch = scheme(SinhStretch(4.0, scale=1.0), 40)
    res = sch.evaluate(0.0, sch.frw(0.0))
    geo = sch.frame(0.0).geo
    dxi = courant_step(res, geo, sch.layout, COURANT_NUMBER)
    assert dxi == pytest.approx(COURANT_NUMBER * geo.dX[0] / sch.frame(0.0).bg.c_s)
    # And with the excised layout the crossing time is taken over the retained cells only.
    excised = scheme(SinhStretch(4.0, scale=1.0), 40, j_e=5)
    res_e = excised.evaluate(0.0, excised.frw(0.0))
    assert courant_step(res_e, geo, excised.layout, COURANT_NUMBER) == pytest.approx(
        COURANT_NUMBER * geo.dX[5] / sch.frame(0.0).bg.c_s
    )


def test_a_non_finite_signal_speed_is_refused():
    sch = scheme(IdentityMap(2.0), 10)
    res = sch.evaluate(0.0, sch.frw(0.0))
    geo = sch.frame(0.0).geo
    bad = np.full_like(res.speeds.Lam, np.nan)
    broken = replace(res, speeds=replace(res.speeds, Lam=bad))
    with pytest.raises(ValueError, match="no Courant step"):
        courant_step(broken, geo, sch.layout, COURANT_NUMBER)


def test_the_step_is_the_courant_step_when_cells_are_small_and_the_cap_when_they_are_large():
    # A fine grid of a small domain: the Courant step is far below the cap. A coarse grid of a huge domain at an
    # early time: the Courant step is of order one in xi and the cap binds (Section 7.6).
    cap = step_cap(EOS)
    fine = scheme(IdentityMap(2.0), 200)
    res = fine.evaluate(0.0, fine.frw(0.0))
    choice = step_size(res, fine.frame(0.0).geo, fine.layout, COURANT_NUMBER, cap)
    assert choice.limit is StepLimit.COURANT
    assert choice.dxi == pytest.approx(courant_step(res, fine.frame(0.0).geo, fine.layout, COURANT_NUMBER))
    coarse = scheme(IdentityMap(48.0), 100)  # a Gaussian of width 8 on Rtilde_max = 6 ell at N = 100
    res = coarse.evaluate(0.0, coarse.frw(0.0))
    choice = step_size(res, coarse.frame(0.0).geo, coarse.layout, COURANT_NUMBER, cap)
    assert choice.limit is StepLimit.CAP
    assert choice.dxi == cap


def test_the_step_cap_has_the_printed_values():
    assert step_cap(EOS) == pytest.approx(0.1316, abs=5e-4)  # eq:num:stepcap: kappa = 0.13 at tol = 1e-5, T_sh = 4
    assert step_cap(EOS, super_horizon_efolds=6.0) == pytest.approx(0.12, abs=5e-3)
    assert step_cap(EOS) == pytest.approx((120.0 * 1e-5 / 4.0) ** 0.25 / EOS.growing_mode_rate)


# --- convergence on the exact Bessel modes (Section 7.3; tab:num:tests row 2, base scheme) ---


def mode_errors(m: Map, k_index: int, N: int, xi_end: float = 1.0) -> tuple[float, float]:
    """The L1 errors of the cell contents and the face velocities after evolving a mode from xi = 0 to xi_end."""
    X_N = float(m.radii(0.0, N)[0][N])
    # Small enough that the scheme's nonlinear response, of relative order B, stays below the truncation error
    # being measured (at B = 1e-4 it floors the density error near 1e-5 and the rates fall off).
    mode = BesselMode(k=J1_ZEROS[k_index] / X_N, B=1e-6)
    sch = Scheme(EOS, m, Layout(N), FaceClosure.FIRST_ORDER, HeldAtFrw())
    geo = sch.frame(0.0).geo
    xi = 0.0
    dy = sch.layout.pack(mode.state(Background.at(EOS, 0.0), geo)) - sch.frw(0.0)
    while xi < xi_end - 1e-12:
        res = sch.evaluate(xi, sch.frw(xi) + dy)
        dxi = min(courant_step(res, geo, sch.layout, COURANT_NUMBER), xi_end - xi)
        dy = advance(sch, Integrator.RK4, xi, dy, dxi)
        xi += dxi
    final = sch.layout.unpack(sch.frw(xi) + dy)
    exact = mode.state(Background.at(EOS, xi), geo)
    err_E = np.sum(np.abs(final.E - exact.E)) / np.sum(np.abs(exact.E - geo.dV))
    err_U = np.sum(np.abs(final.U[1:] - exact.U[1:])) / np.sum(np.abs(exact.U[1:] - geo.X[1:]))
    return err_E, err_U


@pytest.mark.slow
@pytest.mark.parametrize("m", [IdentityMap(2.5), SinhStretch(2.5, scale=1.5)])
@pytest.mark.parametrize("k_index", [0, 1, 2])
def test_the_base_scheme_converges_at_second_order_on_the_exact_bessel_modes(m: Map, k_index: int):
    errors = [mode_errors(m, k_index, N) for N in (40, 80, 160)]
    for field in (0, 1):
        rates = [math.log2(errors[i][field] / errors[i + 1][field]) for i in range(2)]
        assert min(rates) > 1.9, f"field {field}, mode {k_index}: L1 rates {rates} (tab:num:tests asks >= 1.9)"


def test_the_bessel_mode_helper_is_self_consistent():
    # The exact contents are the integral of 1 + delta_rho, and delta_U -> k/3 at the origin by continuity.
    geo = Scheme(EOS, IdentityMap(2.5), Layout(20), FaceClosure.FIRST_ORDER, HeldAtFrw()).frame(0.0).geo
    bg = Background.at(EOS, 0.4)
    mode = BesselMode(k=J1_ZEROS[0] / 2.5, B=1e-3)
    nodes, weights = np.polynomial.legendre.leggauss(12)
    lo, hi = geo.X[:-1], geo.X[1:]
    X = 0.5 * (hi - lo)[:, None] * nodes[None, :] + 0.5 * (hi + lo)[:, None]
    quadrature = 0.5 * (hi - lo) * np.sum(weights[None, :] * X**2 * (1.0 + mode.delta_rho(bg, X)), axis=1)
    assert mode.cell_contents(bg, geo) == pytest.approx(quadrature, rel=1e-12)
    assert mode.delta_U(bg, np.array([0.0]))[0] == pytest.approx(mode.delta_U(bg, np.array([1e-6]))[0], rel=1e-6)
    assert mode.delta_U(bg, np.array([2.5]))[0] == pytest.approx(0.0, abs=1e-15)  # j_1(k X_N) = 0
