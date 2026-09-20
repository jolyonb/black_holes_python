"""Tests of pbh.derived: the FRW identities, the origin identities of Section 7.2, and the hyperbolicity abort."""

import math
from collections.abc import Callable

import numpy as np
import pytest

from pbh.derived import NotHyperbolicError, derive
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, SinhStretch
from pbh.state import State, frw_state
from pbh.stencils import FaceClosure, StencilWeights

EOS = EquationOfState(RADIATION)
type Family = Callable[[float], Map]
FAMILIES: list[Family] = [IdentityMap, lambda R: SinhStretch(R, scale=2.0)]


def setup(
    family: Family, N: int, Rtilde_max: float, xi: float = 0.6, j_e: int = 0
) -> tuple[Geometry, Background, StencilWeights]:
    geo = Geometry.of(*family(Rtilde_max).radii(xi, N))
    return geo, Background.at(EOS, xi), StencilWeights.of(geo, Layout(N, j_e), FaceClosure.FIRST_ORDER)


# --- FRW ---


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("j_e", [0, 3])
def test_on_frw_every_derived_field_has_its_frw_value(family: Family, j_e: int):
    geo, bg, w = setup(family, 40, 6.0, j_e=j_e)
    d = derive(frw_state(geo, j_e), geo, bg, EOS, w)
    cells, faces = w.layout.cells, w.layout.faces
    assert d.rho[cells] == pytest.approx(1.0, rel=1e-14)
    assert d.ephi[cells] == pytest.approx(1.0, rel=1e-14)
    assert d.M[faces] == pytest.approx(geo.X[faces] ** 3, rel=1e-13)
    assert d.mt[max(j_e, 1) :] == pytest.approx(1.0, rel=1e-13)
    assert d.Gammabar2[faces] == pytest.approx(bg.Gammabar2, rel=1e-13)


def test_entries_below_the_excision_face_are_nan_and_the_origin_mt_is_nan():
    geo, bg, w = setup(IdentityMap, 10, 2.0, j_e=3)
    d = derive(frw_state(geo, 3), geo, bg, EOS, w)
    for name in ("rho", "ephi"):
        assert np.all(np.isnan(getattr(d, name)[:3]))
        assert np.all(np.isfinite(getattr(d, name)[3:]))
    for name in ("M", "mt", "Gammabar2", "rho_f", "ephi_f"):
        assert np.all(np.isnan(getattr(d, name)[:3]))
        assert np.all(np.isfinite(getattr(d, name)[3:]))
    geo0, bg0, lay0 = setup(IdentityMap, 10, 2.0)
    d0 = derive(frw_state(geo0), geo0, bg0, EOS, lay0)
    assert math.isnan(d0.mt[0])
    assert d0.M[0] == 0.0
    assert d0.Gammabar2[0] == bg0.Gammabar2


# --- the origin identities of Section 7.2 ---


def test_the_innermost_face_mass_is_the_innermost_density_identically():
    # mt_1 = rho_0: M_1 = 3 E_0 = 3 rho_0 dV_0 and dV_0 = X_1^3 / 3 exactly, so mt_1 = rho_0 (Section 7.2).
    geo, bg, w = setup(lambda R: SinhStretch(R, scale=1.0), 12, 3.0)
    s = frw_state(geo)
    E = s.E.copy()
    E[0] *= 1.37
    d = derive(State(E=E, U=s.U, W=0.0), geo, bg, EOS, w)
    assert d.mt[1] == pytest.approx(d.rho[0], rel=1e-15)


@pytest.mark.parametrize("family", FAMILIES)
def test_exact_contents_of_an_even_density_give_the_printed_mass_exactly(family: Family):
    # Section 7.2: for rho = 1 + r2 X^2 + r4 X^4, exact cell contents give mt_j = 1 + (3/5) r2 X_j^2 + (3/7) r4 X_j^4,
    # an identity of the cumulative sum, to round-off.
    geo, bg, w = setup(family, 30, 2.0)
    r2, r4 = 0.02, -0.003  # small enough that Gammabar^2 stays positive out to the outer face
    X = np.append(geo.X, 0.0)  # a dummy beyond N; only differences between faces 0..N are used
    antiderivative = X**3 / 3 + r2 * X**5 / 5 + r4 * X**7 / 7  # int X^2 rho dX
    E = np.diff(antiderivative[: w.layout.N + 1])
    d = derive(State(E=E, U=geo.X.copy(), W=0.0), geo, bg, EOS, w)
    X_faces = geo.X[1:]
    assert d.mt[1:] == pytest.approx(1 + 0.6 * r2 * X_faces**2 + (3 / 7) * r4 * X_faces**4, rel=1e-13)


def test_the_cumulative_sum_is_accurate_to_round_off_at_large_n():
    # Section 7.2: a sum of positive terms, accurate to 1e-15 at any N. Measured against a compensated sum: at most
    # 1.1e-15 relative at N = 500 and 2000 and 3.4e-15 at N = 4000, a few ulps, far below the N eps a sum could reach.
    geo, bg, w = setup(IdentityMap, 4000, 20.0)
    rng = np.random.default_rng(7)
    E = geo.dV * rng.uniform(0.999, 1.001, size=w.layout.N)  # near FRW, so Gammabar^2 stays positive at X = 20
    d = derive(State(E=E, U=geo.X.copy(), W=0.0), geo, bg, EOS, w)
    exact = np.array([3.0 * math.fsum(E[:j]) for j in range(1, w.layout.N + 1)])
    assert np.max(np.abs(d.M[1:] / exact - 1.0)) < 5e-15


# --- excision bookkeeping ---


def test_after_excision_the_mass_starts_at_m_e_and_sums_only_retained_cells():
    geo, bg, w = setup(IdentityMap, 10, 2.0, j_e=4)
    s = frw_state(geo, 4)
    M_e = 2.5 * s.M_e
    d = derive(State(E=s.E, U=s.U, W=0.0, M_e=M_e), geo, bg, EOS, w)
    assert d.M[4] == M_e
    assert d.M[5:] == pytest.approx(M_e + 3.0 * np.cumsum(s.E[4:]))
    assert d.mt[4] == pytest.approx(M_e / geo.X[4] ** 3)


# --- the hyperbolicity abort ---


def test_a_non_positive_density_aborts_naming_the_cell():
    geo, bg, w = setup(IdentityMap, 10, 2.0)
    s = frw_state(geo)
    E = s.E.copy()
    E[6] = -1e-3
    with pytest.raises(NotHyperbolicError, match=r"rho\[6\]") as info:
        derive(State(E=E, U=s.U, W=0.0), geo, bg, EOS, w)
    assert (info.value.field, info.value.index) == ("rho", 6)
    assert info.value.value == pytest.approx(-1e-3 / geo.dV[6])


def test_a_non_positive_gammabar2_aborts_naming_the_face():
    geo, bg, w = setup(IdentityMap, 10, 2.0)
    s = frw_state(geo)
    E = s.E.copy()
    E[2] *= 400.0  # far too much mass inside face 3: M_3 / X_3 exceeds e^xi + U_3^2
    with pytest.raises(NotHyperbolicError, match=r"Gammabar2\[3\]") as info:
        derive(State(E=E, U=s.U, W=0.0), geo, bg, EOS, w)
    assert (info.value.field, info.value.index) == ("Gammabar2", 3)


def test_a_nan_in_a_retained_cell_aborts_rather_than_passing_silently():
    geo, bg, w = setup(IdentityMap, 10, 2.0)
    s = frw_state(geo)
    E = s.E.copy()
    E[1] = np.nan
    with pytest.raises(NotHyperbolicError, match=r"rho\[1\]"):
        derive(State(E=E, U=s.U, W=0.0), geo, bg, EOS, w)


def test_excised_entries_do_not_trigger_the_abort():
    geo, bg, w = setup(IdentityMap, 10, 2.0, j_e=3)
    s = Layout(10, 3).unpack(Layout(10, 3).pack(frw_state(geo, 3)))  # NaN below j_e
    derive(s, geo, bg, EOS, w)  # no exception
