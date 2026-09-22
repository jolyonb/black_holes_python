"""Tests of pbh.equations and pbh.outer: FRW exactness, the telescoping mass law, reparametrisation, and the closure."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.equations import calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch
from pbh.outer import HeldAtFrw, OuterClosure, OuterInputs, OuterRows
from pbh.state import State, frw_rate, frw_state
from pbh.stencils import FaceClosure, StencilWeights

EOS = EquationOfState(RADIATION)
HELD = HeldAtFrw()
type Family = Callable[[float], Map]
STATIC_FAMILIES: list[Family] = [IdentityMap, lambda R: SinhStretch(R, scale=2.0)]


@dataclass(frozen=True)
class Setup:
    geo: Geometry
    bg: Background
    w: StencilWeights

    @classmethod
    def of(cls, m: Map, N: int, xi: float, j_e: int = 0, closure: FaceClosure = FaceClosure.FIRST_ORDER):
        geo = Geometry.of(*m.radii(xi, N))
        return cls(geo, Background.at(EOS, xi), StencilWeights.of(geo, Layout(N, j_e), closure))

    def run(self, s: State, outer: OuterClosure = HELD):
        return calc_derivs(s, self.geo, self.bg, EOS, self.w, outer, CENTRED_SCHEME)


def smooth_state(su: Setup, amplitude: float = 0.02, seed: int = 0) -> State:
    """A smooth perturbation of FRW that stays hyperbolic: an even bump in the density, an odd one in the velocity."""
    rng = np.random.default_rng(seed)
    geo, lay = su.geo, su.w.layout
    c1, c2 = rng.uniform(0.5, 1.5, size=2)
    E = geo.dV * (1.0 + amplitude * np.exp(-c1 * geo.sbar[:-1]))
    U = geo.X * (1.0 + amplitude * np.exp(-c2 * geo.X**2))
    return State(E=E, U=U, W=0.0, M_e=float(geo.X[lay.j_e]) ** 3)


# --- FRW: a fixed point on every static map, the exact solution on every moving one (Section 7.3) ---


@pytest.mark.parametrize("family", STATIC_FAMILIES)
@pytest.mark.parametrize("j_e", [0, 4])
def test_frw_is_a_fixed_point_on_a_static_map(family: Family, j_e: int):
    su = Setup.of(family(6.0), 40, xi=0.8, j_e=j_e)
    r = su.run(frw_state(su.geo, j_e)).rate
    lay = su.w.layout
    assert np.max(np.abs(r.E[lay.cells])) < 1e-12 * np.max(su.geo.dV)
    assert np.max(np.abs(r.U[lay.faces])) < 1e-12 * np.max(su.geo.X)
    assert r.W == 0.0
    if j_e == 0:
        assert r.M_e == 0.0
    else:
        assert abs(r.M_e) < 1e-12 * su.geo.X[j_e] ** 3  # (2 - 3 alpha) X_e^3 - 3 alpha w X_e^3 cancels to round-off


@pytest.mark.parametrize("family", STATIC_FAMILIES)
@pytest.mark.parametrize("j_e", [0, 4])
def test_frw_is_the_exact_solution_on_a_moving_map(family: Family, j_e: int):
    m = PinnedMap(family(6.0), alpha=float(EOS.alpha), xi_on=0.3)
    su = Setup.of(m, 40, xi=0.8, j_e=j_e)
    r = su.run(frw_state(su.geo, j_e)).rate
    expected = frw_rate(su.geo, j_e)
    lay = su.w.layout
    assert r.E[lay.cells] == pytest.approx(expected.E[lay.cells], rel=1e-12)
    assert r.U[lay.faces] == pytest.approx(expected.U[lay.faces], rel=1e-12)
    assert r.M_e == pytest.approx(expected.M_e, rel=1e-12)


def test_on_frw_the_flux_and_face_fields_have_their_printed_values():
    su = Setup.of(SinhStretch(4.0, scale=2.0), 20, xi=0.8)
    res = su.run(frw_state(su.geo))
    X, alpha, w = su.geo.X, float(EOS.alpha), float(EOS.w)
    assert res.F == pytest.approx(alpha * w * X**3, rel=1e-13)  # F_j = alpha w X_j^3 - X_j^2 (d_xi X)_j, static map
    assert res.speeds.Theta == pytest.approx(0.0, abs=1e-15)  # the background is at rest on a static grid
    assert res.speeds.cE == pytest.approx(alpha * w * X, rel=1e-13)
    assert res.speeds.a == pytest.approx(su.bg.c_s, rel=1e-13)  # the sound speed of Section 5.1
    assert res.speeds.Lam == pytest.approx(su.bg.c_s, rel=1e-13)


# --- the mass law telescopes: d_xi M_j = (2 - 3 alpha) M_j - 3 F_j at every retained face (Section 8.3) ---


@pytest.mark.parametrize("family", STATIC_FAMILIES)
@pytest.mark.parametrize("j_e", [0, 4])
def test_the_cumulative_mass_obeys_the_face_mass_law_at_every_retained_face(family: Family, j_e: int):
    su = Setup.of(family(6.0), 40, xi=0.8, j_e=j_e)
    s = smooth_state(su)
    res = su.run(s)
    lay = su.w.layout
    # d_xi M_j from the rates of the unknowns, as the cumulative sum computes M_j.
    dM = np.full(lay.N + 1, np.nan)
    dM[j_e] = res.rate.M_e
    dM[j_e + 1 :] = res.rate.M_e + 3.0 * np.cumsum(res.rate.E[lay.cells])
    law = EOS.energy_source_rate * res.derived.M[lay.faces] - 3.0 * res.F[lay.faces]
    # The identity is exact; numerically it holds to round-off of the fluxes that the energy rows difference.
    assert dM[lay.faces] == pytest.approx(law, abs=1e-13 * np.max(np.abs(res.F[lay.faces])))


def test_the_total_energy_bookkeeping_is_exact():
    # Summing the energy rows: d_xi sum E = (2 - 3 alpha) sum E - F_N + F_{j_e} (Section 7.2 with the excised face).
    su = Setup.of(SinhStretch(6.0, scale=2.0), 40, xi=0.8, j_e=4)
    s = smooth_state(su)
    res = su.run(s)
    cells = su.w.layout.cells
    total = np.sum(res.rate.E[cells])
    expected = EOS.energy_source_rate * np.sum(s.E[cells]) - res.F[40] + res.F[4]
    assert total == pytest.approx(expected, rel=1e-12)


# --- the layout of the result ---


def test_the_rate_has_nan_below_the_excision_face_and_zero_at_the_origin_velocity():
    su = Setup.of(IdentityMap(4.0), 20, xi=0.8, j_e=3)
    r = su.run(smooth_state(su)).rate
    assert np.all(np.isnan(r.E[:3]))
    assert np.all(np.isnan(r.U[:3]))
    assert np.all(np.isfinite(r.E[3:]))
    assert np.all(np.isfinite(r.U[3:]))
    su0 = Setup.of(IdentityMap(4.0), 20, xi=0.8)
    assert su0.run(smooth_state(su0)).rate.U[0] == 0.0


def test_the_second_order_closure_runs_and_differs_only_at_the_excision_face():
    su1 = Setup.of(SinhStretch(5.0, scale=2.0), 30, xi=0.8, j_e=5, closure=FaceClosure.FIRST_ORDER)
    su2 = Setup.of(SinhStretch(5.0, scale=2.0), 30, xi=0.8, j_e=5, closure=FaceClosure.SECOND_ORDER)
    s = smooth_state(su1)
    r1, r2 = su1.run(s), su2.run(s)
    assert r1.rate.U[5] != r2.rate.U[5]
    assert np.array_equal(r1.rate.U[6:], r2.rate.U[6:])
    # The face value <rho>_{j_e} differs between the closures, so the flux F_{j_e}, the first retained cell's energy
    # rate and the face-mass rate differ too, and by the same amount up to the factor 3.
    assert r1.rate.M_e != r2.rate.M_e
    assert np.array_equal(r1.rate.E[6:], r2.rate.E[6:])
    assert (r1.rate.M_e - r2.rate.M_e) == pytest.approx(3.0 * (r2.rate.E[5] - r1.rate.E[5]), rel=1e-12)


# --- the outer closure ---


def test_the_held_face_follows_the_map_and_uses_the_base_flux():
    su = Setup.of(PinnedMap(SinhStretch(4.0, scale=2.0), alpha=0.5), 20, xi=0.8)
    s = smooth_state(su)
    res = su.run(s)
    N = 20
    assert res.rate.U[N] == su.geo.X_xi[N]
    assert res.F[N] == (res.speeds.cE[N] - su.geo.X_xi[N]) * su.geo.X[N] ** 2 * res.derived.rho_f[N]
    assert res.rate.W == 0.0


@dataclass(frozen=True)
class RecordingClosure(OuterClosure):
    """A test closure that returns fixed rows and keeps what it was given, to check the stage's plumbing."""

    seen: list[OuterInputs]

    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        self.seen.append(inputs)
        return OuterRows(dU_N=1.5, F_N=-2.5, dW=0.25)


def test_the_stage_hands_the_closure_the_face_n_quantities_and_uses_its_rows():
    su = Setup.of(SinhStretch(4.0, scale=2.0), 20, xi=0.8)
    s = smooth_state(su)
    closure = RecordingClosure(seen=[])
    res = su.run(s, closure)
    (inputs,) = closure.seen
    N = 20
    assert inputs.xi == 0.8
    assert inputs.X_N == su.geo.X[N]
    assert inputs.U_N == s.U[N]
    assert inputs.delta_U_N == res.derived.delta_U[N]
    assert inputs.delta_rho_N_1 == res.derived.delta_rho[N - 1]
    assert inputs.rho_f_N == res.derived.rho_f[N]
    assert inputs.ephi_f_N == res.derived.ephi_f[N]
    assert inputs.mt_N == res.derived.mt[N]
    assert inputs.delta_m_N == res.derived.delta_m[N]
    assert inputs.Theta_N == res.speeds.Theta[N]
    assert inputs.cE_N == res.speeds.cE[N]
    assert inputs.dS_N == su.geo.dS[N]
    assert inputs.c_s == su.bg.c_s
    assert res.rate.U[N] == 1.5
    assert res.F[N] == -2.5
    assert res.rate.W == 0.25
    assert res.rate.E[N - 1] == -(-2.5 - res.F[N - 1]) + EOS.energy_source_rate * s.E[N - 1]
