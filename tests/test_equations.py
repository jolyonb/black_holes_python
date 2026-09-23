"""Tests of pbh.equations and pbh.outer: FRW exactness, the telescoping mass law, reparametrisation, and the closure."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest
from whole_state import whole_state_rate

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.equations import calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS, KernelSettings, ViscousFlux
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch
from pbh.outer import PRODUCTION_STRENGTHS, HeldAtFrw, OuterClosure, OuterInputs, OuterRows, OutgoingWave
from pbh.state import State, frw_rate, frw_state
from pbh.stencils import FaceClosure, StencilWeights

EOS = EquationOfState(RADIATION)
AVERAGED_Q = KernelSettings(
    viscous_flux=ViscousFlux.AVERAGED, cap_tension=False
)  # as first printed, but for the end-cell clip
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
    assert res.F[N] == pytest.approx((res.speeds.cE[N] - su.geo.X_xi[N]) * su.geo.X[N] ** 2 * res.derived.rho_f[N])
    assert res.rate.W == 0.0


@dataclass(frozen=True)
class RecordingClosure(OuterClosure):
    """A test closure that returns fixed rows and keeps what it was given, to check the stage's plumbing."""

    seen: list[OuterInputs]

    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        self.seen.append(inputs)
        return OuterRows(delta_dU_N=1.5, delta_F_N=-2.5, dW=0.25)


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
    assert inputs.delta_rho_f_N == res.derived.delta_rho_f[N]
    assert inputs.ephi_f_N == res.derived.ephi_f[N]
    assert inputs.delta_ephi_f_N == res.derived.delta_ephi_f[N]
    assert inputs.mt_N == res.derived.mt[N]
    assert inputs.delta_m_N == res.derived.delta_m[N]
    assert inputs.drift_N == res.speeds.drift[N]
    assert inputs.delta_DU_N == pytest.approx(su.w.velocity_gradient(s.U)[N] - 1.0, abs=1e-12)
    assert inputs.dS_N == su.geo.dS[N]
    assert inputs.c_s == su.bg.c_s
    # the rows come back as deviations from the FRW rows, which are added back for the whole rate
    assert res.deviation_rate.U[N] == 1.5
    assert res.rate.U[N] == 1.5 + su.geo.X_xi[N]
    assert res.delta_F[N] == -2.5
    assert res.F[N] == pytest.approx(-2.5 + EOS.alpha * EOS.w * su.geo.X[N] ** 3)
    assert res.rate.W == 0.25
    delta_E = s.E[N - 1] - su.geo.dV[N - 1]
    assert res.deviation_rate.E[N - 1] == -(-2.5 - res.delta_F[N - 1]) + EOS.energy_source_rate * delta_E


# --- the deviation form against the stage written as printed, on strongly nonlinear states ---


@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, CENTRED_SCHEME, AVERAGED_Q])
@pytest.mark.parametrize("j_e", [0, 5])
@pytest.mark.parametrize("sat", [False, True])
@pytest.mark.parametrize("empty_ends", [False, True])
def test_the_deviation_form_is_the_printed_stage_on_a_strongly_nonlinear_state(
    settings: KernelSettings, j_e: int, sat: bool, empty_ends: bool
):
    # An order-one overdensity and a strong infall on a sinh grid whose interior moves (the outer face may not, under
    # the SAT closure). A slip in the rearranged algebra of the deviation form would show here at order one; what is
    # left is the round-off of the whole-state reference, of the FRW size.
    N, xi = 60, 0.8
    radii, _ = SinhStretch(6.0, scale=2.0).radii(xi, N)
    X_xi = 0.2 * radii * np.exp(-(radii**2))
    X_xi[N:] = 0.0
    geo = Geometry.of(radii, X_xi)
    X = geo.X[: N + 1]
    bg = Background.at(EOS, xi)
    w = StencilWeights.of(geo, Layout(N, j_e), FaceClosure.FIRST_ORDER)
    E = geo.dV * (1.0 + 1.2 * np.exp(-geo.sbar[:-1]) - 0.3 * np.exp(-((geo.Xm - 2.5) ** 2)))
    if empty_ends:  # thin end cells beside denser ones: the reconstruction's positivity clip binds at both. The last
        # cell is only thinned to 0.3, so that the flow across its inner face stays subsonic and the HLL flux there
        # reads the clipped value from its side (an emptier cell's lapse drives the outflow supersonic and hides it).
        E[j_e], E[j_e + 1], E[N - 1] = 1e-6 * E[j_e], 1e-2 * E[j_e + 1], 0.3 * E[N - 1]
    U = X * (1.0 - 0.3 * np.exp(-(X**2) / 2.0))
    state = State(E=E, U=U, W=0.03, M_e=1.4 * float(X[j_e]) ** 3)
    strengths = PRODUCTION_STRENGTHS if sat else None
    outer = OutgoingWave(PRODUCTION_STRENGTHS) if sat else HELD
    res = calc_derivs(state, geo, bg, EOS, w, outer, settings)
    ref = whole_state_rate(state, geo, bg, EOS, w, strengths, settings)
    cells, faces = w.layout.cells, w.layout.faces
    assert np.max(np.abs(res.derived.delta_rho[cells])) > (0.99 if empty_ends else 1.0)  # the state is far from FRW
    assert np.max(np.abs(res.rate.E[cells] - ref.E[cells]) / geo.dV[cells]) < 1e-11
    scale = np.maximum(X[faces], X[1])
    if empty_ends:  # empty cells drive velocity rates far above X, and the reference's round-off with them
        scale = np.maximum(scale, np.abs(ref.U[faces]))
    assert np.nanmax(np.abs(res.rate.U[faces] - ref.U[faces]) / scale) < 1e-12
    assert np.max(np.abs(res.F[faces] - ref.F[faces])) < 1e-12 * X[N] ** 3
    assert res.rate.W == pytest.approx(ref.W, abs=1e-14)
    assert res.rate.M_e == pytest.approx(ref.M_e, rel=1e-12, abs=1e-14)


def vacuum_state(j_e: int, xi: float = 4.0) -> tuple[State, Geometry, Background, StencilWeights]:
    """Both end cells nearly empty beside full neighbours; the cell outside the first one expands hard, so that its
    tension exceeds the fluid pressure several times over; the next-to-last cell is compressed beyond the fluid
    pressure; and an inflow at the outer edge, without which the empty last cell's lapse would make face N - 1
    supersonic outward and its own side of that face would carry no weight. Inside the Hubble radius (xi = 4,
    R_H = 7.4), so that the inflow keeps Gammabar^2 positive.
    """
    N = 40
    radii, _ = SinhStretch(6.0, scale=2.0).radii(xi, N)
    geo = Geometry.of(radii, np.zeros_like(radii))
    X = geo.X[: N + 1]
    rho = 1.0 + 0.8 * np.sin(3.0 * geo.Xm) * np.exp(-(geo.Xm**2) / 2.0)
    rho[j_e], rho[N - 1] = 1e-4, 3e-4  # far enough above round-off of the deviation form (1e-16 / rho)
    X0 = float(X[j_e + 1])
    U = X * (1.0 + 0.6 * np.tanh((X - 1.5) / 0.3) * np.exp(-((X - 1.5) ** 2)))
    U -= 0.5 * X * np.exp(-(((X - X[N]) / 0.5) ** 2))
    U += 40.0 * np.maximum(X - X0, 0.0) * np.exp(-((X - X0) ** 2) / 0.05)
    U[N - 2] += 3.0 * (X[N - 1] - X[N - 2])
    state = State(E=rho * geo.dV[:N], U=U, W=0.0, M_e=0.2 * float(X[j_e]) ** 3)
    return state, geo, Background.at(EOS, xi), StencilWeights.of(geo, Layout(N, j_e), FaceClosure.FIRST_ORDER)


UNCAPPED = KernelSettings(cap_tension=False)
AVERAGED_CAPPED = KernelSettings(viscous_flux=ViscousFlux.AVERAGED)


@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, UNCAPPED, AVERAGED_Q, AVERAGED_CAPPED])
@pytest.mark.parametrize("j_e", [0, 5])
def test_the_deviation_form_is_the_printed_stage_beside_vacuum_with_tension(settings: KernelSettings, j_e: int):
    # Every positivity piece at once: the end-cell clip binds at both ends, one cell is in tension beyond the fluid
    # pressure (capped or not, as the setting says) and one in compression beyond it (which the cap leaves alone), and
    # the one-sided sides that the clip and the density weighting set all carry weight. The reference writes every
    # kernel again, the cap in the force as well as in the flux, so a slip in any of them shows here.
    state, geo, bg, w = vacuum_state(j_e)
    N = w.layout.N
    X = geo.X[: N + 1]
    res = calc_derivs(state, geo, bg, EOS, w, HELD, settings)
    ref = whole_state_rate(state, geo, bg, EOS, w, None, settings)
    k, sp, cells, faces = res.kernels, res.speeds, w.layout.cells, w.layout.faces
    assert k is not None
    assert k.rho_R[j_e] == settings.rho_floor  # the clip binds at both ends
    assert k.rho_L[N] == settings.rho_floor
    assert sp.Theta[N - 1] - sp.a[N - 1] < 0.0  # the last cell's side of face N - 1 carries weight
    q_over_rho = k.q[cells] / res.derived.rho[cells]
    assert np.max(q_over_rho) > float(EOS.w)  # a compression beyond the fluid pressure
    if settings.cap_tension:
        assert np.min(q_over_rho) == pytest.approx(-float(EOS.w), rel=1e-12)  # the cap binds
    else:
        assert np.min(q_over_rho) < -2.0 * float(EOS.w)  # a negative total pressure, as first printed
    # round-off of the whole-state reference is of the FRW size; the vacuum cell's rate is 1e3 times that
    assert np.max(np.abs(res.rate.E[cells] - ref.E[cells]) / (geo.dV[cells] + 1e-3 * np.abs(ref.E[cells]))) < 1e-10
    scale = np.maximum(np.maximum(X[faces], X[1]), np.abs(ref.U[faces]))
    assert np.nanmax(np.abs(res.rate.U[faces] - ref.U[faces]) / scale) < 1e-12
    assert np.max(np.abs(res.F[faces] - ref.F[faces])) < 1e-12 * X[N] ** 3


@pytest.mark.parametrize("j_e", [0, 5])
def test_the_cap_holds_the_tension_at_the_fluid_pressure_and_changes_nothing_else(j_e: int):
    state, geo, bg, w = vacuum_state(j_e)
    cells = w.layout.cells
    on = calc_derivs(state, geo, bg, EOS, w, HELD, PRODUCTION_KERNELS)
    off = calc_derivs(state, geo, bg, EOS, w, HELD, UNCAPPED)
    assert on.kernels is not None
    assert off.kernels is not None
    rho, w_eos = on.derived.rho[cells], float(EOS.w)
    q_on, q_off = on.kernels.q[cells], off.kernels.q[cells]
    assert np.min(q_off / rho) < -2.0 * w_eos  # a tension beyond the fluid pressure, and beyond half of it
    assert np.max(q_off / rho) > w_eos  # and a compression beyond it, which the cap must leave alone
    assert np.array_equal(q_on, np.maximum(q_off, -w_eos * rho))
