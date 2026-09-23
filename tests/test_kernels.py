"""Tests of pbh.kernels: the limiters, the reconstruction, the viscous pressure, the HLL flux, and their effect."""

import math
from dataclasses import dataclass

import numpy as np
import pytest
from modes import mode_errors

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.equations import calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import (
    CENTRED_SCHEME,
    PRODUCTION_KERNELS,
    DensityLimiter,
    KernelSettings,
    ViscousFlux,
    minmod,
    reconstruct_density,
)
from pbh.layout import Layout
from pbh.linearised import energy_norm, relative_scaling
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch
from pbh.outer import HeldAtFrw
from pbh.state import State, frw_rate, frw_state
from pbh.stencils import StencilWeights

EOS = EquationOfState(RADIATION)
HELD = HeldAtFrw()
MINMOD_KERNELS = KernelSettings(density_limiter=DensityLimiter.MINMOD)


@dataclass(frozen=True)
class Setup:
    geo: Geometry
    bg: Background
    w: StencilWeights

    @classmethod
    def of(cls, m: Map, N: int, xi: float = 0.8, j_e: int = 0):
        geo = Geometry.of(*m.radii(xi, N))
        return cls(geo, Background.at(EOS, xi), StencilWeights.of(geo, Layout(N, j_e)))

    def run(self, s: State, settings: KernelSettings = PRODUCTION_KERNELS):
        return calc_derivs(s, self.geo, self.bg, EOS, self.w, HELD, settings)


def smooth_state(su: Setup, amplitude: float = 0.02, seed: int = 0) -> State:
    rng = np.random.default_rng(seed)
    geo, lay = su.geo, su.w.layout
    c1, c2 = rng.uniform(0.5, 1.5, size=2)
    E = geo.dV * (1.0 + amplitude * np.exp(-c1 * geo.sbar[:-1]))
    U = geo.X * (1.0 + amplitude * np.exp(-c2 * geo.X**2))
    return State(E=E, U=U, W=0.0, M_e=float(geo.X[lay.j_e]) ** 3)


# --- the limiters ---


def test_minmod_returns_the_smallest_modulus_when_signs_agree_and_zero_otherwise():
    a = np.array([1.0, -2.0, 3.0, 0.0, 2.0])
    b = np.array([2.0, -1.0, -3.0, 1.0, 2.0])
    c = np.array([0.5, -3.0, 1.0, 1.0, 5.0])
    assert np.array_equal(minmod(a, b), np.array([1.0, -1.0, 0.0, 0.0, 2.0]))
    assert np.array_equal(minmod(a, b, c), np.array([0.5, -1.0, 0.0, 0.0, 2.0]))


# --- the density reconstruction, eq:num:recon ---


@pytest.mark.parametrize("m", [IdentityMap(4.0), SinhStretch(4.0, scale=1.5)])
@pytest.mark.parametrize("limiter", [DensityLimiter.MC, DensityLimiter.MINMOD])
def test_the_reconstruction_is_exact_on_a_field_linear_in_s(m: Map, limiter: DensityLimiter):
    su = Setup.of(m, 24)
    a, b = 1.0, 0.05
    rho = a + b * su.geo.sbar[:-1]  # the shell average of a + b X^2 is its value at sbar
    rho_L, rho_R, _, _ = reconstruct_density(rho - 1.0, su.geo, su.w, limiter, 1e-12)
    exact = a + b * su.geo.X**2
    assert rho_L[1:] == pytest.approx(exact[1:], rel=1e-12)
    assert rho_R[:-1] == pytest.approx(exact[:-1], rel=1e-12)
    assert rho_L[0] == rho_R[0]
    assert rho_R[24] == rho_L[24]


def test_the_mc_coefficients_tend_to_two_away_from_the_origin():
    # r_L, r_R -> 2 where s is locally uniform (Section 7.7); on the uniform grid far out they are close to 2.
    su = Setup.of(IdentityMap(40.0), 400)
    geo = su.geo
    c = 300
    r_L = geo.dS[c] / (geo.sbar[c] - geo.X[c] ** 2)
    r_R = geo.dS[c + 1] / (geo.X[c + 1] ** 2 - geo.sbar[c])
    assert r_L == pytest.approx(2.0, rel=1e-2)  # the approach is first order in dX / X, 0.4 per cent here
    assert r_R == pytest.approx(2.0, rel=1e-2)


@pytest.mark.parametrize("limiter", [DensityLimiter.MC, DensityLimiter.MINMOD])
def test_face_values_stay_between_the_neighbouring_cells_on_rough_data(limiter: DensityLimiter):
    su = Setup.of(IdentityMap(4.0), 30)
    rng = np.random.default_rng(3)
    rho = rng.uniform(0.5, 2.0, size=30)
    rho_L, rho_R, _, _ = reconstruct_density(rho - 1.0, su.geo, su.w, limiter, 1e-12)
    for j in range(2, 29):  # interior faces whose two cells are both limited
        lo, hi = min(rho[j - 1], rho[j]), max(rho[j - 1], rho[j])
        assert lo - 1e-12 <= rho_L[j] <= hi + 1e-12
        assert lo - 1e-12 <= rho_R[j] <= hi + 1e-12


def test_the_floor_is_applied_to_every_face_value():
    su = Setup.of(IdentityMap(4.0), 10)
    rho = np.full(10, 1e-14)
    rho_L, rho_R, delta_L, delta_R = reconstruct_density(rho - 1.0, su.geo, su.w, DensityLimiter.MC, 1e-12)
    assert np.all(rho_L == 1e-12)
    assert np.all(rho_R == 1e-12)
    assert np.all(delta_L == 1e-12 - 1.0)  # a floored value's deviation is the floor's
    assert np.all(delta_R == 1e-12 - 1.0)


def test_the_reconstructed_deviations_are_the_face_values_less_one():
    su = Setup.of(SinhStretch(4.0, scale=2.0), 24)
    rng = np.random.default_rng(5)
    delta_rho = 0.1 * rng.uniform(-1, 1, size=24)
    rho_L, rho_R, delta_L, delta_R = reconstruct_density(delta_rho, su.geo, su.w, DensityLimiter.MC, 1e-12)
    assert delta_L == pytest.approx(rho_L - 1.0, abs=1e-15)
    assert delta_R == pytest.approx(rho_R - 1.0, abs=1e-15)


def test_at_an_excision_face_the_inside_value_is_the_outside_one_and_the_first_slope_is_one_sided():
    su = Setup.of(SinhStretch(4.0, scale=2.0), 24, j_e=5)
    rng = np.random.default_rng(4)
    rho = 1.0 + 0.1 * rng.uniform(-1, 1, size=24)
    rho_L, rho_R, _, _ = reconstruct_density(rho - 1.0, su.geo, su.w, DensityLimiter.MC, 1e-12)
    assert rho_L[5] == rho_R[5]
    slope = (rho[6] - rho[5]) / su.geo.dS[6]  # the single one-sided difference: the clip does not bind here
    assert rho_R[5] == pytest.approx(rho[5] + slope * (su.geo.X[5] ** 2 - su.geo.sbar[5]))
    assert np.all(np.isnan(rho_L[:5]))
    assert np.all(np.isnan(rho_R[:5]))


@pytest.mark.parametrize("j_e", [0, 5])
def test_the_first_retained_cell_is_clipped_where_its_profile_would_turn_negative_inside(j_e: int):
    # A nearly empty first cell beside a denser one: its one-sided slope would carry its profile below zero at its
    # inner face, so the slope is clipped to vanish there, and the outer face value is then
    # rho (X_(e+1)^2 - X_e^2) / (sbar_e - X_e^2), written here from the geometry alone.
    N, e = 24, j_e
    su = Setup.of(SinhStretch(4.0, scale=2.0), N, j_e=j_e)
    rho = np.ones(N)
    rho[e], rho[e + 1] = 1e-6, 1e-2
    rho_L, rho_R, _, _ = reconstruct_density(rho - 1.0, su.geo, su.w, DensityLimiter.MC, 1e-12)
    X2, sbar = su.geo.X**2, su.geo.sbar
    assert rho_L[e + 1] == pytest.approx(1e-6 * (X2[e + 1] - X2[e]) / (sbar[e] - X2[e]), rel=1e-6)
    assert rho_R[e] == 1e-12  # zero at the inner face, then floored


def test_the_last_cell_is_clipped_where_its_profile_would_turn_negative_inside():
    # The mirror image at the outer face: a nearly empty last cell inside a denser one takes a slope that would carry
    # it below zero at the outer face, so the slope is clipped to vanish there.
    N = 24
    su = Setup.of(SinhStretch(4.0, scale=2.0), N)
    rho = np.ones(N)
    rho[N - 1], rho[N - 2] = 1e-6, 1e-2
    rho_L, rho_R, _, _ = reconstruct_density(rho - 1.0, su.geo, su.w, DensityLimiter.MC, 1e-12)
    X2, sbar = su.geo.X**2, su.geo.sbar
    assert rho_R[N - 1] == pytest.approx(1e-6 * (X2[N] - X2[N - 1]) / (X2[N] - sbar[N - 1]), rel=1e-6)
    assert rho_L[N] == 1e-12


# --- FRW: every kernel is inert on the background, on every map ---


@pytest.mark.parametrize(
    "m", [IdentityMap(6.0), SinhStretch(6.0, scale=2.0), PinnedMap(SinhStretch(6.0, scale=2.0), 0.5, 0.3)]
)
@pytest.mark.parametrize("j_e", [0, 4])
@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, MINMOD_KERNELS])
def test_the_kernels_reduce_to_the_base_scheme_on_frw(m: Map, j_e: int, settings: KernelSettings):
    su = Setup.of(m, 40, j_e=j_e)
    s = frw_state(su.geo, j_e)
    res = su.run(s, settings)
    base = su.run(s, CENTRED_SCHEME)
    k = res.kernels
    assert k is not None
    cells, faces = su.w.layout.cells, su.w.layout.faces
    assert np.all(k.J[cells] == 0.0)
    assert np.all(k.q[cells] == 0.0)
    assert np.all(k.Q[j_e : su.w.layout.N] == 0.0)  # Q_N is the outer closure's business and is never formed
    assert k.rho_L[faces] == pytest.approx(1.0, rel=1e-13)
    assert k.rho_R[faces] == pytest.approx(1.0, rel=1e-13)
    assert res.F[faces] == pytest.approx(base.F[faces], rel=1e-13)
    expected = frw_rate(su.geo, j_e)
    assert res.rate.E[cells] == pytest.approx(expected.E[cells], abs=1e-12 * np.max(su.geo.dV))
    assert res.rate.U[faces] == pytest.approx(expected.U[faces], abs=1e-12 * np.max(su.geo.X))


# --- the limited jump and the viscous pressure, eq:num:jump and eq:num:qvisc ---


def test_the_limited_jump_is_half_the_cell_width_squared_times_the_curvature_on_smooth_data():
    # Section 7.7: |J_c| -> (1/2) Delta X_c^2 |d^2 upsilon / dX^2| on smooth data. The minmod slope at a face is biased
    # toward zero by (Delta X / 2) |upsilon''| relative to the exact slope, and that bias is what the jump measures.
    # Compared in the max norm over the domain, since the curvature has nodes.
    def max_ratio(N: int) -> float:
        su = Setup.of(IdentityMap(4.0), N)
        geo = su.geo
        upsilon = 0.01 * np.sin(1.3 * geo.X)  # a smooth odd peculiar velocity
        k = su.run(State(E=geo.dV.copy(), U=geo.X + upsilon, W=0.0)).kernels
        assert k is not None
        predicted = 0.5 * geo.dX**2 * 0.01 * 1.3**2 * np.abs(np.sin(1.3 * geo.Xm))
        c = slice(4, N - 4)
        return float(np.max(np.abs(k.J[c])) / np.max(predicted[c]))

    assert max_ratio(80) == pytest.approx(1.0, abs=0.01)
    assert max_ratio(160) == pytest.approx(1.0, abs=0.005)


def test_the_taper_zeroes_the_pressure_in_the_outermost_cell_and_the_end_row_acts_at_the_excision_face():
    su = Setup.of(SinhStretch(4.0, scale=2.0), 24, j_e=5)
    geo = su.geo
    rng = np.random.default_rng(5)
    s = smooth_state(su)
    U = s.U + 0.01 * rng.normal(size=25) * geo.X  # rough enough for the limiters to bite
    U[:5] = np.nan
    k = su.run(State(E=s.E, U=U, W=0.0, M_e=s.M_e)).kernels
    assert k is not None
    assert k.q[23] == 0.0
    assert k.q_f[5] == k.q[5]
    assert k.Q[5] == pytest.approx(2.0 * geo.sbar[5] * k.q[5] / (geo.X[5] ** 2 * geo.dX[5]))
    assert k.Q[6] == pytest.approx(su.w.gradient_s(geo.sbar[:-1] * k.q)[6] / geo.X[6] ** 2)


# --- the HLL flux, eq:num:hll ---


def test_the_hll_flux_is_the_one_sided_flux_when_both_speeds_point_the_same_way():
    # With Theta > a every characteristic points outward, Lambda^- = 0, and the flux is F(rho_L): pure upwinding.
    su = Setup.of(IdentityMap(4.0), 20)
    geo = su.geo
    s = State(E=geo.dV * 1.1, U=geo.X * 3.0, W=0.0)  # fast outflow relative to the grid
    res = su.run(s)
    k = res.kernels
    assert k is not None
    j = 10
    assert res.speeds.Theta[j] > res.speeds.a[j]
    alpha, w = float(EOS.alpha), float(EOS.w)
    rho, X, U = k.rho_L[j], geo.X[j], s.U[j]
    ephi = rho**EOS.lapse_exponent
    one_sided = (alpha * ((1 + w) * ephi * U - X)) * X**2 * rho + alpha * (ephi * U - X) * X**2 * k.q_f[j]
    assert res.F[j] == pytest.approx(one_sided, rel=1e-12)


# --- what the kernels do: dissipation and convergence ---


@pytest.mark.parametrize(
    "settings",
    [
        PRODUCTION_KERNELS,
        KernelSettings(viscous_flux=ViscousFlux.AVERAGED, cap_tension=False),
    ],  # production, as first printed
)
@pytest.mark.parametrize("m", [IdentityMap(4.0), SinhStretch(4.0, scale=2.0)])
def test_the_kernels_only_remove_energy_near_frw(m: Map, settings: KernelSettings):
    # In the norm of Section 7.4 the rate of change of the energy with the kernels on is never above the base
    # scheme's, for random small perturbations: the kernels are dissipative.
    N = 24
    su = Setup.of(m, N)
    lay = su.w.layout
    H = energy_norm(su.geo, su.bg, EOS, lay)
    T = relative_scaling(su.geo, lay)
    rng = np.random.default_rng(6)
    y_frw = lay.pack(frw_state(su.geo))
    for _ in range(8):
        dy = 1e-4 * rng.normal(size=lay.size) * np.abs(y_frw)
        dy[-2:] = 0.0  # hold the outer velocity and W, as the identity does
        s = lay.unpack(y_frw + dy)
        rate_on = lay.pack(su.run(s, settings).deviation_rate)
        rate_off = lay.pack(su.run(s, CENTRED_SCHEME).deviation_rate)
        z = T * dy
        d_energy_on = 2 * z @ H @ (T * rate_on)
        d_energy_off = 2 * z @ H @ (T * rate_off)
        assert d_energy_on <= d_energy_off + 1e-14 * abs(d_energy_off)


@pytest.mark.slow
@pytest.mark.parametrize("m", [IdentityMap(2.5), SinhStretch(2.5, scale=1.5)])
@pytest.mark.parametrize("k_index", [0, 1, 2])
def test_the_production_scheme_converges_at_second_order_on_the_exact_bessel_modes(m: Map, k_index: int):
    # tab:num:tests row 2: with the kernels, rates >= 1.8 (the kernels cost a constant, not an order).
    # Measured over N = 80, 160, 320 (the paper's row uses 100, 200, 400): the coarsest grids are pre-asymptotic for
    # the second mode, whose density rate is 1.76 from N = 40 to 80 and 1.86 from 80 to 160.
    errors = [mode_errors(m, k_index, N, PRODUCTION_KERNELS) for N in (80, 160, 320)]
    for field in (0, 1):
        rates = [math.log2(errors[i][field] / errors[i + 1][field]) for i in range(2)]
        assert min(rates) > 1.8, f"field {field}, mode {k_index}: L1 rates {rates} (tab:num:tests asks >= 1.8)"
