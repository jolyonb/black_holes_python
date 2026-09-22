"""Tests of pbh.outer: the held face, and the outgoing-wave closure of Section 7.5 with its energy statement."""

from fractions import Fraction

import numpy as np
import pytest
from evolve import evolve
from scipy.linalg import eigh

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS, KernelSettings
from pbh.layout import Layout
from pbh.linearised import energy_norm, jacobian, relative_scaling
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch
from pbh.outer import (
    PRODUCTION_STRENGTHS,
    HeldAtFrw,
    OuterClosure,
    OuterInputs,
    OutgoingWave,
    PenaltyStrengths,
    boundary_ode_coefficients,
    characteristic_pair,
)
from pbh.state import State, frw_rate, frw_state
from pbh.stencils import FaceClosure, StencilWeights
from pbh.timestep import Scheme

RAD = EquationOfState(RADIATION)
UNIFORM = IdentityMap(4.0)
MAPS: list[Map] = [UNIFORM, SinhStretch(4.0, scale=2.0)]
GENERIC = PenaltyStrengths(tau_u=1.5, tau_rho=1.0, tau_W=0.25)  # admissible, off the second-order line, tau_W != 0


# --- the ingredients ---


def test_the_boundary_ode_coefficients_are_the_printed_formulas():
    # At c_s = 1/2, R = 2 the three printed fractions are exact: -(4 + 1)/(4 2 3), -(2 - 1)/(4 2), -(1/2)/(2 3).
    gamma_minus, gamma_plus, gamma_0 = boundary_ode_coefficients(0.5, 2.0)
    assert gamma_minus == float(Fraction(-5, 24))
    assert gamma_plus == float(Fraction(-1, 8))
    assert gamma_0 == float(Fraction(-1, 12))


def test_gamma_minus_is_negative_for_every_sound_speed_with_the_printed_supremum():
    ratio = np.linspace(1e-3, 5.0, 50001)  # c_s / R; the supremum is attained at (sqrt 2 - 1) / 2 = 0.207
    gamma_minus = np.array([boundary_ode_coefficients(r, 1.0)[0] for r in ratio])
    supremum = -(np.sqrt(2.0) - 1.0) / 2.0
    assert np.max(gamma_minus) < 0.0
    assert np.max(gamma_minus) == pytest.approx(supremum, abs=1e-7)
    assert np.max(gamma_minus) <= supremum + 1e-12


def test_the_characteristic_pair_is_the_printed_combination_and_vanishes_on_frw():
    X_N, c_s = 4.0, 0.3
    u_plus, u_minus = characteristic_pair(0.05, 0.1, X_N, c_s)
    kappa = 1.5 * c_s / X_N
    assert u_plus == pytest.approx(0.05 + kappa * 0.1)
    assert u_minus == pytest.approx(0.05 - kappa * 0.1)
    assert characteristic_pair(0.0, 0.0, X_N, c_s) == (0.0, 0.0)


def test_the_production_strengths_are_printed_and_lie_on_the_second_order_line():
    assert PRODUCTION_STRENGTHS == PenaltyStrengths(2.0, 1.0, 0.0)
    assert PRODUCTION_STRENGTHS.is_second_order
    assert PenaltyStrengths(tau_u=1.0, tau_rho=1.5, tau_W=0.5).is_second_order
    assert not GENERIC.is_second_order


@pytest.mark.parametrize("tau", [(1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (4.0, 0.0, 0.0)])
def test_strengths_without_the_energy_statement_are_refused(tau: tuple[float, float, float]):
    # 4 (tau_u + tau_rho - 1) > (tau_u - tau_rho)^2 fails: 0 > 1, 0 > 0, 12 > 16.
    with pytest.raises(ValueError, match="no energy statement"):
        PenaltyStrengths(*tau)


# --- the rows ---


def inputs_off_frw(W: float = 0.02) -> OuterInputs:
    """Face-N quantities of a generic non-FRW state on a static outer face, for radiation, mutually consistent."""
    X_N, U_N, rho_f_N, ephi_f_N, mt_N, DU_N = 4.0, 4.3, 1.15, 0.97, 1.1, 1.3
    return OuterInputs(
        xi=0.5,
        X_N=X_N,
        X_xi_N=0.0,
        U_N=U_N,
        W=W,
        delta_U_N=U_N / X_N - 1.0,
        delta_rho_N_1=0.2,
        rho_f_N=rho_f_N,
        delta_rho_f_N=rho_f_N - 1.0,
        ephi_f_N=ephi_f_N,
        delta_ephi_f_N=ephi_f_N - 1.0,
        mt_N=mt_N,
        delta_m_N=mt_N - 1.0,
        drift_N=0.5 * (ephi_f_N * U_N - X_N),  # alpha (<ephi> U - X), alpha = 1/2
        delta_DU_N=DU_N - 1.0,
        dS_N=2.1,
        c_s=0.35,
    )


def test_the_rows_are_the_printed_formulas_written_out():
    # A second, independent writing of eq:num:sat, term by term and whole, with generic strengths so that every tau
    # enters; the closure returns the rows less their FRW values, d_xi X = 0 and F_FRW = alpha w X^3.
    i, tau = inputs_off_frw(), GENERIC
    alpha, w = 0.5, 1.0 / 3.0
    kappa = 1.5 * i.c_s / i.X_N
    delta_U, delta_rho, delta_m = i.delta_U_N, i.delta_rho_N_1, i.delta_m_N
    u_plus, u_minus = delta_U + kappa * delta_rho, delta_U - kappa * delta_rho
    pen = u_minus - i.W
    Theta_N, DU_N = i.drift_N, 1.0 + i.delta_DU_N  # the grid velocity on a static face, and the gradient
    dU_N = (
        (1.0 - alpha) * i.U_N
        - alpha / 2.0 * i.ephi_f_N * i.X_N * (i.mt_N + 3.0 * w * i.rho_f_N)
        - Theta_N * DU_N
        - tau.tau_u * i.c_s * i.X_N**2 / i.dS_N * pen
    )
    U_star = i.U_N - tau.tau_rho * i.X_N / 2.0 * pen
    F_N = (alpha * ((1.0 + w) * i.ephi_f_N * U_star - i.X_N) - i.X_xi_N) * i.X_N**2 * i.rho_f_N
    gm, gp, g0 = boundary_ode_coefficients(i.c_s, i.X_N)
    dW = gm * i.W + gp * (u_plus - tau.tau_W * pen) + g0 * delta_m
    rows = OutgoingWave(tau).rows(i, RAD)
    assert rows.delta_dU_N == pytest.approx(dU_N, rel=1e-13)
    assert rows.delta_F_N + alpha * w * i.X_N**3 == pytest.approx(F_N, rel=1e-13)
    assert rows.dW == pytest.approx(dW, rel=1e-14)


def test_the_closure_holds_w_at_zero_for_any_other_equation_of_state():
    dust_like = EquationOfState(RADIATION / 2)
    rows = OutgoingWave().rows(inputs_off_frw(W=0.0), dust_like)
    assert rows.dW == 0.0
    assert rows.delta_dU_N != OutgoingWave().rows(inputs_off_frw(W=0.0), RAD).delta_dU_N  # alpha, w differ


def test_the_closure_refuses_a_moving_outer_face():
    pinned = PinnedMap(IdentityMap(4.0), float(RAD.alpha))
    sch = Scheme(RAD, pinned, Layout(8), FaceClosure.FIRST_ORDER, OutgoingWave(), CENTRED_SCHEME)
    with pytest.raises(ValueError, match="static outer face"):
        sch.evaluate(0.3, sch.frw(0.3))


@pytest.mark.parametrize("m", MAPS)
@pytest.mark.parametrize("settings", [CENTRED_SCHEME, PRODUCTION_KERNELS])
def test_frw_is_a_fixed_point_with_the_outgoing_wave_closure(m: Map, settings: KernelSettings):
    sch = Scheme(RAD, m, Layout(24), FaceClosure.FIRST_ORDER, OutgoingWave(), settings)
    xi = 0.7
    res = sch.evaluate(xi, sch.frw(xi))
    expected = frw_rate(sch.frame(xi).geo)
    assert res.rate.E == pytest.approx(expected.E, abs=1e-14)
    assert res.rate.U == pytest.approx(expected.U, abs=1e-14)
    assert res.rate.W == 0.0


# --- linearised at FRW (Section 7.5: the penalties, the energy statement, the spectrum) ---


def linearise(outer: OuterClosure, m: Map = UNIFORM, N: int = 24, xi: float = 0.5):
    """The operator L in the relative variables (delta_rho, delta_U, W) about FRW, with its geometry and background."""
    geo = Geometry.of(*m.radii(xi, N))
    bg = Background.at(RAD, xi)
    lay = Layout(N)
    w = StencilWeights.of(geo, lay, FaceClosure.FIRST_ORDER)
    J = jacobian(frw_state(geo), geo, bg, RAD, w, outer, CENTRED_SCHEME)
    T = relative_scaling(geo, lay)
    return (T[:, None] * J) / T[None, :], geo, bg, lay


def test_the_linearised_penalties_are_the_printed_ones():
    # Section 7.5: linearised at FRW the penalty adds -(tau_u c_s X_N / dS_N) pen to the delta_U,N equation and
    # (tau_rho X_N^3 / 3 dV_N-1) pen to the delta_rho,N-1 equation, and W enters only through pen = u_- - W. So the
    # W column of L is those two coefficients with the sign flipped, and the diagonal entries -tau_u c_s X_N / dS_N
    # and -tau_rho c_s X_N^2 / (2 dV_N-1), read off by comparing two admissible strengths, both damp.
    L, geo, bg, lay = linearise(OutgoingWave(GENERIC))
    N = lay.N
    i_rho, i_U, i_W = N - 1, 2 * N - 1, 2 * N
    X_N, dS_N, dV = geo.X[N], geo.dS[N], geo.dV[N - 1]
    column = np.zeros(2 * N + 1)
    column[i_U] = GENERIC.tau_u * bg.c_s * X_N / dS_N
    column[i_rho] = -GENERIC.tau_rho * X_N**3 / (3.0 * dV)
    gm, gp, _ = boundary_ode_coefficients(bg.c_s, X_N)
    column[i_W] = gm + gp * GENERIC.tau_W  # the ODE row's own W, through gamma_- W and -gamma_+ tau_W pen
    assert L[:, i_W] == pytest.approx(column, abs=1e-9 * np.max(np.abs(column)))
    stronger = PenaltyStrengths(GENERIC.tau_u + 1.0, GENERIC.tau_rho + 1.0, GENERIC.tau_W)
    D = linearise(OutgoingWave(stronger))[0] - L  # the change per unit of tau_u and tau_rho
    assert D[i_U, i_U] == pytest.approx(-bg.c_s * X_N / dS_N, rel=1e-8)
    assert D[i_rho, i_rho] == pytest.approx(-bg.c_s * X_N**2 / (2.0 * dV), rel=1e-8)
    assert D[i_U, i_U] < 0.0
    assert D[i_rho, i_rho] < 0.0


def test_the_linearised_ode_row_is_the_boundary_ode_fed_with_the_interior():
    # d_xi W = gamma_- W + gamma_+ (u_+ - tau_W pen) + gamma_0 delta_m,N with u_+ = delta_U,N + kappa delta_rho,N-1,
    # pen = delta_U,N - kappa delta_rho,N-1 - W and delta_m,N = 3 sum_c dV_c delta_rho,c / X_N^3.
    L, geo, bg, lay = linearise(OutgoingWave(GENERIC))
    N = lay.N
    X_N = geo.X[N]
    gm, gp, g0 = boundary_ode_coefficients(bg.c_s, X_N)
    kappa = 1.5 * bg.c_s / X_N
    row = np.zeros(2 * N + 1)
    row[:N] = g0 * 3.0 * geo.dV / X_N**3
    row[N - 1] += gp * kappa * (1.0 + GENERIC.tau_W)
    row[2 * N - 1] = gp * (1.0 - GENERIC.tau_W)
    row[2 * N] = gm + gp * GENERIC.tau_W
    assert L[2 * N, :] == pytest.approx(row, abs=1e-9 * np.max(np.abs(row)))


@pytest.mark.parametrize("m", MAPS)
def test_no_eigenvalue_of_the_absorbing_operator_grows_faster_than_the_growing_mode(m: Map):
    # Section 7.5: no eigenvalue of the linearised scheme with the boundary has real part above 1; the largest is the
    # absorbing box's growing mode, just below it.
    L, _, _, _ = linearise(OutgoingWave(), m)
    lam = np.linalg.eigvals(L)
    assert np.max(lam.real) <= 1.0 + 1e-8
    assert np.max(lam.real) > 0.9


@pytest.mark.parametrize("m", MAPS)
def test_the_augmented_energy_obeys_the_continuum_bound(m: Map):
    # Section 7.5: face N joins the norm eq:num:norm with weight X_N^3 dS_N / 2, and the boundary energy
    # E_b = c_s X_N^4 (W^2 + delta_m,N^2) / 4 of Section 5.3 is adjoined with u_- replaced by W. The rate of the
    # augmented energy, d/dxi (y^T H y) = y^T (L^T H + H L + dH/dxi) y, is then bounded by 15/4 of the energy
    # (eq:lin:energybound), the weights moving in time through c_s (d_xi c_s = c_s / 2 for radiation).
    L, geo, bg, lay = linearise(OutgoingWave(), m)
    N = lay.N
    X_N = geo.X[N]
    H = energy_norm(geo, bg, RAD, lay)
    H[2 * N - 1, 2 * N - 1] = 0.5 * X_N**3 * geo.dS[N]
    H[2 * N, 2 * N] = 0.25 * bg.c_s * X_N**4
    k_N = 3.0 * geo.dV / X_N**3  # delta_m,N = k_N . delta_rho
    H[:N, :N] += 0.25 * bg.c_s * X_N**4 * np.outer(k_N, k_N)
    dH = np.zeros_like(H)
    dH[:N, :N] = 2.0 * np.diag(2.25 * bg.c_s**2 * geo.dV) * (1.0 - float(RAD.alpha))  # d_xi c_s^2 = c_s^2
    dH[2 * N, 2 * N] = 0.5 * H[2 * N, 2 * N]  # d_xi c_s = c_s / 2
    dH[:N, :N] += 0.5 * 0.25 * bg.c_s * X_N**4 * np.outer(k_N, k_N)
    rate = eigh(L.T @ H + H @ L + dH, H, eigvals_only=True)  # generalised: the growth rates of the energy
    assert bg.c_s <= X_N  # the bound's hypothesis
    assert np.max(rate) <= 3.75


@pytest.mark.parametrize("m", MAPS)
def test_the_spectral_footprint_does_not_depend_on_the_closure(m: Map):
    # Section 7.6: scaled by the Courant step, the eigenvalues of the base scheme form a footprint of radius 2.00-2.02
    # (the acoustic band top 2 c_s / Delta X) independent of the outer closure.
    for outer in (HeldAtFrw(), OutgoingWave()):
        L, geo, bg, _ = linearise(outer, m, N=48)
        radius = np.max(np.abs(np.linalg.eigvals(L))) * np.min(geo.dX) / bg.c_s
        assert radius == pytest.approx(2.0, abs=0.03)


# --- the reflection of an outgoing packet (Table tab:num:tests; slow) ---


def outgoing_packet(geo: Geometry, bg: Background, k: float, X_0: float, sigma: float, amp: float) -> State:
    """FRW plus a compensated wave packet with `u_- = 0`, outgoing to leading order in `1 / (k X)`.

    With `g(X) = amp exp(-(X - X_0)^2 / 2 sigma^2) cos(k (X - X_0))` the density perturbation is
    `X^2 delta_rho = d/dX (X^2 g)`, so the cell contents are exact, `E_c = dV_c + [X^2 g]`, and the mass perturbation
    `3 g / X` is compensated outside the packet; the velocity `delta_U = kappa delta_rho` at the faces puts the packet
    on the outgoing characteristic, `u_- = 0`. The auxiliary scalar starts at the initial `u_-` of the discrete pair.
    """
    X = geo.X
    envelope = amp * np.exp(-0.5 * ((X - X_0) / sigma) ** 2)
    g = envelope * np.cos(k * (X - X_0))
    g_prime = envelope * (-(X - X_0) / sigma**2 * np.cos(k * (X - X_0)) - k * np.sin(k * (X - X_0)))
    E = geo.dV + np.diff(X**2 * g)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_rho = np.where(X > 0.0, g_prime + 2.0 * g / X, 0.0)
        delta_U = np.where(X > 0.0, 1.5 * bg.c_s / X * delta_rho, 0.0)
    U = X * (1.0 + delta_U)
    N = geo.N
    delta_rho_N_1 = (E[N - 1] - geo.dV[N - 1]) / geo.dV[N - 1]
    W = characteristic_pair(float(delta_U[N]), float(delta_rho_N_1), float(X[N]), bg.c_s)[1]
    return State(E=E, U=U, W=W)


def energy(geo: Geometry, bg: Background, state: State, n: int) -> float:
    """The energy of the deviation from FRW in the diagonal norm of Section 7.4, cells `0..n-1` and faces `1..n`."""
    delta_rho = (state.E[:n] - geo.dV[:n]) / geo.dV[:n]
    delta_U = state.U[1 : n + 1] / geo.X[1 : n + 1] - 1.0
    H_cells = 2.25 * bg.c_s**2 * geo.dV[:n]
    H_faces = 0.5 * geo.X[1 : n + 1] ** 3 * geo.dS[1 : n + 1]
    return float(np.sum(H_cells * delta_rho**2) + np.sum(H_faces * delta_U**2))


def reflection(
    N: int, outer: OuterClosure, settings: KernelSettings, Rtilde_max: float = 10.0, k: float = 3.0
) -> float:
    """The reflected fraction of an outgoing packet's amplitude at the outer face, the measurement of Section 7.5.

    The packet (centre `Rtilde_max / 2`, width `2.5 / k`, amplitude 1e-5) is evolved on the uniform grid with the
    closure, and on a grid twice as long with the same cell width and the face held (there the test boundary is an
    interior face), until its trailing edge has left the shorter domain; the reflection is what the two runs then
    differ by inside the shorter domain, relative to the packet, both measured in the energy norm.
    """
    X_0, sigma, amp = 0.5 * Rtilde_max, 2.5 / k, 1e-5
    bg_0 = Background.at(RAD, 0.0)
    distance = Rtilde_max - X_0 + 4.5 * sigma  # the trailing edge, 3.5 sigma behind the centre, plus one sigma
    xi_end = 2.0 * np.log(1.0 + distance / bg_0.tau)  # e^(xi/2) - 1 = distance / tau(0): the sound travel time
    test = Scheme(RAD, IdentityMap(Rtilde_max), Layout(N), FaceClosure.FIRST_ORDER, outer, settings)
    longer = IdentityMap(2.0 * Rtilde_max)
    reference = Scheme(RAD, longer, Layout(2 * N), FaceClosure.FIRST_ORDER, HeldAtFrw(), settings)
    finals: list[State] = []
    for sch in (test, reference):
        geo = sch.frame(0.0).geo
        finals.append(evolve(sch, outgoing_packet(geo, bg_0, k, X_0, sigma, amp), 0.0, xi_end))
    geo, bg = test.frame(xi_end).geo, Background.at(RAD, xi_end)
    geo_ref = reference.frame(xi_end).geo
    difference = State(E=finals[0].E - finals[1].E[:N] + geo.dV, U=finals[0].U - finals[1].U[: N + 1] + geo.X, W=0.0)
    return float(np.sqrt(energy(geo, bg, difference, N) / energy(geo_ref, bg, finals[1], 2 * N)))


@pytest.mark.slow
@pytest.mark.parametrize(("settings", "coefficient"), [(CENTRED_SCHEME, 0.25), (PRODUCTION_KERNELS, 0.29)])
def test_the_outgoing_packet_reflection_is_second_order_with_the_printed_coefficient(
    settings: KernelSettings, coefficient: float
):
    # Table tab:num:tests: reflected amplitude <= 0.35 (k dX_N)^2 for k dX_N <= 0.15; Section 7.5 gives the measured
    # coefficients 0.25 (base scheme) and 0.29 (production kernels, viscous pressure tapered in the last cell).
    k, Rtilde_max = 3.0, 10.0
    R = {N: reflection(N, OutgoingWave(), settings, Rtilde_max, k) for N in (200, 400)}
    for N, r in R.items():
        k_dX = k * Rtilde_max / N  # 0.15 and 0.075
        assert r <= 0.35 * k_dX**2
        assert r == pytest.approx(coefficient * k_dX**2, rel=0.15)
    assert np.log2(R[200] / R[400]) > 1.8  # second order
