"""Tests of pbh.monitors: the stage quantities, the step record, and the pieces they are built from."""

import dataclasses
import math

import numpy as np
import pytest
from modes import J1_ZEROS, mode_state, single_mode

from pbh.eos import RADIATION, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS, DensityLimiter, reconstruct_density
from pbh.layout import Layout
from pbh.maps import IdentityMap, SinhStretch
from pbh.monitors import (
    MonitoredStep,
    StageFluxes,
    StepInputs,
    alternating,
    boundary_energy,
    grid_scale_fraction,
    limiter_clipped,
    monitor_step,
)
from pbh.outer import HeldAtFrw, OutgoingWave, characteristic_pair
from pbh.state import State, frw_state
from pbh.stencils import StencilWeights
from pbh.timestep import RK4, Integrator, Scheme, advance, courant_step
from pbh.types import FloatArray

RAD = EquationOfState(RADIATION)
N = 60
SCHEME = Scheme(RAD, IdentityMap(6.0), Layout(N), OutgoingWave(), PRODUCTION_KERNELS)
THETA = PRODUCTION_KERNELS.theta
WEIGHTS = tuple(float(b) for b in RK4.b)


def evaluate(sch: Scheme, xi: float, state: State):
    f = sch.frame(xi)
    return f, sch.evaluate(xi, sch.layout.pack(state))


def inputs_for(
    sch: Scheme, xi: float, state: State, stages: list[StageFluxes] | None = None, **kw: float
) -> StepInputs:
    f, result = evaluate(sch, xi, state)
    end = StageFluxes.of(result, sch.layout)
    # four identical stages: the mass before is what makes the weighted rate land exactly on the mass after
    rate = float(sch.eos.energy_source_rate) * end.delta_M_total - 3.0 * end.delta_F_N
    return StepInputs(
        step=3,
        xi=xi,
        dxi=0.01,
        limit="courant",
        halvings=0,
        state=state,
        geo=f.geo,
        bg=f.bg,
        result=result,
        stages=stages or [end] * 4,
        weights=WEIGHTS,
        delta_M_total_before=kw.get("delta_M_total_before", end.delta_M_total - 0.01 * rate),
        F_N_integral_before=kw.get("F_N_integral_before", 0.0),
        rate_change=kw.get("rate_change", 0.0),
        far_zone_from=kw.get("far_zone_from", 4.0),
    )


# --- FRW: every deviation monitor is zero, every validity monitor at its background value ---


def test_on_frw_the_record_is_the_background_and_every_deviation_is_zero():
    xi = 0.4
    f = SCHEME.frame(xi)
    state = frw_state(f.geo)
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert isinstance(row, MonitoredStep)
    assert (row.step, row.xi, row.dxi, row.limit) == (3, xi, 0.01, "courant")
    assert row.rho_min == 1.0
    assert row.Gammabar2_min == pytest.approx(1.0)
    assert row.theta_binds == 0
    assert row.M_total == pytest.approx(f.geo.X[N] ** 3)
    assert row.F_N == pytest.approx(
        3.0 * f.geo.X[N] ** 2 * (2.0 / 3.0) * 0.5, rel=1e-12
    )  # alpha (1 + w) - alpha = alpha w
    assert row.bookkeeping_residual < 1e-14
    assert (row.far_zone_delta_rho, row.far_zone_delta_U) == (0.0, 0.0)
    assert (row.W, row.u_plus, row.u_minus, row.penalty, row.delta_rho_N_1) == (0.0,) * 5
    assert row.delta_m_N == pytest.approx(0.0, abs=1e-15)
    assert row.boundary_energy == pytest.approx(0.0, abs=1e-28)  # delta_m,N^2 at round-off
    assert (row.rho_0, row.delta_U_1, row.odd_even_inner) == (1.0, 0.0, 0.0)
    assert (row.grid_scale_cells, row.grid_scale_faces) == (0.0, 0.0)
    assert (row.steepest_log_slope, row.largest_cell_jump, row.largest_cell_jump_outside_3) == (0.0, 0.0, 0.0)
    assert row.largest_four_cell_ratio_outside_5 == 1.0
    assert row.supersonic_faces == 0
    assert row.core_cells == N  # the density never falls to half
    assert (row.viscous_over_pressure_core, row.clipped_cells, row.clipped_in_core, row.reconstruction_jump) == (
        0,
        0,
        0,
        0,
    )
    assert row.companion == 0.0
    assert row.courant_ratio == pytest.approx(
        np.min(f.geo.dX / np.maximum(f.bg.c_s, 0.0)), rel=0.5
    )  # Lambda ~ |Theta| + a


def test_the_full_row_locates_the_minima_and_the_courant_cell():
    xi = 0.4
    f = SCHEME.frame(xi)
    state = frw_state(f.geo)
    E = state.E.copy()
    E[17] *= 1.5  # a mass excess in cell 17: Gammabar^2 drops just outside it, most at face 18
    E[30] *= 0.5  # a low density in cell 30 (a deficit raises Gammabar^2, so the minimum stays at face 18)
    perturbed = State(E=E, U=state.U, W=0.0)
    row = monitor_step(inputs_for(SCHEME, xi, perturbed), RAD, SCHEME.layout)
    assert (row.rho_min, row.rho_min_cell) == (pytest.approx(0.5), 30)
    assert row.Gammabar2_min < 1.0
    assert row.Gammabar2_min_face == 18
    assert row.M_total == pytest.approx(3.0 * np.sum(E))
    assert row.courant_cell in range(N)
    assert row.a_at_courant > 0.0
    fluxes = StageFluxes.of(evaluate(SCHEME, xi, perturbed)[1], SCHEME.layout)
    assert fluxes.F_je == 0.0
    assert fluxes.M_total == row.M_total


# --- the bookkeeping residual over a real step ---


def test_the_bookkeeping_residual_of_an_rk4_step_is_round_off():
    sch = Scheme(RAD, IdentityMap(6.0), Layout(N), HeldAtFrw(), CENTRED_SCHEME)
    xi = 0.0
    f = sch.frame(xi)
    state = mode_state(single_mode(J1_ZEROS[0] / 6.0, 1e-2), f.bg, f.geo)
    y = sch.layout.pack(state)
    dy = y - sch.frw(xi)
    dxi = courant_step(sch.evaluate(xi, y), f.geo, sch.layout, 0.75)
    # the stages, by hand, as the driver will take them from the tableau
    stages: list[StageFluxes] = []
    k: list[FloatArray] = []
    for c_i, a_i in zip(RK4.c, RK4.a, strict=True):
        y_i = y.copy()
        for a_ij, k_j in zip(a_i, k, strict=True):
            if a_ij:
                y_i += dxi * float(a_ij) * k_j
        xi_i = xi + float(c_i) * dxi
        r_i = sch.evaluate(xi_i, y_i)
        k.append(sch.layout.pack(r_i.deviation_rate))
        stages.append(StageFluxes.of(r_i, sch.layout))
    dy_new = advance(sch, Integrator.RK4, xi, dy, dxi)
    new = sch.layout.unpack(sch.frw(xi + dxi) + dy_new)
    inputs = inputs_for(sch, xi + dxi, new, stages, delta_M_total_before=stages[0].delta_M_total)
    inputs = dataclasses.replace(inputs, dxi=dxi)
    row = monitor_step(inputs, RAD, sch.layout)
    assert row.bookkeeping_residual < 1e-13
    assert row.F_N == pytest.approx(sum(b * s.F_N for b, s in zip(WEIGHTS, stages, strict=True)))
    assert row.F_N_integral == pytest.approx(dxi * row.F_N)


# --- the state monitors on constructed states ---


def test_the_outer_boundary_the_centre_and_the_far_zone_read_the_state():
    xi = 0.4
    f = SCHEME.frame(xi)
    X = f.geo.X[: N + 1]
    frw = frw_state(f.geo)
    E = frw.E * (1.0 + 0.01 * np.exp(-(f.geo.Xm**2)))  # a central overdensity, gone by X = 4
    U = X * (1.0 + 0.005 * np.exp(-(X**2)))
    U[0] = 0.0
    state = State(E=E, U=U, W=0.002)
    row = monitor_step(inputs_for(SCHEME, xi, state, far_zone_from=4.0), RAD, SCHEME.layout)
    d = SCHEME.evaluate(xi, SCHEME.layout.pack(state)).derived
    u_plus, u_minus = characteristic_pair(float(d.delta_U[N]), float(d.delta_rho[N - 1]), float(X[N]), f.bg.c_s)
    assert (row.u_plus, row.u_minus, row.W) == (u_plus, u_minus, 0.002)
    assert row.penalty == u_minus - 0.002
    assert row.delta_m_N == pytest.approx(d.mt[N] - 1.0)
    assert row.rho_0 == pytest.approx(d.rho[0])
    assert row.delta_U_1 == pytest.approx(U[1] / X[1] - 1.0)
    assert row.far_zone_delta_rho < 1e-6
    assert row.far_zone_delta_U < 1e-6
    assert row.boundary_energy == pytest.approx(boundary_energy(state, d, f.geo, f.bg, SCHEME.layout))
    assert row.boundary_energy > 0.0
    assert row.core_cells == N  # 1 per cent is far from half the central density


def test_the_steepness_probe_on_a_step_profile():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    rho = np.where(f.geo.Xm[:N] < 1.5, 1.3, 1.0)  # a mild dense core of radius 1.5 on the uniform grid, dX = 0.1
    state = State(E=frw.E * rho, U=frw.U, W=0.0)
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert row.core_cells == N  # 1 / 1.3 is above a half
    assert row.largest_cell_jump == pytest.approx(np.log(1.3))
    assert row.steepest_X == pytest.approx(1.5)
    assert row.largest_cell_jump_outside_3 == 0.0
    assert row.largest_four_cell_ratio_outside_5 == 1.0
    assert row.clipped_cells > 0
    assert row.reconstruction_jump > 0.1
    assert row.viscous_over_pressure_core >= 0.0


def test_the_core_count_and_the_under_resolution_monitors_on_a_gaussian_core():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    rho = 1.0 + 3.0 * np.exp(-((f.geo.Xm[:N] / 0.5) ** 2))  # central density 4, half of it at X = 0.5 sqrt(ln 3)
    state = State(E=frw.E * rho, U=frw.U, W=0.0)
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert row.core_cells == 5  # cells with Xm = 0.05 .. 0.45 lie inside 0.524
    assert row.rho_0 == pytest.approx(rho[0])
    assert row.viscous_over_pressure_core == 0.0  # FRW velocity: no velocity jump, no viscous pressure
    assert row.clipped_in_core <= row.clipped_cells


def test_a_supersonic_zone_is_counted_and_located():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    U = frw.U.copy()
    U[5:10] = f.geo.X[5:10] * 0.05  # strongly infalling faces well inside e^xi/2, where Gammabar^2 stays positive
    state = State(E=frw.E, U=U, W=0.0)
    _, result = evaluate(SCHEME, xi, state)
    supersonic = np.abs(result.speeds.Theta[: N + 1]) > result.speeds.a[: N + 1]
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert 1 <= row.supersonic_faces == int(np.sum(supersonic)) <= 5
    assert row.supersonic_outer_X == pytest.approx(f.geo.X[np.flatnonzero(supersonic)[-1]])


def test_the_companion_and_the_running_integral_pass_through():
    xi = 0.4
    state = frw_state(SCHEME.frame(xi).geo)
    row = monitor_step(inputs_for(SCHEME, xi, state, rate_change=0.6, F_N_integral_before=2.0), RAD, SCHEME.layout)
    assert row.companion == pytest.approx(0.01 / 6.0 * 0.6)
    assert row.F_N_integral == pytest.approx(2.0 + 0.01 * row.F_N)


def test_off_radiation_the_boundary_energy_is_not_defined_and_the_centred_scheme_has_no_kernel_monitors():
    dust_like = EquationOfState(RADIATION / 2)
    sch = Scheme(dust_like, IdentityMap(6.0), Layout(N), OutgoingWave(), CENTRED_SCHEME)
    xi = 0.4
    state = frw_state(sch.frame(xi).geo)
    row = monitor_step(inputs_for(sch, xi, state), dust_like, sch.layout)
    assert np.isnan(row.boundary_energy)
    assert (row.theta_binds, row.clipped_cells, row.reconstruction_jump, row.viscous_over_pressure_core) == (
        0,
        0,
        0,
        0,
    )


def test_an_excised_layout_is_monitored_on_the_retained_cells_only():
    j_e = 5
    layout = Layout(N, j_e=j_e)
    sch = Scheme(RAD, IdentityMap(6.0), layout, OutgoingWave(), PRODUCTION_KERNELS)
    xi = 0.4
    f = sch.frame(xi)
    state = frw_state(f.geo, j_e)
    row = monitor_step(inputs_for(sch, xi, state), RAD, layout)
    assert row.rho_min == 1.0
    assert row.rho_min_cell >= j_e
    assert row.rho_0 == 1.0  # the innermost retained cell
    assert np.isnan(row.boundary_energy)
    assert row.M_total == pytest.approx(f.geo.X[N] ** 3)
    assert row.odd_even_inner == 0.0
    assert row.bookkeeping_residual < 1e-14


def test_the_theta_binds_count_the_cells_whose_slope_the_theta_limiter_scaled():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    E = frw.E.copy()
    E[29:32] *= np.array([1e-4, 1e-8, 1e-4])  # a valley: mc would take cells 29 and 31 down to the valley bottom,
    state = State(E=E, U=frw.U, W=0.0)  # far below theta times theirs, so their slopes are scaled; 30 is a minimum
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert row.theta_binds == 2


def test_on_frw_the_chord_widens_the_bounds_beyond_the_crossover_by_the_pressure_work():
    # On FRW the chord speed is the pressure work alpha w X against a grid velocity of zero, so beyond the crossover
    # X_c = e^((1-alpha) xi) / sqrt(w) the bounds are [-a, alpha w X] instead of [-a, a]: a width ratio of
    # (alpha w X + a) / (2 a), largest at the outermost flux face, and no compression anywhere.
    sch = Scheme(RAD, IdentityMap(24.0), Layout(N), OutgoingWave(), PRODUCTION_KERNELS)
    f = sch.frame(0.0)
    row = monitor_step(inputs_for(sch, 0.0, frw_state(f.geo)), RAD, sch.layout)
    alpha, w = float(RAD.alpha), float(RAD.w)
    a = alpha * math.sqrt(w)  # at xi = 0
    X = f.geo.X[: N + 1]
    assert row.widened_faces == int(np.sum(alpha * w * X[1:N] > 1.001 * a))
    assert row.widening_ratio == pytest.approx((alpha * w * X[N - 1] + a) / (2.0 * a), rel=1e-12)
    assert row.q_over_rho_max == 0.0


# --- the pieces ---


def test_the_alternating_component_and_the_grid_scale_fraction():
    smooth = np.linspace(0.0, 1.0, 11)
    assert alternating(smooth) == pytest.approx(np.zeros(9))
    assert grid_scale_fraction(smooth) == pytest.approx(0.0, abs=1e-15)
    sawtooth = np.array([1.0, -1.0] * 6)
    assert alternating(sawtooth) == pytest.approx(2.0 * sawtooth[1:-1])
    assert grid_scale_fraction(sawtooth) == pytest.approx(2.0)  # the alternating component of a sawtooth is twice it
    assert grid_scale_fraction(np.zeros(5)) == 0.0
    assert grid_scale_fraction(np.ones(2)) == 0.0


def test_the_limiter_clipping_detector_sees_a_kink_and_not_a_smooth_field():
    geo = Geometry.of(*IdentityMap(6.0).radii(0.0, N))
    layout = Layout(N)
    sch = Scheme(RAD, IdentityMap(6.0), layout, OutgoingWave(), PRODUCTION_KERNELS)
    frw = frw_state(geo)
    smooth = State(E=frw.E * (1.0 + 0.01 * geo.sbar[:N] / 36.0), U=frw.U, W=0.0)  # linear in s: exactly reconstructed
    result = sch.evaluate(0.0, layout.pack(smooth))
    assert result.kernels is not None
    assert not np.any(limiter_clipped(result.derived.delta_rho, result.kernels.delta_rho_L, geo, layout))
    kinked = State(E=frw.E * np.where(geo.Xm[:N] < 1.0, 1.2, 1.0), U=frw.U, W=0.0)  # a jump between cells 9 and 10
    result = sch.evaluate(0.0, layout.pack(kinked))
    assert result.kernels is not None
    clipped = limiter_clipped(result.derived.delta_rho, result.kernels.delta_rho_L, geo, layout)
    assert clipped[9]
    assert clipped[10]
    assert not np.any(clipped[:8])
    assert not np.any(clipped[12:])


def test_the_monitor_counts_the_end_cells_when_the_theta_limiter_binds_there():
    sch = SCHEME
    layout, geo = sch.layout, sch.frame(0.0).geo
    frw = frw_state(geo)
    rho = np.ones(N)
    rho[0], rho[1] = 1e-6, 1e-2  # the origin cell's one-sided slope would turn its profile negative at X = 0
    rho[N - 1], rho[N - 2] = 1e-6, 1e-2  # and the last cell's at X_N
    result = sch.evaluate(0.0, layout.pack(State(E=frw.E * rho, U=frw.U, W=0.0)))
    assert result.kernels is not None
    clipped = limiter_clipped(result.derived.delta_rho, result.kernels.delta_rho_L, geo, layout)
    assert clipped[0]
    assert clipped[N - 1]
    mild = np.ones(N)
    mild[0], mild[N - 1] = 1.1, 0.9  # one-sided slopes that keep the faces above theta rho: no limiting
    result = sch.evaluate(0.0, layout.pack(State(E=frw.E * mild, U=frw.U, W=0.0)))
    assert result.kernels is not None
    clipped = limiter_clipped(result.derived.delta_rho, result.kernels.delta_rho_L, geo, layout)
    assert not clipped[0]
    assert not clipped[N - 1]


@pytest.mark.parametrize("j_e", [0, 5])
@pytest.mark.parametrize(("excess", "clipped"), [(1.1, True), (0.9, False)])
def test_the_monitor_sees_an_end_cell_theta_limit_of_a_tenth(j_e: int, excess: float, clipped: bool):
    # The end cells' one-sided slopes set to `excess` times the slope at which the profile reaches theta rho at the far
    # face: 1.1 is scaled back by a tenth, 0.9 not at all. A detector blind to a small limit, or one that reads
    # round-off as one, fails one of the two.
    n = 24
    radii, _ = SinhStretch(4.0, scale=2.0).radii(0.0, n)
    geo = Geometry.of(radii, np.zeros_like(radii))
    layout = Layout(n, j_e)
    weights = StencilWeights.of(geo, layout)
    X2, sbar = geo.X**2, geo.sbar
    rho = np.ones(n)
    e, f = j_e, n - 1
    drop = (1.0 - THETA) * 0.01  # the largest drop below the mean the theta-limiter allows
    rho[e] = 0.01  # rising outward: theta rho at the inner face at slope drop / (sbar_e - X_e^2)
    rho[e + 1] = rho[e] + excess * drop / (sbar[e] - X2[e]) * (sbar[e + 1] - sbar[e])
    rho[f] = 0.01  # falling outward: theta rho at the outer face at slope -drop / (X_N^2 - sbar_f)
    rho[f - 1] = rho[f] + excess * drop / (X2[f + 1] - sbar[f]) * (sbar[f] - sbar[f - 1])
    _, _, delta_L, _, _ = reconstruct_density(rho - 1.0, geo, weights, DensityLimiter.MC, THETA)
    flags = limiter_clipped(rho - 1.0, delta_L, geo, layout)
    assert flags[e] == clipped
    assert flags[f] == clipped


def test_a_plain_step_records_the_first_tier_and_leaves_the_second_unset():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    rho = 1.0 + 3.0 * np.exp(-((f.geo.Xm[:N] / 0.5) ** 2))
    state = State(E=frw.E * rho, U=frw.U, W=0.0)
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout, full=False)
    assert row.rho_0 == pytest.approx(rho[0])
    assert row.rho_min == pytest.approx(1.0, rel=1e-6)
    assert row.bookkeeping_residual < 1e-14
    assert np.isnan(row.u_minus)  # second tier
    assert np.isnan(row.grid_scale_cells)
    assert np.isnan(row.steepest_log_slope)
    assert np.isnan(row.reconstruction_jump)
    assert np.isnan(row.boundary_energy)
    assert (row.rho_min_cell, row.courant_cell, row.core_cells, row.clipped_cells) == (-1, -1, -1, -1)
    assert (row.clipped_in_core, row.supersonic_faces, row.theta_binds) == (-1, -1, -1)


# --- the monitors that were only weakly checked, against independent expectations ---


def test_the_boundary_energy_equals_the_matrix_norm_of_the_linearised_scheme_plus_the_boundary_term():
    # linearised.energy_norm is an independent implementation of eq:num:norm (faces 1..N-1), verified there against
    # the energy identity; face N's weight and E_b are added by hand here.
    from pbh.linearised import energy_norm

    sch = Scheme(RAD, IdentityMap(6.0), Layout(N), OutgoingWave(), CENTRED_SCHEME)
    xi = 0.5
    f = sch.frame(xi)
    state = mode_state(single_mode(1.1, 2e-3), f.bg, f.geo)
    state = State(E=state.E, U=state.U, W=0.003)
    d = sch.evaluate(xi, sch.layout.pack(state)).derived
    X = f.geo.X[: N + 1]
    delta_rho = d.rho - 1.0
    delta_U = state.U[1:] / X[1:] - 1.0
    y = np.concatenate((delta_rho, delta_U, [state.W]))  # (delta_rho, delta_U at 1..N, W) as linearised orders them
    H = energy_norm(f.geo, f.bg, RAD, sch.layout)
    matrix_norm = float(y @ H @ y)
    delta_m_N = float(d.mt[N]) - 1.0
    face_N = 0.5 * X[N] ** 3 * f.geo.dS[N] * (delta_U[-1] ** 2 + delta_m_N**2 / 8.0)
    E_b = 0.25 * f.bg.c_s * X[N] ** 4 * (state.W**2 + delta_m_N**2)
    assert boundary_energy(state, d, f.geo, f.bg, sch.layout) == pytest.approx(matrix_norm + face_N + E_b, rel=1e-10)


def test_the_far_zone_reports_the_largest_deviation_beyond_its_radius_and_nothing_inside():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    E = frw.E.copy()
    E[5] *= 1.01  # inside: must not count
    E[50] *= 1.0002  # cell 50, Xm = 5.05, beyond 4: counts
    U = frw.U.copy()
    U[55] *= 1.0001  # face 55, X = 5.5: counts
    row = monitor_step(inputs_for(SCHEME, xi, State(E=E, U=U, W=0.0), far_zone_from=4.0), RAD, SCHEME.layout)
    assert row.far_zone_delta_rho == pytest.approx(2e-4, rel=1e-8)
    assert row.far_zone_delta_U == pytest.approx(1e-4, rel=1e-8)


def test_the_steepness_values_the_inner_odd_even_amplitude_and_the_four_cell_ratio():
    xi = 0.4
    f = SCHEME.frame(xi)
    Xm = f.geo.Xm[:N]
    frw = frw_state(f.geo)
    rho = np.ones(N)
    rho[:4] = [1.0, 1.02, 1.0, 1.02]  # an inner sawtooth of amplitude 0.01 about 1.01: alternating component 0.02
    rho[Xm > 5.0] = 1.01  # a jump of 1.01 at X = 5 (between cells 49 and 50), seen by the four-cell probe outside 5
    state = State(E=frw.E * rho, U=frw.U, W=0.0)
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert row.odd_even_inner == pytest.approx(0.02 / 1.0)  # relative to rho_0 = 1
    assert row.largest_four_cell_ratio_outside_5 == pytest.approx(1.01)
    assert row.largest_cell_jump_outside_3 == pytest.approx(np.log(1.01))
    # the steepest log slope: ln(1.01) over ln(Xm_50 / Xm_49), unless the inner sawtooth is steeper in ln X
    inner_slope = np.abs(np.log(1.02) / np.log(Xm[1] / Xm[0]))
    outer_slope = np.log(1.01) / np.log(Xm[50] / Xm[49])
    assert row.steepest_log_slope == pytest.approx(max(inner_slope, outer_slope))
    assert row.delta_rho_N_1 == pytest.approx(0.01)


def test_the_reconstruction_jump_and_the_viscous_ratio_are_the_kernels_own_values_in_the_right_places():
    xi = 0.4
    f = SCHEME.frame(xi)
    frw = frw_state(f.geo)
    rho = 1.0 + 3.0 * np.exp(-((f.geo.Xm[:N] / 0.5) ** 2))  # core of 5 cells
    U = frw.U.copy()
    U[1:6] *= 0.9  # a velocity kink inside the core: a viscous pressure there
    state = State(E=frw.E * rho, U=U, W=0.0)
    _, result = evaluate(SCHEME, xi, state)
    assert result.kernels is not None
    k, d = result.kernels, result.derived
    row = monitor_step(inputs_for(SCHEME, xi, state), RAD, SCHEME.layout)
    assert row.core_cells == 5
    assert row.viscous_over_pressure_core == pytest.approx(np.max(np.abs(k.q[:5]) / (d.rho[:5] / 3.0)))
    assert row.viscous_over_pressure_core > 1e-3
    assert row.reconstruction_jump == pytest.approx(np.max(np.abs(k.rho_R[1:N] - k.rho_L[1:N]) / d.rho_f[1:N]))
    assert row.reconstruction_jump > 1e-3  # a smooth core: the jump is the second-order reconstruction error
