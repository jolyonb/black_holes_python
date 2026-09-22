"""Tests of pbh.initial: the growing-mode datum of Sections 5.4 and 7.9 and its sampling on the grid."""

import numpy as np
import pytest
from modes import J1_ZEROS, single_mode
from scipy.special import spherical_jn, spherical_yn

from pbh.derived import NotHyperbolicError, derive
from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.initial import (
    GrowingMode,
    IllPosedDataError,
    ModeExpansion,
    NotCompensatedError,
    cell_contents,
    initial_state,
    j1_over_x,
    j2_over_x2,
    nonlinear_correction,
    project,
)
from pbh.layout import Layout
from pbh.maps import IdentityMap, SinhStretch
from pbh.outer import characteristic_pair
from pbh.profiles import Gaussian
from pbh.stencils import FaceClosure, StencilWeights
from pbh.types import FloatArray

RAD = EquationOfState(RADIATION)
BG_0 = Background.at(RAD, 0.0)


def gaussian_at_peak_compaction(ell: float, C_peak: float = 0.515) -> Gaussian:
    """The Gaussian delta_m of peak compaction C_peak at xi = 0: X^2 delta_m peaks at sqrt 2 ell with 2 ell^2 A / e."""
    return Gaussian(A=C_peak * np.e / (2.0 * ell**2), ell=ell)


# --- the pieces ---


def test_the_bessel_quotients_are_scipys_with_their_limits_at_the_origin():
    x = np.array([0.0, 1e-12, 1e-6, 1e-2, 0.5, 3.0])
    assert j1_over_x(x)[0] == 1.0 / 3.0
    assert j2_over_x2(x)[0] == 1.0 / 15.0
    assert j1_over_x(x[1:]) == pytest.approx(spherical_jn(1, x[1:]) / x[1:], rel=1e-15)
    assert j2_over_x2(x[1:]) == pytest.approx(spherical_jn(2, x[1:]) / x[1:] ** 2, rel=1e-15)
    # and the limits are approached: the series 1/3 - x^2/30 and 1/15 - x^2/210 at x = 1e-6
    assert j1_over_x(x[2:3])[0] == pytest.approx(1.0 / 3.0 - 1e-12 / 30.0, rel=1e-14)
    assert j2_over_x2(x[2:3])[0] == pytest.approx(1.0 / 15.0 - 1e-12 / 210.0, rel=1e-14)


def test_a_single_mode_is_the_closed_form_of_section_5_2():
    # eq:lin:mode, modem, modeU for one wavenumber, written independently: delta_rho = tau k j_0(kR) B j_1(k tau),
    # delta_m = (3 tau / R) j_1(kR) B j_1(k tau), delta_U = -(3 tau / 4R) j_1(kR) B [j_1(k tau) - k tau j_2(k tau)].
    bg = Background.at(RAD, 0.7)
    k, B = 1.3, 2e-3
    mode = single_mode(k, B)
    X = np.array([0.4, 1.0, 2.2])
    z = k * bg.tau
    assert mode.delta_rho(bg, X) == pytest.approx(
        bg.tau * k * spherical_jn(0, k * X) * B * spherical_jn(1, z), rel=1e-14
    )
    assert mode.delta_m(bg, X) == pytest.approx(
        3.0 * bg.tau / X * spherical_jn(1, k * X) * B * spherical_jn(1, z), rel=1e-14
    )
    expected_U = -0.75 * bg.tau / X * spherical_jn(1, k * X) * B * (spherical_jn(1, z) - z * spherical_jn(2, z))
    assert mode.delta_U(bg, X) == pytest.approx(expected_U, rel=1e-14)
    # continuity at the origin and the node at the box edge for k at a zero of j_1
    node = single_mode(J1_ZEROS[0] / 2.5, B)
    assert node.delta_U(bg, np.array([0.0]))[0] == pytest.approx(node.delta_U(bg, np.array([1e-6]))[0], rel=1e-6)
    assert node.delta_U(bg, np.array([2.5]))[0] == pytest.approx(0.0, abs=1e-15)


def test_the_mass_derivatives_agree_with_finite_differences():
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "m", 12.0, BG_0)
    bg = Background.at(RAD, 0.3)
    X = np.array([0.0, 0.5, 1.7, 4.0])
    h = 1e-4
    delta_m, first, second = mode.delta_m_derivatives(bg, X)
    assert delta_m == pytest.approx(mode.delta_m(bg, X), rel=1e-14)
    plus, minus = mode.delta_m(bg, X + h), mode.delta_m(bg, np.abs(X - h))  # delta_m is even
    assert first[1:] == pytest.approx(((plus - minus) / (2 * h))[1:], rel=1e-6)
    assert first[0] == 0.0
    assert second == pytest.approx((plus - 2 * delta_m + minus) / h**2, rel=1e-5)


def test_the_projection_returns_the_coefficients_of_a_known_series():
    Rtilde_max = 6.0
    k = np.arange(1.0, 9.0) * (np.pi / Rtilde_max)
    c = np.array([0.5, -0.2, 0.1, 0.0, 0.05, 0.0, 0.0, 0.01])

    def density(X: FloatArray) -> FloatArray:
        return spherical_jn(0, np.multiply.outer(X, k)) @ c

    def mass(X: FloatArray) -> FloatArray:
        return 3.0 * j1_over_x(np.multiply.outer(X, k)) @ c

    assert project(density, "rho", k, Rtilde_max) == pytest.approx(c, abs=1e-13)
    assert project(mass, "m", k, Rtilde_max) == pytest.approx(c, abs=1e-13)


# --- the growing mode from a profile ---


def test_the_reconstruction_returns_the_input_profile():
    profile = Gaussian(A=0.05, ell=2.0)
    mode, report = GrowingMode.from_profile(profile, "m", 12.0, BG_0)
    X = np.linspace(0.0, 12.0, 241)
    assert mode.delta_m(BG_0, X) == pytest.approx(profile(X), abs=2e-7 * profile.A)  # 150 modes; tail at 12: 1.5e-8
    assert report.power_fraction < 1e-12  # Section 5.4: 5e-43 at ell = 2 in the continuum; quadrature round-off here
    assert abs(report.edge_value) < 2e-8 * profile.A  # a Gaussian delta_m is compensated to its tail
    # and the same perturbation given as a density, through the density projection, is the same mode

    def density(Xq: FloatArray) -> FloatArray:
        return mode.delta_rho(BG_0, Xq)

    again, _ = GrowingMode.from_profile(density, "rho", 12.0, BG_0)
    assert again.delta_rho(BG_0, X) == pytest.approx(density(X), abs=1e-9 * profile.A)
    assert again.B == pytest.approx(mode.B, abs=1e-9 * np.max(np.abs(mode.B)))


def test_data_that_leave_a_mass_excess_at_the_edge_are_refused():
    # A Gaussian delta_rho keeps its whole mass excess inside the box: delta_m at the edge is
    # 3 int X^2 delta_rho / Rtilde_max^3, far above the tolerance.
    with pytest.raises(NotCompensatedError, match="not compensated"):
        GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "rho", 12.0, BG_0)
    # A compensated mass profile on a box too small for its tail: delta_m(6) / A = e^-4.5 = 1e-2.
    with pytest.raises(NotCompensatedError, match="widen the box"):
        GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "m", 6.0, BG_0)
    # and the two-field constructor applies the same check
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "m", 12.0, BG_0)
    bg_i = Background.at(RAD, -1.0)
    with pytest.raises(NotCompensatedError):
        ModeExpansion.from_fields(lambda X: mode.delta_rho(bg_i, X) + 1e-3, lambda X: mode.delta_U(bg_i, X), 12.0, bg_i)


def test_the_power_fractions_of_the_ell_1_gaussian_are_the_prototypes():
    # Section 5.4 quotes 2e-9 at ell = 1 for the input power; on the box at Rtilde_max = 12 the projection gives
    # 4.4e-9, and the output velocity, amplified 26-fold on the mode 0.04 from the zero of j_1, 2.1e-8: these are the
    # Phase B prototype's numbers to three digits, and the second exceeds the tolerance, so the datum is refused.
    _, report = GrowingMode.from_profile(Gaussian(A=0.05, ell=1.0), "m", 12.0, BG_0, power_tolerance=1.0)
    assert report.power_fraction == pytest.approx(4.38e-9, rel=0.01)
    assert report.amplified_fraction == pytest.approx(2.14e-8, rel=0.01)
    assert report.distance_to_first_zero == pytest.approx(0.041, abs=0.001)
    with pytest.raises(IllPosedDataError, match="after the division"):
        GrowingMode.from_profile(Gaussian(A=0.05, ell=1.0), "m", 12.0, BG_0)


def test_a_profile_with_fine_structure_is_refused():
    with pytest.raises(IllPosedDataError, match="cannot be read from one field"):
        GrowingMode.from_profile(Gaussian(A=0.05, ell=0.5), "m", 12.0, BG_0)


def test_far_outside_the_sound_horizon_the_velocity_is_minus_a_quarter_of_the_mass():
    # Section 5.4: delta_U = -delta_m / 4 with the correction factor 1 - z^2 / 5 mode by mode; at ell = 8 the modes
    # that matter have z^2 of order 1 / (3 ell^2) = 0.005, so the departure is a few parts in a thousand.
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.01, ell=8.0), "m", 48.0, BG_0)
    X = np.linspace(0.0, 20.0, 41)
    departure = np.max(np.abs(mode.delta_U(BG_0, X) + mode.delta_m(BG_0, X) / 4.0)) / np.max(
        np.abs(mode.delta_m(BG_0, X) / 4.0)
    )
    assert 1e-4 < departure < 1e-2


def test_the_peak_compaction_of_the_gaussian_is_the_printed_value():
    # Section 5.4: C = X^2 delta_m e^-xi peaks at X = sqrt 2 ell with the value 2 ell^2 A / e.
    ell = 2.0
    mode, _ = GrowingMode.from_profile(gaussian_at_peak_compaction(ell), "m", 12.0, BG_0)
    X = np.linspace(0.0, 8.0, 8001)
    C = X**2 * mode.delta_m(BG_0, X)
    assert np.max(C) == pytest.approx(0.515, rel=1e-6)
    assert X[np.argmax(C)] == pytest.approx(np.sqrt(2.0) * ell, abs=2e-3)


# --- the general expansion of Section 5.4.1 ---


def test_two_fields_of_a_growing_mode_decompose_to_the_growing_branch_alone():
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "m", 12.0, BG_0)
    bg_i = Background.at(RAD, -1.0)
    expansion = ModeExpansion.from_fields(
        lambda X: mode.delta_rho(bg_i, X), lambda X: mode.delta_U(bg_i, X), 12.0, bg_i
    )
    scale = np.max(np.abs(mode.B))
    assert expansion.B == pytest.approx(mode.B, abs=1e-9 * scale)
    assert expansion.C == pytest.approx(np.zeros_like(mode.C), abs=1e-9 * scale)


def test_both_branches_are_recovered_from_their_two_fields_at_any_time():
    # eq:lin:modeinverse against eq:lin:modepair: the system's determinant is one, so an expansion with decaying
    # content is recovered from its density and velocity at the time it is given. The decaying content is a pair of
    # modes whose mass perturbations cancel at the edge, so that the data stay compensated: with
    # j_1(n pi) = (-1)^(n+1) / (n pi), the pair n = 2, 3 needs a_3 = (9/4) a_2.
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "m", 12.0, BG_0)
    bg_i = Background.at(RAD, 0.4)
    z = mode.k * bg_i.tau
    a_2 = 1e-3
    C = np.zeros_like(mode.B)
    C[1] = a_2 / (z[1] * spherical_yn(1, z[1]))
    C[2] = 2.25 * a_2 / (z[2] * spherical_yn(1, z[2]))
    given = ModeExpansion(k=mode.k, B=mode.B, C=C)
    found = ModeExpansion.from_fields(lambda X: given.delta_rho(bg_i, X), lambda X: given.delta_U(bg_i, X), 12.0, bg_i)
    scale = np.max(np.abs(mode.B))
    assert found.B == pytest.approx(given.B, abs=1e-9 * scale)
    assert found.C == pytest.approx(given.C, abs=1e-9 * scale)


def test_decaying_content_outside_the_sound_horizon_falls_as_the_cube_of_the_sound_horizon():
    # Section 5.4.1: for z << 1 the decaying density goes as z y_1(z) ~ -1/z against the growing z j_1(z) ~ z^2/3,
    # so their ratio falls as tau^-3 between two times.
    k = np.array([0.05])
    growing = ModeExpansion(k=k, B=np.array([1.0]), C=np.array([0.0]))
    decaying = ModeExpansion(k=k, B=np.array([0.0]), C=np.array([1e-6]))
    X = np.array([1.0])
    ratios: list[float] = []
    for xi in (0.0, 2.0 * np.log(2.0)):  # tau doubles
        bg = Background.at(RAD, xi)
        ratios.append(float(decaying.delta_rho(bg, X)[0] / growing.delta_rho(bg, X)[0]))
    assert ratios[0] / ratios[1] == pytest.approx(8.0, rel=2e-3)


def test_an_expansion_samples_through_the_door_and_the_growing_mode_without_the_correction_is_its_own_sampling():
    mode, _ = GrowingMode.from_profile(gaussian_at_peak_compaction(2.0), "m", 12.0, BG_0)
    geo = Geometry.of(*IdentityMap(12.0).radii(0.0, 100))
    bg = Background.at(RAD, 0.5)
    sampled = ModeExpansion.state(mode, geo, bg)  # the general sampling, no correction
    uncorrected = mode.state(geo, bg, nonlinear=False)
    assert np.array_equal(sampled.E, uncorrected.E)
    assert np.array_equal(sampled.U, uncorrected.U)
    assert sampled.W == uncorrected.W
    assert sampled.U[0] == 0.0
    assert sampled.W != 0.0


# --- the nonlinear correction ---


def test_the_correction_is_the_printed_quadratic_form():
    X = np.array([0.5, 1.0, 2.0])
    m, m1, m2 = np.array([0.1, 0.08, 0.02]), np.array([-0.05, -0.06, -0.03]), np.array([-0.1, -0.02, 0.01])
    expected = (m**2 + 12 * X * m * m1 + 4 * X**2 * m * m2 - X**2 * m1**2) / 160
    assert nonlinear_correction(X, m, m1, m2) == pytest.approx(expected, rel=1e-15)
    assert nonlinear_correction(X, 2 * m, 2 * m1, 2 * m2) == pytest.approx(4 * expected, rel=1e-15)  # quadratic
    assert nonlinear_correction(X, 0 * m, 0 * m1, m2) == pytest.approx(
        0.0, abs=0.0
    )  # vanishes with delta_m and delta_m'


@pytest.mark.parametrize(("ell", "percent"), [(2.0, 3.6), (4.0, 0.9), (8.0, 0.2)])
def test_the_correction_ratio_is_the_printed_percentage_at_peak_compaction(ell: float, percent: float):
    # Section 7.9: max |delta_U^nl / delta_U^lin| is 3.6, 0.9 and 0.2 per cent for the Gaussian of peak compaction
    # 0.515 at ell = 2, 4 and 8 (verified there).
    Rtilde_max = 6.0 * ell
    mode, _ = GrowingMode.from_profile(gaussian_at_peak_compaction(ell), "m", Rtilde_max, BG_0)
    geo = Geometry.of(*IdentityMap(Rtilde_max).radii(0.0, 600))
    assert 100.0 * mode.correction_ratio(geo, BG_0) == pytest.approx(percent, abs=0.06)


# --- the state ---


@pytest.mark.parametrize("m", [IdentityMap(12.0), SinhStretch(12.0, scale=3.0)])
def test_the_state_holds_the_mass_exactly_with_the_velocity_corrected_and_w_at_the_incoming_amplitude(
    m: IdentityMap | SinhStretch,
):
    mode, _ = GrowingMode.from_profile(gaussian_at_peak_compaction(2.0), "m", 12.0, BG_0)
    geo = Geometry.of(*m.radii(0.0, 200))
    N = geo.N
    X = geo.X[: N + 1]
    state = mode.state(geo, BG_0)
    mt = 3.0 * np.cumsum(state.E)[:-1] / X[1:N] ** 3  # the cumulative sum at faces 1..N-1
    assert mt == pytest.approx(1.0 + mode.delta_m(BG_0, X[1:N]), abs=2e-15)  # eq:num:idata, exact
    assert state.U[0] == 0.0
    delta_m, first, second = mode.delta_m_derivatives(BG_0, X)
    corrected = mode.delta_U(BG_0, X) + nonlinear_correction(X, delta_m, first, second)
    assert state.U[1:] == pytest.approx(X[1:] * (1.0 + corrected[1:]), rel=1e-14)
    delta_U_N = (state.U[N] - X[N]) / X[N]
    delta_rho_N_1 = (state.E[N - 1] - geo.dV[N - 1]) / geo.dV[N - 1]
    assert state.W == characteristic_pair(float(delta_U_N), float(delta_rho_N_1), float(X[N]), BG_0.c_s)[1]
    assert 0.0 < mode.correction_ratio(geo, BG_0) < 0.05
    # and the data are admissible, as derive agrees
    derive(state, geo, BG_0, RAD, StencilWeights.of(geo, Layout(N), FaceClosure.FIRST_ORDER))


def test_the_correction_can_be_switched_off():
    mode, _ = GrowingMode.from_profile(gaussian_at_peak_compaction(2.0), "m", 12.0, BG_0)
    geo = Geometry.of(*IdentityMap(12.0).radii(0.0, 100))
    X = geo.X[: geo.N + 1]
    state = mode.state(geo, BG_0, nonlinear=False)
    assert state.U[1:] == pytest.approx(X[1:] * (1.0 + mode.delta_U(BG_0, X[1:])), rel=1e-14)


def test_inadmissible_data_are_refused_by_the_field_that_fails():
    # Section 5.4: Gammabar^2 > 0 bounds the compaction by 2/3 for radiation; at peak compaction 0.7 it fails.
    geo = Geometry.of(*IdentityMap(12.0).radii(0.0, 100))
    mode, _ = GrowingMode.from_profile(gaussian_at_peak_compaction(2.0, C_peak=0.7), "m", 12.0, BG_0)
    with pytest.raises(NotHyperbolicError, match="Gammabar2"):
        mode.state(geo, BG_0)
    # A density that goes negative: the lowest box mode with a central density of -1/2.
    k_1 = np.pi / 12.0
    z_1 = k_1 * BG_0.tau
    mode = single_mode(k_1, B=-1.5 / (z_1 * spherical_jn(1, z_1)))
    with pytest.raises(NotHyperbolicError, match="rho"):
        mode.state(geo, BG_0)


def test_complete_data_from_any_source_go_through_the_same_door():
    # A converging shell of the test table: a density with a simple-wave velocity of its own, not a growing mode.
    geo = Geometry.of(*IdentityMap(5.0).radii(3.0, 100))
    bg = Background.at(RAD, 3.0)
    X = geo.X[: geo.N + 1]

    def delta_rho(Xq: FloatArray) -> FloatArray:
        return 0.3 * (np.exp(-0.5 * ((Xq - 2.0) / 0.15) ** 2) + np.exp(-0.5 * ((Xq + 2.0) / 0.15) ** 2))

    E = cell_contents(delta_rho, geo)
    U = X - 1.5 * bg.c_s * delta_rho(X)  # u_+ = 0, the ingoing simple wave; U_0 is set by the door
    U[0] = 7.0
    state = initial_state(E, U, geo, bg)
    assert state.U[0] == 0.0
    assert state.U[1:] == pytest.approx(U[1:])
    assert state.E is E
    delta_U_N, delta_rho_N_1 = (U[-1] - X[-1]) / X[-1], (E[-1] - geo.dV[-1]) / geo.dV[-1]
    assert state.W == characteristic_pair(float(delta_U_N), float(delta_rho_N_1), float(X[-1]), bg.c_s)[1]
    assert initial_state(E, U, geo, bg, W=0.25).W == 0.25  # a known incoming amplitude is kept
    with pytest.raises(NotHyperbolicError, match="rho"):
        initial_state(-E, U, geo, bg)


def test_the_quadrature_contents_of_a_density_equal_the_exact_contents_of_its_mass():
    mode, _ = GrowingMode.from_profile(Gaussian(A=0.05, ell=2.0), "m", 12.0, BG_0)
    geo = Geometry.of(*SinhStretch(12.0, scale=3.0).radii(0.0, 150))
    X = geo.X[: geo.N + 1]
    exact = np.diff(X**3 * (1.0 + mode.delta_m(BG_0, X))) / 3.0

    def density(Xq: FloatArray) -> FloatArray:
        return mode.delta_rho(BG_0, Xq)

    assert cell_contents(density, geo) == pytest.approx(exact, rel=1e-13)
