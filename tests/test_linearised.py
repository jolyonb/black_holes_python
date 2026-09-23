"""Tests of pbh.linearised and of the claims of Section 7.4 about the linearised base scheme."""

from fractions import Fraction

import numpy as np
import pytest

from pbh.eos import RADIATION, Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME
from pbh.layout import Layout
from pbh.linearised import energy_norm, jacobian, mass_perturbation, relative_scaling
from pbh.maps import IdentityMap, Map, SinhStretch
from pbh.outer import HeldAtFrw
from pbh.state import frw_state
from pbh.stencils import FaceClosure, StencilWeights
from pbh.types import FloatArray

type ComplexArray = np.ndarray[tuple[int], np.dtype[np.complex128]]

HELD = HeldAtFrw()
MAPS: list[Map] = [IdentityMap(4.0), SinhStretch(4.0, scale=2.0)]


def linearise_about_frw(m: Map, eos: EquationOfState, N: int = 24, xi: float = 0.5):
    """The operator L in the relative variables about FRW, with the geometry and background it was built on."""
    geo = Geometry.of(*m.radii(xi, N))
    bg = Background.at(eos, xi)
    lay = Layout(N)
    w = StencilWeights.of(geo, lay, FaceClosure.FIRST_ORDER)
    J = jacobian(frw_state(geo), geo, bg, eos, w, HELD, CENTRED_SCHEME)
    T = relative_scaling(geo, lay)
    L = (T[:, None] * J) / T[None, :]  # T J T^{-1}
    return L, geo, bg, lay


def interior(L: FloatArray, N: int) -> FloatArray:
    """The operator on the unknowns proper: drop delta_U,N (a datum with the face held) and W (unused)."""
    keep = np.r_[0 : 2 * N - 1]
    return L[np.ix_(keep, keep)]


# --- the Jacobian itself ---


def test_the_jacobian_is_insensitive_to_its_step():
    geo = Geometry.of(*SinhStretch(3.0, scale=2.0).radii(0.5, 16))
    eos = EquationOfState(RADIATION)
    bg, lay = Background.at(eos, 0.5), Layout(16)
    w = StencilWeights.of(geo, lay, FaceClosure.FIRST_ORDER)
    s = frw_state(geo)
    J_a = jacobian(s, geo, bg, eos, w, HELD, CENTRED_SCHEME, relative_step=1e-3)
    J_b = jacobian(s, geo, bg, eos, w, HELD, CENTRED_SCHEME, relative_step=2e-3)
    # Fourth-order differences: doubling the step changes the result at the truncation level, about 1e-9 relative.
    assert np.max(np.abs(J_a - J_b)) < 1e-8 * np.max(np.abs(J_a))


def test_the_relative_scaling_and_the_mass_map_are_the_printed_definitions():
    geo = Geometry.of(*IdentityMap(3.0).radii(0.0, 6))
    lay = Layout(6)
    T = relative_scaling(geo, lay)
    assert np.array_equal(T[:6], 1.0 / geo.dV)
    assert np.array_equal(T[6:12], 1.0 / geo.X[1:])
    assert T[12] == 1.0
    K = mass_perturbation(geo, lay)
    delta_rho = np.array([1.0, 0.5, 0.0, 2.0, 1.0, 1.0])
    delta_E = delta_rho * geo.dV
    delta_m = 3.0 * np.cumsum(delta_E)[:-1] / geo.X[1:6] ** 3  # faces 1..5
    assert K @ delta_rho == pytest.approx(delta_m)


def test_the_relative_variables_need_the_unexcised_grid_and_the_norm_needs_radiation():
    geo = Geometry.of(*IdentityMap(3.0).radii(0.0, 6))
    with pytest.raises(ValueError, match="unexcised"):
        relative_scaling(geo, Layout(6, j_e=2))
    dust_like = EquationOfState(RADIATION / 2)
    with pytest.raises(ValueError, match="radiation"):
        energy_norm(geo, Background.at(dust_like, 0.0), dust_like, Layout(6))


# --- the energy identity eq:num:energyid, as a matrix equation in the norm eq:num:norm (radiation) ---


@pytest.mark.parametrize("m", MAPS)
def test_the_linearised_scheme_satisfies_the_energy_identity_exactly(m: Map):
    eos = EquationOfState(RADIATION)
    L, geo, bg, lay = linearise_about_frw(m, eos)
    N = lay.N
    H = energy_norm(geo, bg, eos, lay)
    # d/dxi (y^T H y) = y^T (L^T H + H L + dH/dxi) y must equal the three printed terms:
    #   2 ||y||^2  -  2 sum_j (X_j^3 dS_j / 2) (delta_U,j + delta_m,j / 4)^2  -  3 c_s^2 X_N^3 delta_U,N delta_rho,N-1.
    # The weight moves in time through c_s^2 on the cells, d_xi c_s^2 = 2 (1 - alpha) c_s^2 (the paper's "using
    # d_xi c_s = c_s / 2 for the time derivative of the weight"); the face weights are static.
    face_weight = 0.5 * geo.X[1:N] ** 3 * geo.dS[1:N]
    dH_dxi = np.zeros_like(H)
    dH_dxi[:N, :N] = 2.0 * (1.0 - float(eos.alpha)) * np.diag(2.25 * bg.c_s**2 * geo.dV)
    B = np.zeros((N - 1, 2 * N + 1))  # the combination delta_U,j + delta_m,j / 4 at faces 1..N-1
    B[:, :N] = mass_perturbation(geo, lay) / 4.0
    B[:, N : 2 * N - 1] = np.eye(N - 1)
    boundary = np.zeros((2 * N + 1, 2 * N + 1))
    i_UN, i_rho = 2 * N - 1, N - 1
    boundary[i_UN, i_rho] = boundary[i_rho, i_UN] = -1.5 * bg.c_s**2 * geo.X[N] ** 3
    lhs = L.T @ H + H @ L + dH_dxi
    rhs = 2.0 * H - 2.0 * B.T @ np.diag(face_weight) @ B + boundary
    assert np.max(np.abs(lhs - rhs)) < 1e-10 * np.max(np.abs(rhs)), "eq:num:energyid, tolerance 1e-10"


@pytest.mark.parametrize("m", MAPS)
def test_the_acoustic_part_is_antisymmetric_in_the_diagonal_weight(m: Map):
    # Section 7.4: the flux divergence and the pressure gradient, symmetrised by H^(1/2), are exactly antisymmetric.
    eos = EquationOfState(RADIATION)
    L, geo, bg, lay = linearise_about_frw(m, eos)
    N = lay.N
    A = acoustic_operator(geo, bg, N)
    h = np.concatenate((2.25 * bg.c_s**2 * geo.dV, 0.5 * geo.X[1:N] ** 3 * geo.dS[1:N]))
    S = np.sqrt(h)[:, None] * A / np.sqrt(h)[None, :]
    assert np.max(np.abs(S + S.T)) < 1e-12 * np.max(np.abs(S))
    # and it is the derivative part of the linearised scheme: L restricted to the acoustic couplings equals A.
    Li = interior(L, N)
    assert Li[:N, N:] == pytest.approx(A[:N, N:], rel=1e-9, abs=1e-9 * np.max(np.abs(A)))


def acoustic_operator(geo: Geometry, bg: Background, N: int) -> FloatArray:
    """The derivative part of the linearised scheme in (delta_rho, delta_U) with delta_U,0 = delta_U,N = 0.

    Section 7.4: the flux divergence -2/3 (X_{j+1}^3 delta_U,j+1 - X_j^3 delta_U,j) / Delta V_c into delta_rho, and the
    pressure gradient -3 c_s^2 (delta_rho,c - delta_rho,c-1) / dS_j into delta_U at faces 1..N-1.
    """
    A = np.zeros((2 * N - 1, 2 * N - 1))
    X3 = geo.X**3
    for c in range(N):
        if c + 1 <= N - 1:
            A[c, N + c] = -2.0 / 3.0 * X3[c + 1] / geo.dV[c]  # the outer face of cell c (delta_U,N = 0 drops out)
        if c >= 1:
            A[c, N + c - 1] = 2.0 / 3.0 * X3[c] / geo.dV[c]  # the inner face of cell c (face 0 carries no velocity)
    for j in range(1, N):
        A[N + j - 1, j] = -3.0 * bg.c_s**2 / geo.dS[j]
        A[N + j - 1, j - 1] = 3.0 * bg.c_s**2 / geo.dS[j]
    return A


# --- the spectrum eq:num:spectrum ---


def sorted_complex(z: ComplexArray) -> ComplexArray:
    """Sort by real part, then by imaginary part, on keys rounded to 1e-8 so that round-off cannot reorder pairs.

    Every oscillatory eigenvalue of eq:num:spectrum has the same real part, so without rounding the order among them
    would be decided by noise in the real parts, differently in two sets that are equal to that noise.
    """
    real, imag = np.round(z.real, 8), np.round(z.imag, 8)
    return z[np.lexsort((imag, real))]


@pytest.mark.parametrize("w", [RADIATION, RADIATION / 2, 2 * RADIATION])
def test_the_spectrum_has_the_isolated_eigenvalue_and_is_symmetric_about_the_printed_real_part(w: Fraction):
    eos = EquationOfState(w)
    L, _, _, lay = linearise_about_frw(IdentityMap(4.0), eos)
    alpha = float(eos.alpha)
    lam = np.linalg.eigvals(interior(L, lay.N)).astype(np.complex128)
    isolated = 2.0 - 3.0 * alpha  # the total mass perturbation, which evolves alone
    k = np.argmin(np.abs(lam - isolated))
    assert abs(lam[k] - isolated) < 1e-8
    rest: ComplexArray = np.delete(lam, k)
    # Every other eigenvalue is (3 - 5 alpha) / 2 +- sqrt(...): the set is symmetric about (3 - 5 alpha) / 2.
    mirrored: ComplexArray = (3.0 - 5.0 * alpha) - rest
    assert sorted_complex(rest) == pytest.approx(sorted_complex(mirrored), abs=1e-7)


def test_the_oscillatory_wavenumbers_are_those_of_the_discrete_radial_laplacian():
    # eq:num:spectrum: the pair for each n has (lambda - (3 - 5 alpha) / 2)^2 = ((1 - alpha) / 2)^2 + alpha
    # - c_s^2 k_n^2, with k_n^2 the eigenvalues of -A_Urho A_rhoU, the Laplacian the same differences define.
    eos = EquationOfState(RADIATION)
    L, geo, bg, lay = linearise_about_frw(IdentityMap(4.0), eos)
    N, alpha = lay.N, float(eos.alpha)
    lam = np.linalg.eigvals(interior(L, N))
    lam = np.delete(lam, np.argmin(np.abs(lam - (2.0 - 3.0 * alpha))))
    k2_from_spectrum = (((1.0 - alpha) / 2.0) ** 2 + alpha - (lam - (3.0 - 5.0 * alpha) / 2.0) ** 2) / bg.c_s**2
    k2_from_spectrum = np.sort(np.unique(np.round(k2_from_spectrum.real, 6)))  # each k_n appears twice, as a pair
    A = acoustic_operator(geo, bg, N)
    laplacian = -(A[N:, :N] @ A[:N, N:]) / bg.c_s**2  # on delta_U at faces 1..N-1
    k2_from_laplacian = np.sort(np.linalg.eigvals(laplacian).real)
    assert k2_from_spectrum == pytest.approx(k2_from_laplacian, rel=1e-5)


def alternation(f: FloatArray) -> float:
    """The fraction of neighbouring entries that differ in sign: one for a sawtooth."""
    changes: FloatArray = (np.signbit(f[1:]) != np.signbit(f[:-1])).astype(np.float64)
    return float(np.mean(changes))


def test_the_sawtooth_is_the_stiffest_direction_not_a_null_one():
    # Section 7.4: the largest singular value of the symmetrised acoustic operator is within one per cent of
    # 2 c_s / min_c Delta X_c, independently of N, and its singular vectors are the sawtooths. The operator is skew in
    # the weighted norm, so its singular values come in equal pairs, the top pair a sawtooth on the cells and one on
    # the faces; which of the two an SVD returns first is the library's choice (it differs between macOS and Linux),
    # so the pair's subspace is tested: its cell pattern and its face pattern must each be a sawtooth.
    eos = EquationOfState(RADIATION)
    for N in (16, 32, 64):
        geo = Geometry.of(*IdentityMap(4.0).radii(0.5, N))
        bg = Background.at(eos, 0.5)
        A = acoustic_operator(geo, bg, N)
        h = np.concatenate((2.25 * bg.c_s**2 * geo.dV, 0.5 * geo.X[1:N] ** 3 * geo.dS[1:N]))
        S = np.sqrt(h)[:, None] * A / np.sqrt(h)[None, :]
        assert np.max(np.abs(S + S.T)) < 1e-14 * np.max(np.abs(S))  # skew
        _, sigma, vt = np.linalg.svd(S)
        assert sigma[0] == pytest.approx(2.0 * bg.c_s / np.min(geo.dX), rel=0.01)
        assert sigma[1] == pytest.approx(sigma[0], rel=1e-12)  # the pair
        pair = vt[:2]
        cells: FloatArray = np.linalg.svd(pair[:, :N])[2][0]  # the cell pattern the pair spans
        faces: FloatArray = np.linalg.svd(pair[:, N:])[2][0]
        assert alternation(cells) > 0.9
        assert alternation(faces) > 0.9


def test_no_eigenvalue_of_the_held_face_operator_grows_faster_than_the_growing_mode():
    # With the face held the bound is ||delta y|| <= e^xi ||delta y(0)||: no eigenvalue has real part above 1.
    eos = EquationOfState(RADIATION)
    L, _, _, lay = linearise_about_frw(SinhStretch(4.0, scale=2.0), eos)
    lam = np.linalg.eigvals(interior(L, lay.N))
    assert np.max(lam.real) <= 1.0 + 1e-8
