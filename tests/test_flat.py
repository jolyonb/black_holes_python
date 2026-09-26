"""Tests of the flat-spacetime limit of Section 7.7: the scheme without gravity, run by the production code path.

With the background coefficient `h = 0` the reference is the fluid at rest, the rows are special-relativistic
hydrodynamics of `P = w rho` in spherical symmetry, and the geometry, the origin and the kernels are the production
ones. The checks: the fluid at rest is an exact fixed point; the stage is the printed stage with the flat terms struck
(`whole_state.py`); energy is conserved exactly inside a rigid wall; the linear standing wave in a rigid sphere, an
exact Bessel mode, converges at second order through the origin; and the closures and layouts that need gravity refuse.
"""

import math
from fractions import Fraction

import numpy as np
import pytest
from whole_state import whole_state_rate

from pbh.eos import RADIATION, Background, EquationOfState, Spacetime
from pbh.equations import calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import CENTRED_SCHEME, PRODUCTION_KERNELS, KernelSettings
from pbh.layout import Layout
from pbh.maps import IdentityMap, Map, PinnedMap, SinhStretch
from pbh.outer import HeldAtFrw, HeldExterior, OuterInputs, OutgoingWave
from pbh.state import State, frw_rate, frw_state
from pbh.stencils import StencilWeights
from pbh.timestep import Scheme, advance_checked, step_size
from pbh.types import FloatArray

EOS = EquationOfState(RADIATION)
STIFF = EquationOfState(Fraction(1))
WALL = HeldAtFrw()  # held at the background: in flat spacetime, at rest
FLAT = Spacetime.FLAT


def flat_scheme(m: Map, N: int, settings: KernelSettings, eos: EquationOfState = EOS) -> Scheme:
    return Scheme(eos, m, Layout(N), WALL, settings, FLAT)


def nonlinear_state(geo: Geometry, seed: int) -> State:
    """A strongly nonlinear state at rest at the wall: rho in [0.2, 3] and |U| < 0.8, both varying on a few cells."""
    rng = np.random.default_rng(seed)
    N = geo.N
    X = geo.X[: N + 1]
    rho = 1.0 + 1.9 * np.sin(2.3 * geo.Xm + rng.uniform(0, 3)) * np.exp(-((geo.Xm - 2.0) ** 2) / 3.0)
    rho = np.clip(rho, 0.2, 3.0)
    U = 0.75 * np.sin(1.7 * X) * np.exp(-((X - 2.5) ** 2) / 2.0) * np.tanh(4.0 * X)
    U[0], U[N] = 0.0, 0.0
    return State(E=rho * geo.dV[:N], U=U, W=0.0)


# --- the background: the fluid at rest, a fixed point on every map ---


def test_the_flat_background_is_the_fluid_at_rest():
    frw = Background.at(EOS, 0.7)
    assert frw == Background.at(EOS, 0.7, Spacetime.FRW)
    assert frw.hubble == 1.0
    flat = Background.at(EOS, 0.7, FLAT)
    assert (flat.a, flat.H, flat.Gammabar2, flat.c_s, flat.hubble) == (1.0, 0.0, 1.0, 0.5 * EOS.sqrt_w, 0.0)
    assert flat.hubble_radius == math.inf  # no horizon ...
    assert flat.tau == math.inf  # ... and no big bang
    geo = Geometry.of(*IdentityMap(4.0).radii(0.0, 10))
    assert np.all(frw_state(geo, 0, 0.0).U == 0.0)


@pytest.mark.parametrize("eos", [EOS, STIFF])
@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, CENTRED_SCHEME])
@pytest.mark.parametrize("m", [IdentityMap(6.0), SinhStretch(6.0, scale=2.0), PinnedMap(IdentityMap(6.0), 0.5, 0.3)])
def test_the_fluid_at_rest_is_an_exact_fixed_point(eos: EquationOfState, settings: KernelSettings, m: Map):
    sch = flat_scheme(m, 40, settings, eos)
    xi = 0.8
    res = sch.evaluate_deviation(xi, np.zeros(sch.layout.size))
    assert np.all(res.deviation_rate.E == 0.0)
    assert np.all(res.deviation_rate.U == 0.0)
    geo = sch.frame(xi).geo
    rest = frw_rate(geo, 0, 0.0)
    assert np.all(res.rate.E == rest.E)  # the moving map's own transport of the uniform fluid, d_xi Delta V
    assert np.all(res.rate.U == 0.0)
    assert np.all(res.derived.Gammabar2 == 1.0)


# --- the stage is the printed stage with the flat terms struck ---


@pytest.mark.parametrize("eos", [EOS, STIFF])
@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, CENTRED_SCHEME])
@pytest.mark.parametrize("moving", [False, True])
def test_the_flat_stage_is_the_printed_stage_on_a_strongly_nonlinear_state(
    eos: EquationOfState, settings: KernelSettings, moving: bool
):
    N, xi = 60, 0.8
    radii, _ = SinhStretch(6.0, scale=2.0).radii(xi, N)
    X_xi = 0.2 * radii * np.exp(-(radii**2)) if moving else np.zeros_like(radii)
    X_xi[N:] = 0.0  # the outer face stays put
    geo = Geometry.of(radii, X_xi)
    bg = Background.at(eos, xi, FLAT)
    w = StencilWeights.of(geo, Layout(N))
    state = nonlinear_state(geo, 1)
    res = calc_derivs(state, geo, bg, eos, w, WALL, settings)
    ref = whole_state_rate(state, geo, bg, eos, w, None, settings)
    cells, faces = w.layout.cells, w.layout.faces
    assert np.max(np.abs(res.derived.delta_rho[cells])) > 1.0  # far from rest
    assert np.max(np.abs(res.rate.E[cells] - ref.E[cells]) / geo.dV[cells]) < 1e-11
    X = geo.X[: N + 1]
    scale = np.maximum(np.maximum(X[faces], X[1]), np.abs(ref.U[faces]))
    assert np.nanmax(np.abs(res.rate.U[faces] - ref.U[faces]) / scale) < 1e-12
    assert np.max(np.abs(res.F[faces] - ref.F[faces])) < 1e-12 * X[N] ** 3
    assert np.max(np.abs(res.derived.Gammabar2[1:] - (1.0 + state.U[1:] ** 2))) < 1e-15  # no gravity
    sp = res.speeds  # no Hubble flow: the energy crosses a face at (1 + w) times the fluid's own velocity
    fluid = float(eos.alpha) * res.derived.ephi_f[faces] * state.U[faces]
    assert np.max(np.abs(sp.cE[faces] - (1.0 + float(eos.w)) * fluid)) < 1e-15
    assert np.max(np.abs(sp.Theta[faces] - (fluid - geo.X_xi[faces]))) < 1e-15


@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, CENTRED_SCHEME])
def test_energy_is_conserved_exactly_inside_the_rigid_wall(settings: KernelSettings):
    # sum_c E_c is the Killing energy on the fluid slices; F_0 = 0 at the origin and F_N = 0 at the wall, and there
    # is no source, so the energy rows telescope to zero.
    sch = flat_scheme(SinhStretch(6.0, scale=2.0), 60, settings)
    geo = sch.frame(0.0).geo
    state = nonlinear_state(geo, 2)
    res = sch.evaluate(0.0, sch.layout.pack(state))
    assert res.F[0] == 0.0
    assert res.F[60] == 0.0
    assert abs(np.sum(res.rate.E)) <= 1e-15 * np.sum(state.E)


def test_the_held_face_is_a_rigid_wall_in_flat_spacetime():
    sch = flat_scheme(IdentityMap(4.0), 20, PRODUCTION_KERNELS)
    geo = sch.frame(0.0).geo
    rho = 1.0 + 0.5 * np.cos(geo.Xm)  # at rest, with any density profile
    res = sch.evaluate(0.0, sch.layout.pack(State(E=rho * geo.dV, U=np.zeros(21), W=0.0)))
    assert res.deviation_rate.U[20] == 0.0
    assert res.delta_F[20] == 0.0


# --- the exact linear solution: the Bessel mode in a rigid sphere, through the origin ---

Z_1 = 4.493409457909064  # the first zero of j1: U = 0 at the wall


def j1(x: FloatArray) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x > 1e-3, (np.sin(x) / x - np.cos(x)) / x, x / 3.0 - x**3 / 30.0)


def evolve_flat(sch: Scheme, state: State, xi_end: float, courant: float = 0.75) -> tuple[State, int]:
    """The production step rule and checked RK4 step from `xi = 0` to `xi_end`, landing on it: no step cap."""
    xi, dy = 0.0, sch.layout.pack(state) - sch.frw(0.0)
    geo = sch.frame(0.0).geo
    res = sch.evaluate_deviation(xi, dy)
    halvings = 0
    while xi < xi_end:
        dxi = step_size(res, geo, sch.layout, courant, math.inf).dxi
        land = xi_end if xi + dxi >= xi_end else None
        step = advance_checked(sch, xi, dy, xi_end - xi if land is not None else dxi, res, land)
        halvings += len(step.refused)
        dy, res = step.dy, step.result
        xi = land if land is not None and not step.refused else xi + step.dxi
    return sch.layout.unpack(sch.frw(xi) + dy), halvings


@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, CENTRED_SCHEME])
def test_the_bessel_mode_in_a_rigid_sphere_converges_at_second_order(settings: KernelSettings):
    # rho = 1 + A j0(kX) cos(omega xi), U = A sqrt(w) / (1 + w) j1(kX) sin(omega xi), omega = alpha sqrt(w) k, and
    # k X_N = z_1. Read an eighth past a period: at whole and half periods the density is at an extremum, its phase
    # error vanishes to leading order, and the centred scheme's density error falls to the nonlinear correction to the
    # linear mode, of relative size A, which does not converge. The cell contents are exact integrals,
    # int j0(kX) X^2 dX = X^2 j1(kX) / k.
    X_N, A = 10.0, 1e-6
    k = Z_1 / X_N
    w, alpha = float(EOS.w), float(EOS.alpha)
    omega = alpha * EOS.sqrt_w * k
    xi_end = 1.125 * 2.0 * math.pi / omega
    errors: list[tuple[float, float]] = []
    for N in (50, 100):
        sch = flat_scheme(IdentityMap(X_N), N, settings)
        geo = sch.frame(0.0).geo
        X = geo.X[: N + 1]
        content = np.diff(X**2 * j1(k * X) / k)
        E0 = geo.dV + A * content
        final, halvings = evolve_flat(sch, State(E=E0, U=np.zeros(N + 1), W=0.0), xi_end)
        assert halvings == 0
        E_exact = geo.dV + A * math.cos(omega * xi_end) * content
        U_exact = A * math.sqrt(w) / (1.0 + w) * j1(k * X) * math.sin(omega * xi_end)
        error_rho = np.sum(np.abs(final.E - E_exact)) / np.sum(np.abs(content)) / A
        error_U = np.sum(np.abs(final.U - U_exact)[:N] * geo.dX) / A
        errors.append((error_rho, error_U))
        assert abs(np.sum(final.E) / np.sum(E0) - 1.0) <= 1e-15
    for coarse, fine in zip(errors[0], errors[1], strict=True):
        assert math.log2(coarse / fine) >= 1.9


# --- what needs gravity refuses flat spacetime ---


def test_the_outgoing_wave_and_held_exterior_closures_and_excision_refuse_flat_spacetime():
    fields = dict.fromkeys(OuterInputs.__dataclass_fields__, 0.1)
    with pytest.raises(ValueError, match="flat-spacetime"):
        OutgoingWave().rows(OuterInputs(**(fields | {"X_xi_N": 0.0, "hubble": 0.0})), EOS)
    with pytest.raises(ValueError, match="flat-spacetime"):
        HeldExterior(rho_N=2.0, ephi_N=0.8).rows(OuterInputs(**(fields | {"hubble": 0.0})), EOS)
    with pytest.raises(ValueError, match="no black hole to excise"):
        Scheme(EOS, IdentityMap(4.0), Layout(20, 3), WALL, PRODUCTION_KERNELS, FLAT)


# --- a shock in flat spacetime: the production kernels capture it (the Taub measurement lives in analysis/) ---


def test_a_diverging_shock_is_captured_in_three_cells_and_keeps_its_books():
    # Rest data 4.013 | 1 across the face at R0 = 25 (compression 2 in the planar limit), to t = alpha xi = 1, with
    # the wall 5 beyond: nothing reaches it, and the far zone stays exactly at rest. The shocked state is the one
    # analysis/v4/experiments/e6_flat_tube measures against Taub (there extrapolated to the front: 1.9725, 0.3005); a
    # wrong flat term that still conserves energy moves it by far more than the tolerance.
    spacing, R0 = 0.006, 25.0
    j0 = round(R0 / spacing)
    N = j0 + math.ceil(5.0 / spacing)
    sch = flat_scheme(IdentityMap(N * spacing), N, PRODUCTION_KERNELS)
    geo = sch.frame(0.0).geo
    rho0 = np.where(np.arange(N) < j0, 4.013, 1.0)
    E0 = rho0 * geo.dV
    final, halvings = evolve_flat(sch, State(E=E0, U=np.zeros(N + 1), W=0.0), 1.0 / float(EOS.alpha))
    assert halvings == 0
    rho = final.E / geo.dV
    front = int(np.argmax(np.abs(np.diff(rho))))
    behind = float(np.median(rho[front - 30 : front - 6]))  # the shocked state, a little behind the front
    U_behind = float(np.median(final.U[front - 29 : front - 5]))
    assert abs(behind - 1.9726) < 1e-3
    assert abs(U_behind - 0.3020) < 1e-3
    taub = math.asinh(EOS.sqrt_w * (behind - 1.0) / ((1.0 + float(EOS.w)) * math.sqrt(behind)))  # eq:eul:taubeta
    assert abs(math.asinh(U_behind) / taub - 1.0) < 1e-2  # Taub, to the defect and the flow behind a spherical front
    between = (rho > 1.0 + 0.1 * (behind - 1.0)) & (rho < 1.0 + 0.9 * (behind - 1.0))
    assert np.sum(between[front - 10 : front + 12]) <= 3
    assert abs(np.sum(final.E) / np.sum(E0) - 1.0) <= 1e-14
    assert np.all(final.U[-200:] == 0.0)
    assert np.max(np.abs(rho[-200:] - 1.0)) <= 1e-15
