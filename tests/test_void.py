"""A strong rarefaction toward vacuum: the pull-apart void, and the kernel defects it and its neighbours exposed.

Near threshold the flow empties regions to densities far below the background: the fluid-orthogonal slicing's lapse
`rho^(-1/4)` makes slice time run fast there, and free expansion lowers the density exponentially in it. A scheme
must then keep every cell positive while its content becomes arbitrarily small, which is a property of each step
and not of convergence: refinement does not supply it. The standard test is two states pulling apart (the "1-2-3"
problem of Einfeldt et al. 1991). Here it is posed on the production equations: FRW at xi = 4, where the Hubble radius
is 7.4, with a peculiar velocity that is inward inside X = 2 and outward outside it, of amplitude V times the Hubble
flow there. The inner fluid falls, reflects at the origin and runs back out into the void it left, meeting the fluid
still falling in: a strong rarefaction beside a strong compression.

With the kernels as first printed (`ViscousFlux.AVERAGED`) the scheme loses positivity at V = 10, at a time that
depends on neither the Courant number (0.75 to 0.05) nor the integrator (RK4 or SSPRK3), and N = 100 survives where
N = 200 and 400 do not: the semi-discrete operator itself drives the cell negative. Four defects, found in turn:

* The viscous work term. Both one-sided fluxes carried the face average `<q>`, so a nearly empty cell beside a
  compression was drained through its face by its neighbour's viscous pressure, and the drain did not fall with the
  cell's own content. `ViscousFlux.DENSITY_WEIGHTED` carries each side's own `q / rho` at that side's reconstructed
  density, as the fluid pressure `w rho` already is in the same flux.
* The end cells' slopes. The first and last retained cells took an unlimited one-sided slope; near vacuum that put a
  face value 40 times above the cell's own density, and the HLL diffusion drained the cell through it. The slope was
  first clipped so that the cell's profile stayed non-negative between its faces, with every face value floored at
  `1e-12`. The theta-limiter replaces both: every cell's slope is scaled, where it must be, so that both its face
  values are at least theta times its density, which also bounds a face value's lapse. It never binds in resolved
  smooth flow but does in violent flow: it is part of the scheme, not a guard.
* The viscous defect again, at the excision face: there the first retained cell's `q` was carried unscaled at a face
  density floored to `1e-12`, whose lapse is a thousand, and the test below measures the drain that gives. It is now
  carried at the face density like the rest, and the theta-limiter holds that face density at theta times the cell's,
  so that the cell now loses content there at its own lapse. Near-threshold collapses lost their first retained cell
  about a third of an e-fold after formation, and this drain is the likely cause, but that is inferred, not
  demonstrated.
* Tension (found by the v3 review). In expansion the viscous pressure is a tension, and beside a nearly empty cell it
  exceeded the fluid pressure some 150-fold (q / rho = -52 against w = 1/3); the total pressure, and the enthalpy the
  flux carries, turned negative, and a face whose flow ran inward pumped energy out of the cell inside it. `cap_tension`
  holds `q >= -w rho`.

With all four fixed the void survives without a trapped surface for V = 10, 15 and 20 at N = 200, 400 and 800, emptying
to 5e-11 of the background (V = 20, N = 800; 7e-11 with the clip and floor the theta-limiter replaced). Stronger pulls
make a black hole at the centre instead: a trapped surface appears for V = 50 at N = 200 and for V = 30 and 50 at N =
400 and 800. Without excision those runs end three ways: with no trapped face left on the grid (V = 50 at N = 200), with
the trapped region still on it (V = 30 at N = 400), or in an abort with Gammabar^2 < 0 at the first faces (V = 50 at N =
400; V = 30 and 50 at N = 800); those stronger pulls were run with the clip and the floor, and the slow tests below
re-assert survival with the theta-limiter. Positivity is measured here, not proved: nothing in the scheme certifies it.
"""

import math

import numpy as np
import pytest
from modes import mode_errors

from pbh.derived import NotHyperbolicError
from pbh.eos import RADIATION, EquationOfState
from pbh.kernels import PRODUCTION_KERNELS, KernelSettings, ViscousFlux, viscous_sides
from pbh.layout import Layout
from pbh.maps import BlendMap, IdentityMap, Map, SinhStretch, Zone
from pbh.outer import OutgoingWave
from pbh.state import State
from pbh.timestep import COURANT_NUMBER, Integrator, Scheme, advance, courant_step

EOS = EquationOfState(RADIATION)
AVERAGED_Q = KernelSettings(
    viscous_flux=ViscousFlux.AVERAGED, cap_tension=False
)  # as first printed, but for the theta-limiter


def pull_apart(N: int, V: float, settings: KernelSettings, xi_end: float = 6.0) -> tuple[bool, float, float]:
    """Evolve the pull-apart void: `(survived, xi reached, smallest cell density seen)`."""
    sch = Scheme(EOS, IdentityMap(12.0), Layout(N), OutgoingWave(), settings)
    xi = 4.0
    geo = sch.frame(xi).geo
    X = geo.X[: N + 1]
    U = X + V * np.tanh((X - 2.0) / 0.2) * np.exp(-((X - 2.0) ** 2) / 2.0) * X / 2.0
    U[0] = 0.0
    dy = sch.layout.pack(State(E=geo.dV[:N].copy(), U=U, W=0.0)) - sch.frw(xi)
    rho_min = 1.0
    try:
        while xi < xi_end - 1e-12:
            res = sch.evaluate_deviation(xi, dy)
            rho_min = min(rho_min, float(np.min(res.derived.rho)))
            dxi = min(courant_step(res, geo, sch.layout, COURANT_NUMBER), xi_end - xi)
            dy = advance(sch, Integrator.RK4, xi, dy, dxi)
            xi += dxi
    except NotHyperbolicError:
        return False, xi, rho_min
    return True, xi, rho_min


def test_the_printed_flux_loses_positivity_in_the_void():
    survived, xi, _ = pull_apart(200, 10.0, AVERAGED_Q)
    assert not survived
    assert 4.85 < xi < 4.95  # about 0.9 e-folds in, whatever the step


def test_the_production_kernels_keep_the_void_positive():
    survived, _, rho_min = pull_apart(200, 10.0, PRODUCTION_KERNELS)
    assert survived
    assert rho_min < 1e-5  # the void is deep: this is not a mild rarefaction


@pytest.mark.slow
@pytest.mark.parametrize(("N", "V"), [(400, 10.0), (200, 15.0), (200, 20.0), (400, 20.0)])
def test_the_production_kernels_keep_stronger_voids_positive(N: int, V: float):
    survived, _, rho_min = pull_apart(N, V, PRODUCTION_KERNELS)
    assert survived
    assert rho_min < 1e-5


# --- the excision face: the first retained cell loses into the hole at its own rate ---


def excised_state(settings: KernelSettings, rho_e: float) -> tuple[float, float]:
    """A nearly empty first retained cell being compressed, beside the hole: `(F_je / E_je, rho^R_je / rho_je)`.

    The face is trapped and the infall is faster outward, so the cell is compressed and its viscous pressure positive.
    Its one-sided slope would carry its profile below zero at the face, so the theta-limiter holds the face value there
    at `theta rho_je`.
    """
    N, j_e = 40, 5
    sch = Scheme(EOS, SinhStretch(4.0, 2.0), Layout(N, j_e), OutgoingWave(), settings)
    geo = sch.frame(0.0).geo
    rho = np.ones(N)
    rho[j_e] = rho_e
    X = geo.X[: N + 1]
    U = -1.5 * X / X[j_e]  # infall, trapping the excision face ...
    U[j_e + 1 :] *= 2.0  # ... and a step faster from the first cell's outer face on: the cell is compressed
    E = np.where(np.arange(N) >= j_e, rho * geo.dV[:N], np.nan)
    M_e = 2.0 * X[j_e]  # 2m/R = 2 at the excision face
    res = sch.evaluate(0.0, sch.layout.pack(State(E=E, U=U, W=0.0, M_e=M_e)))
    assert U[j_e] + math.sqrt(res.derived.Gammabar2[j_e]) < 0.0  # trapped
    assert res.kernels is not None
    return float(res.F[j_e] / E[j_e]), float(res.kernels.rho_R[j_e] / rho_e)


@pytest.mark.parametrize("settings", [PRODUCTION_KERNELS, AVERAGED_Q])
def test_the_first_retained_cell_loses_into_the_hole_at_its_own_lapse(settings: KernelSettings):
    # The loss over the content scales as the cell's own lapse, rho^(-1/4): emptying the cell ten-thousandfold makes it
    # lose tenfold faster, not ten-thousandfold as a drain that ignored its content would. With the face value floored
    # at 1e-12 and q carried unscaled, as first printed, the averaged flux drained this cell at 1e4 times its content
    # per unit xi whatever the content; the theta-limiter keeps the face at theta rho under either viscous flux.
    rate_full, face_full = excised_state(settings, 1e-8)
    rate_empty, face_empty = excised_state(settings, 1e-12)
    theta = PRODUCTION_KERNELS.theta
    assert face_full == pytest.approx(theta, rel=1e-6)
    assert face_empty == pytest.approx(theta, rel=1e-3)
    assert rate_full < 0.0  # into the hole
    assert rate_empty / rate_full == pytest.approx(1e4**0.25, rel=1e-2)


# --- the density-weighted flux costs nothing where the flow is smooth ---


def test_the_production_kernels_are_exact_on_frw():
    sch = Scheme(EOS, SinhStretch(12.0, 3.0), Layout(100), OutgoingWave(), PRODUCTION_KERNELS)
    assert np.all(sch.deviation_rate(0.0, np.zeros(sch.layout.size)) == 0.0)


@pytest.mark.parametrize(
    "m",
    [
        IdentityMap(2.5),
        BlendMap(IdentityMap(2.5), float(EOS.alpha), (Zone(xi_on=0.0, tau_on=0.3, x_t=0.45, Delta_t=0.3),)),
    ],
)
@pytest.mark.parametrize("k_index", [0, 1])
def test_the_printed_flux_converges_at_second_order_on_the_bessel_modes(m: Map, k_index: int):
    # The production form is covered by test_timestep; the printed switch must keep second order too.
    errors = [mode_errors(m, k_index, N, AVERAGED_Q) for N in (40, 80, 160)]
    for field in (0, 1):
        rates = [math.log2(errors[i][field] / errors[i + 1][field]) for i in range(2)]
        assert min(rates) > 1.75, f"field {field}, mode {k_index}: L1 rates {rates}"


def test_the_viscous_pressure_is_carried_at_the_reconstructed_density():
    # Interior faces take each side's own q / rho at that side's face density; the excision face takes the first
    # retained cell's, at its face density. Written from the definition, not from the code.
    N, j_e = 8, 2
    rng = np.random.default_rng(3)
    q, rho = rng.normal(size=N), rng.uniform(0.1, 1.0, size=N)
    rho_L, rho_R = rng.uniform(0.1, 1.0, size=N + 1), rng.uniform(0.1, 1.0, size=N + 1)
    q_f = np.full(N + 1, 7.0)
    q_L, q_R = viscous_sides(q, q_f, rho, rho_L, rho_R, Layout(N, j_e), ViscousFlux.DENSITY_WEIGHTED)
    for j in range(j_e + 1, N):
        assert q_L[j] == pytest.approx(q[j - 1] / rho[j - 1] * rho_L[j])
        assert q_R[j] == pytest.approx(q[j] / rho[j] * rho_R[j])
    assert q_L[j_e] == pytest.approx(q[j_e] / rho[j_e] * rho_R[j_e])
    assert q_R[j_e] == pytest.approx(q[j_e] / rho[j_e] * rho_R[j_e])
    q_L, q_R = viscous_sides(q, q_f, rho, rho_L, rho_R, Layout(N, j_e), ViscousFlux.AVERAGED)
    assert q_L is q_f
    assert q_R is q_f


# --- tension: the viscous pressure in expansion (from the v3 review's test-strength track) ---


def tension_state(settings: KernelSettings, dip: float) -> tuple[float, float, float]:
    """An excised, trapped state whose first retained cell holds 1e-8 of the background, with a velocity dip at its
    outer face: the first cell is compressed and the next expands hard. Returns the fraction of the first cell's
    content leaving through face j_e and through face j_e + 1 in one Courant step, and the least `q / rho`."""
    N, j_e = 40, 8
    sch = Scheme(EOS, IdentityMap(4.0), Layout(N, j_e), OutgoingWave(), settings)
    geo = sch.frame(0.0).geo
    X = geo.X
    rho = np.ones(N)
    rho[j_e] = 1e-8
    E = np.full(N, np.nan)
    E[j_e:] = rho[j_e:] * geo.dV[j_e:]
    U = np.full(N + 1, np.nan)
    U[j_e:] = -1.5 * X[j_e:]
    U[j_e + 1] += dip
    res = sch.evaluate(0.0, sch.layout.pack(State(E=E, U=U, W=0.0, M_e=1.6)))
    assert res.kernels is not None
    dxi = courant_step(res, geo, sch.layout, COURANT_NUMBER)
    q_over_rho = res.kernels.q[j_e:] / res.derived.rho[j_e:]
    return float(-res.F[j_e] * dxi / E[j_e]), float(res.F[j_e + 1] * dxi / E[j_e]), float(np.min(q_over_rho))


@pytest.mark.parametrize("dip", [-2.0, -0.5])
@pytest.mark.parametrize("flux", list(ViscousFlux))
@pytest.mark.parametrize("cap", [False, True])
def test_a_tension_beyond_the_fluid_pressure_no_longer_drains_the_cell_beside_it(
    dip: float, flux: ViscousFlux, cap: bool
):
    # Uncapped, the expanding neighbour's tension reaches q / rho = -52 (or -12), the total pressure and with it the
    # enthalpy its flux carries turn negative, and the inward flow through face j_e + 1 pumps out some 1e7 times the
    # first cell's content in one step; capped at the fluid pressure, that face carries energy in, under either
    # viscous flux. Face j_e was the other defect: with its face value floored, the average took more than the cell's
    # whole content through it in one step. The theta-limiter now holds that face value at theta rho, so the cell loses
    # a share of its own content there under either flux, 15 to 40 per cent of it in this step, the density-weighted
    # flux the smaller share, and the cap does not enter.
    out_e, out_e1, least = tension_state(KernelSettings(viscous_flux=flux, cap_tension=cap), dip)
    if cap:
        assert least == pytest.approx(-float(EOS.w), rel=1e-12)  # the cap binds
        assert out_e1 < 0.0
    else:
        assert least < -10.0
        assert out_e1 > 1e6
    assert 0.0 < out_e < 0.5
    averaged, _, _ = tension_state(KernelSettings(viscous_flux=ViscousFlux.AVERAGED, cap_tension=cap), dip)
    if flux is ViscousFlux.DENSITY_WEIGHTED:
        assert out_e < averaged
