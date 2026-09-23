"""The shock-capturing kernels (paper Section 7.7: eq:num:recon, eq:num:hll, eq:num:jump, eq:num:qvisc).

Three kernels turn the centred base scheme of `equations.py` into one that captures shocks, and they replace exactly
two things there: the energy flux `F_j` and the artificial pressure force `Q_j`. None contains a tunable constant
beyond `c_v = 1`, none is switched on by a detector, and all reduce to the base scheme on FRW of every map:

(a) The transported density is reconstructed to both sides of every face by a piecewise-linear profile in `s = X^2`
    with the monotonized-central limiter (eq:num:recon). The reconstruction is in `s` for the reason of Section 7.2:
    in `X` with mirrored ghosts it would be first order at the first cells. The first and last retained cells take
    their single adjacent one-sided difference, clipped only so far that the cell's linear profile in `s` stays
    non-negative between its faces, and every face value is floored at `rho_floor`. The clip is a positivity limiter:
    it never binds in resolved smooth flow, but it does bind in violent flow, under either viscous flux.
(b) The energy flux is the HLL flux of the one-sided single-variable flux `F_j(rho)`, which carries the lapse of the
    same reconstructed density and the work term of the viscous pressure (eq:num:hll). Without the work term its
    derivative with respect to `X^2 rho` is the grid velocity at that side's own lapse, close to `Theta_j` and so inside
    the HLL bounds `Theta_j +- a_j` unless the lapse jumps across the face; the work term adds a speed of order
    `q / rho`. No donor velocity has to be chosen, and on FRW it is the centred flux of eq:num:energy. It is returned
    as its deviation from the FRW flux, which the energy rows need (`equations.py`), and the reconstruction works in
    the density's deviation for the same reason.
(c) The peculiar velocity `upsilon = U - X` is reconstructed to the cell midpoints from both faces with the minmod
    limiter, and the limited jump across each cell (eq:num:jump), the full jump at a shock and `O(Delta X^2)` where the
    flow is smooth, is fed back as a viscous pressure on the cells (eq:num:qvisc), normalised like a Rusanov term
    with the signal speed. It enters the velocity equation as the areal force `Q_j = X_j^-2 (D_s (sbar q))_j` and the
    energy flux as a pressure: each one-sided flux carries its own cell's `q / rho` at that side's reconstructed
    density (`ViscousFlux.DENSITY_WEIGHTED`; the face average `<q>_j` as first printed is the switch `AVERAGED`),
    and in expansion its tension is capped at the fluid pressure, `q >= -w rho` (`cap_tension`), so that the total
    pressure and the enthalpy the flux carries stay non-negative.
    The peculiar velocity is reconstructed, not `U`, because it vanishes on FRW of every map and because it is what
    makes the momentum dissipation exactly dissipative in the energy weight of Section 7.4; the velocity limiter is
    minmod and never a compressive one (Section 7.7). The taper `q_{N-1} = 0` is the switch-off the outer closure of
    Section 7.5 asks for.

Positivity is measured, not proved: the pull-apart void of test_void and the near-threshold collapses keep every cell
positive with these kernels, but nothing here certifies it, and RK4 is not strong-stability-preserving.

At an excision face `j_e` (Section 8.3) the kernels add their own one-sided rows and no others: the reconstructed
density from inside the face is the value from outside, `rho^L_je = rho^R_je`; the first retained cell's density
slope is its single one-sided difference and the velocity slope at the face the single adjacent difference, as at
faces `0` and `N`; the viscous pressure enters the flux as the first retained cell's `q / rho` at its reconstructed
face density (under `AVERAGED`, its unscaled cell value `<q>_je = q_je`), and its force is the
end row `Q_je = 2 sbar_je q_je / (X_je^2 Delta X_je)`, the one-sided gradient over the half cell with `q` vanishing at
the face.

The centred base scheme remains available as a test switch (`Kernels.CENTRED`); Section 7.4 says why it is never a
production configuration.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np

from pbh.derived import Derived
from pbh.eos import EquationOfState
from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


class Kernels(Enum):
    """Whether the shock-capturing kernels are on (production) or the centred base scheme runs (test switch)."""

    PRODUCTION = "production"
    """The three kernels of Section 7.7."""

    CENTRED = "centred"
    """The centred base flux and no artificial pressure: Section 7.3 as printed, a test switch only."""


class DensityLimiter(Enum):
    """The limiter of the density reconstruction (Table tab:num:params: mc recommended, minmod admissible)."""

    MC = "mc"
    """Monotonized central: monotone in two to three cells."""

    MINMOD = "minmod"
    """More diffusive by one cell per shock, but with an energy certificate everywhere (Section 7.7)."""


class ViscousFlux(Enum):
    """How the viscous work enters the energy flux of eq:num:hll: density-weighted, or as first printed."""

    AVERAGED = "averaged"
    """Both one-sided fluxes carry the face average `<q>_j` of eq:num:stencils, and the excision face the first
    retained cell's unscaled `q_je`: with `cap_tension` off, the scheme as first printed except for the end-cell
    clip of the density reconstruction, which is not switchable."""

    DENSITY_WEIGHTED = "density_weighted"
    """Each one-sided flux carries its own cell's viscous pressure, at that side's reconstructed density (see
    `viscous_sides`), so the HLL weights upwind the work term like the transport. With the average, a nearly empty
    cell next to a strong compression is drained through its face by its neighbour's `q`, at a rate that does not
    vanish with its own content, and its density goes negative in finite time at any step size (test_void)."""


@dataclass(frozen=True)
class KernelSettings:
    """The kernel switches of Table tab:num:params.

    Attributes:
        kernels: Production kernels, or the centred base scheme.
        density_limiter: The limiter of the density reconstruction; the velocity limiter is minmod and not a choice.
        c_v: The one constant of the viscous pressure, `1`.
        rho_floor: The floor on every reconstructed face density, `1e-12`.
        viscous_flux: The viscous work term of the energy flux: density-weighted (production) or the face average.
        cap_tension: Whether the viscous tension is capped at the fluid pressure, `q >= -w rho` (production).
    """

    kernels: Kernels = Kernels.PRODUCTION
    density_limiter: DensityLimiter = DensityLimiter.MC
    c_v: float = 1.0
    rho_floor: float = 1e-12
    viscous_flux: ViscousFlux = ViscousFlux.DENSITY_WEIGHTED
    cap_tension: bool = True


PRODUCTION_KERNELS = KernelSettings()
CENTRED_SCHEME = KernelSettings(kernels=Kernels.CENTRED)


@dataclass(frozen=True)
class KernelResult:
    """What the kernels produce at one evaluation, for the equations and for the monitors.

    Attributes:
        rho_L: The density reconstructed to face `j` from the cell inside it (faces; NaN where not formed).
        rho_R: The density reconstructed to face `j` from the cell outside it.
        delta_rho_L: `rho_L - 1`, formed without subtracting one.
        delta_rho_R: `rho_R - 1`, likewise.
        J: The limited jump of the peculiar velocity across each cell (cells).
        q: The artificial viscous pressure on the cells.
        q_f: Its face value `<q>_j`.
        Q: Its force at the faces, the areal form of eq:num:qvisc.
        F: The HLL energy flux through the retained faces `j < N` (face `N` is the outer closure's).
    """

    rho_L: FloatArray
    rho_R: FloatArray
    delta_rho_L: FloatArray
    delta_rho_R: FloatArray
    J: FloatArray
    q: FloatArray
    q_f: FloatArray
    Q: FloatArray
    F: FloatArray


def minmod(*slopes: FloatArray) -> FloatArray:
    """The minmod of two or three arrays: the one of smallest modulus where all agree in sign, zero otherwise."""
    stacked = np.stack(slopes)
    all_positive = np.all(stacked > 0.0, axis=0)
    all_negative = np.all(stacked < 0.0, axis=0)
    smallest = np.min(np.abs(stacked), axis=0)
    return np.where(all_positive, smallest, np.where(all_negative, -smallest, 0.0))


def reconstruct_density(
    delta_rho: FloatArray, geo: Geometry, w: StencilWeights, limiter: DensityLimiter, floor: float
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The density to both sides of every retained face, piecewise linear in `s = X^2` (eq:num:recon).

    The reconstruction is linear and the limiters see only differences, so it is carried out on the deviation
    `rho - 1` and one is added back: in exact arithmetic the same values, without the rounding of `rho` to its FRW
    size in the slopes and in the flux that uses the deviations.

    Args:
        delta_rho: The cell densities' deviations `rho - 1`, formed without subtracting one (see `derive`).
        geo: The geometry.
        w: The stencil weights, for the retained ranges.
        limiter: mc or minmod for the interior cells.
        floor: The floor applied to every face density.

    Returns:
        `(rho_L, rho_R, delta_rho_L, delta_rho_R)` at the faces: the density from the cell inside the face and from
        the cell outside it, and their deviations; a floored value's deviation is `floor - 1`. At the origin and at an
        excision face `rho_L = rho_R`; at the outer face `rho_R = rho_L`.
    """
    layout = w.layout
    N, j_e = layout.N, layout.j_e
    X2, sbar, dS = geo.X**2, geo.sbar, geo.dS
    # The one-sided slopes d_j across the interior retained faces j_e+1 .. N-1, indexed by face.
    d = np.full(N + 1, np.nan)
    d[j_e + 1 : N] = (delta_rho[j_e + 1 : N] - delta_rho[j_e : N - 1]) / dS[j_e + 1 : N]
    # The limited slope of every retained cell: the interior cells from their two faces, the first and last retained
    # cells from their single adjacent difference, clipped below.
    slope = np.full(N, np.nan)
    c = np.arange(j_e + 1, N - 1)
    d_in, d_out = d[c], d[c + 1]
    if limiter is DensityLimiter.MC:
        r_L = dS[c] / (sbar[c] - X2[c])
        r_R = dS[c + 1] / (X2[c + 1] - sbar[c])
        slope[c] = minmod(0.5 * (d_in + d_out), r_L * d_in, r_R * d_out)
    else:
        slope[c] = minmod(d_in, d_out)
    slope[j_e] = d[j_e + 1]
    slope[N - 1] = d[N - 1]
    # The two end cells keep their one-sided difference for second order at the origin, but not so far that their
    # profile turns negative inside the cell: near vacuum the unlimited slope put a face value tens of times above the
    # cell's own density and the HLL diffusion then drained the cell through it (the pull-apart void at V = 15,
    # test_void). The clip binds only where the linear profile in s would cross zero between the cell's faces: never
    # in resolved smooth flow, but in violent flow it does, under either viscous flux.
    for e in (j_e, N - 1):
        rho_e = 1.0 + delta_rho[e]
        slope[e] = min(max(slope[e], -rho_e / (X2[e + 1] - sbar[e])), rho_e / (sbar[e] - X2[e]))
    delta_L = np.full(N + 1, np.nan)
    delta_R = np.full(N + 1, np.nan)
    inside = slice(j_e, N)  # cell c is inside face c + 1 ...
    delta_L[j_e + 1 : N + 1] = delta_rho[inside] + slope[inside] * (X2[j_e + 1 : N + 1] - sbar[inside])
    delta_R[j_e:N] = delta_rho[inside] + slope[inside] * (X2[j_e:N] - sbar[inside])  # ... and outside face c
    delta_L[j_e] = delta_R[j_e]  # nothing inside the innermost face: transmissive (F_0 = 0 anyway at the origin)
    delta_R[N] = delta_L[N]
    rho_L, rho_R = np.maximum(1.0 + delta_L, floor), np.maximum(1.0 + delta_R, floor)
    delta_L[1.0 + delta_L < floor] = floor - 1.0
    delta_R[1.0 + delta_R < floor] = floor - 1.0
    return rho_L, rho_R, delta_L, delta_R


def viscous_pressure(
    state: State,
    geo: Geometry,
    d: Derived,
    Lam: FloatArray,
    eos: EquationOfState,
    w: StencilWeights,
    c_v: float,
    cap_tension: bool,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """The limited velocity jump, the viscous pressure, its face value and its areal force (eq:num:jump, qvisc).

    Args:
        state: The state, for the face velocities.
        geo: The geometry.
        d: The derived fields, for the cell density and lapse and the face `Gammabar^2`.
        Lam: The signal speeds `Lambda_j` at the faces.
        eos: The equation of state.
        w: The stencil weights.
        c_v: The viscosity constant, `1`.
        cap_tension: Whether the tension is capped at the fluid pressure, `q >= -w rho` (production).

    Returns:
        `(J, q, q_f, Q)`: the jump and the pressure on the cells, the face value and the force at the faces.
    """
    layout = w.layout
    N, j_e = layout.N, layout.j_e
    X, Xm, dX = geo.X, geo.Xm, geo.dX
    alpha, w_eos = float(eos.alpha), float(eos.w)
    cells = layout.cells
    # (c) The peculiar velocity, its slope in each cell, and the minmod-limited slope at each face: the single
    # adjacent difference at the innermost retained face and at the outer face.
    upsilon = X * d.delta_U  # U - X, from the deviation rather than by subtracting (see `derive`)
    if j_e == 0:
        upsilon[0] = 0.0  # U_0 = X_0 = 0
    g = np.full(N, np.nan)
    g[cells] = (upsilon[j_e + 1 : N + 1] - upsilon[j_e:N]) / dX[cells]
    g_f = np.full(N + 1, np.nan)
    g_f[j_e + 1 : N] = minmod(g[j_e : N - 1], g[j_e + 1 : N])
    g_f[j_e] = g[j_e]
    g_f[N] = g[N - 1]
    # The limited jump across each cell: the profiles from its two faces, evaluated at the midpoint.
    J = np.full(N, np.nan)
    inner, outer = slice(j_e, N), slice(j_e + 1, N + 1)
    J[cells] = (upsilon[outer] + g_f[outer] * (Xm[cells] - X[outer])) - (
        upsilon[inner] + g_f[inner] * (Xm[cells] - X[inner])
    )
    # The viscous pressure on the cells, normalised by the signal speed and the inertia factor, tapered at the edge.
    Lam_hat = np.maximum(Lam[inner], Lam[outer])
    Gammabar2_hat = 0.5 * (d.Gammabar2[inner] + d.Gammabar2[outer])
    q = np.full(N, np.nan)
    q[cells] = -0.5 * c_v * Lam_hat * (1.0 + w_eos) * d.rho[cells] / (alpha * d.ephi[cells] * Gammabar2_hat) * J[cells]
    if cap_tension:
        # In expansion the Rusanov-normalised q is a tension. In an under-resolved core or beside a nearly empty
        # cell it can exceed the fluid pressure many times over, making the total pressure, and with it the enthalpy
        # the flux carries, negative, so that a face whose flow runs inward pumps energy outward and drains the cell
        # inside it (test_void). Capping the tension at the fluid pressure keeps the total pressure non-negative; it
        # only shrinks |q|, so the dissipation keeps its sign.
        q[cells] = np.maximum(q[cells], -w_eos * d.rho[cells])
    q[N - 1] = 0.0
    # Its face value (the stencil's, which is the cell behind an excision face) and its areal force: the interior
    # stencil, and the kernels' own end row at the excision face.
    q_f = w.face_average(q)
    Q = np.full(N + 1, np.nan)
    Q[j_e + 1 : N] = w.gradient_s(geo.sbar[:-1] * q)[j_e + 1 : N] / X[j_e + 1 : N] ** 2
    if j_e > 0:
        Q[j_e] = 2.0 * geo.sbar[j_e] * q[j_e] / (X[j_e] ** 2 * dX[j_e])
    else:
        Q[0] = 0.0  # face 0 has no velocity equation
    return J, q, q_f, Q


def viscous_sides(
    q: FloatArray,
    q_f: FloatArray,
    rho: FloatArray,
    rho_L: FloatArray,
    rho_R: FloatArray,
    layout: Layout,
    mode: ViscousFlux,
) -> tuple[FloatArray, FloatArray]:
    """The viscous pressure each one-sided flux of eq:num:hll carries at the faces: `(q_L, q_R)`.

    `AVERAGED` gives `<q>_j` to both, the printed flux. `DENSITY_WEIGHTED` treats `q` as the pressure it is: each side
    carries its own cell's `q / rho` at the density reconstructed to the face on that side, `q_L = (q / rho)_(c-1)
    rho_L`, `q_R = (q / rho)_c rho_R`, just as the fluid pressure in the same flux is `w` times the reconstructed
    density; at the excision face both sides are the first retained cell's, at its reconstructed face density.
    Both agree on FRW (`q = 0`); at the interior faces they differ by `O(Delta X)` times `q` in smooth flow, where
    `q` is `O(Delta X^2)`. At face `N - 1` they differ at `O(q)`, since the taper `q_{N-1} = 0` makes the average
    half the inner cell's value while the density-weighted outer side carries none of it.
    """
    if mode is ViscousFlux.AVERAGED:
        return q_f, q_f
    N, j_e = layout.N, layout.j_e
    inner, outer = slice(j_e, N - 1), slice(j_e + 1, N)
    q_L, q_R = q_f.copy(), q_f.copy()
    q_L[j_e + 1 : N] = q[inner] / rho[inner] * rho_L[j_e + 1 : N]
    q_R[j_e + 1 : N] = q[outer] / rho[outer] * rho_R[j_e + 1 : N]
    if j_e > 0:  # the excision face: both sides are the first retained cell, at its reconstructed face density
        q_L[j_e] = q_R[j_e] = q[j_e] / rho[j_e] * rho_R[j_e]
    return q_L, q_R


def hll_flux(
    rho_L: FloatArray,
    rho_R: FloatArray,
    delta_rho_L: FloatArray,
    delta_rho_R: FloatArray,
    q_L: FloatArray,
    q_R: FloatArray,
    deviation: State,
    geo: Geometry,
    Theta: FloatArray,
    a: FloatArray,
    eos: EquationOfState,
    w: StencilWeights,
) -> FloatArray:
    """The HLL energy flux through the retained faces `j < N` (eq:num:hll) as its deviation from the FRW flux.

    The one-sided flux

        F_j(rho, q) = [alpha ((1 + w) rho^(-w/(1+w)) U_j - X_j) - (d_xi X)_j] X_j^2 rho
                      + alpha (rho^(-w/(1+w)) U_j - X_j) X_j^2 q

    is evaluated on each side, at that side's reconstructed density with the lapse of that same density and at that
    side's viscous pressure `q_L` or `q_R` (`viscous_sides`), and combined with the signal-speed bounds
    `Lambda^+ = max(Theta + a, 0)`, `Lambda^- = min(Theta - a, 0)`.

    The HLL combination is affine in the one-sided fluxes, with weights `Lambda^+ / (Lambda^+ - Lambda^-)` and
    `-Lambda^- / (Lambda^+ - Lambda^-)` summing to one, so subtracting the FRW flux `F_FRW = (alpha w X - d_xi X) X^2`
    from both one-sided fluxes subtracts it from the result. Each one-sided flux is formed as that deviation directly,

        F_j(rho) - F_FRW = X^2 [(alpha w X - d_xi X) (rho - 1) + (1 + w) alpha (e^phi U - X) rho]
                           + alpha (e^phi U - X) X^2 q,          e^phi U - X = X (e^phi - 1) + e^phi delta U,

    in which every term is of the size of the deviation (`equations.py` says why the energy rows need it).

    Args:
        rho_L: The reconstructed density inside each face.
        rho_R: The reconstructed density outside each face.
        delta_rho_L: `rho_L - 1`, formed without cancellation.
        delta_rho_R: `rho_R - 1`, likewise.
        q_L: The viscous pressure the inside flux carries.
        q_R: The viscous pressure the outside flux carries.
        deviation: The state's deviation from FRW, for `delta U`.
        geo: The geometry.
        Theta: The grid velocity `Theta_j` at the faces.
        a: The sound speed `a_j` at the faces.
        eos: The equation of state.
        w: The stencil weights, for the layout.

    Returns:
        `F_j - F_FRW,j` at the retained faces `j < N`; `0` at the origin, where both vanish.
    """
    layout = w.layout
    N, j_e = layout.N, layout.j_e
    faces = slice(j_e, N)  # the interior faces and the innermost one; face N belongs to the outer closure
    X, X_xi, dU = geo.X[faces], geo.X_xi[faces], deviation.U[faces]
    alpha, w_eos = float(eos.alpha), float(eos.w)
    frw_speed = alpha * w_eos * X - X_xi  # the FRW flux is frw_speed X^2
    X2 = X * X

    def one_sided(rho: FloatArray, delta_rho: FloatArray, q: FloatArray) -> FloatArray:
        ephi, delta_ephi = eos.lapse_and_deviation(rho, delta_rho)
        drift = alpha * (X * delta_ephi + ephi * dU)  # alpha (e^phi U - X)
        return X2 * (frw_speed * delta_rho + (1.0 + w_eos) * drift * rho + drift * q)

    Lam_plus = np.maximum(Theta[faces] + a[faces], 0.0)
    Lam_minus = np.minimum(Theta[faces] - a[faces], 0.0)
    delta_F = np.full(N + 1, np.nan)
    delta_F[faces] = (
        Lam_plus * one_sided(rho_L[faces], delta_rho_L[faces], q_L[faces])
        - Lam_minus * one_sided(rho_R[faces], delta_rho_R[faces], q_R[faces])
        + Lam_plus * Lam_minus * X2 * (delta_rho_R[faces] - delta_rho_L[faces])
    ) / (Lam_plus - Lam_minus)
    if j_e == 0:
        delta_F[0] = 0.0
    return delta_F
