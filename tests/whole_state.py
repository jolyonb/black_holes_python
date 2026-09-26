"""An independent reference for the stage: every row written as printed, in the whole-state variables.

`equations.py` forms every row as its deviation from the FRW rate, the FRW parts cancelled in the algebra. This is the
same stage written the plain way, as eq:num:facefields, eq:num:energy, eq:num:velocity, eq:num:hll and eq:num:sat
print it, with the FRW parts left in: it carries round-off of the FRW size and is otherwise the same function of the
state. The tests require the two to agree on strongly nonlinear states, where a slip in the rearranged algebra would
show at order one. It shares with the stage only the derived fields and the stencils. The shock-capturing kernels are
written again here, cell by cell in the whole state: the density reconstruction with its theta-limiter, the
limited jump and the viscous pressure with its tension cap, its face value and its force, and the viscous pressure each
side of a face carries, so that a slip in any of them shows as a disagreement.

Every Hubble, gravity and source term carries the background coefficient `h = bg.hubble` as the flat limit of Section
7.7 strikes it: 1 on FRW, where the rows are as printed, and 0 in flat spacetime.
"""

from dataclasses import dataclass

import numpy as np

from pbh.derived import derive
from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import DensityLimiter, Kernels, KernelSettings, ViscousFlux
from pbh.outer import PenaltyStrengths, boundary_ode_coefficients
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


def minmod3(*values: float) -> float:
    """The argument of least magnitude if all share a sign, else zero."""
    if all(v > 0.0 for v in values):
        return min(values)
    if all(v < 0.0 for v in values):
        return max(values)
    return 0.0


def reconstruct(
    rho: FloatArray, geo: Geometry, N: int, j_e: int, limiter: DensityLimiter, theta: float
) -> tuple[FloatArray, FloatArray]:
    """eq:num:recon and eq:num:theta in the whole density, cell by cell: `(rho_L, rho_R)` on the faces.

    Each retained cell's profile is `rho_c + t_c slope_c (s - sbar_c)`. An interior cell's slope is the MC (or
    minmod) limit of its two one-sided differences in `sbar`; an end cell's is its single one-sided difference. The
    factor `t_c <= 1` is the largest that keeps both face values at least `theta rho_c`.
    """
    X2, sbar = geo.X**2, geo.sbar
    slope = np.full(N, np.nan)
    for c in range(j_e, N):
        d_in = (rho[c] - rho[c - 1]) / (sbar[c] - sbar[c - 1]) if c > j_e else None
        d_out = (rho[c + 1] - rho[c]) / (sbar[c + 1] - sbar[c]) if c < N - 1 else None
        if d_in is not None and d_out is not None:
            if limiter is DensityLimiter.MC:
                r_in = (sbar[c] - sbar[c - 1]) / (sbar[c] - X2[c])
                r_out = (sbar[c + 1] - sbar[c]) / (X2[c + 1] - sbar[c])
                slope[c] = minmod3(0.5 * (d_in + d_out), r_in * d_in, r_out * d_out)
            else:
                slope[c] = minmod3(d_in, d_out)
        else:
            one = d_out if d_in is None else d_in
            assert one is not None
            slope[c] = one
    rho_L, rho_R = np.full(N + 1, np.nan), np.full(N + 1, np.nan)
    for c in range(j_e, N):
        inner, outer = slope[c] * (X2[c] - sbar[c]), slope[c] * (X2[c + 1] - sbar[c])  # the offsets from the mean
        drop = max(-min(inner, outer), 0.0)
        t = min(1.0, (1.0 - theta) * rho[c] / drop) if drop > 0.0 else 1.0
        rho_L[c + 1] = rho[c] + t * outer
        rho_R[c] = rho[c] + t * inner
    rho_L[j_e], rho_R[N] = rho_R[j_e], rho_L[N]
    return rho_L, rho_R


def face_pressures(
    q: FloatArray,
    q_f: FloatArray,
    rho: FloatArray,
    rho_L: FloatArray,
    rho_R: FloatArray,
    N: int,
    j_e: int,
    mode: ViscousFlux,
) -> tuple[FloatArray, FloatArray]:
    """The viscous pressure each one-sided flux carries: the face average, or each side's own `q / rho` at its face
    density, the excision face taking the first retained cell's on both sides."""
    q_L, q_R = q_f.copy(), q_f.copy()
    if mode is ViscousFlux.DENSITY_WEIGHTED:
        for j in range(j_e + 1, N):
            q_L[j] = q[j - 1] / rho[j - 1] * rho_L[j]
            q_R[j] = q[j] / rho[j] * rho_R[j]
        if j_e > 0:
            q_L[j_e] = q_R[j_e] = q[j_e] / rho[j_e] * rho_R[j_e]
    return q_L, q_R


def viscous(
    U: FloatArray,
    geo: Geometry,
    rho: FloatArray,
    ephi: FloatArray,
    Gammabar2: FloatArray,
    Lam: FloatArray,
    alpha: float,
    w_eos: float,
    N: int,
    j_e: int,
    c_v: float,
    cap: bool,
    h: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """eq:num:jump and eq:num:qvisc cell by cell, with the tension cap: `(q, q_f, Q)`.

    The peculiar velocity `U - h X` has a slope in each cell; each face takes the minmod of its two neighbours' (the
    single adjacent one at the end faces), and the jump across a cell is the difference of its two faces' profiles at
    the midpoint. `q = -c_v/2 max(Lam) (1 + w) rho / (alpha ephi <Gammabar^2>) J`, then `q >= -w rho` if capped, and
    zero in the last cell. The force is `X^-2 D_s(sbar q)`, and at an excision face the end row over the half cell.
    """
    X, Xm, dX, sbar, dS = geo.X, geo.Xm, geo.dX, geo.sbar, geo.dS
    ups = U - h * X
    if j_e == 0:
        ups[0] = 0.0
    g = {c: (ups[c + 1] - ups[c]) / dX[c] for c in range(j_e, N)}
    gf = {j_e: g[j_e], N: g[N - 1]}
    for j in range(j_e + 1, N):
        gf[j] = minmod3(g[j - 1], g[j])
    q = np.full(N, np.nan)
    for c in range(j_e, N):
        J = (ups[c + 1] + gf[c + 1] * (Xm[c] - X[c + 1])) - (ups[c] + gf[c] * (Xm[c] - X[c]))
        lam, gb2 = max(Lam[c], Lam[c + 1]), 0.5 * (Gammabar2[c] + Gammabar2[c + 1])
        q[c] = -0.5 * c_v * lam * (1.0 + w_eos) * rho[c] / (alpha * ephi[c] * gb2) * J
        if cap:
            q[c] = max(q[c], -w_eos * rho[c])
    q[N - 1] = 0.0
    q_f, Q = np.full(N + 1, np.nan), np.full(N + 1, np.nan)
    for j in range(j_e + 1, N):
        q_f[j] = 0.5 * (q[j - 1] + q[j])
        Q[j] = 2.0 * X[j] * (sbar[j] * q[j] - sbar[j - 1] * q[j - 1]) / dS[j] / X[j] ** 2
    q_f[j_e] = q[j_e]
    Q[j_e] = 2.0 * sbar[j_e] * q[j_e] / (X[j_e] ** 2 * dX[j_e]) if j_e > 0 else 0.0
    return q, q_f, Q


@dataclass(frozen=True)
class WholeRate:
    """The whole rate of every unknown and the flux, as the reference computes them."""

    E: FloatArray
    U: FloatArray
    W: float
    M_e: float
    F: FloatArray


def whole_state_rate(
    state: State,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    w: StencilWeights,
    strengths: PenaltyStrengths | None,
    settings: KernelSettings,
) -> WholeRate:
    """The stage as printed; the outer face held at FRW when `strengths` is None, the SAT closure otherwise."""
    layout = w.layout
    N, j_e = layout.N, layout.j_e
    cells, faces = layout.cells, layout.faces
    alpha, w_eos = float(eos.alpha), float(eos.w)
    h = bg.hubble
    d = derive(state, geo, bg, eos, w, settings.theta)
    X, X_xi, U = geo.X, geo.X_xi, state.U

    # eq:num:facefields
    Theta = alpha * (U * d.ephi_f - h * X) - X_xi
    cE = alpha * ((1.0 + w_eos) * d.ephi_f * U - h * X)
    a = alpha * eos.sqrt_w * d.ephi_f * np.sqrt(d.Gammabar2)
    Lam = np.abs(Theta) + a
    D_s_rho = w.gradient_s(d.rho)
    D_U = w.velocity_gradient(U)

    # the flux: eq:num:hll with the kernels, the base flux of eq:num:energy without
    F = np.full(N + 1, np.nan)
    if settings.kernels is Kernels.PRODUCTION:
        rho_L, rho_R = reconstruct(d.rho, geo, N, j_e, settings.density_limiter, settings.theta)
        args = (geo, d.rho, d.ephi, d.Gammabar2, Lam, alpha, w_eos, N, j_e, settings.c_v, settings.cap_tension, h)
        q, q_f, Q = viscous(U.copy(), *args)
        q_L, q_R = face_pressures(q, q_f, d.rho, rho_L, rho_R, N, j_e, settings.viscous_flux)
        f = slice(j_e, N)

        def one_sided(rho: FloatArray, q_side: FloatArray) -> FloatArray:
            ephi = rho**eos.lapse_exponent
            transport = (alpha * ((1.0 + w_eos) * ephi * U[f] - h * X[f]) - X_xi[f]) * X[f] ** 2 * rho
            return transport + alpha * (ephi * U[f] - h * X[f]) * X[f] ** 2 * q_side[f]

        F_L, F_R = one_sided(rho_L[f], q_L), one_sided(rho_R[f], q_R)
        with np.errstate(invalid="ignore"):  # 0 / 0 at the origin, whose flux is zero whatever the bounds
            v_L, v_R = F_L / (X[f] ** 2 * rho_L[f]), F_R / (X[f] ** 2 * rho_R[f])  # the chord speeds
        v_L, v_R = np.nan_to_num(v_L), np.nan_to_num(v_R)
        Lp = np.maximum(np.maximum(Theta[f] + a[f], 0.0), np.maximum(v_L, v_R))
        Lm = np.minimum(np.minimum(Theta[f] - a[f], 0.0), np.minimum(v_L, v_R))
        F[f] = (Lp * F_L - Lm * F_R + Lp * Lm * X[f] ** 2 * (rho_R[f] - rho_L[f])) / (Lp - Lm)
    else:
        Q = np.zeros(N + 1)
        F[faces] = (cE[faces] - X_xi[faces]) * X[faces] ** 2 * d.rho_f[faces]
    if j_e == 0:
        F[0] = 0.0

    # the outer face
    if strengths is None:
        dU_N = h * float(X_xi[N])
        F[N] = (cE[N] - X_xi[N]) * X[N] ** 2 * d.rho_f[N]
        dW = 0.0
    else:
        kappa = 1.5 * bg.c_s / X[N]
        delta_U, delta_rho = U[N] / X[N] - 1.0, d.rho[N - 1] - 1.0
        u_plus, u_minus = delta_U + kappa * delta_rho, delta_U - kappa * delta_rho
        pen = u_minus - state.W
        dU_N = (
            (1.0 - alpha) * U[N]
            - 0.5 * alpha * d.ephi_f[N] * X[N] * (d.mt[N] + 3.0 * w_eos * d.rho_f[N])
            - Theta[N] * D_U[N]
            - strengths.tau_u * bg.c_s * X[N] ** 2 / geo.dS[N] * pen
        )
        U_star = U[N] - 0.5 * strengths.tau_rho * X[N] * pen
        F[N] = alpha * ((1.0 + w_eos) * d.ephi_f[N] * U_star - X[N]) * X[N] ** 2 * d.rho_f[N]
        gm, gp, g0 = boundary_ode_coefficients(bg.c_s, float(X[N]))
        dW = gm * state.W + gp * (u_plus - strengths.tau_W * pen) + g0 * (d.mt[N] - 1.0)

    # eq:num:velocity
    dU = np.full(N + 1, np.nan)
    j = slice(max(j_e, 1), N)
    inertia = alpha / (1.0 + w_eos) * d.ephi_f[j] * d.Gammabar2[j] / d.rho_f[j]
    dU[j] = (
        (1.0 - alpha) * h * U[j]
        - inertia * (w_eos * D_s_rho[j] + Q[j])
        - h * 0.5 * alpha * d.ephi_f[j] * X[j] * (d.mt[j] + 3.0 * w_eos * d.rho_f[j])
        - Theta[j] * D_U[j]
    )
    dU[N] = dU_N
    if j_e == 0:
        dU[0] = 0.0

    # eq:num:energy and eq:numbh:mass
    dE = np.full(N, np.nan)
    dE[cells] = -(F[j_e + 1 : N + 1] - F[j_e:N]) + h * eos.energy_source_rate * state.E[cells]
    dM_e = h * eos.energy_source_rate * state.M_e - 3.0 * float(F[j_e]) if j_e > 0 else 0.0
    return WholeRate(E=dE, U=dU, W=dW, M_e=dM_e, F=F)
