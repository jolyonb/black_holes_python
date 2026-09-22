"""An independent reference for the stage: every row written as printed, in the whole-state variables.

`equations.py` forms every row as its deviation from the FRW rate, the FRW parts cancelled in the algebra. This is the
same stage written the plain way, as eq:num:facefields, eq:num:energy, eq:num:velocity, eq:num:hll and eq:num:sat
print it, with the FRW parts left in: it carries round-off of the FRW size and is otherwise the same function of the
state. The tests require the two to agree on strongly nonlinear states, where a slip in the rearranged algebra would
show at order one. It shares with the stage only what the rearrangement did not touch: the derived fields, the
reconstruction, the viscous pressure and the stencils.
"""

from dataclasses import dataclass

import numpy as np

from pbh.derived import derive
from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import Kernels, KernelSettings, reconstruct_density, viscous_pressure
from pbh.outer import PenaltyStrengths, boundary_ode_coefficients
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


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
    d = derive(state, geo, bg, eos, w)
    X, X_xi, U = geo.X, geo.X_xi, state.U

    # eq:num:facefields
    Theta = alpha * (U * d.ephi_f - X) - X_xi
    cE = alpha * ((1.0 + w_eos) * d.ephi_f * U - X)
    a = alpha * eos.sqrt_w * d.ephi_f * np.sqrt(d.Gammabar2)
    Lam = np.abs(Theta) + a
    D_s_rho = w.gradient_s(d.rho)
    D_U = w.velocity_gradient(U)

    # the flux: eq:num:hll with the kernels, the base flux of eq:num:energy without
    F = np.full(N + 1, np.nan)
    if settings.kernels is Kernels.PRODUCTION:
        rho_L, rho_R, _, _ = reconstruct_density(d.rho - 1.0, geo, w, settings.density_limiter, settings.rho_floor)
        _, _, q_f, Q = viscous_pressure(state, geo, d, Lam, eos, w, settings.c_v)
        f = slice(j_e, N)

        def one_sided(rho: FloatArray) -> FloatArray:
            ephi = rho**eos.lapse_exponent
            transport = (alpha * ((1.0 + w_eos) * ephi * U[f] - X[f]) - X_xi[f]) * X[f] ** 2 * rho
            return transport + alpha * (ephi * U[f] - X[f]) * X[f] ** 2 * q_f[f]

        Lp = np.maximum(Theta[f] + a[f], 0.0)
        Lm = np.minimum(Theta[f] - a[f], 0.0)
        F[f] = (Lp * one_sided(rho_L[f]) - Lm * one_sided(rho_R[f]) + Lp * Lm * X[f] ** 2 * (rho_R[f] - rho_L[f])) / (
            Lp - Lm
        )
    else:
        Q = np.zeros(N + 1)
        F[faces] = (cE[faces] - X_xi[faces]) * X[faces] ** 2 * d.rho_f[faces]
    if j_e == 0:
        F[0] = 0.0

    # the outer face
    if strengths is None:
        dU_N = float(X_xi[N])
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
        (1.0 - alpha) * U[j]
        - inertia * (w_eos * D_s_rho[j] + Q[j])
        - 0.5 * alpha * d.ephi_f[j] * X[j] * (d.mt[j] + 3.0 * w_eos * d.rho_f[j])
        - Theta[j] * D_U[j]
    )
    dU[N] = dU_N
    if j_e == 0:
        dU[0] = 0.0

    # eq:num:energy and eq:numbh:mass
    dE = np.full(N, np.nan)
    dE[cells] = -(F[j_e + 1 : N + 1] - F[j_e:N]) + eos.energy_source_rate * state.E[cells]
    dM_e = eos.energy_source_rate * state.M_e - 3.0 * float(F[j_e]) if j_e > 0 else 0.0
    return WholeRate(E=dE, U=dU, W=dW, M_e=dM_e, F=F)
