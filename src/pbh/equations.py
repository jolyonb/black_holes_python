"""One stage of the integrator: the semi-discrete equations (paper Section 7.3; the face-mass law eq:numbh:mass).

A stage receives the time and the state and returns the time derivative of every unknown. In the order the paper
gives (Section 7.3): the map and the geometry are already in hand (cached on a static map, recomputed per stage on a
moving one); the derived fields of `derived.py` come first, with their hyperbolicity check and the face values
`<rho>_j`, `<ephi>_j` of the two even cell fields; then at every face the four speeds of eq:num:facefields,

    Theta_j = alpha (U_j <ephi>_j - X_j) - (d_xi X)_j    the fluid's velocity relative to the grid
    cE_j    = alpha ((1 + w) <ephi>_j U_j - X_j)          the velocity of the energy flux
    a_j     = alpha sqrt(w) <ephi>_j Gammabar_j            the sound speed in the scaled coordinate
    Lambda_j = |Theta_j| + a_j                              the signal speed, which sets the time step;

then the flux through every face and the four rows:

    energy    d_xi E_c = -(F_{c+1} - F_c) + (2 - 3 alpha) E_c                                     eq:num:energy
    velocity  d_xi U_j = (1 - alpha) U_j
                         - alpha / (1 + w) <ephi>_j Gammabar_j^2 / <rho>_j [ w (D_s rho)_j + Q_j ]
                         - alpha / 2 <ephi>_j X_j (mt_j + 3 w <rho>_j)
                         - Theta_j (D_U U)_j                                                       eq:num:velocity
    face mass d_xi M_e = (2 - 3 alpha) M_e - 3 F_{j_e}                                            eq:numbh:mass
    outer     d_xi U_N, F_N, d_xi W                                                                 from the closure

The flux and the artificial pressure force are the two things the shock-capturing kernels of `kernels.py` supply
(Section 7.7). With the kernels off, the centred base scheme runs: the flux is `F_j = (cE_j - (d_xi X)_j) X_j^2
<rho>_j` with the face-averaged density transported (eq:num:energy) and `Q_j` is zero. The paper keeps that scheme
as a test switch and never runs it in production (Section 7.4: undissipated, the modes the energy norm cannot see grow
under refinement).

What the scheme gets exactly (Section 7.3, all verified there and tested here): on FRW every average is `1`, every
gradient `0`, `M_j = X_j^3`, `F_j = alpha w X_j^3 - X_j^2 (d_xi X)_j`, and the rows return exactly `d_xi Delta V_c`
and `(d_xi X)_j`, so FRW is a fixed point on every static map and the exact solution on every moving one; the
constraint holds by construction, since the mass is the cumulative sum; the cumulative mass at every retained face
obeys `d_xi M_j = (2 - 3 alpha) M_j - 3 F_j` because the energy rows telescope into the face-mass row; and the
scheme knows nothing about how the faces were placed, only where they are, so two maps with the same face radii give
the same rates.

The velocity at face `0` is not an unknown; its rate is reported as zero. The face-mass row exists only once cells
are excised. Indexing and the NaN convention are those of `layout.py`.

Round-off. On FRW each row is a sum of terms of the FRW size that cancel: the energy row differences fluxes of size
`alpha w X^3` down to the cell content `X^2 Delta X` and loses a factor of the cell index, the velocity row cancels the
expansion against gravity at the size of `X`. So every row is formed as its deviation from the FRW rate, from the
integrator's deviation (Section 7.6), with the FRW parts cancelled in the algebra rather than in floating point:

    energy    d_xi E_c   = d_xi Delta V_c - (delta F_{c+1} - delta F_c) + (2 - 3 alpha) delta E_c,
    face mass d_xi M_e   = d_xi X_je^3 - 3 delta F_je + (2 - 3 alpha) delta M_e,
    velocity  d_xi U_j   = (d_xi X)_j + (1 - alpha) delta U_j + pressure
                         - alpha / 2 X_j [<ephi - 1>_j (mt_j + 3 w <rho>_j) + delta_m,j + 3 w <rho - 1>_j]
                         - drift_j (1 + delta D_j) + (d_xi X)_j delta D_j,

with `delta F = F - F_FRW`, `F_FRW = (alpha w X - d_xi X) X^2` (so that `-(F_FRW,c+1 - F_FRW,c) + (2 - 3 alpha)
Delta V_c = d_xi Delta V_c` exactly, eq:num:geom), the drift `alpha (<ephi> U - X)` of `Speeds`, which is `Theta` on a
static map, and `delta D = D_U delta U`, the velocity gradient less its FRW value one, which every row of `D_U` gives
exactly on `U = X`. The fluxes are formed as deviations in `kernels.hll_flux` and `outer.base_flux_deviation`.
Every term is then of the size of the deviation; the stage returns this deviation rate, which the integrator
advances, and the whole rate, the FRW rate plus it, which the monitors read.
"""

from dataclasses import dataclass

import numpy as np

from pbh.derived import Derived, derive
from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.kernels import (
    KernelResult,
    Kernels,
    KernelSettings,
    hll_flux,
    reconstruct_density,
    viscous_pressure,
    viscous_sides,
)
from pbh.outer import OuterClosure, OuterInputs
from pbh.state import State, deviation_from_frw, frw_rate
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


@dataclass(frozen=True)
class Speeds:
    """The four speeds of eq:num:facefields, at the retained faces (NaN elsewhere).

    Attributes:
        drift: `alpha (<ephi>_j U_j - X_j)`, the fluid's velocity relative to the Hubble flow in the scaled coordinate:
            zero on FRW, and `Theta_j` on a static map. Formed from the deviation, it gives the other two without
            cancellation, `Theta = drift - d_xi X` and `cE = alpha w X + (1 + w) drift`.
        Theta: The grid velocity `Theta_j`, the fluid's velocity relative to the moving face.
        cE: The energy-flux velocity `cE_j`, at which energy crosses the face: `(1 + w)` times the transport
            because the energy flux carries the pressure work, minus the Hubble flow.
        a: The sound speed `a_j` in the scaled coordinate.
        Lam: The signal speed `Lambda_j = |Theta_j| + a_j`, the fastest characteristic, which sets the time step
            (Section 7.6) and bounds the HLL flux (Section 7.7).
    """

    drift: FloatArray
    Theta: FloatArray
    cE: FloatArray
    a: FloatArray
    Lam: FloatArray


@dataclass(frozen=True)
class DerivsResult:
    """What one evaluation computed: the rate the integrator wants, and the fields the monitors and the finder read.

    Attributes:
        rate: The time derivative of the state, in the shape of a `State`: the FRW rate plus `deviation_rate`.
        deviation_rate: The time derivative of the deviation from FRW, which the integrator advances (Section 7.6).
        derived: The derived fields of this stage.
        speeds: The speeds of this stage.
        F: The energy flux through every retained face; `F_0 = 0`, `F_N` from the outer closure, and `F_{j_e}` the
            flux that also feeds the face-mass row.
        delta_F: The same flux less its FRW value, `F - (alpha w X - d_xi X) X^2`, as the rows use it.
        kernels: What the shock-capturing kernels produced, or `None` when the centred base scheme ran.
    """

    rate: State
    deviation_rate: State
    derived: Derived
    speeds: Speeds
    F: FloatArray
    delta_F: FloatArray
    kernels: KernelResult | None


def speeds(d: Derived, deviation: State, geo: Geometry, eos: EquationOfState, faces: slice) -> Speeds:
    """The four speeds of eq:num:facefields from the derived fields and the deviation, by way of the drift."""
    alpha, w = float(eos.alpha), float(eos.w)
    N = geo.N
    X, ephi_f = geo.X[faces], d.ephi_f[faces]
    drift = np.full(N + 1, np.nan)  # NaN below the retained faces, and so are Theta and cE
    drift[faces] = alpha * (X * d.delta_ephi_f[faces] + ephi_f * deviation.U[faces])  # <ephi> U - X
    a = np.full(N + 1, np.nan)
    a[faces] = alpha * eos.sqrt_w * ephi_f * np.sqrt(d.Gammabar2[faces])
    Theta = drift - geo.X_xi
    return Speeds(drift=drift, Theta=Theta, cE=alpha * w * geo.X + (1.0 + w) * drift, a=a, Lam=np.abs(Theta) + a)


def calc_derivs(
    state: State,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    w: StencilWeights,
    outer: OuterClosure,
    settings: KernelSettings,
    deviation: State | None = None,
) -> DerivsResult:
    """Evaluate the semi-discrete equations once: the rate of every unknown at this time and state.

    Args:
        state: The evolved unknowns.
        geo: The geometry at this stage's time.
        bg: The background at this stage's time.
        eos: The equation of state.
        w: The stencil weights for this geometry, which carry the layout and the excision rows.
        outer: The closure of the outer face.
        settings: The kernel switches: production kernels or the centred base scheme, and their constants.
        deviation: The state's deviation from FRW, if the caller holds it (see `derive`).

    Returns:
        The rate and the fields it was computed from.

    Raises:
        NotHyperbolicError: From the derived fields, if the state has left the hyperbolic domain.
    """
    layout = w.layout
    N, j_e = layout.N, layout.j_e
    alpha, w_eos = float(eos.alpha), float(eos.w)
    cells, faces = layout.cells, layout.faces

    if deviation is None:
        deviation = deviation_from_frw(state, geo, j_e)
    d = derive(state, geo, bg, eos, w, settings.theta, deviation)
    sp = speeds(d, deviation, geo, eos, faces)
    D_s_rho = w.gradient_s(d.delta_rho)  # the same difference as of rho, without its rounding to the FRW size
    delta_D = w.velocity_gradient(deviation.U)  # (D_U U)_j - 1: every row of D_U gives exactly 1 on U = X

    # The energy flux through the retained faces and the artificial pressure force, as the flux's deviation from the
    # FRW flux: from the kernels of Section 7.7, or, with the kernels off, the centred base flux of eq:num:energy, the
    # physical energy flux relative to the moving face, (cE_j - (d_xi X)_j) X_j^2 <rho>_j, and no force.
    X, X_xi = geo.X, geo.X_xi
    F_frw = (alpha * w_eos * X - X_xi) * X**2  # the FRW flux, to which the deviation is added for the whole flux
    kernels = None
    if settings.kernels is Kernels.PRODUCTION:
        rho_L, rho_R, delta_rho_L, delta_rho_R, theta_scale = reconstruct_density(
            d.delta_rho, geo, w, settings.density_limiter, settings.theta
        )
        J, q, q_f, Q = viscous_pressure(state, geo, d, sp.Lam, eos, w, settings.c_v, settings.cap_tension)
        q_L, q_R = viscous_sides(q, q_f, d.rho, rho_L, rho_R, w.layout, settings.viscous_flux)
        delta_F, Lam_plus, Lam_minus, v_L, v_R = hll_flux(
            rho_L, rho_R, delta_rho_L, delta_rho_R, q_L, q_R, deviation, geo, sp.Theta, sp.a, eos, w
        )
        kernels = KernelResult(
            rho_L=rho_L,
            rho_R=rho_R,
            delta_rho_L=delta_rho_L,
            delta_rho_R=delta_rho_R,
            J=J,
            q=q,
            q_f=q_f,
            Q=Q,
            F=F_frw + delta_F,
            theta_scale=theta_scale,
            Lam_plus=Lam_plus,
            Lam_minus=Lam_minus,
            v_L=v_L,
            v_R=v_R,
        )
    else:
        delta_F = np.full(N + 1, np.nan)
        f = faces
        delta_F[f] = X[f] ** 2 * (
            (alpha * w_eos * X[f] - X_xi[f]) * d.delta_rho_f[f] + (1.0 + w_eos) * sp.drift[f] * d.rho_f[f]
        )  # outer.base_flux_deviation, on the arrays
        if j_e == 0:
            delta_F[0] = 0.0
        Q = np.zeros(N + 1)

    # The outer face: the closure supplies the rows the interior cannot.
    rows = outer.rows(
        OuterInputs(
            xi=bg.xi,
            X_N=float(X[N]),
            X_xi_N=float(X_xi[N]),
            U_N=float(state.U[N]),
            W=state.W,
            delta_U_N=float(d.delta_U[N]),
            delta_rho_N_1=float(d.delta_rho[N - 1]),
            rho_f_N=float(d.rho_f[N]),
            delta_rho_f_N=float(d.delta_rho_f[N]),
            ephi_f_N=float(d.ephi_f[N]),
            delta_ephi_f_N=float(d.delta_ephi_f[N]),
            mt_N=float(d.mt[N]),
            delta_m_N=float(d.delta_m[N]),
            drift_N=float(sp.drift[N]),
            delta_DU_N=float(delta_D[N]),
            dS_N=float(geo.dS[N]),
            c_s=bg.c_s,
        ),
        eos,
    )
    delta_F[N] = rows.delta_F_N
    F = F_frw + delta_F  # the whole flux, for the monitors

    # The velocity rows at the interior faces, eq:num:velocity, as their deviation from the FRW rate (d_xi X)_j.
    dU = np.full(N + 1, np.nan)
    j = slice(max(j_e, 1), N)
    Xj, ephi_f, rho_f = X[j], d.ephi_f[j], d.rho_f[j]
    expansion = (1.0 - alpha) * deviation.U[j]  # (1 - alpha) U_j, less the FRW part
    inertia = alpha / (1.0 + w_eos) * ephi_f * d.Gammabar2[j] / rho_f  # alpha / (1 + w) <ephi> Gammabar^2 / <rho>
    pressure = -inertia * (w_eos * D_s_rho[j] + Q[j])  # ... times [w (D_s rho)_j + Q_j], the pressure force
    lapse_mass = d.delta_ephi_f[j] * (d.mt[j] + 3.0 * w_eos * rho_f) + d.delta_m[j] + 3.0 * w_eos * d.delta_rho_f[j]
    gravity = -0.5 * alpha * Xj * lapse_mass  # alpha / 2 <ephi> X (mt + 3 w <rho>), less its FRW value
    advection = -sp.drift[j] * (1.0 + delta_D[j]) + X_xi[j] * delta_D[j]  # Theta (D_U U), less its FRW value
    dU[j] = expansion + pressure + gravity + advection
    dU[N] = rows.delta_dU_N
    if j_e == 0:
        dU[0] = 0.0  # not an unknown: U_0 = 0 always

    # The energy rows, eq:num:energy: what flows out through the outer face minus what flows in through the inner
    # one, plus the source (2 - 3 alpha) E_c; and the face-mass row, eq:numbh:mass, with the same flux F_{j_e}. The
    # FRW parts cancel exactly in the algebra, -(F_FRW,c+1 - F_FRW,c) + (2 - 3 alpha) Delta V_c = d_xi Delta V_c.
    dE = np.full(N, np.nan)
    flux_out, flux_in = delta_F[j_e + 1 : N + 1], delta_F[j_e:N]
    dE[cells] = -(flux_out - flux_in) + eos.energy_source_rate * deviation.E[cells]
    dM_e = eos.energy_source_rate * deviation.M_e - 3.0 * float(delta_F[j_e]) if j_e > 0 else 0.0

    deviation_rate = State(E=dE, U=dU, W=rows.dW, M_e=dM_e)
    frw = frw_rate(geo, j_e)
    rate = State(E=frw.E + dE, U=frw.U + dU, W=rows.dW, M_e=frw.M_e + dM_e)
    return DerivsResult(
        rate=rate,
        deviation_rate=deviation_rate,
        derived=d,
        speeds=sp,
        F=F,
        delta_F=delta_F,
        kernels=kernels,
    )
