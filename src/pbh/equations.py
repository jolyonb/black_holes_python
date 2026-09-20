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

This module implements the centred base scheme: the flux is `F_j = (cE_j - (d_xi X)_j) X_j^2 <rho>_j` with the
face-averaged density transported (eq:num:energy) and the artificial pressure force `Q_j` is zero. The paper keeps
that scheme as a test switch and never runs it in production (Section 7.4: undissipated, the modes the energy norm
cannot see grow under refinement); the production flux and `Q_j` of Section 7.7 replace the two lines marked below
in a later bite, and nothing else changes.

What the scheme gets exactly (Section 7.3, all verified there and tested here): on FRW every average is `1`, every
gradient `0`, `M_j = X_j^3`, `F_j = alpha w X_j^3 - X_j^2 (d_xi X)_j`, and the rows return exactly `d_xi Delta V_c`
and `(d_xi X)_j`, so FRW is a fixed point on every static map and the exact solution on every moving one; the
constraint holds by construction, since the mass is the cumulative sum; the cumulative mass at every retained face
obeys `d_xi M_j = (2 - 3 alpha) M_j - 3 F_j` because the energy rows telescope into the face-mass row; and the
scheme knows nothing about how the faces were placed, only where they are, so two maps with the same face radii give
the same rates.

The velocity at face `0` is not an unknown; its rate is reported as zero. The face-mass row exists only once cells
are excised. Indexing and the NaN convention are those of `layout.py`.
"""

from dataclasses import dataclass

import numpy as np

from pbh.derived import Derived, derive
from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.outer import OuterClosure, OuterInputs
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


@dataclass(frozen=True)
class Speeds:
    """The four speeds of eq:num:facefields, at the retained faces (NaN elsewhere).

    Attributes:
        Theta: The grid velocity `Theta_j`, the fluid's velocity relative to the moving face.
        cE: The energy-flux velocity `cE_j`, at which energy crosses the face: `(1 + w)` times the transport
            because the energy flux carries the pressure work, minus the Hubble flow.
        a: The sound speed `a_j` in the scaled coordinate.
        Lam: The signal speed `Lambda_j = |Theta_j| + a_j`, the fastest characteristic, which sets the time step
            (Section 7.6) and bounds the HLL flux (Section 7.7).
    """

    Theta: FloatArray
    cE: FloatArray
    a: FloatArray
    Lam: FloatArray


@dataclass(frozen=True)
class DerivsResult:
    """What one evaluation computed: the rate the integrator wants, and the fields the monitors and the finder read.

    Attributes:
        rate: The time derivative of the state, in the shape of a `State`.
        derived: The derived fields of this stage.
        speeds: The speeds of this stage.
        F: The energy flux through every retained face; `F_0 = 0`, `F_N` from the outer closure, and `F_{j_e}` the
            flux that also feeds the face-mass row.
    """

    rate: State
    derived: Derived
    speeds: Speeds
    F: FloatArray


def speeds(state: State, geo: Geometry, eos: EquationOfState, d: Derived, faces: slice) -> Speeds:
    """The four speeds of eq:num:facefields from the state, the geometry and the derived fields."""
    alpha = float(eos.alpha)
    N = geo.N
    Theta = np.full(N + 1, np.nan)
    cE = np.full(N + 1, np.nan)
    a = np.full(N + 1, np.nan)
    U, X, X_xi, ephi_f = state.U[faces], geo.X[faces], geo.X_xi[faces], d.ephi_f[faces]
    Theta[faces] = alpha * (U * ephi_f - X) - X_xi
    cE[faces] = alpha * ((1.0 + float(eos.w)) * ephi_f * U - X)
    a[faces] = alpha * eos.sqrt_w * ephi_f * np.sqrt(d.Gammabar2[faces])
    return Speeds(Theta=Theta, cE=cE, a=a, Lam=np.abs(Theta) + a)


def calc_derivs(
    state: State, geo: Geometry, bg: Background, eos: EquationOfState, w: StencilWeights, outer: OuterClosure
) -> DerivsResult:
    """Evaluate the semi-discrete equations once: the rate of every unknown at this time and state.

    Args:
        state: The evolved unknowns.
        geo: The geometry at this stage's time.
        bg: The background at this stage's time.
        eos: The equation of state.
        w: The stencil weights for this geometry, which carry the layout and the excision closure.
        outer: The closure of the outer face.

    Returns:
        The rate and the fields it was computed from.

    Raises:
        NotHyperbolicError: From the derived fields, if the state has left the hyperbolic domain.
    """
    layout = w.layout
    N, j_e = layout.N, layout.j_e
    alpha, w_eos = float(eos.alpha), float(eos.w)
    cells, faces = layout.cells, layout.faces

    d = derive(state, geo, bg, eos, w)
    sp = speeds(state, geo, eos, d, faces)
    D_s_rho = w.gradient_s(d.rho)
    D_U = w.velocity_gradient(state.U)

    # The energy flux through every retained face, eq:num:energy: the physical energy flux relative to the moving
    # face, (cE_j - (d_xi X)_j) X_j^2 <rho>_j. This is the centred base flux; the production flux replaces it later.
    F = np.full(N + 1, np.nan)
    flux_velocity = sp.cE[faces] - geo.X_xi[faces]
    F[faces] = flux_velocity * geo.X[faces] ** 2 * d.rho_f[faces]
    if j_e == 0:
        F[0] = 0.0
    Q = np.zeros(N + 1)  # the artificial pressure force of Section 7.7, zero in the base scheme (replaced later)

    # The outer face: the closure supplies the rows the interior cannot.
    rows = outer.rows(
        OuterInputs(
            xi=bg.xi,
            X_N=float(geo.X[N]),
            X_xi_N=float(geo.X_xi[N]),
            U_N=float(state.U[N]),
            W=state.W,
            rho_N_1=float(d.rho[N - 1]),
            rho_f_N=float(d.rho_f[N]),
            ephi_f_N=float(d.ephi_f[N]),
            mt_N=float(d.mt[N]),
            Theta_N=float(sp.Theta[N]),
            cE_N=float(sp.cE[N]),
            DU_N=float(D_U[N]),
            dS_N=float(geo.dS[N]),
            c_s=bg.c_s,
        ),
        eos,
    )
    F[N] = rows.F_N

    # The velocity rows at the interior faces, eq:num:velocity, term by term as printed.
    dU = np.full(N + 1, np.nan)
    j = slice(max(j_e, 1), N)
    U, X, ephi_f, rho_f = state.U[j], geo.X[j], d.ephi_f[j], d.rho_f[j]
    expansion = (1.0 - alpha) * U  # (1 - alpha) U_j: the background stretching of the velocity variable
    inertia = alpha / (1.0 + w_eos) * ephi_f * d.Gammabar2[j] / rho_f  # alpha / (1 + w) <ephi> Gammabar^2 / <rho>
    pressure = -inertia * (w_eos * D_s_rho[j] + Q[j])  # ... times [w (D_s rho)_j + Q_j], the pressure force
    gravity = -0.5 * alpha * ephi_f * X * (d.mt[j] + 3.0 * w_eos * rho_f)  # alpha / 2 <ephi> X (mt + 3 w <rho>)
    advection = -sp.Theta[j] * D_U[j]  # Theta_j (D_U U)_j: the fluid moving through the grid
    dU[j] = expansion + pressure + gravity + advection
    dU[N] = rows.dU_N
    if j_e == 0:
        dU[0] = 0.0  # not an unknown: U_0 = 0 always

    # The energy rows, eq:num:energy: what flows out through the outer face minus what flows in through the inner
    # one, plus the source (2 - 3 alpha) E_c; and the face-mass row, eq:numbh:mass, with the same flux F_{j_e}.
    dE = np.full(N, np.nan)
    flux_out, flux_in = F[j_e + 1 : N + 1], F[j_e:N]
    dE[cells] = -(flux_out - flux_in) + eos.energy_source_rate * state.E[cells]
    dM_e = eos.energy_source_rate * state.M_e - 3.0 * float(F[j_e]) if j_e > 0 else 0.0

    return DerivsResult(rate=State(E=dE, U=dU, W=rows.dW, M_e=dM_e), derived=d, speeds=sp, F=F)
