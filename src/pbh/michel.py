"""The steady Michel accretion flow of radiation onto a black hole, in the code's variables (paper Section 6.2).

The late-time state near a hole is steady transonic inflow: gas at rest far away falls in, passes the sonic point
at `r_s = 3 M` for radiation, and crosses the horizon at `2 M`. In the fluid-orthogonal slicing the lapse equals
`Gamma`, and for radiation the two first integrals of the flow combine into one algebraic relation at each radius
(eq:exc:cubic),

    N^6 = r^4 U^2 / (108 M^4),   N^2 = 1 + U^2 - 2 M / r,

a cubic in `U^2` whose transonic branch is the single positive root inside `2 M`, the larger of two between `2 M`
and `3 M`, the double root `U^2 = 1/6` at `3 M`, and the smaller beyond. With `N` the lapse and the compression
`rho / rho_inf = N^-4`, this is the flow of Table tab:exc:michel. The accretion rate in units of `n_inf M^2` is
`lambda_c = 6 sqrt 3`, and the accretion law reads `d ln M / dxi = lambda_c epsilon` with `epsilon = H M`.

In the scaled variables, at a time `xi`, a hole of mass `M / R_H` has the areal radius `r = e^(alpha xi) X R_H`,
so `r / M = e^(alpha xi) X / (M / R_H)`; the fields are `rho_tilde = rho / rho_inf`, `ephi = N`,
`U_tilde = e^((1 - alpha) xi) U` and `Gammabar = e^((1 - alpha) xi) Gamma`, with the current background as
`rho_inf`; and the hole's own mass in the cumulative-sum bookkeeping is `M_hole = 2 (M / R_H) e^((2 - 3 alpha) xi)`,
constant in `X`, whose rate `(2 - 3 alpha) M_hole` is eq:numbh:mass with no flux.

This is a test fluid on a fixed geometry: the gas's self-gravity, of relative size `10^2 epsilon^2`, and the
Hubble flow, of relative size `epsilon`, are neglected, so at `epsilon = 10^-8` the flow is an exact solution of
the scheme's equations far below any truncation error. It is the one exact solution with a horizon, and the
excision closure, the mass law, the finder and the near-zone monitors are all held against it.
"""

import math
from dataclasses import dataclass

import numpy as np

from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.initial import cell_contents
from pbh.layout import Layout
from pbh.state import State
from pbh.types import FloatArray

LAMBDA_C = 6.0 * math.sqrt(3.0)
"""The accretion eigenvalue of radiation, eq:exc:lambdac: `(1 + 3w)^((1 + 3w) / 2w) / (4 w^(3/2))` at `w = 1/3`."""

SONIC_RADIUS = 3.0
"""The sonic radius in units of the mass, `(1 + 3w) / (2w)` at `w = 1/3`."""


@dataclass(frozen=True)
class MichelFlow:
    """The flow at radii `r / M`: the lapse `N`, the compression `rho / rho_inf` and the inflow `U < 0`; `Gamma = N`."""

    r: FloatArray
    N: FloatArray
    compression: FloatArray
    U: FloatArray

    @property
    def Gamma(self) -> FloatArray:
        """`Gamma = N` on this flow."""
        return self.N

    @property
    def v(self) -> FloatArray:
        """The velocity ratio `U / Gamma`, `-1` at the horizon and `-1 / sqrt 3` at the sonic point."""
        return self.U / self.N


def inflow_speed_squared(r: float) -> float:
    """`U^2` on the transonic branch at `r / M = r`, from the cubic `(1 + U^2 - 2/r)^3 = r^4 U^2 / 108`."""
    if abs(r - SONIC_RADIUS) < 1e-6:
        return 1.0 / 6.0  # the double root at the sonic point, where the two branches meet (continuously)
    c = 1.0 - 2.0 / r
    k = r**4 / 108.0
    # (u + c)^3 - k u = u^3 + 3c u^2 + (3c^2 - k) u + c^3
    roots = np.roots([1.0, 3.0 * c, 3.0 * c**2 - k, c**3])
    real = np.sort(np.array([float(z.real) for z in roots if abs(z.imag) < 1e-6 * max(1.0, abs(z.real))]))
    positive = real[real > 0.0]
    if positive.size == 0:
        raise ValueError(f"no positive root of the Michel cubic at r / M = {r}")
    if r < SONIC_RADIUS:
        return float(positive[-1])  # inside the horizon the single root; between it and the sonic point the larger
    return float(positive[0])  # beyond the sonic point the smaller


def michel_flow(r_over_M: FloatArray) -> MichelFlow:
    """The flow at the radii `r / M`, on the transonic branch."""
    r = np.asarray(r_over_M, dtype=np.float64)
    U2 = np.array([inflow_speed_squared(float(x)) for x in r.ravel()]).reshape(r.shape)
    N = np.sqrt(1.0 + U2 - 2.0 / r)
    return MichelFlow(r=r, N=N, compression=N**-4, U=-np.sqrt(U2))


def hole_mass_tilde(M_over_RH: float, xi: float, eos: EquationOfState) -> float:
    """The hole's mass in the cumulative-sum bookkeeping, `2 (M / R_H) e^((2 - 3 alpha) xi)`."""
    return 2.0 * M_over_RH * math.exp((2.0 - 3.0 * float(eos.alpha)) * xi)


def michel_state(geo: Geometry, bg: Background, eos: EquationOfState, M_over_RH: float, layout: Layout) -> State:
    """The Michel flow of a hole of mass `M / R_H` sampled on the excised grid at the background's time, radiation.

    The cell contents are the compression integrated over the cells; the face velocities are `e^((1 - alpha) xi)
    U`; the mass inside the excision face is the hole's alone (the gas inside it is not part of the test fluid);
    `W = 0`, the outer face being held.
    """
    if not eos.is_radiation:
        raise ValueError("the Michel flow is written for radiation only")
    N, j_e = layout.N, layout.j_e
    X = geo.X[: N + 1]
    scale = math.exp(float(eos.alpha) * bg.xi) / M_over_RH  # r / M = scale X
    r_inner = scale * float(X[j_e])  # the flow is only evaluated on the retained cells

    def excess(Xq: FloatArray) -> FloatArray:
        """`rho / rho_inf - 1` at the quadrature nodes of the retained cells, zero inside the face."""
        out = np.zeros_like(Xq)
        retained = scale * Xq >= r_inner
        out[retained] = michel_flow(scale * Xq[retained]).compression - 1.0
        return out

    E = np.full(N, np.nan)
    U = np.full(N + 1, np.nan)
    E[j_e:] = cell_contents(excess, geo)[j_e:]
    U[j_e:] = math.exp((1.0 - float(eos.alpha)) * bg.xi) * michel_flow(scale * X[j_e:]).U
    return State(E=E, U=U, W=0.0, M_e=hole_mass_tilde(M_over_RH, bg.xi, eos))


def michel_grid(
    M_over_RH: float, xi: float, eos: EquationOfState, outer_r_over_M: float, dX_over_M: float
) -> tuple[int, float]:
    """The number of cells and the outer radius of a uniform grid in `X` with cells of `dX / M` out to `r / M`."""
    N = round(outer_r_over_M / dX_over_M)
    X_max = outer_r_over_M * M_over_RH * math.exp(-float(eos.alpha) * xi)
    return N, X_max
