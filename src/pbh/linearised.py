"""The scheme linearised about a state: a test instrument and a diagnostic, never part of a step.

This module exists for two things:

* Unit tests of the claims of Section 7.4. The scheme linearised about FRW, `d(delta y)/d xi = L delta y`, must
  satisfy the exact energy identity eq:num:energyid in the norm eq:num:norm, have the closed-form spectrum
  eq:num:spectrum, and have the grid sawtooth as its stiffest direction. `jacobian` builds `L` from the stage of
  `equations.py` itself, so those tests check the equations as written, coefficient by coefficient, including the
  terms that vanish on FRW and that no FRW test can see.
* The Jacobian export of the output specification (Section 11): on request the driver linearises the stage about a
  saved state (FRW, a collapse snapshot, the Michel state) and writes the operator for inspection.

The linearisation is numerical: the Jacobian of the packed rate with respect to the packed state by fourth-order
central differences, each component stepped by a fraction of its own scale. The base scheme is a smooth rational
function of the state, so this is accurate to about `1e-9` relative in the operator and, since the identity
involves only its symmetric part in the weighted norm, to `1e-12` in the identity; the workspace notes record that a
fixed-step forward difference is not, and contaminated an earlier measurement.

The identity is written in the relative variables of Section 7.4,

    delta_rho,c = delta E_c / Delta V_c,   delta_U,j = delta U_j / X_j,   delta_m,j = 3 sum_{i<j} delta E_i / X_j^3,

of which the third is the cumulative sum of the first and not an extra unknown. For radiation the norm is

    ||delta y||^2 = sum_c (9 c_s^2 / 4) Delta V_c delta_rho,c^2
                  + sum_{j=1}^{N-1} (X_j^3 dS_j / 2) (delta_U,j^2 + delta_m,j^2 / 8),                eq:num:norm

so that `d/dxi ||delta y||^2 = 2 ||delta y||^2 - 2 sum_j (X_j^3 dS_j / 2) (delta_U,j + delta_m,j / 4)^2` plus a
boundary term in which the outer velocity enters only as a datum (eq:num:energyid); the weight itself moves in time
through `c_s`, and the time derivative on the left includes that. The functions below give the change of variables
and the norm as matrices, so that the identity can be checked as a matrix equation. All of it is for the unexcised
grid and the outer face held, which is where the paper proves it.
"""

import numpy as np

from pbh.eos import Background, EquationOfState
from pbh.equations import calc_derivs
from pbh.geometry import Geometry
from pbh.kernels import KernelSettings
from pbh.layout import Layout
from pbh.outer import OuterClosure
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


def jacobian(
    state: State,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    w: StencilWeights,
    outer: OuterClosure,
    settings: KernelSettings,
    relative_step: float = 1e-3,
) -> FloatArray:
    """The Jacobian `d(rate) / d(state)` of the stage in the packed variables, by fourth-order central differences.

    Args:
        state: The state to linearise about.
        geo: The geometry at that time.
        bg: The background at that time.
        eos: The equation of state.
        w: The stencil weights.
        outer: The outer closure.
        settings: The kernel switches; the base scheme is smooth, the production kernels are not.
        relative_step: The finite-difference step as a fraction of each component's scale.

    Returns:
        The square matrix `J[i, k] = d rate_i / d y_k` over the packed vector of `w.layout`.
    """
    layout = w.layout
    y0 = layout.pack(state)
    scale = np.maximum(np.abs(y0), np.max(np.abs(y0)) * 1e-3)  # a floor for entries that happen to be near zero

    def rate(y: FloatArray) -> FloatArray:
        return layout.pack(calc_derivs(layout.unpack(y), geo, bg, eos, w, outer, settings).rate)

    J = np.empty((y0.size, y0.size))
    for k in range(y0.size):
        step = relative_step * scale[k]
        e = np.zeros_like(y0)
        e[k] = 1.0
        J[:, k] = (
            -rate(y0 + 2 * step * e) + 8 * rate(y0 + step * e) - 8 * rate(y0 - step * e) + rate(y0 - 2 * step * e)
        ) / (12 * step)
    return J


def relative_scaling(geo: Geometry, layout: Layout) -> FloatArray:
    """The diagonal `T` that takes the packed deviation to the relative variables, `(delta_rho, delta_U, delta W)`.

    `delta_rho,c = delta E_c / Delta V_c` and `delta_U,j = delta U_j / X_j`; `W` is left as it is. An operator `L` in
    the packed variables becomes `T L T^{-1}` in the relative ones.
    """
    if layout.excised:
        raise ValueError("the relative variables of Section 7.4 are defined on the unexcised grid")
    return np.concatenate((1.0 / geo.dV, 1.0 / geo.X[1:], [1.0]))


def mass_perturbation(geo: Geometry, layout: Layout) -> FloatArray:
    """The matrix `K` with `delta_m,j = (K delta_rho)_j` at the faces `1..N-1`: the cumulative sum in relative form.

    `delta_m,j = 3 sum_{i<j} Delta V_i delta_rho,i / X_j^3`, so `K[j-1, i] = 3 Delta V_i / X_j^3` for `i < j`.
    """
    N = layout.N
    K = np.zeros((N - 1, N))
    for j in range(1, N):
        K[j - 1, :j] = 3.0 * geo.dV[:j] / geo.X[j] ** 3
    return K


def energy_norm(geo: Geometry, bg: Background, eos: EquationOfState, layout: Layout) -> FloatArray:
    """The matrix `H` of the norm eq:num:norm in the relative variables `(delta_rho, delta_U, delta W)`, radiation.

    The diagonal weights are `(9/4) c_s^2 Delta V_c` on the cells and `X_j^3 dS_j / 2` on the faces `1..N-1`; the
    outer velocity `delta_U,N` and `W` carry no weight, since the identity treats them as data; and the `delta_m`
    term, the same face weight times `delta_m^2 / 8`, is a quadratic form in `delta_rho` through `K`.
    """
    if not eos.is_radiation:
        raise ValueError("the energy norm of Section 7.4 is printed for radiation only")
    N = layout.N
    face_weight = 0.5 * geo.X[1:N] ** 3 * geo.dS[1:N]
    H = np.diag(np.concatenate((2.25 * bg.c_s**2 * geo.dV, face_weight, [0.0, 0.0])))
    K = mass_perturbation(geo, layout)
    H[:N, :N] += K.T @ np.diag(face_weight / 8.0) @ K
    return H
