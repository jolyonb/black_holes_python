"""The stencils that carry a field across a face (paper eq:num:stencils; at an excision face eq:numbh:rows1).

Three operators, and only three, look across a face:

* the face value `<f>_j` of a cell field `f` (the density or the lapse), the average of the two cells that share
  the face;
* the gradient `(D_s f)_j` of a cell field, a one-cell difference taken in `s = X^2`, which is exact on a field
  linear in `s` and second order down to the origin (a difference in `X` would be first order at the first faces,
  because an even field is not linear in `X` there: Section 7.3);
* the gradient `(D_U U)_j` of the face velocity, which reaches the neighbouring faces because it differentiates a
  face field at a face.

The picture, around face `j` (cells are numbered by their inner face, so cells `j - 1` and `j` share face `j`):

    face j           j-1                j                j+1
                      |-----------------|-----------------|
    cell c                    j-1                j
    cell field f            f_{j-1}             f_j
    face field U   U_{j-1}             U_j             U_{j+1}

        <f>_j     = (f_{j-1} + f_j) / 2
        (D_s f)_j = 2 X_j (f_j - f_{j-1}) / dS_j
        (D_U U)_j = (U_{j+1} - U_{j-1}) / (X_{j+1} - X_{j-1})

This is what the staggering buys: face `j` lies between the two cells it differences, so the average and the one-cell
difference are centred at the face and second order with the shortest possible stencils, and the one-cell
difference sees the grid-scale sawtooth that a collocated centred difference cannot (Section 7.2). The velocity
gradient is the exception: it compares faces two apart, which is harmless because it appears only in the
advection term (Section 7.3).

Three faces are special. At the origin `<f>_0 = f_0` and `(D_s f)_0 = 0` (nothing is imposed there; face `0` carries no
velocity gradient). No ghost values are used, at the origin or anywhere: every stencil at face `j >= 1` reaches only
cells `j - 1`, `j` and faces `j - 1`, `j + 1`, all of which exist, face `0` being a real face with `U_0 = 0`. Odd fields
could be mirrored freely, but at second order that adds nothing, since for an odd field the one-sided difference at the
origin equals the mirrored centred one; an even field must not be mirrored, because a density that a converging front
has cusped at the origin is not smooth there (Section 7.2).

At the outer face `N` no cell field is averaged: the face has one state, the density `rho_hat_N` reconstructed from the
last cell (its one-sided slope in `s`, theta-limited as in eq:num:theta) and the lapse of that same density
(`outer_face_density`, Section 7.5). An extrapolated face value `3/2 f_{N-1} - 1/2 f_{N-2}` would be second
order too, but for the density and the lapse it turns negative beside a nearly empty last cell. No gradient is formed at
face `N` (the outer row of Section 7.5 has no pressure difference), and the velocity gradient is the three-point
one-sided row of eq:num:stencils. At an excision face `j_e` (Section 8.3) the retained data lie on one side only, and
the rows are the first-order ones of eq:numbh:rows1: the face value is the cell behind it, `<f>_je = f_je`, no pressure
gradient is formed, and the velocity gradient is the one retained difference `(U_{j_e+1} - U_{j_e}) / dX_{j_e}`. Their
boundary form is the continuum characteristic flux, so the energy estimate certifies them. Section 8.3 says why the
second-order rows it also discusses are not implemented: no diagonal energy weight certifies them, and near vacuum their
extrapolated face density reaches zero.

Every coefficient of these stencils depends on the geometry alone, so on a static map it is the same at every stage
(Section 7.1). `StencilWeights.of(geo, layout)` computes them once; the caller caches it together with the
geometry, and its methods apply the stencils to a field and do nothing but multiply and subtract, bar the one
theta-limited reconstruction at face `N`.

Outputs are face arrays, `N + 1` long, NaN where the operator is not defined (below `j_e`; `(D_s f)_N`;
`(D_U U)_0`), following the convention of `layout.py`.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.types import FloatArray


@dataclass(frozen=True)
class StencilWeights:
    """The geometry-dependent coefficients of the three stencils, computed once per geometry.

    Attributes:
        layout: Which faces are retained and where the excision rows apply.
        grad_s: `2 X_j / dS_j`, the factor of the one-cell difference in the gradient (faces `1..N-1`; NaN elsewhere).
        centred_U: `1 / (X_{j+1} - X_{j-1})` for the centred velocity gradient (faces `1..N-1`; NaN elsewhere).
        outer_U: The three coefficients of `U_N`, `U_{N-1}`, `U_{N-2}` in the one-sided row at the outer face.
        excision_U: `1 / dX_{j_e}`, the factor of the one retained difference `U_{j_e+1} - U_{j_e}` in the velocity
            gradient at the excision face; never read without one.
        outer_rho: The last cell's `(sbar_{N-1} - sbar_{N-2}, X_{N-1}^2 - sbar_{N-1}, X_N^2 - sbar_{N-1})`: the
            divisor of its one-sided slope in `s`, and the offsets in `s` of its two faces from its mean.
    """

    layout: Layout
    grad_s: FloatArray
    centred_U: FloatArray
    outer_U: tuple[float, float, float]
    excision_U: float
    outer_rho: tuple[float, float, float]

    @classmethod
    def of(cls, geo: Geometry, layout: Layout) -> Self:
        """Compute the weights for this geometry and these retained faces."""
        N, j_e = layout.N, layout.j_e
        X = geo.X
        grad_s = np.full(N + 1, np.nan)
        grad_s[1:N] = 2.0 * X[1:N] / geo.dS[1:N]
        centred_U = np.full(N + 1, np.nan)
        centred_U[1:N] = 1.0 / (X[2 : N + 1] - X[0 : N - 1])
        outer_U = cls._one_sided_three_point(float(geo.dX[N - 1]), float(geo.dX[N - 2]))
        excision_U = 1.0 / float(geo.dX[j_e])  # the one retained difference at an excision face
        X2, sbar = X**2, geo.sbar
        outer_rho = (float(geo.dS[N - 1]), float(X2[N - 1] - sbar[N - 1]), float(X2[N] - sbar[N - 1]))
        return cls(
            layout=layout,
            grad_s=grad_s,
            centred_U=centred_U,
            outer_U=outer_U,
            excision_U=excision_U,
            outer_rho=outer_rho,
        )

    def face_average(self, f: FloatArray) -> FloatArray:
        """The face value `<f>_j` of a cell field (eq:num:stencils, first line; eq:numbh:rows1 at `j_e`).

        Args:
            f: A cell field (`N` entries), the density or the lapse.

        Returns:
            `<f>_j` at the retained faces `j < N`: the two-cell average inside, `f_0` at the origin, and at an
            excision face the cell behind it. NaN at face `N`, whose state is `outer_face_density`'s.
        """
        N, j_e = self.layout.N, self.layout.j_e
        avg = np.full(N + 1, np.nan)
        avg[j_e + 1 : N] = 0.5 * (f[j_e : N - 1] + f[j_e + 1 : N])
        avg[j_e] = f[j_e]  # f_0 at the origin; the cell behind an excision face
        return avg

    def outer_face_density(self, delta_rho: FloatArray, theta: float) -> tuple[float, float]:
        """The density `rho_hat_N` at the outer face and its deviation `rho_hat_N - 1` (Section 7.5, eq:num:theta).

        The last cell's profile in `s`, with its one-sided slope `(rho_{N-1} - rho_{N-2}) / (sbar_{N-1} -
        sbar_{N-2})` scaled by the theta-limiter so that both its face values are at least `theta rho_{N-1}`,
        evaluated at face `N`. It is the reconstruction's own `rho^L_N` (kernels.reconstruct_density), formed here too
        because the face needs it with the kernels off as well.

        Args:
            delta_rho: The cell densities' deviations `rho - 1`.
            theta: The theta-limiter's fraction, `0 < theta < 1`.
        """
        N = self.layout.N
        dS, s_in, s_out = self.outer_rho
        last = delta_rho[N - 1 : N]
        slope = (last - delta_rho[N - 2 : N - 1]) / dS  # formed as the reconstruction forms it, to the last bit
        _, _, delta_out = theta_limited_faces(last, slope * s_in, slope * s_out, theta)
        return 1.0 + float(delta_out[0]), float(delta_out[0])

    def gradient_s(self, f: FloatArray) -> FloatArray:
        """The gradient `(D_s f)_j = 2 X_j (f_j - f_{j-1}) / dS_j` of a cell field (eq:num:stencils, second line).

        A one-cell difference in `s = X^2` converted to a derivative in `X` by the chain rule factor `2 X_j`. Formed
        only for the density and for `sbar q`, never for the lapse (Section 8.3).

        Args:
            f: A cell field (`N` entries).

        Returns:
            `(D_s f)_j` at the retained faces `j < N`: zero at the origin and at an excision face, the one-cell
            difference inside. NaN at face `N`, where no gradient is ever formed.
        """
        N, j_e = self.layout.N, self.layout.j_e
        grad = np.full(N + 1, np.nan)
        grad[j_e + 1 : N] = self.grad_s[j_e + 1 : N] * (f[j_e + 1 : N] - f[j_e : N - 1])
        grad[j_e] = 0.0  # none at the origin, and none formed at an excision face
        return grad

    def velocity_gradient(self, U: FloatArray) -> FloatArray:
        """The velocity gradient `(D_U U)_j` (eq:num:stencils, third and fourth lines; eq:numbh:rows1 at `j_e`).

        Args:
            U: The face velocities (`N + 1` entries).

        Returns:
            `(D_U U)_j` at the retained faces `j >= 1`: the centred two-face difference inside, the three-point
            one-sided row at face `N`, and at an excision face the one retained difference. NaN at the origin, which
            carries no velocity gradient.
        """
        N, j_e = self.layout.N, self.layout.j_e
        grad = np.full(N + 1, np.nan)
        lo, hi = j_e + 1, N  # the centred rows: faces 1 .. N-1 unexcised, j_e+1 .. N-1 excised
        grad[lo:hi] = self.centred_U[lo:hi] * (U[lo + 1 : hi + 1] - U[lo - 1 : hi - 1])
        a_0, a_1, a_2 = self.outer_U
        grad[N] = a_0 * U[N] + a_1 * U[N - 1] + a_2 * U[N - 2]
        if j_e > 0:
            grad[j_e] = self.excision_U * (U[j_e + 1] - U[j_e])
        return grad

    @staticmethod
    def _one_sided_three_point(Delta_1: float, Delta_2: float) -> tuple[float, float, float]:
        """The coefficients of the derivative at the end of three points with spacings `Delta_1` (nearest), `Delta_2`.

        The fourth line of eq:num:stencils, written for the outer face: `(D_U U)_N = a_0 U_N + a_1 U_{N-1} + a_2
        U_{N-2}`, second order and exact on a quadratic.
        """
        return (
            (2.0 * Delta_1 + Delta_2) / (Delta_1 * (Delta_1 + Delta_2)),
            -(Delta_1 + Delta_2) / (Delta_1 * Delta_2),
            Delta_1 / (Delta_2 * (Delta_1 + Delta_2)),
        )


def theta_limited_faces(
    delta_rho: FloatArray, off_in: FloatArray, off_out: FloatArray, theta: float
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The two face values of cells whose linear profiles are scaled by the theta-limiter (eq:num:theta).

    A cell with density `rho_c` and face offsets `off_in`, `off_out` from its mean (the slope times `X^2 - sbar_c` at
    each face) keeps its mean and has its offsets scaled by the one factor `t = min(1, (1 - theta) rho_c / m)`, `m`
    the larger drop below the mean, so that both face values are at least `theta rho_c`. Where no face drops that far,
    `t = 1` and the profile is untouched. Formed on the deviations: a face value is `1 + (delta_rho_c + t off)`.

    Args:
        delta_rho: The cells' deviations `rho_c - 1`.
        off_in: The offset of each cell's inner face value from its mean.
        off_out: The offset of its outer face value.
        theta: The fraction, `0 < theta < 1`.

    Returns:
        `(t, delta_in, delta_out)`: the scale factor of each cell, and the deviations `rho - 1` of its inner and outer
        face values.
    """
    rho = 1.0 + delta_rho
    drop = -np.minimum(np.minimum(off_in, off_out), 0.0)  # the larger drop below the mean, >= 0
    allowed = (1.0 - theta) * rho
    # Where no face drops that far t = 1; the ratio is only ever taken of a drop above `allowed`, so it is never a
    # division by zero, nor by the subnormal drop of a cell barely off the reference, which would overflow.
    t = np.where(drop > allowed, allowed / np.maximum(drop, allowed), 1.0)
    return t, delta_rho + t * off_in, delta_rho + t * off_out
