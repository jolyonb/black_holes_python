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

At the outer face `N` the face value is extrapolated, `<f>_N = 3/2 f_{N-1} - 1/2 f_{N-2}`, no gradient is formed (the
outer row of Section 7.5 has no pressure difference), and the velocity gradient is the three-point one-sided row of
eq:num:stencils. At an excision face `j_e` (Section 8.3) the retained data lie on one side only, and the rows are the
first-order ones of eq:numbh:rows1: the face value is the cell behind it, `<f>_je = f_je`, no pressure gradient is
formed, and the velocity gradient is the one retained difference `(U_{j_e+1} - U_{j_e}) / dX_{j_e}`. Their boundary
form is the continuum characteristic flux, so the energy estimate certifies them. Section 8.3 says why the
second-order rows it also discusses are not implemented: no diagonal energy weight certifies them, and near vacuum
their extrapolated face density reaches zero.

Every coefficient of these stencils depends on the geometry alone, so on a static map it is the same at every stage
(Section 7.1). `StencilWeights.of(geo, layout)` computes them once; the caller caches it together with the
geometry, and its three methods apply the stencils to a field and do nothing but multiply and subtract.

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
    """

    layout: Layout
    grad_s: FloatArray
    centred_U: FloatArray
    outer_U: tuple[float, float, float]
    excision_U: float

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
        return cls(layout=layout, grad_s=grad_s, centred_U=centred_U, outer_U=outer_U, excision_U=excision_U)

    def face_average(self, f: FloatArray) -> FloatArray:
        """The face value `<f>_j` of a cell field (eq:num:stencils, first line; eq:numbh:rows1 at `j_e`).

        Args:
            f: A cell field (`N` entries), the density or the lapse.

        Returns:
            `<f>_j` at the retained faces: the two-cell average inside, the extrapolation at face `N`, `f_0` at the
            origin, and at an excision face the cell behind it.
        """
        N, j_e = self.layout.N, self.layout.j_e
        avg = np.full(N + 1, np.nan)
        avg[j_e + 1 : N] = 0.5 * (f[j_e : N - 1] + f[j_e + 1 : N])
        avg[N] = 1.5 * f[N - 1] - 0.5 * f[N - 2]
        avg[j_e] = f[j_e]  # f_0 at the origin; the cell behind an excision face
        return avg

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
