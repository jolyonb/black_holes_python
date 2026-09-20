"""The stencils that carry a field across a face (paper eq:num:stencils; at an excision face eq:numbh:rows1, rows).

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
eq:num:stencils. At an excision face `j_e` (Section 8.3) the retained data lie on one side only and two closures are
specified: the first-order rows eq:numbh:rows1, the production choice, which take the face value from the cell behind
it, form no pressure gradient, and use the one retained velocity difference; and the second-order rows eq:numbh:rows,
kept as a switch, which extrapolate the face value in `s`, place the single retained slope at the face, and use the
mirror of the outer three-point row. Section 8.3 proves that no diagonal energy weight certifies the second-order rows,
while the first-order closure's boundary form is the continuum characteristic flux; hence the default.

Every coefficient of these stencils depends on the geometry alone, so on a static map it is the same at every stage
(Section 7.1). `StencilWeights.of(geo, layout, closure)` computes them once; the caller caches it together with the
geometry, and its three methods apply the stencils to a field and do nothing but multiply and subtract.

Outputs are face arrays, `N + 1` long, NaN where the operator is not defined (below `j_e`; `(D_s f)_N`;
`(D_U U)_0`), following the convention of `layout.py`.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Self

import numpy as np

from pbh.geometry import Geometry
from pbh.layout import Layout
from pbh.types import FloatArray


class FaceClosure(Enum):
    """The one-sided rows at an excision face (Section 8.3, Table tab:numbh:params)."""

    FIRST_ORDER = "o1"
    """eq:numbh:rows1: the upwind choice, certified by the energy estimate; the production closure."""

    SECOND_ORDER = "o2"
    """eq:numbh:rows: one order more accurate at the face, provably not certifiable; kept as a switch."""


@dataclass(frozen=True)
class StencilWeights:
    """The geometry-dependent coefficients of the three stencils, computed once per geometry.

    Attributes:
        layout: Which faces are retained and where the excision rows apply.
        closure: Which excision rows the weights were built for.
        grad_s: `2 X_j / dS_j`, the factor of the one-cell difference in the gradient (faces `1..N-1`; NaN elsewhere).
        centred_U: `1 / (X_{j+1} - X_{j-1})` for the centred velocity gradient (faces `1..N-1`; NaN elsewhere).
        outer_U: The three coefficients of `U_N`, `U_{N-1}`, `U_{N-2}` in the one-sided row at the outer face.
        face_value_e: For the second-order closure, the weight of `f_{j_e+1} - f_{j_e}` in the extrapolated face
            value at `j_e`; `0` for the first-order closure, whose face value is `f_{j_e}` itself.
        grad_s_e: The factor of `f_{j_e+1} - f_{j_e}` in the gradient at the excision face: `0` for the first-order
            closure (no pressure gradient is formed there), `2 X_{j_e} / dS_{j_e+1}` for the second-order one.
        excision_U: The coefficients of `U_{j_e}`, `U_{j_e+1}`, `U_{j_e+2}` in the velocity gradient at the excision
            face: `(-1, 1, 0) / dX_{j_e}` for the first-order closure, the mirrored three-point row for the second.
    """

    layout: Layout
    closure: FaceClosure
    grad_s: FloatArray
    centred_U: FloatArray
    outer_U: tuple[float, float, float]
    face_value_e: float
    grad_s_e: float
    excision_U: tuple[float, float, float]

    @classmethod
    def of(cls, geo: Geometry, layout: Layout, closure: FaceClosure) -> Self:
        """Compute the weights for this geometry, these retained faces and this excision closure."""
        N, j_e = layout.N, layout.j_e
        X = geo.X
        grad_s = np.full(N + 1, np.nan)
        grad_s[1:N] = 2.0 * X[1:N] / geo.dS[1:N]
        centred_U = np.full(N + 1, np.nan)
        centred_U[1:N] = 1.0 / (X[2 : N + 1] - X[0 : N - 1])
        outer_U = cls._one_sided_three_point(float(geo.dX[N - 1]), float(geo.dX[N - 2]))
        # The rows at the excision face, if there is one; otherwise these are never read.
        face_value_e = grad_s_e = 0.0
        excision_U = (0.0, 0.0, 0.0)
        if j_e > 0:
            if closure is FaceClosure.FIRST_ORDER:
                excision_U = (-1.0 / float(geo.dX[j_e]), 1.0 / float(geo.dX[j_e]), 0.0)
            else:
                face_value_e = float((X[j_e] ** 2 - geo.sbar[j_e]) / geo.dS[j_e + 1])
                grad_s_e = float(2.0 * X[j_e] / geo.dS[j_e + 1])
                a_0, a_1, a_2 = cls._one_sided_three_point(float(geo.dX[j_e]), float(geo.dX[j_e + 1]))
                excision_U = (-a_0, -a_1, -a_2)  # the mirror: the same row with the points counted outward
        return cls(
            layout=layout,
            closure=closure,
            grad_s=grad_s,
            centred_U=centred_U,
            outer_U=outer_U,
            face_value_e=face_value_e,
            grad_s_e=grad_s_e,
            excision_U=excision_U,
        )

    def face_average(self, f: FloatArray) -> FloatArray:
        """The face value `<f>_j` of a cell field (eq:num:stencils, first line; eq:numbh:rows1 or rows at `j_e`).

        Args:
            f: A cell field (`N` entries), the density or the lapse.

        Returns:
            `<f>_j` at the retained faces: the two-cell average inside, the extrapolation at face `N`, `f_0` at the
            origin, and at an excision face the value the closure prescribes.
        """
        N, j_e = self.layout.N, self.layout.j_e
        avg = np.full(N + 1, np.nan)
        avg[j_e + 1 : N] = 0.5 * (f[j_e : N - 1] + f[j_e + 1 : N])
        avg[N] = 1.5 * f[N - 1] - 0.5 * f[N - 2]
        avg[j_e] = f[j_e] + self.face_value_e * (f[j_e + 1] - f[j_e])  # f_0 at the origin; the closure's row at j_e
        return avg

    def gradient_s(self, f: FloatArray) -> FloatArray:
        """The gradient `(D_s f)_j = 2 X_j (f_j - f_{j-1}) / dS_j` of a cell field (eq:num:stencils, second line).

        A one-cell difference in `s = X^2` converted to a derivative in `X` by the chain rule factor `2 X_j`. Formed
        only for the density and for `sbar q`, never for the lapse (Section 8.3).

        Args:
            f: A cell field (`N` entries).

        Returns:
            `(D_s f)_j` at the retained faces `j < N`: zero at the origin, the one-cell difference inside, and at an
            excision face zero (first order) or the single retained slope placed at the face (second order). NaN at
            face `N`, where no gradient is ever formed.
        """
        N, j_e = self.layout.N, self.layout.j_e
        grad = np.full(N + 1, np.nan)
        grad[j_e + 1 : N] = self.grad_s[j_e + 1 : N] * (f[j_e + 1 : N] - f[j_e : N - 1])
        grad[j_e] = self.grad_s_e * (f[j_e + 1] - f[j_e])  # zero at the origin and for the first-order closure
        return grad

    def velocity_gradient(self, U: FloatArray) -> FloatArray:
        """The velocity gradient `(D_U U)_j` (eq:num:stencils, third and fourth lines; eq:numbh:rows1 or rows at `j_e`).

        Args:
            U: The face velocities (`N + 1` entries).

        Returns:
            `(D_U U)_j` at the retained faces `j >= 1`: the centred two-face difference inside, the three-point
            one-sided row at face `N`, and at an excision face the one retained difference (first order) or the
            mirrored three-point row (second order). NaN at the origin, which carries no velocity gradient.
        """
        N, j_e = self.layout.N, self.layout.j_e
        grad = np.full(N + 1, np.nan)
        lo, hi = j_e + 1, N  # the centred rows: faces 1 .. N-1 unexcised, j_e+1 .. N-1 excised
        grad[lo:hi] = self.centred_U[lo:hi] * (U[lo + 1 : hi + 1] - U[lo - 1 : hi - 1])
        a_0, a_1, a_2 = self.outer_U
        grad[N] = a_0 * U[N] + a_1 * U[N - 1] + a_2 * U[N - 2]
        if j_e > 0:
            b_0, b_1, b_2 = self.excision_U
            grad[j_e] = b_0 * U[j_e] + b_1 * U[j_e + 1] + b_2 * U[j_e + 2]
        return grad

    @staticmethod
    def _one_sided_three_point(Delta_1: float, Delta_2: float) -> tuple[float, float, float]:
        """The coefficients of the derivative at the end of three points with spacings `Delta_1` (nearest), `Delta_2`.

        The fourth line of eq:num:stencils, written for the outer face: `(D_U U)_N = a_0 U_N + a_1 U_{N-1} + a_2
        U_{N-2}`, second order and exact on a quadratic. The mirrored row at an excision face is its negative with the
        points counted outward (eq:numbh:rows).
        """
        return (
            (2.0 * Delta_1 + Delta_2) / (Delta_1 * (Delta_1 + Delta_2)),
            -(Delta_1 + Delta_2) / (Delta_1 * Delta_2),
            Delta_1 / (Delta_2 * (Delta_1 + Delta_2)),
        )
