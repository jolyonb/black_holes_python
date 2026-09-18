"""The exact geometry of the cells (paper Section 7.2, eq:num:geom, and the geometry rows of Table tab:num:layout).

Three radial coordinates appear in the code, and only the first two appear in this module:

* The label `x`: the coordinate the grid is uniform in. Faces sit at `x_j = j h`, `j = 0..N`, with `h = x_max / N`.
  It is pure bookkeeping; the scheme contains no `h` except in the Courant number.
* The scaled areal radius `X = Rtilde = R / (a R_H)`: the areal radius of a face in units of the comoving Hubble
  radius, the coordinate every field and stencil of Section 7 is written in. The map of Section 7.1, `X(xi, x)`,
  gives each face its `X_j` analytically, together with `(d_xi X)_j` (how fast the face moves) and `(d_x X)_j`
  (the Jacobian). Before BH formation the map is the identity, `X = x`; after it the map pins the faces near the hole.
* The physical areal radius `R = a R_H X`: what an observer measures. It appears only in the horizon mass and the
  read-out, never in the scheme.

The layout, with `N` cells (here `N = 4`):

    label x      0        h        2h       3h       4h = x_max   5h
                 |--------|--------|--------|--------|- - - - - - -|
    face j       0        1        2        3        4            (5: virtual face)
    cell c           0        1        2        3          (4: virtual cell)
    radius X    X_0 = 0   X_1      X_2      X_3      X_4          X_5

Face `0` is the origin, where `X_0 = 0` exactly; face `N` is the outer boundary. Cells are numbered by their inner
face: cell `c` is the shell between faces `c` and `c + 1`, so at face `j` the cell outside is `j` and the cell inside
is `j - 1`, and every stencil that crosses a face is written that way. A cell has no radius of its own: everything
about it follows from its two face radii. Where a radius for the cell is needed it is the mean-square radius
`sbar_c` (the paper's symbol: `s` for `X^2`, the bar for the shell mean, `c` for the cell), the volume-weighted mean
of `s = X^2` over the shell, `int X^4 dX / int X^2 dX`. An even field is a smooth function of `s`, so its shell
average is its point value at `sbar_c` to second order, and exactly when the field is linear in `s`; neither the
midpoint nor the volume centroid has that property (Section 7.2).

The virtual cell beyond the boundary, between the outer face and one more label step, holds no field; it exists only so
that the difference of mean-square radii across face `N`, `dS_N`, is defined like every other, and the map is evaluated
at its outer face too.

Array indexing follows the picture. Face arrays (`X`, `X_xi`, `X_x`, `dS`) have `N + 1` entries indexed by `j`; cell
arrays (`dV`, `dV_xi`, `dX`, `Xm`) have `N` entries indexed by `c`; `sbar` is a cell array with the virtual cell
appended as entry `N`. An entry that is not a thing (`dS_0`, since no gradient is formed at the origin) is NaN and is
never read. The same convention carries over to the fields in `layout.py`.

The map does not depend on `xi` before formation, so the geometry is computed once and cached by the caller; after
formation the map moves and the geometry is recomputed at every stage (Section 7.1).
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from pbh.types import FloatArray


@dataclass(frozen=True)
class Geometry:
    """The map at the faces and the exact cell geometry built from it, at one time.

    Build it with `Geometry.of(X, X_xi, X_x)` from the map evaluated at the `N + 2` labels `x_0 .. x_{N+1}`, the last
    being the virtual face beyond the outer boundary.

    Attributes:
        X: The scaled areal radius `X_j` of face `j` (faces `0..N`); `X_0 = 0` exactly.
        X_xi: The velocity of the face, `(d_xi X)_j` at fixed label (faces `0..N`); zero on a static map.
        X_x: The Jacobian of the map, `(d_x X)_j` (faces `0..N`); it enters the scheme only through the Courant
            number (eq:num:cfl) and the outer cell width.
        dV: The shell volume `Delta V_c = int_cell X^2 dX` (cells `0..N-1`), the FRW value of the cell energy. There is
            no `4 pi`: the tilde variables absorb it (`mtilde = m / ((4 pi / 3) rho_b R^3)`, eq:newvariablesm), which is
            why the cumulative mass is `M_j = 3 sum_{i<j} E_i` with FRW value `X_j^3` and no `4 pi` appears anywhere.
        dV_xi: Its rate on a moving map, `d_xi Delta V_c = X_{j+1}^2 (d_xi X)_{j+1} - X_j^2 (d_xi X)_j`
            (cells `0..N-1`), the FRW rate of the cell energy in the deviation form of Section 7.6.
        sbar: The mean-square radius `sbar_c = int_cell X^4 dX / Delta V_c`, the shell mean of `s = X^2`
            (cells `0..N`, entry `N` being the virtual outer cell). A shell average of an even field is its point value
            at `sbar_c` up to second order, and exactly for a field linear in `s`; this is what makes the gradient
            stencil and the reconstruction of Section 7 second order down to the origin.
        dS: The difference of mean-square radii across face `j`, `Delta S_j = sbar_c - sbar_{c-1}` for faces
            `1..N`; entry `0` is NaN because no gradient is ever formed at the origin (`(D_s f)_0 = 0` by definition).
        dX: The cell width `Delta X_c = X_{j+1} - X_j` (cells `0..N-1`).
        Xm: The cell midpoint `X_{m,c} = (X_j + X_{j+1}) / 2` (cells `0..N-1`).
    """

    X: FloatArray
    X_xi: FloatArray
    X_x: FloatArray
    dV: FloatArray
    dV_xi: FloatArray
    sbar: FloatArray
    dS: FloatArray
    dX: FloatArray
    Xm: FloatArray

    @property
    def N(self) -> int:
        """The number of cells."""
        return self.dV.shape[0]

    @classmethod
    def of(cls, X: FloatArray, X_xi: FloatArray, X_x: FloatArray) -> Self:
        """Build the geometry from the map evaluated at the faces and the virtual face beyond the boundary.

        Args:
            X: `X_j` at the `N + 2` labels `x_0 .. x_{N+1}`; `X_0` must be exactly zero and the radii strictly
                increasing (the admissibility conditions of the map, Section 7.1).
            X_xi: `(d_xi X)_j` at the same labels.
            X_x: `(d_x X)_j` at the same labels.

        Returns:
            The `Geometry` for the `N` cells, with `sbar` carrying the virtual cell as well.

        Raises:
            ValueError: If the radii do not start at zero or are not strictly increasing.
        """
        if X[0] != 0.0:
            raise ValueError(f"the origin face must have X_0 = 0 exactly, got {X[0]!r}")
        if not np.all(np.diff(X) > 0.0):
            raise ValueError("the face radii must increase strictly: the map has d_x X > 0")
        # Every cell, the virtual one included: the pair (X_-, X_+) of eq:num:geom is (X[:-1], X[1:]). The volume is
        # written as (X_+ - X_-) times a quadratic rather than as a difference of cubes, and the mean-square radius as
        # the ratio of a quartic to that same quadratic with the factor (X_+ - X_-) cancelled by hand, so that the only
        # cancellation is in the one small factor X_+ - X_- and the round-off stays at the N eps floor of Section 7.2.
        X_minus, X_plus = X[:-1], X[1:]
        X_minus2, X_plus2, product = X_minus**2, X_plus**2, X_minus * X_plus
        quadratic = X_minus2 + product + X_plus2  # X_-^2 + X_- X_+ + X_+^2
        quartic = X_minus2**2 + X_plus2**2 + product * (X_minus2 + X_plus2) + product**2  # X_-^4 + ... + X_+^4
        dV_all = (X_plus - X_minus) * quadratic / 3.0
        sbar_all = 0.6 * quartic / quadratic
        # The volume rate on a moving map, eq:num:geom third line, for the real cells only.
        dV_xi = X_plus2[:-1] * X_xi[1:-1] - X_minus2[:-1] * X_xi[:-2]
        # Delta S_j for faces 1..N is the difference of neighbouring cells' sbar, the virtual cell supplying face N.
        dS = np.full(X.shape[0] - 1, np.nan)
        dS[1:] = np.diff(sbar_all)
        return cls(
            X=X[:-1],
            X_xi=X_xi[:-1],
            X_x=X_x[:-1],
            dV=dV_all[:-1],
            dV_xi=dV_xi,
            sbar=sbar_all,
            dS=dS,
            dX=(X_plus - X_minus)[:-1],
            Xm=(0.5 * (X_minus + X_plus))[:-1],
        )
