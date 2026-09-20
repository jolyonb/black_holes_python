"""The fields derived from the state at every stage (paper Table tab:num:layout, below the line; Section 7.3).

Nothing here is evolved. From the stored cell energies and face velocities a stage first forms, in this order:

* the cell density `rho_c = E_c / Delta V_c`, the shell average of `rhotilde` (Section 7.1: the stored energy is
  divided by its volume before anything differences it);
* the cell lapse `ephi_c = rho_c ^ (-w / (1 + w))`, the algebraic lapse of the averaged density (eq:MSphinov);
* the cumulative mass at the faces, `M_j = 3 sum_{i<j} E_i`, or `M_j = M_e + 3 sum_{j_e <= i < j} E_i` once cells
  are excised (Section 8.3), which is how the constraint eq:eul:constraint holds by construction: the mass is never
  evolved, it is the sum of what the cells hold;
* the tilde mass `mt_j = M_j / X_j^3`, unity on FRW;
* `Gammabar_j^2 = e^(2 (1 - alpha) xi) + U_j^2 - M_j / X_j` (eq:num:facefields, first line; eq:newgamma with
  `Rtilde^2 mtilde = M / X`), and at the origin `Gammabar_0^2 = e^(2 (1 - alpha) xi)`, the limit of `M / X ~ X^2`;
* the face values `<rho>_j` and `<ephi>_j` of the two even cell fields, by the one averaging stencil of
  eq:num:stencils, which the velocity equation and the fluxes need at the faces.

Two things are asserted here and nowhere else, because they are the hyperbolicity of the system (Section 7.3):
`rho_c > 0` in every retained cell and `Gammabar_j^2 > 0` at every retained face. Where either fails "the system is
no longer hyperbolic, whatever the cause, and there is nothing to continue": the stage raises `NotHyperbolicError`
naming the entry, and the driver records it as an abort. The lapse is only formed after the density has passed,
since a negative power of a negative density is not a number.

Indexing is that of `layout.py`: cell arrays `N` long, face arrays `N + 1` long, NaN below the excision face, and
`mt_0` NaN because `M_0 / X_0^3` is `0 / 0` (the paper never uses it; `mt_1 = rho_0` identically).
"""

from dataclasses import dataclass

import numpy as np

from pbh.eos import Background, EquationOfState
from pbh.geometry import Geometry
from pbh.state import State
from pbh.stencils import StencilWeights
from pbh.types import FloatArray


class NotHyperbolicError(Exception):
    """A stage found a non-positive density or `Gammabar^2`: the system has left its hyperbolic domain.

    Attributes:
        field: `"rho"` (a cell) or `"Gammabar2"` (a face).
        index: The cell or face index of the first offending entry.
        value: The offending value.
    """

    def __init__(self, field: str, index: int, value: float) -> None:
        super().__init__(f"{field}[{index}] = {value!r} is not positive: the system is no longer hyperbolic")
        self.field = field
        self.index = index
        self.value = value


@dataclass(frozen=True)
class Derived:
    """The fields of Table tab:num:layout derived from the state at one stage.

    Attributes:
        rho: The shell-averaged density `rho_c = E_c / Delta V_c` (cells).
        ephi: The algebraic lapse `ephi_c = rho_c ^ (-w / (1 + w))` (cells).
        M: The cumulative mass `M_j` inside face `j` (faces); `M_0 = 0`, or `M_{j_e} = M_e` once excised.
        mt: The tilde mass `mt_j = M_j / X_j^3` (faces `1..N`; NaN at the origin).
        Gammabar2: `Gammabar_j^2` (faces), the first line of eq:num:facefields.
        rho_f: The face density `<rho>_j` (faces).
        ephi_f: The face lapse `<ephi>_j` (faces).
    """

    rho: FloatArray
    ephi: FloatArray
    M: FloatArray
    mt: FloatArray
    Gammabar2: FloatArray
    rho_f: FloatArray
    ephi_f: FloatArray


def derive(state: State, geo: Geometry, bg: Background, eos: EquationOfState, w: StencilWeights) -> Derived:
    """Form the derived fields of one stage, asserting hyperbolicity on the retained entries.

    Args:
        state: The evolved unknowns at this stage.
        geo: The geometry at this stage's time.
        bg: The background at this stage's time (for the FRW `Gammabar^2`).
        eos: The equation of state (for the lapse exponent).
        w: The stencil weights, which carry the layout and average the cell fields to the faces.

    Returns:
        The `Derived` fields, full length, NaN below the excision face.

    Raises:
        NotHyperbolicError: If any retained `rho_c <= 0` or `Gammabar_j^2 <= 0`.
    """
    layout = w.layout
    cells, faces = layout.cells, layout.faces
    N = layout.N

    rho = np.full(N, np.nan)
    rho[cells] = state.E[cells] / geo.dV[cells]
    _assert_positive(rho, cells, "rho")

    ephi = np.full(N, np.nan)
    ephi[cells] = rho[cells] ** eos.lapse_exponent

    # The mass inside each retained face: what is inside the innermost retained face (nothing, or the excised M_e)
    # plus three times the energy of every retained cell inside it (Section 7.2 and 8.3).
    M = np.full(N + 1, np.nan)
    M[layout.j_e] = state.M_e
    M[layout.j_e + 1 :] = state.M_e + 3.0 * np.cumsum(state.E[cells])

    mt = np.full(N + 1, np.nan)
    inner = max(layout.j_e, 1)  # never face 0, whose mt is 0 / 0
    mt[inner:] = M[inner:] / geo.X[inner:] ** 3

    Gammabar2 = np.full(N + 1, np.nan)
    Gammabar2[inner:] = bg.Gammabar2 + state.U[inner:] ** 2 - M[inner:] / geo.X[inner:]
    if layout.j_e == 0:
        Gammabar2[0] = bg.Gammabar2  # M / X ~ X^2 -> 0 at the origin, and U_0 = 0
    _assert_positive(Gammabar2, faces, "Gammabar2")

    return Derived(
        rho=rho, ephi=ephi, M=M, mt=mt, Gammabar2=Gammabar2, rho_f=w.face_average(rho), ephi_f=w.face_average(ephi)
    )


def _assert_positive(values: FloatArray, retained: slice, name: str) -> None:
    """Raise `NotHyperbolicError` at the first retained entry that is not positive (NaN counts as not positive)."""
    bad = np.flatnonzero(~(values[retained] > 0.0))
    if bad.size:
        index = int(bad[0]) + retained.start
        raise NotHyperbolicError(name, index, float(values[index]))
