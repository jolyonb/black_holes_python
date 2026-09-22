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
  `Rtilde^2 mtilde = M / X`), and at the origin `Gammabar_0^2 = e^(2 (1 - alpha) xi)`, the limit of `M / X ~ X^2`.
  As written, the last two terms are each of order `X^2` and cancel on FRW, which costs a factor `(X / R_H)^2` in
  precision at large radius. They are formed instead from the deviation from FRW,

      U_j^2 - M_j / X_j = delta U_j (U_j + X_j) - delta M_j / X_j,    delta M_j = M_j - X_j^3 = 3 sum_{i<j} delta E_i,

  (plus `delta M_e` once excised) in which every term is small where the flow is near FRW, and nothing large is
  subtracted: `Gammabar^2` is FRW to the bit on the FRW state, at any radius. The deviation is the integrator's own
  (Section 7.6) when the caller passes it, and is otherwise recovered from the state, which keeps the FRW property
  but loses the digits the state lost when the deviation was added to `y_FRW`;
* the relative deviations of Section 7.4, `delta_rho,c = delta E_c / Delta V_c`, `delta_U,j = delta U_j / X_j` and
  `delta_m,j = delta M_j / X_j^3`, which are `rho_c - 1`, `U_j / X_j - 1` and `mt_j - 1` formed without subtracting
  one, for the outer closure and the monitors, which work in them;
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
from pbh.state import State, frw_state
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
        delta_rho: The relative density deviation `rho_c - 1 = delta E_c / Delta V_c` (cells).
        delta_U: The relative velocity deviation `U_j / X_j - 1 = delta U_j / X_j` (faces `1..N`; NaN at the origin).
        delta_m: The relative mass deviation `mt_j - 1 = delta M_j / X_j^3` (faces `1..N`; NaN at the origin).
        Gammabar2: `Gammabar_j^2` (faces), the first line of eq:num:facefields.
        rho_f: The face density `<rho>_j` (faces).
        ephi_f: The face lapse `<ephi>_j` (faces).
    """

    rho: FloatArray
    ephi: FloatArray
    M: FloatArray
    mt: FloatArray
    delta_rho: FloatArray
    delta_U: FloatArray
    delta_m: FloatArray
    Gammabar2: FloatArray
    rho_f: FloatArray
    ephi_f: FloatArray


def derive(
    state: State,
    geo: Geometry,
    bg: Background,
    eos: EquationOfState,
    w: StencilWeights,
    deviation: State | None = None,
) -> Derived:
    """Form the derived fields of one stage, asserting hyperbolicity on the retained entries.

    Args:
        state: The evolved unknowns at this stage.
        geo: The geometry at this stage's time.
        bg: The background at this stage's time (for the FRW `Gammabar^2`).
        eos: The equation of state (for the lapse exponent).
        w: The stencil weights, which carry the layout and average the cell fields to the faces.
        deviation: The state's deviation from FRW, `state - frw_state(geo, j_e)`, if the caller holds it; otherwise
            it is recovered from the state. `Gammabar^2` and the relative deviations read it.

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
    ephi[cells] = eos.lapse(rho[cells])

    # The mass inside each retained face: what is inside the innermost retained face (nothing, or the excised M_e)
    # plus three times the energy of every retained cell inside it (Section 7.2 and 8.3).
    M = np.full(N + 1, np.nan)
    M[layout.j_e] = state.M_e
    M[layout.j_e + 1 :] = state.M_e + 3.0 * np.cumsum(state.E[cells])

    mt = np.full(N + 1, np.nan)
    inner = max(layout.j_e, 1)  # never face 0, whose mt is 0 / 0
    X = geo.X[inner:]
    X3 = X * X * X  # a tenth of the cost of X ** 3, which numpy evaluates as a general power
    mt[inner:] = M[inner:] / X3

    if deviation is None:
        frw = frw_state(geo, layout.j_e)
        deviation = State(E=state.E - frw.E, U=state.U - frw.U, W=state.W, M_e=state.M_e - frw.M_e)
    dM = np.full(N + 1, np.nan)  # M - X^3, the mass against its FRW value, as a sum of small terms
    dM[layout.j_e] = deviation.M_e
    dM[layout.j_e + 1 :] = deviation.M_e + 3.0 * np.cumsum(deviation.E[cells])

    delta_rho = np.full(N, np.nan)
    delta_rho[cells] = deviation.E[cells] / geo.dV[cells]
    delta_U = np.full(N + 1, np.nan)
    delta_U[inner:] = deviation.U[inner:] / X
    delta_m = np.full(N + 1, np.nan)
    delta_m[inner:] = dM[inner:] / X3

    Gammabar2 = np.full(N + 1, np.nan)
    Gammabar2[inner:] = gammabar_squared(bg, X, state.U[inner:], deviation.U[inner:], dM[inner:])
    if layout.j_e == 0:
        Gammabar2[0] = bg.Gammabar2  # M / X ~ X^2 -> 0 at the origin, and U_0 = 0
    _assert_positive(Gammabar2, faces, "Gammabar2")

    return Derived(
        rho=rho,
        ephi=ephi,
        M=M,
        mt=mt,
        delta_rho=delta_rho,
        delta_U=delta_U,
        delta_m=delta_m,
        Gammabar2=Gammabar2,
        rho_f=w.face_average(rho),
        ephi_f=w.face_average(ephi),
    )


def gammabar_squared(bg: Background, X: FloatArray, U: FloatArray, dU: FloatArray, dM: FloatArray) -> FloatArray:
    """`Gammabar^2 = Gammabar_FRW^2 + delta U (U + X) - delta M / X` at faces `X > 0`, without the FRW cancellation.

    `U^2 - M / X` rewritten as `(U - X)(U + X) - (M - X^3) / X`, with `dU = U - X` and `dM = M - X^3` the deviations.
    """
    return bg.Gammabar2 + dU * (U + X) - dM / X


def _assert_positive(values: FloatArray, retained: slice, name: str) -> None:
    """Raise `NotHyperbolicError` at the first retained entry that is not positive (NaN counts as not positive)."""
    bad = np.flatnonzero(~(values[retained] > 0.0))
    if bad.size:
        index = int(bad[0]) + retained.start
        raise NotHyperbolicError(name, index, float(values[index]))
