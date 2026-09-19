"""The evolved unknowns at one time, and their FRW values and rates (paper Table tab:num:layout; Section 7.6).

A `State` holds what Section 7 evolves: the cell energies `E_c` (the energy content `int_cell X^2 rhotilde dX`, whose
FRW value is the shell volume `Delta V_c`), the face velocities `U_j` (`Utilde`, odd, with FRW value `X_j`), the
incoming amplitude `W` at the outer face (Section 7.5, FRW value `0`), and after excision the mass `M_e` inside the
excision face (FRW value `X_{j_e}^3`). The indexing and the NaN convention below the excision face are those of
`layout.py`.

The integrator does not advance the state itself but its deviation from FRW, `delta y = y - y_FRW(xi)` (Section 7.6):
on a static map the two coincide, and on a moving one the deviation form keeps the far zone FRW to round-off where
the direct form keeps it only to the truncation error of the map's motion. `frw_state` gives `y_FRW` at a time and
`frw_rate` its time derivative, which the stage subtracts from its own; both follow from the geometry alone.

A `State` is also the shape of a rate of change: the stage returns `d_xi E_c`, `d_xi U_j`, `d_xi W`, `d_xi M_e` in
the same container, packed by the same rule.
"""

from dataclasses import dataclass

import numpy as np

from pbh.geometry import Geometry
from pbh.types import FloatArray


@dataclass(frozen=True)
class State:
    """The evolved unknowns (or their rates) at one time.

    Attributes:
        E: The cell energies `E_c` (cells `0..N-1`), NaN below the excision face.
        U: The face velocities `U_j` (faces `0..N`); `U_0 = 0` while the origin is in the domain, NaN below the
            excision face otherwise.
        W: The incoming amplitude at the outer face, the auxiliary scalar of the outer condition (Section 7.5).
        M_e: The mass inside the excision face, `M_{j_e}`; `0` while there is no excision.
    """

    E: FloatArray
    U: FloatArray
    W: float
    M_e: float = 0.0

    def __post_init__(self) -> None:
        """Cells and faces must belong to the same grid: `N` cells and `N + 1` faces."""
        if self.U.shape != (self.E.shape[0] + 1,):
            raise ValueError(f"need N + 1 face velocities for N cells, got E {self.E.shape} and U {self.U.shape}")


def frw_state(geo: Geometry, j_e: int = 0) -> State:
    """The FRW state on this geometry: `E_c = Delta V_c`, `U_j = X_j`, `W = 0`, `M_e = X_{j_e}^3` (Section 7.3).

    Args:
        geo: The geometry at the time in question.
        j_e: The index of the excision face, `0` before excision (as in `Layout`); it fixes which face's `X^3` is the
            FRW value of `M_e`.

    Returns:
        The whole FRW state, excised entries included: what is retained is the layout's business.
    """
    return State(E=geo.dV.copy(), U=geo.X.copy(), W=0.0, M_e=float(geo.X[j_e]) ** 3)


def frw_rate(geo: Geometry, j_e: int = 0) -> State:
    """The time derivative of the FRW state on this geometry, `d_xi y_FRW` (Section 7.6).

    `d_xi Delta V_c` is the third line of eq:num:geom, `d_xi X_j` is the map's own velocity, `W` stays zero, and
    `d_xi X_{j_e}^3 = 3 X_{j_e}^2 (d_xi X)_{j_e}` (Section 8.1). All vanish on a static map, where the deviation form
    and the direct form coincide.

    Args:
        geo: The geometry at the time in question.
        j_e: The index of the excision face, `0` before excision (as in `Layout`).

    Returns:
        `d_xi y_FRW` in the shape of a `State`.
    """
    X_e, X_xi_e = float(geo.X[j_e]), float(geo.X_xi[j_e])
    return State(E=geo.dV_xi.copy(), U=geo.X_xi.copy(), W=0.0, M_e=3.0 * X_e**2 * X_xi_e)


def is_finite(state: State, j_e: int = 0) -> bool:
    """Whether every retained entry is finite: the check the driver makes after each step (Section 7.3).

    Args:
        state: The state to check.
        j_e: The index of the excision face, `0` before excision (as in `Layout`); entries below it are NaN by
            convention and are not looked at.
    """
    return bool(
        np.all(np.isfinite(state.E[j_e:]))
        and np.all(np.isfinite(state.U[j_e:]))
        and np.isfinite(state.W)
        and np.isfinite(state.M_e)
    )
