"""The evolved unknowns at one time, and their FRW values and rates (paper Table tab:num:layout; Section 7.6).

A `State` holds what Section 7 evolves: the cell energies `E_c` (the energy content `int_cell X^2 rhotilde dX`, whose
FRW value is the shell volume `Delta V_c`), the face velocities `U_j` (`Utilde`, odd, with FRW value `X_j`), the
incoming amplitude `W` at the outer face (Section 7.5, FRW value `0`), and after excision the mass `M_e` inside the
excision face (FRW value `X_{j_e}^3`). The indexing and the NaN convention below the excision face are those of
`layout.py`.

The integrator does not advance the state itself but its deviation from FRW, `delta y = y - y_FRW(xi)` (Section 7.6):
on a static map the two coincide, and on a moving one the deviation form keeps the far zone FRW to round-off where
the direct form keeps it only to the truncation error of the map's motion. `frw_state` gives `y_FRW` at a time and
`frw_rate` its time derivative, which the stage adds to its deviation rate for the whole rate; both follow from the
geometry alone.

A `State` is also the shape of a rate of change: the stage returns `d_xi E_c`, `d_xi U_j`, `d_xi W`, `d_xi M_e` in
the same container, packed by the same rule.
"""

from dataclasses import dataclass
from typing import Self

import numpy as np

from pbh.eos import EquationOfState
from pbh.geometry import Geometry
from pbh.types import FloatArray, read_only


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

    def plus(self, other: State) -> State:
        """The entrywise sum `self + other`, every field (NaN wherever either has NaN)."""
        return State(E=self.E + other.E, U=self.U + other.U, W=self.W + other.W, M_e=self.M_e + other.M_e)


def frw_state(geo: Geometry, j_e: int = 0, hubble: float = 1.0) -> State:
    """The background state here: `E_c = Delta V_c`, `U_j = h X_j`, `W = 0`, `M_e = X_{j_e}^3` (Section 7.3).

    With `h = 1` it is FRW; with `h = 0` it is the uniform fluid at rest in flat spacetime (Section 7.7).

    Args:
        geo: The geometry at the time in question.
        j_e: The index of the excision face, `0` before excision (as in `Layout`); it fixes which face's `X^3` is the
            FRW value of `M_e`.
        hubble: The background coefficient `h` of `Background`.

    Returns:
        The whole FRW state, excised entries included: what is retained is the layout's business.
    """
    # The scalar power, not `geo.X3[j_e]` (the product `X X X`), which differs in the last bit (see `Geometry`).
    return State(E=geo.dV.copy(), U=hubble * geo.X, W=0.0, M_e=float(geo.X[j_e]) ** 3)


def frw_rate(geo: Geometry, j_e: int = 0, hubble: float = 1.0) -> State:
    """The time derivative of the background state on this geometry, `d_xi y_FRW` (Section 7.6).

    `d_xi Delta V_c` is the third line of eq:num:geom, `d_xi X_j` is the map's own velocity, `W` stays zero, and
    `d_xi X_{j_e}^3 = 3 X_{j_e}^2 (d_xi X)_{j_e}` (Section 8.1). All vanish on a static map, where the deviation form
    and the direct form coincide.

    Args:
        geo: The geometry at the time in question.
        j_e: The index of the excision face, `0` before excision (as in `Layout`).
        hubble: The background coefficient `h` of `Background`; the background velocity is `h X`.

    Returns:
        `d_xi y_FRW` in the shape of a `State`.
    """
    X_e, X_xi_e = float(geo.X[j_e]), float(geo.X_xi[j_e])  # `X_e**2` by pow, not `geo.X2[j_e]` (see `Geometry`)
    return State(E=geo.dV_xi.copy(), U=hubble * geo.X_xi, W=0.0, M_e=3.0 * X_e**2 * X_xi_e)


@dataclass(frozen=True)
class FrwReference:
    """The background on one geometry, as a stage uses it: computed once per frame and shared by its stages.

    Nothing here depends on the state, so every stage on this geometry reuses it (`timestep.Frame`: one per time on a
    moving map, one for the Scheme's lifetime on a static map); its arrays are read-only for that reason.
    `equations.calc_derivs` builds one itself when not handed one.

    Attributes:
        geo: The geometry it was formed on.
        eos: The equation of state.
        j_e: The excision face, as in `Layout`.
        hubble: The background coefficient `h`.
        state: The background state, `frw_state(geo, j_e, h)`.
        rate: Its time derivative, `frw_rate(geo, j_e, h)`.
        frw_speed: `alpha w h X_j - (d_xi X)_j` (faces), the velocity at which the background's energy crosses the
            moving face: the pressure work of the Hubble flow against the face's own motion.
        F_frw: The FRW energy flux `frw_speed X_j^2` (faces), to which a stage adds its deviation.
    """

    geo: Geometry
    eos: EquationOfState
    j_e: int
    hubble: float
    state: State
    rate: State
    frw_speed: FloatArray
    F_frw: FloatArray

    @classmethod
    def of(cls, geo: Geometry, eos: EquationOfState, j_e: int = 0, hubble: float = 1.0) -> Self:
        """The reference on this geometry; `hubble` is the background coefficient `h` of `Background`."""
        alpha, w = eos.alpha_float, eos.w_float
        frw_speed = alpha * w * hubble * geo.X - geo.X_xi
        reference = cls(
            geo=geo,
            eos=eos,
            j_e=j_e,
            hubble=hubble,
            state=frw_state(geo, j_e, hubble),
            rate=frw_rate(geo, j_e, hubble),
            frw_speed=frw_speed,
            F_frw=frw_speed * geo.X2,
        )
        read_only(reference.state.E, reference.state.U, reference.rate.E, reference.rate.U, frw_speed, reference.F_frw)
        return reference

    def belongs_to(self, geo: Geometry, eos: EquationOfState, j_e: int, hubble: float) -> bool:
        """Whether this is the reference of that geometry (the same object), equation of state, `j_e` and `h`."""
        return geo is self.geo and j_e == self.j_e and hubble == self.hubble and eos == self.eos


def deviation_from_frw(state: State, geo: Geometry, j_e: int = 0, hubble: float = 1.0) -> State:
    """The deviation `state - y_FRW` recovered from a whole state, for callers that do not hold the integrator's.

    The integrator's deviation (Section 7.6) is exact where this one keeps only what survived adding the deviation to
    `y_FRW`; the two agree to the rounding of the FRW values, and exactly on the FRW state itself. `hubble` is the
    background coefficient `h` of `Background` (0 in flat spacetime, where the background is the fluid at rest).
    """
    frw = frw_state(geo, j_e, hubble)
    return State(E=state.E - frw.E, U=state.U - frw.U, W=state.W, M_e=state.M_e - frw.M_e)


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
