"""The closure of the outer face (paper Section 7.5, and the held face of the tests in Section 7.8).

The interior rows of Section 7.3 stop one face short of the boundary: the velocity row at face `N` would need a
pressure difference across the face, which needs a density beyond it, and the flux `F_N` through the outer face
into the last cell is not fixed by the interior either. An outer closure supplies exactly those two things, plus the
rate of the auxiliary scalar `W` of Section 7.5 when it carries one:

    d_xi U_N,   F_N,   d_xi W.

Two closures are used before the production one exists. `HeldAtFrw` holds the outer face at its FRW value,
`U_N = X_N`, which the paper uses for the converging-pulse and converging-shell tests of Table tab:num:tests ("with
the outer face held at FRW, `U_N = X_N` in place of eq:num:sat"); the flux is the base flux of eq:num:energy at
face `N` with the extrapolated face density, and `W` is not used. The production closure, the exact outgoing-wave
condition imposed as a penalty (eq:num:sat), and the `W = 0` closure for `w != 1/3` are added in a later bite.

Every closure takes the same `OuterInputs`, the handful of face-`N` quantities the rows can depend on, so that the
stage does not know which closure it is talking to.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from pbh.eos import EquationOfState


@dataclass(frozen=True)
class OuterInputs:
    """What an outer closure may look at: the state and the derived fields at and just inside the outer face.

    Attributes:
        xi: The time.
        X_N: The radius of the outer face.
        X_xi_N: The velocity of the outer face, `(d_xi X)_N`; zero on every admissible map (Section 8.1).
        U_N: The velocity at the outer face.
        W: The auxiliary scalar of Section 7.5.
        rho_N_1: The density of the last cell, `rho_{N-1}`, half a cell inside the face.
        rho_f_N: The extrapolated face density `<rho>_N`.
        ephi_f_N: The extrapolated face lapse `<ephi>_N`.
        mt_N: The tilde mass at the outer face.
        Theta_N: The grid velocity at the outer face.
        cE_N: The energy-flux velocity at the outer face (eq:num:facefields).
        DU_N: The one-sided velocity gradient at the outer face.
        dS_N: The difference of mean-square radii across the outer face.
        c_s: The background sound speed at this time.
    """

    xi: float
    X_N: float
    X_xi_N: float
    U_N: float
    W: float
    rho_N_1: float
    rho_f_N: float
    ephi_f_N: float
    mt_N: float
    Theta_N: float
    cE_N: float
    DU_N: float
    dS_N: float
    c_s: float


@dataclass(frozen=True)
class OuterRows:
    """What an outer closure returns: the three rows the interior cannot supply.

    Attributes:
        dU_N: `d_xi U_N`.
        F_N: The energy flux through the outer face into the last cell.
        dW: `d_xi W`; zero for a closure that carries no `W`.
    """

    dU_N: float
    F_N: float
    dW: float


class OuterClosure(ABC):
    """The interface every outer closure provides."""

    @abstractmethod
    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        """The three outer rows for these face-`N` quantities."""


@dataclass(frozen=True)
class HeldAtFrw(OuterClosure):
    """The outer face held at its FRW value, `U_N = X_N` (Section 7.8; a test closure).

    The velocity follows the face, `d_xi U_N = (d_xi X)_N`, which is zero on a static map; the flux through the face
    is the base flux of eq:num:energy with the extrapolated face density; `W` is not used. A held face reflects, so
    this closure serves only tests whose signals never reach the boundary or are meant to reflect from it.
    """

    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        """`d_xi U_N = (d_xi X)_N`, `F_N = (cE_N - (d_xi X)_N) X_N^2 <rho>_N`, `d_xi W = 0`."""
        F_N = (inputs.cE_N - inputs.X_xi_N) * inputs.X_N**2 * inputs.rho_f_N
        return OuterRows(dU_N=inputs.X_xi_N, F_N=F_N, dW=0.0)
