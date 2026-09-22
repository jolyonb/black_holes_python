"""The closure of the outer face (paper Section 7.5, with the boundary ODE of Section 5.3).

The interior rows of Section 7.3 stop one face short of the boundary: the velocity row at face `N` would need a
pressure difference across the face, which needs a density beyond it, and the flux `F_N` through the outer face
into the last cell is not fixed by the interior either. An outer closure supplies exactly those two things, plus the
rate of the auxiliary scalar `W` of Section 7.5 when it carries one:

    d_xi U_N,   F_N,   d_xi W.

`OutgoingWave` is the production closure, the exact outgoing-wave condition of Section 5.3 imposed as a penalty
(eq:num:sat). Of the two acoustic characteristics at the outer face one leaves the domain and one enters it, so the
boundary may say one thing, about the incoming amplitude `u_-`, and the exact condition says what it must be: the
solution `W` of the boundary ODE eq:lin:bcode, driven by the outgoing amplitude `u_+` and the mass perturbation that
the interior delivers. Rather than overwrite a stored value, which would over-determine the face and reflect, the
scheme keeps face `N` a live unknown, advances `W` with the state, and relaxes both members of the pair towards
`u_- = W` in proportion to the residual `pen = u_- - W`, a simultaneous approximation term (Carpenter 1994). The
strengths `(tau_u, tau_rho, tau_W)` are fixed by the energy estimate and by accuracy, see `PenaltyStrengths`.

`HeldAtFrw` holds the outer face at its FRW value, `U_N = X_N`, which the paper uses for the converging-pulse and
converging-shell tests of Table tab:num:tests ("with the outer face held at FRW, `U_N = X_N` in place of
eq:num:sat"); the flux is the base flux of eq:num:energy at face `N` with the extrapolated face density, and `W` is
not used.

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
        delta_U_N: The relative velocity deviation at the outer face, `U_N / X_N - 1`.
        delta_rho_N_1: The relative density deviation of the last cell, `rho_{N-1} - 1`, half a cell inside the face.
        rho_f_N: The extrapolated face density `<rho>_N`.
        ephi_f_N: The extrapolated face lapse `<ephi>_N`.
        mt_N: The tilde mass at the outer face.
        delta_m_N: Its relative deviation, `mt_N - 1`.
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
    delta_U_N: float
    delta_rho_N_1: float
    rho_f_N: float
    ephi_f_N: float
    mt_N: float
    delta_m_N: float
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


# --- the exact outgoing-wave condition as a penalty (Section 7.5) ---


def boundary_ode_coefficients(c_s: float, R: float) -> tuple[float, float, float]:
    """The coefficients `(gamma_-, gamma_+, gamma_0)` of the boundary ODE eq:lin:bcode at radius `R`.

    The exact outgoing-wave condition of Section 5.3, written in the characteristic variables, is an ODE for the
    incoming amplitude at the boundary,

        d_xi u_- = gamma_- u_- + gamma_+ u_+ + gamma_0 delta_m,
        gamma_- = -(R^2 + 4 c_s^2) / (4 R (R + 2 c_s)),   gamma_+ = -(R - 2 c_s) / (4 R),
        gamma_0 = -c_s / (2 (R + 2 c_s)),

    which needs nothing from outside the domain. `gamma_-` is negative for every `c_s / R` (its supremum is
    `-(sqrt 2 - 1) / 2`), so the incoming amplitude that the curvature and expansion terms generate is damped.
    """
    gamma_minus = -(R * R + 4.0 * c_s * c_s) / (4.0 * R * (R + 2.0 * c_s))
    gamma_plus = -(R - 2.0 * c_s) / (4.0 * R)
    gamma_0 = -c_s / (2.0 * (R + 2.0 * c_s))
    return gamma_minus, gamma_plus, gamma_0


def characteristic_pair(delta_U_N: float, delta_rho_N_1: float, X_N: float, c_s: float) -> tuple[float, float]:
    """The discrete characteristic pair `(u_+, u_-)` at the outer face, eq:lin:charvars on the staggered grid.

    `u_pm = delta_U,N pm kappa delta_rho,N-1` with `kappa = 3 c_s / (2 X_N)`, built from the face-`N` velocity
    deviation, `delta_U,N = U_N / X_N - 1`, and the last cell's density deviation, `delta_rho,N-1 = rho_{N-1} - 1`,
    half a cell inside the face, which is where the staggered pair has a cell field; both are passed as deviations,
    as `derive` forms them. `u_+` is carried outward at `c_s` and `u_-` inward. The initial value of `W` is the
    initial `u_-`, so that the penalty starts at zero.
    """
    kappa = 1.5 * c_s / X_N
    return delta_U_N + kappa * delta_rho_N_1, delta_U_N - kappa * delta_rho_N_1


@dataclass(frozen=True)
class PenaltyStrengths:
    """The three penalty strengths `(tau_u, tau_rho, tau_W)` of eq:num:sat.

    Two constraints choose them (Section 7.5). The energy estimate needs the boundary quadratic form in
    `(u_+, u_-)` to be negative definite, which holds if and only if

        4 (tau_u + tau_rho - 1) > (tau_u - tau_rho)^2,

    and strengths that violate it are refused, since the closure then carries no energy statement. Accuracy needs
    more: the pair is half a cell off the face, an `O(Delta X)` defect, and the flux velocity and the ODE forcing are
    second order only along the line `tau_rho = 2 - tau_u / 2`, `tau_W = 1 - tau_u / 2`; strengths off that line
    make the boundary first order, which `is_second_order` reports. The production choice is `(2, 1, 0)`, the member
    of the line at which the raw pair is itself second order at the face and the flux velocity is the
    characteristic value `U*_N = X_N (1 + (u_+ + W) / 2)`.

    The strengths are unrelated to the sound horizon `tau` of the background.
    """

    tau_u: float
    tau_rho: float
    tau_W: float

    def __post_init__(self) -> None:
        if not 4.0 * (self.tau_u + self.tau_rho - 1.0) > (self.tau_u - self.tau_rho) ** 2:
            raise ValueError(
                f"the penalty strengths {self} carry no energy statement: "
                "4 (tau_u + tau_rho - 1) > (tau_u - tau_rho)^2 fails"
            )

    @property
    def is_second_order(self) -> bool:
        """Whether the strengths lie on the second-order line `tau_rho = 2 - tau_u / 2`, `tau_W = 1 - tau_u / 2`."""
        return self.tau_rho == 2.0 - 0.5 * self.tau_u and self.tau_W == 1.0 - 0.5 * self.tau_u


#: The production strengths `(tau_u, tau_rho, tau_W) = (2, 1, 0)` of Section 7.5.
PRODUCTION_STRENGTHS = PenaltyStrengths(tau_u=2.0, tau_rho=1.0, tau_W=0.0)


@dataclass(frozen=True)
class OutgoingWave(OuterClosure):
    """The exact outgoing-wave condition imposed as a penalty, eq:num:sat: the production closure.

    With `pen = u_- - W` the residual of the condition and `(gamma_-, gamma_+, gamma_0)` the ODE coefficients at
    `R = X_N`, the three rows are

        d_xi U_N = (1 - alpha) U_N - (alpha / 2) <ephi>_N X_N (mt_N + 3 w <rho>_N) - Theta_N (D_U U)_N
                   - tau_u c_s (X_N^2 / dS_N) pen,
        U*_N     = U_N - (tau_rho X_N / 2) pen,
        F_N      = [alpha ((1 + w) <ephi>_N U*_N - X_N) - (d_xi X)_N] X_N^2 <rho>_N,
        d_xi W   = gamma_- W + gamma_+ (u_+ - tau_W pen) + gamma_0 delta_m,N,

    the velocity row of eq:num:velocity without the pressure difference plus the penalty; the last cell's flux with
    the characteristic face velocity `U*_N`; and the boundary ODE fed with the interior's outgoing amplitude. Every
    term vanishes on FRW, where `U_N = X_N`, `rho = mt = 1` and `W = 0`.

    The rows are derived for radiation, the only equation of state besides dust for which the exact condition
    exists. For any other `w` the closure holds `W = 0`, a Sommerfeld-type condition with the same rows, and the
    absorbing and energy statements of Section 7.5 do not apply.

    The outer face must be static, `(d_xi X)_N = 0`: the extra term of eq:lin:bcnonlinear on a moving face is derived
    but not implemented, and every admissible map keeps its outer face fixed (Section 8.1).
    """

    strengths: PenaltyStrengths = PRODUCTION_STRENGTHS

    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        """The three rows of eq:num:sat."""
        if inputs.X_xi_N != 0.0:
            raise ValueError("the outgoing-wave closure needs a static outer face, (d_xi X)_N = 0")
        alpha, w = float(eos.alpha), float(eos.w)
        tau = self.strengths
        X_N, c_s = inputs.X_N, inputs.c_s

        u_plus, u_minus = characteristic_pair(inputs.delta_U_N, inputs.delta_rho_N_1, X_N, c_s)
        pen = u_minus - inputs.W

        expansion = (1.0 - alpha) * inputs.U_N
        gravity = -0.5 * alpha * inputs.ephi_f_N * X_N * (inputs.mt_N + 3.0 * w * inputs.rho_f_N)
        advection = -inputs.Theta_N * inputs.DU_N
        penalty = -tau.tau_u * c_s * X_N**2 / inputs.dS_N * pen
        dU_N = expansion + gravity + advection + penalty

        U_star = inputs.U_N - 0.5 * tau.tau_rho * X_N * pen
        flux_velocity = alpha * ((1.0 + w) * inputs.ephi_f_N * U_star - X_N)
        F_N = flux_velocity * X_N**2 * inputs.rho_f_N

        if eos.is_radiation:
            gamma_minus, gamma_plus, gamma_0 = boundary_ode_coefficients(c_s, X_N)
            dW = gamma_minus * inputs.W + gamma_plus * (u_plus - tau.tau_W * pen) + gamma_0 * inputs.delta_m_N
        else:
            dW = 0.0  # W = 0 for every other equation of state
        return OuterRows(dU_N=dU_N, F_N=F_N, dW=dW)


@dataclass(frozen=True)
class HeldExterior(OuterClosure):
    """The outer face held on a steady exterior (the tests of Section 8.3: the Michel flow outside the near zone).

    The face's density and lapse are the exact steady values, the velocity follows the scaling of the tilde
    variables at a fixed physical radius, `d_xi U_N = (1 - alpha) U_N` (eq:exc:growth), and the flux is the base
    flux with those face values; `W` is not used. The face may move with the pinned map, since the flux is written
    relative to it. A test closure: it presumes the exterior is known.

    Attributes:
        rho_N: The steady density `rho / rho_inf` at the outer face.
        ephi_N: The steady lapse at the outer face.
    """

    rho_N: float
    ephi_N: float

    def rows(self, inputs: OuterInputs, eos: EquationOfState) -> OuterRows:
        """`d_xi U_N = (1 - alpha) U_N`, `F_N = (cE_N - (d_xi X)_N) X_N^2 rho_N` with the held face values."""
        alpha, w = float(eos.alpha), float(eos.w)
        cE_N = alpha * ((1.0 + w) * self.ephi_N * inputs.U_N - inputs.X_N)
        F_N = (cE_N - inputs.X_xi_N) * inputs.X_N**2 * self.rho_N
        return OuterRows(dU_N=(1.0 - alpha) * inputs.U_N, F_N=F_N, dW=0.0)
