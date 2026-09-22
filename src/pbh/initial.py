"""The initial data: a growing-mode perturbation from one profile (paper Sections 5.4 and 7.9).

How it is used. A production datum is one profile in and a `State` out, with two reports on the way:

    bg_0 = Background.at(eos, xi_0)
    mode, report = GrowingMode.from_profile(Gaussian(A, ell), "m", Rtilde_max, bg_0)  # projection and guards
    state = mode.state(geo, bg_0)                                 # sampled, corrected, admissible
    ratio = mode.correction_ratio(geo, bg_0)                      # more than a few per cent: start earlier

Linear data with both fields, and so possibly decaying content, are the general `ModeExpansion` of Section 5.4.1,
of which the growing mode is the `C_n = 0` special case; they are carried to the start time by evaluating the
expansion there, and sampled without the correction, which is derived for the growing solution:

    expansion = ModeExpansion.from_fields(delta_rho, delta_U, Rtilde_max, bg_i)   # never singular
    state = expansion.state(geo, bg_0)

Everything here is for radiation: the mode functions are the Bessel functions of order one of eq:lin:mode, and
eq:lin:idata and eq:num:dUnl are printed for radiation only; the generator refuses any other equation of state.

Complete data from any other source, both fields already known on the grid, go through the same last door:

    state = initial_state(E, U, geo, bg_0)                        # U_0 = 0, W, admissibility

The generator of the initial-data file calls these and records the state with the specification, the `report` and
the `ratio` that produced it; the driver reads that file and never calls this module.

The production data are the compensated one-field growing mode of Section 5.4 with the nonlinear correction of its
velocity (Section 7.9). One profile is the input, the density perturbation `delta_rho` or the mass perturbation
`delta_m` at the start time as a function of radius, the `Profile` contract below (`profiles.py` holds the families),
and everything else follows from it in closed form:

1. Projection (eq:lin:bn). On `0 <= X <= Rtilde_max` the functions `j_0(k_n X)` with `k_n = n pi / Rtilde_max`
   vanish at the edge and are orthogonal with weight `X^2`, so the profile has coefficients `b_n` on them; a mass
   profile has the same coefficients on `3 j_1(k_n X) / (k_n X)`, found without differentiating it.
2. The growing mode (eq:lin:idata). Matching the growing branch of eq:lin:mode, `C_n = 0`, at the start time gives
   the amplitudes `B_n = b_n / (z_n j_1(z_n))`, `z_n = k_n tau_0`, from which the density, the mass and the linear
   velocity follow at any time. The division is ill defined at every zero of `j_1`, the first at `z = 4.4934`, a
   structure finer than about 1.4 sound horizons, so the construction reports the fraction of the power that the
   modes with `z_n > 3` carry, which covers every zero, in the input and in the output velocity, and refuses the
   data if either exceeds `1e-8`. The data are also refused unless compensated, `delta_m(Rtilde_max) = 0` to a
   tolerance (eq:lin:compensated), since the exterior must be FRW for the outer boundary to mean anything.
3. The nonlinear correction (eq:num:dUnl). What the linear recipe omits at the next order of the gradient
   expansion is a quadratic form in `delta_m` with no explicit time dependence, added to the velocity.
4. Sampling (eq:num:idata). The cell contents are exact, `E_c = [X^3 (1 + delta_m)] / 3` across the cell, so the
   cumulative sum returns `mt_j = 1 + delta_m(X_j)` exactly, and the face velocities are point values.

Whatever its source, every datum then goes through `initial_state`: `U_0 = 0`, `W` at the discrete `u_-` so that the
outer penalty starts at zero, and the admissibility check, `Gammabar^2 > 0` at every face and `rho > 0` in every cell.
Complete data from elsewhere, the two-field linear data of Section 5.4.1, or test data with velocities of their own
use that door directly. For the growing mode the diagnostic `max |delta_U^nl / delta_U^lin|` is reported: it is the
size of what a linear recipe would have omitted, and more than a few per cent is a signal to start earlier. All of it
is for radiation, the only equation of state for which eq:lin:idata is printed.

The guard of step 2 and the diagnostic are two thresholds on one parameter, the profile's scales against the horizon
at the start (`z = k tau_0` mode by mode; `epsilon_0 = 1 / (H r_m)` at the compaction peak; the Hubble radius is `sqrt
3 tau` for radiation). The linear recipe captures the linear evolution at every scale, including beyond the point
where the gradient expansion breaks down: the diagnostic is the graded warning that the nonlinear terms it cannot
contain are no longer small, since with the correction the datum is wrong at relative `O(epsilon_0^4)`. The guard is
the cliff: once a mode is in the oscillatory regime one field no longer determines it, and no recipe of this kind can
operate. The guard sees the finest scale present and the diagnostic the peak, so for a profile with a sharp feature on
a broad core the guard can fail while the diagnostic is quiet, and only the two-field data of Section 5.4.1 can carry
such a profile.

For test data given as a density alone, with a velocity that is not the growing mode's, `cell_contents` integrates
the density over each cell by Gauss-Legendre quadrature, exact to round-off for smooth profiles.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pbh.derived import NotHyperbolicError, gammabar_squared
from pbh.eos import Background
from pbh.geometry import Geometry, shell_volumes
from pbh.outer import characteristic_pair
from pbh.state import State
from pbh.types import FloatArray

type Profile = Callable[[FloatArray], FloatArray]
"""A radial profile: the perturbation as a function of the scaled areal radius, vectorised (`profiles.py`)."""

type Field = Literal["rho", "m"]
"""Which perturbation a profile gives: the density `delta_rho` or the mass `delta_m`."""

#: The first zero of j_1. The one-field reconstruction is ill defined at every zero of j_1, and this is the one that
#: a profile's structure reaches first as it gets finer; the guard on z_n > 3 covers all of them at once.
J1_FIRST_ZERO = 4.493409457909064


# --- spherical Bessel quotients that are finite at the origin ---


def j1_over_x(x: FloatArray) -> FloatArray:
    """`j_1(x) / x`, which is `1/3` at the origin; scipy's quotient is accurate to round-off at every `x > 0`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x == 0.0, 1.0 / 3.0, spherical_jn(1, x) / x)


def j2_over_x2(x: FloatArray) -> FloatArray:
    """`j_2(x) / x^2`, which is `1/15` at the origin; scipy's quotient is accurate to round-off at every `x > 0`."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x == 0.0, 1.0 / 15.0, spherical_jn(2, x) / x**2)


# --- the mode expansion and its growing-mode special case ---


class IllPosedDataError(ValueError):
    """The one-field reconstruction cannot read the growing mode from this profile (Section 5.4)."""


class NotCompensatedError(ValueError):
    """The data leave a mass excess at the outer edge, so the exterior is not FRW (eq:lin:compensated)."""


#: The largest `|delta_m(Rtilde_max)|`, relative to the largest `|delta_m|` inside, that counts as compensated.
COMPENSATION_TOLERANCE = 1e-6


def box_wavenumbers(Rtilde_max: float, n_modes: int) -> FloatArray:
    """The wavenumbers `k_n = n pi / Rtilde_max`, `n = 1..n_modes`, whose `j_0(k_n X)` vanish at the outer edge."""
    return np.arange(1, n_modes + 1, dtype=np.float64) * (np.pi / Rtilde_max)


def project(profile: Profile, field: Field, k: FloatArray, Rtilde_max: float) -> FloatArray:
    """The coefficients `b_n` of a density profile on `j_0(k_n X)`, or of a mass profile on `3 j_1(k_n X) / (k_n X)`.

    eq:lin:bn for the density; for the mass the same coefficients are, after an integration by parts,
    `b_n = (2 k_n^3 / 3 Rtilde_max) int X^3 j_1(k_n X) delta_m dX`. Composite Gauss-Legendre quadrature, at least two
    panels per mode.
    """
    n_panels = max(200, 2 * k.size)
    nodes, weights = np.polynomial.legendre.leggauss(16)
    edges = np.linspace(0.0, Rtilde_max, n_panels + 1)
    mid, half = 0.5 * (edges[1:] + edges[:-1]), 0.5 * (edges[1:] - edges[:-1])
    X = (mid[:, None] + half[:, None] * nodes[None, :]).ravel()
    w = (half[:, None] * weights[None, :]).ravel()
    kX = np.outer(k, X)
    if field == "rho":
        return (2.0 * k**2 / Rtilde_max) * (spherical_jn(0, kX) @ (w * X**2 * profile(X)))
    return (2.0 * k**3 / (3.0 * Rtilde_max)) * (spherical_jn(1, kX) @ (w * X**3 * profile(X)))


@dataclass(frozen=True)
class Projection:
    """What the projection reports about a profile.

    Attributes:
        power_fraction: The fraction of the input's power `sum b_n^2 / k_n^2` carried by the modes with `z_n > 3`.
        amplified_fraction: The same fraction of the output velocity's power, after the division by `z_n j_1(z_n)`.
        distance_to_first_zero: The smallest `|z_n - 4.4934|` over the modes: how close one lies to the first of
            the ill-defined points, the one that matters for a profile near the edge of admissibility.
        edge_value: `delta_m(Rtilde_max)` of the reconstruction, the compensation residual (checked).
    """

    power_fraction: float
    amplified_fraction: float
    distance_to_first_zero: float
    edge_value: float


@dataclass(frozen=True)
class ModeExpansion:
    """Linear data as a sum of modes on the box basis `k_n = n pi / Rtilde_max`, both branches (Section 5.4.1).

    Each mode has a growing amplitude `B_n` and a decaying amplitude `C_n` (eq:lin:mode). With `z = k_n tau` and
    every Bessel function evaluated at `z`, the density and velocity coefficients at that time are

        a_n = z [B_n j_1 + C_n y_1],   u_n = -(z / 4) [B_n (j_1 - z j_2) + C_n (y_1 - z y_2)],   eq:lin:modepair

    and the fields are

        delta_rho = sum a_n j_0(k_n X),
        delta_m = sum 3 a_n j_1(k_n X) / (k_n X),
        delta_U = sum 3 u_n j_1(k_n X) / (k_n X).

    Evaluating the fields at another time is the transport of Section 5.4.1: the decaying content given at `tau_i`
    is suppressed by `(tau_i / tau)^3` for modes outside the sound horizon.

    Attributes:
        k: The wavenumbers `k_n`.
        B: The growing amplitudes `B_n`.
        C: The decaying amplitudes `C_n`.
    """

    k: FloatArray
    B: FloatArray
    C: FloatArray

    @classmethod
    def from_fields(
        cls, delta_rho: Profile, delta_U: Profile, Rtilde_max: float, bg_i: Background, n_modes: int = 150
    ) -> Self:
        """The expansion of a density and a velocity given at the time `bg_i`: eq:lin:un and eq:lin:modeinverse.

        The density's coefficients `b_n` are the projection eq:lin:bn and the velocity's `u_n` the projection of
        `delta_U` on `3 j_1(k_n X) / (k_n X)`; inverting eq:lin:modepair at `z_i = k_n tau_i`, whose determinant is
        one by `j_2 y_1 - j_1 y_2 = 1 / z^2`, gives the amplitudes. The system is never singular: the ill-definedness
        at the zeros of `j_1` belongs to the one-field reconstruction alone.
        """
        k = box_wavenumbers(Rtilde_max, n_modes)
        b = project(delta_rho, "rho", k, Rtilde_max)
        u = project(delta_U, "m", k, Rtilde_max)
        z = k * bg_i.tau
        j1, j2 = spherical_jn(1, z), spherical_jn(2, z)
        y1, y2 = spherical_yn(1, z), spherical_yn(2, z)
        B = (y1 - z * y2) * b + 4.0 * y1 * u
        C = -(j1 - z * j2) * b - 4.0 * j1 * u
        expansion = cls(k=k, B=B, C=C)
        expansion.check_compensated(bg_i, Rtilde_max)
        return expansion

    def check_compensated(self, bg: Background, Rtilde_max: float) -> None:
        """Refuse the data unless the exterior is FRW: `delta_m(Rtilde_max)` below `COMPENSATION_TOLERANCE`.

        `mt` at the edge is the mass inside the box in units of the background's, so an FRW exterior needs
        `delta_m(Rtilde_max) = 0` (eq:lin:compensated): the overdensity must be surrounded by an underdensity
        carrying the same mass. Measured relative to the largest `|delta_m|` on the box.
        """
        X = np.linspace(0.0, Rtilde_max, 4 * self.k.size + 1)
        delta_m = self.delta_m(bg, X)
        edge, largest = abs(float(delta_m[-1])), float(np.max(np.abs(delta_m)))
        if edge > COMPENSATION_TOLERANCE * largest:
            raise NotCompensatedError(
                f"delta_m at the outer edge is {edge:.2e}, {edge / largest:.1e} of its largest value "
                f"(tolerance {COMPENSATION_TOLERANCE:.0e}): the profile is not compensated, so the exterior is not "
                "FRW; widen the box or balance the mass excess"
            )

    def _coefficients(self, bg: Background) -> tuple[FloatArray, FloatArray]:
        """The density and velocity coefficients `(a_n, u_n)` at this time, eq:lin:modepair."""
        z = self.k * bg.tau
        j1, j2 = spherical_jn(1, z), spherical_jn(2, z)
        if np.any(self.C != 0.0):
            y1, y2 = spherical_yn(1, z), spherical_yn(2, z)
        else:
            y1 = y2 = np.zeros_like(z)  # the growing mode: no evaluation of the branch that is singular at z = 0
        a = z * (self.B * j1 + self.C * y1)
        u = -0.25 * z * (self.B * (j1 - z * j2) + self.C * (y1 - z * y2))
        return a, u

    def delta_rho(self, bg: Background, X: FloatArray) -> FloatArray:
        """The relative density perturbation at the radii `X`."""
        a, _ = self._coefficients(bg)
        return spherical_jn(0, np.multiply.outer(X, self.k)) @ a

    def delta_m(self, bg: Background, X: FloatArray) -> FloatArray:
        """The mass perturbation `mt - 1` at the radii `X`."""
        a, _ = self._coefficients(bg)
        return 3.0 * j1_over_x(np.multiply.outer(X, self.k)) @ a

    def delta_m_derivatives(self, bg: Background, X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """`(delta_m, delta_m', delta_m'')` at the radii `X`, primes in `X`.

        From the identities `d/dx [j_1/x] = -x [j_2/x^2]` and `d/dx [j_2/x^2] = (j_1/x - 5 j_2/x^2) / x`.
        """
        a, _ = self._coefficients(bg)
        kX = np.multiply.outer(X, self.k)
        f1, f2 = j1_over_x(kX), j2_over_x2(kX)
        first = -3.0 * (f2 * X[..., None]) @ (a * self.k**2)
        second = -3.0 * (f1 - 4.0 * f2) @ (a * self.k**2)
        return 3.0 * f1 @ a, first, second

    def delta_U(self, bg: Background, X: FloatArray) -> FloatArray:
        """The linear velocity perturbation `U / X - 1` at the radii `X`."""
        _, u = self._coefficients(bg)
        return 3.0 * j1_over_x(np.multiply.outer(X, self.k)) @ u

    def state(self, geo: Geometry, bg: Background) -> State:
        """The linear data sampled on the grid at this time: exact cell contents, point velocities, through the door."""
        X = geo.X[: geo.N + 1]
        E = np.diff(X**3 * (1.0 + self.delta_m(bg, X))) / 3.0
        return initial_state(E, X * (1.0 + self.delta_U(bg, X)), geo, bg)


@dataclass(frozen=True, init=False)
class GrowingMode(ModeExpansion):
    """The growing branch alone, `C_n = 0`: the production datum, reconstructed from one profile (Section 5.4)."""

    def __init__(self, k: FloatArray, B: FloatArray) -> None:
        super().__init__(k=k, B=B, C=np.zeros_like(B))

    @classmethod
    def from_profile(
        cls,
        profile: Profile,
        field: Field,
        Rtilde_max: float,
        bg_0: Background,
        n_modes: int = 150,
        power_tolerance: float = 1e-8,
    ) -> tuple[Self, Projection]:
        """The growing mode whose `delta_rho` (`field = "rho"`) or `delta_m` (`field = "m"`) at `bg_0` is the profile.

        Raises:
            IllPosedDataError: If the modes with `z_n > 3` carry more than `power_tolerance` of the power, in the
                input or in the output velocity.
        """
        k = box_wavenumbers(Rtilde_max, n_modes)
        b = project(profile, field, k, Rtilde_max)
        z = k * bg_0.tau
        with np.errstate(divide="ignore", invalid="ignore"):
            B = b / (z * spherical_jn(1, z))
        mode = cls(k, B)
        mode.check_compensated(bg_0, Rtilde_max)  # a property of the input: checked before the guards below
        fine = z > 3.0
        power = b**2 / k**2
        power_fraction = float(np.sum(power[fine]) / np.sum(power))
        if power_fraction > power_tolerance:
            raise IllPosedDataError(
                f"the modes with k tau_0 > 3 carry the fraction {power_fraction:.2e} of the profile's power "
                f"(tolerance {power_tolerance:.0e}): the growing mode cannot be read from one field this close to "
                f"the first zero of j_1 at k tau_0 = {J1_FIRST_ZERO:.4f}; start earlier"
            )
        velocity_power = (z / 4.0 * B * (spherical_jn(1, z) - z * spherical_jn(2, z))) ** 2 / k**2
        amplified_fraction = float(np.sum(velocity_power[fine]) / np.sum(velocity_power))
        if not amplified_fraction <= power_tolerance:  # written so that a mode exactly on a zero, nan, is refused
            raise IllPosedDataError(
                f"after the division by z j_1(z) the modes with k tau_0 > 3 carry the fraction "
                f"{amplified_fraction:.2e} of the velocity's power (tolerance {power_tolerance:.0e}); "
                "start earlier or move Rtilde_max"
            )
        report = Projection(
            power_fraction=power_fraction,
            amplified_fraction=amplified_fraction,
            distance_to_first_zero=float(np.min(np.abs(z - J1_FIRST_ZERO))),
            edge_value=float(mode.delta_m(bg_0, np.array([Rtilde_max]))[0]),
        )
        return mode, report

    def state(self, geo: Geometry, bg: Background, nonlinear: bool = True) -> State:
        """The growing mode sampled on the grid as the paper prescribes (eq:num:idata).

        The cell contents are exact and the face velocities carry the correction eq:num:dUnl unless `nonlinear` is
        off; the correction is derived for the growing solution, which is why it lives here and not on the general
        expansion. Radiation only, as eq:lin:idata and eq:num:dUnl are.
        """
        if not nonlinear:
            return super().state(geo, bg)
        X = geo.X[: geo.N + 1]
        delta_m, first, second = self.delta_m_derivatives(bg, X)
        delta_U = self.delta_U(bg, X) + nonlinear_correction(X, delta_m, first, second)
        E = np.diff(X**3 * (1.0 + delta_m)) / 3.0
        return initial_state(E, X * (1.0 + delta_U), geo, bg)

    def correction_ratio(self, geo: Geometry, bg: Background) -> float:
        """The diagnostic of Section 7.9: `max |delta_U^nl / delta_U^lin|` over the grid's faces.

        Taken over the radii where `|delta_U^lin|` exceeds `1e-3` of its maximum, so that a node of the linear
        velocity does not enter. It is the size of what a linear recipe would have omitted; more than a few per
        cent is a signal to start earlier.
        """
        X = geo.X[: geo.N + 1]
        delta_m, first, second = self.delta_m_derivatives(bg, X)
        linear = self.delta_U(bg, X)
        correction = nonlinear_correction(X, delta_m, first, second)
        significant = np.abs(linear) > 1e-3 * np.max(np.abs(linear))
        return float(np.max(np.abs(correction[significant] / linear[significant])))


# --- the projection, the correction, and the door every datum goes through ---


def nonlinear_correction(X: FloatArray, delta_m: FloatArray, first: FloatArray, second: FloatArray) -> FloatArray:
    """The velocity correction eq:num:dUnl, the quadratic form in `delta_m` that no linear recipe contains.

    `delta_U^nl = [delta_m^2 + 12 X delta_m delta_m' + 4 X^2 delta_m delta_m'' - X^2 (delta_m')^2] / 160`.
    """
    return (delta_m**2 + 12.0 * X * delta_m * first + 4.0 * X**2 * delta_m * second - X**2 * first**2) / 160.0


def initial_state(E: FloatArray, U: FloatArray, geo: Geometry, bg: Background, W: float | None = None) -> State:
    """The state of any complete initial data: cell contents and face velocities from whatever source.

    Every datum passes through here: `U_0 = 0` is imposed, since face `0` carries no unknown; the auxiliary scalar
    is kept if given and otherwise starts at the discrete incoming amplitude `u_-`, so that the outer penalty of
    eq:num:sat starts at zero; and the data are refused unless admissible. The growing-mode recipe above is one
    source; two-field linear data, which know their own `W`, and test data with velocities of their own are others.
    A checkpoint of a run is not an initial datum and does not come through here: it is restored as it was written.
    """
    N = geo.N
    X = geo.X[: N + 1]
    U = U.copy()
    U[0] = 0.0
    check_admissible(E, U, X, bg)
    if W is None:
        delta_U_N = (U[N] - X[N]) / X[N]  # the deviations, formed without subtracting one
        delta_rho_N_1 = (E[N - 1] - geo.dV[N - 1]) / geo.dV[N - 1]
        W = characteristic_pair(float(delta_U_N), float(delta_rho_N_1), float(X[N]), bg.c_s)[1]
    return State(E=E, U=U, W=W)


def check_admissible(E: FloatArray, U: FloatArray, X: FloatArray, bg: Background) -> None:
    """Refuse data with a non-positive density in any cell or `Gammabar^2 <= 0` at any face (Section 5.4).

    `Gammabar^2 = Gammabar_FRW^2 + U^2 - M / X` with `M = 3 sum E` the tilde mass times `X^3` (eq:eul:gamma): the
    linearised bound eq:lin:gammaconstraint, checked on the nonlinear fields. It is stringent at large radius, where it
    constrains `X^2 delta_m`, and for a compensated profile it is a constraint on the interior only. It is formed
    from the deviations, as a stage forms it (`derive`).
    """
    dV = shell_volumes(X)
    rho = E / dV
    if np.any(rho <= 0.0):
        c = int(np.argmin(rho))
        raise NotHyperbolicError("rho", c, float(rho[c]))
    dM = 3.0 * np.cumsum(E - dV)
    Gammabar2 = gammabar_squared(bg, X[1:], U[1:], U[1:] - X[1:], dM)
    if np.any(Gammabar2 <= 0.0):
        j = int(np.argmin(Gammabar2))
        raise NotHyperbolicError("Gammabar2", j + 1, float(Gammabar2[j]))


def cell_contents(delta_rho: Profile, geo: Geometry, n_nodes: int = 12) -> FloatArray:
    """The energy content `int X^2 (1 + delta_rho) dX` of every cell for a density given as a profile.

    The background part is in closed form and the perturbation by Gauss-Legendre quadrature with `n_nodes` per
    cell, which the paper finds exact to round-off for its profiles at twelve nodes.
    """
    N = geo.N
    X = geo.X[: N + 1]
    nodes, weights = np.polynomial.legendre.leggauss(n_nodes)
    mid, half = 0.5 * (X[1:] + X[:-1]), 0.5 * (X[1:] - X[:-1])
    Xq = mid[:, None] + half[:, None] * nodes[None, :]
    return geo.dV + half * (weights[None, :] * Xq**2 * delta_rho(Xq)).sum(axis=1)
