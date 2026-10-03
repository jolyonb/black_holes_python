"""The initial data: the growing mode of a time-independent seed (paper Sections 5.4 and 7.9).

How it is used. A production datum is one seed in and a `State` out, the start time computed from the seed:

    seed = Seed.of(Gaussian(A, ell), "delta_m0", Rtilde_max)     # delta_m0 on the box's mass basis
    peak = seed.peak()                                            # r_m, the peak compaction and its shape q
    xi_0 = seed_start(eos, peak.r_m, epsilon2)                    # the latest start with eps0^2 <= epsilon2
    deviation = seed.deviation(geo, eos, xi_0)                    # grown, second order added, admissible

The seed is the first-order profile of the growing solution, `delta_m = e^xi delta_m0(X) + O(eps^4)` outside the
horizon (radiation), with `X` the scaled radius, the comoving radius in units of the Hubble radius at `xi = 0` and at
leading order the areal coordinate. It fixes the pure growing mode: one free function, every order of the gradient
expansion determined by it, no decaying mode as `t -> 0`. It is given as `delta_m0` itself, as the curvature profile
`K` of the literature (`delta_m0 = K / (1 + alpha)`, `2 K / 3` for radiation) or as the compaction profile
`C0 = X^2 delta_m0` (`SeedForm`). The datum follows from it in closed form:

1. Projection (eq:lin:bn, its mass form). On `0 <= X <= Rtilde_max` the seed has coefficients `b_n` on the functions
   `3 j_1(k_n X) / (k_n X)`, `k_n = n pi / Rtilde_max`, found without differentiating it (`project`).
2. The start. `r_m` is the radius of the peak of `C0`, and `eps0^2 = e^(2 (1 - alpha) xi) / r_m^2`, the Hubble radius
   in units of `r_m` squared, is the expansion parameter at the start; the start is the latest time with `eps0^2` at
   or below a tolerance (`initial.epsilon2` of the configuration), `xi_0 = ln(epsilon2 r_m^2)` for radiation
   (eq:lin:xistart).
3. The linear growth, exact mode by mode (eq:lin:seedmodes): each coefficient is multiplied by its growth,
   `e^xi 3 j_1(z_n) / z_n` with `z_n = k_n tau`, which is the growing mode of eq:lin:mode with `B_n = 9 b_n / k_n^2`
   (the paper's normalisation), so that the density, the mass and the linear velocity follow from the same growing
   branch (eq:lin:modepair).
   Nothing is divided by `j_1`: a seed with structure inside the sound horizon at the start is grown, not
   reconstructed, and the ill-posedness of the former recipe below does not arise.
4. The second order (eq:lin:seed2). The linear gradient terms of the second order are already in the Bessel series;
   what is added is the nonlinear part, `e^(2 xi)` times the quadratic form `second_order` (eq:lin:seedQ) of the seed
   in `delta_m`, and in the velocity its linear companion `-1/4` of it together with the correction eq:num:dUnl
   evaluated on the linear `delta_m`: the datum eq:lin:seeddata, which with the series is the second-order relation
   eq:lin:relation2 (verified in sympy). It is the growing solution up to relative `O(eps0^4)`.
5. Sampling (eq:num:idata). The cell contents are exact, `E_c = [X^3 (1 + delta_m)] / 3` across the cell, so the
   cumulative sum returns `mt_j = 1 + delta_m(X_j)` exactly, and the face velocities are point values; the data are
   formed as their deviation from FRW, so a perturbation below round-off of the background survives.

Every datum then goes through `initial_deviation` (or `initial_state` for data known as fields): `U_0 = 0`, `W` at
the discrete `u_-` so that the outer penalty starts at zero, and the admissibility check, `Gammabar^2 > 0` at every
face and `rho > 0` in every cell. The data are also refused unless compensated, `delta_m(Rtilde_max) = 0` to a
tolerance (eq:lin:compensated), since the exterior must be FRW for the outer boundary to mean anything. Everything
here is for radiation, the only fluid with the exact outgoing-wave outer condition of Section 5, and so the only one
for which the closed forms eq:lin:seedmodes, eq:lin:seedQ and eq:num:dUnl are printed; `seed_start` and
`Seed.deviation` refuse any other equation of state.

Linear data with both fields, and so possibly decaying content, are the general `ModeExpansion` of Section 5.4.1, of
which the growing mode is the `C_n = 0` special case; they are carried to the start time by evaluating the
expansion there, and sampled without the correction, which is derived for the growing solution:

    expansion = ModeExpansion.from_fields(delta_rho, delta_U, Rtilde_max, bg_i)   # never singular
    state = expansion.state(geo, bg_0)

The former recipe of Section 5.4, a profile of `delta_rho` or `delta_m` imposed at a chosen start time and read as the
growing mode by dividing each mode by `z_n j_1(z_n)` (`GrowingMode.from_profile`, with the guard on the modes near the
zeros of `j_1` and the diagnostic `correction_ratio`), is no longer printed and is kept for the recorded experiments of
the analysis, whose harnesses call it. A profile imposed at a start time is a different perturbation from the same
profile as a seed, at relative `O(eps0^2)` (Section 5.4).

For test data given as a density alone, with a velocity that is not the growing mode's, `cell_contents` integrates
the density over each cell by Gauss-Legendre quadrature, exact to round-off for smooth profiles.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
from scipy.special import spherical_jn, spherical_yn

from pbh.derived import NotHyperbolicError, gammabar_squared
from pbh.eos import Background, EquationOfState
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
        """`(delta_m, delta_m', delta_m'')` at the radii `X`, primes in `X` (`mass_series`)."""
        a, _ = self._coefficients(bg)
        return mass_series(a, self.k, X)

    def delta_U(self, bg: Background, X: FloatArray) -> FloatArray:
        """The linear velocity perturbation `U / X - 1` at the radii `X`."""
        _, u = self._coefficients(bg)
        return 3.0 * j1_over_x(np.multiply.outer(X, self.k)) @ u

    def deviation(self, geo: Geometry, bg: Background) -> State:
        """The linear data sampled on the grid at this time, as their deviation from FRW, through the door.

        The exact cell contents are formed as `delta E_c = [X^3 delta_m] / 3` across the cell, never as the
        difference of `[X^3 (1 + delta_m)] / 3` and the shell volume, so that a perturbation below round-off of the
        background survives: the record and the integrator carry the deviation, and nothing on the way adds FRW back.
        """
        X = geo.X[: geo.N + 1]
        return initial_deviation(np.diff(X**3 * self.delta_m(bg, X)) / 3.0, X * self.delta_U(bg, X), geo, bg)

    def state(self, geo: Geometry, bg: Background) -> State:
        """The linear data sampled on the grid at this time: exact cell contents, point velocities, through the door."""
        return with_background(ModeExpansion.deviation(self, geo, bg), geo)  # the general sampling, even on a subclass


@dataclass(frozen=True, init=False)
class GrowingMode(ModeExpansion):
    """The growing branch alone, `C_n = 0` (Section 5.4): the linear part of the production datum (`Seed`)."""

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

        The former recipe, kept for the recorded experiments of the analysis (`experiments/e8`, `e9`, `e10`), whose
        harnesses call it; the production datum is a `Seed`. The amplitudes are `B_n = b_n / (z_n j_1(z_n))`, ill
        defined at every zero of `j_1` (the first at `z = 4.4934`, a structure finer than about 1.4 sound horizons),
        so the fraction of the power carried by the modes with `z_n > 3`, which covers every zero, is reported in the
        input and in the output velocity and the data are refused if either exceeds `power_tolerance`.

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

    def deviation(self, geo: Geometry, bg: Background, nonlinear: bool = True) -> State:
        """The growing mode sampled on the grid as the paper prescribes (eq:num:idata), as its deviation from FRW.

        The cell contents are exact and the face velocities carry the correction eq:num:dUnl unless `nonlinear` is
        off; the correction is derived for the growing solution, which is why it lives here and not on the general
        expansion. Radiation only, as eq:lin:seedmodes and eq:num:dUnl are.
        """
        if not nonlinear:
            return super().deviation(geo, bg)
        X = geo.X[: geo.N + 1]
        delta_m, first, second = self.delta_m_derivatives(bg, X)
        delta_U = self.delta_U(bg, X) + nonlinear_correction(X, delta_m, first, second)
        return initial_deviation(np.diff(X**3 * delta_m) / 3.0, X * delta_U, geo, bg)

    def state(self, geo: Geometry, bg: Background, nonlinear: bool = True) -> State:
        """The growing mode sampled on the grid, FRW plus `deviation`."""
        return with_background(self.deviation(geo, bg, nonlinear), geo)

    def correction_ratio(self, geo: Geometry, bg: Background) -> float:
        """The former recipe's diagnostic, no longer printed: `max |delta_U^nl / delta_U^lin|` over the grid's faces.

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


# --- the seed: the production datum ---


type SeedForm = Literal["delta_m0", "K", "compaction"]
"""How a seed is given: `delta_m0` itself, the curvature profile `K`, or the compaction profile `C0 = X^2 delta_m0`."""

K_TO_DELTA_M0 = 2.0 / 3.0
"""`delta_m0 = K / (1 + alpha)`, the factor `2/3` for radiation, the only fluid the seed is built for."""


def seed_profile(profile: Profile, form: SeedForm) -> Profile:
    """The seed `delta_m0` of a profile given in the form `form` (`SeedForm`).

    A compaction profile is divided by `X^2`, which the projection never evaluates at the origin.
    """
    if form == "K":
        return lambda X: K_TO_DELTA_M0 * profile(X)
    if form == "compaction":
        return lambda X: profile(X) / X**2
    return profile


@dataclass(frozen=True)
class CompactionPeak:
    """The peak of a seed's compaction `C0 = X^2 delta_m0`.

    Attributes:
        r_m: Its radius, the scale against which the start is measured.
        C_m: Its value, the peak compaction of the perturbation at leading order.
        q: Its shape, `-C0''(r_m) r_m^2 / (4 C0(r_m))`, `1` for a Gaussian in `delta_m0` (or in `K`).
    """

    r_m: float
    C_m: float
    q: float


@dataclass(frozen=True)
class Seed:
    """The time-independent seed `delta_m0` on the box's mass basis: `delta_m0 = sum b_n 3 j_1(k_n X) / (k_n X)`.

    Attributes:
        Rtilde_max: The outer edge of the box.
        k: The wavenumbers `k_n = n pi / Rtilde_max`.
        b: The coefficients `b_n`.
    """

    Rtilde_max: float
    k: FloatArray
    b: FloatArray

    @classmethod
    def of(cls, profile: Profile, form: SeedForm, Rtilde_max: float, n_modes: int = 150) -> Self:
        """The seed of a profile given in the form `form`, projected on the box (eq:lin:bn in its mass form)."""
        k = box_wavenumbers(Rtilde_max, n_modes)
        return cls(Rtilde_max, k, project(seed_profile(profile, form), "m", k, Rtilde_max))

    def derivatives(self, X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """`(delta_m0, delta_m0', delta_m0'')` at the radii `X`, primes in `X`, from the series (`mass_series`)."""
        return mass_series(self.b, self.k, X)

    def peak(self) -> CompactionPeak:
        """The peak of `C0 = X^2 delta_m0` on the box: the largest of a sampling, then bisection on `C0'`.

        Raises:
            ValueError: If `C0` has no positive maximum inside the box.
        """
        X = np.linspace(0.0, self.Rtilde_max, 8 * self.k.size + 1)
        C0 = X**2 * self.derivatives(X)[0]
        i = int(np.argmax(C0))
        if not (0 < i < X.size - 1 and C0[i] > 0.0):
            raise ValueError("the seed's compaction X^2 delta_m0 has no positive peak inside the box")
        lo, hi = float(X[i - 1]), float(X[i + 1])
        for _ in range(60):  # C0' = X (2 delta_m0 + X delta_m0') changes sign once, from + to -, across the peak
            mid = 0.5 * (lo + hi)
            d, first, _ = self.derivatives(np.array([mid]))
            if 2.0 * d[0] + mid * first[0] > 0.0:
                lo = mid
            else:
                hi = mid
        r_m = 0.5 * (lo + hi)
        d, first, second = (float(f[0]) for f in self.derivatives(np.array([r_m])))
        C_m = r_m**2 * d
        curvature = 2.0 * d + 4.0 * r_m * first + r_m**2 * second  # C0''
        return CompactionPeak(r_m=r_m, C_m=C_m, q=-curvature * r_m**2 / (4.0 * C_m))

    def growing_mode(self) -> GrowingMode:
        """The linear growing mode of the seed, eq:lin:seedmodes: `B_n = 9 b_n / k_n^2`.

        Each coefficient then grows as `e^xi 3 j_1(z_n) / z_n` with `z_n = k_n tau` (`tau^2 = e^xi / 3`), so that
        `delta_m -> e^xi delta_m0` as `xi -> -inf`.
        """
        return GrowingMode(self.k, 9.0 * self.b / self.k**2)

    def deviation(self, geo: Geometry, eos: EquationOfState, xi: float) -> State:
        """The growing solution of the seed at the time `xi`, sampled on the grid as its deviation from FRW.

        eq:lin:seeddata: `delta_m` is the exact linear growth plus `e^(2 xi) second_order`; the velocity is the linear
        velocity, the linear companion `-e^(2 xi) second_order / 4` of the quadratic part of `delta_m`, and
        eq:num:dUnl on the linear `delta_m` (using the whole `delta_m` there changes it at `O(eps^6)`).

        Raises:
            ValueError: Unless the fluid is radiation, whose coefficients these are.
            NotCompensatedError: Unless the linear data are compensated; and through the door the admissibility
                errors.
        """
        require_radiation(eos)
        bg = Background.at(eos, xi)
        mode = self.growing_mode()
        mode.check_compensated(bg, self.Rtilde_max)
        X = geo.X[: geo.N + 1]
        linear, first, second = mode.delta_m_derivatives(bg, X)
        quadratic = bg.Gammabar2**2 * second_order(X, *self.derivatives(X))  # e^(2 xi) for radiation
        delta_m = linear + quadratic
        delta_U = mode.delta_U(bg, X) - 0.25 * quadratic + nonlinear_correction(X, linear, first, second)
        return initial_deviation(np.diff(X**3 * delta_m) / 3.0, X * delta_U, geo, bg)


def require_radiation(eos: EquationOfState) -> None:
    """Refuse any fluid but radiation, the only one the seed is built for (the outer condition's scope)."""
    if not eos.is_radiation:
        raise ValueError("the seed's growing mode is built for radiation only (the outer condition's scope)")


def seed_start(eos: EquationOfState, r_m: float, epsilon2: float) -> float:
    """The latest start with `eps0^2 = e^(2 (1 - alpha) xi) / r_m^2` at or below `epsilon2`, eq:lin:xistart.

    `ln(epsilon2 r_m^2)` for radiation; `2 (1 - alpha)` is the growth rate of the super-horizon growing mode.

    Raises:
        ValueError: Unless the fluid is radiation, the only one the seed is built for.
    """
    require_radiation(eos)
    return math.log(epsilon2 * r_m**2) / eos.growing_mode_rate


def second_order(X: FloatArray, d: FloatArray, first: FloatArray, second: FloatArray) -> FloatArray:
    """The quadratic part `Q` of `delta_m` at second order in the gradient expansion, per `e^(2 xi)`, eq:lin:seedQ.

    With `d = delta_m0` and primes in `X`, at areal radius `X`:
    `[33 d^2 + 6 X d d' - 3 X^2 d d'' + 2 X^2 (d')^2] / 60`, which is
    `-(1/20) C0 (d'' + 4 d'/X) + (11/20) d^2 + (3/10) X d d' + (1/30) X^2 (d')^2` with `C0 = X^2 d`, written without
    the division by `X`. The linear part of that order, `(d'' + 4 d'/X) / 30`, is in the Bessel series.
    """
    return (33.0 * d**2 + 6.0 * X * d * first - 3.0 * X**2 * d * second + 2.0 * X**2 * first**2) / 60.0


# --- the projection, the correction, and the door every datum goes through ---


def mass_series(a: FloatArray, k: FloatArray, X: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """`(f, f', f'')` at the radii `X` for `f = sum a_n 3 j_1(k_n X) / (k_n X)`, primes in `X`.

    From the identities `d/dx [j_1/x] = -x [j_2/x^2]` and `d/dx [j_2/x^2] = (j_1/x - 5 j_2/x^2) / x`.
    """
    kX = np.multiply.outer(X, k)
    f1, f2 = j1_over_x(kX), j2_over_x2(kX)
    first = -3.0 * (f2 * X[..., None]) @ (a * k**2)
    second = -3.0 * (f1 - 4.0 * f2) @ (a * k**2)
    return 3.0 * f1 @ a, first, second


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
    Data known as their deviation from FRW go through `initial_deviation`, which is this door without the background.
    """
    X = geo.X[: geo.N + 1]
    U = U.copy()
    U[0] = 0.0
    deviation = initial_deviation(E - geo.dV, U - X, geo, bg, W)
    return State(E=E, U=U, W=deviation.W)


def initial_deviation(
    delta_E: FloatArray, delta_U: FloatArray, geo: Geometry, bg: Background, W: float | None = None
) -> State:
    """`initial_state` for data given as their deviation from FRW: `delta E_c = E_c - Delta V_c`, `U_j - X_j`.

    The same door, in the form the record and the integrator carry, so that a perturbation below round-off of the
    background is never added to it and subtracted again. Returns the deviation, `W` included (FRW value `0`).
    """
    N = geo.N
    X = geo.X[: N + 1]
    delta_U = delta_U.copy()
    delta_U[0] = 0.0
    check_admissible(delta_E, delta_U, X, bg)
    if W is None:
        delta_rho_N_1 = delta_E[N - 1] / geo.dV[N - 1]
        W = characteristic_pair(float(delta_U[N] / X[N]), float(delta_rho_N_1), float(X[N]), bg.c_s)[1]
    return State(E=delta_E, U=delta_U, W=W)


def with_background(deviation: State, geo: Geometry) -> State:
    """The state whose deviation from FRW this is, before any excision: `E = Delta V + delta E`, `U = X + delta U`."""
    return State(E=geo.dV + deviation.E, U=geo.X[: geo.N + 1] + deviation.U, W=deviation.W)


def check_admissible(delta_E: FloatArray, delta_U: FloatArray, X: FloatArray, bg: Background) -> None:
    """Refuse data with a non-positive density in any cell or `Gammabar^2 <= 0` at any face (Section 5.4).

    `Gammabar^2 = Gammabar_FRW^2 + U^2 - M / X` with `M = 3 sum E` the tilde mass times `X^3` (eq:eul:gamma): the
    linearised bound eq:lin:gammaconstraint, checked on the nonlinear fields. It is stringent at large radius, where it
    constrains `X^2 delta_m`, and for a compensated profile it is a constraint on the interior only. It is formed
    from the deviations, as a stage forms it (`derive`).
    """
    dV = shell_volumes(X)
    rho = 1.0 + delta_E / dV
    if np.any(rho <= 0.0):
        c = int(np.argmin(rho))
        raise NotHyperbolicError("rho", c, float(rho[c]))
    dM = 3.0 * np.cumsum(delta_E)
    Gammabar2 = gammabar_squared(bg, X[1:], X[1:] + delta_U[1:], delta_U[1:], dM)
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
